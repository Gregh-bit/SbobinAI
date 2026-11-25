import { useState, useRef, useEffect, useLayoutEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  PanelLeft,
  Archive,
  BookmarkPlus,
  ChevronDown,
  AlertTriangle,
  CheckCircle2,
  X,
  Loader2,
  Layers,
  Cpu,
} from 'lucide-react';
import { clsx } from 'clsx';
import type {
  ChatScope,
  ChatSession,
  ChatSessionDetail,
  ChatStreamEvent,
  ChatStudyMode,
  ContextStats,
  CourseMaterial,
  Lecture,
  PersistentChatMessage,
  Subject,
} from '../types';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import { liquidMenuVariants, liquidPopoverContainerClass } from '../lib/animations';
import ChatSidebar from './chat/ChatSidebar';
import ChatMessageItem, { parseThinkingContent } from './chat/ChatMessageItem';
import ChatComposer from './chat/ChatComposer';
import { useDelayedLoading, ChatStreamSkeleton } from './Skeleton';

interface AssistenteCorsoViewProps {
  subject: Subject;
  lectures: Lecture[];
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenAudioSnippet?: (lectureId: number, seconds: number, autoPlay?: boolean) => void;
  onOpenSettings?: () => void;
  onOpenQuiz?: (quizId: number) => void;
  onOpenOralQuestion?: (question: string, topic?: string) => void;
  onOpenQuadernoLecture?: (lectureId: number) => void;
  onLectureCreated?: () => void;
  initialReviewTarget?: {
    chatSessionId?: number | null;
    quizId: number;
    quizTitle: string;
    score: number;
    total: number;
    wrongCount: number;
    key: number;
  } | null;
  onClearInitialReview?: () => void;
}

const DEFAULT_CONTEXT_STATS: ContextStats = {
  used_tokens: 0,
  max_tokens: 8192,
  percentage: 0,
  active_messages_count: 0,
  compacted_messages_count: 0,
  has_compacted_summary: false,
  needs_compaction: false,
};

const EMPTY_SCOPE: ChatScope = { lecture_ids: [], material_ids: [] };

// Cache in memoria per materia: evita flash/reload della chat cambiando scheda
const subjectChatCache = new Map<
  number,
  {
    sessions: ChatSession[];
    activeSession: ChatSessionDetail | null;
    materials: CourseMaterial[];
  }
>();

function getSaturationTone(percentage: number) {
  if (percentage <= 0) {
    return {
      bar: 'bg-emerald-500/20',
      pill: 'bg-white/[0.04] border-white/[0.08] text-zinc-400',
      dot: 'bg-zinc-500',
    };
  }
  if (percentage >= 85) {
    return {
      bar: 'bg-rose-500',
      pill: 'bg-rose-500/[0.12] border-rose-500/30 text-rose-200',
      dot: 'bg-rose-400',
    };
  }
  if (percentage >= 65) {
    return {
      bar: 'bg-amber-400',
      pill: 'bg-amber-500/[0.12] border-amber-500/30 text-amber-200',
      dot: 'bg-amber-400',
    };
  }
  return {
    bar: 'bg-emerald-400',
    pill: 'bg-white/[0.04] border-white/[0.10] text-zinc-300',
    dot: 'bg-emerald-400',
  };
}

function toSessionSummary(detail: ChatSessionDetail): ChatSession {
  return {
    id: detail.id,
    subject_id: detail.subject_id,
    title: detail.title,
    compacted_summary: detail.compacted_summary,
    compacted_up_to_message_id: detail.compacted_up_to_message_id,
    scope: detail.scope,
    context_stats: detail.context_stats,
    message_count: detail.messages.length,
    created_at: detail.created_at,
    updated_at: detail.updated_at,
  };
}

function CompactionDivider({ summary }: { summary?: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="my-5 animate-fade-in w-full">
      <div className="flex items-center gap-3 w-full">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/[0.12] to-white/[0.08]" />
        <button
          type="button"
          onClick={() => summary && setExpanded((prev) => !prev)}
          className={clsx(
            'inline-flex items-center gap-1.5 py-1 text-xs text-zinc-400 transition-colors select-none group',
            summary ? 'hover:text-zinc-200 cursor-pointer' : 'cursor-default opacity-60'
          )}
        >
          <Archive size={13} className="text-zinc-400 group-hover:text-zinc-200 transition-colors flex-shrink-0" />
          <span className="font-normal">Contesto compattato</span>
          {summary && (
            <ChevronDown
              size={12}
              className={clsx(
                'text-zinc-500 group-hover:text-zinc-300 transition-transform duration-200 flex-shrink-0 ml-0.5',
                expanded && 'rotate-180'
              )}
            />
          )}
        </button>
        <div className="h-px flex-1 bg-gradient-to-l from-transparent via-white/[0.12] to-white/[0.08]" />
      </div>

      <AnimatePresence>
        {expanded && summary && (
          <motion.div
            initial={{ opacity: 0, height: 0, y: -4 }}
            animate={{ opacity: 1, height: 'auto', y: 0 }}
            exit={{ opacity: 0, height: 0, y: -4 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden w-full max-w-2xl mx-auto mt-2.5 rounded-2xl bg-[#121622]/85 border border-white/10 px-4 py-3 text-xs text-zinc-300 leading-relaxed whitespace-pre-wrap select-text"
          >
            {summary}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/**
 * ChatGPT 2026 Course Assistant view with persistent multi-chat sidebar,
 * centered Markdown/KaTeX/Citation stream, study modes, scoped RAG context,
 * context saturation management, and Quaderno archiving.
 */
export default function AssistenteCorsoView({
  subject,
  lectures,
  onOpenCitation,
  onOpenAudioSnippet,
  onOpenSettings,
  onOpenQuiz,
  onOpenOralQuestion,
  onOpenQuadernoLecture,
  onLectureCreated,
  initialReviewTarget,
  onClearInitialReview,
}: AssistenteCorsoViewProps) {
  const initialCache = subjectChatCache.get(subject.id);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 900 : true
  );
  const [sessions, setSessions] = useState<ChatSession[]>(() => initialCache?.sessions ?? []);
  const [activeSession, setActiveSession] = useState<ChatSessionDetail | null>(
    () => initialCache?.activeSession ?? null
  );
  const [materials, setMaterials] = useState<CourseMaterial[]>(() => initialCache?.materials ?? []);

  const [isLoadingSessions, setIsLoadingSessions] = useState(() => !initialCache?.activeSession);
  const [isLoadingSessionDetail, setIsLoadingSessionDetail] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const isChatStreamLoading = isLoadingSessionDetail || (isLoadingSessions && !activeSession);
  const showChatStreamSkeleton = useDelayedLoading(isChatStreamLoading, 150, 220);

  const { settings } = useSettings();
  const supportsThinking = settings.llmSupportsThinking !== false;

  const [input, setInput] = useState('');
  const [studyMode, setStudyMode] = useState<ChatStudyMode>('auto');
  const [thinkingMode, setThinkingMode] = useState(false);

  useEffect(() => {
    if (!supportsThinking && thinkingMode) {
      setThinkingMode(false);
    }
  }, [supportsThinking, thinkingMode]);
  const [streamingMessageId, setStreamingMessageId] = useState<number | null>(null);
  const [savingNoteId, setSavingNoteId] = useState<number | null>(null);
  const [savedNoteIds, setSavedNoteIds] = useState<Record<number, boolean>>({});
  const [deletingMessageId, setDeletingMessageId] = useState<number | null>(null);

  const [isCompacting, setIsCompacting] = useState(false);
  const [contextMenuOpen, setContextMenuOpen] = useState(false);
  const [archiveModalOpen, setArchiveModalOpen] = useState(false);
  const [archiveTitle, setArchiveTitle] = useState('');
  const [archiveMode, setArchiveMode] = useState<'quaderno_and_clear' | 'quaderno_keep'>(
    'quaderno_and_clear'
  );
  const [isArchiving, setIsArchiving] = useState(false);

  const [toast, setToast] = useState<{ text: string; tone: 'success' | 'error' } | null>(null);
  const [pendingScope, setPendingScope] = useState<ChatScope | null>(null);

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const streamEndRef = useRef<HTMLDivElement>(null);
  const scrolledSessionIdRef = useRef<number | null>(null);
  const prevMsgCountRef = useRef<number>(0);
  const nearBottomRef = useRef(true);
  const clientKeyMapRef = useRef<Map<number, string>>(new Map());
  const abortControllerRef = useRef<AbortController | null>(null);
  const streamTokenRef = useRef(0);
  const stopSyncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeSessionRef = useRef<ChatSessionDetail | null>(activeSession);
  const activeSessionIdRef = useRef<number | null>(null);
  const sessionsRef = useRef<ChatSession[]>(sessions);
  const loadSeqRef = useRef(0);
  const contextMenuRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(true);
  const pendingScopeRef = useRef<ChatScope | null>(null);
  const regeneratingSnapshotRef = useRef<{
    chatId: number;
    messageId: number;
    prevMessage: PersistentChatMessage;
  } | null>(null);

  activeSessionRef.current = activeSession;
  activeSessionIdRef.current = activeSession?.id ?? null;
  sessionsRef.current = sessions;

  const updatePendingScope = useCallback((next: ChatScope | null) => {
    pendingScopeRef.current = next;
    setPendingScope(next);
  }, []);

  const showToast = useCallback((text: string, tone: 'success' | 'error' = 'success') => {
    setToast({ text, tone });
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 3200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  // Close context menu dropdown on outside click
  useEffect(() => {
    if (!contextMenuOpen) return;
    const handler = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [contextMenuOpen]);

  // Auto-close sidebar on small screen resize (< 768px)
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Close archive modal or context menu on Escape key
  useEffect(() => {
    if (!archiveModalOpen && !contextMenuOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (contextMenuOpen) {
          setContextMenuOpen(false);
          return;
        }
        if (archiveModalOpen && !isArchiving) {
          setArchiveModalOpen(false);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [archiveModalOpen, contextMenuOpen, isArchiving]);

  const abortActiveStream = useCallback(() => {
    streamTokenRef.current += 1;
    if (stopSyncTimerRef.current !== null) {
      clearTimeout(stopSyncTimerRef.current);
      stopSyncTimerRef.current = null;
    }
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {
        // ignore
      }
      abortControllerRef.current = null;
    }
    setStreamingMessageId(null);
  }, []);

  // Cleanup active stream on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortActiveStream();
    };
  }, [abortActiveStream]);

  const syncSessionIntoList = useCallback((detail: ChatSessionDetail, moveToTop = false) => {
    const summary = toSessionSummary(detail);
    setSessions((prev) => {
      const exists = prev.some((s) => s.id === summary.id);
      if (!exists) return [summary, ...prev];
      if (moveToTop) {
        return [summary, ...prev.filter((s) => s.id !== summary.id)];
      }
      return prev.map((s) => (s.id === summary.id ? summary : s));
    });
  }, []);

  // Initial load + legacy localStorage migration when subject changes
  useEffect(() => {
    let cancelled = false;
    abortActiveStream();
    regeneratingSnapshotRef.current = null;
    updatePendingScope(null);
    const cached = subjectChatCache.get(subject.id);
    if (cached?.activeSession) {
      setSessions(cached.sessions);
      setActiveSession(cached.activeSession);
      setMaterials(cached.materials);
      setIsLoadingSessions(false);
    } else {
      setSessions([]);
      setActiveSession(null);
      setSavedNoteIds({});
      setIsLoadingSessions(true);
    }

    const legacyKey = `sbobinai_chat_subject_${subject.id}`;
    const migratedKey = `sbobinai_chat_migrated_${subject.id}`;

    const init = async () => {
      // Load materials in parallel (non-blocking if fails)
      api.subjects
        .getMaterials(subject.id)
        .then((mats) => {
          if (!cancelled) setMaterials(mats);
        })
        .catch(() => {
          if (!cancelled) setMaterials([]);
        });

      let importedSession: ChatSessionDetail | null = null;

      // One-time legacy localStorage migration
      try {
        const rawLegacy = localStorage.getItem(legacyKey);
        const alreadyMigrated = localStorage.getItem(migratedKey);
        if (rawLegacy && !alreadyMigrated) {
          const parsed = JSON.parse(rawLegacy);
          if (Array.isArray(parsed)) {
            const meaningful = parsed.filter(
              (m: any) =>
                m &&
                typeof m.content === 'string' &&
                m.content.trim().length > 0 &&
                !(m.id === '1' && m.role === 'assistant')
            );
            if (meaningful.length > 0) {
              importedSession = await api.chats.importLegacy(subject.id, {
                title: 'Chat precedente',
                messages: meaningful.map((m: any) => ({
                  role: m.role === 'user' ? 'user' : 'assistant',
                  content: String(m.content),
                  thinking: typeof m.thinking === 'string' ? m.thinking : undefined,
                  context_used:
                    typeof m.contextUsed === 'string'
                      ? m.contextUsed
                      : typeof m.context_used === 'string'
                        ? m.context_used
                        : undefined,
                })),
              });
            }
          }
          localStorage.setItem(migratedKey, '1');
          localStorage.removeItem(legacyKey);
        }
      } catch {
        // Keep legacy data untouched if import fails so user doesn't lose messages
      }

      if (cancelled) return;

      try {
        const list = await api.chats.list(subject.id);
        if (cancelled) return;

        const applyPendingScopeIfAny = async (session: ChatSessionDetail): Promise<ChatSessionDetail> => {
          const scoped = pendingScopeRef.current;
          if (!scoped) return session;
          updatePendingScope(null);
          try {
            const updated = await api.chats.update(session.id, { scope: scoped });
            return updated;
          } catch {
            return { ...session, scope: scoped };
          }
        };

        if (list.length > 0) {
          setSessions(list);
          if (importedSession && list.some((s) => s.id === importedSession!.id)) {
            const resolved = await applyPendingScopeIfAny(importedSession);
            if (!cancelled) {
              setActiveSession(resolved);
              syncSessionIntoList(resolved);
            }
          } else {
            const preferredId =
              activeSessionIdRef.current && list.some((s) => s.id === activeSessionIdRef.current)
                ? activeSessionIdRef.current
                : list[0].id;
            try {
              const firstDetail = await api.chats.get(preferredId);
              if (!cancelled) {
                const resolved = await applyPendingScopeIfAny(firstDetail);
                if (!cancelled) {
                  setActiveSession(resolved);
                  syncSessionIntoList(resolved);
                }
              }
            } catch {
              if (!cancelled && list.length > 1 && preferredId !== list[0].id) {
                try {
                  const fallbackDetail = await api.chats.get(list[0].id);
                  if (!cancelled) {
                    const resolved = await applyPendingScopeIfAny(fallbackDetail);
                    if (!cancelled) {
                      setActiveSession(resolved);
                      syncSessionIntoList(resolved);
                    }
                  }
                } catch {
                  if (!cancelled) setActiveSession(null);
                }
              } else if (!cancelled) {
                setActiveSession(null);
              }
            }
          }
        } else if (importedSession) {
          const resolved = await applyPendingScopeIfAny(importedSession);
          if (!cancelled) {
            setSessions([toSessionSummary(resolved)]);
            setActiveSession(resolved);
          }
        } else {
          // Nessuna chat presente: stato iniziale pulito senza creare sessioni vuote nel database
          if (!cancelled) {
            setSessions([]);
            setActiveSession(null);
          }
        }
      } catch (err: any) {
        if (!cancelled) {
          showToast(err?.message || 'Impossibile caricare le chat del corso', 'error');
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSessions(false);
        }
      }
    };

    void init();
    return () => {
      cancelled = true;
    };
  }, [subject.id, abortActiveStream, syncSessionIntoList, showToast, updatePendingScope]);

  // Sincronizza la cache in memoria per la materia corrente (evita flash cambiando tab)
  useEffect(() => {
    if (sessions.length > 0 || activeSession) {
      subjectChatCache.set(subject.id, { sessions, activeSession, materials });
    } else {
      subjectChatCache.delete(subject.id);
    }
  }, [subject.id, sessions, activeSession, materials]);

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    nearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  // Posizionamento istantaneo in fondo PRIMA del paint quando si apre/cambia chat (zero scroll visibile),
  // e scroll fluido durante invio, streaming e arrivo token.
  useLayoutEffect(() => {
    const el = scrollContainerRef.current;
    if (!el || isLoadingSessionDetail || showChatStreamSkeleton) return;

    const currentSessionId = activeSession?.id ?? null;
    const msgCount = activeSession?.messages.length ?? 0;

    if (currentSessionId !== scrolledSessionIdRef.current) {
      if (currentSessionId !== null && msgCount > 0) {
        el.scrollTop = el.scrollHeight;
        scrolledSessionIdRef.current = currentSessionId;
        prevMsgCountRef.current = msgCount;
        nearBottomRef.current = true;
      } else if (currentSessionId !== null) {
        scrolledSessionIdRef.current = currentSessionId;
        prevMsgCountRef.current = 0;
        nearBottomRef.current = true;
      }
      return;
    }

    if (nearBottomRef.current) {
      const raf = requestAnimationFrame(() => {
        if (scrollContainerRef.current && nearBottomRef.current) {
          scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
        }
      });
      prevMsgCountRef.current = msgCount;
      return () => cancelAnimationFrame(raf);
    }
    prevMsgCountRef.current = msgCount;
  }, [activeSession?.id, activeSession?.messages, streamingMessageId, isLoadingSessionDetail, showChatStreamSkeleton]);

  const handleSelectSession = async (sessionId: number) => {
    if (sessionId === activeSessionIdRef.current && activeSession) return;
    abortActiveStream();
    const seq = ++loadSeqRef.current;
    setIsLoadingSessionDetail(true);
    try {
      const detail = await api.chats.get(sessionId);
      if (!isMountedRef.current || loadSeqRef.current !== seq) return;
      setActiveSession(detail);
      syncSessionIntoList(detail);
    } catch (err: any) {
      if (isMountedRef.current && loadSeqRef.current === seq) {
        showToast(err?.message || 'Errore apertura conversazione', 'error');
      }
    } finally {
      if (isMountedRef.current && loadSeqRef.current === seq) {
        setIsLoadingSessionDetail(false);
      }
    }
  };

  const handleCreateSession = async () => {
    if (isCreatingSession) return;
    // Avoid creating duplicate empty sessions if current session already has 0 messages
    if (activeSession && activeSession.messages.length === 0) {
      return;
    }
    abortActiveStream();
    setIsCreatingSession(true);
    try {
      const created = await api.chats.create(subject.id);
      if (!isMountedRef.current) return;
      setActiveSession(created);
      syncSessionIntoList(created, true);
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Impossibile creare una nuova chat', 'error');
      }
    } finally {
      if (isMountedRef.current) {
        setIsCreatingSession(false);
      }
    }
  };

  const handleRenameSession = async (sessionId: number, newTitle: string) => {
    try {
      const updated = await api.chats.update(sessionId, { title: newTitle });
      if (!isMountedRef.current) return;
      syncSessionIntoList(updated);
      if (activeSessionIdRef.current === sessionId) {
        setActiveSession(updated);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Errore rinomina chat', 'error');
      }
    }
  };

  const handleDeleteSession = async (sessionId: number) => {
    const isDeletingActive =
      activeSessionIdRef.current === sessionId || activeSession?.id === sessionId;

    if (isDeletingActive) {
      abortActiveStream();
    }

    try {
      await api.chats.delete(sessionId);
      if (!isMountedRef.current) return;

      const remaining = sessionsRef.current.filter((s) => s.id !== sessionId);
      sessionsRef.current = remaining;
      setSessions(remaining);

      // Se non ci sono più chat per questa materia, azzera completamente la sessione attiva e la cache
      if (remaining.length === 0) {
        setActiveSession(null);
        subjectChatCache.delete(subject.id);
        return;
      }

      // Se la chat eliminata era quella aperta a schermo, passa alla prima chat rimanente
      if (isDeletingActive) {
        try {
          const nextDetail = await api.chats.get(remaining[0].id);
          if (!isMountedRef.current) return;
          setActiveSession(nextDetail);
        } catch {
          if (isMountedRef.current) {
            setActiveSession(null);
          }
        }
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Errore eliminazione chat', 'error');
      }
    }
  };

  const handleScopeChange = async (newScope: ChatScope) => {
    if (!activeSession) {
      updatePendingScope(newScope);
      return;
    }
    const prevScope = activeSession.scope;
    setActiveSession((prev) => (prev ? { ...prev, scope: newScope } : prev));
    try {
      const updated = await api.chats.update(activeSession.id, { scope: newScope });
      if (!isMountedRef.current) return;
      if (activeSessionIdRef.current === updated.id) {
        setActiveSession((prev) =>
          prev ? { ...prev, scope: updated.scope, context_stats: updated.context_stats } : updated
        );
      }
      syncSessionIntoList(updated);
    } catch (err: any) {
      if (!isMountedRef.current) return;
      setActiveSession((prev) => (prev ? { ...prev, scope: prevScope } : prev));
      showToast(err?.message || 'Impossibile salvare il filtro fonti', 'error');
    }
  };

  /**
   * Unified NDJSON stream consumer shared by send and regenerate flows.
   */
  const runStreamIntoSession = useCallback(
    async (params: {
      chatId: number;
      targetAssistantId: number;
      optimisticUserId?: number;
      prevMessage?: PersistentChatMessage;
      invoke: (
        onEvent: (ev: ChatStreamEvent) => void,
        signal: AbortSignal
      ) => Promise<void>;
    }) => {
      const { chatId, targetAssistantId, optimisticUserId, prevMessage, invoke } = params;
      const controller = new AbortController();
      abortControllerRef.current = controller;
      const myStreamToken = ++streamTokenRef.current;

      let rawBuffer = '';
      let streamAction = null as PersistentChatMessage['action'];
      let currentAssistantId = targetAssistantId;

      const isCurrentContext = () =>
        isMountedRef.current &&
        streamTokenRef.current === myStreamToken &&
        activeSessionIdRef.current === chatId;

      const handleStreamFailure = (detail: string, isInterrupt: boolean) => {
        if (prevMessage && !rawBuffer.trim()) {
          setActiveSession((prev) => {
            if (!prev || prev.id !== chatId) return prev;
            return {
              ...prev,
              messages: prev.messages.map((m) =>
                m.id === currentAssistantId ? prevMessage : m
              ),
            };
          });
          showToast(detail, 'error');
          return;
        }
        if (prevMessage) {
          // Regenerate failed mid-stream: resync with SQLite so persisted message stays consistent
          setActiveSession((prev) => {
            if (!prev || prev.id !== chatId) return prev;
            return {
              ...prev,
              messages: prev.messages.map((m) =>
                m.id === currentAssistantId ? prevMessage : m
              ),
            };
          });
          showToast(detail, 'error');
          return;
        }
        setActiveSession((prev) => {
          if (!prev || prev.id !== chatId) return prev;
          return {
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === currentAssistantId
                ? {
                    ...m,
                    content: m.content
                      ? `${m.content}\n\n*(${isInterrupt ? 'Risposta interrotta' : 'Errore'}: ${detail})*`
                      : `Impossibile completare la risposta: ${detail}`,
                  }
                : m
            ),
          };
        });
      };

      let tokenRafId: number | null = null;
      let pendingBufferUpdate = false;

      const flushTokenUpdate = () => {
        if (!pendingBufferUpdate) return;
        pendingBufferUpdate = false;
        const parsed = parseThinkingContent(rawBuffer);
        setActiveSession((prev) => {
          if (!prev || prev.id !== chatId) return prev;
          return {
            ...prev,
            messages: prev.messages.map((m) =>
              m.id === currentAssistantId
                ? {
                    ...m,
                    content: rawBuffer,
                    thinking: parsed.thinking || m.thinking,
                  }
                : m
            ),
          };
        });
      };

      try {
        await invoke((ev) => {
          if (!isCurrentContext()) return;

          if (ev.type === 'meta') {
            if (tokenRafId !== null) {
              cancelAnimationFrame(tokenRafId);
              tokenRafId = null;
            }
            flushTokenUpdate();
            setActiveSession((prev) => {
              if (!prev || prev.id !== chatId) return prev;
              const updatedMsgs = prev.messages.map((m) => {
                if (optimisticUserId !== undefined && m.id === optimisticUserId) {
                  const stableKey = clientKeyMapRef.current.get(optimisticUserId);
                  if (stableKey) clientKeyMapRef.current.set(ev.user_message.id, stableKey);
                  return ev.user_message;
                }
                if (m.id === currentAssistantId) {
                  return {
                    ...m,
                    context_used: ev.context_used,
                    citations: ev.citations,
                  };
                }
                return m;
              });
              return {
                ...prev,
                messages: updatedMsgs,
                context_stats: ev.context_stats,
              };
            });
            return;
          }

          if (ev.type === 'token') {
            rawBuffer += ev.text;
            pendingBufferUpdate = true;
            if (tokenRafId === null) {
              tokenRafId = requestAnimationFrame(() => {
                tokenRafId = null;
                flushTokenUpdate();
              });
            }
            return;
          }

          if (ev.type === 'action') {
            if (tokenRafId !== null) {
              cancelAnimationFrame(tokenRafId);
              tokenRafId = null;
            }
            flushTokenUpdate();
            streamAction = ev.action;
            if (ev.action?.type === 'saved_to_quaderno') {
              onLectureCreated?.();
            }
            setActiveSession((prev) => {
              if (!prev || prev.id !== chatId) return prev;
              return {
                ...prev,
                messages: prev.messages.map((m) =>
                  m.id === currentAssistantId ? { ...m, action: ev.action } : m
                ),
              };
            });
            return;
          }

          if (ev.type === 'done') {
            if (tokenRafId !== null) {
              cancelAnimationFrame(tokenRafId);
              tokenRafId = null;
            }
            flushTokenUpdate();
            const finalizedMsg: PersistentChatMessage = {
              ...ev.assistant_message,
              followups:
                ev.assistant_message.followups?.length > 0
                  ? ev.assistant_message.followups
                  : ev.followups || [],
              action: ev.assistant_message.action ?? streamAction ?? null,
            };
            const stableKey = clientKeyMapRef.current.get(targetAssistantId);
            if (stableKey) clientKeyMapRef.current.set(finalizedMsg.id, stableKey);
            currentAssistantId = finalizedMsg.id;

            const currentSession = activeSessionRef.current;
            const nextDetailForSync: ChatSessionDetail | null =
              currentSession && currentSession.id === chatId
                ? {
                    ...currentSession,
                    title: ev.session_title || currentSession.title,
                    context_stats: ev.context_stats,
                    messages: currentSession.messages.map((m) =>
                      m.id === targetAssistantId || m.id === finalizedMsg.id ? finalizedMsg : m
                    ),
                  }
                : null;

            setActiveSession((prev) => {
              if (!prev || prev.id !== chatId) return prev;
              return {
                ...prev,
                title: ev.session_title || prev.title,
                context_stats: ev.context_stats,
                messages: prev.messages.map((m) =>
                  m.id === targetAssistantId || m.id === finalizedMsg.id ? finalizedMsg : m
                ),
              };
            });

            if (nextDetailForSync) {
              syncSessionIntoList(nextDetailForSync, true);
            }
            return;
          }

          if (ev.type === 'error') {
            handleStreamFailure(ev.detail, false);
          }
        }, controller.signal);
      } catch (err: any) {
        if (err?.name === 'AbortError') return;
        if (!isCurrentContext()) return;
        handleStreamFailure(String(err?.message || err), true);
      } finally {
        if (tokenRafId !== null) {
          cancelAnimationFrame(tokenRafId);
          tokenRafId = null;
        }
        flushTokenUpdate();
        if (streamTokenRef.current === myStreamToken) {
          abortControllerRef.current = null;
          regeneratingSnapshotRef.current = null;
          if (isMountedRef.current) {
            setStreamingMessageId(null);
          }
        }
      }
    },
    [syncSessionIntoList, showToast]
  );

  const handleSend = async (
    overrideText?: string,
    overrideMode?: ChatStudyMode,
    explicitSession?: ChatSessionDetail
  ) => {
    const rawText = (overrideText ?? input).trim();
    const effectiveMode = overrideMode ?? studyMode;
    const hasMode = effectiveMode && effectiveMode !== 'auto';
    if (!rawText && !hasMode) return;
    if (streamingMessageId !== null) return;

    let targetSession = explicitSession || activeSessionRef.current || activeSession;

    if (!targetSession) {
      try {
        const initialScope = pendingScopeRef.current || undefined;
        updatePendingScope(null);
        targetSession = await api.chats.create(
          subject.id,
          initialScope ? { scope: initialScope } : undefined
        );
        if (!isMountedRef.current) return;
        setActiveSession(targetSession);
        syncSessionIntoList(targetSession, true);
      } catch (err: any) {
        if (isMountedRef.current) {
          showToast(err?.message || 'Impossibile avviare la chat', 'error');
        }
        return;
      }
    }

    const tempUserId = -Date.now();
    const tempAssistantId = tempUserId - 1;
    const nowIso = new Date().toISOString();

    clientKeyMapRef.current.set(tempUserId, `client-msg-${tempUserId}`);
    clientKeyMapRef.current.set(tempAssistantId, `client-msg-${tempAssistantId}`);
    nearBottomRef.current = true;

    const optimisticUser: PersistentChatMessage = {
      id: tempUserId,
      session_id: targetSession.id,
      role: 'user',
      content: rawText,
      thinking: '',
      mode: effectiveMode,
      context_used: '',
      citations: [],
      action: null,
      followups: [],
      token_estimate: 0,
      created_at: nowIso,
    };

    const optimisticAssistant: PersistentChatMessage = {
      id: tempAssistantId,
      session_id: targetSession.id,
      role: 'assistant',
      content: '',
      thinking: '',
      mode: effectiveMode,
      context_used: '',
      citations: [],
      action: null,
      followups: [],
      token_estimate: 0,
      created_at: nowIso,
    };

    if (!overrideText) {
      setInput('');
      setStudyMode('auto');
    }
    setStreamingMessageId(tempAssistantId);
    setActiveSession((prev) =>
      prev && prev.id === targetSession!.id
        ? { ...prev, messages: [...prev.messages, optimisticUser, optimisticAssistant] }
        : prev
    );

    const chatId = targetSession.id;
    const scope = targetSession.scope || EMPTY_SCOPE;

    await runStreamIntoSession({
      chatId,
      targetAssistantId: tempAssistantId,
      optimisticUserId: tempUserId,
      invoke: (onEvent, signal) =>
        api.chats.streamMessage(
          chatId,
          {
            message: rawText,
            mode: effectiveMode,
            scope,
            enable_thinking: supportsThinking && thinkingMode,
          },
          onEvent,
          signal
        ),
    });
  };

  const lastReviewKeyRef = useRef<number | null>(null);

  useEffect(() => {
    if (!initialReviewTarget || lastReviewKeyRef.current === initialReviewTarget.key) return;
    lastReviewKeyRef.current = initialReviewTarget.key;

    const runReviewHandoff = async () => {
      const { chatSessionId, quizTitle, score, total } = initialReviewTarget;
      onClearInitialReview?.();

      let targetSession = activeSession;
      if (chatSessionId && (!targetSession || targetSession.id !== chatSessionId)) {
        try {
          const detail = await api.chats.get(chatSessionId);
          if (isMountedRef.current) {
            setActiveSession(detail);
            syncSessionIntoList(detail);
            targetSession = detail;
          }
        } catch {
          // fallback
        }
      } else if (!chatSessionId) {
        try {
          const cleanTitle = `Ripasso: ${quizTitle.replace(/^Quiz:\s*/i, '')}`.slice(0, 100);
          const fresh = await api.chats.create(subject.id, { title: cleanTitle });
          if (isMountedRef.current) {
            setActiveSession(fresh);
            syncSessionIntoList(fresh, true);
            targetSession = fresh;
          }
        } catch (err: any) {
          console.warn('[Assistente] Errore avvio sessione ripasso:', err);
        }
      }

      const percent = total > 0 ? Math.round((score / total) * 100) : 0;
      let reviewPrompt = '';
      if (score === total) {
        reviewPrompt = `Ho completato il quiz "${quizTitle}" con il punteggio massimo (${score}/${total}, 100%)! Quali argomenti correlati possiamo approfondire per prepararmi all'esame?`;
      } else {
        reviewPrompt = `Ho completato il quiz "${quizTitle}" con un punteggio di ${score} su ${total} (${percent}%). Aiutami a ripassare dove ho sbagliato e a capire i concetti chiave.`;
      }

      setTimeout(() => {
        if (isMountedRef.current) {
          void handleSend(reviewPrompt, 'auto', targetSession || undefined);
        }
      }, 150);
    };

    void runReviewHandoff();
  }, [initialReviewTarget, activeSession, subject.id, onClearInitialReview, handleSelectSession]);

  const handleRegenerate = async (messageId: number) => {
    if (!activeSession || streamingMessageId !== null || messageId <= 0) return;
    const prevMessage = activeSession.messages.find((m) => m.id === messageId);
    if (!prevMessage) return;

    const chatId = activeSession.id;
    regeneratingSnapshotRef.current = { chatId, messageId, prevMessage };

    setStreamingMessageId(messageId);
    setActiveSession((prev) => {
      if (!prev || prev.id !== chatId) return prev;
      return {
        ...prev,
        messages: prev.messages.map((m) =>
          m.id === messageId
            ? {
                ...m,
                content: '',
                thinking: '',
                citations: [],
                action: null,
                followups: [],
              }
            : m
        ),
      };
    });

    await runStreamIntoSession({
      chatId,
      targetAssistantId: messageId,
      prevMessage,
      invoke: (onEvent, signal) =>
        api.chats.regenerateStream(
          chatId,
          messageId,
          onEvent,
          signal,
          supportsThinking && thinkingMode
        ),
    });
  };

  const handleEditMessage = async (messageId: number, newText: string, rollbackQuiz: boolean = true) => {
    const text = newText.trim();
    if (!activeSession || streamingMessageId !== null || messageId <= 0 || !text) return;

    const userIdx = activeSession.messages.findIndex((m) => m.id === messageId);
    if (userIdx === -1) return;

    const userMsg = activeSession.messages[userIdx];
    const tempAssistantId = -Date.now();
    const nowIso = new Date().toISOString();
    clientKeyMapRef.current.set(tempAssistantId, `client-msg-${tempAssistantId}`);
    nearBottomRef.current = true;

    const updatedUserMsg: PersistentChatMessage = {
      ...userMsg,
      content: text,
    };

    const optimisticAssistant: PersistentChatMessage = {
      id: tempAssistantId,
      session_id: activeSession.id,
      role: 'assistant',
      content: '',
      thinking: '',
      mode: userMsg.mode || studyMode,
      context_used: '',
      citations: [],
      action: null,
      followups: [],
      token_estimate: 0,
      created_at: nowIso,
    };

    // Optimistically truncate all messages subsequent to edited user message
    const truncatedMessages = [
      ...activeSession.messages.slice(0, userIdx),
      updatedUserMsg,
      optimisticAssistant,
    ];

    setStreamingMessageId(tempAssistantId);
    setActiveSession((prev) =>
      prev && prev.id === activeSession.id
        ? { ...prev, messages: truncatedMessages }
        : prev
    );

    const chatId = activeSession.id;
    const scope = activeSession.scope || EMPTY_SCOPE;
    const effectiveMode = (userMsg.mode as ChatStudyMode) || studyMode;

    await runStreamIntoSession({
      chatId,
      targetAssistantId: tempAssistantId,
      optimisticUserId: messageId,
      invoke: (onEvent, signal) =>
        api.chats.editAndStreamMessage(
          chatId,
          messageId,
          {
            message: text,
            mode: effectiveMode,
            scope,
            enable_thinking: supportsThinking && thinkingMode,
            rollback_quiz: rollbackQuiz,
          },
          onEvent,
          signal
        ),
    });
  };

  const handleStopStream = () => {
    const activeMsgId = streamingMessageId;
    const sessionIdToRefresh = activeSession?.id ?? activeSessionIdRef.current;
    const regenSnap = regeneratingSnapshotRef.current;
    regeneratingSnapshotRef.current = null;
    abortActiveStream();

    if (regenSnap && regenSnap.messageId === activeMsgId) {
      const currentMsg = activeSession?.messages.find((m) => m.id === activeMsgId);
      const hasPartial = Boolean(currentMsg?.content.trim());

      setActiveSession((prev) => {
        if (!prev || prev.id !== regenSnap.chatId) return prev;
        return {
          ...prev,
          messages: prev.messages.map((m) =>
            m.id === regenSnap.messageId ? regenSnap.prevMessage : m
          ),
        };
      });

      if (hasPartial) {
        showToast('Rigenerazione interrotta — risposta salvata ripristinata');
        void api.chats
          .get(regenSnap.chatId)
          .then((detail) => {
            if (!isMountedRef.current || activeSessionIdRef.current !== detail.id) return;
            setActiveSession(detail);
            syncSessionIntoList(detail);
          })
          .catch(() => {});
      }
      api.ai.stop(true).catch(() => {});
      return;
    }

    if (activeMsgId !== null) {
      setActiveSession((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          messages: prev.messages.map((m) =>
            m.id === activeMsgId
              ? {
                  ...m,
                  content: m.content
                    ? `${m.content}\n\n*(Generazione interrotta)*`
                    : 'Generazione interrotta su richiesta.',
                }
              : m
          ),
        };
      });
    }
    api.ai.stop(true).catch(() => {});

    // Quietly refresh persisted messages so any negative optimistic temp IDs
    // get replaced with real positive SQLite IDs (re-enabling Regenerate/Delete/SaveNote).
    if (sessionIdToRefresh != null) {
      const refreshToken = streamTokenRef.current;
      stopSyncTimerRef.current = setTimeout(() => {
        stopSyncTimerRef.current = null;
        if (
          !isMountedRef.current ||
          activeSessionIdRef.current !== sessionIdToRefresh ||
          streamTokenRef.current !== refreshToken
        ) {
          return;
        }
        void api.chats
          .get(sessionIdToRefresh)
          .then((detail) => {
            if (
              !isMountedRef.current ||
              activeSessionIdRef.current !== detail.id ||
              streamTokenRef.current !== refreshToken
            ) {
              return;
            }
            setActiveSession((prev) => {
              if (!prev || prev.id !== detail.id) return prev;
              if (detail.messages.length >= prev.messages.length) {
                return detail;
              }
              const persistedByRoleAndContent = new Map<string, PersistentChatMessage>();
              for (const m of detail.messages) {
                persistedByRoleAndContent.set(`${m.role}:${m.content}`, m);
              }
              const mergedMessages = prev.messages.map((m) => {
                if (m.id > 0) {
                  return detail.messages.find((dm) => dm.id === m.id) || m;
                }
                const matched = persistedByRoleAndContent.get(`${m.role}:${m.content}`);
                return matched || m;
              });
              return {
                ...detail,
                messages: mergedMessages,
              };
            });
            syncSessionIntoList(detail);
          })
          .catch(() => {});
      }, 250);
    }
  };

  const handleSaveNote = async (messageId: number) => {
    if (!activeSession || messageId <= 0) return;
    setSavingNoteId(messageId);
    try {
      const res = await api.chats.saveNote(activeSession.id, { message_id: messageId });
      if (!isMountedRef.current) return;
      setSavedNoteIds((prev) => ({ ...prev, [messageId]: true }));
      onLectureCreated?.();
      showToast(`Salvato nel Quaderno: "${res.title}"`);
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Impossibile salvare nel Quaderno', 'error');
      }
    } finally {
      if (isMountedRef.current) {
        setSavingNoteId(null);
      }
    }
  };

  const handleDeleteMessage = async (messageId: number) => {
    if (!activeSession || messageId <= 0 || streamingMessageId !== null) return;
    const msgIdx = activeSession.messages.findIndex((m) => m.id === messageId);
    if (msgIdx === -1) return;

    const previousMessages = activeSession.messages;
    const previousContextStats = activeSession.context_stats;
    const truncatedMessages = activeSession.messages.slice(0, msgIdx);

    // Immediate optimistic update truncating the message and all subsequent messages
    setActiveSession((prev) =>
      prev && prev.id === activeSession.id
        ? { ...prev, messages: truncatedMessages }
        : prev
    );
    setDeletingMessageId(messageId);

    try {
      const updated = await api.chats.deleteMessage(activeSession.id, messageId);
      if (!isMountedRef.current) return;
      if (activeSessionIdRef.current === updated.id) {
        setActiveSession(updated);
      }
      syncSessionIntoList(updated);
    } catch (err: any) {
      if (isMountedRef.current) {
        // Rollback optimistic update on failure
        setActiveSession((prev) =>
          prev && prev.id === activeSession.id
            ? { ...prev, messages: previousMessages, context_stats: previousContextStats }
            : prev
        );
        showToast(err?.message || 'Impossibile eliminare il messaggio', 'error');
      }
    } finally {
      if (isMountedRef.current) {
        setDeletingMessageId(null);
      }
    }
  };

  const handleCompactContext = async () => {
    if (!activeSession || isCompacting || streamingMessageId !== null) return;
    const prevCompactedId = activeSession.compacted_up_to_message_id || 0;
    const prevCompactedCount = activeSession.context_stats?.compacted_messages_count || 0;
    setContextMenuOpen(false);
    setIsCompacting(true);
    try {
      const updated = await api.chats.compact(activeSession.id);
      if (!isMountedRef.current) return;
      if (activeSessionIdRef.current === updated.id) {
        setActiveSession(updated);
      }
      syncSessionIntoList(updated);
      const nextCompactedId = updated.compacted_up_to_message_id || 0;
      const nextCompactedCount = updated.context_stats?.compacted_messages_count || 0;
      if (nextCompactedId > prevCompactedId || nextCompactedCount > prevCompactedCount) {
        showToast('Contesto compattato con successo');
      } else {
        showToast('Servono almeno 3 messaggi attivi per eseguire la compattazione.');
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Impossibile compattare il contesto', 'error');
      }
    } finally {
      if (isMountedRef.current) {
        setIsCompacting(false);
      }
    }
  };

  const openArchiveModal = () => {
    if (!activeSession) return;
    setContextMenuOpen(false);
    setArchiveTitle(
      activeSession.title && activeSession.title !== 'Nuova chat'
        ? activeSession.title
        : `${subject.name} — Appunti chat`
    );
    setArchiveMode('quaderno_and_clear');
    setArchiveModalOpen(true);
  };

  const handleConfirmArchive = async () => {
    if (!activeSession || isArchiving) return;
    setIsArchiving(true);
    try {
      const res = await api.chats.archive(activeSession.id, {
        title: archiveTitle.trim() || undefined,
        mode: archiveMode,
      });
      if (!isMountedRef.current) return;
      if (activeSessionIdRef.current === res.session.id) {
        setActiveSession(res.session);
      }
      syncSessionIntoList(res.session);
      setArchiveModalOpen(false);
      onLectureCreated?.();
      showToast(
        archiveMode === 'quaderno_and_clear'
          ? `Archiviato in "${res.title}" e chat pulita`
          : `Copia salvata in "${res.title}"`
      );
    } catch (err: any) {
      if (isMountedRef.current) {
        showToast(err?.message || 'Impossibile archiviare nel Quaderno', 'error');
      }
    } finally {
      if (isMountedRef.current) {
        setIsArchiving(false);
      }
    }
  };

  const hasInstalledLLM = Boolean(settings.llmDownloaded);

  const contextStats = activeSession?.context_stats || DEFAULT_CONTEXT_STATS;
  const isZeroSaturation =
    !hasInstalledLLM ||
    !contextStats.used_tokens ||
    !contextStats.percentage ||
    contextStats.used_tokens === 0 ||
    contextStats.percentage === 0;
  const saturationPct = isZeroSaturation
    ? 0
    : Math.min(100, Math.max(0, Math.round(contextStats.percentage || 0)));
  const saturationTone = getSaturationTone(saturationPct);
  const showSaturationBanner =
    !isZeroSaturation && (contextStats.needs_compaction || saturationPct >= 75);

  const messages = activeSession?.messages || [];
  const compactedUpToId = activeSession?.compacted_up_to_message_id || 0;

  // Determine where to place the compaction divider: strictly at the end of compacted messages
  const lastCompactedIndex = useMemo(() => {
    if (!activeSession?.compacted_summary) return -1;
    if (messages.length === 0) return -1;

    if (compactedUpToId > 0) {
      let idx = -1;
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].id > 0 && messages[i].id <= compactedUpToId) {
          idx = i;
        }
      }
      if (idx !== -1) return idx;
      return messages.length - 1;
    }

    return messages.length - 1;
  }, [messages, compactedUpToId, activeSession?.compacted_summary]);

  const lastAssistantMessageId = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') return messages[i].id;
    }
    return null;
  }, [messages]);

  return (
    <div className="flex-1 flex h-full overflow-hidden content-canvas select-none relative">
      {/* Collapsible Multi-Chat Sidebar */}
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSession?.id ?? null}
        isOpen={sidebarOpen}
        isLoading={isLoadingSessions}
        isCreating={isCreatingSession}
        onSelectSession={(id) => void handleSelectSession(id)}
        onCreateSession={() => void handleCreateSession()}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
      />

      {/* Main Chat Column */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden relative">
        {/* Top Bar */}
        <header className="px-4 sm:px-6 py-2.5 border-b border-white/[0.06] flex items-center justify-between gap-3 flex-shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <button
              type="button"
              onClick={() => setSidebarOpen((prev) => !prev)}
              title={sidebarOpen ? 'Nascondi cronologia chat' : 'Mostra cronologia chat'}
              className="p-1.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.09] border border-white/[0.08] text-zinc-300 hover:text-white transition-colors cursor-pointer flex-shrink-0"
            >
              <PanelLeft size={15} />
            </button>
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-200 truncate">
                <span className="truncate">{activeSession?.title || 'Nuova chat'}</span>
                <span className="text-zinc-600 hidden sm:inline">·</span>
                <span className="text-zinc-400 font-normal truncate hidden sm:inline">
                  {subject.name}
                </span>
              </div>
            </div>
          </div>

          {/* Spazio Chat (Visibile SOLO quando un modello LLM è installato) */}
          {hasInstalledLLM && (
            <div ref={contextMenuRef} className="relative flex items-center gap-2 flex-shrink-0">
              <motion.button
                type="button"
                layout
                transition={{
                  type: 'spring',
                  stiffness: 420,
                  damping: 32,
                  mass: 0.8,
                }}
                onClick={() => setContextMenuOpen((prev) => !prev)}
                title={contextMenuOpen ? "Chiudi (Esc)" : "Stato della memoria e opzioni per liberare spazio"}
                className={clsx(
                  'relative h-7 rounded-full flex items-center justify-center select-none cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40',
                  contextMenuOpen
                    ? 'w-7 liquid-glass border border-white/[0.22] text-white hover:bg-white/[0.14] shadow-liquid-sm'
                    : clsx('px-2.5 border text-[11px] font-medium hover:brightness-110 shadow-sm', saturationTone.pill)
                )}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.96 }}
              >
                <AnimatePresence mode="popLayout" initial={false}>
                  {contextMenuOpen ? (
                    <motion.div
                      key="close"
                      initial={{ opacity: 0, scale: 0.85, rotate: -45 }}
                      animate={{ opacity: 1, scale: 1, rotate: 0 }}
                      exit={{ opacity: 0, scale: 0.85, rotate: 45 }}
                      transition={{ duration: 0.16, ease: 'easeOut' }}
                      className="w-7 h-7 flex items-center justify-center flex-shrink-0"
                    >
                      <X size={13} strokeWidth={2.3} className="text-white" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="pill"
                      initial={{ opacity: 0, scale: 0.88 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.88 }}
                      transition={{ duration: 0.16, ease: 'easeOut' }}
                      className="flex items-center gap-1.5 whitespace-nowrap overflow-hidden flex-shrink-0"
                    >
                      {/* Indicatore circolare SVG che si riempie */}
                      <div className="relative w-3.5 h-3.5 flex items-center justify-center flex-shrink-0">
                        <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                          <circle
                            cx="18"
                            cy="18"
                            r="14"
                            fill="none"
                            stroke="rgba(255, 255, 255, 0.16)"
                            strokeWidth="4"
                          />
                          <circle
                            cx="18"
                            cy="18"
                            r="14"
                            fill="none"
                            stroke={
                              saturationPct < 60
                                ? '#34d399' // Smeraldo / Verde
                                : saturationPct < 85
                                ? '#fbbf24' // Ambra / Giallo
                                : '#f43f5e' // Rosa-Rosso
                            }
                            strokeWidth="4"
                            strokeDasharray="87.96"
                            strokeDashoffset={Math.max(0, 87.96 - (87.96 * saturationPct) / 100)}
                            strokeLinecap="round"
                            className="transition-all duration-500 ease-out"
                          />
                        </svg>
                      </div>

                      <span>Spazio</span>
                      <span className="tabular-nums text-white/70 font-normal">
                        {saturationPct}%
                      </span>
                      <ChevronDown
                        size={11}
                        className="opacity-60"
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.button>

              {/* Menu a Bolla Liquid Glass con animazione spring standard */}
              <AnimatePresence>
                {contextMenuOpen && (
                  <motion.div
                    variants={liquidMenuVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                    style={{ transformOrigin: 'top right' }}
                    className={clsx(
                      "absolute right-0 top-full mt-2 w-80 max-w-[calc(100vw-2rem)]",
                      liquidPopoverContainerClass,
                      "p-3.5 space-y-3"
                    )}
                  >
                    {/* Header memoria */}
                    <div className="px-1 py-1 space-y-2">
                      <div className="flex items-center justify-between text-xs font-semibold text-white">
                        <span>Memoria della chat</span>
                        <span className="tabular-nums font-mono text-white/80">
                          {saturationPct}%
                        </span>
                      </div>

                      {/* Barra di riempimento precisa */}
                      <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden p-0.5">
                        <div
                          className={clsx(
                            'h-full rounded-full transition-all duration-300',
                            saturationTone.bar
                          )}
                          style={{ width: `${Math.max(4, saturationPct)}%` }}
                        />
                      </div>

                      <p className="text-[11px] text-white/60 leading-relaxed bg-white/[0.03] p-2.5 rounded-2xl border border-white/[0.06]">
                        {saturationPct >= 80
                          ? "La memoria è quasi piena. Libera spazio o archivia la chat nel Quaderno per continuare a ricevere risposte accurate."
                          : "Man mano che la conversazione cresce, la memoria si riempie. Puoi compattarla o salvarla nel Quaderno in ogni momento."}
                      </p>
                    </div>

                    {/* Azioni rapide */}
                    <div className="space-y-1.5 border-t border-white/[0.08] pt-2.5">
                      <button
                        type="button"
                        disabled={
                          !activeSession || isCompacting || streamingMessageId !== null || messages.length === 0
                        }
                        onClick={() => void handleCompactContext()}
                        className="w-full flex items-start gap-3 p-2.5 rounded-2xl text-left text-zinc-100 hover:bg-white/[0.08] transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed group"
                      >
                        <div className="w-8 h-8 rounded-xl bg-indigo-500/20 border border-indigo-400/30 flex items-center justify-center text-indigo-300 flex-shrink-0 mt-0.5 group-hover:scale-105 transition-transform">
                          {isCompacting ? (
                            <Loader2 size={15} className="animate-spin text-indigo-300" />
                          ) : (
                            <Layers size={15} />
                          )}
                        </div>
                        <div className="min-w-0">
                          <div className="text-xs font-semibold text-white">Libera memoria (Compatta)</div>
                          <div className="text-[11px] text-white/50 leading-snug mt-0.5">
                            Riassume i messaggi precedenti e fa spazio per continuare a chattare
                          </div>
                        </div>
                      </button>

                      <button
                        type="button"
                        disabled={!activeSession || streamingMessageId !== null || messages.length === 0}
                        onClick={openArchiveModal}
                        className="w-full flex items-start gap-3 p-2.5 rounded-2xl text-left text-zinc-100 hover:bg-white/[0.08] transition-all cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed group"
                      >
                        <div className="w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-300 flex-shrink-0 mt-0.5 group-hover:scale-105 transition-transform">
                          <BookmarkPlus size={15} />
                        </div>
                        <div className="min-w-0">
                          <div className="text-xs font-semibold text-white">Salva nel Quaderno</div>
                          <div className="text-[11px] text-white/50 leading-snug mt-0.5">
                            Trasforma la discussione in un appunto del corso.
                          </div>
                        </div>
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}
        </header>

        {/* Apple Liquid Glass Subtle Context Saturation Pill */}
        {showSaturationBanner && (
          <div className="px-6 py-2 bg-white/[0.02] border-b border-white/[0.06] flex items-center justify-between gap-3 text-xs text-zinc-400 flex-shrink-0 animate-fade-in backdrop-blur-sm select-none">
            <div className="flex items-center gap-2 min-w-0">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400/90 flex-shrink-0 animate-pulse" />
              <span className="truncate">
                Memoria conversazione quasi piena ({saturationPct}%)
              </span>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                type="button"
                disabled={isCompacting || streamingMessageId !== null}
                onClick={() => void handleCompactContext()}
                className="px-2.5 py-1 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-zinc-200 hover:text-white border border-white/[0.10] text-[11px] font-medium transition-all cursor-pointer disabled:opacity-40"
              >
                {isCompacting ? 'Compattazione...' : 'Compatta'}
              </button>
              <button
                type="button"
                disabled={streamingMessageId !== null}
                onClick={openArchiveModal}
                className="text-[11px] text-zinc-400 hover:text-zinc-200 transition-colors cursor-pointer px-1.5 py-1"
              >
                Salva nel Quaderno
              </button>
            </div>
          </div>
        )}

        {/* Toast notification */}
        {toast && (
          <div className="pointer-events-none absolute top-14 left-1/2 -translate-x-1/2 z-40 animate-fade-in">
            <div
              className={clsx(
                'inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border shadow-xl backdrop-blur-md text-xs font-medium',
                toast.tone === 'error'
                  ? 'bg-rose-950/90 border-rose-500/40 text-rose-200'
                  : 'bg-[#121622]/[0.96] border-white/[0.14] text-emerald-200'
              )}
            >
              {toast.tone === 'error' ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}
              <span>{toast.text}</span>
            </div>
          </div>
        )}

        {/* Centered Reading Stream */}
        <div ref={scrollContainerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto [scrollbar-gutter:stable] py-6">
          <div className="max-w-3xl mx-auto w-full px-4 sm:px-6">
            {isChatStreamLoading || showChatStreamSkeleton ? (
              showChatStreamSkeleton ? (
                <ChatStreamSkeleton />
              ) : (
                <div className="min-h-[240px]" aria-busy="true" />
              )
            ) : messages.length === 0 ? (
              /* Empty Welcome State: Pure ChatGPT 2026 Style */
              <div className="py-24 sm:py-32 flex flex-col items-center text-center animate-fade-in max-w-md mx-auto select-none">
                <h2 className="text-2xl sm:text-3xl font-medium tracking-tight text-zinc-100">
                  Da dove cominciamo?
                </h2>
                <p className="text-xs sm:text-sm text-zinc-400 mt-2.5 leading-relaxed font-normal">
                  Chiedi spiegazioni sulle lezioni di <span className="text-zinc-200 font-medium">{subject.name}</span>, approfondisci i materiali o crea un quiz.
                </p>
                {!hasInstalledLLM && onOpenSettings && (
                  <button
                    type="button"
                    onClick={onOpenSettings}
                    className="mt-4 px-3.5 py-1.5 rounded-full bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/30 text-amber-200 text-xs font-medium transition-all flex items-center gap-2 cursor-pointer shadow-sm select-none"
                  >
                    <Cpu size={13} className="text-amber-300" />
                    <span>Scarica un modello LLM nelle Impostazioni</span>
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-6 pb-6">
                {messages.map((msg, idx) => {
                  const itemKey = clientKeyMapRef.current.get(msg.id) || `msg-${msg.id}`;
                  const subsequentMessages = messages.slice(idx + 1);
                  const hasSubsequentQuizActions = subsequentMessages.some(
                    (m) => m.role === 'assistant' && m.action && (m.action.type === 'quiz_created' || m.action.type === 'quiz_deleted')
                  );
                  return (
                    <div key={itemKey}>
                      <ChatMessageItem
                        message={msg}
                        isStreamingThis={streamingMessageId === msg.id}
                        isAnyStreaming={streamingMessageId !== null}
                        isLastAssistant={msg.id === lastAssistantMessageId}
                        isSavedToQuaderno={Boolean(savedNoteIds[msg.id])}
                        isSavingNote={savingNoteId === msg.id}
                        isDeleting={deletingMessageId === msg.id}
                        hasSubsequentQuizActions={hasSubsequentQuizActions}
                        onOpenCitation={onOpenCitation}
                        onOpenAudioSnippet={onOpenAudioSnippet}
                        onOpenSettings={onOpenSettings}
                        onOpenQuiz={onOpenQuiz}
                        onOpenOralQuestion={onOpenOralQuestion}
                        onOpenQuadernoLecture={onOpenQuadernoLecture}
                        onRegenerate={(id) => void handleRegenerate(id)}
                        onSaveNote={(id) => void handleSaveNote(id)}
                        onDelete={(id) => void handleDeleteMessage(id)}
                        onEdit={(id, newText, rollbackQuiz) => void handleEditMessage(id, newText, rollbackQuiz)}
                      />
                      {idx === lastCompactedIndex && (
                        <CompactionDivider summary={activeSession?.compacted_summary} />
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            <div ref={streamEndRef} />
          </div>
        </div>

        {/* Floating ChatGPT 2026 Pill Composer */}
        <ChatComposer
          value={input}
          onChange={setInput}
          onSend={() => void handleSend()}
          onStop={handleStopStream}
          isStreaming={streamingMessageId !== null}
          disabled={isLoadingSessions && !activeSession}
          subjectName={subject.name}
          studyMode={studyMode}
          onStudyModeChange={setStudyMode}
          thinkingMode={thinkingMode}
          onThinkingModeChange={setThinkingMode}
          supportsThinking={supportsThinking}
          scope={activeSession?.scope || pendingScope || EMPTY_SCOPE}
          onScopeChange={(nextScope) => void handleScopeChange(nextScope)}
          lectures={lectures}
          materials={materials}
        />
      </div>

      {/* Salva nel Quaderno Modal */}
      {archiveModalOpen && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget && !isArchiving) {
              setArchiveModalOpen(false);
            }
          }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-md p-4 animate-fade-in select-none"
        >
          <div className="w-full max-w-md rounded-2xl liquid-modal border border-white/[0.14] shadow-liquid-lg p-5 space-y-4 animate-scale-up">
            <div className="flex items-start justify-between gap-3 border-b border-white/[0.08] pb-3">
              <div>
                <h3 className="text-sm font-semibold text-white">
                  Salva nel Quaderno
                </h3>
                <p className="text-xs text-white/50 mt-0.5">
                  Esporta la conversazione negli appunti del corso.
                </p>
              </div>
              <button
                type="button"
                onClick={() => !isArchiving && setArchiveModalOpen(false)}
                className="w-7 h-7 rounded-xl flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
              >
                <X size={15} />
              </button>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-white/80">
                Titolo
              </label>
              <input
                type="text"
                value={archiveTitle}
                onChange={(e) => setArchiveTitle(e.target.value)}
                placeholder="Es. Sintesi della discussione"
                className="w-full rounded-xl liquid-input px-3.5 py-2.5 text-xs text-white placeholder-white/30 focus:outline-none select-text"
              />
            </div>

            <div className="space-y-2 pt-1">
              {/* Opzione 1: Salva e pulisci chat */}
              <div
                role="button"
                tabIndex={0}
                onClick={() => setArchiveMode('quaderno_and_clear')}
                className={clsx(
                  'w-full p-3 rounded-xl border text-left transition-all cursor-pointer flex items-start gap-3 select-none',
                  archiveMode === 'quaderno_and_clear'
                    ? 'bg-emerald-500/[0.12] border-emerald-400/40 text-white shadow-liquid-xs'
                    : 'bg-white/[0.03] border-white/[0.08] text-white/70 hover:bg-white/[0.06] hover:border-white/[0.14]'
                )}
              >
                <div className={clsx(
                  'w-4 h-4 rounded-full border mt-0.5 flex items-center justify-center flex-shrink-0 transition-colors',
                  archiveMode === 'quaderno_and_clear'
                    ? 'border-emerald-400 bg-emerald-400'
                    : 'border-white/30 bg-transparent'
                )}>
                  {archiveMode === 'quaderno_and_clear' && (
                    <div className="w-1.5 h-1.5 rounded-full bg-black" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-semibold text-white">
                    Salva e pulisci chat
                  </div>
                  <div className="text-[11px] text-white/50 mt-0.5 leading-relaxed">
                    Crea il capitolo nel Quaderno e azzera la chat per continuare.
                  </div>
                </div>
              </div>

              {/* Opzione 2: Salva una copia e mantieni chat */}
              <div
                role="button"
                tabIndex={0}
                onClick={() => setArchiveMode('quaderno_keep')}
                className={clsx(
                  'w-full p-3 rounded-xl border text-left transition-all cursor-pointer flex items-start gap-3 select-none',
                  archiveMode === 'quaderno_keep'
                    ? 'bg-emerald-500/[0.12] border-emerald-400/40 text-white shadow-liquid-xs'
                    : 'bg-white/[0.03] border-white/[0.08] text-white/70 hover:bg-white/[0.06] hover:border-white/[0.14]'
                )}
              >
                <div className={clsx(
                  'w-4 h-4 rounded-full border mt-0.5 flex items-center justify-center flex-shrink-0 transition-colors',
                  archiveMode === 'quaderno_keep'
                    ? 'border-emerald-400 bg-emerald-400'
                    : 'border-white/30 bg-transparent'
                )}>
                  {archiveMode === 'quaderno_keep' && (
                    <div className="w-1.5 h-1.5 rounded-full bg-black" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-semibold text-white">
                    Salva una copia e mantieni chat
                  </div>
                  <div className="text-[11px] text-white/50 mt-0.5 leading-relaxed">
                    Esporta gli appunti nel Quaderno mantenendo intatta la chat.
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-white/[0.08]">
              <button
                type="button"
                disabled={isArchiving}
                onClick={() => setArchiveModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer disabled:opacity-50"
              >
                Annulla
              </button>
              <button
                type="button"
                disabled={isArchiving}
                onClick={() => void handleConfirmArchive()}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-white text-[#0d1017] hover:bg-white/90 text-xs font-semibold transition-all cursor-pointer shadow-liquid-sm disabled:opacity-50"
              >
                {isArchiving && <Loader2 size={13} className="animate-spin" />}
                <span>{isArchiving ? 'Salvataggio...' : 'Salva nel Quaderno'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
