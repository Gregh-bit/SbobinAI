import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  PanelLeft,
  Archive,
  BookmarkPlus,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  X,
  Loader2,
  Layers,
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
import ChatSidebar from './chat/ChatSidebar';
import ChatMessageItem, { parseThinkingContent } from './chat/ChatMessageItem';
import ChatComposer from './chat/ChatComposer';

interface AssistenteCorsoViewProps {
  subject: Subject;
  lectures: Lecture[];
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenQuiz?: (quizId: number) => void;
  onOpenOralQuestion?: (question: string, topic?: string) => void;
  onLectureCreated?: () => void;
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
      pill: 'bg-rose-500/12 border-rose-500/30 text-rose-200',
      dot: 'bg-rose-400',
    };
  }
  if (percentage >= 65) {
    return {
      bar: 'bg-amber-400',
      pill: 'bg-amber-500/12 border-amber-500/30 text-amber-200',
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
    <div className="my-5 animate-fade-in">
      <div className="flex items-center gap-3">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-400/30 to-indigo-400/20" />
        <button
          type="button"
          onClick={() => summary && setExpanded((prev) => !prev)}
          className={clsx(
            'inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-400/25 text-[11px] font-medium text-indigo-200/90 transition-colors',
            summary ? 'hover:bg-indigo-500/20 cursor-pointer' : 'cursor-default'
          )}
        >
          <Archive size={11} className="text-indigo-300" />
          <span>Contesto precedente compattato — Riassunto attivo in memoria</span>
          {summary &&
            (expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />)}
        </button>
        <div className="h-px flex-1 bg-gradient-to-l from-transparent via-indigo-400/30 to-indigo-400/20" />
      </div>

      {expanded && summary && (
        <div className="mt-2.5 mx-auto max-w-2xl rounded-2xl bg-zinc-900/80 border border-indigo-400/20 px-4 py-3 text-xs text-zinc-300 leading-relaxed whitespace-pre-wrap select-text">
          {summary}
        </div>
      )}
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
  onOpenQuiz,
  onOpenOralQuestion,
  onLectureCreated,
}: AssistenteCorsoViewProps) {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 900 : true
  );
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<ChatSessionDetail | null>(null);
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);

  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isLoadingSessionDetail, setIsLoadingSessionDetail] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);

  const [input, setInput] = useState('');
  const [studyMode, setStudyMode] = useState<ChatStudyMode>('auto');
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

  const streamEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const streamTokenRef = useRef(0);
  const activeSessionIdRef = useRef<number | null>(null);
  const loadSeqRef = useRef(0);
  const contextMenuRef = useRef<HTMLDivElement>(null);
  const isMountedRef = useRef(true);
  const pendingScopeRef = useRef<ChatScope | null>(null);
  const regeneratingSnapshotRef = useRef<{
    chatId: number;
    messageId: number;
    prevMessage: PersistentChatMessage;
  } | null>(null);

  activeSessionIdRef.current = activeSession?.id ?? null;

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

  // Close archive modal on Escape key
  useEffect(() => {
    if (!archiveModalOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isArchiving) {
        setArchiveModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [archiveModalOpen, isArchiving]);

  const abortActiveStream = useCallback(() => {
    streamTokenRef.current += 1;
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
    setSessions([]);
    setActiveSession(null);
    setSavedNoteIds({});
    setIsLoadingSessions(true);

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
            const firstDetail = await api.chats.get(list[0].id);
            if (!cancelled) {
              const resolved = await applyPendingScopeIfAny(firstDetail);
              if (!cancelled) {
                setActiveSession(resolved);
                syncSessionIntoList(resolved);
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
          const initialScope = pendingScopeRef.current || undefined;
          updatePendingScope(null);
          const created = await api.chats.create(subject.id, initialScope ? { scope: initialScope } : undefined);
          if (!cancelled) {
            setSessions([toSessionSummary(created)]);
            setActiveSession(created);
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

  // Smooth auto-scroll when messages update or streaming starts
  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.id, activeSession?.messages.length, streamingMessageId]);

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
    if (activeSessionIdRef.current === sessionId) {
      abortActiveStream();
    }
    try {
      await api.chats.delete(sessionId);
      if (!isMountedRef.current) return;
      const remaining = sessions.filter((s) => s.id !== sessionId);
      setSessions(remaining);

      if (activeSessionIdRef.current === sessionId) {
        if (remaining.length > 0) {
          const nextDetail = await api.chats.get(remaining[0].id);
          if (!isMountedRef.current) return;
          setActiveSession(nextDetail);
        } else {
          const fresh = await api.chats.create(subject.id);
          if (!isMountedRef.current) return;
          setSessions([toSessionSummary(fresh)]);
          setActiveSession(fresh);
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

      try {
        await invoke((ev) => {
          if (!isCurrentContext()) return;

          if (ev.type === 'meta') {
            setActiveSession((prev) => {
              if (!prev || prev.id !== chatId) return prev;
              const updatedMsgs = prev.messages.map((m) => {
                if (optimisticUserId !== undefined && m.id === optimisticUserId) {
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
            return;
          }

          if (ev.type === 'action') {
            streamAction = ev.action;
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
            const finalizedMsg: PersistentChatMessage = {
              ...ev.assistant_message,
              followups:
                ev.assistant_message.followups?.length > 0
                  ? ev.assistant_message.followups
                  : ev.followups || [],
              action: ev.assistant_message.action ?? streamAction ?? null,
            };
            currentAssistantId = finalizedMsg.id;

            setActiveSession((prev) => {
              if (!prev || prev.id !== chatId) return prev;
              const nextDetail: ChatSessionDetail = {
                ...prev,
                title: ev.session_title || prev.title,
                context_stats: ev.context_stats,
                messages: prev.messages.map((m) =>
                  m.id === targetAssistantId || m.id === finalizedMsg.id ? finalizedMsg : m
                ),
              };
              syncSessionIntoList(nextDetail, true);
              return nextDetail;
            });
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

  const handleSend = async (overrideText?: string, overrideMode?: ChatStudyMode) => {
    const rawText = (overrideText ?? input).trim();
    const effectiveMode = overrideMode ?? studyMode;
    const text = rawText || (effectiveMode === 'quiz' ? 'Genera un quiz a scelta multipla sui temi principali del corso' : '');
    if (!text || streamingMessageId !== null) return;

    let targetSession = activeSession;

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

    const optimisticUser: PersistentChatMessage = {
      id: tempUserId,
      session_id: targetSession.id,
      role: 'user',
      content: text,
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

    if (!overrideText) setInput('');
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
          { message: text, mode: effectiveMode, scope },
          onEvent,
          signal
        ),
    });
  };

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
        api.chats.regenerateStream(chatId, messageId, onEvent, signal),
    });
  };

  const handleEditMessage = async (messageId: number, newText: string) => {
    const text = newText.trim();
    if (!activeSession || streamingMessageId !== null || messageId <= 0 || !text) return;

    const userIdx = activeSession.messages.findIndex((m) => m.id === messageId);
    if (userIdx === -1) return;

    const userMsg = activeSession.messages[userIdx];
    const tempAssistantId = -Date.now();
    const nowIso = new Date().toISOString();

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
          { message: text, mode: effectiveMode, scope },
          onEvent,
          signal
        ),
    });
  };

  const handleStopStream = () => {
    const activeMsgId = streamingMessageId;
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

  const { settings } = useSettings();
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
  const effectiveUsedTokens = isZeroSaturation ? 0 : contextStats.used_tokens;
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
                <span className="truncate">{activeSession?.title || 'Assistente del Corso'}</span>
                <span className="text-zinc-600 hidden sm:inline">·</span>
                <span className="text-zinc-400 font-normal truncate hidden sm:inline">
                  {subject.name}
                </span>
              </div>
            </div>
          </div>

          {/* Calm Default Context Saturation Pill + Actions */}
          <div ref={contextMenuRef} className="relative flex items-center gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={() => setContextMenuOpen((prev) => !prev)}
              title="Stato memoria contesto e azioni di compattazione"
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-full border text-[11px] font-medium transition-all cursor-pointer',
                saturationTone.pill,
                'hover:brightness-110'
              )}
            >
              <span className={clsx('w-1.5 h-1.5 rounded-full', saturationTone.dot)} />
              <div className="w-12 h-1.5 rounded-full bg-white/10 overflow-hidden hidden sm:block">
                <div
                  className={clsx('h-full rounded-full transition-all duration-300', saturationTone.bar)}
                  style={{ width: `${saturationPct}%` }}
                />
              </div>
              <span className="tabular-nums">
                {!hasInstalledLLM ? (
                  <>
                    <span className="text-zinc-400 font-normal hidden md:inline">Nessun LLM scaricato · </span>
                    0 / {contextStats.max_tokens.toLocaleString('it-IT')} (0%)
                  </>
                ) : (
                  `${effectiveUsedTokens.toLocaleString('it-IT')} / ${contextStats.max_tokens.toLocaleString('it-IT')} (${saturationPct}%)`
                )}
              </span>
              <ChevronDown size={12} className="opacity-70" />
            </button>

            {contextMenuOpen && (
              <div className="absolute right-0 top-full mt-2 w-72 rounded-2xl bg-zinc-900/95 border border-white/12 shadow-2xl backdrop-blur-xl p-2 z-40 animate-fade-in">
                <div className="px-2.5 py-1.5 border-b border-white/[0.07] mb-1">
                  <div className="flex items-center justify-between text-[11px] text-zinc-300 font-medium">
                    <span>Saturazione finestra contesto</span>
                    <span className="tabular-nums">{saturationPct}%</span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden mt-1.5">
                    <div
                      className={clsx('h-full rounded-full transition-all', saturationTone.bar)}
                      style={{ width: `${saturationPct}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-zinc-400 mt-1.5">
                    {!hasInstalledLLM
                      ? 'Nessun LLM scaricato — barra inattiva (0 token)'
                      : `${contextStats.active_messages_count} messaggi attivi${
                          contextStats.compacted_messages_count > 0
                            ? ` · ${contextStats.compacted_messages_count} compattati`
                            : ''
                        }`}
                  </p>
                </div>

                <button
                  type="button"
                  disabled={!activeSession || isCompacting || streamingMessageId !== null || messages.length === 0}
                  onClick={() => void handleCompactContext()}
                  className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-xl text-left text-zinc-200 hover:bg-white/[0.07] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {isCompacting ? (
                    <Loader2 size={14} className="mt-0.5 text-indigo-300 animate-spin flex-shrink-0" />
                  ) : (
                    <Layers size={14} className="mt-0.5 text-indigo-300 flex-shrink-0" />
                  )}
                  <div>
                    <div className="text-xs font-medium">Compatta contesto</div>
                    <div className="text-[10px] text-zinc-400 leading-snug">
                      Riassume i messaggi precedenti liberando token in memoria
                    </div>
                  </div>
                </button>

                <button
                  type="button"
                  disabled={!activeSession || streamingMessageId !== null || messages.length === 0}
                  onClick={openArchiveModal}
                  className="w-full flex items-start gap-2.5 px-2.5 py-2 rounded-xl text-left text-zinc-200 hover:bg-white/[0.07] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <BookmarkPlus size={14} className="mt-0.5 text-emerald-300 flex-shrink-0" />
                  <div>
                    <div className="text-xs font-medium">Salva nel Quaderno (DB) e Cancella</div>
                    <div className="text-[10px] text-zinc-400 leading-snug">
                      Esporta la chat come capitolo nel Quaderno e libera la finestra
                    </div>
                  </div>
                </button>
              </div>
            )}
          </div>
        </header>

        {/* Subtle Saturation Warning Banner when >= 75% or needs_compaction */}
        {showSaturationBanner && (
          <div className="px-4 sm:px-6 py-2 bg-amber-500/10 border-b border-amber-500/20 flex flex-wrap items-center justify-between gap-2 text-xs animate-fade-in">
            <div className="flex items-center gap-2 text-amber-200">
              <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
              <span>
                Memoria della conversazione al {saturationPct}%. Compatta il contesto o salva nel
                Quaderno per mantenere massima precisione.
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                disabled={isCompacting || streamingMessageId !== null}
                onClick={() => void handleCompactContext()}
                className="px-2.5 py-1 rounded-full bg-amber-500/20 hover:bg-amber-500/30 text-amber-100 border border-amber-400/30 text-[11px] font-medium transition-colors cursor-pointer disabled:opacity-50"
              >
                {isCompacting ? 'Compattazione...' : 'Compatta contesto'}
              </button>
              <button
                type="button"
                disabled={streamingMessageId !== null}
                onClick={openArchiveModal}
                className="px-2.5 py-1 rounded-full bg-white/[0.07] hover:bg-white/[0.12] text-zinc-100 border border-white/15 text-[11px] font-medium transition-colors cursor-pointer"
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
                  : 'bg-zinc-900/95 border-emerald-500/35 text-emerald-200'
              )}
            >
              {toast.tone === 'error' ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}
              <span>{toast.text}</span>
            </div>
          </div>
        )}

        {/* Centered Reading Stream */}
        <div className="flex-1 overflow-y-auto py-6">
          <div className="max-w-3xl mx-auto w-full px-4 sm:px-6">
            {isLoadingSessionDetail ? (
              <div className="py-16 flex flex-col items-center justify-center gap-3 text-zinc-400 text-xs">
                <Loader2 size={20} className="animate-spin text-emerald-400" />
                <span>Caricamento conversazione...</span>
              </div>
            ) : messages.length === 0 ? (
              /* Empty Welcome State: Pure ChatGPT 2026 Style */
              <div className="py-24 sm:py-32 flex flex-col items-center text-center animate-fade-in max-w-md mx-auto select-none">
                <h2 className="text-2xl sm:text-3xl font-medium tracking-tight text-zinc-100">
                  Da dove cominciamo?
                </h2>
                <p className="text-xs sm:text-sm text-zinc-400 mt-2.5 leading-relaxed font-normal">
                  Chiedi spiegazioni sulle lezioni di <span className="text-zinc-200 font-medium">{subject.name}</span>, approfondisci i materiali o crea un quiz.
                </p>
              </div>
            ) : (
              <div className="space-y-6 pb-4">
                {messages.map((msg, idx) => (
                  <div key={msg.id}>
                    <ChatMessageItem
                      message={msg}
                      isStreamingThis={streamingMessageId === msg.id}
                      isAnyStreaming={streamingMessageId !== null}
                      isLastAssistant={msg.id === lastAssistantMessageId}
                      isSavedToQuaderno={Boolean(savedNoteIds[msg.id])}
                      isSavingNote={savingNoteId === msg.id}
                      isDeleting={deletingMessageId === msg.id}
                      onOpenCitation={onOpenCitation}
                      onOpenQuiz={onOpenQuiz}
                      onOpenOralQuestion={onOpenOralQuestion}
                      onRegenerate={(id) => void handleRegenerate(id)}
                      onSaveNote={(id) => void handleSaveNote(id)}
                      onDelete={(id) => void handleDeleteMessage(id)}
                      onEdit={(id, newText) => void handleEditMessage(id, newText)}
                    />
                    {idx === lastCompactedIndex && (
                      <CompactionDivider summary={activeSession?.compacted_summary} />
                    )}
                  </div>
                ))}
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
          scope={activeSession?.scope || pendingScope || EMPTY_SCOPE}
          onScopeChange={(nextScope) => void handleScopeChange(nextScope)}
          lectures={lectures}
          materials={materials}
        />
      </div>

      {/* Lightweight Archive to Quaderno Modal */}
      {archiveModalOpen && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget && !isArchiving) {
              setArchiveModalOpen(false);
            }
          }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-sm p-4 animate-fade-in"
        >
          <div className="w-full max-w-md rounded-3xl bg-zinc-900 border border-white/12 shadow-2xl p-5 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-zinc-100">
                  Salva conversazione nel Quaderno
                </h3>
                <p className="text-xs text-zinc-400 mt-0.5">
                  L&apos;IA trasformerà la chat in un capitolo ordinato salvato nel DB del corso.
                </p>
              </div>
              <button
                type="button"
                onClick={() => !isArchiving && setArchiveModalOpen(false)}
                className="p-1.5 rounded-full text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.08] cursor-pointer"
              >
                <X size={15} />
              </button>
            </div>

            <div className="space-y-1.5">
              <label className="text-[11px] font-medium text-zinc-300">
                Titolo capitolo nel Quaderno
              </label>
              <input
                type="text"
                value={archiveTitle}
                onChange={(e) => setArchiveTitle(e.target.value)}
                placeholder="Es. Riepilogo Teorema di Bayes"
                className="w-full rounded-xl bg-zinc-950/80 border border-white/12 px-3 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-emerald-400/50 select-text"
              />
            </div>

            <div className="space-y-2">
              <button
                type="button"
                onClick={() => setArchiveMode('quaderno_and_clear')}
                className={clsx(
                  'w-full p-3 rounded-2xl border text-left transition-all cursor-pointer',
                  archiveMode === 'quaderno_and_clear'
                    ? 'bg-emerald-500/12 border-emerald-400/35 text-zinc-100'
                    : 'bg-white/[0.03] border-white/[0.08] text-zinc-300 hover:bg-white/[0.06]'
                )}
              >
                <div className="text-xs font-medium">Salva nel Quaderno e pulisci chat</div>
                <div className="text-[11px] text-zinc-400 mt-0.5">
                  Crea il capitolo nel Quaderno e azzera la finestra di contesto per continuare.
                </div>
              </button>

              <button
                type="button"
                onClick={() => setArchiveMode('quaderno_keep')}
                className={clsx(
                  'w-full p-3 rounded-2xl border text-left transition-all cursor-pointer',
                  archiveMode === 'quaderno_keep'
                    ? 'bg-emerald-500/12 border-emerald-400/35 text-zinc-100'
                    : 'bg-white/[0.03] border-white/[0.08] text-zinc-300 hover:bg-white/[0.06]'
                )}
              >
                <div className="text-xs font-medium">Salva una copia e mantieni chat</div>
                <div className="text-[11px] text-zinc-400 mt-0.5">
                  Esporta gli appunti nel Quaderno mantenendo intatta la conversazione attuale.
                </div>
              </button>
            </div>

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                disabled={isArchiving}
                onClick={() => setArchiveModalOpen(false)}
                className="px-4 py-2 rounded-full bg-white/[0.05] hover:bg-white/[0.10] text-xs font-medium text-zinc-300 transition-colors cursor-pointer disabled:opacity-50"
              >
                Annulla
              </button>
              <button
                type="button"
                disabled={isArchiving}
                onClick={() => void handleConfirmArchive()}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-emerald-500 hover:bg-emerald-400 text-zinc-950 text-xs font-semibold transition-colors cursor-pointer disabled:opacity-50"
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
