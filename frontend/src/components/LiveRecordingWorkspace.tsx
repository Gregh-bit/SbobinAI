import React, { useRef, useEffect, useLayoutEffect, useState, useCallback, useMemo } from 'react';
import {
  HelpCircle,
  MessageSquare,
  Sparkles,
  ArrowUp,
  Trash2,
  Square,
  ChevronRight,
  ChevronLeft,
  X,
  Eye,
  Code2,
  ExternalLink,
  Loader2,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { clsx } from 'clsx';
import type {
  ChatMessage,
  DetectedQuestion,
  MaterialPageOut,
  PersistentChatMessage,
  TranscriptChunk,
} from '../types';
import SegmentedControl from './SegmentedControl';
import MarkdownContent from './MarkdownContent';
import ChatMessageItem from './chat/ChatMessageItem';
import { SlidePageSkeleton } from './Skeleton';
import { api } from '../lib/api';

interface Props {
  subjectId?: number;
  subjectName?: string;
  chunks: TranscriptChunk[];
  isPaused?: boolean;
  audioStatus?: 'idle' | 'listening' | 'transcribing';
  detectedQuestions: DetectedQuestion[];
  onDeleteQuestion?: (id: string) => void;
  onAnswerQuestion?: (id: string) => void;
  answeringQuestionIds?: Set<string>;
  chatMessages: ChatMessage[];
  chatInput: string;
  setChatInput: (input: string) => void;
  isSendingChat: boolean;
  handleChatSend: (text?: string, options?: { enableThinking?: boolean }) => void;
  handleCancelChat?: () => void;
  inspectorTab: 'domande' | 'chat';
  setInspectorTab: (tab: 'domande' | 'chat') => void;
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenSettings?: () => void;
  onLectureCreated?: () => void;
}

// ── Isolamento Rendering Live: Contenitore Trascrizione Lineare con Macchina da Scrivere ────────────
interface LiveTranscriptContainerProps {
  chunks: TranscriptChunk[];
  isPaused: boolean;
  audioStatus: 'idle' | 'listening' | 'transcribing';
}

export const LiveTranscriptContainer = React.memo(function LiveTranscriptContainer({
  chunks,
  isPaused,
  audioStatus,
}: LiveTranscriptContainerProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef<boolean>(true);

  // Unione fluida e lineare di tutti i chunk senza forzare mai interruzioni di riga
  const fullTargetText = useMemo(() => {
    if (!chunks || chunks.length === 0) return '';
    let result = '';
    for (const c of chunks) {
      const raw = (c.text || '').trim();
      if (!raw) continue;
      if (!result) {
        result = raw;
      } else {
        const lastChar = result.slice(-1);
        const firstChar = raw.charAt(0);
        const isPunctuation = /^[.,?!:;]/.test(firstChar);
        if (!/\s$/.test(lastChar) && !isPunctuation) {
          result += ' ' + raw;
        } else {
          result += (isPunctuation ? '' : ' ') + raw;
        }
      }
    }
    return result;
  }, [chunks]);

  // Se chunks sono già presenti al montaggio iniziale, mostriamo direttamente il testo pregresso
  const [displayedLength, setDisplayedLength] = useState<number>(() => {
    return fullTargetText.length;
  });

  const displayedLengthRef = useRef<number>(displayedLength);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const charAccumulatorRef = useRef<number>(0);

  useEffect(() => {
    displayedLengthRef.current = displayedLength;
  }, [displayedLength]);

  // Rileva se l'utente è vicino al fondo o se sta scrollando manualmente per leggere
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceToBottom < 75;
  }, []);

  // Ciclo macchina da scrivere a scorrimento naturale sincronizzato al frame rate (60-120fps)
  useEffect(() => {
    const targetLen = fullTargetText.length;

    // Se il testo si riduce (es. rimozione duplicati o reset)
    if (displayedLengthRef.current > targetLen) {
      setDisplayedLength(targetLen);
      displayedLengthRef.current = targetLen;
      charAccumulatorRef.current = 0;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    // Se il ritardo è consistente (>300 caratteri, es. riconnessione di sessione attiva),
    // aggiorna subito alla fine senza far attendere l'utente
    if (targetLen - displayedLengthRef.current > 300) {
      setDisplayedLength(targetLen);
      displayedLengthRef.current = targetLen;
      charAccumulatorRef.current = 0;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    if (displayedLengthRef.current === targetLen) {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
      return;
    }

    const stepAnimation = (timestamp: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp;
      }
      const dt = Math.min(0.08, (timestamp - lastTimeRef.current) / 1000);
      lastTimeRef.current = timestamp;

      const remaining = targetLen - displayedLengthRef.current;
      if (remaining <= 0) {
        animFrameRef.current = null;
        charAccumulatorRef.current = 0;
        return;
      }

      // Cadenza battiti macchina da scrivere:
      // - Per battitura normale: ~34 caratteri/s (~29ms a lettera, cadenza naturale e piacevole)
      // - Per chunk corposi: accelera progressivamente fino a ~95 caratteri/s per non rimanere indietro
      const charsPerSec = Math.min(110, Math.max(34, remaining * 2.8));
      charAccumulatorRef.current += dt * charsPerSec;

      const step = Math.floor(charAccumulatorRef.current);
      if (step > 0) {
        charAccumulatorRef.current -= step;
        const nextLen = Math.min(targetLen, displayedLengthRef.current + step);
        displayedLengthRef.current = nextLen;
        setDisplayedLength(nextLen);
      }

      if (displayedLengthRef.current < targetLen) {
        animFrameRef.current = requestAnimationFrame(stepAnimation);
      } else {
        animFrameRef.current = null;
        charAccumulatorRef.current = 0;
      }
    };

    lastTimeRef.current = 0;
    if (!animFrameRef.current) {
      animFrameRef.current = requestAnimationFrame(stepAnimation);
    }

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
        animFrameRef.current = null;
      }
    };
  }, [fullTargetText]);

  const displayedText = useMemo(() => {
    if (displayedLength <= 0) return '';
    return fullTargetText.slice(0, displayedLength);
  }, [fullTargetText, displayedLength]);

  const isTyping = displayedLength < fullTargetText.length;

  // Auto-scroll fluido solo se l'utente si trova in fondo alla pagina (senza scroll thrashing)
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el || !isNearBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [displayedText, isPaused]);

  return (
    <section className="flex-1 flex flex-col h-full overflow-hidden relative pb-8 content-canvas">
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-12 py-10"
      >
        <div className="max-w-2xl mx-auto space-y-4">
          {/* Testo lineare unico e continuo (display inline), zero scritte di caricamento o interruzioni di riga forzate */}
          <div className="relative text-[16px] leading-[1.85] font-sans text-white/95 font-normal tracking-wide select-text break-words">
            <span>{displayedText}</span>

            {/* Cursore verticale stile Mac (macchina da scrivere): bianco minimal, nessun bagliore hover */}
            <span
              className={clsx(
                "inline-block pointer-events-none transition-opacity duration-150 align-baseline",
                (isTyping || audioStatus === 'transcribing') ? "opacity-100" : "animate-caret-blink"
              )}
              style={{
                width: '2px',
                height: '1.2em',
                backgroundColor: '#ffffff',
                boxShadow: 'none',
                borderRadius: '1px',
                marginLeft: displayedText ? '2.5px' : '0px',
                marginRight: '1px',
                verticalAlign: '-0.15em',
              }}
              aria-hidden="true"
            />
          </div>
        </div>
      </div>
    </section>
  );
});

// ── Adattatore ChatMessage -> PersistentChatMessage per riuso esatto di ChatMessageItem (ChatGPT 2026) ──
function toChatMessageItemModel(
  msg: ChatMessage,
  index: number,
  isStreamingThis: boolean
): PersistentChatMessage {
  return {
    id: isStreamingThis ? -(index + 1) : index + 1,
    session_id: 0,
    role: msg.role,
    content: msg.content,
    thinking: msg.thinking || '',
    mode: 'auto',
    citations: msg.citations ?? [],
    followups: [],
    context_used: msg.contextUsed || '',
    token_estimate: 0,
    created_at: new Date(msg.timestamp).toISOString(),
  };
}

function formatQuestionAnswer(raw: string): string {
  if (!raw) return '';
  const trimmed = raw.trim();
  if (
    trimmed.includes('modello LLM non ancora scaricato') ||
    trimmed.includes('Risposta non disponibile') ||
    trimmed.includes('modello LLM non installato') ||
    trimmed.includes('Scarica un modello LLM dalle Impostazioni')
  ) {
    return 'Scarica un modello LLM dalle [[action:settings|Impostazioni]] per attivare le risposte rapide.';
  }
  return raw;
}

// ── LiveRecordingWorkspace / LiveLecture Container ───────────────────────────
function LiveRecordingWorkspaceComponent({
  subjectId,
  subjectName,
  chunks,
  isPaused = false,
  audioStatus = 'idle',
  detectedQuestions,
  onDeleteQuestion,
  onAnswerQuestion,
  answeringQuestionIds,
  chatMessages,
  chatInput,
  setChatInput,
  isSendingChat,
  handleChatSend,
  handleCancelChat,
  inspectorTab,
  setInspectorTab,
  onOpenCitation,
  onOpenSettings,
}: Props) {
  const [isComposerMultiline, setIsComposerMultiline] = useState(false);
  const [isComposerOverflow, setIsComposerOverflow] = useState(false);

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatNearBottomRef = useRef(true);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const isMountedRef = useRef(true);
  const previewSeqRef = useRef(0);

  const [citationPreview, setCitationPreview] = useState<{
    sourceId: string;
    page: number;
    nonPaginatedNote?: string;
  } | null>(null);
  const [previewData, setPreviewData] = useState<MaterialPageOut | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewMode, setPreviewMode] = useState<'visual' | 'text'>('visual');

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Auto-grow textarea stile ChatGPT 2026 (senza flicker e senza scrollbar fantasma in Safari)
  useLayoutEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    if (!chatInput) {
      el.style.height = '24px';
      setIsComposerMultiline(false);
      setIsComposerOverflow(false);
      return;
    }
    el.style.height = 'auto';
    const scrollH = el.scrollHeight;
    if (scrollH <= 32 && !chatInput.includes('\n')) {
      el.style.height = '24px';
      setIsComposerMultiline(false);
      setIsComposerOverflow(false);
    } else {
      el.style.height = `${Math.min(scrollH, 120)}px`;
      setIsComposerMultiline(true);
      setIsComposerOverflow(scrollH > 120);
    }
  }, [chatInput, inspectorTab]);

  // Snap istantaneo al fondo aprendo la tab chat (zero animazione/flash dall'alto)
  useLayoutEffect(() => {
    if (inspectorTab !== 'chat') return;
    const el = chatScrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
      chatNearBottomRef.current = true;
    }
  }, [inspectorTab]);

  // Follow fluido durante l'arrivo di nuovi messaggi/token (con doppio frame per catturare il layout Framer Motion)
  useLayoutEffect(() => {
    if (inspectorTab !== 'chat') return;
    const el = chatScrollRef.current;
    if (!el || !chatNearBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
    const raf = requestAnimationFrame(() => {
      if (chatScrollRef.current && chatNearBottomRef.current) {
        chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
        if (chatEndRef.current) {
          chatEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [chatMessages, isSendingChat, inspectorTab]);

  const handleChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    chatNearBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 72;
  }, []);

  const sendLiveChat = useCallback(
    (overrideText?: string) => {
      chatNearBottomRef.current = true;
      handleChatSend(overrideText, { enableThinking: false });
    },
    [handleChatSend]
  );

  const closeCitationPreview = useCallback(() => {
    previewSeqRef.current += 1;
    setCitationPreview(null);
    setPreviewData(null);
    setPreviewError(null);
    setPreviewLoading(false);
  }, []);

  const fetchCitationPage = useCallback(
    async (sid: number, filename: string, page: number) => {
      const seq = ++previewSeqRef.current;
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const res = await api.subjects.getMaterialPage(sid, filename, page);
        if (!isMountedRef.current || seq !== previewSeqRef.current) return;
        setPreviewData(res);
        if (res.page && res.page !== page) {
          setCitationPreview((prev) => (prev ? { ...prev, page: res.page } : prev));
        }
      } catch (err) {
        if (!isMountedRef.current || seq !== previewSeqRef.current) return;
        setPreviewData(null);
        setPreviewError(err instanceof Error ? err.message : 'Impossibile caricare la pagina.');
      } finally {
        if (isMountedRef.current && seq === previewSeqRef.current) {
          setPreviewLoading(false);
        }
      }
    },
    []
  );

  const handleCitationOpen = useCallback(
    (sourceId: string, page?: number) => {
      if (subjectId == null) {
        onOpenCitation?.(sourceId, page);
        return;
      }
      const safePage = Math.max(1, page ?? 1);
      setPreviewData(null);
      setPreviewError(null);
      setPreviewMode('visual');

      if (/^lecture[-_]?\d+$/i.test(sourceId) || /^infocorso(?:-\d+)?$/i.test(sourceId)) {
        setCitationPreview({
          sourceId,
          page: safePage,
          nonPaginatedNote:
            'Questa citazione fa riferimento a una lezione o alle informazioni del corso. Potrai consultarla nei Materiali al termine della registrazione.',
        });
        setPreviewLoading(false);
        return;
      }

      setCitationPreview({ sourceId, page: safePage });
      void fetchCitationPage(subjectId, sourceId, safePage);
    },
    [subjectId, onOpenCitation, fetchCitationPage]
  );

  useEffect(() => {
    if (!citationPreview) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        closeCitationPreview();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [citationPreview, closeCitationPreview]);

  return (
    <div className="flex-1 flex flex-row w-full h-full overflow-hidden relative select-none" style={{ transform: 'translateZ(0)' }}>
      {/* ── Foglio Centrale (Isolato tramite React.memo) ── */}
      <LiveTranscriptContainer
        chunks={chunks}
        isPaused={isPaused}
        audioStatus={audioStatus}
      />

      {/* ── Inspector Laterale Destro (30% larghezza) ── */}
      <aside className="w-80 xl:w-[356px] flex-shrink-0 flex flex-col h-full border-l border-white/[0.08] bg-[#111520]/85 backdrop-blur-2xl z-10" style={{ transform: 'translateZ(0)' }}>
        {/* 2 Tab a pillola in alto con SegmentedControl */}
        <div className="p-3 border-b border-white/[0.06]">
          <SegmentedControl
            size="sm"
            layoutId="inspector-tab"
            fullWidth
            options={[
              {
                id: 'domande' as const,
                label: 'Domande',
                icon: HelpCircle,
                badge: detectedQuestions.length > 0 ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                ) : undefined,
              },
              {
                id: 'chat' as const,
                label: 'Assistente AI',
                icon: MessageSquare,
              },
            ]}
            value={inspectorTab}
            onChange={setInspectorTab}
          />
        </div>

        {/* ── Pannello 1: Rilevamento Domande del Prof ── */}
        {inspectorTab === 'domande' && (
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {detectedQuestions.length === 0 ? (
              <div className="py-16 text-center">
                <HelpCircle size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-2" />
                <p className="text-xs text-white/40 leading-normal">Nessuna domanda del professore rilevata finora.</p>
                <p className="text-[11px] text-white/25 mt-1 leading-normal">
                  Le domande del docente vengono intercettate automaticamente: puoi richiedere la risposta con l&apos;AI con un click.
                </p>
              </div>
            ) : (
              detectedQuestions.map((q) => {
                const isAnswering = Boolean(answeringQuestionIds?.has(q.id));
                const hasAnswer = Boolean(q.answer || q.suggestedAnswer);

                return (
                  <div
                    key={q.id}
                    className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.09] hover:border-white/[0.14] transition-colors space-y-2.5 animate-slide-up group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-[12.5px] font-medium text-white/90 leading-snug">
                        {q.question}
                      </p>
                      {onDeleteQuestion && (
                        <button
                          type="button"
                          onClick={() => onDeleteQuestion(q.id)}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded-lg text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer flex items-center justify-center flex-shrink-0"
                          title="Rimuovi domanda"
                        >
                          <Trash2 size={11} />
                        </button>
                      )}
                    </div>

                    {/* Stato 1: Risposta pronta o in streaming token-by-token */}
                    {hasAnswer && (
                      <>
                        <div className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.06] text-xs text-white/80 space-y-1">
                          <div className="flex items-center gap-1.5 text-[10px] text-white/50 font-mono leading-normal" style={{ transform: 'translateZ(0)' }}>
                            <Sparkles size={10} className={clsx(isAnswering && 'animate-pulse text-emerald-300')} />
                            <span className={clsx(isAnswering && 'chat-shimmer-text')}>
                              {isAnswering ? 'Risposta in corso...' : 'Risposta rapida'}
                            </span>
                          </div>
                          <div className="text-[11.5px] font-sans text-white/90 leading-relaxed select-text">
                            <MarkdownContent
                              content={formatQuestionAnswer(q.answer || q.suggestedAnswer || '')}
                              onOpenSettings={onOpenSettings}
                              onOpenCitation={onOpenCitation}
                            />
                          </div>
                        </div>
                        {!isAnswering && (
                          <div className="pt-0.5">
                            <button
                              type="button"
                              onClick={() => {
                                setInspectorTab('chat');
                                sendLiveChat(`Approfondisci la risposta alla domanda del docente: "${q.question}"`);
                              }}
                              className="w-full h-8 inline-flex items-center justify-center gap-1.5 px-3 rounded-full text-xs font-medium text-white/75 hover:text-white bg-white/[0.05] hover:bg-white/[0.10] border border-white/[0.10] transition-all cursor-pointer whitespace-nowrap"
                            >
                              <MessageSquare size={12} className="text-white/60 flex-shrink-0" />
                              <span>Approfondisci in Chat</span>
                            </button>
                          </div>
                        )}
                      </>
                    )}

                    {/* Stato 2: Generazione risposta AI in corso (prima del primo token) */}
                    {isAnswering && !hasAnswer && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/[0.05] border border-white/[0.10] text-white/75 text-xs animate-fade-in" style={{ transform: 'translateZ(0)' }}>
                        <Loader2 size={13} className="animate-spin text-white/60 flex-shrink-0" style={{ willChange: 'transform' }} />
                        <span className="font-medium text-[11px] leading-normal chat-shimmer-text">
                          Elaborazione immediata...
                        </span>
                      </div>
                    )}

                    {/* Stato 3: Controlli minimali su una sola riga */}
                    {!hasAnswer && !isAnswering && (
                      <div className="flex items-center gap-2 pt-0.5 animate-fade-in">
                        <button
                          type="button"
                          onClick={() => onAnswerQuestion?.(q.id)}
                          className="flex-1 h-8 inline-flex items-center justify-center gap-1.5 px-3 rounded-full text-xs font-medium text-white bg-white/[0.10] hover:bg-white/[0.16] border border-white/[0.14] shadow-liquid-sm transition-all cursor-pointer whitespace-nowrap"
                        >
                          <Sparkles size={12} className="text-white/80 flex-shrink-0" />
                          <span>Rispondi</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setInspectorTab('chat');
                            sendLiveChat(`Spiega e rispondi alla domanda del docente: "${q.question}"`);
                          }}
                          title="Approfondisci in Chat"
                          aria-label="Approfondisci in Chat"
                          className="w-8 h-8 rounded-full inline-flex items-center justify-center text-white/65 hover:text-white bg-white/[0.05] hover:bg-white/[0.12] border border-white/[0.10] transition-all cursor-pointer flex-shrink-0"
                        >
                          <MessageSquare size={13} />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* ── Pannello 2: Chat Live (Stile ChatGPT 2026 identico all'Assistente Corso) ── */}
        {inspectorTab === 'chat' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            <div
              ref={chatScrollRef}
              onScroll={handleChatScroll}
              className="flex-1 overflow-y-auto [scrollbar-gutter:stable] px-3.5 py-4"
            >
              {chatMessages.length === 0 ? (
                /* Empty Welcome State: Pure ChatGPT 2026 Style (respiro pieno nella sidebar) */
                <div className="h-full min-h-[260px] flex flex-col items-center justify-center text-center px-1.5 animate-fade-in select-none">
                  <h2 className="text-[20px] font-medium tracking-tight text-white/95">
                    Da dove cominciamo?
                  </h2>
                  <p className="text-[12.5px] text-white/50 mt-2 leading-relaxed max-w-[295px]">
                    Chiedi spiegazioni sulla lezione
                    {subjectName ? (
                      <>
                        {' '}di <span className="text-white/85 font-medium">{subjectName}</span>
                      </>
                    ) : null}
                    {' '}o sui materiali del corso.
                  </p>
                  <div className="flex flex-col gap-2 w-full mt-6">
                    <button
                      type="button"
                      onClick={() =>
                        sendLiveChat("Non ho capito l'ultimo concetto, puoi rispiegarmelo in modo semplice con un esempio?")
                      }
                      className="w-full px-4 py-2.5 rounded-2xl text-[12px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.09] hover:border-white/[0.16] text-white/80 hover:text-white transition-all cursor-pointer leading-snug"
                    >
                      Rispiega l&apos;ultimo concetto con un esempio
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        sendLiveChat('Cosa ha detto il docente negli ultimi 2 minuti?')
                      }
                      className="w-full px-4 py-2.5 rounded-2xl text-[12px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.09] hover:border-white/[0.16] text-white/80 hover:text-white transition-all cursor-pointer leading-snug"
                    >
                      Riassumi gli ultimi 2 minuti di lezione
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3.5 pb-4">
                  {chatMessages.map((msg, idx) => {
                    const isStreamingThis =
                      isSendingChat &&
                      idx === chatMessages.length - 1 &&
                      msg.role === 'assistant';
                    return (
                      <ChatMessageItem
                        key={msg.id}
                        message={toChatMessageItemModel(msg, idx, isStreamingThis)}
                        isStreamingThis={isStreamingThis}
                        isAnyStreaming={isSendingChat}
                        compact
                        onOpenCitation={handleCitationOpen}
                        onOpenSettings={onOpenSettings}
                      />
                    );
                  })}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* ChatGPT 2026 Floating Pill Composer */}
            <div className="px-3 pb-3 pt-1.5 flex-shrink-0">
              <div
                className={clsx(
                  'rounded-[24px] bg-[#121622]/90 border border-white/[0.12] focus-within:border-white/[0.24] shadow-xl transition-all pl-3.5 pr-1.5 py-1.5 flex gap-2',
                  isComposerMultiline ? 'items-end' : 'items-center'
                )}
              >
                <textarea
                  ref={composerRef}
                  rows={1}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      if (!isSendingChat && chatInput.trim()) {
                        sendLiveChat();
                      }
                    }
                  }}
                  placeholder="Chiedi qualsiasi cosa..."
                  className={clsx(
                    'flex-1 min-w-0 resize-none bg-transparent pl-0.5 py-[1px] text-[12.5px] leading-[22px] text-zinc-100 placeholder-zinc-500 focus:outline-none max-h-[120px] select-text block',
                    isComposerOverflow ? 'overflow-y-auto' : 'overflow-hidden'
                  )}
                />

                <div className={clsx('flex items-center flex-shrink-0', isComposerMultiline && 'mb-0.5')}>
                  {isSendingChat ? (
                    <button
                      type="button"
                      onClick={() => {
                        if (handleCancelChat) {
                          handleCancelChat();
                        } else {
                          api.ai.stop(true).catch(() => {});
                        }
                      }}
                      title="Interrompi generazione"
                      className="w-7 h-7 rounded-full flex items-center justify-center bg-white text-[#0d1017] hover:bg-white/90 transition-all cursor-pointer flex-shrink-0 shadow-md"
                    >
                      <Square size={9} fill="currentColor" strokeWidth={0} />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => sendLiveChat()}
                      disabled={!chatInput.trim()}
                      title="Invia messaggio"
                      className={clsx(
                        'w-7 h-7 rounded-full flex items-center justify-center transition-all flex-shrink-0',
                        chatInput.trim()
                          ? 'bg-white text-[#0d1017] hover:bg-white/90 cursor-pointer shadow-md'
                          : 'bg-white/[0.08] text-white/30 cursor-not-allowed'
                      )}
                    >
                      <ArrowUp size={15} strokeWidth={2.5} />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* ── Inline Slide / PDF Preview Overlay (non interrompe la registrazione) ── */}
      {citationPreview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/70 backdrop-blur-md animate-fade-in"
          onClick={closeCitationPreview}
        >
          <div
            className="w-full max-w-3xl max-h-[88vh] flex flex-col rounded-3xl liquid-modal border border-white/[0.14] shadow-liquid-lg p-5 sm:p-6 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-white/[0.08]">
              <div className="flex items-center gap-2.5 min-w-0 flex-1">
                <div className="w-8 h-8 rounded-xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center flex-shrink-0">
                  <FileText size={15} className="text-emerald-300" />
                </div>
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold text-white truncate">
                    {previewData?.title || citationPreview.sourceId}
                  </h3>
                  <div className="flex items-center gap-2 text-[10px] font-mono text-white/50">
                    <span className="truncate max-w-[240px]">
                      {previewData?.filename || citationPreview.sourceId}
                    </span>
                    {!citationPreview.nonPaginatedNote && (
                      <span>
                        · Pagina {previewData?.page ?? citationPreview.page}
                        {previewData?.page_count ? ` / ${previewData.page_count}` : ''}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 flex-shrink-0">
                {previewData?.kind === 'image' && previewData.image_data_url && (
                  <button
                    type="button"
                    onClick={() => setPreviewMode((m) => (m === 'visual' ? 'text' : 'visual'))}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-[11px] font-medium text-white transition-all cursor-pointer"
                  >
                    {previewMode === 'visual' ? (
                      <>
                        <Code2 size={12} />
                        <span>Testo</span>
                      </>
                    ) : (
                      <>
                        <Eye size={12} className="text-blue-300" />
                        <span>Slide</span>
                      </>
                    )}
                  </button>
                )}

                {subjectId != null && !citationPreview.nonPaginatedNote && (
                  <a
                    href={api.subjects.getMaterialFileUrl(subjectId, citationPreview.sourceId)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.10] text-[11px] font-medium text-white/80 hover:text-white transition-all"
                  >
                    <ExternalLink size={11} />
                    <span className="hidden sm:inline">Apri file</span>
                  </a>
                )}

                <button
                  type="button"
                  onClick={closeCitationPreview}
                  title="Chiudi (Esc)"
                  className="w-7 h-7 rounded-full flex items-center justify-center text-white/50 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
                >
                  <X size={14} />
                </button>
              </div>
            </div>

            {/* Pagination controls */}
            {!citationPreview.nonPaginatedNote && subjectId != null && (
              <div className="py-2 px-3 mt-3 rounded-2xl bg-white/[0.03] border border-white/[0.07] flex items-center justify-between">
                <button
                  type="button"
                  disabled={citationPreview.page <= 1 || previewLoading}
                  onClick={() => {
                    const nextPage = Math.max(1, citationPreview.page - 1);
                    setCitationPreview({ ...citationPreview, page: nextPage });
                    void fetchCitationPage(subjectId, citationPreview.sourceId, nextPage);
                  }}
                  className="flex items-center gap-1 px-3 py-1 rounded-full bg-white/[0.06] hover:bg-white/[0.12] disabled:opacity-35 disabled:cursor-not-allowed text-[11px] font-medium text-white transition-all cursor-pointer"
                >
                  <ChevronLeft size={13} />
                  <span>Precedente</span>
                </button>

                <span className="text-[11px] font-mono text-white/60">
                  Pagina {previewData?.page ?? citationPreview.page}
                  {previewData?.page_count ? ` / ${previewData.page_count}` : ''}
                </span>

                <button
                  type="button"
                  disabled={
                    previewLoading ||
                    Boolean(previewData?.page_count && citationPreview.page >= previewData.page_count)
                  }
                  onClick={() => {
                    const nextPage = citationPreview.page + 1;
                    setCitationPreview({ ...citationPreview, page: nextPage });
                    void fetchCitationPage(subjectId, citationPreview.sourceId, nextPage);
                  }}
                  className="flex items-center gap-1 px-3 py-1 rounded-full bg-white/[0.06] hover:bg-white/[0.12] disabled:opacity-35 disabled:cursor-not-allowed text-[11px] font-medium text-white transition-all cursor-pointer"
                >
                  <span>Successiva</span>
                  <ChevronRight size={13} />
                </button>
              </div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-y-auto py-4 min-h-[240px]">
              {citationPreview.nonPaginatedNote ? (
                <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/[0.08] text-xs text-white/75 leading-relaxed">
                  {citationPreview.nonPaginatedNote}
                </div>
              ) : previewLoading ? (
                <SlidePageSkeleton />
              ) : previewError ? (
                <div className="p-5 rounded-2xl bg-red-500/[0.08] border border-red-500/25 text-center space-y-2.5">
                  <AlertCircle size={22} className="text-red-400 mx-auto" />
                  <p className="text-xs text-red-200">{previewError}</p>
                  {subjectId != null && (
                    <button
                      type="button"
                      onClick={() => void fetchCitationPage(subjectId, citationPreview.sourceId, citationPreview.page)}
                      className="px-3.5 py-1.5 rounded-full bg-white/[0.10] hover:bg-white/[0.16] text-[11px] text-white cursor-pointer"
                    >
                      Riprova
                    </button>
                  )}
                </div>
              ) : previewData?.kind === 'image' && previewData.image_data_url && previewMode === 'visual' ? (
                <div className="flex items-center justify-center rounded-2xl bg-black/40 border border-white/[0.08] p-3">
                  <img
                    src={previewData.image_data_url}
                    alt={`${previewData.title || citationPreview.sourceId} - Pagina ${previewData.page}`}
                    className="max-w-full max-h-[58vh] w-auto h-auto object-contain rounded-xl shadow-lg select-text"
                  />
                </div>
              ) : previewData?.text ? (
                <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-xs leading-relaxed text-white/85 whitespace-pre-wrap select-text">
                  {previewData.text}
                </div>
              ) : (
                <div className="py-12 text-center text-xs text-white/40 italic">
                  Nessun contenuto estratto per questa pagina.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const LiveRecordingWorkspace = React.memo(LiveRecordingWorkspaceComponent);
export const LiveLecture = LiveRecordingWorkspace;
export default LiveRecordingWorkspace;
