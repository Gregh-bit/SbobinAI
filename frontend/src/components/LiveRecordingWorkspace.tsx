import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import {
  HelpCircle,
  MessageSquare,
  Sparkles,
  Send,
  Trash2,
  Square,
  Copy,
  Check,
  Brain,
  ChevronDown,
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
import type { ChatMessage, DetectedQuestion, MaterialPageOut, TranscriptChunk } from '../types';
import SegmentedControl from './SegmentedControl';
import MarkdownContent from './MarkdownContent';
import { parseThinkingContent } from './chat/ChatMessageItem';
import { api } from '../lib/api';

interface Props {
  subjectId?: number;
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
  handleChatSend: (text?: string) => void;
  handleCancelChat?: () => void;
  inspectorTab: 'domande' | 'chat';
  setInspectorTab: (tab: 'domande' | 'chat') => void;
  onOpenCitation?: (sourceId: string, page?: number) => void;
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
  const transcriptEndRef = useRef<HTMLDivElement>(null);

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

  const [displayedLength, setDisplayedLength] = useState<number>(() => {
    // Se chunks sono già presenti al montaggio (es. recupero sessione attiva)
    const initialLen = chunks && chunks.length > 0
      ? chunks.map(c => (c.text || '').trim()).filter(Boolean).join(' ').length
      : 0;
    return initialLen > 300 ? initialLen : 0;
  });

  const displayedLengthRef = useRef<number>(displayedLength);
  const animFrameRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    displayedLengthRef.current = displayedLength;
  }, [displayedLength]);

  // Ciclo macchina da scrivere a scorrimento naturale
  useEffect(() => {
    const targetLen = fullTargetText.length;
    // Se all'inizio o se il ritardo accumulato è eccessivo (>350 caratteri su restore)
    if (displayedLengthRef.current === 0 && targetLen > 350) {
      setDisplayedLength(targetLen);
      displayedLengthRef.current = targetLen;
      return;
    }

    const stepAnimation = (timestamp: number) => {
      if (displayedLengthRef.current < targetLen) {
        if (timestamp - lastTickRef.current >= 18) {
          lastTickRef.current = timestamp;
          const remaining = targetLen - displayedLengthRef.current;
          // Passo dinamico per un battito naturale da macchina da scrivere:
          // 1-2 caratteri per parlato normale, fino a 4-8 caratteri se arriva un chunk corposo
          const step = Math.max(1, Math.min(remaining, Math.ceil(remaining / 7)));
          const nextLen = displayedLengthRef.current + step;
          displayedLengthRef.current = nextLen;
          setDisplayedLength(nextLen);
        }
        animFrameRef.current = requestAnimationFrame(stepAnimation);
      } else {
        animFrameRef.current = null;
      }
    };

    if (displayedLengthRef.current < targetLen) {
      if (!animFrameRef.current) {
        animFrameRef.current = requestAnimationFrame(stepAnimation);
      }
    } else if (displayedLengthRef.current > targetLen) {
      setDisplayedLength(targetLen);
      displayedLengthRef.current = targetLen;
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

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [displayedText, isPaused]);

  return (
    <section className="flex-1 flex flex-col h-full overflow-hidden relative pb-8 content-canvas">
      <div className="flex-1 overflow-y-auto px-12 py-10">
        <div className="max-w-2xl mx-auto space-y-4">
          {isPaused && (
            <div className="flex items-center justify-center mb-4 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs shadow-liquid-sm">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="font-medium">Registrazione in Pausa · Hardware a riposo (0% CPU/GPU)</span>
              </div>
            </div>
          )}

          {/* Testo lineare unico e continuo (display inline), zero scritte di caricamento o interruzioni di riga forzate */}
          <div className="relative text-[16px] leading-[1.85] font-sans text-white/95 font-normal tracking-wide select-text break-words">
            <span>{displayedText}</span>

            {/* Cursore verticale animato stile playhead/karaoke (effetto macchina da scrivere) */}
            <span
              className={clsx(
                "inline-block rounded-full pointer-events-none transition-opacity duration-150 align-baseline",
                (isTyping || audioStatus === 'transcribing') ? "opacity-100" : "animate-caret-blink"
              )}
              style={{
                width: '2.5px',
                height: '1.25em',
                background: 'linear-gradient(180deg, #ffffff 0%, #38bdf8 100%)',
                boxShadow: '0 0 8px #38bdf8, 0 0 14px rgba(56, 189, 248, 0.65)',
                marginLeft: displayedText ? '3px' : '0px',
                marginRight: '2px',
                verticalAlign: '-0.15em',
              }}
              aria-hidden="true"
            />
          </div>

          <div ref={transcriptEndRef} />
        </div>
      </div>
    </section>
  );
});

interface LiveAssistantMessageProps {
  message: ChatMessage;
  isStreaming: boolean;
  onOpenCitation?: (sourceId: string, page?: number) => void;
}

function LiveAssistantMessage({
  message,
  isStreaming,
  onOpenCitation,
}: LiveAssistantMessageProps) {
  const [copied, setCopied] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (copyTimeoutRef.current !== null) {
        clearTimeout(copyTimeoutRef.current);
        copyTimeoutRef.current = null;
      }
    };
  }, []);

  if (message.role === 'user') {
    return (
      <div className="flex flex-col items-end max-w-[88%] ml-auto text-xs leading-relaxed animate-fade-in">
        <div className="px-3.5 py-2 rounded-2xl rounded-br-sm bg-zinc-800/85 text-zinc-100 border border-white/10 break-words select-text">
          <p className="text-xs whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  const parsed = parseThinkingContent(message.content);
  const isThinkingStreaming = isStreaming && parsed.isThinkingStreaming;
  const waitingFirstToken = isStreaming && !parsed.visible && !isThinkingStreaming;

  const handleCopy = () => {
    const textToCopy = parsed.visible || message.content;
    if (!textToCopy) return;
    navigator.clipboard
      ?.writeText(textToCopy)
      .then(() => {
        if (!isMountedRef.current) return;
        setCopied(true);
        if (copyTimeoutRef.current !== null) {
          clearTimeout(copyTimeoutRef.current);
        }
        copyTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            setCopied(false);
          }
          copyTimeoutRef.current = null;
        }, 1500);
      })
      .catch(() => {});
  };

  return (
    <div className="group flex flex-col items-start w-full text-xs leading-relaxed animate-fade-in">
      <div className="flex items-center justify-between w-full mb-1">
        <span className="inline-flex items-center gap-1.5 text-[10px] font-medium text-zinc-400">
          <Sparkles size={10} className="text-emerald-400" />
          <span>SbobinAI</span>
        </span>
        {!isStreaming && (parsed.visible || message.content) && (
          <button
            type="button"
            onClick={handleCopy}
            title="Copia risposta"
            className="opacity-0 group-hover:opacity-100 inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.08] transition-all cursor-pointer"
          >
            {copied ? (
              <>
                <Check size={10} className="text-emerald-400" />
                <span className="text-emerald-300">Copiato</span>
              </>
            ) : (
              <>
                <Copy size={10} />
                <span>Copia</span>
              </>
            )}
          </button>
        )}
      </div>

      {(parsed.thinking || isThinkingStreaming) && (
        <div className="w-full mb-1.5">
          <button
            type="button"
            onClick={() => setThinkingExpanded((prev) => !prev)}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-[10px] transition-colors cursor-pointer"
          >
            <Brain size={10} className="text-indigo-300/80" />
            <span className={clsx(isThinkingStreaming ? 'chat-shimmer-text font-medium' : 'text-zinc-300')}>
              {isThinkingStreaming ? 'Ragionamento...' : 'Ragionamento'}
            </span>
            {thinkingExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          </button>
          {thinkingExpanded && parsed.thinking && (
            <div className="mt-1.5 pl-2.5 border-l-2 border-white/10 text-[11px] text-zinc-400 whitespace-pre-wrap max-h-40 overflow-y-auto select-text">
              {parsed.thinking}
            </div>
          )}
        </div>
      )}

      {waitingFirstToken && (
        <div className="flex items-center gap-2 py-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="chat-shimmer-text text-[11px] font-medium">
            SbobinAI sta formulando la spiegazione...
          </span>
        </div>
      )}

      {parsed.visible && (
        <div className="w-full text-zinc-100 break-words select-text">
          <MarkdownContent
            content={parsed.visible}
            onOpenCitation={onOpenCitation}
            className="chat-prose"
          />
        </div>
      )}
    </div>
  );
}

// ── LiveRecordingWorkspace / LiveLecture Container ───────────────────────────
function LiveRecordingWorkspaceComponent({
  subjectId,
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
}: Props) {
  const chatEndRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

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
    <div className="flex-1 flex flex-row w-full h-full overflow-hidden relative select-none">
      {/* ── Foglio Centrale (Isolato tramite React.memo) ── */}
      <LiveTranscriptContainer
        chunks={chunks}
        isPaused={isPaused}
        audioStatus={audioStatus}
      />

      {/* ── Inspector Laterale Destro (30% larghezza) ── */}
      <aside className="w-80 flex-shrink-0 flex flex-col h-full border-l border-white/[0.08] bg-[#111520]/85 backdrop-blur-2xl z-10">
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
            <div className="flex items-center justify-between pb-1">
              <span className="text-[10px] uppercase font-mono tabular-nums tracking-wider text-white/40 leading-normal">
                Domande Rilevate ({detectedQuestions.length})
              </span>
            </div>

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
                    className="p-3.5 rounded-2xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-2 animate-slide-up group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-xs font-semibold text-white/90 leading-snug">
                        &ldquo;{q.question}&rdquo;
                      </p>
                      {onDeleteQuestion && (
                        <button
                          type="button"
                          onClick={() => onDeleteQuestion(q.id)}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded-lg text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer flex items-center justify-center"
                          title="Rimuovi domanda"
                        >
                          <Trash2 size={11} />
                        </button>
                      )}
                    </div>

                    {/* Stato 1: Risposta pronta (generata su click) */}
                    {hasAnswer && (
                      <>
                        <div className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.06] text-xs text-white/80 space-y-1">
                          <div className="flex items-center gap-1.5 text-[10px] text-amber-300/80 font-mono leading-normal">
                            <Sparkles size={10} />
                            <span>Risposta flash suggerita:</span>
                          </div>
                          <div className="text-[11px] font-sans text-white/90 leading-relaxed">
                            <MarkdownContent content={q.answer || q.suggestedAnswer || ''} />
                          </div>
                        </div>
                        <div className="pt-1">
                          <button
                            type="button"
                            onClick={() => {
                              setInspectorTab('chat');
                              handleChatSend(`Approfondisci la risposta alla domanda del docente: "${q.question}"`);
                            }}
                            className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-white/70 hover:text-white bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-all cursor-pointer leading-normal"
                          >
                            <MessageSquare size={12} className="text-indigo-400" />
                            <span className="leading-normal">Approfondisci in Chat</span>
                          </button>
                        </div>
                      </>
                    )}

                    {/* Stato 2: Generazione risposta AI in corso */}
                    {isAnswering && !hasAnswer && (
                      <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs animate-fade-in">
                        <Loader2 size={13} className="animate-spin text-amber-400 flex-shrink-0" />
                        <span className="font-medium text-[11px] leading-normal">Generazione risposta AI in corso...</span>
                      </div>
                    )}

                    {/* Stato 3: Senza risposta e non in generazione: 2 pulsanti chiari */}
                    {!hasAnswer && !isAnswering && (
                      <div className="flex items-center gap-2 pt-1 animate-fade-in">
                        <button
                          type="button"
                          onClick={() => onAnswerQuestion?.(q.id)}
                          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-amber-200 bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/30 shadow-liquid-sm transition-all cursor-pointer group/btn leading-normal"
                        >
                          <Sparkles size={12} className="text-amber-400 group-hover/btn:scale-110 transition-transform" />
                          <span className="leading-normal">Rispondi con l&apos;AI</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            setInspectorTab('chat');
                            handleChatSend(`Spiega e rispondi alla domanda del docente: "${q.question}"`);
                          }}
                          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-white/80 hover:text-white bg-white/[0.05] hover:bg-white/[0.10] border border-white/[0.10] shadow-liquid-sm transition-all cursor-pointer leading-normal"
                        >
                          <MessageSquare size={12} className="text-zinc-400" />
                          <span className="leading-normal">Chiedi in Chat</span>
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* ── Pannello 2: Chat Live "Non ho capito!" ── */}
        {inspectorTab === 'chat' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {chatMessages.length === 0 ? (
                <div className="py-16 text-center space-y-3">
                  <MessageSquare size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-1" />
                  <p className="text-xs text-white/50">Non hai capito un concetto spiegato dal docente?</p>
                  <div className="flex flex-col gap-1.5 max-w-[200px] mx-auto pt-2">
                    <button
                      type="button"
                      onClick={() => handleChatSend("Non ho capito l'ultimo concetto, puoi rispiegarmelo con un esempio pratico?")}
                      className="px-2.5 py-1.5 rounded-xl text-[11px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/70 hover:text-white transition-all cursor-pointer"
                    >
                      "Non ho capito l'ultimo concetto..."
                    </button>
                    <button
                      type="button"
                      onClick={() => handleChatSend("Qual è la differenza principale tra i due termini appena spiegati?")}
                      className="px-2.5 py-1.5 rounded-xl text-[11px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/70 hover:text-white transition-all cursor-pointer"
                    >
                      "Qual è la differenza tra i due termini?"
                    </button>
                  </div>
                </div>
              ) : (
                chatMessages.map((msg, idx) => (
                  <LiveAssistantMessage
                    key={msg.id}
                    message={msg}
                    isStreaming={
                      isSendingChat &&
                      idx === chatMessages.length - 1 &&
                      msg.role === 'assistant'
                    }
                    onOpenCitation={handleCitationOpen}
                  />
                ))
              )}
              {isSendingChat &&
                (chatMessages.length === 0 ||
                  chatMessages[chatMessages.length - 1]?.role === 'user') && (
                  <div className="flex items-center gap-2 py-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="chat-shimmer-text text-[11px] font-medium">
                      SbobinAI sta formulando la spiegazione...
                    </span>
                  </div>
                )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input Bar */}
            <div className="p-3 border-t border-white/[0.08] bg-black/20">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                      e.preventDefault();
                      handleChatSend();
                    }
                  }}
                  placeholder="Chiedi al volo..."
                  className="flex-1 px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                />
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
                    title="Interrompi risposta IA e libera RAM"
                    className="px-2.5 h-7 rounded-xl flex items-center gap-1.5 bg-red-500/25 hover:bg-red-500/35 text-red-300 border border-red-500/40 text-[11px] font-semibold transition-all cursor-pointer animate-pulse"
                  >
                    <Square size={10} fill="currentColor" strokeWidth={0} />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleChatSend()}
                    disabled={!chatInput.trim()}
                    className="w-7 h-7 rounded-xl flex items-center justify-center bg-white/[0.14] hover:bg-white/[0.22] text-white transition-all disabled:opacity-30 border border-white/[0.10] cursor-pointer"
                  >
                    <Send size={12} strokeWidth={2} />
                  </button>
                )}
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
            className="w-full max-w-3xl max-h-[88vh] flex flex-col rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg p-5 sm:p-6 animate-slide-up"
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
                <div className="h-64 flex flex-col items-center justify-center gap-2 text-xs text-white/50 font-mono">
                  <Loader2 size={22} className="animate-spin text-white/50" />
                  <span>Caricamento pagina {citationPreview.page}...</span>
                </div>
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
