import { useState, useRef, useEffect } from 'react';
import {
  Copy,
  Check,
  RotateCcw,
  BookmarkPlus,
  Trash2,
  ChevronDown,
  BookOpen,
  Brain,
  Pencil,
  Sparkles,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { PersistentChatMessage } from '../../types';
import MarkdownContent from '../MarkdownContent';
import ChatActionCard from './ChatActionCard';
import { getStudyModeMeta } from './ChatComposer';
import Text3DFlip from './Text3DFlip';

// Passaggi dinamici stile ChatGPT 2026 per la consultazione RAG e il ragionamento
const RAG_SEARCH_STEPS = [
  'Analisi delle fonti...',
  'Consultazione degli appunti...',
  'Ricerca nei materiali del corso...',
  'Sintesi delle spiegazioni...',
];

const THINKING_REASONING_STEPS = [
  'Ragionamento sui materiali...',
  'Elaborazione dei concetti...',
  'Strutturazione della risposta...',
];

export interface ParsedThinkingResult {
  thinking: string;
  visible: string;
  isThinkingStreaming: boolean;
}

/**
 * Splits `<think>...</think>` reasoning blocks (including partial streaming tags)
 * from the visible assistant prose.
 */
export function parseThinkingContent(rawText: string): ParsedThinkingResult {
  if (!rawText) {
    return { thinking: '', visible: '', isThinkingStreaming: false };
  }

  const openTag = '<think>';
  const closeTag = '</think>';
  const openIdx = rawText.indexOf(openTag);

  if (openIdx === -1) {
    const trimmed = rawText.trimStart();
    if (trimmed.length > 0 && trimmed.length < openTag.length && openTag.startsWith(trimmed)) {
      return { thinking: '', visible: '', isThinkingStreaming: true };
    }
    return { thinking: '', visible: rawText, isThinkingStreaming: false };
  }

  const afterOpen = openIdx + openTag.length;
  const closeIdx = rawText.indexOf(closeTag, afterOpen);

  if (closeIdx === -1) {
    let partialThinking = rawText.slice(afterOpen);
    // Prevent partial closing tag flicker like `</thi` at the end of stream
    const lastLt = partialThinking.lastIndexOf('<');
    if (lastLt !== -1 && closeTag.startsWith(partialThinking.slice(lastLt))) {
      partialThinking = partialThinking.slice(0, lastLt);
    }
    return {
      thinking: partialThinking.trimStart(),
      visible: rawText.slice(0, openIdx).trim(),
      isThinkingStreaming: true,
    };
  }

  const thinking = rawText.slice(afterOpen, closeIdx).trim();
  const visible = (rawText.slice(0, openIdx) + rawText.slice(closeIdx + closeTag.length)).trimStart();
  return { thinking, visible, isThinkingStreaming: false };
}

export interface ChatMessageItemProps {
  message: PersistentChatMessage;
  isStreamingThis?: boolean;
  isAnyStreaming?: boolean;
  isLastAssistant?: boolean;
  isSavedToQuaderno?: boolean;
  isSavingNote?: boolean;
  isDeleting?: boolean;
  compact?: boolean;
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenAudioSnippet?: (lectureId: number, seconds: number, autoPlay?: boolean) => void;
  onOpenSettings?: () => void;
  onOpenQuiz?: (quizId: number) => void;
  onOpenOralQuestion?: (question: string, topic?: string) => void;
  onOpenQuadernoLecture?: (lectureId: number) => void;
  onRegenerate?: (messageId: number) => void;
  onSaveNote?: (messageId: number) => void;
  onDelete?: (messageId: number) => void;
  hasSubsequentQuizActions?: boolean;
  onEdit?: (messageId: number, newText: string, rollbackQuiz?: boolean) => void;
  onSelectFollowup?: (text: string) => void;
}

/**
 * Renders a single user or assistant message using the asymmetric ChatGPT 2026 layout:
 * clean dark pill for user turns (with inline edit & deletion confirmation) and
 * borderless Markdown+KaTeX prose for assistant turns with an expandable "Fonti consultate" menu.
 */
export default function ChatMessageItem({
  message,
  isStreamingThis = false,
  isAnyStreaming = false,
  isLastAssistant = false,
  isSavedToQuaderno = false,
  isSavingNote = false,
  isDeleting = false,
  compact = false,
  hasSubsequentQuizActions = false,
  onOpenCitation,
  onOpenAudioSnippet,
  onOpenSettings,
  onOpenQuiz,
  onOpenOralQuestion,
  onOpenQuadernoLecture,
  onRegenerate,
  onSaveNote,
  onDelete,
  onEdit,
}: ChatMessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [traceExpanded, setTraceExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const [revertQuiz, setRevertQuiz] = useState(true);
  const [showEditConfirmModal, setShowEditConfirmModal] = useState(false);
  const [showDeleteConfirmModal, setShowDeleteConfirmModal] = useState(false);

  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);
  const thinkingContainerRef = useRef<HTMLDivElement | null>(null);

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

  useEffect(() => {
    if (!showEditConfirmModal && !showDeleteConfirmModal) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowEditConfirmModal(false);
        setShowDeleteConfirmModal(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showEditConfirmModal, showDeleteConfirmModal]);

  const handleCopy = (text: string) => {
    if (!text) return;
    navigator.clipboard
      ?.writeText(text)
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
        }, 1600);
      })
      .catch(() => {});
  };

  if (message.role === 'user') {
    const canDelete = message.id > 0 && !isAnyStreaming && !isDeleting && Boolean(onDelete);
    const canEdit = message.id > 0 && !isAnyStreaming && Boolean(onEdit);
    const modeMeta = getStudyModeMeta(message.mode);

    const actionButtons = (
      <>
        <button
          type="button"
          onClick={() => {
            const fullText = modeMeta
              ? (message.content ? `[${modeMeta.label}] ${message.content}` : `[${modeMeta.label}]`)
              : message.content;
            handleCopy(fullText);
          }}
          title="Copia messaggio"
          className="p-1 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.06] transition-colors cursor-pointer"
        >
          {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
        </button>
        {canEdit && (
          <button
            type="button"
            onClick={() => {
              setEditText(message.content);
              setIsEditing(true);
            }}
            title="Modifica messaggio"
            className="p-1 rounded-lg text-zinc-500 hover:text-indigo-300 hover:bg-indigo-500/10 transition-colors cursor-pointer"
          >
            <Pencil size={12} />
          </button>
        )}
        {canDelete && (
          <button
            type="button"
            onClick={() => setShowDeleteConfirmModal(true)}
            title="Elimina messaggio"
            className="p-1 rounded-lg text-zinc-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer"
          >
            <Trash2 size={12} />
          </button>
        )}
      </>
    );

    return (
      <div
        className={clsx(
          'group flex flex-col items-end relative w-full',
          compact ? 'gap-0' : 'gap-1',
          message.id < 0 && 'animate-fade-in'
        )}
      >
        {isEditing ? (
          <div className="w-full max-w-[85%] bg-[#121622]/95 border border-white/[0.14] rounded-2xl p-3 shadow-lg flex flex-col gap-2">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={Math.min(8, Math.max(2, editText.split('\n').length))}
              className="w-full bg-black/30 border border-white/[0.10] rounded-xl p-2.5 text-white/95 text-[13px] leading-relaxed resize-y focus:outline-none focus:border-white/[0.24]"
              placeholder="Modifica il tuo messaggio..."
              autoFocus
            />
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  setIsEditing(false);
                  setEditText(message.content);
                }}
                className="px-3.5 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-xs font-medium text-zinc-300 hover:text-white transition-colors cursor-pointer"
              >
                Annulla
              </button>
              <button
                type="button"
                disabled={!editText.trim() || editText.trim() === message.content.trim()}
                onClick={() => setShowEditConfirmModal(true)}
                className="px-3.5 py-1.5 rounded-full bg-white text-[#0d1017] hover:bg-white/90 text-xs font-semibold transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Conferma e Invia
              </button>
            </div>
          </div>
        ) : compact ? (
          <div className="flex items-center justify-end gap-1 w-full">
            <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 pointer-events-none group-hover:pointer-events-auto focus-within:pointer-events-auto transition-opacity flex-shrink-0">
              {actionButtons}
            </div>
            <div className="bg-white/[0.08] border border-white/[0.11] rounded-3xl text-white/95 leading-relaxed shadow-sm select-text px-3.5 py-2 max-w-[88%] text-[12.5px]">
              <div className="flex flex-wrap items-center gap-2">
                {modeMeta && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-sky-500/15 border border-sky-400/30 text-sky-300 text-xs font-medium select-none shadow-sm flex-shrink-0">
                    <modeMeta.icon size={12} className={clsx(modeMeta.colorClass, 'flex-shrink-0')} />
                    <span>{modeMeta.label}</span>
                  </span>
                )}
                {message.content && (
                  <span className="whitespace-pre-wrap break-words">{message.content}</span>
                )}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-white/[0.08] border border-white/[0.11] rounded-3xl text-white/95 leading-relaxed shadow-sm select-text px-4 py-2.5 max-w-[80%] text-[13px]">
              <div className="flex flex-wrap items-center gap-2">
                {modeMeta && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-sky-500/15 border border-sky-400/30 text-sky-300 text-xs font-medium select-none shadow-sm flex-shrink-0">
                    <modeMeta.icon size={12} className={clsx(modeMeta.colorClass, 'flex-shrink-0')} />
                    <span>{modeMeta.label}</span>
                  </span>
                )}
                {message.content && (
                  <span className="whitespace-pre-wrap break-words">{message.content}</span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-1 pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
              {actionButtons}
            </div>
          </>
        )}

        {/* Minimal edit confirmation dialog */}
        {showEditConfirmModal && (
          <div
            onClick={(e) => {
              if (e.target === e.currentTarget) setShowEditConfirmModal(false);
            }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in"
          >
            <div className="w-full max-w-sm rounded-3xl liquid-modal p-5 flex flex-col gap-3.5 text-left">
              <div className="text-sm font-semibold text-zinc-100">Modifica messaggio</div>
              <p className="text-xs text-zinc-300 leading-relaxed">
                Le risposte successive verranno cancellate per generare una nuova risposta aggiornata.
              </p>

              {hasSubsequentQuizActions && (
                <label className="flex items-start gap-2.5 p-3 rounded-2xl bg-white/[0.04] border border-white/[0.08] cursor-pointer hover:bg-white/[0.07] transition-colors select-none">
                  <input
                    type="checkbox"
                    checked={revertQuiz}
                    onChange={(e) => setRevertQuiz(e.target.checked)}
                    className="mt-0.5 rounded border-zinc-600 text-indigo-500 focus:ring-indigo-500/30"
                  />
                  <div className="flex flex-col text-left">
                    <span className="text-xs font-medium text-zinc-200">Ripristina quiz allo stato precedente</span>
                    <span className="text-[11px] text-zinc-400 leading-normal">
                      Annulla le modifiche alle domande effettuate dai messaggi successivi
                    </span>
                  </div>
                </label>
              )}

              <div className="flex items-center justify-end gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setShowEditConfirmModal(false)}
                  className="px-4 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-zinc-300 hover:text-white text-xs font-medium transition-colors cursor-pointer"
                >
                  Annulla
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowEditConfirmModal(false);
                    setIsEditing(false);
                    onEdit?.(message.id, editText.trim(), revertQuiz);
                  }}
                  className="px-4 py-1.5 rounded-full bg-white text-[#0d1017] hover:bg-white/90 text-xs font-semibold transition-colors cursor-pointer"
                >
                  Conferma
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Minimal delete confirmation dialog */}
        {showDeleteConfirmModal && (
          <div
            onClick={(e) => {
              if (e.target === e.currentTarget) setShowDeleteConfirmModal(false);
            }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fade-in"
          >
            <div className="w-full max-w-sm rounded-3xl liquid-modal p-5 flex flex-col gap-3 text-left">
              <div className="text-sm font-semibold text-zinc-100">Elimina messaggio</div>
              <p className="text-xs text-zinc-300 leading-relaxed">
                Eliminare questo messaggio e tutta la conversazione successiva?
              </p>
              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowDeleteConfirmModal(false)}
                  className="px-4 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-zinc-300 hover:text-white text-xs font-medium transition-colors cursor-pointer"
                >
                  Annulla
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowDeleteConfirmModal(false);
                    onDelete?.(message.id);
                  }}
                  className="px-4 py-1.5 rounded-full bg-red-500/20 hover:bg-red-500/30 text-red-200 border border-red-500/35 text-xs font-medium transition-colors cursor-pointer"
                >
                  Elimina
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Assistant turn: combine explicit `message.thinking` with any inline `<think>` block
  const parsed = parseThinkingContent(message.content);
  const combinedThinking = (message.thinking || parsed.thinking).trim();
  const visibleContent = parsed.visible;
  const isThinkingStreaming = isStreamingThis && parsed.isThinkingStreaming;
  const waitingFirstToken = isStreamingThis && !visibleContent && !isThinkingStreaming;

  const isPersisted = message.id > 0;
  const isMissingLlmNotice =
    !isStreamingThis &&
    (visibleContent.includes('modello LLM non è ancora stato scaricato') ||
      visibleContent.includes('Nessun modello LLM scaricato'));
  const showActions = !isStreamingThis && isPersisted && !isMissingLlmNotice;

  // Rendi "Impostazioni" un link sottolineato cliccabile anche nei messaggi già salvati
  const renderedContent =
    isMissingLlmNotice && onOpenSettings
      ? visibleContent.replace(/\*\*(Impostazioni)\*\*/gi, '[[action:settings|$1]]')
      : visibleContent;

  // ChatGPT 2026 dynamic status header con roll continuo tra fasi successive
  const isBusy = isStreamingThis && !visibleContent;
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (!isBusy) {
      setStepIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setStepIndex((prev) => prev + 1);
    }, 3500);
    return () => clearInterval(timer);
  }, [isBusy]);

  useEffect(() => {
    if (isThinkingStreaming && traceExpanded && thinkingContainerRef.current) {
      thinkingContainerRef.current.scrollTop = thinkingContainerRef.current.scrollHeight;
    }
  }, [combinedThinking, isThinkingStreaming, traceExpanded]);

  // Approccio A + Antigravity hierarchical timeline: visibile solo con dati reali
  const hasRealSources = Array.isArray(message.citations) && message.citations.length > 0;
  const hasThinking = Boolean(combinedThinking && combinedThinking.trim().length > 0);
  const showTrace = !isBusy && (hasThinking || hasRealSources);

  const traceHeaderLabel =
    hasThinking && hasRealSources
      ? 'Fonti e ragionamento del corso'
      : hasThinking
        ? 'Ragionamento del modello'
        : `${message.citations.length} ${message.citations.length === 1 ? 'fonte consultata' : 'fonti consultate'}`;

  return (
    <div
      className={clsx(
        'group flex flex-col items-start w-full max-w-full',
        (isStreamingThis || message.id < 0 || isMissingLlmNotice) && 'animate-fade-in'
      )}
    >
      {/* 1. Stato durante lo streaming attivo */}
      {isBusy && (
        <div className="w-full mb-1">
          {isThinkingStreaming ? (
            <div>
              <button
                type="button"
                onClick={() => setTraceExpanded((prev) => !prev)}
                className={clsx(
                  'inline-flex items-center py-1 font-normal text-zinc-400 hover:text-zinc-200 transition-colors group select-none cursor-pointer',
                  compact ? 'text-[11.5px]' : 'text-xs sm:text-[13px]'
                )}
                title={traceExpanded ? 'Nascondi passaggi' : 'Mostra ragionamento in tempo reale'}
              >
                <div className="relative inline-flex items-center min-h-[20px] gap-1.5 whitespace-nowrap">
                  <Brain size={compact ? 13 : 14} className="text-indigo-400 animate-pulse flex-shrink-0" />
                  <Text3DFlip
                    text={THINKING_REASONING_STEPS[stepIndex % THINKING_REASONING_STEPS.length]}
                    rotateDirection="top"
                    wordClassName="chat-shimmer-text font-medium"
                  />
                  <ChevronDown
                    size={compact ? 12 : 13}
                    className={clsx(
                      'text-zinc-500 hover:text-zinc-300 transition-transform duration-200 ml-0.5',
                      traceExpanded && 'rotate-180'
                    )}
                  />
                </div>
              </button>
              <div
                aria-hidden={!traceExpanded}
                className={clsx(
                  'grid w-full min-w-0 transition-[grid-template-rows,opacity] duration-200 ease-out',
                  traceExpanded
                    ? 'grid-rows-[1fr] opacity-100'
                    : 'grid-rows-[0fr] opacity-0 pointer-events-none'
                )}
              >
                <div className="overflow-hidden min-h-0 w-full min-w-0">
                  <div className="pl-3.5 border-l border-white/10 flex flex-col w-full py-1.5 mb-1">
                    <div className="inline-flex items-center gap-1.5 text-[12px] font-medium text-zinc-300 mb-1 select-none">
                      <Brain size={13} className="text-indigo-400 flex-shrink-0" />
                      <span>Ragionamento in corso...</span>
                    </div>
                    <div
                      ref={thinkingContainerRef}
                      className="ml-5 text-xs sm:text-[12.5px] text-zinc-400/90 leading-relaxed whitespace-pre-wrap select-text max-h-72 overflow-y-auto"
                    >
                      {combinedThinking || <span className="text-zinc-500 italic">Avvio del ragionamento del modello...</span>}
                      <span className="inline-block w-1.5 h-3.5 ml-1 bg-indigo-400 animate-pulse align-middle" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : waitingFirstToken ? (
            <div className="inline-flex items-center py-1 font-normal text-zinc-400 text-xs sm:text-[13px] gap-1.5 select-none">
              <Sparkles size={compact ? 13 : 14} className="text-zinc-400 animate-pulse flex-shrink-0" />
              <Text3DFlip
                text={RAG_SEARCH_STEPS[stepIndex % RAG_SEARCH_STEPS.length]}
                rotateDirection="top"
                wordClassName="chat-shimmer-text font-medium"
              />
            </div>
          ) : null}
        </div>
      )}

      {/* 2. Badge unificato stile timeline Antigravity (visibile SOLO se ci sono dati reali) */}
      {showTrace && (
        <div className="w-full min-w-0 mb-1.5 flex flex-col items-start">
          <button
            type="button"
            onClick={() => setTraceExpanded((prev) => !prev)}
            className={clsx(
              'inline-flex items-center py-0.5 gap-1.5 font-normal text-zinc-400 hover:text-zinc-200 transition-colors group select-none cursor-pointer',
              compact ? 'text-[11.5px]' : 'text-xs sm:text-[13px]'
            )}
            title={traceExpanded ? 'Nascondi dettagli' : 'Mostra passaggi ed elaborazione'}
          >
            {hasThinking ? (
              <Brain size={compact ? 13 : 14} className="text-zinc-400 group-hover:text-indigo-300 transition-colors flex-shrink-0" />
            ) : (
              <BookOpen size={compact ? 13 : 14} className="text-zinc-400 group-hover:text-zinc-200 transition-colors flex-shrink-0" />
            )}
            <span>{traceHeaderLabel}</span>
            <ChevronDown
              size={compact ? 12 : 13}
              className={clsx(
                'text-zinc-500 group-hover:text-zinc-300 transition-transform duration-200 ml-0.5',
                traceExpanded && 'rotate-180'
              )}
            />
          </button>

          {/* Timeline / Roadmap gerarchica espansa stile Antigravity (CSS Grid, zero layout shift) */}
          <div
            aria-hidden={!traceExpanded}
            className={clsx(
              'grid w-full min-w-0 max-w-2xl transition-[grid-template-rows,opacity] duration-200 ease-out',
              traceExpanded
                ? 'grid-rows-[1fr] opacity-100'
                : 'grid-rows-[0fr] opacity-0 pointer-events-none'
            )}
          >
            <div className="overflow-hidden min-h-0 w-full min-w-0">
              <div className="pl-3.5 border-l border-white/10 flex flex-col w-full space-y-3.5 py-1.5">
                {/* Step 1: Fonti e materiali consultati (se presenti) */}
                {hasRealSources && (
                  <div className="flex flex-col space-y-1.5">
                    <div className="inline-flex items-center gap-1.5 text-[12px] font-medium text-zinc-300 select-none">
                      <BookOpen size={13} className="text-zinc-400 flex-shrink-0" />
                      <span>Materiali e fonti consultate</span>
                    </div>
                    <div className="flex flex-col space-y-1 ml-5">
                      {message.citations.map((cit, idx) => {
                        const isLiveLecture = cit.source_type === 'live_lecture';
                        const isLecture = cit.source_type === 'lecture';
                        const isQuaderno = cit.source_type === 'quaderno' || cit.source_type === 'note';
                        const title = cit.title || cit.label || 'materiale del corso';

                        let description = '';
                        if (isLiveLecture) {
                          description = 'Trascrizione in diretta della lezione';
                        } else if (isLecture) {
                          description =
                            typeof cit.page === 'number'
                              ? `Trascrizione di "${title}" · min. ${cit.page}`
                              : `Trascrizione di "${title}"`;
                        } else if (isQuaderno) {
                          description = title ? `Appunti del Quaderno · "${title}"` : 'Appunti del Quaderno';
                        } else {
                          description =
                            typeof cit.page === 'number'
                              ? `Slide di "${title}" · pag. ${cit.page}`
                              : `Slide di "${title}"`;
                        }

                        return (
                          <div
                            key={`${cit.source_id}-${cit.page ?? 'x'}-${idx}`}
                            onClick={
                              isLiveLecture
                                ? undefined
                                : isLecture && onOpenAudioSnippet && typeof cit.page === 'number'
                                  ? () => onOpenAudioSnippet(Number(cit.source_id), Number(cit.page) * 60)
                                  : isQuaderno && onOpenQuadernoLecture && typeof cit.source_id === 'number'
                                    ? () => onOpenQuadernoLecture(Number(cit.source_id))
                                    : onOpenCitation
                                      ? () => onOpenCitation(cit.source_id, cit.page ?? undefined)
                                      : undefined
                            }
                            className={clsx(
                              'text-[11.5px] text-zinc-400 hover:text-zinc-200 transition-colors flex items-center gap-2 py-0.5 select-none',
                              (onOpenCitation || (isLecture && onOpenAudioSnippet) || (isQuaderno && onOpenQuadernoLecture)) &&
                                !isLiveLecture
                                ? 'cursor-pointer group/cit'
                                : 'cursor-default'
                            )}
                          >
                            <span className="w-1 h-1 rounded-full bg-zinc-600 group-hover/cit:bg-zinc-400 flex-shrink-0" />
                            <span className="truncate">{description}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Step 2: Ragionamento del modello (se presente, testo libero senza box e senza tasto copia) */}
                {hasThinking && (
                  <div className="flex flex-col space-y-1.5">
                    <div className="inline-flex items-center gap-1.5 text-[12px] font-medium text-zinc-300 select-none">
                      <Brain size={13} className="text-indigo-400 flex-shrink-0" />
                      <span>Ragionamento del modello</span>
                    </div>
                    <div
                      ref={thinkingContainerRef}
                      className="ml-5 text-xs sm:text-[12.5px] text-zinc-400/90 leading-relaxed whitespace-pre-wrap select-text max-h-80 overflow-y-auto"
                    >
                      {combinedThinking}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Borderless Flowing Markdown + KaTeX + Citations Prose */}
      {renderedContent && (
        <div className="w-full max-w-full text-zinc-100 select-text mt-1.5">
          <MarkdownContent
            content={renderedContent}
            onOpenCitation={onOpenCitation}
            onOpenAudioSnippet={onOpenAudioSnippet}
            onOpenSettings={onOpenSettings}
            className="chat-prose"
          />
        </div>
      )}

      {/* Interactive Study Action Card */}
      {message.action && (
        <div className="w-full">
          <ChatActionCard
            action={message.action}
            onOpenQuiz={onOpenQuiz}
            onOpenOralQuestion={onOpenOralQuestion}
            onOpenQuadernoLecture={onOpenQuadernoLecture}
          />
        </div>
      )}


      {/* Assistant Micro-Actions Toolbar (Copy, Regenerate on last, Save to Quaderno) */}
      {showActions && (
        <div className="w-full mt-2 flex items-center gap-1 text-[11px] text-zinc-400">
          <button
            type="button"
            onClick={() => handleCopy(visibleContent || message.content)}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-white/[0.06] hover:text-zinc-100 transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check size={12} className="text-emerald-400" />
                <span className="text-emerald-300">Copiato</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Copia</span>
              </>
            )}
          </button>

          {isLastAssistant && onRegenerate && (
            <button
              type="button"
              disabled={isAnyStreaming}
              onClick={() => onRegenerate(message.id)}
              title="Rigenera questa risposta"
              className="inline-flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-white/[0.06] hover:text-zinc-100 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw size={12} />
              <span>Rigenera</span>
            </button>
          )}

          {onSaveNote && (
            <button
              type="button"
              disabled={isSavingNote || isAnyStreaming}
              onClick={() => onSaveNote(message.id)}
              title="Salva questa risposta come appunto nel Quaderno"
              className={clsx(
                'inline-flex items-center gap-1 px-2 py-1 rounded-lg transition-colors cursor-pointer disabled:opacity-50',
                isSavedToQuaderno
                  ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25'
                  : 'hover:bg-white/[0.06] hover:text-zinc-100'
              )}
            >
              {isSavedToQuaderno ? <Check size={12} /> : <BookmarkPlus size={12} />}
              <span>
                {isSavingNote
                  ? 'Salvataggio...'
                  : isSavedToQuaderno
                    ? 'Salvato nel Quaderno'
                    : 'Salva nel Quaderno'}
              </span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
