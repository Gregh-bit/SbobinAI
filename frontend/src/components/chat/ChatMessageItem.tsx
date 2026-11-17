import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Copy,
  Check,
  RotateCcw,
  BookmarkPlus,
  Trash2,
  ChevronDown,
  ChevronRight,
  BookOpen,
  Brain,
  Pencil,
  CheckCircle2,
  Sparkles,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { PersistentChatMessage } from '../../types';
import MarkdownContent from '../MarkdownContent';
import ChatActionCard from './ChatActionCard';

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
  onOpenCitation?: (sourceId: string, page?: number) => void;
  onOpenQuiz?: (quizId: number) => void;
  onOpenOralQuestion?: (question: string, topic?: string) => void;
  onRegenerate?: (messageId: number) => void;
  onSaveNote?: (messageId: number) => void;
  onDelete?: (messageId: number) => void;
  onEdit?: (messageId: number, newText: string) => void;
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
  onOpenCitation,
  onOpenQuiz,
  onOpenOralQuestion,
  onRegenerate,
  onSaveNote,
  onDelete,
  onEdit,
}: ChatMessageItemProps) {
  const [copied, setCopied] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);
  const [sourcesExpanded, setSourcesExpanded] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const [showEditConfirmModal, setShowEditConfirmModal] = useState(false);
  const [showDeleteConfirmModal, setShowDeleteConfirmModal] = useState(false);

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

    return (
      <div className="group flex flex-col items-end gap-1 animate-fade-in relative w-full">
        {isEditing ? (
          <div className="w-full max-w-[85%] bg-zinc-800/95 border border-white/15 rounded-2xl p-3 shadow-lg flex flex-col gap-2">
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={Math.min(8, Math.max(2, editText.split('\n').length))}
              className="w-full bg-zinc-900/90 border border-white/10 rounded-xl p-2.5 text-zinc-100 text-[13px] leading-relaxed resize-y focus:outline-none focus:ring-1 focus:ring-indigo-400/50"
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
                className="px-3 py-1 rounded-xl text-xs text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.06] transition-colors cursor-pointer"
              >
                Annulla
              </button>
              <button
                type="button"
                disabled={!editText.trim() || editText.trim() === message.content.trim()}
                onClick={() => setShowEditConfirmModal(true)}
                className="px-3.5 py-1 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Conferma e Invia
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-zinc-800/80 border border-white/10 rounded-3xl px-4 py-2.5 max-w-[80%] text-zinc-100 text-[13px] leading-relaxed shadow-sm select-text">
              <p className="whitespace-pre-wrap break-words">{message.content}</p>
            </div>
            <div className="flex items-center gap-1 pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                type="button"
                onClick={() => handleCopy(message.content)}
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
            <div className="w-full max-w-sm rounded-2xl bg-zinc-900 border border-white/10 p-5 shadow-2xl flex flex-col gap-3 text-left">
              <div className="text-sm font-semibold text-zinc-100">Modifica messaggio</div>
              <p className="text-xs text-zinc-300 leading-relaxed">
                Modificare questo messaggio cancellerà tutte le risposte successive e l&apos;IA ne genererà una nuova con il testo aggiornato. Continuare?
              </p>
              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowEditConfirmModal(false)}
                  className="px-3.5 py-1.5 rounded-xl bg-white/[0.06] hover:bg-white/[0.1] text-zinc-300 text-xs font-medium transition-colors cursor-pointer"
                >
                  Annulla
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowEditConfirmModal(false);
                    setIsEditing(false);
                    onEdit?.(message.id, editText.trim());
                  }}
                  className="px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors cursor-pointer"
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
            <div className="w-full max-w-sm rounded-2xl bg-zinc-900 border border-white/10 p-5 shadow-2xl flex flex-col gap-3 text-left">
              <div className="text-sm font-semibold text-zinc-100">Elimina messaggio</div>
              <p className="text-xs text-zinc-300 leading-relaxed">
                Eliminare questo messaggio e tutta la conversazione successiva?
              </p>
              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowDeleteConfirmModal(false)}
                  className="px-3.5 py-1.5 rounded-xl bg-white/[0.06] hover:bg-white/[0.1] text-zinc-300 text-xs font-medium transition-colors cursor-pointer"
                >
                  Annulla
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowDeleteConfirmModal(false);
                    onDelete?.(message.id);
                  }}
                  className="px-3.5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium transition-colors cursor-pointer"
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

  const hasCitations = Array.isArray(message.citations) && message.citations.length > 0;
  const hasContextUsed = Boolean(message.context_used && message.context_used.trim());
  const isPersisted = message.id > 0;
  const showActions = !isStreamingThis && isPersisted;

  // ChatGPT 2026 dynamic status header
  const isBusy = isStreamingThis && !visibleContent;
  let headerLabel = 'Fonti ed elaborazione del corso';
  if (isThinkingStreaming) {
    headerLabel = 'Ragionamento sui materiali in corso...';
  } else if (waitingFirstToken) {
    headerLabel = 'Analisi delle fonti e appunti del corso...';
  }

  return (
    <div className="group flex flex-col items-start w-full max-w-full animate-fade-in">
      {/* Assistant Header: ChatGPT 2026 unified status & collapsible sources (Apple Dynamic Island Slide & Blur) */}
      <div className="w-full mb-1">
        <motion.button
          layout="position"
          type="button"
          onClick={() => setSourcesExpanded((prev) => !prev)}
          className="inline-flex items-center gap-1.5 py-1 text-xs sm:text-[13px] font-normal text-zinc-400 hover:text-zinc-200 transition-colors cursor-pointer group select-none"
          title="Mostra fonti ed elaborazione del corso"
        >
          <div className="relative overflow-hidden inline-flex items-center min-h-[20px]">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={headerLabel}
                initial={{ y: 8, opacity: 0, filter: 'blur(4px)' }}
                animate={{ y: 0, opacity: 1, filter: 'blur(0px)' }}
                exit={{ y: -8, opacity: 0, filter: 'blur(4px)' }}
                transition={{
                  duration: 0.24,
                  ease: [0.16, 1, 0.3, 1],
                }}
                className="inline-flex items-center gap-1.5"
              >
                {isBusy ? (
                  <Sparkles size={14} className="text-zinc-400 animate-pulse flex-shrink-0" />
                ) : (
                  <BookOpen size={14} className="text-zinc-400 group-hover:text-zinc-200 transition-colors flex-shrink-0" />
                )}
                <span className={clsx(isBusy && 'chat-shimmer-text font-medium')}>
                  {headerLabel}
                </span>
              </motion.div>
            </AnimatePresence>
          </div>
          <ChevronDown
            size={13}
            className={clsx(
              'text-zinc-500 group-hover:text-zinc-300 transition-transform duration-200 flex-shrink-0',
              sourcesExpanded && 'rotate-180'
            )}
          />
        </motion.button>

        <AnimatePresence>
          {sourcesExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0, y: -4 }}
              animate={{ opacity: 1, height: 'auto', y: 0 }}
              exit={{ opacity: 0, height: 0, y: -4 }}
              transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden"
            >
              <div className="mt-1.5 pl-3.5 border-l border-white/10 flex flex-col w-full max-w-xl space-y-1.5">
                {hasCitations ? (
                  message.citations.map((cit, idx) => {
                    const isLecture = cit.source_type === 'lecture';
                    const isQuaderno = cit.source_type === 'quaderno' || cit.source_type === 'note';
                    const title = cit.title || cit.label || 'materiale del corso';

                    let description = '';
                    if (isLecture) {
                      description =
                        typeof cit.page === 'number'
                          ? `Consultata la trascrizione di "${title}" al min. ${cit.page}`
                          : `Consultata la trascrizione di "${title}"`;
                    } else if (isQuaderno) {
                      description = title ? `Esaminati gli appunti del Quaderno: "${title}"` : 'Esaminati gli appunti del Quaderno';
                    } else {
                      description =
                        typeof cit.page === 'number'
                          ? `Verificate le slide di "${title}" a pag. ${cit.page}`
                          : `Verificate le slide di "${title}"`;
                    }

                    return (
                      <div
                        key={`${cit.source_id}-${cit.page ?? 'x'}-${idx}`}
                        onClick={() => onOpenCitation?.(cit.source_id, cit.page ?? undefined)}
                        className="text-zinc-400 text-xs py-0.5 leading-relaxed hover:text-zinc-100 transition-colors cursor-pointer select-text"
                      >
                        {description}
                      </div>
                    );
                  })
                ) : hasContextUsed ? (
                  <div className="text-zinc-400 text-xs py-0.5 leading-relaxed select-text">
                    Esaminati i materiali del corso: {message.context_used}
                  </div>
                ) : (
                  <div className="text-zinc-400 text-xs py-0.5 leading-relaxed select-text">
                    Analisi diretta dei documenti e del contesto del corso
                  </div>
                )}

                {/* Embedded Reasoning step inside sources if present */}
                {combinedThinking && (
                  <div className="mt-1 pt-1.5 border-t border-white/[0.06] space-y-1">
                    <button
                      type="button"
                      onClick={() => setThinkingExpanded((prev) => !prev)}
                      className="text-[11px] font-medium text-zinc-400 hover:text-zinc-200 flex items-center gap-1.5 transition-colors cursor-pointer"
                    >
                      <Brain size={12} className="text-zinc-400" />
                      <span>Passaggi di ragionamento</span>
                      {thinkingExpanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                    </button>
                    {thinkingExpanded && (
                      <div className="text-xs text-zinc-400/90 leading-relaxed whitespace-pre-wrap pl-2 border-l border-white/10 max-h-56 overflow-y-auto select-text font-mono text-[11px] mt-1">
                        {combinedThinking}
                      </div>
                    )}
                  </div>
                )}

                {!isStreamingThis && (
                  <div className="text-zinc-400 text-xs py-1 leading-relaxed flex items-center gap-2 border-t border-white/[0.06] mt-1 pt-1.5 font-medium">
                    <CheckCircle2 size={13} className="text-zinc-400 flex-shrink-0" />
                    <span>Consultazione materiali completata</span>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Borderless Flowing Markdown + KaTeX + Citations Prose */}
      {visibleContent && (
        <div className="w-full max-w-full text-zinc-100 select-text mt-1.5">
          <MarkdownContent
            content={visibleContent}
            onOpenCitation={onOpenCitation}
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
