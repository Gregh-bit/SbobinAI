import { ArrowRight, BookMarked, CheckCircle2, HelpCircle, Mic } from 'lucide-react';
import type { ChatStudyAction } from '../../types';

export interface ChatActionCardProps {
  action: ChatStudyAction;
  onOpenQuiz?: (quizId: number) => void;
  onOpenOralQuestion?: (question: string, topic?: string) => void;
}

/**
 * Interactive study card rendered directly beneath an assistant message when
 * the backend emits a structured study action (quiz, oral prompt, or Quaderno save).
 */
export default function ChatActionCard({
  action,
  onOpenQuiz,
  onOpenOralQuestion,
}: ChatActionCardProps) {
  if (action.type === 'quiz_created') {
    const canOpen = typeof action.quiz_id === 'number' && action.quiz_id > 0 && Boolean(onOpenQuiz);
    return (
      <div className="mt-3 rounded-2xl bg-zinc-900/80 border border-emerald-500/20 px-4 py-3 flex flex-wrap items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center flex-shrink-0 text-emerald-300">
            <HelpCircle size={17} strokeWidth={1.9} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider font-semibold text-emerald-400">
                Quiz pronto
              </span>
              {typeof action.num_questions === 'number' && action.num_questions > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 tabular-nums">
                  {action.num_questions} {action.num_questions === 1 ? 'domanda' : 'domande'}
                </span>
              )}
            </div>
            <p className="text-xs font-medium text-zinc-100 truncate mt-0.5">
              {action.title || 'Quiz di verifica generato'}
            </p>
          </div>
        </div>

        <button
          type="button"
          disabled={!canOpen}
          onClick={() => {
            if (typeof action.quiz_id === 'number') {
              onOpenQuiz?.(action.quiz_id);
            }
          }}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-200 border border-emerald-500/35 text-xs font-medium transition-all cursor-pointer disabled:opacity-45 disabled:cursor-not-allowed"
        >
          <span>Apri nella schermata Quiz</span>
          <ArrowRight size={13} />
        </button>
      </div>
    );
  }

  if (action.type === 'oral_question') {
    const questionText = action.question?.trim() || '';
    const canAnswer = Boolean(questionText && onOpenOralQuestion);
    return (
      <div className="mt-3 rounded-2xl bg-zinc-900/80 border border-indigo-500/25 px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-sm">
        <div className="flex items-start gap-3 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/25 flex items-center justify-center flex-shrink-0 text-indigo-300 mt-0.5">
            <Mic size={16} strokeWidth={1.9} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-300">
                Simulazione Orale
              </span>
              {action.topic && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-200/90 border border-indigo-500/20 truncate max-w-[220px]">
                  {action.topic}
                </span>
              )}
            </div>
            {questionText && (
              <p className="text-xs text-zinc-200 leading-relaxed mt-1 line-clamp-2">
                &ldquo;{questionText}&rdquo;
              </p>
            )}
          </div>
        </div>

        <button
          type="button"
          disabled={!canAnswer}
          onClick={() => {
            if (questionText) {
              onOpenOralQuestion?.(questionText, action.topic ?? undefined);
            }
          }}
          className="inline-flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-full bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-200 border border-indigo-500/35 text-xs font-medium transition-all cursor-pointer flex-shrink-0 disabled:opacity-45 disabled:cursor-not-allowed"
        >
          <span>Rispondi all&apos;Orale</span>
          <ArrowRight size={13} />
        </button>
      </div>
    );
  }

  if (action.type === 'saved_to_quaderno') {
    return (
      <div className="mt-3 rounded-2xl bg-zinc-900/75 border border-emerald-500/20 px-4 py-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/25 flex items-center justify-center flex-shrink-0 text-emerald-300">
            <BookMarked size={14} />
          </div>
          <div className="min-w-0">
            <p className="text-xs font-medium text-zinc-100 truncate">
              {action.title || 'Appunto salvato nel Quaderno'}
            </p>
            <p className="text-[11px] text-zinc-400">Disponibile nella scheda Quaderno del corso</p>
          </div>
        </div>
        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-300 flex-shrink-0">
          <CheckCircle2 size={13} />
          <span>Salvato</span>
        </span>
      </div>
    );
  }

  return null;
}
