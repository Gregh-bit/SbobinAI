import { useState, useRef, useEffect } from 'react';
import {
  Plus,
  MessageSquare,
  Pencil,
  Trash2,
  Check,
  X,
  Archive,
  PanelLeftClose,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { ChatSession } from '../../types';
import { useDelayedLoading, SkeletonBlock } from '../Skeleton';

export interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: number | null;
  isOpen: boolean;
  isLoading?: boolean;
  isCreating?: boolean;
  onSelectSession: (sessionId: number) => void;
  onCreateSession: () => void;
  onRenameSession: (sessionId: number, newTitle: string) => Promise<void> | void;
  onDeleteSession: (sessionId: number) => Promise<void> | void;
  onToggleSidebar?: () => void;
}

function formatRelativeTime(isoDate: string): string {
  const ts = Date.parse(isoDate);
  if (!Number.isFinite(ts)) return '';
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return 'Adesso';
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m fa`;
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours}h fa`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays === 1) return 'Ieri';
  if (diffDays < 7) return `${diffDays}g fa`;
  return new Date(ts).toLocaleDateString('it-IT', { day: 'numeric', month: 'short' });
}

/**
 * Collapsible ChatGPT-style left sidebar listing persistent chat sessions
 * for the current course with inline rename and compact delete confirmation.
 */
export default function ChatSidebar({
  sessions,
  activeSessionId,
  isOpen,
  isLoading = false,
  isCreating = false,
  onSelectSession,
  onCreateSession,
  onRenameSession,
  onDeleteSession,
  onToggleSidebar,
}: ChatSidebarProps) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [draftTitle, setDraftTitle] = useState('');
  const isCancellingRenameRef = useRef(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);
  const isListLoading = isLoading && sessions.length === 0;
  const showListSkeleton = useDelayedLoading(isListLoading, 140, 220);

  useEffect(() => {
    if (editingId !== null) {
      editInputRef.current?.focus();
      editInputRef.current?.select();
    }
  }, [editingId]);

  const startRename = (session: ChatSession) => {
    isCancellingRenameRef.current = false;
    setConfirmDeleteId(null);
    setEditingId(session.id);
    setDraftTitle(session.title);
  };

  const commitRename = async (session: ChatSession) => {
    if (isCancellingRenameRef.current) {
      isCancellingRenameRef.current = false;
      return;
    }
    const trimmed = draftTitle.trim();
    setEditingId(null);
    if (!trimmed || trimmed === session.title) return;
    setBusyId(session.id);
    try {
      await onRenameSession(session.id, trimmed);
    } finally {
      setBusyId(null);
    }
  };

  const handleConfirmDelete = async (sessionId: number) => {
    setBusyId(sessionId);
    try {
      await onDeleteSession(sessionId);
    } finally {
      setBusyId(null);
      setConfirmDeleteId(null);
    }
  };

  return (
    <>
      {/* Mobile / narrow backdrop when open */}
      {isOpen && (
        <div
          onClick={onToggleSidebar}
          className="md:hidden fixed inset-0 z-20 bg-black/50 backdrop-blur-sm"
          aria-hidden="true"
        />
      )}

      <aside
        aria-label="Cronologia chat"
        className={clsx(
          'flex-shrink-0 flex flex-col h-full border-r border-white/[0.07] bg-[#0e121c]/90 backdrop-blur-xl transition-all duration-200 ease-out overflow-hidden z-30',
          'fixed md:relative inset-y-0 left-0',
          isOpen
            ? 'w-64 opacity-100 translate-x-0'
            : 'w-0 -translate-x-full md:translate-x-0 opacity-0 border-r-0 pointer-events-none'
        )}
      >
        <div className="w-64 flex flex-col h-full">
          {/* Top bar: Nuova chat pill + mobile close */}
          <div className="p-3 flex items-center gap-2 border-b border-white/[0.06]">
            <button
              type="button"
              onClick={onCreateSession}
              disabled={isCreating}
              className="flex-1 flex items-center justify-center gap-2 px-3.5 py-2 rounded-full bg-white/[0.09] hover:bg-white/[0.15] active:scale-[0.99] text-zinc-100 border border-white/[0.12] text-xs font-medium transition-all cursor-pointer disabled:opacity-50"
            >
              <Plus size={14} strokeWidth={2.2} className="text-emerald-400" />
              <span>{isCreating ? 'Creazione...' : 'Nuova chat'}</span>
            </button>

            {onToggleSidebar && (
              <button
                type="button"
                onClick={onToggleSidebar}
                title="Chiudi barra laterale"
                className="md:hidden p-2 rounded-full text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.08] transition-colors cursor-pointer"
              >
                <PanelLeftClose size={15} />
              </button>
            )}
          </div>

          {/* Session list */}
          <div className="flex-1 overflow-y-auto px-2 py-2.5 space-y-1">
            {isListLoading || showListSkeleton ? (
              showListSkeleton ? (
                <div
                  aria-busy="true"
                  aria-label="Caricamento conversazioni"
                  className="space-y-1.5 px-1 py-1.5 animate-fade-in select-none"
                >
                  {(['w-4/5', 'w-2/3', 'w-3/4'] as const).map((titleWidth, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl px-2.5 py-2.5 bg-white/[0.03] border border-white/[0.06] space-y-2"
                    >
                      <SkeletonBlock className={clsx('h-3 rounded-full', titleWidth)} />
                      <SkeletonBlock className="h-2 w-1/3 rounded-full bg-white/[0.04]" />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-36" aria-busy="true" />
              )
            ) : sessions.length === 0 ? (
              <div className="px-3 py-8 text-center">
                <MessageSquare size={18} className="mx-auto text-zinc-600 mb-2" />
                <p className="text-[11px] text-zinc-400 leading-relaxed">
                  Nessuna conversazione salvata. Inizia una nuova chat con l&apos;assistente.
                </p>
              </div>
            ) : (
              sessions.map((session) => {
                const isActive = session.id === activeSessionId;
                const isEditing = editingId === session.id;
                const isConfirmingDelete = confirmDeleteId === session.id;
                const isBusy = busyId === session.id;
                const relTime = formatRelativeTime(session.updated_at || session.created_at);

                return (
                  <div
                    key={session.id}
                    className={clsx(
                      'group relative rounded-xl px-2.5 py-2 transition-all border',
                      isActive
                        ? 'bg-white/[0.09] border-white/[0.12] text-zinc-100 shadow-sm'
                        : 'bg-transparent border-transparent text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200'
                    )}
                  >
                    {isEditing ? (
                      <div className="flex items-center gap-1">
                        <input
                          ref={editInputRef}
                          type="text"
                          value={draftTitle}
                          onChange={(e) => setDraftTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              void commitRename(session);
                            } else if (e.key === 'Escape') {
                              e.preventDefault();
                              isCancellingRenameRef.current = true;
                              setEditingId(null);
                            }
                          }}
                          onBlur={() => void commitRename(session)}
                          className="w-full bg-[#121622]/90 border border-white/20 rounded-lg px-2 py-1 text-xs text-zinc-100 focus:outline-none focus:border-emerald-400/60"
                        />
                      </div>
                    ) : isConfirmingDelete ? (
                      <div className="flex items-center justify-between gap-1.5 py-0.5">
                        <span className="text-[11px] text-rose-300 font-medium truncate">
                          Eliminare chat?
                        </span>
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={() => void handleConfirmDelete(session.id)}
                            title="Conferma eliminazione"
                            className="p-1 rounded-lg bg-rose-500/20 hover:bg-rose-500/35 text-rose-300 border border-rose-500/30 transition-colors cursor-pointer disabled:opacity-50"
                          >
                            <Check size={12} />
                          </button>
                          <button
                            type="button"
                            disabled={isBusy}
                            onClick={() => setConfirmDeleteId(null)}
                            title="Annulla"
                            className="p-1 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] text-zinc-300 transition-colors cursor-pointer"
                          >
                            <X size={12} />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div
                        onClick={() => {
                          setConfirmDeleteId(null);
                          onSelectSession(session.id);
                        }}
                        className="flex items-start justify-between gap-1.5 cursor-pointer"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-medium truncate leading-snug">
                              {session.title || 'Nuova chat'}
                            </span>
                            {session.compacted_up_to_message_id > 0 && (
                              <span
                                title="Contesto compattato in memoria"
                                className="flex-shrink-0 text-indigo-400/80"
                              >
                                <Archive size={10} />
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-1.5 mt-0.5 text-[10px] text-zinc-500 tabular-nums">
                            <span>
                              {session.message_count}{' '}
                              {session.message_count === 1 ? 'messaggio' : 'messaggi'}
                            </span>
                            {relTime && (
                              <>
                                <span>·</span>
                                <span>{relTime}</span>
                              </>
                            )}
                          </div>
                        </div>

                        <div
                          className={clsx(
                            'flex items-center gap-0.5 flex-shrink-0 transition-opacity',
                            isActive
                              ? 'opacity-70 group-hover:opacity-100'
                              : 'opacity-0 group-hover:opacity-100'
                          )}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            onClick={() => startRename(session)}
                            title="Rinomina chat"
                            className="p-1 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.08] transition-colors cursor-pointer"
                          >
                            <Pencil size={11} />
                          </button>
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteId(session.id)}
                            title="Elimina chat"
                            className="p-1 rounded-lg text-zinc-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </aside>
    </>
  );
}
