import { useState, useEffect } from 'react';
import { Plus, BookOpen, Settings, Trash2, Pencil } from 'lucide-react';
import { clsx } from 'clsx';
import type { Subject } from '../types';
import NewSubjectModal from './NewSubjectModal';
import { SidebarSkeleton, useDelayedLoading } from './Skeleton';

interface Props {
  subjects: Subject[];
  selectedId: number | null;
  onSelect: (subject: Subject) => void;
  onAdd: (data: { name: string; description: string; keywords: string; color: string }) => Promise<void>;
  onUpdate?: (id: number, data: { name: string; description: string; keywords: string; color: string }) => Promise<void>;
  onDelete?: (id: number) => Promise<void>;
  onMoveLecture?: (lectureId: number, targetSubjectId: number) => Promise<void>;
  isLoading?: boolean;
  isRecording?: boolean;
  onOpenSettings?: () => void;
  [key: string]: any;
}

export default function Sidebar({ subjects, selectedId, onSelect, onAdd, onUpdate, onDelete, onMoveLecture, isLoading = false, isRecording = false, onOpenSettings }: Props) {
  const [showModal, setShowModal] = useState(false);
  const [subjectToEdit, setSubjectToEdit] = useState<Subject | null>(null);
  const [subjectToDelete, setSubjectToDelete] = useState<Subject | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [dragOverSubjectId, setDragOverSubjectId] = useState<number | null>(null);
  const [isDraggingActive, setIsDraggingActive] = useState(false);
  const isInitialLoading = Boolean(isLoading) && subjects.length === 0;
  const showSkeleton = useDelayedLoading(isInitialLoading, 140, 220);

  // Chiusura modal di eliminazione corso con ESC e tracciamento globale dello stato di drag
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isDeleting) {
        e.stopPropagation();
        setSubjectToDelete(null);
      }
    };
    const handleDragStart = () => {
      setIsDraggingActive(true);
    };
    const handleDragEndGlobal = () => {
      setIsDraggingActive(false);
      setDragOverSubjectId(null);
    };

    window.addEventListener('keydown', handleKey);
    window.addEventListener('dragstart', handleDragStart);
    window.addEventListener('dragend', handleDragEndGlobal);
    window.addEventListener('drop', handleDragEndGlobal);
    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('dragstart', handleDragStart);
      window.removeEventListener('dragend', handleDragEndGlobal);
      window.removeEventListener('drop', handleDragEndGlobal);
    };
  }, [subjectToDelete, isDeleting]);

  return (
    <>
      <aside className="w-64 h-full flex flex-col flex-shrink-0 liquid-glass border-r border-white/[0.08] select-none [contain:paint]">
        {/* macOS Native Window Controls Drag Region */}
        <div data-tauri-drag-region className="h-14 relative flex items-center justify-center flex-shrink-0 px-4">
          <img
            src="/logo-wordmark.png"
            alt="SbobinAI"
            className="h-[18px] max-h-[18px] w-auto object-contain pointer-events-none select-none opacity-90"
          />
        </div>

        {/* Section Header */}
        <div
          className="px-4 pt-3 pb-2 flex items-center justify-between"
          onDragOver={() => {
            if (dragOverSubjectId !== null) setDragOverSubjectId(null);
          }}
        >
          <span className="text-[11px] font-semibold tracking-wider text-white/45 uppercase">
            Corsi
          </span>
          <span className="text-[11px] text-white/40 font-mono">
            {isInitialLoading || showSkeleton ? '' : subjects.length}
          </span>
        </div>

        {/* Course List */}
        <div
          className="flex-1 overflow-y-auto px-2 pt-1.5 pb-2 space-y-1"
          onDragOver={(e) => {
            // Se il mouse sta passando sullo spazio vuoto del contenitore (sotto i corsi)
            if (e.target === e.currentTarget && dragOverSubjectId !== null) {
              setDragOverSubjectId(null);
            }
          }}
          onDragLeave={(e) => {
            // Se il mouse esce completamente dall'area della lista dei corsi, azzera l'evidenziazione
            if (!e.currentTarget.contains(e.relatedTarget as Node)) {
              setDragOverSubjectId(null);
            }
          }}
        >
          {subjects.length === 0 ? (
            isInitialLoading || showSkeleton ? (
              showSkeleton ? <SidebarSkeleton /> : null
            ) : (
              <div className="py-8 px-3 text-center">
                <BookOpen size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-2" />
                <p className="text-xs text-white/50">Nessun corso attivo.</p>
                <p className="text-[11px] text-white/35 mt-0.5">Aggiungine uno in basso.</p>
              </div>
            )
          ) : (
            subjects.map((s) => {
              const isSelected = selectedId === s.id;
              // Durante una registrazione live il corso non può essere cambiato
              const isLocked = isRecording && !isSelected;
              // Una lezione non può essere spostata nel corso in cui si trova già
              const canBeDragTarget = !isLocked && !isSelected;
              const isDragTarget = canBeDragTarget && dragOverSubjectId === s.id;

              return (
                <div
                  key={s.id}
                  onClick={() => {
                    if (!isLocked) onSelect(s);
                  }}
                  onDragEnter={(e) => {
                    if (canBeDragTarget) {
                      e.preventDefault();
                      e.stopPropagation();
                      setDragOverSubjectId(s.id);
                    } else if (dragOverSubjectId !== null) {
                      setDragOverSubjectId(null);
                    }
                  }}
                  onDragOver={(e) => {
                    if (canBeDragTarget) {
                      e.preventDefault();
                      e.stopPropagation();
                      e.dataTransfer.dropEffect = 'move';
                      if (dragOverSubjectId !== s.id) {
                        setDragOverSubjectId(s.id);
                      }
                    } else if (dragOverSubjectId !== null) {
                      setDragOverSubjectId(null);
                    }
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                      setDragOverSubjectId((prev) => (prev === s.id ? null : prev));
                    }
                  }}
                  onDrop={async (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setDragOverSubjectId(null);
                    if (!canBeDragTarget) return;
                    try {
                      const raw = e.dataTransfer.getData('application/json') || e.dataTransfer.getData('text/plain');
                      if (!raw) return;
                      let parsed: any = null;
                      try {
                        parsed = JSON.parse(raw);
                      } catch {
                        if (raw.startsWith('lecture:')) {
                          parsed = { type: 'sbobinai-lecture', lectureId: Number(raw.replace('lecture:', '')) };
                        }
                      }
                      if (parsed && (parsed.type === 'sbobinai-lecture' || parsed.lectureId)) {
                        const targetLecId = Number(parsed.lectureId);
                        if (!targetLecId) return;
                        if (parsed.sourceSubjectId === s.id) return;
                        if (onMoveLecture) {
                          await onMoveLecture(targetLecId, s.id);
                        }
                      }
                    } catch (err) {
                      console.warn('[Sidebar] Errore elaborazione drop lezione:', err);
                    }
                  }}
                  title={
                    isLocked
                      ? 'Termina la registrazione per cambiare corso'
                      : isSelected
                      ? s.name
                      : 'Trascina qui una lezione dal Quaderno per spostarla in questo corso'
                  }
                  aria-disabled={isLocked || undefined}
                  className={clsx(
                    'w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-left transition-colors duration-150 group relative select-none box-border',
                    isLocked ? 'cursor-not-allowed opacity-40' : 'cursor-pointer',
                    isDragTarget
                      ? 'bg-blue-500/20 text-white border border-transparent'
                      : isSelected
                      ? 'bg-white/[0.12] text-white shadow-liquid-sm border border-white/[0.14]'
                      : 'text-white/70 hover:bg-white/[0.06] hover:text-white/95 border border-transparent'
                  )}
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0 pointer-events-none"
                    style={{ backgroundColor: s.color }}
                  />
                  <span className="flex-1 truncate text-xs font-medium tracking-tight pr-2 pointer-events-none">
                    {s.name}
                  </span>
                  {s.lecture_count > 0 && (
                    <span
                      className={clsx(
                        'text-[11px] font-mono px-2 py-0.5 rounded-md flex-shrink-0 transition-colors pointer-events-none',
                        isDragTarget
                          ? 'bg-blue-400/25 text-blue-100'
                          : isSelected
                          ? 'bg-white/20 text-white'
                          : 'text-white/40 group-hover:text-white/60'
                      )}
                    >
                      {s.lecture_count}
                    </span>
                  )}
                  {/* Action buttons (overlay in hover per modifica/eliminazione; nascosto durante il drag) */}
                  {!isRecording && !isDraggingActive && !isDragTarget && (onUpdate || onDelete) && (
                    <div className="absolute right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity bg-[#161a24] px-1 py-0.5 rounded-lg border border-white/[0.08] shadow-sm">
                      {onUpdate && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSubjectToEdit(s);
                          }}
                          title={`Modifica ${s.name}`}
                          className="w-5 h-5 flex items-center justify-center rounded-md text-white/40 hover:text-white hover:bg-white/[0.12] transition-colors cursor-pointer"
                        >
                          <Pencil size={11} strokeWidth={1.75} />
                        </button>
                      )}
                      {onDelete && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteError(null);
                            setSubjectToDelete(s);
                          }}
                          title={`Elimina ${s.name}`}
                          className="w-5 h-5 flex items-center justify-center rounded-md text-white/30 hover:text-red-400 hover:bg-red-500/15 transition-colors cursor-pointer"
                        >
                          <Trash2 size={11} strokeWidth={1.75} />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Actions: Ingranaggio Impostazioni & + Nuovo Corso (both h-9 / 36px) */}
        <div
          className="p-3 border-t border-white/[0.08] flex items-center gap-2"
          onDragOver={() => {
            if (dragOverSubjectId !== null) setDragOverSubjectId(null);
          }}
        >
          <button
            type="button"
            onClick={onOpenSettings}
            title="Impostazioni"
            className="w-9 h-9 rounded-xl flex items-center justify-center text-white/50 hover:text-white hover:bg-white/[0.08] border border-white/[0.08] hover:border-white/[0.16] transition-all flex-shrink-0 cursor-pointer"
          >
            <Settings size={15} strokeWidth={1.75} />
          </button>
          <button
            type="button"
            onClick={() => {
              setSubjectToEdit(null);
              setShowModal(true);
            }}
            className="flex-1 h-9 flex items-center justify-center gap-1.5 px-3 rounded-xl text-xs font-medium text-white/80 hover:text-white hover:bg-white/[0.08] border border-white/[0.08] hover:border-white/[0.16] transition-all cursor-pointer"
          >
            <Plus size={13} strokeWidth={2} />
            <span>Nuovo Corso</span>
          </button>
        </div>
      </aside>

      <NewSubjectModal
        isOpen={showModal || !!subjectToEdit}
        subjectToEdit={subjectToEdit}
        onClose={() => {
          setShowModal(false);
          setSubjectToEdit(null);
        }}
        onSubmit={async (data) => {
          if (subjectToEdit && onUpdate) {
            await onUpdate(subjectToEdit.id, data);
          } else {
            await onAdd(data);
          }
        }}
      />

      {/* Confirmation Modal Eliminazione Corso */}
      {subjectToDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in"
          onClick={() => !isDeleting && setSubjectToDelete(null)}
        >
          <div
            className="w-full max-w-sm rounded-3xl liquid-modal p-6 space-y-4 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-red-500/15 border border-red-500/25 flex items-center justify-center text-red-400 flex-shrink-0">
                <Trash2 size={18} strokeWidth={2} />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-white">Elimina Corso</h3>
                <p className="text-xs text-white/50 truncate">{subjectToDelete.name}</p>
              </div>
            </div>

            <p className="text-xs leading-relaxed text-white/70">
              Sei sicuro di voler eliminare questo corso? Verranno rimossi tutti i materiali, appunti e registrazioni associati.
            </p>

            {deleteError && (
              <p className="text-xs leading-relaxed text-red-300 bg-red-500/10 border border-red-500/25 rounded-xl px-3 py-2">
                Impossibile eliminare il corso: {deleteError}
              </p>
            )}

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setSubjectToDelete(null)}
                disabled={isDeleting}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
              >
                Annulla
              </button>
              <button
                type="button"
                onClick={async () => {
                  if (!onDelete || !subjectToDelete) return;
                  setIsDeleting(true);
                  setDeleteError(null);
                  try {
                    await onDelete(subjectToDelete.id);
                    setSubjectToDelete(null);
                  } catch (err) {
                    setDeleteError(err instanceof Error ? err.message : String(err));
                  } finally {
                    setIsDeleting(false);
                  }
                }}
                disabled={isDeleting}
                className="px-4 py-1.5 rounded-full text-xs font-semibold text-red-300 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {isDeleting ? 'Eliminazione...' : 'Elimina Corso'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
