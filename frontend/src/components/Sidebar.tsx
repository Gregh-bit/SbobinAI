import { useState } from 'react';
import { Plus, BookOpen, Settings, Trash2, Pencil } from 'lucide-react';
import { clsx } from 'clsx';
import type { Subject } from '../types';
import NewSubjectModal from './NewSubjectModal';

interface Props {
  subjects: Subject[];
  selectedId: number | null;
  onSelect: (subject: Subject) => void;
  onAdd: (data: { name: string; description: string; keywords: string; color: string }) => Promise<void>;
  onUpdate?: (id: number, data: { name: string; description: string; keywords: string; color: string }) => Promise<void>;
  onDelete?: (id: number) => Promise<void>;
  isLoading: boolean;
  isRecording?: boolean;
  onOpenSettings?: () => void;
}

export default function Sidebar({ subjects, selectedId, onSelect, onAdd, onUpdate, onDelete, isRecording = false, onOpenSettings }: Props) {
  const [showModal, setShowModal] = useState(false);
  const [subjectToEdit, setSubjectToEdit] = useState<Subject | null>(null);
  const [subjectToDelete, setSubjectToDelete] = useState<Subject | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  return (
    <>
      <aside className="w-64 h-full flex flex-col flex-shrink-0 liquid-glass border-r border-white/[0.08] select-none [contain:paint]">
        {/* macOS Native Window Controls Drag Region */}
        <div data-tauri-drag-region className="h-14 px-4 flex items-center gap-2.5 flex-shrink-0">
          {/* Spazio riservato ai Traffic Lights nativi macOS (72px) */}
          <div className="w-[72px] h-4 pointer-events-none" />
          <img
            src="/logo-wordmark.png"
            alt="SbobinAI"
            className="h-4.5 w-auto object-contain pointer-events-none select-none opacity-90 hover:opacity-100 transition-opacity"
          />
        </div>

        {/* Section Header */}
        <div className="px-4 pt-3 pb-2 flex items-center justify-between">
          <span className="text-[11px] font-semibold tracking-wider text-white/45 uppercase">
            Corsi
          </span>
          <span className="text-[11px] text-white/40 font-mono">
            {subjects.length}
          </span>
        </div>

        {/* Course List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {subjects.length === 0 ? (
            <div className="py-8 px-3 text-center">
              <BookOpen size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-2" />
              <p className="text-xs text-white/50">Nessun corso attivo.</p>
              <p className="text-[11px] text-white/35 mt-0.5">Aggiungine uno in basso.</p>
            </div>
          ) : (
            subjects.map((s) => {
              const isSelected = selectedId === s.id;
              // Durante una registrazione live il corso non può essere cambiato
              const isLocked = isRecording && !isSelected;
              return (
                <div
                  key={s.id}
                  onClick={() => {
                    if (!isLocked) onSelect(s);
                  }}
                  title={isLocked ? 'Termina la registrazione per cambiare corso' : undefined}
                  aria-disabled={isLocked || undefined}
                  className={clsx(
                    'w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-left transition-all duration-150 group',
                    isLocked ? 'cursor-not-allowed opacity-40' : 'cursor-pointer',
                    isSelected
                      ? 'bg-white/[0.12] text-white shadow-liquid-sm border border-white/[0.14]'
                      : 'text-white/70 hover:bg-white/[0.05] hover:text-white/95 border border-transparent'
                  )}
                >
                  <span
                    className="w-2 h-2 rounded-full flex-shrink-0 transition-transform group-hover:scale-110"
                    style={{ backgroundColor: s.color }}
                  />
                  <span className="flex-1 truncate text-xs font-medium tracking-tight">
                    {s.name}
                  </span>
                  {s.lecture_count > 0 && (
                    <span className={clsx(
                      'text-[11px] font-mono px-2 py-0.5 rounded-md flex-shrink-0 transition-opacity',
                      isSelected ? 'bg-white/20 text-white' : 'text-white/40 group-hover:text-white/60',
                      (onDelete || onUpdate) && !isRecording && 'group-hover:hidden'
                    )}>
                      {s.lecture_count}
                    </span>
                  )}
                  <div className={clsx('hidden items-center gap-1 flex-shrink-0', !isRecording && 'group-hover:flex')}>
                    {onUpdate && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSubjectToEdit(s);
                        }}
                        title={`Modifica ${s.name}`}
                        className="w-5 h-5 flex items-center justify-center rounded-md text-white/40 hover:text-white hover:bg-white/[0.10] transition-colors"
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
                        className="w-5 h-5 flex items-center justify-center rounded-md text-white/30 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <Trash2 size={11} strokeWidth={1.75} />
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Actions: Ingranaggio Impostazioni & + Nuovo Corso (both h-9 / 36px) */}
        <div className="p-3 border-t border-white/[0.08] flex items-center gap-2">
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
            className="w-full max-w-sm rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg p-6 space-y-4 animate-slide-up"
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
