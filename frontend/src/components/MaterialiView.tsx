// components/MaterialiView.tsx — Sezione Materiali: Tutta la memoria del database del corso (File + Registrazioni + InfoCorso)
import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Upload, FileText, Presentation, File, Trash2, Plus, 
  Search, Sparkles, X, Info, Mic, RotateCcw
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { CourseMaterial, Lecture, Subject } from '../types';
import AddMaterialModal from './AddMaterialModal';
import MarkdownContent from './MarkdownContent';

interface Props {
  subject: Subject;
  onDeleteLecturePermanently?: (id: number) => Promise<void>;
  onRestoreLectureToQuaderno?: (id: number) => Promise<void>;
  onLectureCreated?: (lecture: Lecture) => void;
}

export default function MaterialiView({ 
  subject, 
  onDeleteLecturePermanently, 
  onRestoreLectureToQuaderno,
  onLectureCreated,
}: Props) {
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [previewMaterial, setPreviewMaterial] = useState<CourseMaterial | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<CourseMaterial | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Guardia contro risposte obsolete (dismount o richieste superate)
  const isMountedRef = useRef(true);
  const loadSeqRef = useRef(0);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const loadMaterials = useCallback(async () => {
    const seq = ++loadSeqRef.current;
    const isStale = () => !isMountedRef.current || seq !== loadSeqRef.current;
    setLoading(true);
    try {
      const data = await api.subjects.getMaterials(subject.id);
      if (isStale()) return;
      setMaterials(data);
    } catch {
      if (isStale()) return;
      // Fallback locale
      const fallbackInfo: CourseMaterial = {
        id: `infocorso-${subject.id}`,
        name: 'InfoCorso.md',
        type: 'doc',
        size: '1.2 KB',
        uploadedAt: 'Auto-importato',
        pageCount: 1,
        content: `# Informazioni del Corso — ${subject.name}\n\n## Descrizione\n${subject.description || 'Nessuna descrizione specificata.'}`,
        isAutoImported: true,
      };
      setMaterials([fallbackInfo]);
    } finally {
      if (!isStale()) setLoading(false);
    }
  }, [subject]);

  useEffect(() => {
    loadMaterials();
  }, [loadMaterials]);

  const filtered = materials.filter((m) =>
    m.name.toLowerCase().includes(search.toLowerCase())
  );

  const handleDeleteItem = (mat: CourseMaterial, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteError(null);
    setItemToDelete(mat);
  };

  const confirmDeleteItem = async () => {
    if (!itemToDelete) return;
    setIsDeleting(true);
    setDeleteError(null);
    try {
      if (itemToDelete.lecture_id) {
        if (onDeleteLecturePermanently) {
          await onDeleteLecturePermanently(itemToDelete.lecture_id);
        } else {
          await api.lectures.delete(itemToDelete.lecture_id);
        }
      } else if (!itemToDelete.isAutoImported) {
        // In caso di errore il materiale NON viene rimosso dalla lista
        await api.subjects.deleteMaterialFile(subject.id, itemToDelete.name);
      }
      setMaterials((prev) => prev.filter((m) => m.id !== itemToDelete.id));
      if (previewMaterial?.id === itemToDelete.id) setPreviewMaterial(null);
      setItemToDelete(null);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsDeleting(false);
    }
  };

  const handleRestore = async (mat: CourseMaterial, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (!mat.lecture_id) return;
    if (onRestoreLectureToQuaderno) {
      await onRestoreLectureToQuaderno(mat.lecture_id);
    } else {
      await api.lectures.update(mat.lecture_id, { in_quaderno: true });
    }
    setMaterials((prev) =>
      prev.map((m) => (m.id === mat.id ? { ...m, in_quaderno: true } : m))
    );
    if (previewMaterial?.id === mat.id) {
      setPreviewMaterial((prev) => (prev ? { ...prev, in_quaderno: true } : null));
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-black/15 select-none relative">
      {/* Top Bar / Search & Upload */}
      <div className="px-8 py-3.5 border-b border-white/[0.06] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={12} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Cerca nella memoria..."
              className="pl-8 pr-3 py-1.5 rounded-full liquid-input text-xs text-white placeholder-white/20 w-52 focus:outline-none"
            />
          </div>
          <span className="text-[11px] text-white/40 font-mono">
            {filtered.length} {filtered.length === 1 ? 'elemento' : 'elementi salvati'}
          </span>
        </div>

        <div>
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full liquid-glass-pill text-xs font-semibold text-white hover:bg-white/[0.16] shadow-liquid-sm transition-all"
          >
            <Plus size={12} strokeWidth={2.5} />
            <span>Aggiungi Materiale</span>
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-8">
        {loading ? (
          <div className="h-64 flex items-center justify-center text-xs text-white/30 font-mono">
            Caricamento memoria del corso...
          </div>
        ) : filtered.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-center">
            <Upload size={24} strokeWidth={1.5} className="text-white/20 mb-3" />
            <h4 className="text-sm font-medium text-white/70">Nessun elemento presente</h4>
            <p className="text-xs text-white/30 mt-1 max-w-xs">
              Aggiungi slide, PDF, appunti o registrazioni per indicizzarle nella memoria di {subject.name}.
            </p>
            <button
              onClick={() => setShowAddModal(true)}
              className="mt-3 px-3.5 py-1.5 rounded-full liquid-glass-pill text-xs font-medium text-white hover:bg-white/[0.12] transition-all"
            >
              + Aggiungi il primo materiale
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filtered.map((mat) => {
              const isLecture = mat.type === 'lecture';
              const isInQuaderno = mat.in_quaderno !== false;

              return (
                <div
                  key={mat.id}
                  onClick={() => setPreviewMaterial(mat)}
                  className={clsx(
                    'group p-4 rounded-2xl liquid-glass border cursor-pointer transition-all duration-200 flex flex-col justify-between',
                    mat.isAutoImported
                      ? 'border-white/[0.18] bg-white/[0.06] hover:bg-white/[0.09] shadow-liquid'
                      : isLecture
                      ? 'border-white/[0.12] bg-white/[0.04] hover:bg-white/[0.08] hover:border-white/[0.20]'
                      : 'border-white/[0.08] hover:border-white/[0.16] hover:bg-white/[0.06]'
                  )}
                >
                  <div>
                    <div className="flex items-start justify-between mb-3">
                      <div className="w-10 h-10 rounded-xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center group-hover:scale-105 transition-transform">
                        {mat.isAutoImported ? (
                          <Info size={18} strokeWidth={1.75} className="text-emerald-400" />
                        ) : isLecture ? (
                          <Mic size={18} strokeWidth={1.75} className="text-amber-400" />
                        ) : mat.type === 'slide' ? (
                          <Presentation size={18} strokeWidth={1.75} className="text-orange-400" />
                        ) : mat.type === 'pdf' ? (
                          <FileText size={18} strokeWidth={1.75} className="text-red-400" />
                        ) : (
                          <File size={18} strokeWidth={1.75} className="text-blue-400" />
                        )}
                      </div>

                      <div className="flex items-center gap-1">
                        {!mat.isAutoImported && (
                          <button
                            onClick={(e) => handleDeleteItem(mat, e)}
                            title="Elimina definitivamente da tutto"
                            className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all"
                          >
                            <Trash2 size={12} strokeWidth={1.75} />
                          </button>
                        )}

                        {mat.isAutoImported && (
                          <span className="text-[9px] font-mono text-emerald-300 bg-emerald-500/15 border border-emerald-500/30 px-1.5 py-0.5 rounded-full flex items-center gap-1">
                            <Sparkles size={9} />
                            InfoCorso
                          </span>
                        )}
                      </div>
                    </div>

                    <h4 className="text-xs font-semibold tracking-tight text-white/95 line-clamp-2 leading-snug">
                      {mat.name}
                    </h4>

                    {/* Stato Lezione nel Quaderno */}
                    {isLecture && (
                      <div className="mt-1.5 flex items-center gap-1.5">
                        <span className={clsx(
                          'text-[9px] font-mono px-1.5 py-0.2 rounded-full border',
                          isInQuaderno
                            ? 'text-emerald-300/90 bg-emerald-500/10 border-emerald-500/20'
                            : 'text-amber-300/90 bg-amber-500/10 border-amber-500/20'
                        )}>
                          {isInQuaderno ? 'Nel Quaderno' : 'Solo in Memoria'}
                        </span>

                        {!isInQuaderno && (
                          <button
                            onClick={(e) => handleRestore(mat, e)}
                            className="text-[9px] font-mono text-white/60 hover:text-white underline underline-offset-2 flex items-center gap-0.5"
                            title="Ripristina nel Quaderno"
                          >
                            <RotateCcw size={8} />
                            <span>Ripristina</span>
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-[10px] text-white/40 font-mono">
                    <span>{mat.size}</span>
                    <span>{mat.uploadedAt}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ── Document Preview Modal (Liquid Glass) ── */}
      {previewMaterial && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/60 backdrop-blur-md animate-fade-in"
          onClick={() => setPreviewMaterial(null)}
        >
          <div 
            className="w-full max-w-xl max-h-[80vh] flex flex-col rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg p-6 animate-slide-up"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center">
                  {previewMaterial.type === 'lecture' ? (
                    <Mic size={16} className="text-amber-400" />
                  ) : previewMaterial.isAutoImported ? (
                    <Info size={16} className="text-emerald-400" />
                  ) : (
                    <FileText size={16} className="text-blue-400" />
                  )}
                </div>
                <div>
                  <h3 className="text-sm font-semibold tracking-tight text-white">
                    {previewMaterial.name}
                  </h3>
                  <div className="flex items-center gap-2 mt-0.5 text-[10px] font-mono text-white/40">
                    <span>{previewMaterial.size}</span>
                    <span>·</span>
                    <span>{previewMaterial.uploadedAt}</span>
                    {previewMaterial.lecture_id && (
                      <span className={clsx(
                        'px-1.5 py-0.2 rounded-full border',
                        previewMaterial.in_quaderno !== false
                          ? 'text-emerald-300 border-emerald-500/20'
                          : 'text-amber-300 border-amber-500/20'
                      )}>
                        {previewMaterial.in_quaderno !== false ? 'Presente nel Quaderno' : 'Archiviato solo in Memoria'}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {previewMaterial.lecture_id && previewMaterial.in_quaderno === false && (
                  <button
                    onClick={() => handleRestore(previewMaterial)}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white transition-all"
                  >
                    <RotateCcw size={11} />
                    <span>Ripristina nel Quaderno</span>
                  </button>
                )}

                <button
                  onClick={() => setPreviewMaterial(null)}
                  className="w-6 h-6 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors"
                >
                  <X size={14} strokeWidth={2} />
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto py-5 pr-2 space-y-4 text-xs leading-relaxed text-white/80">
              {previewMaterial.content ? (
                <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/[0.06]">
                  <MarkdownContent content={previewMaterial.content} className="quaderno-markdown" />
                </div>
              ) : (
                <div className="py-12 text-center text-white/40 italic">
                  Nessun contenuto di testo memorizzato per questo file.
                </div>
              )}
            </div>

            {/* Modal Footer: Action to delete permanently */}
            {!previewMaterial.isAutoImported && (
              <div className="pt-3 border-t border-white/[0.06] flex items-center justify-between text-xs">
                <span className="text-[10px] text-white/30 font-mono">
                  Elemento salvato nel database locale
                </span>
                <button
                  onClick={(e) => handleDeleteItem(previewMaterial, e)}
                  className="flex items-center gap-1 px-3 py-1 rounded-full text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 size={12} />
                  <span>Elimina definitivamente da tutto</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Modal Selettore Tipologia Materiale ── */}
      <AddMaterialModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        subject={subject}
        onMaterialAdded={(newMat) => {
          setMaterials((prev) => [newMat, ...prev]);
        }}
        onLectureCreated={(newLecture) => {
          if (onLectureCreated) {
            onLectureCreated(newLecture);
          }
        }}
      />

      {/* ── Modal Conferma Eliminazione Materiale (Liquid Glass) ── */}
      {itemToDelete && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in"
          onClick={() => !isDeleting && setItemToDelete(null)}
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
                <h3 className="text-sm font-semibold text-white">Elimina Materiale</h3>
                <p className="text-xs text-white/50 truncate">{itemToDelete.name}</p>
              </div>
            </div>

            <p className="text-xs leading-relaxed text-white/70">
              {itemToDelete.lecture_id
                ? 'Eliminare definitivamente questa lezione dalla memoria del corso? Verrà rimossa sia da Materiali che dal Quaderno.'
                : `Eliminare definitivamente "${itemToDelete.name}" dal disco e dalla memoria del corso?`}
            </p>

            {deleteError && (
              <p className="text-xs leading-relaxed text-red-300 bg-red-500/10 border border-red-500/25 rounded-xl px-3 py-2">
                Eliminazione non riuscita: {deleteError}
              </p>
            )}

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setItemToDelete(null)}
                disabled={isDeleting}
                className="px-3.5 py-1.5 rounded-full text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
              >
                Annulla
              </button>
              <button
                type="button"
                onClick={confirmDeleteItem}
                disabled={isDeleting}
                className="px-4 py-1.5 rounded-full text-xs font-semibold text-red-300 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 transition-all flex items-center gap-1.5 cursor-pointer"
              >
                {isDeleting ? 'Eliminazione...' : 'Elimina Definitivamente'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
