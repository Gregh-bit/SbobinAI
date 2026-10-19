// components/MaterialiFinderModal.tsx — Finder macOS Liquid Glass per importare file da "Materiali" nel "Quaderno"
import { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  X, Search, Check, Plus, FileText, Presentation, 
  File, Mic, Info, BookOpen, Loader2
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { CourseMaterial, Lecture, Subject } from '../types';
import SegmentedControl from './SegmentedControl';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  subject: Subject;
  lecturesInQuaderno: Lecture[];
  onIncludeLectureInQuaderno: (lectureId: number) => Promise<void>;
  onImportMaterialAsLecture: (data: {
    title: string;
    processed_notes?: string;
    source_type?: string;
    date?: string;
    duration_seconds?: number;
  }) => Promise<Lecture>;
  onOpenAddMaterialModal?: () => void;
}

type FilterCategory = 'all' | 'audio' | 'notes' | 'slide_pdf';

export default function MaterialiFinderModal({
  isOpen,
  onClose,
  subject,
  lecturesInQuaderno,
  onIncludeLectureInQuaderno,
  onImportMaterialAsLecture,
  onOpenAddMaterialModal,
}: Props) {
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<FilterCategory>('all');
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Set of lecture IDs that are currently in Quaderno
  const inQuadernoLectureIds = useMemo(() => {
    return new Set(lecturesInQuaderno.map((l) => l.id));
  }, [lecturesInQuaderno]);

  // Set of titles in Quaderno (for matching file-based notes)
  const inQuadernoTitles = useMemo(() => {
    return new Set(lecturesInQuaderno.map((l) => l.title.toLowerCase().trim()));
  }, [lecturesInQuaderno]);

  const loadMaterials = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.subjects.getMaterials(subject.id);
      setMaterials(data);
    } catch {
      // Fallback base
      setMaterials([]);
    } finally {
      setLoading(false);
    }
  }, [subject.id]);

  useEffect(() => {
    if (isOpen) {
      loadMaterials();
      setSearch('');
      setCategoryFilter('all');
      setProcessingId(null);
    }
  }, [isOpen, loadMaterials]);

  // Close on Escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const isMaterialInQuaderno = (mat: CourseMaterial): boolean => {
    if (mat.lecture_id) {
      return inQuadernoLectureIds.has(mat.lecture_id);
    }
    const cleanName = mat.name.replace(/\.[^/.]+$/, '').toLowerCase().trim();
    return inQuadernoTitles.has(cleanName) || inQuadernoTitles.has(mat.name.toLowerCase().trim());
  };

  const filteredMaterials = useMemo(() => {
    return materials.filter((m) => {
      const matchesSearch = m.name.toLowerCase().includes(search.toLowerCase());
      if (!matchesSearch) return false;

      if (categoryFilter === 'all') return true;
      if (categoryFilter === 'audio') return m.type === 'lecture' || m.type === 'audio';
      if (categoryFilter === 'notes') return m.type === 'doc';
      if (categoryFilter === 'slide_pdf') return m.type === 'pdf' || m.type === 'slide';
      return true;
    });
  }, [materials, search, categoryFilter]);

  const handleInclude = async (mat: CourseMaterial) => {
    setProcessingId(mat.id);
    try {
      if (mat.lecture_id) {
        // È una lezione già registrata/salvata che era nascosta dal Quaderno
        await onIncludeLectureInQuaderno(mat.lecture_id);
      } else {
        // È un file caricato (PDF, Slide, Appunti .md, o InfoCorso)
        const cleanTitle = mat.name.replace(/\.[^/.]+$/, '');
        let initialNotes = mat.content;
        if (!initialNotes) {
          if (mat.type === 'pdf' || mat.type === 'slide') {
            initialNotes = `# ${cleanTitle}\n\nDocumento di studio importato da **Materiali** (${mat.type.toUpperCase()}).\n\n- File: \`${mat.name}\`\n- Dimensione: ${mat.size}\n- Indicizzato nella memoria del corso.`;
          } else {
            initialNotes = `# ${cleanTitle}\n\nAppunti importati da **Materiali**.`;
          }
        }
        await onImportMaterialAsLecture({
          title: cleanTitle,
          processed_notes: initialNotes,
          source_type: 'imported',
          date: new Date().toISOString(),
          duration_seconds: mat.duration_seconds || undefined,
        });
      }
      // Ricarica la lista per mostrare lo stato aggiornato
      await loadMaterials();
    } catch {
      setErrorMsg('Impossibile importare questo elemento nel Quaderno');
      setTimeout(() => setErrorMsg(null), 4000);
    } finally {
      setProcessingId(null);
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in select-none"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-2xl rounded-2xl liquid-glass border border-white/[0.14] shadow-liquid-lg overflow-hidden animate-slide-up flex flex-col max-h-[82vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Modal Sheet Header ── */}
        <div className="h-12 px-5 border-b border-white/[0.08] flex items-center justify-between flex-shrink-0 bg-white/[0.02]">
          <div className="flex items-center gap-2.5">
            <BookOpen size={15} strokeWidth={1.75} className="text-white/60" />
            <span className="text-xs font-semibold tracking-tight text-white/90">
              Finder Memoria · {subject.name}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="relative">
              <Search size={11} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-white/30" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filtra materiali..."
                className="pl-7 pr-3 py-1 rounded-full liquid-input text-xs text-white placeholder-white/30 w-44 focus:outline-none"
              />
            </div>
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors"
              title="Chiudi (Esc)"
            >
              <X size={13} strokeWidth={2} />
            </button>
          </div>
        </div>

        {/* ── Filter Tabs Segmented Control ── */}
        <div className="px-5 py-2.5 border-b border-white/[0.06] flex items-center justify-between gap-3">
          <SegmentedControl
            size="sm"
            layoutId="finder-category-filter"
            options={[
              { id: 'all' as FilterCategory, label: `Tutti (${materials.length})` },
              { id: 'audio' as FilterCategory, label: 'Lezioni & Audio' },
              { id: 'notes' as FilterCategory, label: 'Appunti' },
              { id: 'slide_pdf' as FilterCategory, label: 'Slide & PDF' },
            ]}
            value={categoryFilter}
            onChange={(val) => setCategoryFilter(val as FilterCategory)}
          />

          <span className="text-[11px] text-white/40 font-mono flex-shrink-0">
            {lecturesInQuaderno.length} nel Quaderno
          </span>
        </div>

        {/* Error Notification Banner */}
        {errorMsg && (
          <div className="px-5 py-2 bg-red-500/15 border-b border-red-500/25 flex items-center justify-between text-xs text-red-300">
            <span>{errorMsg}</span>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-red-400 hover:text-white"
            >
              <X size={12} />
            </button>
          </div>
        )}

        {/* ── Finder List of Materials ── */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {loading ? (
            <div className="h-48 flex items-center justify-center gap-2 text-xs text-white/40 font-mono">
              <Loader2 size={14} className="animate-spin text-white/60" />
              <span>Esplorazione memoria del corso...</span>
            </div>
          ) : filteredMaterials.length === 0 ? (
            <div className="h-48 flex flex-col items-center justify-center text-center p-6">
              <BookOpen size={20} strokeWidth={1.5} className="text-white/20 mb-2" />
              <p className="text-xs font-semibold text-white/70">Nessun materiale trovato</p>
              <p className="text-[11px] text-white/40 mt-1 max-w-xs">
                Non ci sono file in questa categoria. Puoi aggiungerne di nuovi nella sezione Materiali.
              </p>
            </div>
          ) : (
            filteredMaterials.map((mat) => {
              const inQuaderno = isMaterialInQuaderno(mat);
              const isProcessing = processingId === mat.id;

              return (
                <div
                  key={mat.id}
                  className={clsx(
                    'p-3 rounded-xl border flex items-center justify-between transition-all group',
                    inQuaderno
                      ? 'bg-white/[0.02] border-white/[0.06] opacity-80'
                      : 'bg-white/[0.05] border-white/[0.10] hover:bg-white/[0.08] hover:border-white/[0.18] shadow-liquid-sm'
                  )}
                >
                  <div className="flex items-center gap-3 min-w-0 pr-4">
                    <div className="w-9 h-9 rounded-xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center flex-shrink-0">
                      {mat.isAutoImported ? (
                        <Info size={16} strokeWidth={2} className="text-emerald-400" />
                      ) : mat.type === 'lecture' || mat.type === 'audio' ? (
                        <Mic size={16} strokeWidth={2} className="text-amber-400" />
                      ) : mat.type === 'slide' ? (
                        <Presentation size={16} strokeWidth={2} className="text-orange-400" />
                      ) : mat.type === 'pdf' ? (
                        <FileText size={16} strokeWidth={2} className="text-rose-400" />
                      ) : (
                        <File size={16} strokeWidth={2} className="text-blue-400" />
                      )}
                    </div>

                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-white/95 truncate">
                          {mat.name}
                        </span>
                        {mat.isAutoImported && (
                          <span className="text-[9px] font-mono text-emerald-300 bg-emerald-500/15 border border-emerald-500/30 px-1.5 rounded-full">
                            InfoCorso
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-white/40 font-mono mt-0.5">
                        <span>
                          {mat.type === 'lecture'
                            ? 'Registrazione Lezione'
                            : mat.type === 'pdf'
                            ? 'Documento PDF'
                            : mat.type === 'slide'
                            ? 'Slide / Presentazione'
                            : 'Appunti di testo'}
                        </span>
                        <span>·</span>
                        <span>{mat.size}</span>
                        <span>·</span>
                        <span>{mat.uploadedAt}</span>
                      </div>
                    </div>
                  </div>

                  {/* Azione destra: Includi o già presente */}
                  <div className="flex-shrink-0">
                    {inQuaderno ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-mono text-emerald-300 bg-emerald-500/15 border border-emerald-500/30">
                        <Check size={11} strokeWidth={2.5} />
                        <span>Nel Quaderno</span>
                      </span>
                    ) : (
                      <button
                        onClick={() => handleInclude(mat)}
                        disabled={isProcessing}
                        className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-white/[0.14] hover:bg-white/[0.24] text-white text-xs font-medium border border-white/[0.18] shadow-liquid-sm transition-all disabled:opacity-50"
                      >
                        {isProcessing ? (
                          <Loader2 size={11} className="animate-spin text-white" />
                        ) : (
                          <Plus size={11} strokeWidth={2.5} />
                        )}
                        <span>Includi nel Quaderno</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* ── Footer ── */}
        <div className="p-4 border-t border-white/[0.08] flex items-center justify-between flex-shrink-0 bg-white/[0.02]">
          <div className="text-[11px] text-white/40">
            {onOpenAddMaterialModal && (
              <button
                onClick={() => {
                  onClose();
                  onOpenAddMaterialModal();
                }}
                className="text-white/70 hover:text-white underline underline-offset-2 transition-colors"
              >
                + Carica o crea un nuovo file in Materiali
              </button>
            )}
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl liquid-glass-pill border border-white/[0.12] text-xs font-medium text-white hover:bg-white/[0.12] transition-colors"
          >
            Fatto
          </button>
        </div>
      </div>
    </div>
  );
}

