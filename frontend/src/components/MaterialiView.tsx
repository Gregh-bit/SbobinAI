// components/MaterialiView.tsx — Sezione Materiali: Tutta la memoria del database del corso (File + Registrazioni + InfoCorso)
import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Upload, FileText, Presentation, File, Trash2, Plus, 
  Search, Sparkles, X, Info, Mic, RotateCcw,
  ChevronLeft, ChevronRight, Eye, Code2, ExternalLink, Loader2, AlertCircle
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { CourseMaterial, Lecture, MaterialPageOut, Subject } from '../types';
import AddMaterialModal from './AddMaterialModal';
import MarkdownContent from './MarkdownContent';

export interface MaterialTarget {
  sourceId: string;
  page?: number;
  key: number;
}

interface Props {
  subject: Subject;
  onDeleteLecturePermanently?: (id: number) => Promise<void>;
  onRestoreLectureToQuaderno?: (id: number) => Promise<void>;
  onLectureCreated?: (lecture: Lecture) => void;
  initialTarget?: MaterialTarget | null;
  onClearInitialTarget?: () => void;
}

function isPaginatedFileMaterial(mat: CourseMaterial): boolean {
  return !mat.isAutoImported && !mat.lecture_id && (mat.type === 'pdf' || mat.type === 'slide' || mat.type === 'doc');
}

function isSlideMaterial(mat: CourseMaterial, pageData: MaterialPageOut | null): boolean {
  const name = (pageData?.filename || mat.name).toLowerCase();
  return mat.type === 'slide' || /\.(ppt|pptx|key|odp)$/.test(name);
}

export default function MaterialiView({ 
  subject, 
  onDeleteLecturePermanently, 
  onRestoreLectureToQuaderno,
  onLectureCreated,
  initialTarget,
  onClearInitialTarget,
}: Props) {
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [previewMaterial, setPreviewMaterial] = useState<CourseMaterial | null>(null);
  const [viewerPage, setViewerPage] = useState<number>(1);
  const [pageInput, setPageInput] = useState<string>('1');
  const [pageData, setPageData] = useState<MaterialPageOut | null>(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'visual' | 'text'>('visual');
  const [showAddModal, setShowAddModal] = useState(false);
  const [itemToDelete, setItemToDelete] = useState<CourseMaterial | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Guardia contro risposte obsolete (dismount o richieste superate)
  const isMountedRef = useRef(true);
  const loadSeqRef = useRef(0);
  const pageReqSeqRef = useRef(0);
  const lastTargetKeyRef = useRef<number | null>(null);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const closePreviewModal = useCallback(() => {
    setPreviewMaterial(null);
    setPageData(null);
    setPageError(null);
    setPageLoading(false);
    onClearInitialTarget?.();
  }, [onClearInitialTarget]);

  const openMaterialViewer = useCallback((mat: CourseMaterial, page = 1) => {
    const safePage = Math.max(1, page);
    setPreviewMaterial(mat);
    setViewerPage(safePage);
    setPageInput(String(safePage));
    setPageData(null);
    setPageError(null);
    setViewMode('visual');
  }, []);

  // Gestione tasto Escape a livelli e frecce paginazione nel viewer
  useEffect(() => {
    if (!itemToDelete && !previewMaterial) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        if (itemToDelete) {
          if (!isDeleting) setItemToDelete(null);
          return;
        }
        if (previewMaterial) {
          closePreviewModal();
          return;
        }
      }

      if (!itemToDelete && previewMaterial && isPaginatedFileMaterial(previewMaterial)) {
        const activeTag = (document.activeElement as HTMLElement | null)?.tagName;
        if (activeTag === 'INPUT' || activeTag === 'TEXTAREA') return;
        const total = pageData?.page_count ?? previewMaterial.pageCount ?? 0;
        if (e.key === 'ArrowLeft' && viewerPage > 1 && !pageLoading) {
          e.preventDefault();
          const next = viewerPage - 1;
          setViewerPage(next);
          setPageInput(String(next));
        } else if (e.key === 'ArrowRight' && (!total || viewerPage < total) && !pageLoading) {
          e.preventDefault();
          const next = viewerPage + 1;
          setViewerPage(next);
          setPageInput(String(next));
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [itemToDelete, previewMaterial, isDeleting, closePreviewModal, pageData?.page_count, viewerPage, pageLoading]);

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

  // Deep-link da citazioni dell'Assistente (o Live Workspace)
  useEffect(() => {
    if (loading || !initialTarget || lastTargetKeyRef.current === initialTarget.key) return;
    lastTargetKeyRef.current = initialTarget.key;

    const src = initialTarget.sourceId;
    const srcLower = src.toLowerCase();
    const matched = materials.find(
      (m) =>
        m.id === src ||
        m.id.toLowerCase() === srcLower ||
        m.name === src ||
        m.name.toLowerCase() === srcLower ||
        (m.lecture_id != null &&
          (String(m.lecture_id) === src ||
            `lecture_${m.lecture_id}` === srcLower ||
            `lecture-${m.lecture_id}` === srcLower))
    );

    if (matched) {
      openMaterialViewer(matched, initialTarget.page ?? 1);
      return;
    }

    const lectureMatch = /^lecture[-_]?(\d+)$/i.exec(src);
    if (lectureMatch) {
      const lecId = Number(lectureMatch[1]);
      openMaterialViewer(
        {
          id: src,
          name: `Lezione #${lecId}`,
          type: 'lecture',
          size: '',
          uploadedAt: 'Citazione',
          lecture_id: lecId,
          content: 'Questa lezione non è più presente nella memoria del corso oppure non contiene ancora appunti salvati.',
        },
        1
      );
      return;
    }

    const isInfoCorsoRef = /^infocorso(?:-\d+)?$/i.test(src) || srcLower === 'infocorso.md';
    if (isInfoCorsoRef) {
      const autoMat = materials.find((m) => m.isAutoImported);
      openMaterialViewer(
        autoMat ?? {
          id: `infocorso-${subject.id}`,
          name: 'InfoCorso.md',
          type: 'doc',
          size: '',
          uploadedAt: 'Auto-importato',
          pageCount: 1,
          isAutoImported: true,
          content: `# Informazioni del Corso — ${subject.name}\n\n## Descrizione\n${subject.description || 'Nessuna descrizione specificata.'}`,
        },
        1
      );
      return;
    }

    const targetMat: CourseMaterial = {
      id: src,
      name: src,
      type: 'doc',
      size: '',
      uploadedAt: 'Citazione',
      pageCount: 1,
      isAutoImported: true,
      content: `### Documento non più disponibile\n\nIl file **"${src}"** citato nella conversazione è stato rimosso dalla memoria del corso e non è più consultabile.`,
    };

    openMaterialViewer(targetMat, 1);
  }, [loading, initialTarget, materials, openMaterialViewer, subject.id, subject.name, subject.description]);

  // Caricamento pagina visuale / testo estratto per file paginati (PDF / Slide / Doc)
  const fetchViewerPage = useCallback(
    async (filename: string, page: number) => {
      const seq = ++pageReqSeqRef.current;
      setPageLoading(true);
      setPageError(null);
      try {
        const res = await api.subjects.getMaterialPage(subject.id, filename, page);
        if (!isMountedRef.current || seq !== pageReqSeqRef.current) return;
        setPageData(res);
        if (res.page && res.page !== page) {
          setViewerPage(res.page);
          setPageInput(String(res.page));
        }
      } catch (err) {
        if (!isMountedRef.current || seq !== pageReqSeqRef.current) return;
        setPageData(null);
        setPageError(err instanceof Error ? err.message : 'Impossibile caricare questa pagina.');
      } finally {
        if (isMountedRef.current && seq === pageReqSeqRef.current) {
          setPageLoading(false);
        }
      }
    },
    [subject.id]
  );

  useEffect(() => {
    if (!previewMaterial || !isPaginatedFileMaterial(previewMaterial)) {
      setPageData(null);
      setPageError(null);
      setPageLoading(false);
      return;
    }
    void fetchViewerPage(previewMaterial.name, viewerPage);
  }, [previewMaterial, viewerPage, fetchViewerPage]);

  const goToPage = (targetPage: number) => {
    if (!previewMaterial) return;
    const total = pageData?.page_count ?? previewMaterial.pageCount ?? 0;
    const clamped = total > 0 ? Math.min(Math.max(1, targetPage), total) : Math.max(1, targetPage);
    setViewerPage(clamped);
    setPageInput(String(clamped));
  };

  const handlePageInputSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const parsed = parseInt(pageInput, 10);
    if (Number.isNaN(parsed)) {
      setPageInput(String(viewerPage));
      return;
    }
    goToPage(parsed);
  };

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
      if (previewMaterial?.id === itemToDelete.id) closePreviewModal();
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
                  onClick={() => openMaterialViewer(mat, 1)}
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

      {/* ── Visual Slide / PDF / Document Page Viewer Modal (Liquid Glass) ── */}
      {previewMaterial && (() => {
        const paginated = isPaginatedFileMaterial(previewMaterial);
        const totalPages = pageData?.page_count ?? previewMaterial.pageCount ?? 0;
        const currentPage = pageData?.page ?? viewerPage;
        const unitLabel = isSlideMaterial(previewMaterial, pageData) ? 'Slide' : 'Pagina';
        const hasImagePage = Boolean(pageData?.kind === 'image' && pageData?.image_data_url);
        const displayTitle = (pageData?.title && pageData.title.trim()) || previewMaterial.name;

        return (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/70 backdrop-blur-md animate-fade-in"
            onClick={closePreviewModal}
          >
            <div
              className={clsx(
                'w-full flex flex-col rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg p-5 sm:p-6 animate-slide-up',
                paginated ? 'max-w-4xl max-h-[90vh]' : 'max-w-2xl max-h-[82vh]'
              )}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-white/[0.08]">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <div className="w-9 h-9 rounded-xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center flex-shrink-0">
                    {previewMaterial.type === 'lecture' ? (
                      <Mic size={16} className="text-amber-400" />
                    ) : previewMaterial.isAutoImported ? (
                      <Info size={16} className="text-emerald-400" />
                    ) : previewMaterial.type === 'slide' ? (
                      <Presentation size={16} className="text-orange-400" />
                    ) : previewMaterial.type === 'pdf' ? (
                      <FileText size={16} className="text-red-400" />
                    ) : (
                      <File size={16} className="text-blue-400" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="text-sm font-semibold tracking-tight text-white truncate max-w-md">
                        {displayTitle}
                      </h3>
                      <span className="px-2 py-0.5 rounded-full bg-white/[0.06] border border-white/[0.10] text-[10px] font-mono text-white/60 truncate max-w-[200px]">
                        {pageData?.filename || previewMaterial.name}
                      </span>
                      {paginated && (
                        <span className="px-2 py-0.5 rounded-full bg-blue-500/15 border border-blue-400/30 text-[10px] font-mono text-blue-200 flex-shrink-0">
                          {unitLabel} {currentPage}{totalPages > 0 ? ` / ${totalPages}` : ''}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 mt-1 text-[10px] font-mono text-white/40">
                      {previewMaterial.size && <span>{previewMaterial.size}</span>}
                      {previewMaterial.size && previewMaterial.uploadedAt && <span>·</span>}
                      {previewMaterial.uploadedAt && <span>{previewMaterial.uploadedAt}</span>}
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

                <div className="flex items-center gap-2 flex-shrink-0">
                  {hasImagePage && (
                    <button
                      type="button"
                      onClick={() => setViewMode((m) => (m === 'visual' ? 'text' : 'visual'))}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-[11px] font-medium text-white transition-all cursor-pointer"
                    >
                      {viewMode === 'visual' ? (
                        <>
                          <Code2 size={12} className="text-white/70" />
                          <span>Mostra testo estratto</span>
                        </>
                      ) : (
                        <>
                          <Eye size={12} className="text-blue-300" />
                          <span>Mostra pagina visuale</span>
                        </>
                      )}
                    </button>
                  )}

                  {paginated && (
                    <a
                      href={api.subjects.getMaterialFileUrl(subject.id, previewMaterial.name)}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Apri file originale"
                      className="flex items-center gap-1 px-2.5 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.10] text-[11px] font-medium text-white/80 hover:text-white transition-all"
                    >
                      <ExternalLink size={11} />
                      <span className="hidden sm:inline">Apri file</span>
                    </a>
                  )}

                  {previewMaterial.lecture_id && previewMaterial.in_quaderno === false && (
                    <button
                      type="button"
                      onClick={() => handleRestore(previewMaterial)}
                      className="flex items-center gap-1.5 px-3 py-1 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white transition-all cursor-pointer"
                    >
                      <RotateCcw size={11} />
                      <span>Ripristina nel Quaderno</span>
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={closePreviewModal}
                    className="w-7 h-7 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
                    title="Chiudi (Esc)"
                  >
                    <X size={14} strokeWidth={2} />
                  </button>
                </div>
              </div>

              {/* Pagination Bar (solo per file paginati) */}
              {paginated && (
                <div className="py-2.5 px-3 mt-3 rounded-2xl bg-white/[0.03] border border-white/[0.07] flex items-center justify-between gap-3">
                  <button
                    type="button"
                    onClick={() => goToPage(viewerPage - 1)}
                    disabled={viewerPage <= 1 || pageLoading}
                    className="flex items-center gap-1 px-3 py-1 rounded-full bg-white/[0.06] hover:bg-white/[0.12] disabled:opacity-35 disabled:cursor-not-allowed text-[11px] font-medium text-white transition-all cursor-pointer"
                  >
                    <ChevronLeft size={13} />
                    <span>Precedente</span>
                  </button>

                  <form onSubmit={handlePageInputSubmit} className="flex items-center gap-2 text-[11px] font-mono text-white/60">
                    <span>{unitLabel}</span>
                    <input
                      type="number"
                      min={1}
                      max={totalPages > 0 ? totalPages : undefined}
                      value={pageInput}
                      onChange={(e) => setPageInput(e.target.value)}
                      onBlur={() => handlePageInputSubmit()}
                      aria-label="Vai alla pagina"
                      className="w-14 px-2 py-1 rounded-lg bg-black/40 border border-white/[0.14] text-center text-white text-xs font-mono focus:outline-none focus:border-white/35 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    {totalPages > 0 && <span className="text-white/40">/ {totalPages}</span>}
                  </form>

                  <button
                    type="button"
                    onClick={() => goToPage(viewerPage + 1)}
                    disabled={(totalPages > 0 && viewerPage >= totalPages) || pageLoading}
                    className="flex items-center gap-1 px-3 py-1 rounded-full bg-white/[0.06] hover:bg-white/[0.12] disabled:opacity-35 disabled:cursor-not-allowed text-[11px] font-medium text-white transition-all cursor-pointer"
                  >
                    <span>Successiva</span>
                    <ChevronRight size={13} />
                  </button>
                </div>
              )}

              {/* Modal Content */}
              <div className="flex-1 overflow-y-auto py-4 pr-1 space-y-4 text-xs leading-relaxed text-white/80 min-h-[260px]">
                {paginated ? (
                  pageLoading ? (
                    <div className="h-full min-h-[320px] rounded-2xl bg-white/[0.02] border border-white/[0.06] flex flex-col items-center justify-center gap-3 p-8">
                      <Loader2 size={24} className="animate-spin text-white/50" />
                      <p className="text-xs text-white/50 font-mono">
                        Caricamento {unitLabel.toLowerCase()} {viewerPage}...
                      </p>
                      <div className="w-full max-w-md space-y-2.5 pt-2 opacity-30">
                        <div className="h-3 bg-white/20 rounded-full w-4/5 mx-auto animate-pulse" />
                        <div className="h-36 bg-white/10 rounded-2xl w-full animate-pulse" />
                        <div className="h-3 bg-white/15 rounded-full w-2/3 mx-auto animate-pulse" />
                      </div>
                    </div>
                  ) : pageError ? (
                    <div className="rounded-2xl bg-red-500/[0.08] border border-red-500/25 p-6 text-center space-y-3">
                      <AlertCircle size={26} className="text-red-400 mx-auto" />
                      <div className="space-y-1">
                        <h4 className="text-xs font-semibold text-red-200">
                          Impossibile caricare la {unitLabel.toLowerCase()} {viewerPage}
                        </h4>
                        <p className="text-[11px] text-red-300/80 max-w-md mx-auto">
                          {pageError}
                        </p>
                      </div>
                      <div className="flex items-center justify-center gap-2 pt-1">
                        <button
                          type="button"
                          onClick={() => void fetchViewerPage(previewMaterial.name, viewerPage)}
                          className="px-3.5 py-1.5 rounded-full bg-white/[0.10] hover:bg-white/[0.16] border border-white/[0.16] text-[11px] font-medium text-white transition-all cursor-pointer"
                        >
                          Riprova
                        </button>
                        <a
                          href={api.subjects.getMaterialFileUrl(subject.id, previewMaterial.name)}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3.5 py-1.5 rounded-full bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.10] text-[11px] font-medium text-white/80 transition-all"
                        >
                          Apri file originale
                        </a>
                      </div>
                      {previewMaterial.content && (
                        <div className="mt-4 pt-4 border-t border-white/[0.08] text-left">
                          <MarkdownContent content={previewMaterial.content} className="quaderno-markdown" />
                        </div>
                      )}
                    </div>
                  ) : hasImagePage && viewMode === 'visual' ? (
                    <div className="flex flex-col items-center justify-center rounded-2xl bg-black/40 border border-white/[0.08] p-3 sm:p-4 min-h-[340px]">
                      <img
                        src={pageData!.image_data_url!}
                        alt={`${displayTitle} - ${unitLabel} ${currentPage}`}
                        className="max-w-full max-h-[60vh] w-auto h-auto object-contain rounded-xl shadow-[0_12px_40px_rgba(0,0,0,0.65)] select-text"
                      />
                    </div>
                  ) : pageData?.text ? (
                    <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/[0.06] select-text">
                      <div className="whitespace-pre-wrap font-sans text-[13px] leading-relaxed text-white/85">
                        {pageData.text}
                      </div>
                    </div>
                  ) : previewMaterial.content ? (
                    <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/[0.06] select-text">
                      <MarkdownContent content={previewMaterial.content} className="quaderno-markdown" />
                    </div>
                  ) : (
                    <div className="py-12 text-center text-white/40 italic">
                      Nessun contenuto estratto per questa {unitLabel.toLowerCase()}.
                    </div>
                  )
                ) : previewMaterial.content ? (
                  <div className="bg-white/[0.02] p-5 rounded-2xl border border-white/[0.06] select-text">
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
                    type="button"
                    onClick={(e) => handleDeleteItem(previewMaterial, e)}
                    className="flex items-center gap-1 px-3 py-1 rounded-full text-red-400 hover:bg-red-500/10 transition-colors cursor-pointer"
                  >
                    <Trash2 size={12} />
                    <span>Elimina definitivamente da tutto</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      })()}

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
