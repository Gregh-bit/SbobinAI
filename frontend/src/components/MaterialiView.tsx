// components/MaterialiView.tsx — Sezione Materiali: Tutta la memoria del database del corso (File + Registrazioni + InfoCorso)
import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, FileText, Presentation, File, Trash2, Plus, 
  Search, Sparkles, X, Info, Mic, RotateCcw,
  ChevronLeft, ChevronRight, Eye, Code2, ExternalLink, Loader2, AlertCircle,
  BookOpen, ChevronDown, Check
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { CourseMaterial, Lecture, MaterialPageOut, Subject } from '../types';
import MarkdownContent from './MarkdownContent';
import { liquidMenuVariants } from '../lib/animations';
import {
  useDelayedLoading,
  MaterialiGridSkeleton,
  SlidePageSkeleton,
} from './Skeleton';

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
  if (mat.isAutoImported || mat.lecture_id) return false;
  const name = mat.name.toLowerCase();
  if (name.endsWith('.md') || name.endsWith('.txt')) return false;
  return mat.type === 'pdf' || mat.type === 'slide';
}

function isSlideMaterial(mat: CourseMaterial, pageData: MaterialPageOut | null): boolean {
  const name = (pageData?.filename || mat.name).toLowerCase();
  return mat.type === 'slide' || /\.(ppt|pptx|key|odp)$/.test(name);
}

export default function MaterialiView({ 
  subject, 
  onDeleteLecturePermanently, 
  onRestoreLectureToQuaderno,
  onLectureCreated: _onLectureCreated,
  initialTarget,
  onClearInitialTarget,
}: Props) {
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [loading, setLoading] = useState(true);
  const showGridSkeleton = useDelayedLoading(loading, 150, 220);
  const subjectRef = useRef(subject);
  subjectRef.current = subject;
  const [search, setSearch] = useState('');
  const [previewMaterial, setPreviewMaterial] = useState<CourseMaterial | null>(null);
  const [viewerPage, setViewerPage] = useState<number>(1);
  const [pageInput, setPageInput] = useState<string>('1');
  const [pageData, setPageData] = useState<MaterialPageOut | null>(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'visual' | 'text'>('visual');
  const [itemToDelete, setItemToDelete] = useState<CourseMaterial | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Drag & drop e caricamento rapido
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tagMenuMaterialId, setTagMenuMaterialId] = useState<string | null>(null);

  useEffect(() => {
    const handleClickOutside = () => setTagMenuMaterialId(null);
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, []);

  const handleUploadFiles = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    setUploadingCount((prev) => prev + files.length);
    setUploadError(null);

    for (const file of files) {
      try {
        const newMat = await api.subjects.uploadMaterialFile(subject.id, file);
        // Il corso può cambiare mentre l'upload è in corso: non inserire il
        // risultato del corso precedente nella griglia attualmente visibile.
        if (subjectRef.current.id !== subject.id) continue;
        setMaterials((prev) => {
          // Mantieni InfoCorso.md sempre in cima, poi il file appena caricato subito dopo (dal più recente)
          const pinned = prev.filter((m) => m.isAutoImported);
          const others = prev.filter((m) => !m.isAutoImported && m.id !== newMat.id && m.name !== newMat.name);
          return [...pinned, newMat, ...others];
        });
      } catch (err) {
        if (subjectRef.current.id === subject.id) {
          setUploadError(err instanceof Error ? err.message : 'Errore nel caricamento del file.');
        }
      } finally {
        setUploadingCount((prev) => Math.max(0, prev - 1));
      }
    }
  }, [subject.id]);

  // Listener globali drag-and-drop compatibili con Safari, Chrome, Firefox e Tauri
  useEffect(() => {
    const isFileDrag = (dt: DataTransfer | null): boolean => {
      if (!dt) return false;
      if (dt.types) {
        const types = Array.from(dt.types);
        if (types.includes('Files') || types.includes('application/x-moz-file')) return true;
      }
      if (dt.items && dt.items.length > 0) return true;
      if (dt.files && dt.files.length > 0) return true;
      return false;
    };

    const onWinDragEnter = (e: DragEvent) => {
      if (!isFileDrag(e.dataTransfer)) return;
      e.preventDefault();
      dragCounter.current += 1;
      setIsDragging(true);
    };

    const onWinDragOver = (e: DragEvent) => {
      if (!isFileDrag(e.dataTransfer)) return;
      e.preventDefault();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
      setIsDragging(true);
    };

    const onWinDragLeave = (e: DragEvent) => {
      if (!isFileDrag(e.dataTransfer)) return;
      e.preventDefault();
      dragCounter.current = Math.max(0, dragCounter.current - 1);
      if (dragCounter.current === 0 || e.relatedTarget === null) {
        dragCounter.current = 0;
        setIsDragging(false);
      }
    };

    const onWinDrop = (e: DragEvent) => {
      if (!isFileDrag(e.dataTransfer)) return;
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);
      const dropped = e.dataTransfer?.files ? Array.from(e.dataTransfer.files) : [];
      if (dropped.length > 0) {
        void handleUploadFiles(dropped);
      }
    };

    const resetDrag = () => {
      dragCounter.current = 0;
      setIsDragging(false);
    };

    window.addEventListener('dragenter', onWinDragEnter);
    window.addEventListener('dragover', onWinDragOver);
    window.addEventListener('dragleave', onWinDragLeave);
    window.addEventListener('drop', onWinDrop);
    window.addEventListener('dragend', resetDrag);
    window.addEventListener('blur', resetDrag);
    return () => {
      window.removeEventListener('dragenter', onWinDragEnter);
      window.removeEventListener('dragover', onWinDragOver);
      window.removeEventListener('dragleave', onWinDragLeave);
      window.removeEventListener('drop', onWinDrop);
      window.removeEventListener('dragend', resetDrag);
      window.removeEventListener('blur', resetDrag);
    };
  }, [handleUploadFiles]);

  const handleSelectTag = async (mat: CourseMaterial, newTag: 'notes' | 'slide' | 'doc') => {
    setTagMenuMaterialId(null);
    const previousSourceType = mat.source_type;
    setMaterials((prev) =>
      prev.map((item) => (item.id === mat.id ? { ...item, source_type: newTag } : item))
    );
    try {
      await api.subjects.updateMaterialTag(subject.id, mat.name, newTag);
    } catch (err) {
      console.error('Errore aggiornamento tag:', err);
      if (isMountedRef.current) {
        setMaterials((prev) =>
          prev.map((item) =>
            item.id === mat.id ? { ...item, source_type: previousSourceType } : item
          )
        );
      }
    }
  };

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
      const currentSubject = subjectRef.current;
      const fallbackInfo: CourseMaterial = {
        id: `infocorso-${subject.id}`,
        name: 'InfoCorso.md',
        type: 'doc',
        size: '1.2 KB',
        uploadedAt: 'Auto-importato',
        pageCount: 1,
        content: `# Informazioni del Corso — ${currentSubject.name}\n\n## Descrizione\n${currentSubject.description || 'Nessuna descrizione specificata.'}`,
        isAutoImported: true,
      };
      setMaterials([fallbackInfo]);
    } finally {
      if (!isStale()) setLoading(false);
    }
  }, [subject.id]);

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

    // Se il file citato non esiste nei materiali del corso, pulisci il target senza aprire modali ingannevoli
    onClearInitialTarget?.();
  }, [loading, initialTarget, materials, openMaterialViewer, onClearInitialTarget, subject.id, subject.name, subject.description]);

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
      }
      if (!itemToDelete.isAutoImported) {
        try {
          await api.subjects.deleteMaterialFile(subject.id, itemToDelete.name);
        } catch {
          // Se era solo una lezione virtuale senza file fisico su disco, ignora il 404
        }
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
    <div className="flex-1 flex flex-col overflow-hidden content-canvas select-none relative">
      {/* Hidden file input for native OS picker */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.ppt,.pptx,.key,.odp,.md,.txt,.doc,.docx,.mp3,.wav,.m4a,.aac,.ogg,.flac"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            void handleUploadFiles(Array.from(e.target.files));
            e.target.value = '';
          }
        }}
      />

      {/* Drag & Drop Visual Overlay (transizione fluida GPU-accelerated) */}
      <div
        aria-hidden={!isDragging}
        className={clsx(
          'absolute inset-4 z-50 rounded-3xl bg-[#090b12]/88 backdrop-blur-xl border-2 border-dashed border-sky-400/75 flex flex-col items-center justify-center gap-3.5 p-8 pointer-events-none shadow-liquid-lg transition-all duration-200 ease-out',
          isDragging ? 'opacity-100 scale-100' : 'opacity-0 scale-[0.97]'
        )}
      >
        <div
          className={clsx(
            'w-16 h-16 rounded-2xl bg-sky-500/20 border border-sky-400/45 flex items-center justify-center text-sky-300 shadow-[0_0_32px_rgba(56,189,248,0.35)] transition-transform duration-300',
            isDragging ? 'scale-100 animate-bounce' : 'scale-90'
          )}
        >
          <Upload size={30} strokeWidth={2} />
        </div>
        <div className="text-center space-y-1">
          <h3 className="text-base font-semibold text-white tracking-tight">Rilascia qui i tuoi file</h3>
          <p className="text-xs text-white/60">
            Slide (PDF, PPTX), Appunti (.md, .txt) o Registrazioni Audio (.mp3, .wav, .m4a)
          </p>
        </div>
      </div>

      {/* Top Bar / Search, Status & Minimal Add Button */}
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
          {uploadingCount > 0 && (
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-sky-500/15 border border-sky-400/30 text-[11px] font-mono text-sky-300 animate-pulse">
              <Loader2 size={11} className="animate-spin" />
              Caricamento di {uploadingCount} file...
            </span>
          )}
          {uploadError && (
            <span className="flex items-center gap-1.5 pl-2.5 pr-1.5 py-0.5 rounded-full bg-red-500/15 border border-red-500/30 text-[11px] font-mono text-red-300">
              <AlertCircle size={11} />
              {uploadError}
              <button 
                type="button" 
                onClick={() => setUploadError(null)} 
                title="Chiudi avviso"
                aria-label="Chiudi avviso"
                className="ml-0.5 w-5 h-5 rounded-full flex items-center justify-center text-red-200/70 hover:text-white hover:bg-white/[0.10] transition-colors cursor-pointer"
              >
                <X size={13} />
              </button>
            </span>
          )}
        </div>

        <div>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full liquid-glass-pill text-xs font-semibold text-white hover:bg-white/[0.16] shadow-liquid-sm transition-all cursor-pointer"
            title="Apri il Finder o Esplora File per caricare materiale"
          >
            <Plus size={13} strokeWidth={2.5} />
            <span>Aggiungi</span>
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-8">
        {loading || showGridSkeleton ? (
          showGridSkeleton ? (
            <MaterialiGridSkeleton />
          ) : (
            <div className="min-h-[16rem]" aria-busy="true" />
          )
        ) : filtered.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-center">
            <Upload size={24} strokeWidth={1.5} className="text-white/20 mb-3" />
            <h4 className="text-sm font-medium text-white/70">Nessun elemento presente</h4>
            <p className="text-xs text-white/30 mt-1 max-w-xs">
              Trascina qui i tuoi file o clicca Aggiungi per includerli nei materiali del corso.
            </p>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="mt-3.5 px-4 py-1.5 rounded-full liquid-glass-pill text-xs font-semibold text-white hover:bg-white/[0.16] shadow-liquid-sm transition-all cursor-pointer"
            >
              + Aggiungi il primo materiale
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {filtered.map((mat) => {
              const isLecture = mat.type === 'lecture';
              const isImportedNotes = mat.source_type === 'notes' || mat.name.toLowerCase().endsWith('.md');
              const isInQuaderno = mat.in_quaderno !== false;
              const tag = mat.isAutoImported
                ? 'infocorso'
                : (mat.source_type || (mat.type === 'slide' || mat.type === 'pdf' ? 'slide' : mat.type === 'audio' ? 'audio' : isImportedNotes ? 'notes' : 'doc'));

              return (
                <div
                  key={mat.id}
                  onClick={() => openMaterialViewer(mat, 1)}
                  className={clsx(
                    'group p-4 rounded-2xl liquid-glass border cursor-pointer transition-all duration-200 flex flex-col justify-between relative',
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
                        ) : tag === 'notes' ? (
                          <BookOpen size={18} strokeWidth={1.75} className="text-sky-400" />
                        ) : tag === 'slide' ? (
                          <Presentation size={18} strokeWidth={1.75} className="text-orange-400" />
                        ) : mat.type === 'pdf' ? (
                          <FileText size={18} strokeWidth={1.75} className="text-red-400" />
                        ) : mat.type === 'audio' || tag === 'audio' ? (
                          <Mic size={18} strokeWidth={1.75} className="text-purple-400" />
                        ) : (
                          <File size={18} strokeWidth={1.75} className="text-blue-400" />
                        )}
                      </div>

                      <div className="flex items-center gap-1.5">
                        {/* Tag Pill con selettore rapido */}
                        {!mat.isAutoImported && (
                          <div className="relative">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setTagMenuMaterialId((prev) => (prev === mat.id ? null : mat.id));
                              }}
                              className={clsx(
                                'text-[10px] font-mono px-2 py-0.5 rounded-full border flex items-center gap-1 transition-all cursor-pointer',
                                tag === 'notes' && 'text-sky-300 bg-sky-500/15 border-sky-400/30 hover:bg-sky-500/25',
                                tag === 'slide' && 'text-amber-300 bg-amber-500/15 border-amber-400/30 hover:bg-amber-500/25',
                                tag === 'audio' && 'text-purple-300 bg-purple-500/15 border-purple-400/30 hover:bg-purple-500/25',
                                tag === 'doc' && 'text-white/80 bg-white/[0.08] border-white/[0.15] hover:bg-white/[0.14]'
                              )}
                              title="Modifica tipologia materiale (Appunti vs Slide vs Doc)"
                            >
                              {tag === 'notes' && <BookOpen size={9} />}
                              {tag === 'slide' && <Presentation size={9} />}
                              {tag === 'audio' && <Mic size={9} />}
                              {tag === 'doc' && <FileText size={9} />}
                              <span>{tag === 'notes' ? 'Appunti' : tag === 'slide' ? 'Slide' : tag === 'audio' ? 'Audio' : 'Doc'}</span>
                              <ChevronDown size={8} className="opacity-60" />
                            </button>

                            <AnimatePresence>
                              {tagMenuMaterialId === mat.id && (
                                <motion.div
                                  variants={liquidMenuVariants}
                                  initial="initial"
                                  animate="animate"
                                  exit="exit"
                                  style={{ transformOrigin: 'top right' }}
                                  className="absolute right-0 top-full mt-1 z-30 w-36 rounded-2xl liquid-glass border border-white/[0.18] shadow-liquid-lg p-1.5 space-y-0.5 bg-[#0d0f18]/98 backdrop-blur-2xl"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      void handleSelectTag(mat, 'notes');
                                    }}
                                    className={clsx(
                                      'w-full flex items-center justify-between px-2.5 py-1.5 rounded-xl text-left text-[11px] transition-colors cursor-pointer',
                                      tag === 'notes' ? 'bg-sky-500/20 text-sky-200 font-semibold' : 'text-white/70 hover:bg-white/[0.08] hover:text-white'
                                    )}
                                  >
                                    <div className="flex items-center gap-1.5">
                                      <BookOpen size={11} className="text-sky-400" />
                                      <span>Appunti</span>
                                    </div>
                                    {tag === 'notes' && <Check size={10} />}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      void handleSelectTag(mat, 'slide');
                                    }}
                                    className={clsx(
                                      'w-full flex items-center justify-between px-2.5 py-1.5 rounded-xl text-left text-[11px] transition-colors cursor-pointer',
                                      tag === 'slide' ? 'bg-amber-500/20 text-amber-200 font-semibold' : 'text-white/70 hover:bg-white/[0.08] hover:text-white'
                                    )}
                                  >
                                    <div className="flex items-center gap-1.5">
                                      <Presentation size={11} className="text-orange-400" />
                                      <span>Slide</span>
                                    </div>
                                    {tag === 'slide' && <Check size={10} />}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      void handleSelectTag(mat, 'doc');
                                    }}
                                    className={clsx(
                                      'w-full flex items-center justify-between px-2.5 py-1.5 rounded-xl text-left text-[11px] transition-colors cursor-pointer',
                                      tag === 'doc' ? 'bg-white/20 text-white font-semibold' : 'text-white/70 hover:bg-white/[0.08] hover:text-white'
                                    )}
                                  >
                                    <div className="flex items-center gap-1.5">
                                      <FileText size={11} className="text-white/70" />
                                      <span>Documento</span>
                                    </div>
                                    {tag === 'doc' && <Check size={10} />}
                                  </button>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        )}

                        {!mat.isAutoImported && (
                          <button
                            onClick={(e) => handleDeleteItem(mat, e)}
                            title="Elimina definitivamente da tutto"
                            className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer"
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
                          'text-[9px] font-mono px-1.5 py-0.5 rounded-full border',
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
              className="w-full max-w-4xl max-h-[88vh] flex flex-col rounded-3xl liquid-modal border border-white/[0.14] shadow-liquid-lg p-5 sm:p-7 animate-slide-up"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header — Minimale, orientato al documento, zero ridondanze */}
              <div className="flex items-center justify-between gap-4 pb-4 border-b border-white/[0.08]">
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <div className="w-10 h-10 rounded-2xl liquid-glass-subtle border border-white/[0.10] flex items-center justify-center flex-shrink-0">
                    {previewMaterial.type === 'lecture' ? (
                      <Mic size={18} className="text-amber-400" />
                    ) : previewMaterial.isAutoImported ? (
                      <Info size={18} className="text-emerald-400" />
                    ) : previewMaterial.type === 'slide' ? (
                      <Presentation size={18} className="text-orange-400" />
                    ) : previewMaterial.type === 'pdf' ? (
                      <FileText size={18} className="text-red-400" />
                    ) : (
                      <File size={18} className="text-blue-400" />
                    )}
                  </div>

                  <div className="min-w-0 flex-1">
                    <h3 className="text-base font-semibold tracking-tight text-white truncate max-w-xl">
                      {displayTitle}
                    </h3>
                    <div className="flex items-center gap-2 mt-0.5 text-[11px] font-mono text-white/40">
                      {previewMaterial.size && <span>{previewMaterial.size}</span>}
                      {previewMaterial.size && previewMaterial.uploadedAt && previewMaterial.uploadedAt !== 'Caricato' && (
                        <span>·</span>
                      )}
                      {previewMaterial.uploadedAt && previewMaterial.uploadedAt !== 'Caricato' && (
                        <span>{previewMaterial.uploadedAt}</span>
                      )}
                      {paginated && (
                        <>
                          <span>·</span>
                          <span className="text-blue-300 font-medium">
                            {unitLabel} {currentPage}{totalPages > 0 ? ` / ${totalPages}` : ''}
                          </span>
                        </>
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
                          <span>Testo</span>
                        </>
                      ) : (
                        <>
                          <Eye size={12} className="text-blue-300" />
                          <span>Visuale</span>
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
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white transition-all cursor-pointer"
                    >
                      <RotateCcw size={11} />
                      <span>Ripristina</span>
                    </button>
                  )}

                  {!previewMaterial.isAutoImported && (
                    <button
                      type="button"
                      onClick={(e) => handleDeleteItem(previewMaterial, e)}
                      title="Elimina materiale"
                      className="w-8 h-8 rounded-full flex items-center justify-center text-white/40 hover:text-red-400 hover:bg-red-500/15 border border-transparent hover:border-red-500/25 transition-all cursor-pointer"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={closePreviewModal}
                    className="w-8 h-8 rounded-full flex items-center justify-center bg-white/[0.06] hover:bg-white/[0.12] text-white/60 hover:text-white border border-white/[0.08] transition-all cursor-pointer"
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

              {/* Modal Content — Il centro è il documento */}
              <div className="flex-1 overflow-y-auto overflow-x-hidden [overscroll-behavior-x:none] py-3 pr-1 space-y-4 text-xs leading-relaxed text-white/80 min-h-[300px]">
                {(() => {
                  const isMarkdown = previewMaterial.name.toLowerCase().endsWith('.md') || previewMaterial.name.toLowerCase().endsWith('.markdown') || pageData?.kind === 'markdown';

                  if (isMarkdown) {
                    return (
                      <div className="bg-black/25 p-6 sm:p-8 rounded-2xl border border-white/[0.06] select-text shadow-inner">
                        <MarkdownContent content={previewMaterial.content || pageData?.text || ''} className="quaderno-markdown text-[14px]" />
                      </div>
                    );
                  }

                  if (paginated) {
                    if (pageLoading) {
                      return <SlidePageSkeleton />;
                    }

                    if (pageError) {
                      return (
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
                      );
                    }

                    if (hasImagePage && viewMode === 'visual') {
                      return (
                        <div className="flex flex-col items-center justify-center rounded-2xl bg-black/40 border border-white/[0.08] p-3 sm:p-4 min-h-[340px]">
                          <img
                            src={pageData!.image_data_url!}
                            alt={`${displayTitle} - ${unitLabel} ${currentPage}`}
                            className="max-w-full max-h-[64vh] w-auto h-auto object-contain rounded-xl shadow-[0_12px_40px_rgba(0,0,0,0.65)] select-text"
                          />
                        </div>
                      );
                    }

                    if (pageData?.text) {
                      return (
                        <div className="bg-black/25 p-6 sm:p-8 rounded-2xl border border-white/[0.06] select-text shadow-inner">
                          <div className="whitespace-pre-wrap font-sans text-[13.5px] leading-relaxed text-white/90">
                            {pageData.text}
                          </div>
                        </div>
                      );
                    }

                    if (previewMaterial.content) {
                      return (
                        <div className="bg-black/25 p-6 sm:p-8 rounded-2xl border border-white/[0.06] select-text shadow-inner">
                          <MarkdownContent content={previewMaterial.content} className="quaderno-markdown text-[14px]" />
                        </div>
                      );
                    }

                    return (
                      <div className="py-12 text-center text-white/40 italic">
                        Nessun contenuto estratto per questa {unitLabel.toLowerCase()}.
                      </div>
                    );
                  }

                  if (previewMaterial.content) {
                    return (
                      <div className="bg-black/25 p-6 sm:p-8 rounded-2xl border border-white/[0.06] select-text shadow-inner">
                        <MarkdownContent content={previewMaterial.content} className="quaderno-markdown text-[14px]" />
                      </div>
                    );
                  }

                  return (
                    <div className="py-12 text-center text-white/40 italic">
                      Nessun contenuto di testo memorizzato per questo file.
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        );
      })()}

      {/* ── Modal Conferma Eliminazione Materiale (Liquid Glass) ── */}
      {itemToDelete && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in"
          onClick={() => !isDeleting && setItemToDelete(null)}
        >
          <div 
            className="w-full max-w-sm rounded-3xl liquid-modal border border-white/[0.14] shadow-liquid-lg p-6 space-y-4 animate-slide-up"
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
