import { useState, useMemo, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { 
  Calendar, Clock, FileText, Trash2, Sparkles, 
  ChevronRight, ChevronLeft, BookOpen, AlignLeft, Plus, Pencil, 
  Check, X, Search, HelpCircle, Square, Loader2,
  PanelLeftClose, PanelLeftOpen, AlertTriangle,
  Play, Pause, RotateCcw, Bookmark as BookmarkIcon,
  FileDown, Music, Share2, Settings, Mic, Presentation,
  FolderInput, GripVertical
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import type { Lecture, Subject, DetectedQuestion, Bookmark, KaraokeLine, CourseMaterial } from '../types';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import { saveTextFile } from '../lib/tauri';
import SegmentedControl from './SegmentedControl';
import MarkdownContent from './MarkdownContent';
import { AppleLyricsView } from './AppleLyricsView';
import { ChapterGenerationSkeleton } from './Skeleton';
import {
  liquidMenuVariants,
  liquidLayoutTransition,
  liquidPopoverContainerClass,
} from '../lib/animations';

function normalizeKaraokeLines(raw: any): KaraokeLine[] {
  if (!raw) return [];
  const list = Array.isArray(raw) ? raw : (raw.lines || raw.segments || []);
  if (!Array.isArray(list)) return [];

  // Rileva se i timestamp sono già in millisecondi analizzando la durata di parole o segmenti
  let isMs = false;
  for (const item of list) {
    if (!item) continue;
    const words = item.words || [];
    for (const w of words) {
      const dur = parseFloat(w.end ?? 0) - parseFloat(w.start ?? 0);
      if (dur >= 15) {
        isMs = true;
        break;
      }
    }
    if (isMs) break;
    const segDur = parseFloat(item.lineEnd ?? item.end ?? 0) - parseFloat(item.lineStart ?? item.start ?? 0);
    if (segDur >= 30) {
      isMs = true;
      break;
    }
  }

  const mult = isMs ? 1 : 1000;

  return list.map((item: any) => {
    const rawStart = parseFloat(item.lineStart ?? item.start ?? 0);
    const rawEnd = parseFloat(item.lineEnd ?? item.end ?? 0);

    const wordsSrc = item.words || [];
    const words = wordsSrc.map((w: any) => {
      const wText = String(w.text || w.word || '').trim();
      const wStart = parseFloat(w.start ?? 0) * mult;
      const wEnd = parseFloat(w.end ?? 0) * mult;
      return {
        text: wText,
        start: Math.round(wStart),
        end: Math.round(wEnd),
      };
    }).filter((w: any) => w.text.length > 0);

    const lineStart = Math.round(rawStart * mult);
    const lineEnd = Math.round(rawEnd * mult);

    return {
      lineStart: words.length > 0 ? Math.min(lineStart, words[0].start) : lineStart,
      lineEnd: words.length > 0 ? Math.max(lineEnd, words[words.length - 1].end) : lineEnd,
      words,
    };
  }).filter((line: any) => line.words.length > 0);
}

interface Props {
  isActive?: boolean;
  subject: Subject;
  allSubjects?: Subject[];
  lectures: Lecture[];
  onMoveLecture?: (lectureId: number, targetSubjectId: number) => Promise<void>;
  onDeletePermanently: (id: number) => Promise<void>;
  onRemoveFromQuadernoOnly: (id: number) => Promise<void>;
  onRestoreToQuaderno: (id: number) => Promise<void>;
  onAddLecture: (data: {
    title: string;
    date?: string;
    processed_notes?: string;
    raw_transcript?: string;
    duration_seconds?: number;
    source_type?: string;
    audio_path?: string;
  }) => Promise<Lecture>;
  onUpdateLecture: (
    id: number,
    data: Partial<{
      title: string;
      processed_notes: string;
      raw_transcript: string;
      date: string;
      detected_questions: string | null;
    }>
  ) => Promise<Lecture>;
  onOpenAddMaterialModal?: () => void;
  onOpenSettings?: () => void;
  onStartLive?: () => void;
  initialLectureId?: number | null;
  onClearInitialLectureId?: () => void;
  initialAudioTarget?: {
    lectureId: number;
    seconds: number;
    autoPlay?: boolean;
    key: number;
  } | null;
  onClearInitialAudioTarget?: () => void;
  onOpenAudioSnippet?: (lectureId: number, seconds: number, autoPlay?: boolean) => void;
  onOpenCitation?: (sourceId: string, page?: number) => void;
  /** Aggiorna solo lo stato locale di una lezione (il backend ha già persistito la modifica) */
  onLocalLectureUpdate?: (id: number, updater: (lecture: Lecture) => Lecture) => void;
}

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('it-IT', {
      day: 'numeric', month: 'long', year: 'numeric'
    });
  } catch {
    return dateStr;
  }
}

function formatDuration(seconds: number | null): string {
  if (!seconds || seconds <= 0) return '';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0 && m > 0) return `${h}h ${m}m`;
  if (h > 0) return `${h}h`;
  if (m > 0) return `${m}m`;
  return `${s}s`;
}

function formatSeconds(sec: number): string {
  if (isNaN(sec) || sec < 0) return '00:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}


export default function QuadernoView({
  isActive = true,
  subject, 
  allSubjects,
  lectures, 
  initialLectureId,
  onClearInitialLectureId,
  initialAudioTarget,
  onClearInitialAudioTarget,
  onOpenAudioSnippet,
  onOpenCitation,
  onMoveLecture,
  onDeletePermanently,
  onRemoveFromQuadernoOnly,
  onRestoreToQuaderno,
  onAddLecture, 
  onUpdateLecture,
  onStartLive,
  onOpenAddMaterialModal,
  onOpenSettings,
  onLocalLectureUpdate,
}: Props) {
  const { settings } = useSettings();
  const [selectedLectureId, setSelectedLectureId] = useState<number | null>(
    lectures.length > 0 ? lectures[0].id : null
  );
  const [activeTab, setActiveTab] = useState<'notes' | 'transcript' | 'questions'>('notes');
  const [search, setSearch] = useState('');
  const [sortOrder] = useState<'newest' | 'oldest'>('newest');

  // Second sidebar collapsible state
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  // Menu Sposta in altro corso
  const [showMoveMenu, setShowMoveMenu] = useState(false);
  const [isMovingLecture, setIsMovingLecture] = useState(false);
  const [draggingLectureId, setDraggingLectureId] = useState<number | null>(null);
  const moveMenuRef = useRef<HTMLDivElement | null>(null);

  // Chiudi il menu sposta se si clicca all'esterno
  useEffect(() => {
    if (!showMoveMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (moveMenuRef.current && !moveMenuRef.current.contains(e.target as Node)) {
        setShowMoveMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showMoveMenu]);

  // Audio Player & Bookmarks State
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Export State
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [exportNotice, setExportNotice] = useState<string | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);

  // Chiudi il menu esporta se si clicca altrove nello schermo
  useEffect(() => {
    if (!showExportMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showExportMenu]);

  // ── Sincronizzazione Trascrizione Stile Apple Music ─────────────────────
  const [karaokeLines, setKaraokeLines] = useState<KaraokeLine[] | null>(null);
  const [loadingKaraoke, setLoadingKaraoke] = useState(false);
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const loadedKaraokeKeyRef = useRef<string | null>(null);

  // Pausa audio e chiudi popover quando si cambia scheda fuori dal Quaderno
  useEffect(() => {
    if (isActive) return;
    setShowAddMenu(false);
    setShowMoveMenu(false);
    setShowExportMenu(false);
    setShowDeleteModal(false);
    if (audioRef.current && !audioRef.current.paused) {
      try {
        audioRef.current.pause();
      } catch {}
      setIsPlaying(false);
    }
  }, [isActive]);

  // State per generazione risposta domanda specifica con IA
  const [answeringQuestionId, setAnsweringQuestionId] = useState<string | null>(null);
  const [questionAnswerError, setQuestionAnswerError] = useState<string | null>(null);

  // Menu Eliminazione (bolla Liquid Glass identica a Esporta)
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const deleteMenuRef = useRef<HTMLDivElement | null>(null);

  // Chiudi il menu elimina se si clicca altrove nello schermo
  useEffect(() => {
    if (!showDeleteModal) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (deleteMenuRef.current && !deleteMenuRef.current.contains(e.target as Node)) {
        setShowDeleteModal(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showDeleteModal]);

  // ── Menu Aggiungi (+) nel Quaderno: Liquid Glass Popover ──
  const [showAddMenu, setShowAddMenu] = useState(false);
  const [addMenuMode, setAddMenuMode] = useState<'main' | 'materials'>('main');
  const [materials, setMaterials] = useState<CourseMaterial[]>([]);
  const [isCreatingChapter, setIsCreatingChapter] = useState(false);
  const [importingMaterialId, setImportingMaterialId] = useState<string | null>(null);
  const [menuHeight, setMenuHeight] = useState<number | undefined>(undefined);
  const addMenuRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);

  // Misurazione continua dell'altezza per animazione fluida di espansione/ridimensionamento
  useLayoutEffect(() => {
    if (!showAddMenu) {
      setMenuHeight(undefined);
      return;
    }
    const el = contentRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    if (rect.height > 0) {
      setMenuHeight(Math.round(rect.height));
    }

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const h = entry.borderBoxSize?.[0]?.blockSize ?? entry.contentRect.height;
        if (h > 0) {
          setMenuHeight(Math.round(h));
        }
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [showAddMenu, addMenuMode, materials.length]);

  // Chiudi il menu aggiungi se si clicca altrove nello schermo
  useEffect(() => {
    if (!showAddMenu) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) {
        setShowAddMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showAddMenu]);

  // Caricamento materiali del corso per subview del popover
  const loadMaterials = useCallback(async () => {
    try {
      const data = await api.subjects.getMaterials(subject.id);
      setMaterials(data);
    } catch {
      setMaterials([]);
    }
  }, [subject.id]);

  useEffect(() => {
    if (showAddMenu) {
      loadMaterials();
    } else {
      setAddMenuMode('main');
    }
  }, [showAddMenu, loadMaterials]);

  // Materiali disponibili (non ancora presenti nel Quaderno)
  const inQuadernoLectureIds = useMemo(() => {
    return new Set(lectures.map((l) => l.id));
  }, [lectures]);

  const inQuadernoTitles = useMemo(() => {
    return new Set(lectures.map((l) => l.title.toLowerCase().trim()));
  }, [lectures]);

  const isMaterialInQuaderno = useCallback((mat: CourseMaterial): boolean => {
    if (mat.lecture_id) {
      return inQuadernoLectureIds.has(mat.lecture_id);
    }
    const cleanName = mat.name.replace(/\.[^/.]+$/, '').toLowerCase().trim();
    return inQuadernoTitles.has(cleanName) || inQuadernoTitles.has(mat.name.toLowerCase().trim());
  }, [inQuadernoLectureIds, inQuadernoTitles]);

  const availableMaterials = useMemo(() => {
    return materials.filter((m) => !isMaterialInQuaderno(m));
  }, [materials, isMaterialInQuaderno]);

  const handleCreateEmptyChapter = async () => {
    setIsCreatingChapter(true);
    try {
      const created = await onAddLecture({
        title: 'Nuovo capitolo',
        processed_notes: '',
        raw_transcript: '',
        source_type: 'custom',
        date: new Date().toISOString(),
      });
      setSelectedLectureId(created.id);
      setEditTitle(created.title || 'Nuovo capitolo');
      setEditNotes('');
      setEditTab('editor');
      setIsEditing(true);
      setShowAddMenu(false);
    } catch (err) {
      console.error('Errore creazione capitolo:', err);
    } finally {
      setIsCreatingChapter(false);
    }
  };

  const handleImportMaterial = async (mat: CourseMaterial) => {
    setImportingMaterialId(mat.id);
    try {
      if (mat.lecture_id) {
        pendingRestoreIdRef.current = mat.lecture_id;
        await onRestoreToQuaderno(mat.lecture_id);
        setSelectedLectureId(mat.lecture_id);
      } else {
        const cleanTitle = mat.name.replace(/\.[^/.]+$/, '');
        let initialNotes = mat.content;
        if (!initialNotes || initialNotes.trim().length === 0) {
          try {
            const freshList = await api.subjects.getMaterials(subject.id);
            const freshMat = freshList.find((m) => m.name === mat.name || m.id === mat.id);
            if (freshMat?.content && freshMat.content.trim().length > 0) {
              initialNotes = freshMat.content;
            }
          } catch (e) {
            console.warn('[QuadernoView] Impossibile recuperare testo aggiornato:', e);
          }
        }

        if (!initialNotes || initialNotes.trim().length === 0) {
          if (mat.type === 'pdf' || mat.type === 'slide') {
            initialNotes = `# ${cleanTitle}\n\nDocumento importato da **Materiali** (${mat.type.toUpperCase()}).\n\n- File: \`${mat.name}\`\n- Dimensione: ${mat.size}\n- Indicizzato nella memoria del corso.`;
          } else {
            initialNotes = `# ${cleanTitle}\n\nAppunti importati da **Materiali**.`;
          }
        }
        const created = await onAddLecture({
          title: cleanTitle,
          processed_notes: initialNotes,
          audio_path: mat.name,
          source_type: 'imported',
          date: new Date().toISOString(),
          duration_seconds: mat.duration_seconds || undefined,
        });
        setMaterials((prev) =>
          prev.map((m) => (m.id === mat.id || m.name === mat.name ? { ...m, lecture_id: created.id, in_quaderno: true } : m))
        );
        setSelectedLectureId(created.id);
      }
      setShowAddMenu(false);
    } catch (err) {
      console.error('Errore importazione materiale:', err);
    } finally {
      setImportingMaterialId(null);
    }
  };

  // ── Gestione Tasto ESC per Sottofinestre (Menu Esportazione, Aggiungi e Eliminazione) ──
  useEffect(() => {
    if (!showDeleteModal && !showExportMenu && !showAddMenu) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.stopPropagation();
      if (showAddMenu) {
        setShowAddMenu(false);
        return;
      }
      if (showDeleteModal) {
        setShowDeleteModal(false);
        return;
      }
      if (showExportMenu) {
        setShowExportMenu(false);
        return;
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [showDeleteModal, showExportMenu, showAddMenu]);

  // Edit Mode per lezione selezionata
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [editTab, setEditTab] = useState<'editor' | 'preview'>('editor');
  const [isSavingEdit, setIsSavingEdit] = useState(false);
  const [generatingLectureId, setGeneratingLectureId] = useState<number | null>(null);
  const [summaryProgress, setSummaryProgress] = useState<{ stage: string; message: string; progress: number } | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [missingLlmNotice, setMissingLlmNotice] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const initialNotesRef = useRef<string | null>(null);
  const pendingRestoreIdRef = useRef<number | null>(null);

  const hasLLM = Boolean(settings.llmDownloaded);

  useEffect(() => {
    if (hasLLM) {
      setMissingLlmNotice(false);
    }
  }, [hasLLM]);

  useEffect(() => {
    setMissingLlmNotice(false);
  }, [selectedLectureId]);

  // Mantiene selectedLectureId sincronizzato a un ID valido in `lectures`
  // così la digitazione nella barra di ricerca non cambia `selectedLecture` né ri-scarica segnalibri/timestamp
  useEffect(() => {
    if (pendingRestoreIdRef.current !== null) {
      const targetId = pendingRestoreIdRef.current;
      if (lectures.some((l) => l.id === targetId)) {
        pendingRestoreIdRef.current = null;
        if (selectedLectureId !== targetId) {
          setSelectedLectureId(targetId);
        }
        return;
      }
      if (selectedLectureId === targetId) {
        return;
      }
    }
    if (lectures.length === 0) {
      if (selectedLectureId !== null) {
        setSelectedLectureId(null);
      }
      return;
    }
    const exists = selectedLectureId !== null && lectures.some((l) => l.id === selectedLectureId);
    if (!exists) {
      setSelectedLectureId(lectures[0].id);
    }
  }, [lectures, selectedLectureId]);

  useEffect(() => {
    if (initialLectureId && lectures.some((l) => l.id === initialLectureId)) {
      setSelectedLectureId(initialLectureId);
      onClearInitialLectureId?.();
    }
  }, [initialLectureId, lectures, onClearInitialLectureId]);

  const isPlaceholderAnswer = (ans?: string) => {
    if (!ans || !ans.trim()) return true;
    const lower = ans.toLowerCase();
    return (
      lower.includes('modello llm non ancora scaricato') ||
      lower.includes('modello llm non installato') ||
      lower.includes('non ancora scaricato') ||
      lower.includes('domanda rilevata dal docente') ||
      lower.includes('domanda rilevata') ||
      lower.includes('risposta non disponibile') ||
      lower.includes('generazione risposta flash') ||
      lower.includes('parlato del docente')
    );
  };

  const formatQuestionAnswer = (raw?: string): string => {
    if (!raw) return '';
    const trimmed = raw.trim();
    if (
      trimmed.includes('modello LLM non ancora scaricato') ||
      trimmed.includes('Risposta non disponibile') ||
      trimmed.includes('modello LLM non installato') ||
      trimmed.includes('Scarica un modello LLM dalle Impostazioni')
    ) {
      return 'Scarica un modello LLM dalle [[action:settings|Impostazioni]] per attivare le risposte rapide.';
    }
    return raw;
  };

  const handleAnswerQuestion = async (questionId: string) => {
    if (!selectedLecture) return;
    const lectureId = selectedLecture.id;
    setAnsweringQuestionId(questionId);
    setQuestionAnswerError(null);
    try {
      const res = await api.lectures.answerQuestion(lectureId, questionId);
      if (res?.question && onLocalLectureUpdate) {
        // Il backend ha già salvato la risposta: aggiorniamo solo lo stato locale,
        // partendo dallo stato più recente della lezione (nessun PATCH da stato obsoleto)
        onLocalLectureUpdate(lectureId, (lecture) => {
          let list: DetectedQuestion[] = [];
          try {
            const parsed = lecture.detected_questions ? JSON.parse(lecture.detected_questions) : [];
            list = Array.isArray(parsed) ? parsed : [];
          } catch {
            list = [];
          }
          const merged = list.map((q, idx) =>
            (q.id ? String(q.id) === questionId : String(idx) === questionId)
              ? { ...q, ...res.question, suggestedAnswer: res.answer || res.question.suggestedAnswer }
              : q
          );
          return { ...lecture, detected_questions: JSON.stringify(merged) };
        });
      }
    } catch (err: any) {
      console.error('Errore risposta domanda con IA:', err);
      const rawMsg = String(err?.message || '').trim();
      if (rawMsg.includes('Failed to fetch') || rawMsg.includes('NetworkError') || rawMsg.includes('Load failed')) {
        setQuestionAnswerError('Il motore di intelligenza artificiale locale non risponde. Verifica nelle Impostazioni che il modello sia attivo e riprova.');
      } else {
        setQuestionAnswerError(rawMsg || 'Impossibile generare la risposta alla domanda con l\'IA.');
      }
    } finally {
      setAnsweringQuestionId(null);
    }
  };

  const handleGenerateSummary = async () => {
    if (!selectedLecture) return;
    if (!hasLLM) {
      setSummaryError(null);
      if (
        !selectedLecture.processed_notes &&
        Boolean(selectedLecture.raw_transcript?.trim()) &&
        onOpenSettings
      ) {
        setMissingLlmNotice(false);
        onOpenSettings();
        return;
      }
      setMissingLlmNotice(true);
      if (activeTab !== 'notes') {
        setActiveTab('notes');
      }
      return;
    }
    setMissingLlmNotice(false);
    if (activeTab !== 'notes') {
      setActiveTab('notes');
    }
    const targetLectureId = selectedLecture.id;
    initialNotesRef.current = selectedLecture.processed_notes || null;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setGeneratingLectureId(targetLectureId);
    setSummaryProgress({ stage: 'start', message: 'Inizializzazione motore IA e analisi trascrizione...', progress: 5 });
    setSummaryError(null);
    try {
      const updated = await api.lectures.generateSummaryStream(
        targetLectureId,
        (ev) => {
          if (ev.message || typeof ev.progress === 'number') {
            setSummaryProgress({
              stage: ev.stage || 'progress',
              message: ev.message || 'Elaborazione capitolo in corso...',
              progress: typeof ev.progress === 'number' ? ev.progress : 40,
            });
          }
        },
        { signal: controller.signal }
      );
      if (updated?.status === 'cancelled') {
        return;
      }
      if (updated?.processed_notes) {
        // Il backend ha già salvato processed_notes su SQLite e indicizzato LanceDB: aggiorna solo lo stato locale
        if (onLocalLectureUpdate) {
          onLocalLectureUpdate(targetLectureId, (lec) => ({
            ...lec,
            processed_notes: updated.processed_notes,
          }));
        } else {
          await onUpdateLecture(targetLectureId, {
            processed_notes: updated.processed_notes,
          });
        }
      }
    } catch (err: any) {
      const isAborted =
        err?.name === 'AbortError' ||
        controller.signal.aborted ||
        abortControllerRef.current === null ||
        String(err?.message || '').toLowerCase().includes('abort');

      if (isAborted) {
        return;
      }

      const rawMsg = String(err?.message || '');
      if (rawMsg.includes('Failed to fetch') || rawMsg.includes('NetworkError') || rawMsg.includes('Load failed')) {
        setSummaryError('Il motore di intelligenza artificiale locale non risponde. Verifica nelle Impostazioni che il modello sia installato e riprova.');
      } else {
        setSummaryError(rawMsg || 'Errore durante la generazione della sintesi in locale');
      }
    } finally {
      setGeneratingLectureId((curr) => (curr === targetLectureId ? null : curr));
      setSummaryProgress(null);
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
    }
  };

  const handleCancelSummary = async (e?: React.MouseEvent) => {
    e?.preventDefault();
    e?.stopPropagation();
    setGeneratingLectureId(null);
    setSummaryProgress(null);
    setSummaryError(null);
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {
        // ignore
      }
      abortControllerRef.current = null;
    }
    try {
      await api.ai.stop(true);
    } catch (err: any) {
      console.warn('Errore arresto IA:', err);
    }
  };

  // Filtra e ordina le lezioni
  const filteredLectures = useMemo(() => {
    return lectures
      .filter((l) => l.title.toLowerCase().includes(search.toLowerCase()) || (l.processed_notes || '').toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => {
        const timeA = new Date(a.date).getTime();
        const timeB = new Date(b.date).getTime();
        return sortOrder === 'newest' ? timeB - timeA : timeA - timeB;
      });
  }, [lectures, search, sortOrder]);

  const selectedLecture = useMemo(
    () => lectures.find((l) => l.id === selectedLectureId) ?? lectures[0] ?? null,
    [lectures, selectedLectureId]
  );

  // Indica se la sintesi è in corso ESCLUSIVAMENTE per la lezione attualmente visualizzata
  const isGeneratingSummary = Boolean(selectedLecture && generatingLectureId === selectedLecture.id);

  // Se la sintesi è stata completata in background (arriva tramite polling o dal backend), sblocca l'interfaccia
  useEffect(() => {
    if (generatingLectureId && selectedLecture?.id === generatingLectureId && selectedLecture?.processed_notes && selectedLecture.processed_notes.trim().length > 0) {
      if (!initialNotesRef.current || selectedLecture.processed_notes !== initialNotesRef.current) {
        setGeneratingLectureId(null);
        setSummaryProgress(null);
      }
    }
  }, [generatingLectureId, selectedLecture?.id, selectedLecture?.processed_notes]);

  // Domande del prof salvate per questa specifica lezione
  const parsedQuestions = useMemo<DetectedQuestion[]>(() => {
    if (!selectedLecture?.detected_questions) return [];
    try {
      const parsed = JSON.parse(selectedLecture.detected_questions);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }, [selectedLecture?.detected_questions]);

  // Carica i segnalibri e pulisce gli errori quando cambia la lezione selezionata
  useEffect(() => {
    setSummaryError(null);
    setQuestionAnswerError(null);
    setMissingLlmNotice(false);
    setShowExportMenu(false);
    setShowDeleteModal(false);
    initialNotesRef.current = null;
    // Ignora risposte arrivate dopo il cambio di lezione
    let alive = true;
    if (selectedLecture?.id) {
      api.bookmarks.listByLecture(selectedLecture.id)
        .then((bms) => {
          if (alive) setBookmarks(Array.isArray(bms) ? bms : []);
        })
        .catch(() => {
          if (alive) setBookmarks([]);
        });
    } else {
      setBookmarks([]);
    }
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      } catch {}
    }
    setIsPlaying(false);
    setCurrentTime(0);
    setAudioDuration(selectedLecture?.duration_seconds || 0);
    return () => {
      alive = false;
      if (audioRef.current) {
        try {
          audioRef.current.pause();
        } catch {}
      }
    };
  }, [selectedLecture?.id]);

  // Carica i timestamp parola per parola SOLO quando si apre la scheda Trascrizione (evita calcoli pesanti su 'notes')
  useEffect(() => {
    if (!selectedLecture) {
      setKaraokeLines(null);
      setLoadingKaraoke(false);
      loadedKaraokeKeyRef.current = null;
      return;
    }
    const cacheKey = `${selectedLecture.id}:${selectedLecture.raw_transcript?.length ?? 0}`;
    if (loadedKaraokeKeyRef.current !== cacheKey) {
      setKaraokeLines(null);
    }
    if (!isActive || activeTab !== 'transcript') {
      return;
    }
    if (loadedKaraokeKeyRef.current === cacheKey) {
      setLoadingKaraoke(false);
      return;
    }
    let isMounted = true;
    let pollTimeout: ReturnType<typeof setTimeout> | null = null;
    setLoadingKaraoke(true);

    const fetchTimestamps = (isInitial: boolean) => {
      api.lectures.getTimestamps(selectedLecture.id)
        .then((res: any) => {
          if (!isMounted) return;
          const parsed = normalizeKaraokeLines(res);
          if (parsed.length > 0) {
            setKaraokeLines(parsed);
            if (res?.cached !== false) {
              loadedKaraokeKeyRef.current = cacheKey;
            }
          }
          // Se la risposta è una stima iniziale transitoria (cached: false),
          // rischedula un fetch dopo 3.5s per caricare i timestamp definitivi parola per parola di Whisper
          if (res?.cached === false && isInitial) {
            pollTimeout = setTimeout(() => {
              if (isMounted) fetchTimestamps(false);
            }, 3500);
          }
        })
        .catch((err) => {
          console.warn('[Karaoke] Timestamps non disponibili:', err);
        })
        .finally(() => {
          if (isMounted && isInitial) setLoadingKaraoke(false);
        });
    };

    fetchTimestamps(true);

    return () => {
      isMounted = false;
      if (pollTimeout) clearTimeout(pollTimeout);
    };
  }, [isActive, activeTab, selectedLecture?.id, selectedLecture?.raw_transcript]);

  const handleSeekAudio = (seconds: number) => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    const maxDur = (audioDuration && !isNaN(audioDuration) && audioDuration > 0)
      ? audioDuration
      : (audioRef.current.duration > 0 ? audioRef.current.duration : 0);
    const target = maxDur > 0 ? Math.max(0, Math.min(maxDur, seconds)) : Math.max(0, seconds);
    audioRef.current.currentTime = target;
    setCurrentTime(target);
    if (!isPlaying) {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((e) => console.warn('Audio play prevented:', e));
    }
  };

  const handleCopyTranscript = (textToCopy?: string) => {
    const text = typeof textToCopy === 'string' && textToCopy.trim().length > 0
      ? textToCopy
      : selectedLecture?.raw_transcript;
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedTranscript(true);
    setTimeout(() => setCopiedTranscript(false), 2000);
  };

  const toggleAudioPlay = () => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((e) => console.warn('Audio play prevented:', e));
    }
  };

  const handleJumpToBookmark = (seconds: number) => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, seconds);
    setCurrentTime(seconds);
    audioRef.current
      .play()
      .then(() => setIsPlaying(true))
      .catch(() => {});
  };

  const lastAudioTargetKeyRef = useRef<number | null>(null);

  const handleJumpToAudioSnippet = useCallback(
    (lectureId: number, seconds: number, autoPlay: boolean = true) => {
      onOpenAudioSnippet?.(lectureId, seconds, autoPlay);
      if (selectedLectureId !== lectureId) {
        setSelectedLectureId(lectureId);
      }
      setActiveTab('transcript');

      const applySeek = () => {
        if (audioRef.current) {
          audioRef.current.currentTime = Math.max(0, seconds);
          setCurrentTime(seconds);
          if (autoPlay) {
            audioRef.current
              .play()
              .then(() => setIsPlaying(true))
              .catch(() => {});
          }
        } else {
          setCurrentTime(seconds);
        }
      };

      setTimeout(applySeek, 80);
      setTimeout(applySeek, 300);
    },
    [selectedLectureId]
  );

  useEffect(() => {
    if (!initialAudioTarget || lastAudioTargetKeyRef.current === initialAudioTarget.key) return;
    lastAudioTargetKeyRef.current = initialAudioTarget.key;

    const { lectureId, seconds, autoPlay } = initialAudioTarget;
    handleJumpToAudioSnippet(lectureId, seconds, autoPlay !== false);
    onClearInitialAudioTarget?.();
  }, [initialAudioTarget, handleJumpToAudioSnippet, onClearInitialAudioTarget]);

  const handleSkipAudio = (offsetSeconds: number) => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    const maxDur = (audioDuration && !isNaN(audioDuration) && audioDuration > 0)
      ? audioDuration
      : (audioRef.current.duration > 0 ? audioRef.current.duration : 99999);
    const newTime = Math.max(0, Math.min(maxDur, (audioRef.current.currentTime || 0) + offsetSeconds));
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  // ── Funzioni di Esportazione Markdown & Testo ──────────────────────────────
  const downloadMarkdownFile = async (
    content: string,
    filename: string,
    filterName = 'Markdown',
    extension = 'md'
  ) => {
    setShowExportMenu(false);
    const savedPath = await saveTextFile(filename, content, filterName, extension);
    if (savedPath) {
      setExportNotice('File salvato con successo in: ' + savedPath);
      setTimeout(() => setExportNotice(null), 4000);
    }
  };

  const handleExportSingleLecture = async () => {
    if (!selectedLecture) return;
    const dateFormatted = formatDate(selectedLecture.date);
    const durFormatted = formatDuration(selectedLecture.duration_seconds);

    let md = `# ${selectedLecture.title}\n\n`;
    md += `- **Corso:** ${subject.name}\n`;
    md += `- **Data Lezione:** ${dateFormatted}\n`;
    if (durFormatted) md += `- **Durata Audio:** ${durFormatted}\n`;
    md += `\n---\n\n`;

    md += `## 1. Appunti di Studio & Schema Concettuale\n\n`;
    md += (selectedLecture.processed_notes || '_Nessun appunto elaborato per questa lezione._').trim();
    md += `\n\n---\n\n`;

    if (bookmarks.length > 0) {
      md += `## 2. Segnalibri Esame Rilevati a Lezione\n\n`;
      bookmarks.forEach((b, idx) => {
        md += `- **[${formatSeconds(b.timestamp_seconds)}]** ${b.note || `Segnalibro #${idx + 1}`}\n`;
      });
      md += `\n---\n\n`;
    }

    if (parsedQuestions.length > 0) {
      md += `## 3. Domande del Docente Rilevate\n\n`;
      parsedQuestions.forEach((q, idx) => {
        md += `### Domanda ${idx + 1}: "${q.question}"\n\n`;
        if (q.suggestedAnswer && !isPlaceholderAnswer(q.suggestedAnswer)) {
          md += `**Risposta SbobinAI:**\n${q.suggestedAnswer}\n\n`;
        }
      });
      md += `\n---\n\n`;
    }

    if (selectedLecture.raw_transcript) {
      md += `## 4. Trascrizione Integrale\n\n`;
      md += selectedLecture.raw_transcript.trim();
      md += `\n`;
    }

    const cleanFilename = `${subject.name}_${selectedLecture.title}`.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.md';
    await downloadMarkdownFile(md, cleanFilename, 'Markdown', 'md');
  };

  const handleExportFullQuaderno = async () => {
    const today = new Date().toLocaleDateString('it-IT', { day: 'numeric', month: 'long', year: 'numeric' });
    let md = `# Quaderno di Studio — ${subject.name}\n\n`;
    md += `*Generato da SbobinAI il ${today}*\n\n`;
    if (subject.description) {
      md += `> ${subject.description}\n\n`;
    }
    md += `## Indice Cronologico delle Lezioni (${lectures.length})\n\n`;
    lectures.forEach((l, idx) => {
      md += `${idx + 1}. **${l.title}** (${formatDate(l.date)})\n`;
    });
    md += `\n\n---\n\n`;

    lectures.forEach((l, idx) => {
      md += `# Lezione ${idx + 1}: ${l.title}\n`;
      md += `*Data: ${formatDate(l.date)}*\n\n`;
      md += `### Appunti e Concetti Chiave\n\n`;
      md += (l.processed_notes || '_Appunti non disponibili._').trim() + '\n\n';
      if (l.raw_transcript) {
        md += `<details>\n<summary>Trascrizione Integrale</summary>\n\n${l.raw_transcript.trim()}\n\n</details>\n\n`;
      }
      md += `\n---\n\n`;
    });

    const cleanFilename = `Quaderno_${subject.name}`.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.md';
    await downloadMarkdownFile(md, cleanFilename, 'Markdown', 'md');
  };

  const handleExportTranscriptTxt = async () => {
    if (!selectedLecture || !selectedLecture.raw_transcript) return;
    const cleanFilename = `Trascrizione_${subject.name}_${selectedLecture.title}`.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.txt';
    await downloadMarkdownFile(selectedLecture.raw_transcript, cleanFilename, 'Testo', 'txt');
  };

  const handleDownloadAudio = () => {
    if (!selectedLecture || !selectedLecture.audio_path) return;
    const ext = selectedLecture.audio_path.split('.').pop() || 'wav';
    const cleanFilename = `Audio_${subject.name}_${selectedLecture.title}`.replace(/[^a-zA-Z0-9_\-]/g, '_') + `.${ext}`;
    const audioUrl = api.lectures.getAudioUrl(selectedLecture.id);
    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = cleanFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setShowExportMenu(false);
    setExportNotice(`Download audio avviato: ${cleanFilename}`);
    setTimeout(() => setExportNotice(null), 3500);
  };


  // Inizia modifica lezione
  const startEditing = () => {
    if (!selectedLecture) return;
    setEditTitle(selectedLecture.title);
    setEditNotes(selectedLecture.processed_notes || '');
    setEditTab('editor');
    setIsEditing(true);
  };

  const cancelEditing = () => {
    setIsEditing(false);
  };

  const saveEditing = async () => {
    if (!selectedLecture) return;
    setIsSavingEdit(true);
    try {
      await onUpdateLecture(selectedLecture.id, {
        title: editTitle.trim() || selectedLecture.title,
        processed_notes: editNotes.trim(),
      });
      setIsEditing(false);
    } catch {
      // Error handled in parent
    } finally {
      setIsSavingEdit(false);
    }
  };

  const handleDeleteSavedQuestion = async (targetId: string) => {
    if (!selectedLecture) return;
    const updated = parsedQuestions.filter((q, idx) => (q.id || String(idx)) !== targetId);
    try {
      await onUpdateLecture(selectedLecture.id, {
        detected_questions: updated.length > 0 ? JSON.stringify(updated) : null,
      });
    } catch {
      // Error handled in parent
    }
  };

  return (
    <div className="flex-1 flex overflow-hidden select-none">
      {/* ── Left Column: Chronological Index (w-72) ── */}
      <aside
        aria-label="Indice lezioni"
        className={clsx(
          'flex-shrink-0 flex flex-col h-full border-r border-white/[0.08] liquid-glass-subtle transition-all duration-200 ease-out overflow-hidden z-20',
          !isSidebarCollapsed
            ? 'w-72 opacity-100'
            : 'w-0 opacity-0 border-r-0 pointer-events-none'
        )}
      >
        <div className="w-72 flex flex-col h-full">
          {/* Top Bar: Search Bar with + and Collapse */}
          <div ref={addMenuRef} className="p-3 border-b border-white/[0.06] flex items-center gap-1.5 relative">
            <div className="relative flex-1 min-w-0">
              <Search size={11} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-white/30" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filtra lezioni..."
                className="w-full pl-7 pr-2.5 py-1.5 rounded-xl liquid-input text-[11px] text-white placeholder-white/25 focus:outline-none"
              />
            </div>

            {/* Bottone + per importare o aggiungere una lezione */}
            <button
              type="button"
              onClick={() => {
                setShowAddMenu((prev) => !prev);
                if (showAddMenu) setAddMenuMode('main');
              }}
              className={clsx(
                "w-7 h-7 rounded-xl flex items-center justify-center transition-all cursor-pointer flex-shrink-0 shadow-liquid-sm select-none",
                showAddMenu
                  ? "bg-white/[0.22] text-white border border-white/30 shadow-liquid-md scale-105"
                  : "liquid-glass hover:bg-white/[0.14] text-white/80 hover:text-white border border-white/[0.12] hover:scale-105"
              )}
              title={showAddMenu ? "Chiudi menu" : "Aggiungi o importa nel Quaderno"}
            >
              <AnimatePresence mode="wait" initial={false}>
                {showAddMenu ? (
                  <motion.span
                    key="close-add"
                    initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                    animate={{ rotate: 0, opacity: 1, scale: 1 }}
                    exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                    transition={{ duration: 0.16 }}
                    className="flex items-center justify-center"
                  >
                    <X size={13} strokeWidth={2.4} className="text-white" />
                  </motion.span>
                ) : (
                  <motion.span
                    key="plus-add"
                    initial={{ rotate: 90, opacity: 0, scale: 0.6 }}
                    animate={{ rotate: 0, opacity: 1, scale: 1 }}
                    exit={{ rotate: -90, opacity: 0, scale: 0.6 }}
                    transition={{ duration: 0.16 }}
                    className="flex items-center justify-center"
                  >
                    <Plus size={13} strokeWidth={2.2} className="text-white/90" />
                  </motion.span>
                )}
              </AnimatePresence>
            </button>

            {/* Bottone per ritirare la barra laterale */}
            <button
              type="button"
              onClick={() => setIsSidebarCollapsed(true)}
              className="w-7 h-7 rounded-xl flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-all cursor-pointer flex-shrink-0"
              title="Nascondi barra indice lezioni"
            >
              <PanelLeftClose size={13} strokeWidth={1.75} />
            </button>

            {/* Popover Liquid Glass ad espansione fluida (allineato all'indice, zero sovrapposizione laterale) */}
            <AnimatePresence>
              {showAddMenu && (
                <motion.div
                  variants={liquidMenuVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  style={{ transformOrigin: 'calc(100% - 46px) top' }}
                  className={clsx(
                    "absolute left-3 right-3 top-[calc(100%+6px)] z-50 text-left overflow-hidden",
                    liquidPopoverContainerClass
                  )}
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Contenitore interno ad altezza animata fluida */}
                  <motion.div
                    animate={{ height: menuHeight ?? 'auto' }}
                    transition={liquidLayoutTransition}
                    className="overflow-hidden"
                  >
                    <div ref={contentRef}>
                      <AnimatePresence mode="wait" initial={false}>
                        {addMenuMode === 'main' ? (
                          <motion.div
                            key="main"
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -8 }}
                            transition={{ duration: 0.14 }}
                            className="space-y-0.5"
                          >
                            {/* 1. Nuovo appunto vuoto */}
                            <button
                              type="button"
                              onClick={() => void handleCreateEmptyChapter()}
                              disabled={isCreatingChapter}
                              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-2xl text-left hover:bg-white/[0.08] active:bg-white/[0.12] transition-colors cursor-pointer group disabled:opacity-40"
                            >
                              <div className="w-7 h-7 rounded-xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-300 flex-shrink-0 group-hover:scale-105 transition-transform">
                                {isCreatingChapter ? (
                                  <Loader2 size={13} className="animate-spin text-emerald-300" />
                                ) : (
                                  <Pencil size={13} />
                                )}
                              </div>
                              <div className="min-w-0 flex-1">
                                <div className="text-[12px] font-semibold text-white/95 leading-tight">Nuovo appunto</div>
                                <div className="text-[10px] text-white/45 leading-none mt-0.5">Crea e scrivi subito</div>
                              </div>
                            </button>

                            {/* 2. Da Materiali */}
                            <button
                              type="button"
                              onClick={() => setAddMenuMode('materials')}
                              className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-2xl text-left hover:bg-white/[0.08] active:bg-white/[0.12] transition-colors cursor-pointer group"
                            >
                              <div className="w-7 h-7 rounded-xl bg-sky-500/20 border border-sky-400/30 flex items-center justify-center text-sky-300 flex-shrink-0 group-hover:scale-105 transition-transform">
                                <BookOpen size={13} />
                              </div>
                              <div className="min-w-0 flex-1">
                                <div className="text-[12px] font-semibold text-white/95 leading-tight">Da Materiali</div>
                                <div className="text-[10px] text-white/45 leading-none mt-0.5">
                                  Importa file del corso
                                </div>
                              </div>
                              <ChevronRight size={13} className="text-white/30 group-hover:text-white/80 group-hover:translate-x-0.5 transition-all" />
                            </button>

                            {/* 3. Registra lezione */}
                            {onStartLive && (
                              <button
                                type="button"
                                onClick={() => {
                                  setShowAddMenu(false);
                                  onStartLive();
                                }}
                                className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-2xl text-left hover:bg-white/[0.08] active:bg-white/[0.12] transition-colors cursor-pointer group"
                              >
                                <div className="w-7 h-7 rounded-xl bg-rose-500/20 border border-rose-400/30 flex items-center justify-center text-rose-300 flex-shrink-0 group-hover:scale-105 transition-transform">
                                  <Mic size={13} />
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="text-[12px] font-semibold text-white/95 leading-tight">Registra lezione</div>
                                  <div className="text-[10px] text-white/45 leading-none mt-0.5">Audio e trascrizione live</div>
                                </div>
                              </button>
                            )}
                          </motion.div>
                        ) : (
                          /* Subview: Da Materiali */
                          <motion.div
                            key="materials"
                            initial={{ opacity: 0, x: 8 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 8 }}
                            transition={{ duration: 0.14 }}
                            className="space-y-1"
                          >
                            <div className="flex items-center justify-between px-1 py-0.5 border-b border-white/[0.08] pb-1.5 mb-1">
                              <button
                                type="button"
                                onClick={() => setAddMenuMode('main')}
                                className="flex items-center gap-1 text-[11px] font-medium text-white/70 hover:text-white transition-colors cursor-pointer -ml-1 px-1.5 py-1 rounded-xl hover:bg-white/[0.08]"
                              >
                                <ChevronLeft size={13} />
                                <span>Indietro</span>
                              </button>
                            </div>

                            <div className="max-h-56 overflow-y-auto space-y-0.5 pr-0.5">
                              {availableMaterials.length === 0 ? (
                                <div className="py-6 px-3 text-center space-y-1">
                                  <Check size={15} className="mx-auto text-emerald-400/80" />
                                  <div className="text-[12px] font-semibold text-white/90">Tutto incluso</div>
                                  <p className="text-[10px] text-white/40 leading-tight">
                                    Tutti i file di questo corso sono già nel Quaderno.
                                  </p>
                                </div>
                              ) : (
                                availableMaterials.map((mat) => {
                                  const isImporting = importingMaterialId === mat.id;
                                  const cleanName = mat.name.replace(/\.[^/.]+$/, '');
                                  const isAudio = mat.type === 'lecture' || mat.type === 'audio';
                                  const isSlide = mat.type === 'pdf' || mat.type === 'slide';

                                  return (
                                    <button
                                      key={mat.id}
                                      type="button"
                                      onClick={() => void handleImportMaterial(mat)}
                                      disabled={isImporting}
                                      className="w-full flex items-center gap-2 px-2 py-1.5 rounded-xl text-left hover:bg-white/[0.08] active:bg-white/[0.12] transition-colors cursor-pointer group disabled:opacity-50"
                                    >
                                      <div className="w-5 h-5 rounded-md bg-white/[0.06] border border-white/[0.08] flex items-center justify-center text-white/60 flex-shrink-0 group-hover:text-white">
                                        {isImporting ? (
                                          <Loader2 size={10} className="animate-spin text-white" />
                                        ) : isAudio ? (
                                          <Mic size={10} className="text-rose-400" />
                                        ) : isSlide ? (
                                          <Presentation size={10} className="text-amber-400" />
                                        ) : (
                                          <FileText size={10} className="text-sky-400" />
                                        )}
                                      </div>
                                      <div className="min-w-0 flex-1">
                                        <div className="text-[11px] font-medium text-white/90 truncate group-hover:text-white">
                                          {cleanName}
                                        </div>
                                        <div className="text-[9.5px] text-white/40 leading-none mt-0.5 truncate">
                                          {isAudio
                                            ? mat.duration_seconds
                                              ? formatSeconds(mat.duration_seconds)
                                              : 'Audio lezione'
                                            : mat.size || 'Documento'}
                                        </div>
                                      </div>
                                      <Plus size={11} className="text-white/20 group-hover:text-white/70 flex-shrink-0" />
                                    </button>
                                  );
                                })
                              )}
                            </div>

                            {onOpenAddMaterialModal && (
                              <div className="pt-1.5 border-t border-white/[0.06] mt-1">
                                <button
                                  type="button"
                                  onClick={() => {
                                    setShowAddMenu(false);
                                    onOpenAddMaterialModal();
                                  }}
                                  className="w-full text-center text-[10px] text-white/50 hover:text-white py-1 transition-colors cursor-pointer rounded-lg hover:bg-white/[0.04]"
                                >
                                  + Carica nuovo file in Materiali
                                </button>
                              </div>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Lectures List */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredLectures.length === 0 ? (
              <div className="py-12 px-3 text-center">
                <BookOpen size={20} strokeWidth={1.5} className="mx-auto text-white/15 mb-2" />
                <p className="text-xs text-white/40">Nessuna lezione in {subject.name}</p>
                <button
                  onClick={() => {
                    setShowAddMenu(true);
                    setAddMenuMode('materials');
                  }}
                  className="mt-3 text-[11px] text-white/60 hover:text-white underline underline-offset-4 cursor-pointer"
                >
                  + Importa da Materiali
                </button>
              </div>
            ) : (
              filteredLectures.map((l) => {
                const isSelected = selectedLecture?.id === l.id;
                const isImported = l.source_type === 'imported';
                const isDragging = draggingLectureId === l.id;

                return (
                  <div
                    key={l.id}
                    draggable={!isEditing}
                    onDragStart={(e) => {
                      e.dataTransfer.setData(
                        'application/json',
                        JSON.stringify({
                          type: 'sbobinai-lecture',
                          lectureId: l.id,
                          lectureTitle: l.title,
                          sourceSubjectId: subject.id,
                        })
                      );
                      e.dataTransfer.setData('text/plain', `lecture:${l.id}`);
                      e.dataTransfer.effectAllowed = 'move';
                      setDraggingLectureId(l.id);
                    }}
                    onDragEnd={() => {
                      setDraggingLectureId(null);
                    }}
                    onClick={() => {
                      pendingRestoreIdRef.current = null;
                      setSelectedLectureId(l.id);
                      setIsEditing(false);
                    }}
                    title="Clicca per aprire. Trascina sul nome di un altro corso nella barra laterale a sinistra per spostarla"
                    className={clsx(
                      'p-3 rounded-2xl cursor-pointer transition-colors duration-150 border text-left group select-none',
                      isDragging
                        ? 'opacity-35 scale-[0.98] border-dashed border-blue-400/50 bg-blue-500/10 cursor-grabbing'
                        : isSelected
                        ? 'bg-white/[0.12] border-white/[0.16] shadow-liquid-sm'
                        : 'bg-transparent border-transparent hover:bg-white/[0.04] hover:border-white/[0.06]'
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-1.5 min-w-0 flex-1">
                        <GripVertical size={11} className="text-white/20 opacity-0 group-hover:opacity-60 transition-opacity flex-shrink-0 cursor-grab" />
                        <h4 className={clsx(
                          'text-xs font-semibold tracking-tight leading-snug line-clamp-1',
                          isSelected ? 'text-white' : 'text-white/80 group-hover:text-white'
                        )}>
                          {l.title}
                        </h4>
                      </div>
                      <ChevronRight size={12} className={clsx(
                        'flex-shrink-0 transition-opacity',
                        isSelected ? 'text-white/60' : 'opacity-0 group-hover:opacity-40'
                      )} />
                    </div>

                    <div className="flex items-center justify-between mt-2 text-[11px] text-white/50 font-mono">
                      <div className="flex items-center gap-1.5">
                        <Calendar size={11} strokeWidth={1.5} />
                        <span>{new Date(l.date).toLocaleDateString('it-IT', { day: 'numeric', month: 'short' })}</span>
                        {l.duration_seconds && l.duration_seconds > 0 ? (
                          <span>· {Math.floor(l.duration_seconds / 60)}m</span>
                        ) : null}
                      </div>

                      {/* Tag Source */}
                      <span className={clsx(
                        'text-[10px] px-2 py-0.5 rounded-full border',
                        isImported
                          ? 'bg-purple-500/10 border-purple-500/25 text-purple-300'
                          : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                      )}>
                        {isImported ? 'Importata' : 'Registrata'}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </aside>

      {/* ── Right Column: Clean Reading & Editing Sheet (flex-1) ── */}
      {selectedLecture ? (
        <div className="flex-1 flex flex-col overflow-hidden content-canvas relative">
          {/* Sheet Header — Clean macOS Document Header */}
          <div className="px-8 pt-6 pb-4 border-b border-white/[0.06] bg-white/[0.015] flex flex-col gap-4 flex-shrink-0">
            {/* Row 1: Title & Metadata on the left, Action buttons on the right */}
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1 flex items-start gap-3">
                {isSidebarCollapsed && (
                  <button
                    type="button"
                    onClick={() => setIsSidebarCollapsed(false)}
                    title="Mostra indice lezioni"
                    aria-label="Mostra indice lezioni"
                    className="w-8 h-8 rounded-full liquid-glass hover:bg-white/[0.14] text-white/70 hover:text-white border border-white/[0.14] transition-all cursor-pointer flex items-center justify-center flex-shrink-0 shadow-liquid-sm"
                  >
                    <PanelLeftOpen size={14} strokeWidth={1.75} />
                  </button>
                )}
                <div className="min-w-0 flex-1">
                  {isEditing ? (
                  <div>
                    <label className="block text-xs font-sans font-medium text-white/50 mb-1.5 uppercase tracking-wider">
                      Titolo Lezione
                    </label>
                    <input
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      placeholder="Titolo della lezione..."
                      className="w-full px-3.5 py-2 rounded-xl liquid-input text-lg font-bold text-white focus:outline-none"
                      autoFocus
                    />
                  </div>
                ) : (
                  <>
                    <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white leading-tight break-words">
                      {selectedLecture.title}
                    </h1>

                    {/* Metadata Subtitle: Date & Duration */}
                    <div className="flex items-center gap-2 mt-2 text-xs text-white/50 font-mono whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <Calendar size={13} strokeWidth={1.75} className="text-white/40" />
                        <span>{formatDate(selectedLecture.date)}</span>
                      </div>

                      {selectedLecture.duration_seconds && selectedLecture.duration_seconds > 0 ? (
                        <>
                          <span className="text-white/20">·</span>
                          <div className="flex items-center gap-1.5">
                            <Clock size={13} strokeWidth={1.75} className="text-white/40" />
                            <span>{formatDuration(selectedLecture.duration_seconds)}</span>
                          </div>
                        </>
                      ) : null}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Actions Toolbar */}
            <div className="flex items-center gap-2 flex-shrink-0 pt-0.5">
                {isEditing ? (
                  <>
                    <button
                      type="button"
                      onClick={cancelEditing}
                      disabled={isSavingEdit}
                      className="h-8 px-3.5 rounded-full text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer flex items-center gap-1.5"
                    >
                      <X size={13} strokeWidth={2} />
                      <span>Annulla</span>
                    </button>
                    <button
                      type="button"
                      onClick={saveEditing}
                      disabled={isSavingEdit}
                      className="h-8 px-4 rounded-full bg-white/[0.18] hover:bg-white/[0.26] border border-white/[0.22] text-xs font-semibold text-white shadow-liquid-sm transition-all cursor-pointer flex items-center gap-1.5"
                    >
                      <Check size={13} strokeWidth={2.5} />
                      <span>{isSavingEdit ? 'Salvataggio...' : 'Salva Modifiche'}</span>
                    </button>
                  </>
                ) : (
                  <>
                    {selectedLecture.raw_transcript && (
                      <div className="inline-flex">
                        <AnimatePresence mode="popLayout" initial={false}>
                          {isGeneratingSummary ? (
                            <motion.button
                              key="stop-summary"
                              type="button"
                              onClick={handleCancelSummary}
                              initial={{ opacity: 0, scale: 0.92 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0, scale: 0.92 }}
                              transition={{ duration: 0.16, ease: 'easeOut' }}
                              className="h-8 px-3.5 rounded-full bg-red-500/20 hover:bg-red-500/30 text-xs font-semibold text-red-200 hover:text-white border border-red-500/35 transition-colors flex items-center gap-1.5 cursor-pointer shadow-liquid-sm whitespace-nowrap"
                              title="Interrompi subito l'inferenza IA e libera le risorse del computer"
                            >
                              <span
                                aria-hidden="true"
                                className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse flex-shrink-0"
                              />
                              <Square size={10} fill="currentColor" strokeWidth={0} />
                              <span>Interrompi</span>
                            </motion.button>
                          ) : (
                            <motion.button
                              key="start-summary"
                              type="button"
                              onClick={handleGenerateSummary}
                              initial={{ opacity: 0, scale: 0.92 }}
                              animate={{ opacity: hasLLM ? 1 : 0.75, scale: 1 }}
                              whileHover={{ opacity: 1 }}
                              exit={{ opacity: 0, scale: 0.92 }}
                              transition={{ duration: 0.16, ease: 'easeOut' }}
                              className={clsx(
                                'h-8 px-3.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/90 border border-white/[0.14] transition-colors flex items-center gap-1.5 cursor-pointer shadow-liquid-sm whitespace-nowrap'
                              )}
                              title={
                                hasLLM
                                  ? 'Genera o rigenera la sintesi strutturata del Quaderno'
                                  : 'Richiede un modello LLM — clicca per configurare'
                              }
                            >
                              <Sparkles size={12} strokeWidth={2} className="text-amber-300" />
                              <span>Sintesi IA</span>
                            </motion.button>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    <div className="inline-flex">
                      <button
                        type="button"
                        onClick={startEditing}
                        className="h-8 px-3.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/90 border border-white/[0.14] transition-colors flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                        title="Modifica appunti o titolo"
                      >
                        <Pencil size={12} strokeWidth={2} />
                        <span>Modifica</span>
                      </button>
                    </div>

                    {/* Bottone Sposta con Morphing in 'X' & Pannello a Bolla Liquid Glass */}
                    {allSubjects && allSubjects.length > 1 && onMoveLecture && (
                      <div className="relative" ref={moveMenuRef}>
                        <button
                          type="button"
                          onClick={() => {
                            setShowDeleteModal(false);
                            setShowExportMenu(false);
                            setShowMoveMenu((prev) => !prev);
                          }}
                          className={clsx(
                            "w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer shadow-liquid-sm select-none",
                            showMoveMenu
                              ? "bg-white/[0.25] text-white border border-white/40 shadow-liquid-md scale-105"
                              : "liquid-glass-pill hover:bg-white/[0.18] text-white/90 border border-white/[0.14] hover:scale-105"
                          )}
                          title={showMoveMenu ? "Chiudi menu" : "Sposta in un altro corso"}
                        >
                          <AnimatePresence mode="wait" initial={false}>
                            {showMoveMenu ? (
                              <motion.span
                                key="close"
                                initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                                animate={{ rotate: 0, opacity: 1, scale: 1 }}
                                exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                                transition={{ duration: 0.16 }}
                                className="flex items-center justify-center"
                              >
                                <X size={14} strokeWidth={2.5} className="text-white" />
                              </motion.span>
                            ) : (
                              <motion.span
                                key="move"
                                initial={{ rotate: 90, opacity: 0, scale: 0.6 }}
                                animate={{ rotate: 0, opacity: 1, scale: 1 }}
                                exit={{ rotate: -90, opacity: 0, scale: 0.6 }}
                                transition={{ duration: 0.16 }}
                                className="flex items-center justify-center"
                              >
                                <FolderInput size={13} strokeWidth={2} className="text-white/90" />
                              </motion.span>
                            )}
                          </AnimatePresence>
                        </button>

                        {/* Pannello a Bolla Liquid Glass */}
                        <AnimatePresence>
                          {showMoveMenu && (
                            <motion.div
                              variants={liquidMenuVariants}
                              initial="initial"
                              animate="animate"
                              exit="exit"
                              style={{ transformOrigin: 'top right' }}
                              className={clsx("absolute right-0 top-11 w-64 space-y-1.5 p-1.5", liquidPopoverContainerClass)}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <div className="px-3 py-1.5 text-[10px] font-semibold text-white/45 uppercase tracking-wider">
                                Sposta nel corso
                              </div>
                              <div className="max-h-56 overflow-y-auto space-y-1">
                                {allSubjects
                                  .filter((s) => s.id !== subject.id)
                                  .map((s) => (
                                    <button
                                      key={s.id}
                                      type="button"
                                      disabled={isMovingLecture}
                                      onClick={async () => {
                                        if (!selectedLecture) return;
                                        const movedId = selectedLecture.id;
                                        const nextCandidate = lectures.find((l) => l.id !== movedId);
                                        setSelectedLectureId(nextCandidate ? nextCandidate.id : null);
                                        setShowMoveMenu(false);
                                        setIsMovingLecture(true);
                                        try {
                                          await onMoveLecture(movedId, s.id);
                                          setExportNotice(`Spostata in "${s.name}"`);
                                          setTimeout(() => setExportNotice(null), 3000);
                                        } catch (err) {
                                          console.error('Errore spostamento:', err);
                                          setSelectedLectureId(movedId);
                                        } finally {
                                          setIsMovingLecture(false);
                                        }
                                      }}
                                      className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer disabled:opacity-50"
                                    >
                                      <span
                                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                                        style={{ backgroundColor: s.color }}
                                      />
                                      <span className="block truncate font-semibold flex-1">
                                        {s.name}
                                      </span>
                                      <span className="text-[10px] text-white/40 font-mono">
                                        {s.lecture_count || 0} lez.
                                      </span>
                                    </button>
                                  ))}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {/* Bottone Esporta con Morphing in 'X' & Pannello a Bolla Liquid Glass */}
                    <div className="relative" ref={exportMenuRef}>
                      <button
                        type="button"
                        onClick={() => {
                          setShowDeleteModal(false);
                          setShowExportMenu((prev) => !prev);
                        }}
                        className={clsx(
                          "w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer shadow-liquid-sm select-none",
                          showExportMenu
                            ? "bg-white/[0.25] text-white border border-white/40 shadow-liquid-md scale-105"
                            : "liquid-glass-pill hover:bg-white/[0.18] text-white/90 border border-white/[0.14] hover:scale-105"
                        )}
                        title={showExportMenu ? "Chiudi menu" : "Esporta e scarica materiali"}
                      >
                        <AnimatePresence mode="wait" initial={false}>
                          {showExportMenu ? (
                            <motion.span
                              key="close"
                              initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                              animate={{ rotate: 0, opacity: 1, scale: 1 }}
                              exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                              transition={{ duration: 0.16 }}
                              className="flex items-center justify-center"
                            >
                              <X size={14} strokeWidth={2.5} className="text-white" />
                            </motion.span>
                          ) : (
                            <motion.span
                              key="export"
                              initial={{ rotate: 90, opacity: 0, scale: 0.6 }}
                              animate={{ rotate: 0, opacity: 1, scale: 1 }}
                              exit={{ rotate: -90, opacity: 0, scale: 0.6 }}
                              transition={{ duration: 0.16 }}
                              className="flex items-center justify-center"
                            >
                              <Share2 size={13} strokeWidth={2} className="text-white/90" />
                            </motion.span>
                          )}
                        </AnimatePresence>
                      </button>

                      {/* Pannello a Bolla Liquid Glass (alta opacità, zero sovrapposizione visiva) */}
                      <AnimatePresence>
                        {showExportMenu && (
                          <motion.div 
                            variants={liquidMenuVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                            style={{ transformOrigin: 'top right' }}
                            className={clsx("absolute right-0 top-11 w-72 space-y-1.5", liquidPopoverContainerClass)}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={handleExportSingleLecture}
                              className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer"
                            >
                              <FileDown size={15} className="text-amber-300 flex-shrink-0" />
                              <div className="min-w-0">
                                <span className="block truncate font-semibold">Lezione completa (.md)</span>
                                <span className="block text-[10px] text-white/45">Appunti, domande e trascrizione</span>
                              </div>
                            </button>

                            {selectedLecture.raw_transcript && (
                              <button
                                type="button"
                                onClick={handleExportTranscriptTxt}
                                className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer"
                              >
                                <FileText size={15} className="text-emerald-300 flex-shrink-0" />
                                <div className="min-w-0">
                                  <span className="block truncate font-semibold">Trascrizione</span>
                                  <span className="block text-[10px] text-white/45">Trascrizione integrale</span>
                                </div>
                              </button>
                            )}

                            {selectedLecture.audio_path && (
                              <button
                                type="button"
                                onClick={handleDownloadAudio}
                                className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer"
                              >
                                <Music size={15} className="text-cyan-300 flex-shrink-0" />
                                <div className="min-w-0">
                                  <span className="block truncate font-semibold">Registrazione</span>
                                </div>
                              </button>
                            )}

                            <div className="h-px bg-white/[0.10] my-1" />

                            <button
                              type="button"
                              onClick={handleExportFullQuaderno}
                              className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer"
                            >
                              <BookOpen size={15} className="text-blue-300 flex-shrink-0" />
                              <div className="min-w-0">
                                <span className="block truncate font-semibold">Intero Quaderno (.md)</span>
                                <span className="block text-[10px] text-white/45">{lectures.length} {lectures.length === 1 ? 'lezione' : 'lezioni'} del corso</span>
                              </div>
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Bottone Elimina con Morphing in 'X' & Pannello a Bolla Liquid Glass (identico a Esporta) */}
                    <div className="relative" ref={deleteMenuRef}>
                      <button
                        type="button"
                        onClick={() => {
                          setShowExportMenu(false);
                          setShowDeleteModal((prev) => !prev);
                        }}
                        title={showDeleteModal ? 'Chiudi menu' : 'Rimuovi lezione'}
                        className={clsx(
                          'w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer shadow-liquid-sm select-none',
                          showDeleteModal
                            ? 'bg-red-500/25 text-red-200 border border-red-400/45 shadow-liquid-md scale-105'
                            : 'bg-red-500/[0.12] hover:bg-red-500/20 text-red-400 hover:text-red-300 border border-red-500/25 hover:border-red-400/40 hover:scale-105'
                        )}
                      >
                        <AnimatePresence mode="wait" initial={false}>
                          {showDeleteModal ? (
                            <motion.span
                              key="close-delete"
                              initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                              animate={{ rotate: 0, opacity: 1, scale: 1 }}
                              exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                              transition={{ duration: 0.16 }}
                              className="flex items-center justify-center"
                            >
                              <X size={14} strokeWidth={2.5} className="text-red-200" />
                            </motion.span>
                          ) : (
                            <motion.span
                              key="trash"
                              initial={{ rotate: 90, opacity: 0, scale: 0.6 }}
                              animate={{ rotate: 0, opacity: 1, scale: 1 }}
                              exit={{ rotate: -90, opacity: 0, scale: 0.6 }}
                              transition={{ duration: 0.16 }}
                              className="flex items-center justify-center"
                            >
                              <Trash2 size={14} strokeWidth={1.75} />
                            </motion.span>
                          )}
                        </AnimatePresence>
                      </button>

                      <AnimatePresence>
                        {showDeleteModal && (
                          <motion.div
                            variants={liquidMenuVariants}
                            initial="initial"
                            animate="animate"
                            exit="exit"
                            style={{ transformOrigin: 'top right' }}
                            className={clsx("absolute right-0 top-11 w-72 space-y-1.5", liquidPopoverContainerClass)}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              type="button"
                              onClick={async () => {
                                try {
                                  await onRemoveFromQuadernoOnly(selectedLecture.id);
                                  setShowDeleteModal(false);
                                } catch (err) {
                                  console.error('Impossibile rimuovere dal quaderno:', err);
                                }
                              }}
                              className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-white/90 hover:text-white hover:bg-white/[0.12] transition-colors flex items-center gap-2.5 cursor-pointer"
                            >
                              <BookOpen size={15} className="text-white/65 flex-shrink-0" />
                              <div className="min-w-0">
                                <span className="block truncate font-semibold">Rimuovi dal Quaderno</span>
                                <span className="block text-[10px] text-white/45">Resta salvata nei Materiali</span>
                              </div>
                            </button>

                            <div className="h-px bg-white/[0.10] my-1" />

                            <button
                              type="button"
                              onClick={async () => {
                                try {
                                  await onDeletePermanently(selectedLecture.id);
                                  setShowDeleteModal(false);
                                } catch (err) {
                                  console.error('Impossibile eliminare la lezione:', err);
                                }
                              }}
                              className="w-full px-3 py-2 rounded-2xl text-left text-xs font-medium text-red-300 hover:text-red-200 hover:bg-red-500/15 transition-colors flex items-center gap-2.5 cursor-pointer"
                            >
                              <Trash2 size={15} className="text-red-400 flex-shrink-0" />
                              <div className="min-w-0">
                                <span className="block truncate font-semibold">Elimina definitivamente</span>
                                <span className="block text-[10px] text-red-300/60">Cancella audio, testo e appunti</span>
                              </div>
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Avviso Esportazione Completata */}
            {exportNotice && (
              <div className="px-3.5 py-1.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-xs text-emerald-200 flex items-center gap-2 animate-fade-in">
                <Check size={13} className="text-emerald-400 flex-shrink-0" />
                <span>{exportNotice}</span>
              </div>
            )}

            {/* Row 2: Segmented Control Tabs (Quaderno / Trascrizione / Domande) */}
            {!isEditing && (
              <div className="flex items-center pt-1 border-t border-white/[0.04]">
                <SegmentedControl
                  key={`quaderno-seg-${selectedLecture?.id ?? 'default'}`}
                  size="sm"
                  layoutId={`quaderno-tab-${selectedLecture?.id ?? 'default'}`}
                  options={[
                    { id: 'notes' as const, label: 'Quaderno', icon: FileText },
                    { id: 'transcript' as const, label: 'Trascrizione', icon: AlignLeft },
                    {
                      id: 'questions' as const,
                      label: 'Domande',
                      icon: HelpCircle,
                    },
                  ]}
                  value={activeTab}
                  onChange={setActiveTab}
                />
              </div>
            )}
          </div>

          {/* Sheet Body */}
          <div className="flex-1 overflow-y-auto overflow-x-hidden px-6 sm:px-12 py-8 pb-32 [overscroll-behavior-x:none]">
            <div className="max-w-3xl mx-auto space-y-6 w-full min-w-0">
              {isEditing ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-white/60">Appunti del Quaderno (Markdown / Testo)</label>
                    <div className="flex items-center gap-1 bg-white/[0.04] p-0.5 rounded-lg border border-white/[0.06]">
                      <button
                        type="button"
                        onClick={() => setEditTab('editor')}
                        className={clsx(
                          'px-2.5 py-0.5 rounded-md text-[11px] font-medium transition-all cursor-pointer',
                          editTab === 'editor'
                            ? 'bg-white/[0.16] text-white shadow-liquid-sm'
                            : 'text-white/40 hover:text-white/70'
                        )}
                      >
                        Editor
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditTab('preview')}
                        className={clsx(
                          'px-2.5 py-0.5 rounded-md text-[11px] font-medium transition-all cursor-pointer',
                          editTab === 'preview'
                            ? 'bg-white/[0.16] text-white shadow-liquid-sm'
                            : 'text-white/40 hover:text-white/70'
                        )}
                      >
                        Anteprima
                      </button>
                    </div>
                  </div>
                  {editTab === 'editor' ? (
                    <textarea
                      rows={18}
                      value={editNotes}
                      onChange={(e) => setEditNotes(e.target.value)}
                      placeholder="Scrivi o incolla qui gli appunti della lezione in formato Markdown..."
                      className="w-full p-4 rounded-2xl liquid-input text-xs text-white/90 leading-relaxed font-mono placeholder-white/20 focus:outline-none transition-all resize-y"
                    />
                  ) : (
                    <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.06] min-h-[300px]">
                      {editNotes.trim() ? (
                        <MarkdownContent
                          content={editNotes}
                          className="quaderno-markdown"
                          onOpenAudioSnippet={handleJumpToAudioSnippet}
                          onOpenCitation={onOpenCitation}
                          onOpenSettings={onOpenSettings}
                        />
                      ) : (
                        <p className="text-xs text-white/40 italic">Nessun contenuto da visualizzare nell'anteprima.</p>
                      )}
                    </div>
                  )}
                </div>
              ) : activeTab === 'notes' ? (
                <>
                  {!isGeneratingSummary && missingLlmNotice && (
                    <div className="p-4 rounded-2xl liquid-glass border border-amber-400/25 bg-amber-500/[0.07] shadow-liquid-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3.5 text-left animate-fade-in">
                      <div className="flex items-start gap-3 min-w-0">
                        <div className="w-8 h-8 rounded-xl bg-amber-400/15 border border-amber-400/25 flex items-center justify-center text-amber-300 flex-shrink-0 mt-0.5">
                          <AlertTriangle size={15} strokeWidth={2} />
                        </div>
                        <div className="min-w-0 space-y-0.5">
                          <h4 className="text-xs font-semibold text-white tracking-tight">
                            Nessun modello LLM scaricato
                          </h4>
                          <p className="text-xs text-white/65 leading-relaxed">
                            Scarica un modello IA dalle Impostazioni per generare automaticamente il capitolo del Quaderno.
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0 self-end sm:self-center">
                        {onOpenSettings && (
                          <button
                            type="button"
                            onClick={onOpenSettings}
                            className="h-8 px-3.5 rounded-full bg-white/[0.14] hover:bg-white/[0.22] border border-white/[0.20] text-xs font-semibold text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                          >
                            <Settings size={12} strokeWidth={2} />
                            <span>Apri Impostazioni</span>
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => setMissingLlmNotice(false)}
                          className="w-7 h-7 rounded-full flex items-center justify-center text-white/45 hover:text-white hover:bg-white/[0.10] transition-colors cursor-pointer"
                          title="Chiudi avviso"
                          aria-label="Chiudi avviso"
                        >
                          <X size={13} strokeWidth={2} />
                        </button>
                      </div>
                    </div>
                  )}

                  {isGeneratingSummary ? (
                    <ChapterGenerationSkeleton
                      statusMessage={summaryProgress?.message}
                      progress={summaryProgress?.progress}
                    />
                  ) : selectedLecture.processed_notes ? (
                    <div className="space-y-4">
                      {summaryError && (
                        <div className="p-3.5 rounded-2xl bg-red-500/10 border border-red-500/25 flex items-start gap-2.5 text-left animate-fade-in">
                          <AlertTriangle size={15} className="text-red-400 flex-shrink-0 mt-0.5" />
                          <div className="flex-1 space-y-1">
                            <p className="text-xs font-medium text-red-300 leading-relaxed">
                              {summaryError}
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => setSummaryError(null)}
                            className="w-6 h-6 rounded-full flex items-center justify-center text-red-200/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer flex-shrink-0"
                            title="Chiudi avviso"
                            aria-label="Chiudi avviso"
                          >
                            <X size={13} />
                          </button>
                        </div>
                      )}
                      <div className="p-4 sm:p-6 rounded-3xl bg-white/[0.02] border border-white/[0.06] shadow-liquid-sm w-full min-w-0 overflow-hidden">
                        <MarkdownContent
                          content={selectedLecture.processed_notes}
                          className="quaderno-markdown"
                          onOpenAudioSnippet={handleJumpToAudioSnippet}
                          onOpenCitation={onOpenCitation}
                          onOpenSettings={onOpenSettings}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/[0.06] text-center space-y-3">
                      <Sparkles size={24} strokeWidth={1.5} className="mx-auto text-amber-300 mb-2" />
                      <h3 className="text-sm font-semibold text-white/90">Nessun appunto nel Quaderno</h3>
                      <p className="text-xs text-white/50 max-w-sm mx-auto leading-relaxed">
                        {selectedLecture.raw_transcript?.trim()
                          ? hasLLM
                            ? 'La trascrizione è pronta. Usa "Sintesi IA" in alto per generare il capitolo, oppure "Modifica" per scrivere manualmente.'
                            : 'La trascrizione è pronta. Scarica un modello LLM dalle Impostazioni per generare il capitolo con l\'IA, oppure usa Modifica in alto.'
                          : 'Non ci sono ancora note per questa lezione. Usa "Modifica" in alto per scriverle o incollarle.'}
                      </p>

                      {selectedLecture.raw_transcript?.trim() && !hasLLM && onOpenSettings && (
                        <div className="pt-1">
                          <button
                            type="button"
                            onClick={onOpenSettings}
                            className="h-8 px-3.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/85 hover:text-white border border-white/[0.14] transition-all inline-flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                          >
                            <Settings size={12} strokeWidth={2} />
                            <span>Apri Impostazioni</span>
                          </button>
                        </div>
                      )}

                      {summaryError && (
                        <div className="max-w-md mx-auto p-3 rounded-2xl bg-red-500/10 border border-red-500/25 flex items-start gap-2.5 text-left animate-fade-in">
                          <AlertTriangle size={15} className="text-red-400 flex-shrink-0 mt-0.5" />
                          <div className="flex-1 space-y-1">
                            <p className="text-xs font-medium text-red-300 leading-relaxed">
                              {summaryError}
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => setSummaryError(null)}
                            className="w-6 h-6 rounded-full flex items-center justify-center text-red-200/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer flex-shrink-0"
                            title="Chiudi avviso"
                            aria-label="Chiudi avviso"
                          >
                            <X size={13} />
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </>
              ) : activeTab === 'transcript' ? (
                <div className="animate-fade-in">
                  <AppleLyricsView
                    lines={karaokeLines}
                    audioRef={audioRef}
                    audioPath={selectedLecture.audio_path || ''}
                    isPlaying={isPlaying}
                    onSeek={handleSeekAudio}
                    rawTranscript={selectedLecture.raw_transcript || ''}
                    onCopyTranscript={handleCopyTranscript}
                    copiedTranscript={copiedTranscript}
                    isLoadingTimestamps={loadingKaraoke}
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  {questionAnswerError && (
                    <div className="p-3.5 rounded-2xl bg-red-500/10 border border-red-500/25 flex items-start gap-2.5 text-left animate-fade-in">
                      <AlertTriangle size={15} className="text-red-400 flex-shrink-0 mt-0.5" />
                      <p className="flex-1 text-xs font-medium text-red-300 leading-relaxed">
                        {questionAnswerError}
                      </p>
                      <button
                        type="button"
                        onClick={() => setQuestionAnswerError(null)}
                        className="w-5 h-5 rounded-full flex items-center justify-center text-red-200/60 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer flex-shrink-0"
                        title="Chiudi avviso"
                        aria-label="Chiudi avviso"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  )}

                  {parsedQuestions.length === 0 ? (
                    <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/[0.06] text-center space-y-2.5">
                      <HelpCircle size={22} strokeWidth={1.5} className="mx-auto text-white/30" />
                      <h3 className="text-sm font-semibold text-white/90">Nessuna domanda rilevata</h3>
                      <p className="text-xs text-white/50 max-w-sm mx-auto leading-relaxed">
                        Non sono state intercettate domande del docente durante questa lezione.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {parsedQuestions.map((q, idx) => {
                        const hasModel = Boolean(settings.llmDownloaded);
                        const isPending = isPlaceholderAnswer(q.suggestedAnswer);
                        // Fallback sull'indice: il backend riconosce le domande anche per posizione
                        const qKey = q.id || String(idx);
                        const isAnswering = answeringQuestionId === qKey;

                        return (
                          <div
                            key={qKey}
                            className="p-4 rounded-2xl bg-white/[0.03] border border-white/[0.09] hover:border-white/[0.14] transition-colors space-y-3 animate-slide-up group"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <p className="text-xs sm:text-[13px] font-medium text-white/95 leading-relaxed">
                                &ldquo;{q.question}&rdquo;
                              </p>
                              <div className="flex items-center gap-1 flex-shrink-0">
                                {hasModel && !isPending && (
                                  <button
                                    type="button"
                                    disabled={isAnswering}
                                    onClick={() => handleAnswerQuestion(qKey)}
                                    title="Rigenera risposta"
                                    aria-label="Rigenera risposta"
                                    className="w-7 h-7 rounded-lg flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-all cursor-pointer disabled:opacity-40"
                                  >
                                    <RotateCcw size={13} className={isAnswering ? 'animate-spin text-amber-300' : ''} />
                                  </button>
                                )}
                                <button
                                  type="button"
                                  onClick={() => handleDeleteSavedQuestion(qKey)}
                                  title="Elimina domanda"
                                  aria-label="Elimina domanda"
                                  className="w-7 h-7 rounded-lg flex items-center justify-center text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer"
                                >
                                  <Trash2 size={13} strokeWidth={1.75} />
                                </button>
                              </div>
                            </div>

                            {/* Stato risposta */}
                            {isPending ? (
                              <div className="pt-2.5 border-t border-white/[0.06]">
                                {!hasModel ? (
                                  <div className="p-3.5 rounded-2xl liquid-glass border border-amber-400/25 bg-amber-500/[0.07] shadow-liquid-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-left">
                                    <div className="flex items-start gap-3 min-w-0">
                                      <div className="w-8 h-8 rounded-xl bg-amber-400/15 border border-amber-400/25 flex items-center justify-center text-amber-300 flex-shrink-0 mt-0.5">
                                        <AlertTriangle size={14} strokeWidth={2} />
                                      </div>
                                      <div className="min-w-0 space-y-0.5">
                                        <h4 className="text-xs font-semibold text-white tracking-tight">
                                          Nessun modello LLM scaricato
                                        </h4>
                                        <p className="text-xs text-white/65 leading-relaxed">
                                          Scarica un modello IA dalle Impostazioni per generare risposte alle domande del docente.
                                        </p>
                                      </div>
                                    </div>
                                    {onOpenSettings && (
                                      <button
                                        type="button"
                                        onClick={onOpenSettings}
                                        className="h-8 px-3.5 rounded-full bg-white/[0.14] hover:bg-white/[0.22] border border-white/[0.20] text-xs font-semibold text-white transition-all flex items-center gap-1.5 flex-shrink-0 self-end sm:self-center cursor-pointer shadow-liquid-sm"
                                      >
                                        <Settings size={12} strokeWidth={2} />
                                        <span>Apri Impostazioni</span>
                                      </button>
                                    )}
                                  </div>
                                ) : (
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="text-xs text-white/45">
                                      Nessuna risposta memorizzata per questa domanda.
                                    </span>
                                    <button
                                      type="button"
                                      disabled={isAnswering}
                                      onClick={() => handleAnswerQuestion(qKey)}
                                      className="h-8 px-3.5 rounded-full bg-white/[0.10] hover:bg-white/[0.16] text-white text-xs font-medium border border-white/[0.14] shadow-liquid-sm transition-all inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50 whitespace-nowrap flex-shrink-0"
                                    >
                                      {isAnswering ? (
                                        <>
                                          <Loader2 size={12} className="animate-spin text-white/75" />
                                          <span className="chat-shimmer-text">SbobinAI sta rispondendo...</span>
                                        </>
                                      ) : (
                                        <>
                                          <Sparkles size={12} className="text-amber-300" />
                                          <span>Rispondi con l&apos;IA</span>
                                        </>
                                      )}
                                    </button>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="pt-2 border-t border-white/[0.06]">
                                <div className="text-xs text-white/85 leading-relaxed bg-white/[0.03] p-3.5 rounded-xl border border-white/[0.06] select-text">
                                  <MarkdownContent
                                    content={formatQuestionAnswer(q.suggestedAnswer!)}
                                    onOpenAudioSnippet={handleJumpToAudioSnippet}
                                    onOpenCitation={onOpenCitation}
                                    onOpenSettings={onOpenSettings}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ── Apple Music Floating Liquid Glass Audio Player Pill ── */}
          {selectedLecture.audio_path && (
            <div className="absolute bottom-6 left-0 right-0 z-30 flex flex-col items-center pointer-events-none px-6">
              {/* Segnalibri Esame: Floating mini-pill dock above player */}
              {bookmarks.length > 0 && (
                <div className="pointer-events-auto mb-2 flex items-center gap-1.5 overflow-x-auto max-w-xl py-1 px-3.5 rounded-full bg-[#0c0d14]/88 backdrop-blur-xl border border-white/[0.12] shadow-[0_8px_24px_rgba(0,0,0,0.5)] no-scrollbar animate-fade-in">
                  <span className="text-[10px] font-mono uppercase text-amber-400 flex items-center gap-1 flex-shrink-0 mr-1 font-semibold">
                    <BookmarkIcon size={11} className="fill-amber-400/20 text-amber-400" />
                    <span>Segnalibri:</span>
                  </span>
                  {bookmarks.map((b) => {
                    const isNear = Math.abs(currentTime - b.timestamp_seconds) <= 2;
                    return (
                      <button
                        key={b.id}
                        type="button"
                        onClick={() => handleJumpToBookmark(b.timestamp_seconds)}
                        title={`Salta al secondo ${Math.round(b.timestamp_seconds)}`}
                        className={clsx(
                          'px-2.5 py-0.5 rounded-full text-[10.5px] font-mono flex items-center gap-1 transition-all flex-shrink-0 cursor-pointer border',
                          isNear
                            ? 'bg-amber-500/25 border-amber-500/50 text-amber-200 shadow-liquid-sm scale-105'
                            : 'bg-white/[0.06] hover:bg-white/[0.12] border-white/[0.08] text-white/70'
                        )}
                      >
                        <span className="text-amber-400 text-[9px]">▶</span>
                        <span>{formatSeconds(b.timestamp_seconds)}</span>
                        {b.note && <span className="text-white/40 truncate max-w-[120px]">· {b.note}</span>}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Floating Pill Player Container — Apple Music Liquid Glass Capsule */}
              <div className="pointer-events-auto w-full max-w-xl h-[52px] px-4 rounded-full bg-[#121622]/78 hover:bg-[#121622]/86 backdrop-blur-xl border border-white/[0.20] shadow-[0_20px_50px_rgba(0,0,0,0.65),inset_0_1px_1px_rgba(255,255,255,0.30)] flex items-center gap-3 transition-colors">
                {/* HTML Audio element — cache-buster su audio_path evita errori 416 in Safari quando .wav diventa .m4a */}
                <audio
                  ref={audioRef}
                  src={`${api.lectures.getAudioUrl(selectedLecture.id)}?v=${encodeURIComponent(selectedLecture.audio_path || '')}`}
                  preload="metadata"
                  onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
                  onLoadedMetadata={() => setAudioDuration(audioRef.current?.duration || selectedLecture.duration_seconds || 0)}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                  onError={() => {
                    setIsPlaying(false);
                    console.warn('[Audio] Errore caricamento sorgente audio.');
                  }}
                />

                {/* Play / Pause button — Apple Music circular button */}
                <button
                  type="button"
                  onClick={toggleAudioPlay}
                  className="w-8 h-8 rounded-full bg-white text-black hover:bg-white/90 active:scale-95 flex items-center justify-center shadow-[0_2px_12px_rgba(255,255,255,0.35)] transition-all cursor-pointer flex-shrink-0"
                  title={isPlaying ? 'Pausa' : 'Riproduci'}
                >
                  {isPlaying ? (
                    <Pause size={13} className="fill-current" />
                  ) : (
                    <Play size={13} className="fill-current ml-0.5" />
                  )}
                </button>

                {/* Skip -10s */}
                <button
                  type="button"
                  onClick={() => handleSkipAudio(-10)}
                  className="w-7 h-7 rounded-full text-white/60 hover:text-white hover:bg-white/[0.10] flex items-center justify-center transition-all cursor-pointer flex-shrink-0 active:scale-95"
                  title="Riavvolgi 10 secondi"
                >
                  <RotateCcw size={13} />
                </button>

                {/* Current Time */}
                <span className="text-[11px] font-mono text-white/85 min-w-[32px] text-right font-medium tabular-nums">
                  {formatSeconds(currentTime)}
                </span>

                {/* Scrubber Range Slider */}
                <div className="flex-1 relative flex items-center group">
                  <input
                    type="range"
                    min={0}
                    max={audioDuration || (audioRef.current?.duration || 100)}
                    step={0.1}
                    value={currentTime}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      setCurrentTime(val);
                      if (audioRef.current) audioRef.current.currentTime = val;
                    }}
                    className="w-full h-1 group-hover:h-1.5 bg-white/[0.16] rounded-full appearance-none cursor-pointer accent-white transition-all"
                  />
                </div>

                {/* Total Duration */}
                <span className="text-[11px] font-mono text-white/45 min-w-[32px] tabular-nums">
                  {formatSeconds(audioDuration || selectedLecture.duration_seconds || 0)}
                </span>

                {/* Skip +10s */}
                <button
                  type="button"
                  onClick={() => handleSkipAudio(10)}
                  className="w-7 h-7 rounded-full text-white/60 hover:text-white hover:bg-white/[0.10] flex items-center justify-center transition-all cursor-pointer flex-shrink-0 active:scale-95"
                  title="Avanza 10 secondi"
                >
                  <RotateCcw size={13} className="scale-x-[-1]" />
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center relative">
          {isSidebarCollapsed && (
            <button
              type="button"
              onClick={() => setIsSidebarCollapsed(false)}
              title="Mostra indice lezioni"
              aria-label="Mostra indice lezioni"
              className="absolute top-4 left-4 w-9 h-9 rounded-full liquid-glass hover:bg-white/[0.14] text-white/75 hover:text-white flex items-center justify-center border border-white/[0.14] shadow-liquid-sm transition-all cursor-pointer"
            >
              <PanelLeftOpen size={15} strokeWidth={1.75} />
            </button>
          )}
          <BookOpen size={32} strokeWidth={1.5} className="text-white/20 mb-3" />
          <h3 className="text-sm font-semibold text-white/80">Nessuna lezione selezionata</h3>
          <p className="text-xs text-white/40 max-w-xs mt-1">Seleziona una lezione dall'indice o importane una dai materiali.</p>
        </div>
      )}

    </div>
  );
}
