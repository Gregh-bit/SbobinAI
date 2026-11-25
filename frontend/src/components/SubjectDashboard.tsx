// components/SubjectDashboard.tsx — Dashboard Corso con Segmented Control e viste
import { useState, useEffect, useCallback, useMemo, useRef, lazy, Suspense } from 'react';
import { 
  Mic, BookOpen, Layers, HelpCircle, MessageSquare, 
  Loader2, Download, AlertTriangle, Square, Pause, Play, Bookmark 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import { useLiveSession, formatElapsed } from '../lib/useLiveSession';
import { shortcutLabel } from '../lib/platform';
import type { CourseTab, Lecture, Subject } from '../types';
import QuadernoView from './QuadernoView';
import SegmentedControl from './SegmentedControl';
import ErrorBoundary from './ErrorBoundary';
import {
  ChatStreamSkeleton,
  LiveWorkspaceSkeleton,
  MaterialiGridSkeleton,
  QuadernoSkeleton,
  QuizWizardSkeleton,
  SkeletonPill,
  useDelayedLoading,
} from './Skeleton';

// Lazy loading viste secondarie e live workspace per apertura istantanea (<100ms)
const MaterialiView = lazy(() => import('./MaterialiView'));
const QuizRipassoView = lazy(() => import('./QuizRipassoView'));
const AssistenteCorsoView = lazy(() => import('./AssistenteCorsoView'));
const LiveRecordingWorkspace = lazy(() => import('./LiveRecordingWorkspace'));

interface Props {
  subject: Subject;
  allSubjects?: Subject[];
  onOpenSettings?: () => void;
  onRecordingChange?: (isRecording: boolean) => void;
  onLectureCountChange?: (subjectId: number, count: number) => void;
  onMoveLecture?: (lectureId: number, targetSubjectId: number) => Promise<void>;
  lastMovedLectureId?: number | null;
  [key: string]: any;
}

// Polling delle lezioni dopo lo stop finché il backend non ha generato gli appunti
const NOTES_POLL_INTERVAL_MS = 5000;
const NOTES_POLL_MAX_MS = 3 * 60 * 1000;

const TABS: { id: CourseTab; label: string; icon: typeof BookOpen }[] = [
  { id: 'quaderno', label: 'Quaderno', icon: BookOpen },
  { id: 'materiali', label: 'Materiali', icon: Layers },
  { id: 'quiz', label: 'Quiz', icon: HelpCircle },
  { id: 'assistente', label: 'Assistente', icon: MessageSquare },
];

export default function SubjectDashboard({ subject, allSubjects, onOpenSettings, onRecordingChange, onLectureCountChange, onMoveLecture, lastMovedLectureId }: Props) {
  const { settings } = useSettings();
  const [activeTab, setActiveTab] = useState<CourseTab>('quaderno');
  const [hasMountedQuaderno, setHasMountedQuaderno] = useState(true);
  const [lectures, setLectures] = useState<Lecture[]>([]);
  const quadernoLectures = useMemo(
    () => lectures.filter((l) => l.in_quaderno !== false),
    [lectures]
  );

  useEffect(() => {
    if (activeTab === 'quaderno') {
      setHasMountedQuaderno(true);
    }
  }, [activeTab]);
  const [loading, setLoading] = useState(true);
  const [hasLoadedLectures, setHasLoadedLectures] = useState(false);
  const showLecturesSkeleton = useDelayedLoading(loading, 150, 220);
  const [startingLive, setStartingLive] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [activeRecordingLecture, setActiveRecordingLecture] = useState<Lecture | null>(null);
  const [isClosingLive, setIsClosingLive] = useState(false);
  const isCapsuleExpanded = Boolean(activeRecordingLecture) && !isClosingLive;

  // Guardie contro risposte asincrone obsolete (dismount o richieste superate)
  const isMountedRef = useRef(true);
  const loadSeqRef = useRef(0);
  const notesPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (notesPollRef.current) clearInterval(notesPollRef.current);
    };
  }, []);

  const loadLectures = useCallback(async (isInitial = false): Promise<Lecture[] | null> => {
    const seq = ++loadSeqRef.current;
    if (isInitial) {
      setLoading(true);
      setHasLoadedLectures(false);
    }
    try {
      const data = await api.lectures.listBySubject(subject.id);
      if (!isMountedRef.current || seq !== loadSeqRef.current) return null;
      setLectures(data);
      setHasLoadedLectures(true);
      return data;
    } catch {
      // Ignora silenziosamente
      return null;
    } finally {
      if (isMountedRef.current && seq === loadSeqRef.current) setLoading(false);
    }
  }, [subject.id]);

  // Se una lezione è stata spostata da un'azione esterna (es. drag sulla Sidebar), la rimuoviamo dallo stato locale
  useEffect(() => {
    if (lastMovedLectureId) {
      setLectures((prev) => prev.filter((l) => l.id !== lastMovedLectureId));
    }
  }, [lastMovedLectureId]);

  const handleMoveLecture = useCallback(async (lectureId: number, targetSubjectId: number) => {
    // 1. Rimuovi immediatamente dalla lista locale per aggiornare la UI all'istante
    setLectures((prev) => prev.filter((l) => l.id !== lectureId));
    onLectureCountChange?.(subject.id, Math.max(0, lectures.length - 1));
    const targetSub = allSubjects?.find((s) => s.id === targetSubjectId);
    if (targetSub) {
      onLectureCountChange?.(targetSubjectId, (targetSub.lecture_count || 0) + 1);
    }

    // 2. Esegui la chiamata di spostamento
    if (onMoveLecture) {
      await onMoveLecture(lectureId, targetSubjectId);
    } else {
      await api.lectures.move(lectureId, targetSubjectId);
    }

    // 3. Sincronizza lo stato reale con il backend
    void loadLectures(false);
  }, [onMoveLecture, subject.id, lectures.length, allSubjects, onLectureCountChange, loadLectures]);

  // Stato per handoff cross-tab (Citazioni -> Materiali, Audio Snippet -> Quaderno, Quiz/Orale -> QuizRipassoView)
  const [materialTarget, setMaterialTarget] = useState<{
    sourceId: string;
    page?: number;
    key: number;
  } | null>(null);
  const [audioSnippetTarget, setAudioSnippetTarget] = useState<{
    lectureId: number;
    seconds: number;
    autoPlay?: boolean;
    key: number;
  } | null>(null);
  const [pendingQuizId, setPendingQuizId] = useState<number | null>(null);
  const [pendingOralQuestion, setPendingOralQuestion] = useState<{
    question: string;
    topic?: string;
    key: number;
  } | null>(null);
  const [pendingChatReview, setPendingChatReview] = useState<{
    chatSessionId?: number | null;
    quizId: number;
    quizTitle: string;
    score: number;
    total: number;
    wrongCount: number;
    key: number;
  } | null>(null);

  // Mantieni sincronizzato il badge del numero lezioni nella Sidebar senza refresh
  useEffect(() => {
    if (!loading && hasLoadedLectures) {
      onLectureCountChange?.(subject.id, lectures.length);
    }
  }, [loading, hasLoadedLectures, lectures.length, subject.id, onLectureCountChange]);

  useEffect(() => {
    setMaterialTarget(null);
    setAudioSnippetTarget(null);
    setPendingQuizId(null);
    setPendingOralQuestion(null);
    loadLectures(true);
    return () => {
      if (notesPollRef.current) {
        clearInterval(notesPollRef.current);
        notesPollRef.current = null;
      }
    };
  }, [loadLectures]);

  const handleOpenCitation = useCallback((sourceId: string, page?: number) => {
    setMaterialTarget({ sourceId, page: page ?? 1, key: Date.now() });
    setActiveTab('materiali');
  }, []);

  const handleOpenAudioSnippet = useCallback(
    (lectureId: number, seconds: number, autoPlay: boolean = true) => {
      setAudioSnippetTarget({ lectureId, seconds, autoPlay, key: Date.now() });
      setActiveTab('quaderno');
    },
    []
  );

  const handleOpenQuiz = useCallback((quizId: number) => {
    setPendingQuizId(quizId);
    setActiveTab('quiz');
  }, []);

  const handleOpenOralQuestion = useCallback((question: string, topic?: string) => {
    setPendingOralQuestion({ question, topic, key: Date.now() });
    setActiveTab('quiz');
  }, []);

  const handleReviewInChat = useCallback(
    (review: {
      quizId: number;
      chatSessionId?: number | null;
      quizTitle: string;
      score: number;
      total: number;
      wrongCount: number;
    }) => {
      setPendingChatReview({ ...review, key: Date.now() });
      setActiveTab('assistente');
    },
    []
  );

  const handleLectureCreatedFromAssistant = useCallback(() => {
    void loadLectures(false);
  }, [loadLectures]);

  const [pendingQuadernoLectureId, setPendingQuadernoLectureId] = useState<number | null>(null);
  const handleOpenQuadernoLecture = useCallback(async (lectureId: number) => {
    await loadLectures(false);
    setPendingQuadernoLectureId(lectureId);
    setActiveTab('quaderno');
  }, [loadLectures]);

  const handleClearMaterialTarget = useCallback(() => setMaterialTarget(null), []);
  const handleClearAudioSnippetTarget = useCallback(() => setAudioSnippetTarget(null), []);
  const handleClearPendingQuiz = useCallback(() => setPendingQuizId(null), []);
  const handleClearPendingOralQuestion = useCallback(() => setPendingOralQuestion(null), []);
  const handleClearPendingChatReview = useCallback(() => setPendingChatReview(null), []);

  // Notifica al genitore lo stato di registrazione (blocca il cambio corso nella Sidebar)
  useEffect(() => {
    onRecordingChange?.(activeRecordingLecture !== null);
  }, [activeRecordingLecture, onRecordingChange]);

  // Prefetch proattivo del modulo di registrazione in background:
  // quando l'utente preme "Avvia", il componente è già in memoria e si monta a 0ms senza jank.
  useEffect(() => {
    void import('./LiveRecordingWorkspace');
  }, []);

  useEffect(() => {
    return () => onRecordingChange?.(false);
  }, [onRecordingChange]);

  const activeRecordingLectureRef = useRef<Lecture | null>(null);
  activeRecordingLectureRef.current = activeRecordingLecture;

  const pollUntilNotesReady = useCallback((lectureId: number) => {
    if (notesPollRef.current) clearInterval(notesPollRef.current);
    const startedAt = Date.now();
    // Usa llmDownloaded (NON modelsDownloaded che controlla solo Whisper!):
    // se nessun LLM è scaricato, fa solo 2 controlli rapidi (~12s) per l'audio M4A / coda finale.
    const hasLLM = Boolean(settings.llmDownloaded);
    const maxPollMs = hasLLM ? NOTES_POLL_MAX_MS : 12000;
    const isDone = (data: Lecture[] | null) => {
      const lec = data?.find((l) => l.id === lectureId);
      if (!lec) return false;
      if (!hasLLM) {
        return Boolean(lec.audio_path && lec.audio_path.endsWith('.m4a'));
      }
      return Boolean(lec.processed_notes && lec.audio_path);
    };
    notesPollRef.current = setInterval(async () => {
      const data = await loadLectures(false);
      if (isDone(data) || Date.now() - startedAt > maxPollMs) {
        if (notesPollRef.current) clearInterval(notesPollRef.current);
        notesPollRef.current = null;
      }
    }, NOTES_POLL_INTERVAL_MS);
  }, [loadLectures, settings.llmDownloaded]);

  const handleRecordingFinished = useCallback(() => {
    const finished = activeRecordingLectureRef.current;
    // Fase 1: richiude la Dynamic Island a 60/120 FPS e carica IN PARALLELO le lezioni aggiornate
    // così QuadernoView si monta già con raw_transcript, durata e audio_path freschi (zero flash!)
    setIsClosingLive(true);
    const preloadPromise = loadLectures(false).catch(() => null);
    window.setTimeout(async () => {
      const data = await Promise.race([
        preloadPromise,
        new Promise<Lecture[] | null>((resolve) => window.setTimeout(() => resolve(null), 350)),
      ]);
      if (!isMountedRef.current) return;
      // Fase 2: smonta il workspace live e rimonta il Quaderno con dati già aggiornati
      setActiveRecordingLecture(null);
      setIsClosingLive(false);
      if (!finished) return;
      const lec = data?.find((l) => l.id === finished.id);
      const needsPoll = settings.llmDownloaded
        ? !lec || !lec.processed_notes || !lec.audio_path
        : !lec || !lec.audio_path || !lec.audio_path.endsWith('.m4a');
      if (needsPoll) pollUntilNotesReady(finished.id);
    }, 190);
  }, [loadLectures, pollUntilNotesReady, settings.llmDownloaded]);

  const liveSession = useLiveSession({
    subject,
    lecture: activeRecordingLecture,
    onStop: handleRecordingFinished,
  });

  const handleStartLive = async () => {
    setStartingLive(true);
    setIsClosingLive(false);
    setLiveError(null);
    try {
      const today = new Date().toLocaleDateString('it-IT', { day: 'numeric', month: 'long' });
      const newLec = await api.lectures.create({
        subject_id: subject.id,
        title: `Lezione del ${today}`,
      });
      setActiveRecordingLecture(newLec);
      setLectures((prev) => [newLec, ...prev]);
    } catch (e: unknown) {
      console.error('Errore avvio lezione:', e);
      setLiveError('Impossibile avviare la lezione. Assicurati che il motore locale sia attivo.');
    } finally {
      setStartingLive(false);
    }
  };

  // In caso di errore la lezione resta in lista e l'errore viene propagato al chiamante
  const handleDeleteLecture = async (id: number) => {
    try {
      await api.lectures.delete(id);
      setLectures((prev) => prev.filter((l) => l.id !== id));
    } catch (e) {
      console.error('Impossibile eliminare la lezione:', e);
      throw e;
    }
  };

  const handleLocalLectureUpdate = useCallback((id: number, updater: (lecture: Lecture) => Lecture) => {
    setLectures((prev) => prev.map((l) => (l.id === id ? updater(l) : l)));
  }, []);

  const handleUpdateLecture = async (
    id: number,
    data: Partial<{
      title: string;
      processed_notes: string;
      raw_transcript: string;
      date: string;
      detected_questions: string | null;
    }>
  ) => {
    try {
      const updated = await api.lectures.update(id, data);
      setLectures((prev) => prev.map((l) => (l.id === id ? updated : l)));
      return updated;
    } catch (e: unknown) {
      console.error('Errore aggiornamento lezione:', e);
      throw e;
    }
  };

  const handleCreateCustomLecture = async (data: {
    title: string;
    date?: string;
    processed_notes?: string;
    raw_transcript?: string;
    duration_seconds?: number;
    source_type?: string;
  }) => {
    try {
      const created = await api.lectures.create({
        subject_id: subject.id,
        source_type: data.source_type || 'imported',
        ...data,
      });
      setLectures((prev) => [created, ...prev]);
      return created;
    } catch (e: unknown) {
      console.error('Errore creazione lezione:', e);
      throw e;
    }
  };

  const handleRemoveFromQuadernoOnly = async (id: number) => {
    try {
      const updated = await api.lectures.update(id, { in_quaderno: false });
      setLectures((prev) => prev.map((l) => (l.id === id ? updated : l)));
    } catch (e) {
      console.error('Errore rimozione dal quaderno:', e);
      throw e;
    }
  };

  const handleRestoreToQuaderno = async (id: number) => {
    try {
      const updated = await api.lectures.update(id, { in_quaderno: true });
      setLectures((prev) => prev.map((l) => (l.id === id ? updated : l)));
    } catch (e) {
      console.error('Errore ripristino nel quaderno:', e);
      throw e;
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden select-none [contain:paint]">
      {/* ── Apple Segmented Control Top Bar (Tauri Drag Region) ── */}
      <header
        data-tauri-drag-region
        className="h-14 px-6 border-b border-white/[0.08] liquid-glass flex items-center justify-between gap-4 flex-shrink-0 z-20 relative"
      >
        {/* Left: Minimal course indicator or live recording lecture title */}
        <div className="flex-1 min-w-0 flex items-center gap-2" data-tauri-drag-region="false">
          {isCapsuleExpanded && activeRecordingLecture ? (
            <div className="flex items-center gap-2 text-xs font-medium text-white/85 animate-fade-in truncate">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse flex-shrink-0" />
              <span className="truncate">{activeRecordingLecture.title}</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 min-w-0 animate-fade-in">
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: subject.color }}
              />
              <span className="text-xs font-medium text-white/55 truncate">
                {subject.name}
              </span>
            </div>
          )}
        </div>

        {/* Center: Apple-style Liquid Glass Segmented Control — collassa durante la registrazione così la Dynamic Island ha tutto lo spazio */}
        <div 
          data-tauri-drag-region="false"
          className={clsx(
            'flex justify-center items-center h-full',
            !isCapsuleExpanded ? 'flex-shrink-0 w-full max-w-[440px]' : 'w-0 overflow-hidden'
          )}
        >
          <AnimatePresence mode="wait" initial={false}>
            {!isCapsuleExpanded ? (
              <motion.div
                key="segmented-tabs"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ duration: 0.18, ease: [0.16, 1, 0.3, 1] }}
                className="w-full"
              >
                <SegmentedControl
                  options={TABS}
                  value={activeTab}
                  onChange={setActiveTab}
                  layoutId="subject-dashboard-active-tab"
                  fullWidth
                  size="md"
                />
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        {/* Right: Dynamic Recording Controller — iOS Dynamic Island morphing capsule (fluida 60fps senza blur annidato) */}
        <div
          data-tauri-drag-region="false"
          className={clsx(
            'flex items-center justify-end z-30',
            isCapsuleExpanded ? 'flex-shrink-0' : 'flex-1 min-w-0'
          )}
        >
          <motion.div
            layout
            transition={{
              type: 'spring',
              stiffness: 420,
              damping: 32,
              mass: 0.8,
            }}
            className={clsx(
              'relative h-9 rounded-full flex items-center select-none flex-shrink-0 transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40',
              !isCapsuleExpanded
                ? 'bg-white/[0.08] hover:bg-white/[0.14] border border-white/20 hover:border-white/30 cursor-pointer shadow-liquid-sm'
                : 'bg-[#0d0f17]/95 border border-white/15 shadow-liquid-lg'
            )}
            role={!activeRecordingLecture ? 'button' : undefined}
            tabIndex={!activeRecordingLecture ? 0 : undefined}
            aria-label={!activeRecordingLecture ? 'Registra lezione' : undefined}
            onKeyDown={(e) => {
              if (!activeRecordingLecture && !startingLive && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                handleStartLive();
              }
            }}
            onClick={!activeRecordingLecture && !startingLive ? handleStartLive : undefined}
            whileHover={!isCapsuleExpanded ? { scale: 1.03 } : undefined}
            whileTap={!isCapsuleExpanded ? { scale: 0.97 } : undefined}
          >
            <AnimatePresence mode="popLayout" initial={false}>
              {!isCapsuleExpanded ? (
                <motion.div
                  key="mic-bubble"
                  initial={{ opacity: 0, scale: 0.88 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.88 }}
                  transition={{ duration: 0.16, ease: 'easeOut' }}
                  className="h-9 px-3.5 gap-1.5 flex items-center justify-center flex-shrink-0 whitespace-nowrap"
                  title="Registra Lezione"
                >
                  {startingLive ? (
                    <Loader2 size={14} className="animate-spin text-white/60" />
                  ) : (
                    <>
                      <Mic size={13} strokeWidth={2.2} className="text-red-400" />
                      <span className="text-[11.5px] font-medium text-white/90">Registra</span>
                    </>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="player-expanded"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.92 }}
                  transition={{ duration: 0.16, ease: 'easeOut' }}
                  className="flex items-center gap-1.5 px-1.5 py-1 whitespace-nowrap"
                >
                  {/* Waveform indicator + Timer */}
                  <div className="h-7 flex items-center gap-2 pl-2 pr-2 border-r border-white/10">
                    <div className="flex items-center gap-0.5 h-3.5">
                      <span
                        className={clsx(
                          'w-0.5 bg-red-400 rounded-full transition-all',
                          liveSession.isConnected && !liveSession.isPaused && liveSession.elapsed > 0
                            ? 'animate-wave'
                            : 'h-2 opacity-40'
                        )}
                        style={{ height: '8px', animationDelay: '0ms' }}
                      />
                      <span
                        className={clsx(
                          'w-0.5 bg-red-400 rounded-full transition-all',
                          liveSession.isConnected && !liveSession.isPaused && liveSession.elapsed > 0
                            ? 'animate-wave'
                            : 'h-3.5 opacity-40'
                        )}
                        style={{ height: '14px', animationDelay: '180ms' }}
                      />
                      <span
                        className={clsx(
                          'w-0.5 bg-red-400 rounded-full transition-all',
                          liveSession.isConnected && !liveSession.isPaused && liveSession.elapsed > 0
                            ? 'animate-wave'
                            : 'h-2 opacity-40'
                        )}
                        style={{ height: '8px', animationDelay: '360ms' }}
                      />
                    </div>
                    <span className="text-[12px] font-mono font-semibold text-white/95 tracking-wide tabular-nums">
                      {formatElapsed(liveSession.elapsed)}
                    </span>
                  </div>

                  {/* Quick Bookmark (⌘⇧B / Ctrl+Shift+B) Button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      liveSession.handleBookmark();
                    }}
                    title={`Segnalibro Esame (${shortcutLabel('B', true)})`}
                    className={clsx(
                      'h-7 flex items-center gap-1.5 px-2.5 rounded-full text-[11.5px] font-medium transition-all border cursor-pointer select-none',
                      liveSession.lastBookmarkPinned && Date.now() - liveSession.lastBookmarkPinned < 2000
                        ? 'bg-amber-500/25 border-amber-500/50 text-amber-200 scale-105'
                        : 'bg-white/[0.08] hover:bg-white/[0.14] border-white/[0.10] text-zinc-200 hover:text-white'
                    )}
                  >
                    <Bookmark size={11} strokeWidth={2} className="text-amber-400 fill-amber-400/20" />
                    <span>Segnalibro</span>
                    {liveSession.bookmarks.length > 0 && (
                      <span className="text-[10px] font-mono bg-amber-400/25 text-amber-300 px-1.5 rounded-full">
                        {liveSession.bookmarks.length}
                      </span>
                    )}
                  </button>

                  {/* Pause / Resume */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      liveSession.togglePause();
                    }}
                    title={liveSession.isPaused ? 'Riprendi Registrazione' : 'Metti in Pausa (Calore Zero)'}
                    className={clsx(
                      'h-7 px-2.5 rounded-full flex items-center gap-1.5 border transition-all cursor-pointer text-[11.5px] font-medium shadow-sm select-none',
                      liveSession.isPaused
                        ? 'bg-amber-500/25 border-amber-500/40 text-amber-200 hover:bg-amber-500/35 shadow-liquid-sm animate-pulse'
                        : 'bg-white/[0.08] hover:bg-white/[0.14] border-white/[0.10] text-zinc-200 hover:text-white'
                    )}
                  >
                    {liveSession.isPaused ? (
                      <>
                        <Play size={11} strokeWidth={2.5} className="ml-0.5 fill-current" />
                        <span>Riprendi</span>
                      </>
                    ) : (
                      <>
                        <Pause size={11} strokeWidth={2.5} />
                        <span>Pausa</span>
                      </>
                    )}
                  </button>

                  {/* Termina */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      liveSession.handleStop();
                    }}
                    disabled={liveSession.isStopping}
                    title="Termina e salva la registrazione"
                    className="h-7 flex items-center gap-1.5 px-3 rounded-full bg-red-500/20 hover:bg-red-500/30 text-red-200 hover:text-white text-[11.5px] font-medium border border-red-500/30 shadow-liquid-sm transition-all cursor-pointer disabled:opacity-60 disabled:cursor-wait select-none"
                  >
                    {liveSession.isStopping ? (
                      <Loader2 size={11} className="animate-spin text-red-300" />
                    ) : (
                      <Square size={10} strokeWidth={2.5} className="fill-current text-red-400" />
                    )}
                    <span>{liveSession.isStopping ? 'Salvataggio...' : 'Termina'}</span>
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </header>

      {/* ── Banner Modelli IA da scaricare (Risparmio Dati) ── */}
      {!settings.modelsDownloaded && settings.whisperModel === 'mock_mode' && !activeRecordingLecture && (
        <div 
          onClick={onOpenSettings}
          className="px-6 py-2 bg-amber-500/10 border-b border-amber-500/20 flex items-center justify-between text-xs text-amber-200/90 cursor-pointer hover:bg-amber-500/15 transition-all flex-shrink-0 group"
        >
          <div className="flex items-center gap-2.5">
            <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
            <span className="font-semibold text-amber-300">Modelli non scaricati</span>
          </div>
          <button 
            type="button"
            className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/30 text-[11px] font-semibold text-amber-200 transition-all flex-shrink-0 shadow-liquid-sm"
          >
            <Download size={11} />
            <span>Scarica Modelli</span>
          </button>
        </div>
      )}

      {/* ── Banner errore avvio registrazione live ── */}
      {liveError && !activeRecordingLecture && (
        <div className="px-6 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center justify-between text-xs text-red-200/90 flex-shrink-0 animate-fade-in">
          <div className="flex items-center gap-2.5">
            <AlertTriangle size={14} className="text-red-400 flex-shrink-0" />
            <span>{liveError}</span>
          </div>
          <button
            type="button"
            onClick={() => setLiveError(null)}
            className="text-red-300/60 hover:text-red-200 text-xs px-2 py-0.5 rounded cursor-pointer"
          >
            Chiudi
          </button>
        </div>
      )}

      {/* ── Banner stato connessione registrazione live ── */}
      {activeRecordingLecture && !isClosingLive && !liveSession.isStopping && (liveSession.connectionError || !liveSession.isConnected || liveSession.isReconnecting) && (
        <div
          className={clsx(
            'px-6 py-2 border-b flex items-start gap-2.5 text-xs flex-shrink-0 animate-fade-in',
            liveSession.connectionError
              ? 'bg-red-500/10 border-red-500/20 text-red-200/90'
              : 'bg-amber-500/10 border-amber-500/20 text-amber-200/90'
          )}
        >
          {liveSession.connectionError ? (
            <AlertTriangle size={14} className="text-red-400 flex-shrink-0 mt-0.5" />
          ) : (
            <Loader2 size={13} className="animate-spin text-amber-400 flex-shrink-0 mt-0.5" />
          )}
          <span className="flex-1 min-w-0 leading-snug break-words select-text">
            {liveSession.connectionError
              ? `Registrazione non attiva: ${liveSession.connectionError}`
              : liveSession.isReconnecting
              ? 'Riconnessione al motore di registrazione in corso...'
              : 'Connessione al motore di registrazione in corso...'}
          </span>
        </div>
      )}

      {/* ── Active View / Live Recording Workspace ── */}
      <main className="flex-1 flex overflow-hidden relative w-full h-full" style={{ transform: 'translateZ(0)' }}>
        {activeRecordingLecture ? (
          <Suspense fallback={<LiveWorkspaceSkeleton />}>
            <LiveRecordingWorkspace
              subjectId={subject.id}
              subjectName={subject.name}
              chunks={liveSession.chunks}
              isPaused={liveSession.isPaused}
              audioStatus={liveSession.audioStatus}
              detectedQuestions={liveSession.detectedQuestions}
              onDeleteQuestion={liveSession.handleDeleteQuestion}
              onAnswerQuestion={liveSession.handleAnswerQuestion}
              answeringQuestionIds={liveSession.answeringQuestionIds}
              chatMessages={liveSession.chatMessages}
              chatInput={liveSession.chatInput}
              setChatInput={liveSession.setChatInput}
              isSendingChat={liveSession.isSendingChat}
              handleChatSend={liveSession.handleChatSend}
              handleCancelChat={liveSession.handleCancelChat}
              inspectorTab={liveSession.inspectorTab}
              setInspectorTab={liveSession.setInspectorTab}
              onOpenSettings={onOpenSettings}
            />
          </Suspense>
        ) : loading || showLecturesSkeleton ? (
          showLecturesSkeleton ? (
            <QuadernoSkeleton />
          ) : (
            <div className="flex-1 flex w-full h-full" />
          )
        ) : (
          <div className="flex-1 flex overflow-hidden w-full h-full relative">
            {hasMountedQuaderno && (
              <div
                className={
                  activeTab === 'quaderno'
                    ? 'flex-1 flex overflow-hidden w-full h-full animate-fade-in'
                    : 'hidden'
                }
                aria-hidden={activeTab !== 'quaderno'}
              >
                <QuadernoView
                  key={subject.id}
                  isActive={activeTab === 'quaderno'}
                  subject={subject}
                  allSubjects={allSubjects}
                  lectures={quadernoLectures}
                  onMoveLecture={handleMoveLecture}
                  initialLectureId={pendingQuadernoLectureId}
                  onClearInitialLectureId={() => setPendingQuadernoLectureId(null)}
                  initialAudioTarget={audioSnippetTarget}
                  onClearInitialAudioTarget={handleClearAudioSnippetTarget}
                  onOpenAudioSnippet={handleOpenAudioSnippet}
                  onOpenCitation={handleOpenCitation}
                  onDeletePermanently={handleDeleteLecture}
                  onRemoveFromQuadernoOnly={handleRemoveFromQuadernoOnly}
                  onRestoreToQuaderno={handleRestoreToQuaderno}
                  onAddLecture={handleCreateCustomLecture}
                  onUpdateLecture={handleUpdateLecture}
                  onStartLive={handleStartLive}
                  onOpenAddMaterialModal={() => setActiveTab('materiali')}
                  onOpenSettings={onOpenSettings}
                  onLocalLectureUpdate={handleLocalLectureUpdate}
                />
              </div>
            )}

            {activeTab !== 'quaderno' && (
              <div key={activeTab} className="flex-1 flex overflow-hidden w-full h-full animate-fade-in">
                {activeTab === 'materiali' && (
                  <Suspense
                    fallback={
                      <div className="flex-1 flex flex-col overflow-hidden bg-black/15 select-none animate-fade-in">
                        <div className="px-8 py-3.5 border-b border-white/[0.06] flex items-center justify-between flex-shrink-0">
                          <div className="flex items-center gap-3">
                            <SkeletonPill tone="soft" className="h-7 w-52" />
                            <SkeletonPill tone="soft" className="h-3 w-24" />
                          </div>
                          <SkeletonPill tone="strong" className="h-7 w-24" />
                        </div>
                        <div className="flex-1 overflow-hidden p-8">
                          <MaterialiGridSkeleton />
                        </div>
                      </div>
                    }
                  >
                    <MaterialiView
                      subject={subject}
                      key={subject.id}
                      initialTarget={materialTarget}
                      onClearInitialTarget={handleClearMaterialTarget}
                      onDeleteLecturePermanently={handleDeleteLecture}
                      onRestoreLectureToQuaderno={handleRestoreToQuaderno}
                      onLectureCreated={(newLecture) => {
                        setLectures((prev) => {
                          const exists = prev.some((l) => l.id === newLecture.id);
                          if (exists) return prev.map((l) => (l.id === newLecture.id ? newLecture : l));
                          return [newLecture, ...prev];
                        });
                      }}
                    />
                  </Suspense>
                )}
                {activeTab === 'quiz' && (
                  <ErrorBoundary fallbackTitle="Impossibile caricare la schermata Quiz">
                    <Suspense fallback={<QuizWizardSkeleton />}>
                      <QuizRipassoView
                        key={subject.id}
                        subject={subject}
                        lectures={lectures}
                        onOpenSettings={onOpenSettings}
                        initialQuizId={pendingQuizId}
                        onClearInitialQuiz={handleClearPendingQuiz}
                        initialOralQuestion={pendingOralQuestion}
                        onClearInitialOralQuestion={handleClearPendingOralQuestion}
                        onReviewInChat={handleReviewInChat}
                      />
                    </Suspense>
                  </ErrorBoundary>
                )}
                {activeTab === 'assistente' && (
                  <ErrorBoundary fallbackTitle="Impossibile caricare l'Assistente">
                    <Suspense
                      fallback={
                        <div className="flex-1 overflow-hidden content-canvas px-6 sm:px-8 py-6">
                          <div className="max-w-3xl mx-auto">
                            <ChatStreamSkeleton />
                          </div>
                        </div>
                      }
                    >
                      <AssistenteCorsoView
                        key={subject.id}
                        subject={subject}
                        lectures={lectures}
                        onOpenCitation={handleOpenCitation}
                        onOpenAudioSnippet={handleOpenAudioSnippet}
                        onOpenSettings={onOpenSettings}
                        onOpenQuiz={handleOpenQuiz}
                        onOpenOralQuestion={handleOpenOralQuestion}
                        onOpenQuadernoLecture={handleOpenQuadernoLecture}
                        onLectureCreated={handleLectureCreatedFromAssistant}
                        initialReviewTarget={pendingChatReview}
                        onClearInitialReview={handleClearPendingChatReview}
                      />
                    </Suspense>
                  </ErrorBoundary>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
