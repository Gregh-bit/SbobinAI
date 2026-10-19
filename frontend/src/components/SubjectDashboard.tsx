// components/SubjectDashboard.tsx — Dashboard Corso con Segmented Control e viste
import { useState, useEffect, useCallback, useRef, lazy, Suspense } from 'react';
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

// Lazy loading viste secondarie e live workspace per apertura istantanea (<100ms)
const MaterialiView = lazy(() => import('./MaterialiView'));
const QuizRipassoView = lazy(() => import('./QuizRipassoView'));
const AssistenteCorsoView = lazy(() => import('./AssistenteCorsoView'));
const LiveRecordingWorkspace = lazy(() => import('./LiveRecordingWorkspace'));

interface Props {
  subject: Subject;
  onStartLecture?: (lecture: Lecture) => void;
  onOpenSettings?: () => void;
  onRecordingChange?: (isRecording: boolean) => void;
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

export default function SubjectDashboard({ subject, onOpenSettings, onRecordingChange }: Props) {
  const { settings } = useSettings();
  const [activeTab, setActiveTab] = useState<CourseTab>('quaderno');
  const [lectures, setLectures] = useState<Lecture[]>([]);
  const [loading, setLoading] = useState(true);
  const [startingLive, setStartingLive] = useState(false);
  const [activeRecordingLecture, setActiveRecordingLecture] = useState<Lecture | null>(null);

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
    if (isInitial) setLoading(true);
    try {
      const data = await api.lectures.listBySubject(subject.id);
      if (!isMountedRef.current || seq !== loadSeqRef.current) return null;
      setLectures(data);
      return data;
    } catch {
      // Ignora silenziosamente
      return null;
    } finally {
      if (isMountedRef.current && seq === loadSeqRef.current) setLoading(false);
    }
  }, [subject.id]);

  useEffect(() => {
    loadLectures(true);
  }, [loadLectures]);

  // Notifica al genitore lo stato di registrazione (blocca il cambio corso nella Sidebar)
  useEffect(() => {
    onRecordingChange?.(activeRecordingLecture !== null);
  }, [activeRecordingLecture, onRecordingChange]);

  useEffect(() => {
    return () => onRecordingChange?.(false);
  }, [onRecordingChange]);

  const activeRecordingLectureRef = useRef<Lecture | null>(null);
  activeRecordingLectureRef.current = activeRecordingLecture;

  const pollUntilNotesReady = useCallback((lectureId: number) => {
    if (notesPollRef.current) clearInterval(notesPollRef.current);
    const startedAt = Date.now();
    const isDone = (data: Lecture[] | null) => {
      const lec = data?.find((l) => l.id === lectureId);
      if (!lec) return false;
      // Il backend genera gli appunti solo se la trascrizione è sufficientemente lunga
      return Boolean(lec.processed_notes) || (lec.raw_transcript || '').trim().length <= 30;
    };
    notesPollRef.current = setInterval(async () => {
      const data = await loadLectures(false);
      if (isDone(data) || Date.now() - startedAt > NOTES_POLL_MAX_MS) {
        if (notesPollRef.current) clearInterval(notesPollRef.current);
        notesPollRef.current = null;
      }
    }, NOTES_POLL_INTERVAL_MS);
  }, [loadLectures]);

  const handleRecordingFinished = useCallback(() => {
    const finished = activeRecordingLectureRef.current;
    setActiveRecordingLecture(null);
    loadLectures(false).then((data) => {
      if (!finished || !isMountedRef.current) return;
      const lec = data?.find((l) => l.id === finished.id);
      if (!lec || !lec.processed_notes) pollUntilNotesReady(finished.id);
    });
  }, [loadLectures, pollUntilNotesReady]);

  const liveSession = useLiveSession({
    subject,
    lecture: activeRecordingLecture,
    onStop: handleRecordingFinished,
  });

  const handleStartLive = async () => {
    setStartingLive(true);
    try {
      const today = new Date().toLocaleDateString('it-IT', { day: 'numeric', month: 'long' });
      const newLec = await api.lectures.create({
        subject_id: subject.id,
        title: `Lezione del ${today}`,
      });
      setActiveRecordingLecture(newLec);
    } catch (e: unknown) {
      console.error('Errore avvio lezione:', e);
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
    }
  };

  const handleRestoreToQuaderno = async (id: number) => {
    try {
      const updated = await api.lectures.update(id, { in_quaderno: true });
      setLectures((prev) => prev.map((l) => (l.id === id ? updated : l)));
    } catch (e) {
      console.error('Errore ripristino nel quaderno:', e);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden select-none [contain:paint]">
      {/* ── Apple Segmented Control Top Bar (Tauri Drag Region) ── */}
      <header
        data-tauri-drag-region
        className="h-14 px-6 border-b border-white/[0.08] liquid-glass flex items-center justify-between gap-4 flex-shrink-0 z-20 relative"
      >
        {/* Left: Clean space or live recording lecture title */}
        <div className="flex-1 min-w-0 flex items-center gap-2" data-tauri-drag-region="false">
          {activeRecordingLecture ? (
            <div className="flex items-center gap-2 text-xs font-medium text-white/85 animate-fade-in truncate">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse flex-shrink-0" />
              <span className="truncate">{activeRecordingLecture.title}</span>
            </div>
          ) : null}
        </div>

        {/* Center: Apple-style Liquid Glass Segmented Control — Centrato in modo responsive */}
        {!activeRecordingLecture && (
          <div 
            data-tauri-drag-region="false"
            className="flex-shrink-0 w-full max-w-[440px] animate-fade-in"
          >
            <SegmentedControl
              options={TABS}
              value={activeTab}
              onChange={setActiveTab}
              layoutId="subject-dashboard-active-tab"
              fullWidth
              size="md"
            />
          </div>
        )}

        {/* Right: Dynamic Recording Controller — iOS Dynamic Island morphing capsule */}
        <div data-tauri-drag-region="false" className={clsx('flex items-center justify-end z-30', activeRecordingLecture ? 'flex-1' : 'flex-1 min-w-0')}>
          <motion.div
            layout
            transition={{
              type: 'spring',
              stiffness: 360,
              damping: 26,
              mass: 0.8,
            }}
            className={clsx(
              'relative h-8 rounded-full overflow-hidden flex items-center select-none shadow-liquid-sm transition-colors duration-200',
              !activeRecordingLecture
                ? 'w-8 min-w-[32px] justify-center liquid-glass-pill border border-white/[0.22] hover:border-white/[0.38] hover:bg-white/[0.12] cursor-pointer'
                : 'w-auto liquid-capsule border border-white/[0.18] shadow-liquid-lg'
            )}
            onClick={!activeRecordingLecture && !startingLive ? handleStartLive : undefined}
            whileHover={!activeRecordingLecture ? { scale: 1.06 } : undefined}
            whileTap={!activeRecordingLecture ? { scale: 0.94 } : undefined}
          >
            <AnimatePresence mode="popLayout" initial={false}>
              {!activeRecordingLecture ? (
                <motion.div
                  key="mic-bubble"
                  layout="position"
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.5 }}
                  transition={{ duration: 0.15 }}
                  className="w-8 h-8 flex items-center justify-center flex-shrink-0"
                  title="Registra Lezione"
                >
                  {startingLive ? (
                    <Loader2 size={13} className="animate-spin text-white/60" />
                  ) : (
                    <Mic size={14} strokeWidth={2} className="text-red-400" />
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="player-expanded"
                  layout="position"
                  initial={{ opacity: 0, filter: 'blur(4px)' }}
                  animate={{ opacity: 1, filter: 'blur(0px)' }}
                  exit={{ opacity: 0, filter: 'blur(4px)' }}
                  transition={{ duration: 0.22, delay: 0.06 }}
                  className="flex items-center gap-2 p-1 pl-3 pr-1.5 whitespace-nowrap"
                >
                  {/* Waveform indicator + Timer */}
                  <div className="flex items-center gap-2 font-mono text-[11px] text-white/80 pr-2 border-r border-white/[0.12]">
                    <div className="flex items-center gap-0.5 h-3">
                      <span className={clsx('w-0.5 bg-red-400 rounded-full', !liveSession.isPaused && 'animate-wave')} style={{ height: '8px', animationDelay: '0ms' }} />
                      <span className={clsx('w-0.5 bg-red-400 rounded-full', !liveSession.isPaused && 'animate-wave')} style={{ height: '12px', animationDelay: '200ms' }} />
                      <span className={clsx('w-0.5 bg-red-400 rounded-full', !liveSession.isPaused && 'animate-wave')} style={{ height: '7px', animationDelay: '400ms' }} />
                    </div>
                    <span className="font-semibold tracking-tight">{formatElapsed(liveSession.elapsed)}</span>
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
                      'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium transition-all border cursor-pointer',
                      liveSession.lastBookmarkPinned && Date.now() - liveSession.lastBookmarkPinned < 2000
                        ? 'bg-amber-500/25 border-amber-500/50 text-amber-200 scale-105'
                        : 'bg-white/[0.08] hover:bg-white/[0.16] border-white/[0.10] text-white/90'
                    )}
                  >
                    <Bookmark size={11} strokeWidth={2} className="text-amber-400 fill-amber-400/20" />
                    <span>Segnalibro</span>
                    <span className="text-[9px] font-mono text-white/40 ml-0.5">{shortcutLabel('B', true)}</span>
                    {liveSession.bookmarks.length > 0 && (
                      <span className="text-[9px] font-mono bg-amber-400/25 text-amber-300 px-1 rounded-full">
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
                      "h-6.5 px-2.5 rounded-full flex items-center gap-1.5 border transition-all cursor-pointer text-[11px] font-medium shadow-sm",
                      liveSession.isPaused
                        ? "bg-amber-500/25 border-amber-500/40 text-amber-200 hover:bg-amber-500/35 shadow-liquid-sm animate-pulse"
                        : "bg-white/[0.08] hover:bg-white/[0.16] border-white/[0.10] text-white/90"
                    )}
                  >
                    {liveSession.isPaused ? (
                      <>
                        <Play size={10} strokeWidth={2.5} className="ml-0.5 fill-current" />
                        <span>In Pausa</span>
                      </>
                    ) : (
                      <>
                        <Pause size={10} strokeWidth={2.5} />
                        <span>Pausa</span>
                      </>
                    )}
                  </button>

                  {/* Termina e Rielabora */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      liveSession.handleStop();
                    }}
                    disabled={liveSession.isStopping}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.18] hover:bg-white/[0.28] text-white text-[11px] font-semibold border border-white/[0.22] shadow-liquid-sm transition-all cursor-pointer disabled:opacity-60 disabled:cursor-wait"
                  >
                    {liveSession.isStopping ? (
                      <Loader2 size={10} className="animate-spin" />
                    ) : (
                      <Square size={9} strokeWidth={2.5} className="fill-white" />
                    )}
                    <span>{liveSession.isStopping ? 'Salvataggio...' : 'Termina e Rielabora'}</span>
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

      {/* ── Banner stato connessione registrazione live ── */}
      {activeRecordingLecture && !liveSession.isStopping && (liveSession.connectionError || !liveSession.isConnected) && (
        <div
          className={clsx(
            'px-6 py-2 border-b flex items-center gap-2.5 text-xs flex-shrink-0',
            liveSession.connectionError
              ? 'bg-red-500/10 border-red-500/20 text-red-200/90'
              : 'bg-white/[0.04] border-white/[0.08] text-white/60'
          )}
        >
          {liveSession.connectionError ? (
            <AlertTriangle size={14} className="text-red-400 flex-shrink-0" />
          ) : (
            <Loader2 size={13} className="animate-spin text-white/50 flex-shrink-0" />
          )}
          <span className="truncate">
            {liveSession.connectionError
              ? `Registrazione non attiva: ${liveSession.connectionError}`
              : 'Connessione al motore di registrazione in corso...'}
          </span>
        </div>
      )}

      {/* ── Active View / Live Recording Workspace ── */}
      <main className="flex-1 flex overflow-hidden relative w-full h-full">
        {activeRecordingLecture ? (
          <Suspense fallback={<div className="flex-1 flex items-center justify-center w-full h-full"><Loader2 size={24} className="animate-spin text-white/30" /></div>}>
            <LiveRecordingWorkspace
              chunks={liveSession.chunks}
              isPaused={liveSession.isPaused}
              audioStatus={liveSession.audioStatus}
              detectedQuestions={liveSession.detectedQuestions}
              onDeleteQuestion={liveSession.handleDeleteQuestion}
              chatMessages={liveSession.chatMessages}
              chatInput={liveSession.chatInput}
              setChatInput={liveSession.setChatInput}
              isSendingChat={liveSession.isSendingChat}
              handleChatSend={liveSession.handleChatSend}
              handleCancelChat={liveSession.handleCancelChat}
              inspectorTab={liveSession.inspectorTab}
              setInspectorTab={liveSession.setInspectorTab}
            />
          </Suspense>
        ) : loading ? (
          <div className="flex-1 flex items-center justify-center w-full h-full">
            <Loader2 size={20} className="animate-spin text-white/30" />
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              transition={{ duration: 0.15 }}
              className="flex-1 flex overflow-hidden w-full h-full"
            >
              {activeTab === 'quaderno' && (
                <QuadernoView
                  key={subject.id}
                  subject={subject}
                  lectures={lectures.filter((l) => l.in_quaderno !== false)}
                  onDeletePermanently={handleDeleteLecture}
                  onRemoveFromQuadernoOnly={handleRemoveFromQuadernoOnly}
                  onRestoreToQuaderno={handleRestoreToQuaderno}
                  onAddLecture={handleCreateCustomLecture}
                  onUpdateLecture={handleUpdateLecture}
                  onOpenAddMaterialModal={() => setActiveTab('materiali')}
                  onOpenSettings={onOpenSettings}
                  onLocalLectureUpdate={handleLocalLectureUpdate}
                />
              )}
              {activeTab === 'materiali' && (
                <Suspense fallback={<div className="flex-1 flex items-center justify-center"><Loader2 size={24} className="animate-spin text-white/30" /></div>}>
                  <MaterialiView 
                    subject={subject} 
                    key={subject.id}
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
                <Suspense fallback={<div className="flex-1 flex items-center justify-center"><Loader2 size={24} className="animate-spin text-white/30" /></div>}>
                  <QuizRipassoView
                    key={subject.id}
                    subject={subject}
                    lectures={lectures}
                    onOpenSettings={onOpenSettings}
                  />
                </Suspense>
              )}
              {activeTab === 'assistente' && (
                <Suspense fallback={<div className="flex-1 flex items-center justify-center"><Loader2 size={24} className="animate-spin text-white/30" /></div>}>
                  <AssistenteCorsoView key={subject.id} subject={subject} />
                </Suspense>
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}
