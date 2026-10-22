// components/QuizRipassoView.tsx — Sezione Quiz & Ripasso Accademico (Wizard a Schermate + Kill Processo IA)
import { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import { 
  HelpCircle, Mic, CheckCircle2, XCircle, ArrowRight, 
  RotateCcw, AlertCircle, Loader2, 
  BookOpen, Award, ArrowLeft, RefreshCw,
  Settings, Cpu, Check, StopCircle, ShieldAlert
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import type { Lecture, ModelsStatusResponse, OralEvaluationResult, QuizQuestion, Subject } from '../types';

interface Props {
  subject: Subject;
  lectures?: Lecture[];
  onOpenSettings?: () => void;
}

export default function QuizRipassoView({ subject, lectures = [], onOpenSettings }: Props) {
  const { settings } = useSettings();

  // ── 0. Verifica Reale Disponibilità Modello LLM ──────────────────────────────
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [checkingModels, setCheckingModels] = useState<boolean>(true);

  const fetchModelsStatus = () => {
    setCheckingModels(true);
    api.models.status()
      .then((st) => setModelsStatus(st))
      .catch((err) => console.warn('[Quiz] Verifica modelli fallita:', err))
      .finally(() => setCheckingModels(false));
  };

  useEffect(() => {
    fetchModelsStatus();
  }, [settings.modelsDownloaded, settings.llmModel]);

  const hasInstalledLLM = useMemo(() => {
    if (modelsStatus) {
      const hasDownloadedLlm = modelsStatus.llm?.some((m) => m.downloaded || m.status === 'completed');
      if (hasDownloadedLlm) return true;
      const hasInstalledSummary = modelsStatus.installed_models?.some((m) => m.type === 'llm');
      if (hasInstalledSummary) return true;
      if (modelsStatus.llm && modelsStatus.llm.length > 0) return false;
    }
    return Boolean(settings.modelsDownloaded);
  }, [modelsStatus, settings.modelsDownloaded]);

  // ── Wizard Navigation State ─────────────────────────────────────────────────
  // wizardStep: 1 = Selezione Lezioni, 2 = Scelta Modalità (Quiz vs Orale)
  const [wizardStep, setWizardStep] = useState<1 | 2>(1);
  const [slideDirection, setSlideDirection] = useState<'forward' | 'backward'>('forward');
  const [selectedLectureIds, setSelectedLectureIds] = useState<number[]>([]);

  const handleGoToStep2 = () => {
    if (selectedLectureIds.length === 0) return;
    setSlideDirection('forward');
    setWizardStep(2);
  };

  const handleBackToStep1 = () => {
    setSlideDirection('backward');
    setWizardStep(1);
  };

  // activeMode: 'none' (nel wizard), 'quiz' (quiz a risposta multipla), 'oral' (interrogazione vocale)
  const [activeMode, setActiveMode] = useState<'none' | 'quiz' | 'oral'>('none');

  // Kill Processo & Gestione Abort
  const abortControllerRef = useRef<AbortController | null>(null);
  const [isKillingProcess, setIsKillingProcess] = useState<boolean>(false);
  const [killNotice, setKillNotice] = useState<string | null>(null);

  // ── Stato Quiz ─────────────────────────────────────────────────────────────
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [selectedOpt, setSelectedOpt] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [isGeneratingQuiz, setIsGeneratingQuiz] = useState(false);
  const [quizError, setQuizError] = useState<string | null>(null);
  const [isQuizCompleted, setIsQuizCompleted] = useState(false);

  // ── Stato Orale ────────────────────────────────────────────────────────────
  const [oralQuestion, setOralQuestion] = useState<string>('');
  const [oralTopic, setOralTopic] = useState<string>('');
  const [isGeneratingOralQ, setIsGeneratingOralQ] = useState(false);
  const [isRecordingAnswer, setIsRecordingAnswer] = useState(false);
  const [recordingElapsed, setRecordingElapsed] = useState(0);
  const [isEvaluatingOral, setIsEvaluatingOral] = useState(false);
  const [oralResult, setOralResult] = useState<OralEvaluationResult | null>(null);
  const [oralError, setOralError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<any>(null);

  // Modello LLM configurato
  const llmModelId = settings.llmModel || settings.ollamaModel || undefined;

  // Pre-seleziona tutte le lezioni al primo caricamento
  useEffect(() => {
    if (lectures.length > 0 && selectedLectureIds.length === 0) {
      setSelectedLectureIds(lectures.map((l) => l.id));
    }
  }, [lectures]);

  const getValidLectureIds = (): number[] | undefined => {
    const valid = selectedLectureIds.filter((id) => lectures.some((l) => l.id === id));
    return valid.length > 0 ? valid : undefined;
  };

  // Cleanup globale al dismount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop();
          mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
        } catch {}
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // ── Toggle Selezione Lezioni (Schermata 1) ──────────────────────────────────
  const toggleLectureSelection = (id: number) => {
    setSelectedLectureIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAllLectures = () => {
    setSelectedLectureIds(lectures.map((l) => l.id));
  };

  const deselectAllLectures = () => {
    setSelectedLectureIds([]);
  };

  // ── Kill Processo IA & Eliminazione Cache (Richiesto) ───────────────────────
  const handleKillAIProcess = async () => {
    // 1. Reset IMMEDIATO e sincrono di tutta la UI per evitare qualsiasi flash di schermate intermedie
    setActiveMode('none');
    setIsGeneratingQuiz(false);
    setIsGeneratingOralQ(false);
    setIsEvaluatingOral(false);
    setIsRecordingAnswer(false);
    setIsQuizCompleted(false);
    setQuestions([]);
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setOralQuestion('');
    setOralResult(null);
    setQuizError(null);
    setOralError(null);
    setIsKillingProcess(true);

    // 2. Ferma microfono se in ascolto e timer all'istante
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      } catch {}
      mediaRecorderRef.current = null;
    }

    // 3. Abort controller immediato sul client
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {}
      abortControllerRef.current = null;
    }

    // 4. Chiamata al backend per stoppare l'inferenza e liberare cache Metal/GPU
    try {
      await api.study.abort(subject.id).catch(() => {});
      setKillNotice('Processo IA interrotto e cache Metal liberata.');
      setTimeout(() => setKillNotice(null), 3000);
    } finally {
      setIsKillingProcess(false);
    }
  };

  // ── 1. Avvio Quiz a Scelta Multipla con Animazione Espansione ───────────────
  const handleStartQuiz = () => {
    setQuizError(null);
    setActiveMode('quiz');
    executeQuizGeneration();
  };

  const executeQuizGeneration = async () => {
    setIsGeneratingQuiz(true);
    setQuizError(null);
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const res = await api.study.generateQuiz(
        subject.id,
        {
          lecture_ids: getValidLectureIds(),
          model_id: llmModelId,
        },
        controller.signal
      );

      if (res?.questions && res.questions.length > 0) {
        setQuestions(res.questions);
        setCurrentIdx(0);
        setSelectedOpt(null);
        setScore(0);
        setIsQuizCompleted(false);
      } else {
        throw new Error('Nessuna domanda restituita dal modello IA.');
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        // Interrotto intenzionalmente dall'utente: nessun errore da mostrare
        return;
      }
      setQuizError(err?.message || 'Errore durante la generazione del quiz con Qwen 3.5');
    } finally {
      setIsGeneratingQuiz(false);
      abortControllerRef.current = null;
    }
  };

  const handleSelectOption = (idx: number) => {
    if (selectedOpt !== null) return;
    setSelectedOpt(idx);
    const currQ = questions[currentIdx];
    const correct = currQ.correct_index ?? currQ.correctIndex ?? 0;
    if (idx === correct) {
      setScore((s) => s + 1);
    }
  };

  const handleNextQuestion = () => {
    if (currentIdx + 1 < questions.length) {
      setCurrentIdx((i) => i + 1);
      setSelectedOpt(null);
    } else {
      setIsQuizCompleted(true);
    }
  };

  const handleRestartQuiz = () => {
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setIsQuizCompleted(false);
  };

  // ── 2. Avvio Interrogazione Vocale con Animazione Espansione ────────────────
  const handleStartOralMode = () => {
    setOralResult(null);
    setOralError(null);
    setActiveMode('oral');
    if (!oralQuestion) {
      handleGenerateOralQuestion();
    }
  };

  const handleGenerateOralQuestion = async () => {
    setIsGeneratingOralQ(true);
    setOralError(null);
    setOralResult(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const res = await api.study.getOralQuestion(
        subject.id,
        {
          lecture_ids: getValidLectureIds(),
          model_id: llmModelId,
        },
        controller.signal
      );
      if (res?.question) {
        setOralQuestion(res.question);
        setOralTopic(res.topic || subject.name);
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setOralError(err?.message || "Errore durante la generazione della domanda d'esame.");
    } finally {
      setIsGeneratingOralQ(false);
      abortControllerRef.current = null;
    }
  };

  const handleStartVoiceRecording = async () => {
    setOralError(null);
    setOralResult(null);
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const activeStream = stream;
      audioChunksRef.current = [];
      const mimeType = ['audio/webm', 'audio/mp4'].find(
        (type) => typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(type)
      );
      const mediaRecorder = mimeType
        ? new MediaRecorder(activeStream, { mimeType })
        : new MediaRecorder(activeStream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorder.mimeType || mimeType || '' });
        activeStream.getTracks().forEach((track) => track.stop());
        mediaRecorderRef.current = null;
        await handleSendOralAudio(audioBlob);
      };

      mediaRecorder.start(250);
      setIsRecordingAnswer(true);
      setRecordingElapsed(0);

      timerIntervalRef.current = setInterval(() => {
        setRecordingElapsed((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      stream?.getTracks().forEach((track) => track.stop());
      mediaRecorderRef.current = null;
      setOralError('Impossibile accedere al microfono: ' + (err?.message || 'Permesso negato'));
    }
  };

  const handleStopVoiceRecording = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    setIsRecordingAnswer(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  const handleSendOralAudio = async (blob: Blob) => {
    setIsEvaluatingOral(true);
    setOralError(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const result = await api.study.evaluateOral(
        subject.id,
        blob,
        oralQuestion,
        llmModelId,
        controller.signal
      );
      setOralResult(result);
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setOralError(err?.message || 'Errore durante la valutazione della risposta con Whisper e Qwen.');
    } finally {
      setIsEvaluatingOral(false);
      abortControllerRef.current = null;
    }
  };

  const currentQ = questions[currentIdx];
  const currentCorrectIdx = currentQ ? (currentQ.correct_index ?? currentQ.correctIndex ?? 0) : 0;

  // ═════════════════════════════════════════════════════════════════════════════
  // SCHERMATA 0: Rilevamento LLM Mancante (SOLO SE NESSUN LLM INSTALLATO)
  // Non si può procedere al passo 1 senza LLM.
  // ═════════════════════════════════════════════════════════════════════════════
  if (!checkingModels && !hasInstalledLLM) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 select-none relative bg-black/20">
        <div className="max-w-md w-full p-8 rounded-3xl liquid-glass border border-white/[0.12] shadow-[0_24px_60px_rgba(0,0,0,0.6)] text-center space-y-6 animate-scale-in">
          {/* Glowing Icon */}
          <div className="w-16 h-16 rounded-3xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center text-amber-300 mx-auto shadow-[0_0_30px_rgba(245,158,11,0.25)]">
            <Cpu size={32} strokeWidth={1.75} />
          </div>

          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/15 border border-amber-500/30 text-[11px] font-mono text-amber-300">
              <ShieldAlert size={12} />
              <span>Modello LLM Richiesto</span>
            </div>
            <h2 className="text-xl font-bold tracking-tight text-white/95">
              Installa un LLM per il Ripasso
            </h2>
            <p className="text-xs text-white/50 leading-relaxed max-w-sm mx-auto">
              Per generare quiz universitari e simulare interrogazioni vocali d'esame è necessario aver installato almeno un modello linguistico locale in memoria.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] text-left text-xs space-y-2">
            <div className="flex items-center justify-between text-white/70 font-medium">
              <span>Modello consigliato:</span>
              <span className="font-mono text-amber-300 font-semibold">Qwen 3.5 4B</span>
            </div>
            <p className="text-[11px] text-white/40 leading-normal">
              Ottimizzato per Apple Silicon Metal 4-bit (zero allucinazioni e calcolo in locale senza cloud).
            </p>
          </div>

          {onOpenSettings && (
            <button
              type="button"
              onClick={onOpenSettings}
              className="w-full py-3.5 px-5 rounded-2xl bg-white/[0.14] hover:bg-white/[0.22] active:scale-95 text-xs font-semibold text-white border border-white/[0.20] shadow-liquid-md transition-all flex items-center justify-center gap-2 cursor-pointer group"
            >
              <Settings size={15} className="text-amber-300 group-hover:rotate-45 transition-transform" />
              <span>Apri Impostazioni per Scaricare il Modello</span>
              <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
            </button>
          )}

          <p className="text-[11px] text-white/40 font-sans">
            Il wizard si sbloccherà automaticamente non appena il modello risulterà installato.
          </p>
        </div>
      </div>
    );
  }

  // Varianti per l'animazione di scorrimento orizzontale del Wizard (Pagina che scorre a sinistra/destra)
  const pageSlideVariants: Variants = {
    enter: (direction: 'forward' | 'backward') => ({
      x: direction === 'forward' ? 70 : -70,
      opacity: 0,
      filter: 'blur(3px)',
    }),
    center: {
      x: 0,
      opacity: 1,
      filter: 'blur(0px)',
      transition: {
        x: { type: 'spring', stiffness: 420, damping: 32 },
        opacity: { duration: 0.16, ease: 'easeOut' },
        filter: { duration: 0.16 },
      },
    },
    exit: (direction: 'forward' | 'backward') => ({
      x: direction === 'forward' ? -70 : 70,
      opacity: 0,
      filter: 'blur(3px)',
      transition: {
        x: { type: 'spring', stiffness: 420, damping: 32 },
        opacity: { duration: 0.13, ease: 'easeIn' },
        filter: { duration: 0.13 },
      },
    }),
  };

  // ═════════════════════════════════════════════════════════════════════════════
  // SCHERMATE DEL WIZARD (Quando un LLM è installato)
  // ═════════════════════════════════════════════════════════════════════════════
  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-black/15 select-none relative p-6 sm:p-8">
      {/* Toast Notifica Kill Processo */}
      {killNotice && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 z-50 animate-slide-down">
          <div className="px-4 py-2 rounded-full liquid-glass border border-emerald-500/30 text-emerald-300 text-xs font-medium flex items-center gap-2 shadow-[0_10px_30px_rgba(0,0,0,0.6)] backdrop-blur-2xl">
            <Check size={14} className="text-emerald-400" />
            <span>{killNotice}</span>
          </div>
        </div>
      )}

      <div className="max-w-4xl w-full mx-auto flex-1 flex flex-col justify-center overflow-x-hidden overflow-y-auto">

        {/* ─────────────────────────────────────────────────────────────────────
            WIZARD: SCELTA LEZIONI (STEP 1) & SCELTA MODALITÀ (STEP 2)
            Con animazione fluida a scorrimento orizzontale e cards espandibili in-place
        ───────────────────────────────────────────────────────────────────── */}
        <AnimatePresence mode="wait" custom={slideDirection}>
          {wizardStep === 1 ? (
            <motion.div
              key="wizard-step-1"
              custom={slideDirection}
              variants={pageSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              className="w-full my-auto py-4 flex flex-col items-center"
            >
              {/* Intestazione Step 1 (Senza "Passo 1 di 2") */}
              <div className="text-center space-y-1.5 mb-6">
                <h2 className="text-2xl font-bold tracking-tight text-white/95">
                  Quali lezioni vuoi ripassare?
                </h2>
                <p className="text-xs text-white/50 leading-relaxed max-w-md mx-auto">
                  Scegli gli argomenti d'esame. L'IA formulerà le domande attingendo esclusivamente dai materiali delle lezioni selezionate.
                </p>
              </div>

              {/* Riga Centrale: Spacer SX + Scheda Lista Lezioni + Freccia DX */}
              <div className="flex items-center justify-center gap-5 w-full max-w-3xl mx-auto">
                {/* Spacer a sinistra per bilanciamento simmetrico perfetto */}
                <div className="w-12 h-12 flex-shrink-0 invisible pointer-events-none" aria-hidden="true" />

                {/* Scheda Lista Lezioni (Check box per selezionare i file) */}
                <div className="flex-1 min-w-0 max-w-xl">
                  {lectures.length === 0 ? (
                    <div className="p-8 rounded-3xl liquid-glass border border-white/[0.08] text-center space-y-3">
                      <BookOpen size={28} className="text-white/20 mx-auto" />
                      <h4 className="text-sm font-semibold text-white/80">Nessuna lezione trovata</h4>
                      <p className="text-xs text-white/40 max-w-xs mx-auto">
                        Registra o importa una lezione in {subject.name} per poter avviare la simulazione di ripasso.
                      </p>
                    </div>
                  ) : (
                    <div className="p-5 rounded-3xl liquid-glass border border-white/[0.10] shadow-[0_16px_40px_rgba(0,0,0,0.4)] space-y-4">
                      <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                        <div className="flex items-center gap-2">
                          <BookOpen size={14} className="text-amber-400" />
                          <span className="text-xs font-semibold text-white/80">
                            Lezioni del Corso
                          </span>
                          <span className="px-2 py-0.5 rounded-full bg-white/[0.08] text-[10px] font-mono text-white/60">
                            {selectedLectureIds.length} di {lectures.length} selezionate
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={selectAllLectures}
                            className="text-[11px] text-white/50 hover:text-white transition-colors cursor-pointer"
                          >
                            Seleziona tutte
                          </button>
                          <span className="text-white/20">•</span>
                          <button
                            type="button"
                            onClick={deselectAllLectures}
                            className="text-[11px] text-white/50 hover:text-white transition-colors cursor-pointer"
                          >
                            Deseleziona
                          </button>
                        </div>
                      </div>

                      {/* Grid Lezioni */}
                      <div className="grid grid-cols-1 gap-2 max-h-72 overflow-y-auto pr-1">
                        {lectures.map((l) => {
                          const isSelected = selectedLectureIds.includes(l.id);
                          const isImported = l.source_type === 'imported';

                          return (
                            <div
                              key={l.id}
                              onClick={() => toggleLectureSelection(l.id)}
                              className={clsx(
                                'p-3.5 rounded-2xl border transition-all cursor-pointer flex items-center justify-between gap-3 group select-none',
                                isSelected
                                  ? 'bg-white/[0.10] border-white/[0.22] text-white shadow-liquid-sm'
                                  : 'bg-white/[0.02] border-white/[0.05] text-white/60 hover:bg-white/[0.05] hover:border-white/[0.10]'
                              )}
                            >
                              <div className="flex items-center gap-3 min-w-0">
                                <div className={clsx(
                                  'w-5 h-5 rounded-lg border flex items-center justify-center transition-all flex-shrink-0',
                                  isSelected
                                    ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                                    : 'border-white/20 text-transparent group-hover:border-white/40'
                                )}>
                                  <Check size={12} strokeWidth={2.5} />
                                </div>
                                <div className="min-w-0">
                                  <h4 className={clsx(
                                    'text-xs font-semibold tracking-tight truncate',
                                    isSelected ? 'text-white' : 'text-white/80'
                                  )}>
                                    {l.title}
                                  </h4>
                                  <p className="text-[10px] text-white/40 font-mono mt-0.5">
                                    {new Date(l.date).toLocaleDateString('it-IT', { day: 'numeric', month: 'short' })}
                                    {l.duration_seconds && l.duration_seconds > 0 ? ` · ${Math.floor(l.duration_seconds / 60)}m` : ''}
                                  </p>
                                </div>
                              </div>

                              <span className={clsx(
                                'text-[10px] font-mono px-2 py-0.5 rounded-full border flex-shrink-0',
                                isImported
                                  ? 'bg-purple-500/10 border-purple-500/25 text-purple-300'
                                  : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                              )}>
                                {isImported ? 'Importata' : 'Registrata'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Freccia Destra al lato del box per passare alla schermata 2 */}
                <div className="w-12 h-12 flex-shrink-0 flex items-center justify-center">
                  <button
                    type="button"
                    disabled={selectedLectureIds.length === 0}
                    onClick={handleGoToStep2}
                    title={selectedLectureIds.length === 0 ? "Seleziona almeno una lezione per proseguire" : "Avanti: Scegli Modalità"}
                    className={clsx(
                      'w-12 h-12 rounded-2xl flex items-center justify-center border select-none transition-all duration-200 group',
                      selectedLectureIds.length > 0
                        ? 'liquid-glass bg-white/[0.06] text-white border-white/[0.12] shadow-liquid-sm hover:scale-110 active:scale-95 cursor-pointer'
                        : 'bg-white/[0.02] border-white/[0.06] text-white/20 cursor-not-allowed opacity-30'
                    )}
                  >
                    <ArrowRight
                      size={20}
                      strokeWidth={2.2}
                      className={clsx(
                        "transition-transform duration-200 ease-out",
                        selectedLectureIds.length > 0 && "group-hover:translate-x-1"
                      )}
                    />
                  </button>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="wizard-step-2"
              custom={slideDirection}
              variants={pageSlideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              className="w-full my-auto py-4 flex flex-col items-center"
            >
              {/* Intestazione Step 2 (Senza "Passo 2 di 2", collassa fluidamente quando attiva una modalità) */}
              <div className={clsx(
                "text-center space-y-1.5 transition-all duration-300 ease-out overflow-hidden select-none",
                activeMode === 'none' ? "max-h-24 opacity-100 mb-6" : "max-h-0 opacity-0 mb-0 pointer-events-none"
              )}>
                <h2 className="text-2xl font-bold tracking-tight text-white/95">
                  Come vuoi metterti alla prova?
                </h2>
                <p className="text-xs text-white/50 leading-relaxed max-w-md mx-auto">
                  Scegli il formato del ripasso per {subject.name}.
                </p>
              </div>

              {/* Riga Centrale: Freccia SX + Cards Modalità + Spacer DX */}
              <div className="flex items-center justify-center gap-5 w-full max-w-4xl mx-auto">
                {/* Freccia Sinistra al centro per tornare alla schermata 1 (scompare se modalità attiva) */}
                <div className={clsx(
                  "transition-all duration-300 ease-out flex items-center justify-center",
                  activeMode === 'none' ? "w-12 h-12 opacity-100 flex-shrink-0" : "w-0 h-0 opacity-0 overflow-hidden pointer-events-none"
                )}>
                  <button
                    type="button"
                    onClick={handleBackToStep1}
                    title="Torna alla selezione lezioni"
                    className="w-12 h-12 rounded-2xl liquid-glass bg-white/[0.06] text-white border border-white/[0.12] shadow-liquid-sm hover:scale-110 active:scale-95 flex items-center justify-center transition-all duration-200 cursor-pointer select-none group"
                  >
                    <ArrowLeft
                      size={20}
                      strokeWidth={2.2}
                      className="transition-transform duration-200 ease-out group-hover:-translate-x-1"
                    />
                  </button>
                </div>

                {/* Box Centrale Contenitore Cards (si allarga da max-w-xl a max-w-3xl quando una modalità è attiva) */}
                <div className={clsx(
                  "flex-1 min-w-0 transition-all duration-300 ease-out",
                  activeMode === 'none' ? "max-w-xl" : "max-w-3xl"
                )}>
                  <div className={clsx(
                    "flex items-stretch w-full transition-all duration-300 ease-out",
                    activeMode === 'none' ? "gap-4" : "gap-0"
                  )}>

                    {/* ══════════════════════════════════════════════════════════
                        CARD 1: QUIZ A SCELTA MULTIPLA
                        Quando premuta: Card 2 si nasconde (w-0, opacity-0) e questa si allarga
                    ══════════════════════════════════════════════════════════ */}
                    <div
                      onClick={activeMode === 'none' ? handleStartQuiz : undefined}
                      className={clsx(
                        'rounded-3xl liquid-glass border transition-all duration-300 ease-out flex flex-col justify-between overflow-hidden',
                        activeMode === 'none' && [
                          'flex-1 min-w-0 p-6 group shadow-liquid-sm cursor-pointer select-none',
                          'border-white/[0.10] hover:border-blue-400/40 hover:bg-white/[0.08] hover:scale-[1.02] active:scale-95'
                        ],
                        activeMode === 'quiz' && [
                          'flex-1 w-full min-w-0 p-6 sm:p-8 cursor-default max-h-[82vh] overflow-y-auto',
                          'border-blue-500/30 bg-white/[0.05] shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-blue-500/20'
                        ],
                        activeMode === 'oral' && [
                          'flex-[0_0_0%] w-0 min-w-0 p-0 m-0 border-0 opacity-0 pointer-events-none scale-95'
                        ]
                      )}
                    >
                      {activeMode === 'none' ? (
                        /* Preview Card Quiz nel Wizard */
                        <div className="flex flex-col justify-between h-full space-y-4">
                          <div>
                            <div className="w-12 h-12 rounded-2xl liquid-glass-subtle border border-blue-500/25 flex items-center justify-center text-blue-400 mb-4 group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(59,130,246,0.15)]">
                              <HelpCircle size={24} strokeWidth={1.75} />
                            </div>
                            <h3 className="text-base font-semibold tracking-tight text-white/95">
                              Quiz a Scelta Multipla
                            </h3>
                            <p className="text-xs text-white/50 mt-1.5 leading-relaxed">
                              Genera una batteria di 5 domande d'esame dai tuoi appunti, con calcolo del punteggio e spiegazione analitica degli errori.
                            </p>
                          </div>

                          <div className="mt-4 pt-4 border-t border-white/[0.06] flex items-center justify-between">
                            <span className="text-xs font-semibold text-blue-300 group-hover:text-blue-200 flex items-center gap-1.5">
                              <span>Avvia Quiz (5 domande)</span>
                              <ArrowRight size={13} strokeWidth={2} className="group-hover:translate-x-1 transition-transform" />
                            </span>
                          </div>
                        </div>
                      ) : activeMode === 'quiz' ? (
                        /* Contenuto Attivo Quiz (Espanso nella Card) */
                        <div className="w-full flex flex-col space-y-5 animate-fade-in">
                          {/* Top Navigation Bar con Tasto Esci */}
                          <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                            <button
                              type="button"
                              onClick={() => {
                                if (isGeneratingQuiz) {
                                  handleKillAIProcess();
                                } else {
                                  setActiveMode('none');
                                  setQuestions([]);
                                  setCurrentIdx(0);
                                  setSelectedOpt(null);
                                  setScore(0);
                                  setIsQuizCompleted(false);
                                }
                              }}
                              className="text-xs text-white/50 hover:text-white transition-colors flex items-center gap-1.5 cursor-pointer py-1"
                            >
                              <ArrowLeft size={13} />
                              <span>Torna alla Selezione Modalità</span>
                            </button>

                            {!isGeneratingQuiz && questions.length > 0 && !isQuizCompleted && (
                              <div className="flex items-center gap-3 text-xs font-mono text-white/50">
                                <span>Domanda {currentIdx + 1} di {questions.length}</span>
                                <span>·</span>
                                <span className="text-amber-300 font-semibold">{score} corrette</span>
                              </div>
                            )}
                          </div>

                          {/* STATO 1: Generazione Quiz in corso con Tasto Kill Processo */}
                          {isGeneratingQuiz ? (
                            <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center space-y-6">
                              <div className="w-16 h-16 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 mx-auto shadow-[0_0_30px_rgba(59,130,246,0.2)]">
                                <Loader2 size={30} className="animate-spin text-blue-400" />
                              </div>

                              <div className="space-y-1.5">
                                <h3 className="text-lg font-semibold text-white/95">
                                  Generazione Quiz in corso...
                                </h3>
                                <p className="text-xs text-white/50 max-w-sm mx-auto leading-relaxed">
                                  Qwen 3.5 sta analizzando le trascrizioni e gli appunti delle {selectedLectureIds.length} lezioni selezionate per formulare 5 domande d'esame.
                                </p>
                              </div>

                              {/* Shimmering Skeleton Preview */}
                              <div className="w-full max-w-md mx-auto space-y-2 pt-2 opacity-35">
                                <div className="h-3 bg-white/20 rounded-full w-5/6 mx-auto animate-pulse" />
                                <div className="h-3 bg-white/10 rounded-full w-full animate-pulse" />
                                <div className="h-3 bg-white/15 rounded-full w-4/6 mx-auto animate-pulse" />
                              </div>

                              {/* TASTO KILL DEL PROCESSO IA */}
                              <div className="pt-4 border-t border-white/[0.06] flex flex-col items-center gap-2">
                                <button
                                  type="button"
                                  onClick={handleKillAIProcess}
                                  disabled={isKillingProcess}
                                  className="px-5 py-2.5 rounded-full bg-red-500/15 hover:bg-red-500/25 border border-red-500/35 text-red-300 hover:text-red-200 text-xs font-semibold flex items-center gap-2 cursor-pointer transition-all active:scale-95 shadow-[0_4px_16px_rgba(239,68,68,0.25)] hover:border-red-500/50"
                                >
                                  <StopCircle size={14} className="text-red-400" />
                                  <span>{isKillingProcess ? 'Interruzione in corso...' : 'Interrompi Processo IA e Libera Cache'}</span>
                                </button>
                                <span className="text-[11px] text-white/40 font-sans">
                                  Ferma l'inferenza immediatamente e svuota la memoria Metal/GPU
                                </span>
                              </div>
                            </div>
                          ) : quizError ? (
                            /* Errore generazione con opzione riprova o torna indietro */
                            <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-red-500/30 text-center space-y-5">
                              <AlertCircle size={32} className="text-red-400 mx-auto" />
                              <div>
                                <h3 className="text-sm font-semibold text-white">Generazione Non Riuscita</h3>
                                <p className="text-xs text-red-300/80 mt-1 max-w-sm mx-auto">{quizError}</p>
                              </div>
                              <div className="flex items-center justify-center gap-3">
                                <button
                                  type="button"
                                  onClick={() => setActiveMode('none')}
                                  className="px-4 py-2 rounded-xl liquid-glass hover:bg-white/[0.10] text-xs font-medium text-white/80 border border-white/[0.12] cursor-pointer"
                                >
                                  Torna al Menu
                                </button>
                                <button
                                  type="button"
                                  onClick={executeQuizGeneration}
                                  className="px-4 py-2 rounded-xl bg-white/[0.14] hover:bg-white/[0.22] text-xs font-semibold text-white border border-white/[0.20] cursor-pointer"
                                >
                                  Riprova
                                </button>
                              </div>
                            </div>
                          ) : !isQuizCompleted && currentQ ? (
                            /* Domanda Attiva del Quiz */
                            <div className="space-y-6">
                              <div>
                                <span className="text-[10px] font-mono text-white/40 uppercase tracking-wider block mb-1">
                                  Tratta da: {currentQ.source_lecture || subject.name}
                                </span>
                                <h3 className="text-base sm:text-lg font-semibold tracking-tight text-white/95 leading-snug">
                                  {currentQ.question}
                                </h3>
                              </div>

                              <div className="space-y-2.5">
                                {currentQ.options.map((opt, idx) => {
                                  const isChosen = selectedOpt === idx;
                                  const isCorrect = idx === currentCorrectIdx;
                                  const showResult = selectedOpt !== null;

                                  return (
                                    <button
                                      key={idx}
                                      type="button"
                                      onClick={() => handleSelectOption(idx)}
                                      disabled={showResult}
                                      className={clsx(
                                        'w-full p-4 rounded-2xl text-left text-xs sm:text-[13px] tracking-tight transition-all flex items-center justify-between border cursor-pointer',
                                        showResult && isCorrect
                                          ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-100 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
                                          : showResult && isChosen && !isCorrect
                                          ? 'bg-red-500/20 border-red-500/40 text-red-200'
                                          : 'liquid-glass-subtle border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.16] text-white/85'
                                      )}
                                    >
                                      <span className="leading-relaxed pr-3">{opt}</span>
                                      {showResult && isCorrect && <CheckCircle2 size={16} className="text-emerald-400 flex-shrink-0" />}
                                      {showResult && isChosen && !isCorrect && <XCircle size={16} className="text-red-400 flex-shrink-0" />}
                                    </button>
                                  );
                                })}
                              </div>

                              {selectedOpt !== null && (
                                <div className="pt-4 border-t border-white/[0.08] space-y-4 animate-fade-in">
                                  <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.06] text-xs leading-relaxed text-white/70">
                                    <span className="text-[10px] font-mono text-white/40 uppercase block mb-1">Spiegazione accademica:</span>
                                    {currentQ.explanation}
                                  </div>

                                  <div className="flex justify-end">
                                    <button
                                      type="button"
                                      onClick={handleNextQuestion}
                                      className="px-5 py-2.5 rounded-2xl bg-white/[0.14] hover:bg-white/[0.22] text-white text-xs font-semibold border border-white/[0.18] shadow-liquid-sm transition-all flex items-center gap-1.5 cursor-pointer"
                                    >
                                      <span>{currentIdx + 1 < questions.length ? 'Prossima Domanda' : 'Vedi Risultati'}</span>
                                      <ArrowRight size={13} strokeWidth={2} />
                                    </button>
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            /* Risultato Finale Quiz */
                            <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center space-y-6">
                              <div className="w-16 h-16 rounded-3xl liquid-glass-subtle border border-amber-400/25 flex items-center justify-center text-amber-300 mx-auto shadow-liquid-sm">
                                <Award size={32} strokeWidth={1.75} />
                              </div>

                              <div>
                                <h3 className="text-xl font-bold text-white">Quiz Completato!</h3>
                                <p className="text-xs text-white/50 mt-1">
                                  Hai risposto correttamente a <span className="text-white font-bold">{score}</span> su <span className="text-white font-bold">{questions.length}</span> domande.
                                </p>
                                <div className="mt-3 inline-block px-4 py-1.5 rounded-full bg-white/[0.08] border border-white/[0.12] font-mono text-sm text-white font-semibold">
                                  Punteggio: {questions.length > 0 ? Math.round((score / questions.length) * 100) : 0}%
                                </div>
                              </div>

                              <div className="flex items-center justify-center gap-3 pt-2">
                                <button
                                  type="button"
                                  onClick={handleRestartQuiz}
                                  className="px-4 py-2 rounded-xl liquid-glass-subtle hover:bg-white/[0.10] text-xs font-medium text-white/80 border border-white/[0.10] transition-all flex items-center gap-1.5 cursor-pointer"
                                >
                                  <RotateCcw size={13} />
                                  <span>Ricomincia Questo Quiz</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={executeQuizGeneration}
                                  className="px-4 py-2 rounded-xl bg-white/[0.14] hover:bg-white/[0.22] text-xs font-semibold text-white border border-white/[0.18] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                                >
                                  <RefreshCw size={13} />
                                  <span>Genera Altre 5 Domande</span>
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : null}
                    </div>

                    {/* ══════════════════════════════════════════════════════════
                        CARD 2: INTERROGAZIONE VOCALE CON IA
                        Quando premuta: Card 1 si nasconde (w-0, opacity-0) e questa si allarga
                    ══════════════════════════════════════════════════════════ */}
                    <div
                      onClick={activeMode === 'none' ? handleStartOralMode : undefined}
                      className={clsx(
                        'rounded-3xl liquid-glass border transition-all duration-300 ease-out flex flex-col justify-between overflow-hidden',
                        activeMode === 'none' && [
                          'flex-1 min-w-0 p-6 group shadow-liquid-sm cursor-pointer select-none',
                          'border-white/[0.10] hover:border-emerald-400/40 hover:bg-white/[0.08] hover:scale-[1.02] active:scale-95'
                        ],
                        activeMode === 'oral' && [
                          'flex-1 w-full min-w-0 p-6 sm:p-8 cursor-default max-h-[82vh] overflow-y-auto',
                          'border-emerald-500/30 bg-white/[0.05] shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-emerald-500/20'
                        ],
                        activeMode === 'quiz' && [
                          'flex-[0_0_0%] w-0 min-w-0 p-0 m-0 border-0 opacity-0 pointer-events-none scale-95'
                        ]
                      )}
                    >
                      {activeMode === 'none' ? (
                        /* Preview Card Orale nel Wizard */
                        <div className="flex flex-col justify-between h-full space-y-4">
                          <div>
                            <div className="w-12 h-12 rounded-2xl liquid-glass-subtle border border-emerald-500/25 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-110 transition-transform shadow-[0_0_20px_rgba(16,185,129,0.15)]">
                              <Mic size={24} strokeWidth={1.75} />
                            </div>
                            <h3 className="text-base font-semibold tracking-tight text-white/95">
                              Interrogazione Vocale con IA
                            </h3>
                            <p className="text-xs text-white/50 mt-1.5 leading-relaxed">
                              L'assistente ti pone una domanda d'esame orale. Rispondi al microfono e ricevi un voto in trentesimi, punti di forza e lacune.
                            </p>
                          </div>

                          <div className="mt-4 pt-4 border-t border-white/[0.06] flex items-center justify-between">
                            <span className="text-xs font-semibold text-emerald-300 group-hover:text-emerald-200 flex items-center gap-1.5">
                              <span>Simula Colloquio Orale</span>
                              <ArrowRight size={13} strokeWidth={2} className="group-hover:translate-x-1 transition-transform" />
                            </span>
                          </div>
                        </div>
                      ) : activeMode === 'oral' ? (
                        /* Contenuto Attivo Orale (Espanso nella Card) */
                        <div className="w-full flex flex-col space-y-5 animate-fade-in">
                          {/* Header Orale */}
                          <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                            <button
                              type="button"
                              onClick={() => {
                                if (isGeneratingOralQ || isEvaluatingOral) {
                                  handleKillAIProcess();
                                } else {
                                  handleStopVoiceRecording();
                                  setActiveMode('none');
                                  setOralQuestion('');
                                  setOralResult(null);
                                  setOralError(null);
                                }
                              }}
                              className="text-xs text-white/50 hover:text-white transition-colors flex items-center gap-1.5 cursor-pointer py-1"
                            >
                              <ArrowLeft size={13} />
                              <span>Torna alla Selezione Modalità</span>
                            </button>

                            {!isGeneratingOralQ && !isEvaluatingOral && (
                              <button
                                type="button"
                                disabled={isRecordingAnswer}
                                onClick={handleGenerateOralQuestion}
                                className="text-[11px] font-mono text-emerald-400/80 hover:text-emerald-300 transition-colors flex items-center gap-1 cursor-pointer disabled:opacity-40"
                              >
                                <RefreshCw size={11} />
                                <span>Cambia Domanda</span>
                              </button>
                            )}
                          </div>

                          <div className="space-y-6">
                            {/* Domanda d'esame */}
                            <div className="text-center space-y-2">
                              <div className="w-12 h-12 rounded-2xl liquid-glass-subtle border border-emerald-500/25 flex items-center justify-center text-emerald-400 mx-auto">
                                <Mic size={24} strokeWidth={1.75} />
                              </div>
                              <span className="text-[10px] font-mono text-emerald-400/80 uppercase tracking-wider block">
                                Colloquio Orale · {oralTopic || subject.name}
                              </span>

                              {isGeneratingOralQ ? (
                                <div className="py-4 space-y-3">
                                  <div className="flex items-center justify-center gap-2 text-xs text-white/60">
                                    <Loader2 size={15} className="animate-spin text-emerald-400" />
                                    <span>Il docente sta formulando la domanda d'esame...</span>
                                  </div>

                                  {/* Tasto Kill durante formulazione domanda */}
                                  <div className="pt-2">
                                    <button
                                      type="button"
                                      onClick={handleKillAIProcess}
                                      disabled={isKillingProcess}
                                      className="px-4 py-1.5 rounded-full bg-red-500/15 hover:bg-red-500/25 border border-red-500/35 text-red-300 text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition-all mx-auto active:scale-95 shadow-[0_4px_16px_rgba(239,68,68,0.25)] hover:border-red-500/50"
                                    >
                                      <StopCircle size={13} className="text-red-400" />
                                      <span>Interrompi Generazione</span>
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <h3 className="text-base sm:text-lg font-semibold tracking-tight text-white/95 leading-relaxed max-w-xl mx-auto">
                                  "{oralQuestion || 'Descrivi i concetti cardine affrontati nelle lezioni.'}"
                                </h3>
                              )}
                            </div>

                            {oralError && (
                              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/25 text-xs text-red-300 text-center">
                                {oralError}
                              </div>
                            )}

                            {/* Pulsante Microfono & Stato Registrazione */}
                            {!isGeneratingOralQ && !isEvaluatingOral && (
                              <div className="pt-2 flex flex-col items-center gap-3">
                                <button
                                  type="button"
                                  onClick={isRecordingAnswer ? handleStopVoiceRecording : handleStartVoiceRecording}
                                  className={clsx(
                                    'w-20 h-20 rounded-full flex items-center justify-center transition-all border cursor-pointer',
                                    isRecordingAnswer
                                      ? 'bg-red-500/25 border-red-500/60 text-red-400 animate-pulse shadow-[0_0_30px_rgba(239,68,68,0.35)] scale-105'
                                      : 'liquid-glass hover:bg-white/[0.14] border-white/[0.18] text-white hover:scale-105 shadow-liquid-md'
                                  )}
                                >
                                  <Mic size={28} strokeWidth={2} />
                                </button>

                                <div className="text-center">
                                  <p className="text-xs text-white/70 font-medium">
                                    {isRecordingAnswer
                                      ? `In ascolto... (${recordingElapsed}s) Clicca per terminare e inviare`
                                      : 'Clicca sul microfono e rispondi a voce'}
                                  </p>
                                  <p className="text-[11px] text-white/50 font-sans mt-0.5">
                                    {isRecordingAnswer
                                      ? 'Parla chiaramente esponendo i concetti chiave'
                                      : 'L\'IA valuterà rigore concettuale e completezza'}
                                  </p>
                                </div>

                                {isRecordingAnswer && (
                                  <button
                                    type="button"
                                    onClick={handleKillAIProcess}
                                    className="text-[11px] text-red-400/80 hover:text-red-300 transition-colors flex items-center gap-1 cursor-pointer pt-1"
                                  >
                                    <StopCircle size={12} />
                                    <span>Annulla registrazione</span>
                                  </button>
                                )}
                              </div>
                            )}

                            {/* STATO 2: Valutazione Orale in corso con Tasto Kill Processo */}
                            {isEvaluatingOral && (
                              <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] text-center space-y-4">
                                <div className="flex items-center justify-center gap-3 text-xs text-emerald-300">
                                  <Loader2 size={16} className="animate-spin text-emerald-400" />
                                  <span>Trascrizione Whisper & Valutazione con Qwen 3.5 in corso...</span>
                                </div>

                                {/* Tasto Kill durante valutazione */}
                                <div className="flex flex-col items-center gap-1.5 pt-1">
                                  <button
                                    type="button"
                                    onClick={handleKillAIProcess}
                                    disabled={isKillingProcess}
                                    className="px-5 py-2 rounded-full bg-red-500/15 hover:bg-red-500/25 border border-red-500/35 text-red-300 text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition-all active:scale-95 shadow-[0_4px_16px_rgba(239,68,68,0.25)] hover:border-red-500/50"
                                  >
                                    <StopCircle size={14} className="text-red-400" />
                                    <span>{isKillingProcess ? 'Interruzione...' : 'Interrompi Valutazione e Libera Memoria'}</span>
                                  </button>
                                  <span className="text-[10px] text-white/30 font-mono">
                                    Ferma Whisper/LLM immediatamente e svuota la cache Metal
                                  </span>
                                </div>
                              </div>
                            )}

                            {/* Scheda Risultato Valutazione Orale */}
                            {oralResult && (
                              <div className="pt-4 border-t border-white/[0.08] space-y-4 text-left animate-slide-up">
                                {/* Riga Voto */}
                                <div className="flex items-center justify-between p-4 rounded-2xl bg-white/[0.04] border border-white/[0.10]">
                                  <div>
                                    <span className="text-[10px] font-mono text-white/40 uppercase block">Esito Colloquio</span>
                                    <h4 className="text-sm font-semibold text-white">Valutazione della Commissione</h4>
                                  </div>
                                  <div className={clsx(
                                    'px-3.5 py-1.5 rounded-xl border text-sm font-bold font-mono',
                                    oralResult.score_num >= 27
                                      ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                                      : oralResult.score_num >= 22
                                      ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                                      : 'bg-red-500/20 border-red-500/40 text-red-300'
                                  )}>
                                    Voto: {oralResult.grade}
                                  </div>
                                </div>

                                {/* Trascrizione risposta studente */}
                                {oralResult.student_transcript && (
                                  <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-xs text-white/70">
                                    <span className="text-[10px] font-mono text-white/40 uppercase block mb-1">Cosa hai detto a voce:</span>
                                    <p className="italic">"{oralResult.student_transcript}"</p>
                                  </div>
                                )}

                                {/* Punti di forza */}
                                <div className="p-3.5 rounded-2xl bg-emerald-500/[0.06] border border-emerald-500/20 text-xs text-emerald-200/90 space-y-1">
                                  <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider block font-semibold">
                                    ✓ Cosa hai detto bene:
                                  </span>
                                  <p className="leading-relaxed">{oralResult.good_points}</p>
                                </div>

                                {/* Lacune ed errori */}
                                <div className="p-3.5 rounded-2xl bg-amber-500/[0.06] border border-amber-500/20 text-xs text-amber-200/90 space-y-1">
                                  <span className="text-[10px] font-mono text-amber-400 uppercase tracking-wider block font-semibold">
                                    ! Cosa hai dimenticato o imprecisioni:
                                  </span>
                                  <p className="leading-relaxed">{oralResult.missing_or_errors}</p>
                                </div>

                                {/* Risposta ideale */}
                                <div className="p-3.5 rounded-2xl bg-blue-500/[0.06] border border-blue-500/20 text-xs text-blue-200/90 space-y-1">
                                  <span className="text-[10px] font-mono text-blue-400 uppercase tracking-wider block font-semibold">
                                    ★ Come riformularla all'esame in modo perfetto:
                                  </span>
                                  <p className="leading-relaxed">{oralResult.ideal_formulation}</p>
                                </div>

                                <div className="flex justify-end pt-2">
                                  <button
                                    type="button"
                                    onClick={handleGenerateOralQuestion}
                                    className="px-4 py-2 rounded-xl bg-white/[0.12] hover:bg-white/[0.20] text-xs font-semibold text-white border border-white/[0.16] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                                  >
                                    <RefreshCw size={13} />
                                    <span>Prossima Domanda d'Esame</span>
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : null}
                    </div>

                  </div>
                </div>

                {/* Spacer a destra per bilanciamento simmetrico perfetto (scompare se modalità attiva) */}
                <div className={clsx(
                  "transition-all duration-300 ease-out",
                  activeMode === 'none' ? "w-12 h-12 flex-shrink-0 invisible pointer-events-none" : "w-0 h-0 opacity-0 overflow-hidden pointer-events-none"
                )} aria-hidden="true" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
}
