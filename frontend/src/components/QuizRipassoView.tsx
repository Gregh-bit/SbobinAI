// components/QuizRipassoView.tsx — Sezione Quiz & Ripasso (Scelta Multipla Reale + Interrogazione Vocale MLX)
import { useState, useRef, useEffect } from 'react';
import { 
  HelpCircle, Mic, CheckCircle2, XCircle, ArrowRight, 
  RotateCcw, Volume2, Sparkles, AlertTriangle, Loader2, 
  BookOpen, Square, CheckSquare, Award, ArrowLeft, RefreshCw
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import type { Lecture, OralEvaluationResult, QuizQuestion, Subject } from '../types';

interface Props {
  subject: Subject;
  lectures?: Lecture[];
  onOpenSettings?: () => void;
}

export default function QuizRipassoView({ subject, lectures = [], onOpenSettings }: Props) {
  const { settings } = useSettings();
  const hasModel = Boolean(settings.modelsDownloaded);

  const [mode, setMode] = useState<'overview' | 'quiz' | 'oral'>('overview');

  // Selezione lezioni su cui esercitarsi
  const [selectedLectureIds, setSelectedLectureIds] = useState<number[]>([]);

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

  // Modello LLM scelto dall'utente nelle Impostazioni
  const llmModelId = settings.llmModel || settings.ollamaModel || undefined;

  // Seleziona di default tutte le lezioni
  useEffect(() => {
    if (lectures.length > 0 && selectedLectureIds.length === 0) {
      setSelectedLectureIds(lectures.map((l) => l.id));
    }
  }, [lectures]);

  // Solo id di lezioni ancora esistenti (evita id obsoleti dopo eliminazioni)
  const getValidLectureIds = (): number[] | undefined => {
    const valid = selectedLectureIds.filter((id) => lectures.some((l) => l.id === id));
    return valid.length > 0 ? valid : undefined;
  };

  // Cleanup timer e microfono al dismount (senza inviare l'audio)
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      const rec = mediaRecorderRef.current;
      if (rec) {
        rec.onstop = null;
        rec.ondataavailable = null;
        if (rec.state !== 'inactive') {
          try {
            rec.stop();
          } catch {
            // ignore
          }
        }
        rec.stream.getTracks().forEach((track) => track.stop());
        mediaRecorderRef.current = null;
      }
    };
  }, []);

  const toggleLectureSelection = (id: number) => {
    setSelectedLectureIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAllLectures = () => {
    setSelectedLectureIds(lectures.map((l) => l.id));
  };

  // ── 1. Avvio Quiz a Scelta Multipla ─────────────────────────────────────────
  const handleStartQuiz = async () => {
    setQuizError(null);
    setIsGeneratingQuiz(true);
    try {
      const res = await api.study.generateQuiz(subject.id, {
        lecture_ids: getValidLectureIds(),
        model_id: llmModelId,
      });

      if (res?.questions && res.questions.length > 0) {
        setQuestions(res.questions);
        setCurrentIdx(0);
        setSelectedOpt(null);
        setScore(0);
        setIsQuizCompleted(false);
        setMode('quiz');
      } else {
        throw new Error('Nessuna domanda restituita dal modello IA.');
      }
    } catch (err: any) {
      setQuizError(err?.message || 'Errore durante la generazione del quiz con Qwen 3.5');
    } finally {
      setIsGeneratingQuiz(false);
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

  // ── 2. Avvio Interrogazione Vocale ──────────────────────────────────────────
  const handleStartOralMode = async () => {
    setMode('oral');
    setOralResult(null);
    setOralError(null);
    if (!oralQuestion) {
      await handleGenerateOralQuestion();
    }
  };

  const handleGenerateOralQuestion = async () => {
    setIsGeneratingOralQ(true);
    setOralError(null);
    setOralResult(null);
    try {
      const res = await api.study.getOralQuestion(subject.id, {
        lecture_ids: getValidLectureIds(),
        model_id: llmModelId,
      });
      if (res?.question) {
        setOralQuestion(res.question);
        setOralTopic(res.topic || subject.name);
      }
    } catch (err: any) {
      setOralError(err?.message || 'Errore durante la generazione della domanda d\'esame.');
    } finally {
      setIsGeneratingOralQ(false);
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
      // Scegli un formato supportato dalla webview (WebKit non supporta audio/webm)
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
      // Rilascia il microfono se la registrazione non è partita
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
    try {
      const result = await api.study.evaluateOral(subject.id, blob, oralQuestion, llmModelId);
      setOralResult(result);
    } catch (err: any) {
      setOralError(err?.message || 'Errore durante la valutazione della risposta orale con Whisper e Qwen.');
    } finally {
      setIsEvaluatingOral(false);
    }
  };

  const currentQ = questions[currentIdx];
  const currentCorrectIdx = currentQ ? (currentQ.correct_index ?? currentQ.correctIndex ?? 0) : 0;

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-black/15 select-none relative p-6 sm:p-8">
      <div className="max-w-4xl w-full mx-auto flex-1 flex flex-col overflow-y-auto">

        {/* ── 1. MODALITÀ OVERVIEW (SELEZIONE) ── */}
        {mode === 'overview' && (
          <div className="space-y-6 my-auto animate-fade-in py-4">
            <div className="text-center max-w-xl mx-auto space-y-1.5">
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full liquid-glass-subtle text-[11px] font-mono text-amber-300 border border-amber-400/25 mb-1 shadow-liquid-sm">
                <Sparkles size={11} className="text-amber-400" />
                <span>Simulazione Esame & Auto-Apprendimento</span>
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-white/95">
                Ripasso Universitario · {subject.name}
              </h2>
              <p className="text-xs text-white/50 leading-relaxed">
                Esercitati sulle lezioni registrate e sui materiali con Qwen 3.5 ed ottieni correzioni istantanee e feedback vocale d'esame.
              </p>
            </div>

            {/* Avviso Modello Mancante */}
            {!hasModel && (
              <div className="max-w-xl mx-auto p-3.5 rounded-2xl bg-amber-500/[0.08] border border-amber-500/25 flex items-center justify-between gap-3 shadow-liquid-sm">
                <div className="flex items-center gap-2.5 text-xs text-amber-200">
                  <AlertTriangle size={16} className="text-amber-400 flex-shrink-0" />
                  <span>Modello LLM non scaricato. Scarica Qwen 3.5 dalle Impostazioni per generare i quiz.</span>
                </div>
                {onOpenSettings && (
                  <button
                    onClick={onOpenSettings}
                    className="px-3 py-1 rounded-xl bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 text-xs font-medium cursor-pointer transition-all flex-shrink-0"
                  >
                    Impostazioni
                  </button>
                )}
              </div>
            )}

            {/* Selezione Lezioni */}
            {lectures.length > 0 && (
              <div className="max-w-xl mx-auto p-4 rounded-2xl liquid-glass border border-white/[0.08] space-y-2.5 shadow-liquid-sm">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-mono uppercase tracking-wider text-white/50 flex items-center gap-1.5">
                    <BookOpen size={12} />
                    <span>Seleziona Lezioni per il Ripasso ({selectedLectureIds.length}/{lectures.length})</span>
                  </span>
                  <button
                    onClick={selectAllLectures}
                    className="text-[10px] text-white/40 hover:text-white transition-colors cursor-pointer"
                  >
                    Seleziona tutte
                  </button>
                </div>

                <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto p-1">
                  {lectures.map((l) => {
                    const isSelected = selectedLectureIds.includes(l.id);
                    return (
                      <button
                        key={l.id}
                        type="button"
                        onClick={() => toggleLectureSelection(l.id)}
                        className={clsx(
                          'px-2.5 py-1 rounded-xl text-[11px] font-medium transition-all flex items-center gap-1.5 border cursor-pointer',
                          isSelected
                            ? 'bg-white/[0.14] border-white/[0.22] text-white shadow-liquid-sm'
                            : 'bg-white/[0.03] border-white/[0.06] text-white/40 hover:text-white/70'
                        )}
                      >
                        {isSelected ? <CheckSquare size={12} className="text-amber-300" /> : <Square size={12} />}
                        <span className="truncate max-w-[180px]">{l.title}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {quizError && (
              <div className="max-w-xl mx-auto p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300">
                {quizError}
              </div>
            )}

            {/* Griglia Selezione Modalità */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
              {/* Card 1: Quiz a Scelta Multipla */}
              <div 
                onClick={!isGeneratingQuiz && hasModel ? handleStartQuiz : undefined}
                className={clsx(
                  'p-6 rounded-3xl liquid-glass border transition-all flex flex-col justify-between group shadow-liquid-sm',
                  hasModel
                    ? 'border-white/[0.08] hover:border-white/[0.20] hover:bg-white/[0.06] cursor-pointer'
                    : 'border-white/[0.05] opacity-60 cursor-not-allowed'
                )}
              >
                <div>
                  <div className="w-11 h-11 rounded-2xl liquid-glass-subtle border border-blue-500/25 flex items-center justify-center text-blue-400 mb-4 group-hover:scale-105 transition-transform">
                    <HelpCircle size={22} strokeWidth={1.75} />
                  </div>
                  <h3 className="text-base font-semibold tracking-tight text-white/95">
                    Quiz a Scelta Multipla
                  </h3>
                  <p className="text-xs text-white/50 mt-1.5 leading-relaxed">
                    Genera una batteria di 5 domande d'esame dai tuoi appunti, con calcolo del punteggio e spiegazione analitica degli errori.
                  </p>
                </div>

                <div className="mt-6 pt-4 border-t border-white/[0.06] flex items-center justify-between">
                  <span className="text-xs font-semibold text-blue-300 group-hover:text-blue-200 flex items-center gap-1.5">
                    {isGeneratingQuiz ? (
                      <>
                        <Loader2 size={13} className="animate-spin text-blue-300" />
                        <span>Generazione con Qwen 3.5...</span>
                      </>
                    ) : (
                      <>
                        <span>Genera Quiz (5 domande)</span>
                        <ArrowRight size={13} strokeWidth={2} />
                      </>
                    )}
                  </span>
                </div>
              </div>

              {/* Card 2: Interrogazione Vocale */}
              <div 
                onClick={hasModel ? handleStartOralMode : undefined}
                className={clsx(
                  'p-6 rounded-3xl liquid-glass border transition-all flex flex-col justify-between group shadow-liquid-sm',
                  hasModel
                    ? 'border-white/[0.08] hover:border-white/[0.20] hover:bg-white/[0.06] cursor-pointer'
                    : 'border-white/[0.05] opacity-60 cursor-not-allowed'
                )}
              >
                <div>
                  <div className="w-11 h-11 rounded-2xl liquid-glass-subtle border border-emerald-500/25 flex items-center justify-center text-emerald-400 mb-4 group-hover:scale-105 transition-transform">
                    <Mic size={22} strokeWidth={1.75} />
                  </div>
                  <h3 className="text-base font-semibold tracking-tight text-white/95">
                    Interrogazione Vocale con IA
                  </h3>
                  <p className="text-xs text-white/50 mt-1.5 leading-relaxed">
                    L'assistente ti pone una domanda d'esame orale. Rispondi al microfono e ricevi un voto in trentesimi, punti di forza e lacune.
                  </p>
                </div>

                <div className="mt-6 pt-4 border-t border-white/[0.06] flex items-center justify-between">
                  <span className="text-xs font-semibold text-emerald-300 group-hover:text-emerald-200 flex items-center gap-1.5">
                    <span>Simula Colloquio Orale</span>
                    <ArrowRight size={13} strokeWidth={2} />
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── 2. MODALITÀ QUIZ A SCELTA MULTIPLA ── */}
        {mode === 'quiz' && (
          <div className="flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full py-4 animate-fade-in">
            {/* Header del quiz */}
            <div className="flex items-center justify-between mb-4">
              <button
                type="button"
                onClick={() => setMode('overview')}
                className="text-xs text-white/50 hover:text-white transition-colors flex items-center gap-1 cursor-pointer"
              >
                <ArrowLeft size={13} />
                <span>Torna al Menu Ripasso</span>
              </button>
              <div className="flex items-center gap-3 text-xs font-mono text-white/50">
                <span>Domanda {currentIdx + 1} di {questions.length}</span>
                <span>·</span>
                <span className="text-amber-300 font-semibold">{score} corrette</span>
              </div>
            </div>

            {!isQuizCompleted && currentQ ? (
              <div className="p-8 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-lg space-y-6">
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
              <div className="p-8 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-lg text-center space-y-6">
                <div className="w-16 h-16 rounded-3xl liquid-glass-subtle border border-amber-400/25 flex items-center justify-center text-amber-300 mx-auto shadow-liquid-sm">
                  <Award size={32} strokeWidth={1.75} />
                </div>

                <div>
                  <h3 className="text-xl font-bold text-white">Quiz Completato!</h3>
                  <p className="text-xs text-white/50 mt-1">
                    Hai risposto correttamente a <span className="text-white font-bold">{score}</span> su <span className="text-white font-bold">{questions.length}</span> domande.
                  </p>
                  <div className="mt-3 inline-block px-4 py-1.5 rounded-full bg-white/[0.08] border border-white/[0.12] font-mono text-sm text-white font-semibold">
                    Punteggio: {Math.round((score / questions.length) * 100)}%
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
                    onClick={handleStartQuiz}
                    className="px-4 py-2 rounded-xl bg-white/[0.14] hover:bg-white/[0.22] text-xs font-semibold text-white border border-white/[0.18] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                  >
                    <RefreshCw size={13} />
                    <span>Genera Altre 5 Domande</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── 3. MODALITÀ INTERROGAZIONE VOCALE (ESAME ORALE) ── */}
        {mode === 'oral' && (
          <div className="flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full py-4 animate-fade-in">
            {/* Header Orale */}
            <div className="flex items-center justify-between mb-4">
              <button
                type="button"
                onClick={() => {
                  handleStopVoiceRecording();
                  setMode('overview');
                }}
                className="text-xs text-white/50 hover:text-white transition-colors flex items-center gap-1 cursor-pointer"
              >
                <ArrowLeft size={13} />
                <span>Torna al Menu Ripasso</span>
              </button>

              <button
                type="button"
                disabled={isGeneratingOralQ || isRecordingAnswer}
                onClick={handleGenerateOralQuestion}
                className="text-[11px] font-mono text-emerald-400/80 hover:text-emerald-300 transition-colors flex items-center gap-1 cursor-pointer disabled:opacity-40"
              >
                <RefreshCw size={11} className={clsx(isGeneratingOralQ && 'animate-spin')} />
                <span>Cambia Domanda</span>
              </button>
            </div>

            <div className="p-8 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-lg space-y-6">
              <div className="text-center space-y-2">
                <div className="w-12 h-12 rounded-2xl liquid-glass-subtle border border-emerald-500/25 flex items-center justify-center text-emerald-400 mx-auto">
                  <Volume2 size={24} strokeWidth={1.75} />
                </div>
                <span className="text-[10px] font-mono text-emerald-400/80 uppercase tracking-wider block">
                  Colloquio Orale · {oralTopic || subject.name}
                </span>

                {isGeneratingOralQ ? (
                  <div className="py-4 flex items-center justify-center gap-2 text-xs text-white/40">
                    <Loader2 size={14} className="animate-spin text-emerald-400" />
                    <span>Il docente sta formulando la domanda...</span>
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
              <div className="pt-2 flex flex-col items-center gap-3">
                <button
                  type="button"
                  disabled={isEvaluatingOral || isGeneratingOralQ}
                  onClick={isRecordingAnswer ? handleStopVoiceRecording : handleStartVoiceRecording}
                  className={clsx(
                    'w-20 h-20 rounded-full flex items-center justify-center transition-all border cursor-pointer disabled:opacity-40',
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
                      ? `In ascolto... (${recordingElapsed}s) Clicca per terminare`
                      : isEvaluatingOral
                      ? 'Trascrizione Whisper & Valutazione con Qwen 3.5 in corso...'
                      : 'Clicca sul microfono e rispondi a voce'}
                  </p>
                  <p className="text-[11px] text-white/40 font-mono mt-0.5">
                    {isRecordingAnswer
                      ? 'Parla chiaramente spiegando i concetti'
                      : 'L\'IA valuterà rigore concettuale e completezza'}
                  </p>
                </div>
              </div>

              {/* Loading Spinner durante valutazione */}
              {isEvaluatingOral && (
                <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-center gap-3 text-xs text-emerald-300">
                  <Loader2 size={16} className="animate-spin text-emerald-400" />
                  <span>Elaborazione accademica in corso (Metal Unified Memory)...</span>
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
        )}

      </div>
    </div>
  );
}
