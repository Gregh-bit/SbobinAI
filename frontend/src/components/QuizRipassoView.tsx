// components/QuizRipassoView.tsx — Sezione Quiz & Ripasso Accademico (Dashboard 2 Colonne + Kill Processo IA)
import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  HelpCircle, Mic, CheckCircle2, XCircle, ArrowRight, 
  RotateCcw, AlertCircle, Loader2, 
  BookOpen, Award, ArrowLeft, RefreshCw,
  Settings, Check, StopCircle, Sparkles, ChevronDown, Trash2, GraduationCap, X
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import { useDelayedLoading, QuizCardSkeleton, QuizWizardSkeleton, SkeletonPill } from './Skeleton';
import { liquidMenuVariants, liquidPopoverContainerClass } from '../lib/animations';
import type {
  Lecture,
  ModelsStatusResponse,
  OralEvaluationResult,
  QuizQuestion,
  QuizResultItem,
  QuizSetOut,
  StudentConcept,
  StudentMasterySummary,
  Subject,
} from '../types';

export interface OralQuestionTarget {
  question: string;
  topic?: string;
  key: number;
}

interface Props {
  subject: Subject;
  lectures?: Lecture[];
  onOpenSettings?: () => void;
  initialQuizId?: number | null;
  onClearInitialQuiz?: () => void;
  initialOralQuestion?: OralQuestionTarget | null;
  onClearInitialOralQuestion?: () => void;
  onReviewInChat?: (review: {
    quizId: number;
    chatSessionId?: number | null;
    quizTitle: string;
    score: number;
    total: number;
    wrongCount: number;
  }) => void;
}

function mapQuizSetToQuestions(quizSet: QuizSetOut): QuizQuestion[] {
  return quizSet.questions.map((q, idx) => ({
    id: `saved-${quizSet.id}-${idx}`,
    question: q.question,
    concept: (q as any).concept || '',
    options: q.options,
    correct_index: q.correct_index,
    explanation: q.explanation,
    source_lecture: quizSet.source_label || quizSet.title,
  }));
}

function formatQuizDisplayTitle(title: string): string {
  if (!title) return 'Quiz di ripasso';
  let cleaned = title.replace(/^quiz(\s*:\s*|\s+su\s+|\s+per\s+|\s+di\s+)?/i, '').trim();
  cleaned = cleaned.replace(/^(mi\s+)?fammi\s+un\s+quiz\s+(su\s+|di\s+|per\s+)?/i, '').trim();
  cleaned = cleaned.replace(/^(mi\s+)?fai\s+un\s+quiz\s+(su\s+|di\s+|per\s+)?/i, '').trim();
  if (!cleaned) return title;
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function formatQuizSourceLabel(sourceLabel?: string): { primary: string; extraCount: number } {
  if (!sourceLabel) return { primary: '', extraCount: 0 };
  const parts = sourceLabel.split(' · ').map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return { primary: '', extraCount: 0 };
  const primary = parts[0];
  const extraCount = parts.length - 1;
  return { primary, extraCount };
}

function formatConceptTopic(topic: string): string {
  if (!topic) return 'Concetto';
  let clean = topic.replace(/^['"«]/, '').replace(/['"»]$/, '').trim();
  clean = clean.replace(/\s+(in\s+relazione\s+all[a-z]?|in\s+relazione\s+a|in\s+base\s+a|dell['’]?|all['’]?|di|del|dello|della|dei|degli|delle|in|con|su|da|per|ad|ed|a|e|o)$/i, '').trim();
  clean = clean.replace(/^(obiettivo\s+principale\s+dell['’]?|scopo\s+principale\s+dell['’]?|ruolo\s+principale\s+dell['’]?|l['’]|d['’]|un['’]|il\s+|la\s+|lo\s+|i\s+|gli\s+|le\s+)/i, '').trim();
  if (!clean) return 'Concetto';
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

export default function QuizRipassoView({
  subject,
  lectures = [],
  onOpenSettings,
  initialQuizId,
  onClearInitialQuiz,
  initialOralQuestion,
  onClearInitialOralQuestion,
  onReviewInChat,
}: Props) {
  const {
    settings,
    modelsStatus: contextModelsStatus,
    setModelsStatus: setContextModelsStatus,
  } = useSettings();

  // ── 0. Verifica Reale Disponibilità Modello LLM ──────────────────────────────
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(
    () => contextModelsStatus
  );
  const [checkingModels, setCheckingModels] = useState<boolean>(
    () => contextModelsStatus === null && !settings.llmDownloaded
  );
  const modelsStatusRef = useRef<ModelsStatusResponse | null>(contextModelsStatus);
  modelsStatusRef.current = modelsStatus ?? contextModelsStatus;

  useEffect(() => {
    if (contextModelsStatus) {
      setModelsStatus(contextModelsStatus);
      setCheckingModels(false);
    }
  }, [contextModelsStatus]);

  useEffect(() => {
    let cancelled = false;
    if (!modelsStatusRef.current && !settings.llmDownloaded) {
      setCheckingModels(true);
    }
    api.models
      .status()
      .then((st) => {
        if (cancelled) return;
        setModelsStatus(st);
        setContextModelsStatus(st);
      })
      .catch((err) => console.warn('[Quiz] Verifica modelli fallita:', err))
      .finally(() => {
        if (!cancelled) setCheckingModels(false);
      });
    return () => {
      cancelled = true;
    };
  }, [settings.modelsDownloaded, settings.llmDownloaded, settings.llmModel, setContextModelsStatus]);

  const effectiveModelsStatus = modelsStatus ?? contextModelsStatus;

  const hasInstalledLLM = useMemo(() => {
    if (effectiveModelsStatus) {
      const hasDownloadedLlm = effectiveModelsStatus.llm?.some(
        (m) => m.downloaded || m.status === 'completed'
      );
      if (hasDownloadedLlm) return true;
      const hasInstalledSummary = effectiveModelsStatus.installed_models?.some(
        (m) => m.type === 'llm'
      );
      if (hasInstalledSummary) return true;
      if (effectiveModelsStatus.llm && effectiveModelsStatus.llm.length > 0) return false;
    }
    return Boolean(
      settings.llmDownloaded || effectiveModelsStatus?.llm?.some((m) => m.downloaded)
    );
  }, [effectiveModelsStatus, settings.llmDownloaded]);

  // ── Modalità Attiva: 'none' (Dashboard), 'quiz' (Player Quiz), 'oral' (Player Orale) ─
  const [activeMode, setActiveMode] = useState<'none' | 'quiz' | 'oral'>('none');
  const [selectedLectureIds, setSelectedLectureIds] = useState<number[]>([]);

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
  const [isLoadingSavedQuiz, setIsLoadingSavedQuiz] = useState(false);
  const [quizError, setQuizError] = useState<string | null>(null);
  const [isQuizCompleted, setIsQuizCompleted] = useState(false);
  const [activeQuizMeta, setActiveQuizMeta] = useState<{
    id?: number;
    title: string;
    source_label?: string;
    chat_session_id?: number | null;
  } | null>(null);
  const [userAnswers, setUserAnswers] = useState<QuizResultItem[]>([]);

  // ── Quiz salvati dall'Assistente / Recenti ──────────────────────────────────
  const [savedQuizSets, setSavedQuizSets] = useState<QuizSetOut[]>([]);
  const [loadingSavedQuizzes, setLoadingSavedQuizzes] = useState(true);
  const [deletingQuizId, setDeletingQuizId] = useState<number | null>(null);
  const [confirmDeleteQuizId, setConfirmDeleteQuizId] = useState<number | null>(null);

  const isCheckingReadiness =
    activeMode === 'none' &&
    !isLoadingSavedQuiz &&
    initialQuizId == null &&
    !initialOralQuestion &&
    (checkingModels || (!hasInstalledLLM && loadingSavedQuizzes));
  const showReadinessSkeleton = useDelayedLoading(isCheckingReadiness, 150, 220);
  const showSavedQuizSkeleton = useDelayedLoading(isLoadingSavedQuiz, 150, 220);
  const showSavedListLoading = useDelayedLoading(
    loadingSavedQuizzes && savedQuizSets.length === 0,
    150,
    220
  );

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
  const killNoticeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef(true);
  const lastOralKeyRef = useRef<number | null>(null);

  // Modello LLM configurato
  const llmModelId = settings.llmModel || settings.ollamaModel || undefined;

  const applySavedQuizSet = useCallback((quizSet: QuizSetOut) => {
    const mapped = mapQuizSetToQuestions(quizSet);
    setActiveQuizMeta({
      id: quizSet.id,
      title: quizSet.title,
      source_label: quizSet.source_label,
      chat_session_id: quizSet.chat_session_id,
    });
    setQuestions(mapped);
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setUserAnswers([]);
    setIsQuizCompleted(false);
    setIsGeneratingQuiz(false);
    setIsLoadingSavedQuiz(false);
    setQuizError(mapped.length === 0 ? 'Questo quiz salvato non contiene domande.' : null);
    setActiveMode('quiz');
  }, []);

  // ── Mastery Tracking Summary (Progresso Concetti Consolidati) ──────────────
  const [masterySummary, setMasterySummary] = useState<StudentMasterySummary | null>(null);
  const [showMasteryDrawer, setShowMasteryDrawer] = useState(false);
  const masteryDrawerRef = useRef<HTMLDivElement | null>(null);
  const masteryButtonRef = useRef<HTMLButtonElement | null>(null);

  const loadMastery = useCallback(async () => {
    try {
      const summary = await api.study.getMasterySummary(subject.id);
      if (isMountedRef.current) {
        setMasterySummary(summary);
      }
    } catch {
      // ignore
    }
  }, [subject.id]);

  useEffect(() => {
    void loadMastery();
  }, [loadMastery]);

  // Click outside per chiudere il menu a tendina "Concetti"
  useEffect(() => {
    if (!showMasteryDrawer) return;
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        masteryDrawerRef.current &&
        !masteryDrawerRef.current.contains(target) &&
        masteryButtonRef.current &&
        !masteryButtonRef.current.contains(target)
      ) {
        setShowMasteryDrawer(false);
        setConfirmClearAllConcepts(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showMasteryDrawer]);

  // ── Gestione Concetti (Filtri, Eliminazione Singola e Reset Totale) ──────────
  const [conceptFilter, setConceptFilter] = useState<'all' | 'da_consolidare' | 'consolidato'>('all');
  const [deletingConceptId, setDeletingConceptId] = useState<number | null>(null);
  const [confirmClearAllConcepts, setConfirmClearAllConcepts] = useState(false);
  const [isClearingAllConcepts, setIsClearingAllConcepts] = useState(false);
  const [togglingConceptId, setTogglingConceptId] = useState<number | null>(null);

  const filteredConcepts = useMemo(() => {
    if (!masterySummary) return [];
    if (conceptFilter === 'all') return masterySummary.concepts;
    return masterySummary.concepts.filter((c) => c.status === conceptFilter);
  }, [masterySummary, conceptFilter]);

  const handleDeleteConcept = async (conceptId: number) => {
    try {
      setDeletingConceptId(conceptId);
      await api.study.deleteConcept(subject.id, conceptId);
      setMasterySummary((prev) => {
        if (!prev) return null;
        const remaining = prev.concepts.filter((c) => c.id !== conceptId);
        const cons = remaining.filter((c) => c.status === 'consolidato').length;
        const toCons = remaining.filter((c) => c.status === 'da_consolidare').length;
        return {
          ...prev,
          total_tracked: remaining.length,
          consolidated_count: cons,
          to_consolidate_count: toCons,
          concepts: remaining,
        };
      });
    } catch (err) {
      console.error('Errore durante eliminazione concetto:', err);
    } finally {
      setDeletingConceptId(null);
    }
  };

  const handleClearAllConcepts = async () => {
    try {
      setIsClearingAllConcepts(true);
      await api.study.clearAllConcepts(subject.id);
      setMasterySummary({
        subject_id: subject.id,
        total_tracked: 0,
        consolidated_count: 0,
        to_consolidate_count: 0,
        concepts: [],
      });
      setConfirmClearAllConcepts(false);
    } catch (err) {
      console.error('Errore durante azzeramento concetti:', err);
    } finally {
      setIsClearingAllConcepts(false);
    }
  };

  const handleToggleConceptStatus = async (concept: StudentConcept) => {
    const nextStatus = concept.status === 'consolidato' ? 'da_consolidare' : 'consolidato';
    try {
      setTogglingConceptId(concept.id);
      setMasterySummary((prev) => {
        if (!prev) return null;
        const updated = prev.concepts.map((c) =>
          c.id === concept.id ? { ...c, status: nextStatus } : c
        );
        const cons = updated.filter((c) => c.status === 'consolidato').length;
        const toCons = updated.filter((c) => c.status === 'da_consolidare').length;
        return {
          ...prev,
          consolidated_count: cons,
          to_consolidate_count: toCons,
          concepts: updated,
        };
      });
      await api.study.markConcept(subject.id, {
        topic: concept.topic,
        status: nextStatus,
      });
    } catch (err) {
      console.error('Errore aggiornamento stato concetto:', err);
      void loadMastery();
    } finally {
      setTogglingConceptId(null);
    }
  };

  // Carica elenco quiz salvati per il corso
  useEffect(() => {
    let cancelled = false;
    setLoadingSavedQuizzes(true);
    api.study
      .listQuizSets(subject.id)
      .then((list) => {
        if (!cancelled) setSavedQuizSets(list);
      })
      .catch((err) => {
        console.warn('[Quiz] Caricamento quiz salvati fallito:', err);
      })
      .finally(() => {
        if (!cancelled) setLoadingSavedQuizzes(false);
      });
    return () => {
      cancelled = true;
    };
  }, [subject.id]);

  // Deep-link: carica quiz salvato dall'Assistente
  useEffect(() => {
    if (initialQuizId == null) return;
    let cancelled = false;

    setActiveMode('quiz');
    setIsLoadingSavedQuiz(true);
    setIsGeneratingQuiz(false);
    setQuizError(null);
    setActiveQuizMeta(null);
    setQuestions([]);
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setIsQuizCompleted(false);

    api.study
      .getQuizSet(initialQuizId)
      .then((quizSet) => {
        if (cancelled) return;
        applySavedQuizSet(quizSet);
        setSavedQuizSets((prev) =>
          prev.some((q) => q.id === quizSet.id) ? prev : [quizSet, ...prev]
        );
        onClearInitialQuiz?.();
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setIsLoadingSavedQuiz(false);
        setQuizError(
          err instanceof Error
            ? err.message
            : "Impossibile caricare il quiz salvato dall'Assistente."
        );
        onClearInitialQuiz?.();
      });

    return () => {
      cancelled = true;
    };
  }, [initialQuizId, applySavedQuizSet, onClearInitialQuiz]);

  // Deep-link: avvia simulazione orale con domanda proposta dall'Assistente
  useEffect(() => {
    if (!initialOralQuestion || lastOralKeyRef.current === initialOralQuestion.key) return;
    lastOralKeyRef.current = initialOralQuestion.key;

    setActiveMode('oral');
    setOralQuestion(initialOralQuestion.question);
    setOralTopic(initialOralQuestion.topic || subject.name);
    setOralResult(null);
    setOralError(null);
    setIsGeneratingOralQ(false);
    setIsEvaluatingOral(false);
    onClearInitialOralQuestion?.();
  }, [initialOralQuestion, subject.name, onClearInitialOralQuestion]);

  const hasInitializedLecturesRef = useRef(false);
  const prevSubjectIdRef = useRef(subject.id);

  // Reset selezione lezioni al cambio materia
  useEffect(() => {
    if (prevSubjectIdRef.current !== subject.id) {
      prevSubjectIdRef.current = subject.id;
      hasInitializedLecturesRef.current = false;
      setSelectedLectureIds([]);
    }
  }, [subject.id]);

  // Pre-seleziona tutte le lezioni solo al primo caricamento; sui poll successivi
  // rimuove eventuali ID obsoleti (lezioni eliminate) senza mai annullare "Deseleziona"
  useEffect(() => {
    if (lectures.length === 0) {
      setSelectedLectureIds((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    if (!hasInitializedLecturesRef.current) {
      hasInitializedLecturesRef.current = true;
      setSelectedLectureIds(lectures.map((l) => l.id));
      return;
    }
    const validIds = new Set(lectures.map((l) => l.id));
    setSelectedLectureIds((prev) => {
      const pruned = prev.filter((id) => validIds.has(id));
      return pruned.length === prev.length ? prev : pruned;
    });
  }, [lectures]);

  const getValidLectureIds = (): number[] | undefined => {
    const valid = selectedLectureIds.filter((id) => lectures.some((l) => l.id === id));
    return valid.length > 0 ? valid : undefined;
  };

  const cancelVoiceRecording = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setIsRecordingAnswer(false);
    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.onstop = null;
        mediaRecorderRef.current.onerror = null;
        mediaRecorderRef.current.ondataavailable = null;
        if (mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
        mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      } catch {}
      mediaRecorderRef.current = null;
    }
    audioChunksRef.current = [];
  }, []);

  // Cleanup globale al dismount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (killNoticeTimeoutRef.current) {
        clearTimeout(killNoticeTimeoutRef.current);
        killNoticeTimeoutRef.current = null;
      }
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      if (mediaRecorderRef.current) {
        try {
          mediaRecorderRef.current.onstop = null;
          mediaRecorderRef.current.onerror = null;
          mediaRecorderRef.current.ondataavailable = null;
          if (mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
          }
          mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
        } catch {}
        mediaRecorderRef.current = null;
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // ── Selezione Lezioni ──────────────────────────────────────────────────────
  const toggleLectureSelection = (id: number) => {
    setSelectedLectureIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAllLectures = () => {
    setSelectedLectureIds(lectures.map((l) => l.id));
  };

  const deselectAllLectures = () => {
    hasInitializedLecturesRef.current = true;
    setSelectedLectureIds([]);
  };

  // ── Eliminazione Quiz Salvato (Inline, zero popup Safari) ───────────────────
  const handleDeleteSavedQuiz = async (quizId: number) => {
    try {
      setDeletingQuizId(quizId);
      await api.study.deleteQuizSet(quizId);
      setSavedQuizSets((prev) => prev.filter((q) => q.id !== quizId));
      setConfirmDeleteQuizId(null);
      if (activeQuizMeta?.id === quizId) {
        setActiveMode('none');
        setActiveQuizMeta(null);
        setQuestions([]);
      }
    } catch (err) {
      console.error('Errore durante eliminazione quiz:', err);
    } finally {
      setDeletingQuizId(null);
    }
  };

  // ── Kill Processo IA & Eliminazione Cache ───────────────────────────────────
  const handleKillAIProcess = async () => {
    setActiveMode('none');
    setIsGeneratingQuiz(false);
    setIsLoadingSavedQuiz(false);
    setIsGeneratingOralQ(false);
    setIsEvaluatingOral(false);
    setIsRecordingAnswer(false);
    setIsQuizCompleted(false);
    setActiveQuizMeta(null);
    setQuestions([]);
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setOralQuestion('');
    setOralResult(null);
    setQuizError(null);
    setOralError(null);
    setIsKillingProcess(true);

    cancelVoiceRecording();

    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {}
      abortControllerRef.current = null;
    }

    try {
      await api.study.abort(subject.id).catch(() => {});
      if (!isMountedRef.current) return;
      if (killNoticeTimeoutRef.current) {
        clearTimeout(killNoticeTimeoutRef.current);
      }
      setKillNotice('Processo IA interrotto e cache Metal liberata.');
      killNoticeTimeoutRef.current = setTimeout(() => {
        if (isMountedRef.current) setKillNotice(null);
        killNoticeTimeoutRef.current = null;
      }, 3000);
    } finally {
      if (isMountedRef.current) setIsKillingProcess(false);
    }
  };

  // ── 1. Avvio Quiz a Scelta Multipla ────────────────────────────────────────
  const handleStartQuiz = () => {
    setQuizError(null);
    setActiveMode('quiz');
    executeQuizGeneration();
  };

  const executeQuizGeneration = async () => {
    setIsGeneratingQuiz(true);
    setIsLoadingSavedQuiz(false);
    setActiveQuizMeta(null);
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
        return;
      }
      setQuizError(err?.message || "Errore durante la generazione del quiz con l'IA locale");
    } finally {
      setIsGeneratingQuiz(false);
      abortControllerRef.current = null;
    }
  };

  const handleSelectOption = (idx: number) => {
    if (selectedOpt !== null) return;
    setSelectedOpt(idx);
    const currQ = questions[currentIdx];
    const correct = currQ.correct_index ?? (currQ as any).correctIndex ?? 0;
    const isCorrect = idx === correct;
    if (isCorrect) {
      setScore((s) => s + 1);
    }
    const item: QuizResultItem = {
      question: currQ.question,
      concept: currQ.concept || '',
      options: currQ.options,
      selected_index: idx,
      correct_index: correct,
      is_correct: isCorrect,
      explanation: currQ.explanation || '',
    };
    setUserAnswers((prev) => {
      const copy = [...prev];
      copy[currentIdx] = item;
      return copy;
    });
  };

  const handleNextQuestion = () => {
    if (currentIdx + 1 < questions.length) {
      setCurrentIdx((i) => i + 1);
      setSelectedOpt(null);
    } else {
      setIsQuizCompleted(true);
      if (activeQuizMeta?.id) {
        const finalPercent = questions.length > 0 ? Math.round((score / questions.length) * 100) : 0;
        api.study
          .submitQuizResults(activeQuizMeta.id, {
            score_percent: finalPercent,
            results: userAnswers,
          })
          .then(() => {
            void loadMastery();
          })
          .catch((err) => console.warn('[Quiz] Salvataggio risultati fallito:', err));
      }
    }
  };

  const handleRestartQuiz = () => {
    setCurrentIdx(0);
    setSelectedOpt(null);
    setScore(0);
    setUserAnswers([]);
    setIsQuizCompleted(false);
  };

  const handleExpandQuiz = async () => {
    if (!activeQuizMeta?.id) {
      executeQuizGeneration();
      return;
    }
    setIsGeneratingQuiz(true);
    setQuizError(null);
    try {
      const updated = await api.study.expandQuizSet(activeQuizMeta.id, 5);
      const prevCount = questions.length;
      applySavedQuizSet(updated);
      setSavedQuizSets((prev) =>
        prev.map((q) => (q.id === updated.id ? updated : q))
      );
      setCurrentIdx(prevCount);
      setSelectedOpt(null);
      setIsQuizCompleted(false);
    } catch (err: any) {
      setQuizError(err?.message || "Errore durante l'aggiunta di nuove domande al quiz.");
    } finally {
      setIsGeneratingQuiz(false);
    }
  };

  // ── 2. Avvio Interrogazione Vocale ──────────────────────────────────────────
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
      setOralError(err?.message || "Errore durante la valutazione della risposta con l'IA locale.");
    } finally {
      setIsEvaluatingOral(false);
      abortControllerRef.current = null;
    }
  };

  const currentQ = questions[currentIdx];
  const currentCorrectIdx = currentQ ? (currentQ.correct_index ?? (currentQ as any).correctIndex ?? 0) : 0;

  // ═════════════════════════════════════════════════════════════════════════════
  // SCHERMATA DI VERIFICA INIZIALE
  // ═════════════════════════════════════════════════════════════════════════════
  if (isCheckingReadiness || showReadinessSkeleton) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 select-none relative content-canvas">
        {showReadinessSkeleton ? (
          <QuizWizardSkeleton />
        ) : (
          <div className="min-h-[320px]" aria-busy="true" />
        )}
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // SCHERMATA 0: Rilevamento LLM Mancante
  // ═════════════════════════════════════════════════════════════════════════════
  if (
    !checkingModels &&
    !loadingSavedQuizzes &&
    !hasInstalledLLM &&
    activeMode === 'none' &&
    savedQuizSets.length === 0 &&
    !isLoadingSavedQuiz &&
    initialQuizId == null &&
    !initialOralQuestion
  ) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-12 select-none relative content-canvas">
        <div className="max-w-md w-full p-8 rounded-3xl liquid-glass border border-white/[0.12] shadow-[0_24px_60px_rgba(0,0,0,0.6)] text-center space-y-6 animate-scale-in">
          <div className="space-y-2">
            <h2 className="text-xl font-bold tracking-tight text-white/95">
              Installa un modello per continuare
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
              Ottimizzato per Apple Silicon Metal. Elaborazione interamente sul tuo dispositivo.
            </p>
          </div>

          {onOpenSettings && (
            <button
              type="button"
              onClick={onOpenSettings}
              className="w-full py-3.5 px-5 rounded-full bg-white/[0.14] hover:bg-white/[0.22] active:scale-95 text-xs font-semibold text-white border border-white/[0.20] shadow-liquid-md transition-all flex items-center justify-center gap-2 cursor-pointer group"
            >
              <Settings size={15} className="text-amber-300 group-hover:rotate-45 transition-transform" />
              <span>Vai alle impostazioni e scarica un modello</span>
              <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
            </button>
          )}
        </div>
      </div>
    );
  }

  // ═════════════════════════════════════════════════════════════════════════════
  // CORPO PRINCIPALE
  // ═════════════════════════════════════════════════════════════════════════════
  return (
    <div className="flex-1 flex flex-col overflow-y-auto content-canvas select-none relative p-6 sm:p-8">
      {/* Toast Notifica Kill Processo */}
      {killNotice && (
        <div className="fixed top-6 left-1/2 -translate-x-1/2 z-50 animate-slide-down">
          <div className="px-4 py-2 rounded-full liquid-glass border border-emerald-500/30 text-emerald-300 text-xs font-medium flex items-center gap-2 shadow-[0_10px_30px_rgba(0,0,0,0.6)] backdrop-blur-2xl">
            <Check size={14} className="text-emerald-400" />
            <span>{killNotice}</span>
          </div>
        </div>
      )}

      <div className="max-w-5xl w-full mx-auto flex-1 flex flex-col justify-start">
        {/* ══════════════════════════════════════════════════════════════════════
            1. DASHBOARD PRINCIPALE: LIQUID GLASS 2-COLONNE
            Visualizzata quando activeMode === 'none'
        ══════════════════════════════════════════════════════════════════════ */}
        {activeMode === 'none' && (
          <div className="space-y-6 animate-fade-in py-1">
            {/* Header Sezione + Pill "Concetti" con Popover Spring Dropdown */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-1">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white/95">
                  Ripasso
                </h1>
                <p className="text-xs text-white/50 mt-1">
                  Simulazioni d&apos;esame e quiz sui tuoi materiali.
                </p>
              </div>

              {/* Pill Singola "Concetti" con Popover Spring Animato */}
              <div className="relative self-start sm:self-center">
                <button
                  ref={masteryButtonRef}
                  type="button"
                  onClick={() => setShowMasteryDrawer((prev) => !prev)}
                  className={clsx(
                    "inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all select-none cursor-pointer shadow-liquid-sm group",
                    showMasteryDrawer
                      ? "bg-white/[0.12] text-white border border-white/[0.22] shadow-liquid-md"
                      : "bg-white/[0.05] hover:bg-white/[0.09] text-white/80 hover:text-white border border-white/[0.10]"
                  )}
                >
                  <GraduationCap size={14} className="text-amber-400 group-hover:scale-110 transition-transform" />
                  <span>Concetti</span>
                  {masterySummary && (masterySummary.consolidated_count > 0 || masterySummary.to_consolidate_count > 0) && (
                    <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-white/[0.08] text-white/70">
                      <span className="text-emerald-400 font-semibold">{masterySummary.consolidated_count}</span>
                      <span className="text-white/20">/</span>
                      <span className="text-amber-300 font-semibold">{masterySummary.to_consolidate_count}</span>
                    </span>
                  )}
                  <ChevronDown
                    size={12}
                    className={clsx(
                      "text-white/40 group-hover:text-white/70 transition-transform duration-200",
                      showMasteryDrawer && "rotate-180"
                    )}
                  />
                </button>

                {/* Popover a Bolla Liquid Glass (stesso standard di Quaderno / Esporta) */}
                <AnimatePresence>
                  {showMasteryDrawer && (
                    <motion.div
                      ref={masteryDrawerRef}
                      variants={liquidMenuVariants}
                      initial="initial"
                      animate="animate"
                      exit="exit"
                      style={{ transformOrigin: 'top right' }}
                      className={clsx(
                        "absolute right-0 top-full mt-2 w-[380px] h-[375px] max-w-[calc(100vw-2rem)] flex flex-col p-3 gap-2.5",
                        liquidPopoverContainerClass
                      )}
                      onClick={(e) => e.stopPropagation()}
                    >
                      {/* Filtri a Schede (Segmented Tabs) + Tasto Azzera compatto (Zero ripetizioni, zero X superflua) */}
                      {masterySummary && masterySummary.total_tracked > 0 && (
                        <div className="flex items-center gap-2 pb-1.5 border-b border-white/[0.08] shrink-0">
                          <div className="flex-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center gap-1 text-[11px]">
                            <button
                              type="button"
                              onClick={() => setConceptFilter('all')}
                              className={clsx(
                                "flex-1 py-1 px-1.5 rounded-lg font-medium transition-all text-center cursor-pointer select-none whitespace-nowrap text-[11px]",
                                conceptFilter === 'all'
                                  ? "bg-white/[0.12] text-white shadow-liquid-xs border border-white/[0.12]"
                                  : "text-white/60 hover:text-white border border-transparent"
                              )}
                            >
                              Tutti ({masterySummary.total_tracked})
                            </button>
                            <button
                              type="button"
                              onClick={() => setConceptFilter('da_consolidare')}
                              className={clsx(
                                "flex-1 py-1 px-1.5 rounded-lg font-medium transition-all text-center flex items-center justify-center gap-1 cursor-pointer select-none whitespace-nowrap text-[11px]",
                                conceptFilter === 'da_consolidare'
                                  ? "bg-amber-500/20 text-amber-200 border border-amber-500/30 shadow-liquid-xs"
                                  : "text-amber-300/70 hover:text-amber-200 border border-transparent"
                              )}
                            >
                              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                              <span>Da rivedere ({masterySummary.to_consolidate_count})</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => setConceptFilter('consolidato')}
                              className={clsx(
                                "flex-1 py-1 px-1.5 rounded-lg font-medium transition-all text-center flex items-center justify-center gap-1 cursor-pointer select-none whitespace-nowrap text-[11px]",
                                conceptFilter === 'consolidato'
                                  ? "bg-emerald-500/20 text-emerald-200 border border-emerald-500/30 shadow-liquid-xs"
                                  : "text-emerald-300/70 hover:text-emerald-200 border border-transparent"
                              )}
                            >
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                              <span>Consolidati ({masterySummary.consolidated_count})</span>
                            </button>
                          </div>

                          {confirmClearAllConcepts ? (
                            <div className="flex items-center gap-1 px-2 py-1 rounded-xl bg-red-500/20 border border-red-500/35 text-[10px] animate-scale-in shrink-0">
                              <span className="text-red-200 font-medium whitespace-nowrap">Azzera?</span>
                              <button
                                type="button"
                                disabled={isClearingAllConcepts}
                                onClick={() => void handleClearAllConcepts()}
                                className="px-1.5 py-0.5 rounded bg-red-500/40 hover:bg-red-500/60 text-red-100 font-bold transition cursor-pointer"
                              >
                                {isClearingAllConcepts ? <Loader2 size={9} className="animate-spin" /> : 'Sì'}
                              </button>
                              <button
                                type="button"
                                onClick={() => setConfirmClearAllConcepts(false)}
                                className="p-0.5 rounded text-white/40 hover:text-white transition cursor-pointer"
                              >
                                <X size={10} />
                              </button>
                            </div>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setConfirmClearAllConcepts(true)}
                              title="Azzera tutti i concetti"
                              className="p-2 rounded-xl text-white/40 hover:text-red-400 hover:bg-red-500/15 border border-white/[0.06] hover:border-red-500/25 transition cursor-pointer flex-shrink-0"
                            >
                              <Trash2 size={12} />
                            </button>
                          )}
                        </div>
                      )}

                      {/* 3. Lista Concetti (Altezza stabile flex-1 min-h-0: zero scatti o ridimensionamenti al cambio tab) */}
                      {(!masterySummary || masterySummary.concepts.length === 0) ? (
                        <div className="flex-1 min-h-0 flex flex-col items-center justify-center py-4 px-4 text-center space-y-2">
                          <GraduationCap size={24} className="text-white/20 mx-auto" />
                          <p className="text-xs text-white/70 font-semibold">Nessun concetto tracciato</p>
                          <p className="text-[11px] text-white/40 leading-relaxed max-w-xs mx-auto">
                            Completa un quiz per tracciare automaticamente i concetti compresi e quelli da consolidare.
                          </p>
                        </div>
                      ) : filteredConcepts.length === 0 ? (
                        <div className="flex-1 min-h-0 flex flex-col items-center justify-center py-4 px-4 text-center space-y-1">
                          <p className="text-xs text-white/60 font-medium">Nessun concetto in questa categoria</p>
                          <p className="text-[11px] text-white/40">
                            {conceptFilter === 'da_consolidare'
                              ? 'Tutti i concetti risultano consolidati! Ottimo lavoro.'
                              : 'Non hai ancora concetti consolidati in questo corso.'}
                          </p>
                        </div>
                      ) : (
                        <div className="flex-1 min-h-0 overflow-y-auto space-y-1.5 pr-1 custom-scrollbar">
                          {filteredConcepts.map((c) => {
                            const isConsolidated = c.status === 'consolidato';
                            const displayTopic = formatConceptTopic(c.topic);
                            const subtitle = isConsolidated
                              ? (c.correct_count > 0 ? `${c.correct_count} ${c.correct_count === 1 ? 'risposta corretta' : 'risposte corrette'}` : 'Concetto compreso')
                              : (c.error_count > 0 ? `${c.error_count} ${c.error_count === 1 ? 'errore da consolidare' : 'errori da consolidare'}` : 'Da ripassare nei prossimi quiz');

                            return (
                              <div
                                key={c.id}
                                className="group p-2.5 rounded-2xl bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.06] hover:border-white/[0.14] flex items-center justify-between gap-3 text-xs transition-all shadow-liquid-xs"
                              >
                                <div className="flex items-center gap-2.5 min-w-0 flex-1">
                                  <div
                                    className={clsx(
                                      "w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0 border",
                                      isConsolidated
                                        ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.15)]"
                                        : "bg-amber-500/10 border-amber-500/25 text-amber-400 shadow-[0_0_12px_rgba(245,158,11,0.15)]"
                                    )}
                                  >
                                    {isConsolidated ? <CheckCircle2 size={13} /> : <AlertCircle size={13} />}
                                  </div>

                                  <div className="min-w-0 flex-1">
                                    <div className="font-semibold text-white/95 truncate tracking-tight text-[12px]">
                                      {displayTopic}
                                    </div>
                                    <div className="text-[10px] text-white/45 truncate font-mono mt-0.5">
                                      {subtitle}
                                    </div>
                                  </div>
                                </div>

                                <div className="flex items-center gap-1.5 flex-shrink-0">
                                  {/* Badge / Bottone Interattivo di Cambio Stato */}
                                  <button
                                    type="button"
                                    disabled={togglingConceptId === c.id}
                                    onClick={() => void handleToggleConceptStatus(c)}
                                    title={isConsolidated ? "Clicca per segnare da consolidare" : "Clicca per segnare consolidato"}
                                    className={clsx(
                                      'text-[10px] font-mono font-medium px-2.5 py-0.5 rounded-full border transition-all cursor-pointer select-none flex items-center gap-1',
                                      isConsolidated
                                        ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/25'
                                        : 'bg-amber-500/15 border-amber-500/30 text-amber-300 hover:bg-amber-500/25'
                                    )}
                                  >
                                    {togglingConceptId === c.id && <Loader2 size={9} className="animate-spin" />}
                                    <span>{isConsolidated ? 'Consolidato' : 'Da rivedere'}</span>
                                  </button>

                                  {/* Tasto Eliminazione Singolo Concetto */}
                                  <button
                                    type="button"
                                    disabled={deletingConceptId === c.id}
                                    onClick={() => void handleDeleteConcept(c.id)}
                                    title="Elimina definitivamente concetto"
                                    className="p-1 rounded-lg text-white/30 hover:text-red-400 hover:bg-red-500/15 border border-transparent hover:border-red-500/25 transition-all cursor-pointer opacity-0 group-hover:opacity-100 flex-shrink-0"
                                  >
                                    {deletingConceptId === c.id ? (
                                      <Loader2 size={11} className="animate-spin text-red-300" />
                                    ) : (
                                      <Trash2 size={11} />
                                    )}
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* 4. Footer descrittivo */}
                      {masterySummary && masterySummary.total_tracked > 0 && (
                        <div className="pt-2 border-t border-white/[0.06] text-[10px] text-white/40 text-center leading-relaxed shrink-0">
                          L&apos;IA adatta le spiegazioni e i quiz in base ai concetti da consolidare.
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* Layout 2 Colonne Bilanciato */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* COLONNA SINISTRA (7 col): Selettore Lezioni + Azioni Rapide 1-Click */}
              <div className="lg:col-span-7 flex flex-col space-y-4">
                {/* Scheda Lista Lezioni */}
                <div className="p-5 rounded-3xl liquid-glass border border-white/[0.10] shadow-[0_16px_40px_rgba(0,0,0,0.4)] space-y-3.5 flex-1 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                      <div className="flex items-center gap-2">
                        <BookOpen size={14} className="text-amber-400" />
                        <span className="text-xs font-semibold text-white/85">
                          Materiali
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
                          Tutte
                        </button>
                        <span className="text-white/20">•</span>
                        <button
                          type="button"
                          onClick={deselectAllLectures}
                          className="text-[11px] text-white/50 hover:text-white transition-colors cursor-pointer"
                        >
                          Nessuna
                        </button>
                      </div>
                    </div>

                    {lectures.length === 0 ? (
                      <div className="py-8 text-center space-y-2">
                        <BookOpen size={24} className="text-white/20 mx-auto" />
                        <h4 className="text-xs font-semibold text-white/80">Nessuna lezione registrata</h4>
                        <p className="text-[11px] text-white/40 max-w-xs mx-auto leading-relaxed">
                          Registra o importa una lezione in {subject.name} per formulare domande d&apos;esame.
                        </p>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 gap-2 pt-3 max-h-56 overflow-y-auto pr-1">
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
                                  ? 'bg-white/[0.09] border-white/[0.20] text-white shadow-liquid-sm'
                                  : 'bg-white/[0.02] border-white/[0.05] text-white/60 hover:bg-white/[0.05] hover:border-white/[0.10]'
                              )}
                            >
                              <div className="flex items-center gap-3 min-w-0">
                                <div className={clsx(
                                  'w-4 h-4 rounded-md border flex items-center justify-center transition-all flex-shrink-0',
                                  isSelected
                                    ? 'bg-amber-400/20 border-amber-400/50 text-amber-300'
                                    : 'border-white/20 text-transparent group-hover:border-white/40'
                                )}>
                                  <Check size={11} strokeWidth={2.5} />
                                </div>
                                <div className="min-w-0">
                                  <h4 className={clsx(
                                    'text-xs font-medium tracking-tight truncate',
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
                                'text-[9px] font-mono px-2 py-0.5 rounded-full border flex-shrink-0',
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
                    )}
                  </div>
                </div>

                {/* Due Tessere Azione Diretta (Quiz vs Orale) */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Azione 1: Quiz a Scelta Multipla */}
                  <div
                    onClick={selectedLectureIds.length > 0 ? handleStartQuiz : undefined}
                    className={clsx(
                      'p-5 rounded-3xl liquid-glass border transition-all flex flex-col justify-between group shadow-liquid-sm select-none',
                      selectedLectureIds.length > 0
                        ? 'border-blue-500/25 hover:border-blue-400/50 hover:bg-white/[0.08] hover:scale-[1.01] active:scale-95 cursor-pointer bg-gradient-to-br from-blue-500/[0.08] to-transparent'
                        : 'border-white/[0.06] bg-white/[0.02] opacity-50 cursor-not-allowed'
                    )}
                  >
                    <div className="space-y-3">
                      <div className="w-11 h-11 rounded-2xl liquid-glass-subtle border border-blue-500/30 flex items-center justify-center text-blue-400 group-hover:scale-110 transition-transform shadow-[0_0_16px_rgba(59,130,246,0.2)]">
                        <HelpCircle size={22} strokeWidth={1.8} />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold tracking-tight text-white/95">
                          Quiz a Scelta Multipla
                        </h3>
                        <p className="text-[11px] text-white/50 mt-1 leading-relaxed">
                          5 domande formulate dai tuoi appunti, spiegazioni ragionate e tracciamento errori.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-xs font-semibold text-blue-300 group-hover:text-blue-200">
                      <span>Avvia Quiz</span>
                      <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>

                  {/* Azione 2: Colloquio Orale con IA */}
                  <div
                    onClick={selectedLectureIds.length > 0 ? handleStartOralMode : undefined}
                    className={clsx(
                      'p-5 rounded-3xl liquid-glass border transition-all flex flex-col justify-between group shadow-liquid-sm select-none',
                      selectedLectureIds.length > 0
                        ? 'border-emerald-500/25 hover:border-emerald-400/50 hover:bg-white/[0.08] hover:scale-[1.01] active:scale-95 cursor-pointer bg-gradient-to-br from-emerald-500/[0.08] to-transparent'
                        : 'border-white/[0.06] bg-white/[0.02] opacity-50 cursor-not-allowed'
                    )}
                  >
                    <div className="space-y-3">
                      <div className="w-11 h-11 rounded-2xl liquid-glass-subtle border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform shadow-[0_0_16px_rgba(16,185,129,0.2)]">
                        <Mic size={22} strokeWidth={1.8} />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold tracking-tight text-white/95">
                          Colloquio Orale
                        </h3>
                        <p className="text-[11px] text-white/50 mt-1 leading-relaxed">
                          Domanda d&apos;esame vocale a sorpresa. Rispondi al microfono e ricevi un voto in 30esimi.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-xs font-semibold text-emerald-300 group-hover:text-emerald-200">
                      <span>Simula Orale</span>
                      <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
                    </div>
                  </div>
                </div>
              </div>

              {/* COLONNA DESTRA (5 col): Archivio Quiz */}
              <div className="lg:col-span-5 flex flex-col">
                <div className="p-5 rounded-3xl liquid-glass border border-white/[0.10] shadow-[0_16px_40px_rgba(0,0,0,0.4)] space-y-4 flex-1 flex flex-col">
                  <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                    <div className="flex items-center gap-2">
                      <Sparkles size={14} className="text-blue-400" />
                      <span className="text-xs font-semibold text-white/85">
                        Quiz
                      </span>
                    </div>
                    {savedQuizSets.length > 0 && (
                      <span className="px-2 py-0.5 rounded-full bg-white/[0.08] text-[10px] font-mono text-white/60">
                        {savedQuizSets.length} {savedQuizSets.length === 1 ? 'quiz' : 'quiz'}
                      </span>
                    )}
                  </div>

                  {/* Lista o Empty State */}
                  {savedQuizSets.length === 0 && showSavedListLoading ? (
                    <div className="flex-1 flex items-center justify-center gap-2 text-xs text-white/40 py-12">
                      <Loader2 size={14} className="animate-spin text-blue-400" />
                      <span>Caricamento archivio quiz...</span>
                    </div>
                  ) : savedQuizSets.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center p-8 text-center space-y-2">
                      <HelpCircle size={26} className="text-white/20" />
                      <h4 className="text-xs font-semibold text-white/70">Nessun quiz salvato</h4>
                      <p className="text-[11px] text-white/40 max-w-xs leading-relaxed">
                        I quiz generati compariranno qui per essere ripassati.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1 flex-1">
                      {savedQuizSets.map((qs) => {
                        const displayTitle = formatQuizDisplayTitle(qs.title);
                        const sourceInfo = formatQuizSourceLabel(qs.source_label);
                        const isConfirming = confirmDeleteQuizId === qs.id;

                        return (
                          <div
                            key={qs.id}
                            onClick={() => {
                              if (!isConfirming) {
                                applySavedQuizSet(qs);
                              }
                            }}
                            className="p-3.5 rounded-2xl bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.07] hover:border-blue-400/35 transition-all flex items-center justify-between gap-3 text-left group cursor-pointer shadow-liquid-xs select-none"
                          >
                            <div className="min-w-0 flex-1">
                              <h4 className="text-xs font-semibold text-white/90 truncate group-hover:text-white">
                                {displayTitle}
                              </h4>
                              <div className="flex items-center gap-1.5 mt-1 text-[10px] font-mono">
                                {sourceInfo.primary && (
                                  <span className="truncate text-blue-300/80 font-medium">
                                    {sourceInfo.primary}
                                  </span>
                                )}
                                {sourceInfo.extraCount > 0 && (
                                  <span className="text-[9px] text-white/40 font-mono flex-shrink-0">
                                    +{sourceInfo.extraCount}
                                  </span>
                                )}
                                {(sourceInfo.primary || sourceInfo.extraCount > 0) && (
                                  <span className="text-white/20">·</span>
                                )}
                                <span className="flex-shrink-0 text-white/50">{qs.questions.length} domande</span>
                                <span className="text-white/20">·</span>
                                <span className="flex-shrink-0 text-white/40">
                                  {new Date(qs.created_at).toLocaleDateString('it-IT', { day: 'numeric', month: 'short' })}
                                </span>
                              </div>
                            </div>

                            <div className="flex items-center gap-2 flex-shrink-0">
                              {/* Conferma Eliminazione Inline (Zero popup Safari, compatto e non distorce la riga) */}
                              {isConfirming ? (
                                <div
                                  className="flex items-center gap-1.5 px-2 py-0.5 rounded-xl bg-red-500/20 border border-red-500/35 text-xs animate-scale-in"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <span className="text-[10px] text-red-200 font-medium whitespace-nowrap">Eliminare?</span>
                                  <button
                                    type="button"
                                    disabled={deletingQuizId === qs.id}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      void handleDeleteSavedQuiz(qs.id);
                                    }}
                                    className="px-2 py-0.5 rounded-lg bg-red-500/40 hover:bg-red-500/60 text-[10px] text-red-100 font-bold transition cursor-pointer"
                                  >
                                    {deletingQuizId === qs.id ? (
                                      <Loader2 size={10} className="animate-spin" />
                                    ) : (
                                      'Sì'
                                    )}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setConfirmDeleteQuizId(null);
                                    }}
                                    className="p-1 rounded-md text-white/40 hover:text-white hover:bg-white/[0.08] transition cursor-pointer"
                                  >
                                    <X size={11} />
                                  </button>
                                </div>
                              ) : (
                                <>
                                  {/* Score Pill — Trofeo solo per punteggi alti (>= 70%) */}
                                  {qs.score_percent !== null && qs.score_percent !== undefined ? (
                                    <span
                                      className={clsx(
                                        'px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold border flex items-center gap-1',
                                        qs.score_percent >= 70
                                          ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                                          : 'bg-white/[0.06] border-white/[0.12] text-white/70'
                                      )}
                                    >
                                      {qs.score_percent >= 70 && <Award size={10} />}
                                      <span>{qs.score_percent}%</span>
                                    </span>
                                  ) : qs.completed ? (
                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-blue-500/15 border border-blue-500/30 text-blue-300">
                                      Completato
                                    </span>
                                  ) : (
                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/[0.05] border border-white/[0.10] text-white/50">
                                      Nuovo
                                    </span>
                                  )}

                                  {/* Tasto Cestino compatto su Hover (senza freccia ridondante a fianco) */}
                                  <button
                                    type="button"
                                    title="Elimina quiz"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setConfirmDeleteQuizId(qs.id);
                                    }}
                                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-white/40 hover:text-red-400 hover:bg-red-500/15 border border-transparent hover:border-red-500/30 transition-all cursor-pointer flex-shrink-0"
                                  >
                                    <Trash2 size={13} />
                                  </button>
                                </>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            2. WORKSPACE ATTIVO: QUIZ A SCELTA MULTIPLA
            Visualizzato quando activeMode === 'quiz'
        ══════════════════════════════════════════════════════════════════════ */}
        {activeMode === 'quiz' && (
          <div className="max-w-3xl w-full mx-auto my-auto py-4 animate-scale-in">
            <div className="rounded-3xl liquid-glass border border-blue-500/30 bg-white/[0.05] shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-blue-500/20 p-6 sm:p-8 space-y-5">
              {/* Top Navigation Bar Unificata (1 sola riga essenziale) */}
              <div className="flex items-center justify-between pb-3.5 border-b border-white/[0.08] gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <button
                    type="button"
                    onClick={() => {
                      if (isGeneratingQuiz) {
                        handleKillAIProcess();
                      } else {
                        setActiveMode('none');
                        setIsLoadingSavedQuiz(false);
                        setActiveQuizMeta(null);
                        setQuizError(null);
                        setQuestions([]);
                        setCurrentIdx(0);
                        setSelectedOpt(null);
                        setScore(0);
                        setIsQuizCompleted(false);
                      }
                    }}
                    className="text-xs text-white/60 hover:text-white transition-colors flex items-center gap-1.5 cursor-pointer py-1 select-none flex-shrink-0 group"
                  >
                    <ArrowLeft size={13} className="group-hover:-translate-x-0.5 transition-transform" />
                    <span className="font-medium">Ripasso</span>
                  </button>

                  {activeQuizMeta && (
                    <>
                      <span className="text-white/20 select-none">•</span>
                      <span className="text-xs text-white/80 font-medium truncate">
                        {formatQuizDisplayTitle(activeQuizMeta.title)}
                      </span>
                    </>
                  )}
                </div>

                {!isGeneratingQuiz && !isLoadingSavedQuiz && !showSavedQuizSkeleton && questions.length > 0 && !isQuizCompleted && (
                  <div className="flex items-center gap-2 text-xs font-mono text-white/50 flex-shrink-0">
                    <span>{currentIdx + 1}/{questions.length}</span>
                    <span className="text-white/20">•</span>
                    <span className="text-amber-300 font-semibold">{score} corrette</span>
                  </div>
                )}
              </div>

              {/* STATO 0: Caricamento Quiz salvato dall'Assistente */}
              {isLoadingSavedQuiz || showSavedQuizSkeleton ? (
                showSavedQuizSkeleton ? (
                  <QuizCardSkeleton />
                ) : (
                  <div className="min-h-[280px]" aria-busy="true" />
                )
              ) : isGeneratingQuiz ? (
                <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center space-y-6">
                  <div className="w-16 h-16 rounded-full bg-blue-500/10 border border-blue-500/30 flex items-center justify-center text-blue-400 mx-auto shadow-[0_0_30px_rgba(59,130,246,0.2)]">
                    <Loader2 size={30} className="animate-spin text-blue-400" />
                  </div>

                  <div className="space-y-1.5">
                    <h3 className="text-lg font-semibold text-white/95">
                      Generazione Quiz in corso...
                    </h3>
                    <p className="text-xs text-white/50 max-w-sm mx-auto leading-relaxed">
                      L&apos;IA locale sta analizzando i materiali del corso per formulare le domande...
                    </p>
                  </div>

                  {/* Shimmering Skeleton Preview */}
                  <div className="w-full max-w-md mx-auto space-y-2.5 pt-2">
                    <SkeletonPill className="h-3 w-5/6 mx-auto" />
                    <SkeletonPill className="h-3 w-full" />
                    <SkeletonPill className="h-3 w-4/6 mx-auto" />
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
                      Ferma l&apos;inferenza immediatamente e svuota la memoria Metal/GPU
                    </span>
                  </div>
                </div>
              ) : quizError ? (
                /* Errore generazione */
                <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-red-500/30 text-center space-y-5">
                  <AlertCircle size={32} className="text-red-400 mx-auto" />
                  <div>
                    <h3 className="text-sm font-semibold text-white">Generazione Non Riuscita</h3>
                    <p className="text-xs text-red-300/80 mt-1 max-w-sm mx-auto">{quizError}</p>
                  </div>
                  <div className="flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => {
                        setActiveMode('none');
                        setQuizError(null);
                        setActiveQuizMeta(null);
                      }}
                      className="px-4 py-2 rounded-full liquid-glass hover:bg-white/[0.10] text-xs font-medium text-white/80 border border-white/[0.12] cursor-pointer"
                    >
                      Torna al Menu
                    </button>
                    <button
                      type="button"
                      onClick={executeQuizGeneration}
                      className="px-4 py-2 rounded-full bg-white/[0.14] hover:bg-white/[0.22] text-xs font-semibold text-white border border-white/[0.20] cursor-pointer"
                    >
                      Riprova
                    </button>
                  </div>
                </div>
              ) : !isQuizCompleted && currentQ ? (
                /* Domanda Attiva del Quiz */
                <div className="space-y-5">
                  <div>
                    <h3 className="text-base sm:text-lg font-semibold tracking-tight text-white leading-snug">
                      {currentQ.question}
                    </h3>
                  </div>

                  <div className="space-y-2">
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
                            'w-full p-3 sm:p-3.5 rounded-xl text-left text-xs sm:text-[13px] tracking-tight transition-all flex items-center gap-3 border cursor-pointer select-none',
                            showResult && isCorrect
                              ? 'bg-emerald-500/15 border-emerald-500/35 text-emerald-100 shadow-[0_0_15px_rgba(16,185,129,0.12)]'
                              : showResult && isChosen && !isCorrect
                              ? 'bg-red-500/15 border-red-500/35 text-red-200'
                              : 'bg-white/[0.03] border-white/[0.08] hover:bg-white/[0.07] hover:border-white/[0.15] text-white/90'
                          )}
                        >
                          <span className={clsx(
                            'w-6 h-6 rounded-lg text-xs font-mono font-medium flex items-center justify-center flex-shrink-0 transition-colors',
                            showResult && isCorrect
                              ? 'bg-emerald-500/25 text-emerald-200'
                              : showResult && isChosen && !isCorrect
                              ? 'bg-red-500/25 text-red-200'
                              : 'bg-white/[0.06] text-white/50'
                          )}>
                            {String.fromCharCode(65 + idx)}
                          </span>
                          <span className="leading-relaxed flex-1">{opt}</span>
                          {showResult && isCorrect && <CheckCircle2 size={16} className="text-emerald-400 flex-shrink-0" />}
                          {showResult && isChosen && !isCorrect && <XCircle size={16} className="text-red-400 flex-shrink-0" />}
                        </button>
                      );
                    })}
                  </div>

                  {selectedOpt !== null && (
                    <div className="pt-3 border-t border-white/[0.06] space-y-3 animate-fade-in">
                      <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] text-xs leading-relaxed text-white/70">
                        {currentQ.explanation}
                      </div>

                      <div className="flex justify-end">
                        <button
                          type="button"
                          onClick={handleNextQuestion}
                          className="px-4 py-2 rounded-xl bg-white/[0.16] hover:bg-white/[0.24] text-white text-xs font-semibold border border-white/[0.20] shadow-liquid-sm transition-all flex items-center gap-1.5 cursor-pointer"
                        >
                          <span>{currentIdx + 1 < questions.length ? 'Prossima' : 'Risultati'}</span>
                          <ArrowRight size={13} strokeWidth={2} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* Risultato Finale Quiz */
                <div className="p-6 sm:p-8 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center space-y-6">
                  <div className="w-14 h-14 rounded-2xl bg-amber-400/10 border border-amber-400/20 flex items-center justify-center text-amber-300 mx-auto">
                    <Award size={26} strokeWidth={1.75} />
                  </div>

                  <div className="space-y-1">
                    <div className="text-3xl sm:text-4xl font-bold tracking-tight text-white font-mono">
                      {questions.length > 0 ? Math.round((score / questions.length) * 100) : 0}%
                    </div>
                    <p className="text-xs text-white/45">
                      {score} su {questions.length} corrette
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                    <button
                      type="button"
                      onClick={() => {
                        onReviewInChat?.({
                          quizId: activeQuizMeta?.id ?? 0,
                          chatSessionId: activeQuizMeta?.chat_session_id ?? null,
                          quizTitle: activeQuizMeta?.title || `Quiz su ${subject.name}`,
                          score,
                          total: questions.length,
                          wrongCount: Math.max(0, questions.length - score),
                        });
                      }}
                      className="px-3.5 py-2 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-xs font-semibold text-emerald-300 border border-emerald-500/30 transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                    >
                      <Sparkles size={13} className="text-emerald-400" />
                      <span>Ripassa in chat</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleRestartQuiz}
                      className="px-3.5 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-xs font-medium text-white/80 border border-white/[0.10] transition-all flex items-center gap-1.5 cursor-pointer"
                    >
                      <RotateCcw size={12} />
                      <span>Rifai</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleExpandQuiz}
                      disabled={isGeneratingQuiz}
                      className="px-3.5 py-2 rounded-xl bg-white/[0.12] hover:bg-white/[0.18] text-xs font-semibold text-white border border-white/[0.16] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm disabled:opacity-50"
                    >
                      {isGeneratingQuiz ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                      <span>Altre domande</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveMode('none');
                        setActiveQuizMeta(null);
                        setQuestions([]);
                      }}
                      className="px-3.5 py-2 rounded-xl bg-transparent hover:bg-white/[0.06] text-xs font-medium text-white/50 hover:text-white border border-transparent hover:border-white/[0.08] transition-all cursor-pointer"
                    >
                      <span>Esci</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════════
            3. WORKSPACE ATTIVO: COLLOQUIO ORALE CON IA
            Visualizzato quando activeMode === 'oral'
        ══════════════════════════════════════════════════════════════════════ */}
        {activeMode === 'oral' && (
          <div className="max-w-3xl w-full mx-auto my-auto py-4 animate-scale-in">
            <div className="rounded-3xl liquid-glass border border-emerald-500/30 bg-white/[0.05] shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-emerald-500/20 p-6 sm:p-8 space-y-5">
              {/* Header Orale */}
              <div className="flex items-center justify-between pb-3 border-b border-white/[0.08]">
                <button
                  type="button"
                  onClick={() => {
                    if (isGeneratingOralQ || isEvaluatingOral) {
                      handleKillAIProcess();
                    } else {
                      cancelVoiceRecording();
                      setActiveMode('none');
                      setOralQuestion('');
                      setOralResult(null);
                      setOralError(null);
                    }
                  }}
                  className="text-xs text-white/60 hover:text-white transition-colors flex items-center gap-1.5 cursor-pointer py-1 select-none"
                >
                  <ArrowLeft size={13} />
                  <span>Torna a Ripasso</span>
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
                        <span>Il docente sta formulando la domanda d&apos;esame...</span>
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
                      &quot;{oralQuestion || 'Descrivi i concetti cardine affrontati nelle lezioni.'}&quot;
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

                {/* Valutazione Orale in corso con Tasto Kill Processo */}
                {isEvaluatingOral && (
                  <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] text-center space-y-4">
                    <div className="flex items-center justify-center gap-3 text-xs text-emerald-300">
                      <Loader2 size={16} className="animate-spin text-emerald-400" />
                      <span>Trascrizione Whisper &amp; Valutazione IA in corso...</span>
                    </div>

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
                    {oralResult.warning && (
                      <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/25 text-xs text-amber-200 flex items-start gap-2">
                        <AlertCircle size={14} className="text-amber-400 flex-shrink-0 mt-0.5" />
                        <span>{oralResult.warning}</span>
                      </div>
                    )}

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

                    {oralResult.student_transcript && (
                      <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-xs text-white/70">
                        <span className="text-[10px] font-mono text-white/40 uppercase block mb-1">Cosa hai detto a voce:</span>
                        <p className="italic">&quot;{oralResult.student_transcript}&quot;</p>
                      </div>
                    )}

                    <div className="p-3.5 rounded-2xl bg-emerald-500/[0.06] border border-emerald-500/20 text-xs text-emerald-200/90 space-y-1">
                      <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider block font-semibold">
                        ✓ Cosa hai detto bene:
                      </span>
                      <p className="leading-relaxed">{oralResult.good_points}</p>
                    </div>

                    <div className="p-3.5 rounded-2xl bg-amber-500/[0.06] border border-amber-500/20 text-xs text-amber-200/90 space-y-1">
                      <span className="text-[10px] font-mono text-amber-400 uppercase tracking-wider block font-semibold">
                        ! Cosa hai dimenticato o imprecisioni:
                      </span>
                      <p className="leading-relaxed">{oralResult.missing_or_errors}</p>
                    </div>

                    <div className="p-3.5 rounded-2xl bg-blue-500/[0.06] border border-blue-500/20 text-xs text-blue-200/90 space-y-1">
                      <span className="text-[10px] font-mono text-blue-400 uppercase tracking-wider block font-semibold">
                        ★ Come riformularla all&apos;esame in modo perfetto:
                      </span>
                      <p className="leading-relaxed">{oralResult.ideal_formulation}</p>
                    </div>

                    <div className="flex justify-end pt-2">
                      <button
                        type="button"
                        onClick={handleGenerateOralQuestion}
                        className="px-4 py-2 rounded-full bg-white/[0.12] hover:bg-white/[0.20] text-xs font-semibold text-white border border-white/[0.16] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                      >
                        <RefreshCw size={13} />
                        <span>Prossima Domanda d&apos;Esame</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
