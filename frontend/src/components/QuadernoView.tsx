import { useState, useMemo, useRef, useEffect } from 'react';
import { 
  Calendar, Clock, FileText, Trash2, Sparkles, 
  ChevronRight, BookOpen, AlignLeft, Plus, Pencil, 
  Check, X, Search, HelpCircle, Square, Loader2,
  PanelLeftClose, PanelLeftOpen, AlertTriangle,
  Play, Pause, RotateCcw, Bookmark as BookmarkIcon,
  FileDown, Music, Share2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import type { Lecture, Subject, DetectedQuestion, Bookmark, KaraokeLine } from '../types';
import { api } from '../lib/api';
import { useSettings } from '../lib/SettingsContext';
import MaterialiFinderModal from './MaterialiFinderModal';
import SegmentedControl from './SegmentedControl';
import MarkdownContent from './MarkdownContent';
import { AppleLyricsView } from './AppleLyricsView';

function normalizeKaraokeLines(raw: any): KaraokeLine[] {
  if (!raw) return [];
  const list = Array.isArray(raw) ? raw : (raw.lines || raw.segments || []);
  if (!Array.isArray(list)) return [];

  const maxEnd = list.length > 0 ? Math.max(...list.map((it: any) => parseFloat(it.lineEnd ?? it.end ?? 0))) : 0;
  const mult = maxEnd > 0 && maxEnd < 100000 ? 1000 : 1;

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
  subject: Subject;
  lectures: Lecture[];
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
  subject, 
  lectures, 
  onDeletePermanently,
  onRemoveFromQuadernoOnly,
  onRestoreToQuaderno,
  onAddLecture, 
  onUpdateLecture,
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

  // State per generazione risposta domanda specifica con IA
  const [answeringQuestionId, setAnsweringQuestionId] = useState<string | null>(null);

  // Modal Eliminazione
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Modal Finder Memoria (Materiali -> Quaderno)
  const [showFinderModal, setShowFinderModal] = useState(false);

  // Edit Mode per lezione selezionata
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [editTab, setEditTab] = useState<'editor' | 'preview'>('editor');
  const [isSavingEdit, setIsSavingEdit] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const isPlaceholderAnswer = (ans?: string) => {
    if (!ans || !ans.trim()) return true;
    const lower = ans.toLowerCase();
    return (
      lower.includes('modello llm non ancora scaricato') ||
      lower.includes('modello llm non installato') ||
      lower.includes('non ancora scaricato') ||
      lower.includes('domanda rilevata dal docente') ||
      lower.includes('risposta non disponibile')
    );
  };

  const handleAnswerQuestion = async (questionId: string) => {
    if (!selectedLecture) return;
    const lectureId = selectedLecture.id;
    setAnsweringQuestionId(questionId);
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
    } finally {
      setAnsweringQuestionId(null);
    }
  };

  const handleGenerateSummary = async () => {
    if (!selectedLecture) return;
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setIsGeneratingSummary(true);
    setSummaryError(null);
    try {
      const updated = await api.lectures.generateSummary(selectedLecture.id, { signal: controller.signal });
      if (updated?.status === 'cancelled') {
        return;
      }
      if (updated?.processed_notes) {
        await onUpdateLecture(selectedLecture.id, {
          processed_notes: updated.processed_notes,
        });
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        return;
      }
      setSummaryError(err?.message || 'Errore durante la generazione della sintesi in locale');
    } finally {
      setIsGeneratingSummary(false);
      abortControllerRef.current = null;
    }
  };

  const handleCancelSummary = async (e?: React.MouseEvent) => {
    e?.preventDefault();
    e?.stopPropagation();
    setIsGeneratingSummary(false);
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

  const selectedLecture = lectures.find((l) => l.id === selectedLectureId) || filteredLectures[0] || null;

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

  // Carica i segnalibri quando cambia la lezione selezionata
  useEffect(() => {
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
    setIsPlaying(false);
    setCurrentTime(0);
    setAudioDuration(selectedLecture?.duration_seconds || 0);
    return () => {
      alive = false;
    };
  }, [selectedLecture?.id]);

  // Carica i timestamp parola per parola all'apertura della lezione (Stile Apple Music)
  useEffect(() => {
    setKaraokeLines(null);
    if (!selectedLecture) {
      setLoadingKaraoke(false);
      return;
    }
    let isMounted = true;
    setLoadingKaraoke(true);
    api.lectures.getTimestamps(selectedLecture.id)
      .then((res: any) => {
        if (isMounted) {
          const parsed = normalizeKaraokeLines(res);
          setKaraokeLines(parsed.length > 0 ? parsed : null);
        }
      })
      .catch((err) => {
        console.warn('[Karaoke] Timestamps non disponibili:', err);
      })
      .finally(() => {
        if (isMounted) setLoadingKaraoke(false);
      });
    return () => { isMounted = false; };
  }, [selectedLecture?.id]);

  const handleSeekAudio = (seconds: number) => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, seconds);
    setCurrentTime(seconds);
    if (!isPlaying) {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch((e) => console.warn('Audio play prevented:', e));
    }
  };

  const handleCopyTranscript = () => {
    if (!selectedLecture?.raw_transcript) return;
    navigator.clipboard.writeText(selectedLecture.raw_transcript);
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

  const handleSkipAudio = (offsetSeconds: number) => {
    if (!selectedLecture?.audio_path || !audioRef.current) return;
    const maxDur = audioDuration || (audioRef.current.duration > 0 ? audioRef.current.duration : 99999);
    const newTime = Math.max(0, Math.min(maxDur, (audioRef.current.currentTime || 0) + offsetSeconds));
    audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  // ── Funzioni di Esportazione Markdown ───────────────────────────────────────
  const downloadMarkdownFile = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    setExportNotice(`Esportato con successo: ${filename}`);
    setTimeout(() => setExportNotice(null), 3500);
  };

  const handleExportSingleLecture = () => {
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
    downloadMarkdownFile(md, cleanFilename);
  };

  const handleExportFullQuaderno = () => {
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
    downloadMarkdownFile(md, cleanFilename);
  };

  const handleExportTranscriptTxt = () => {
    if (!selectedLecture || !selectedLecture.raw_transcript) return;
    const cleanFilename = `Trascrizione_${subject.name}_${selectedLecture.title}`.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.txt';
    const blob = new Blob([selectedLecture.raw_transcript], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = cleanFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    setExportNotice(`Trascrizione salvata: ${cleanFilename}`);
    setTimeout(() => setExportNotice(null), 3500);
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
      {!isSidebarCollapsed && (
        <div className="w-72 flex flex-col border-r border-white/[0.08] liquid-glass-subtle flex-shrink-0 animate-fade-in">
          {/* Top Bar: Search Bar with + and Collapse */}
          <div className="p-3 border-b border-white/[0.06] flex items-center gap-1.5">
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
              onClick={() => setShowFinderModal(true)}
              className="w-7 h-7 rounded-xl flex items-center justify-center liquid-glass hover:bg-white/[0.14] text-white/80 hover:text-white border border-white/[0.12] transition-all cursor-pointer flex-shrink-0"
              title="Importa o aggiungi lezione nel Quaderno"
            >
              <Plus size={13} strokeWidth={2.2} />
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
          </div>

          {/* Lectures List */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredLectures.length === 0 ? (
              <div className="py-12 px-3 text-center">
                <BookOpen size={20} strokeWidth={1.5} className="mx-auto text-white/15 mb-2" />
                <p className="text-xs text-white/40">Nessuna lezione in {subject.name}</p>
                <button
                  onClick={() => setShowFinderModal(true)}
                  className="mt-3 text-[11px] text-white/60 hover:text-white underline underline-offset-4"
                >
                  + Importa da Materiali
                </button>
              </div>
            ) : (
              filteredLectures.map((l) => {
                const isSelected = selectedLecture?.id === l.id;
                const isImported = l.source_type === 'imported';

                return (
                  <div
                    key={l.id}
                    onClick={() => {
                      setSelectedLectureId(l.id);
                      setIsEditing(false);
                    }}
                    className={clsx(
                      'p-3 rounded-2xl cursor-pointer transition-all duration-150 border text-left group',
                      isSelected
                        ? 'bg-white/[0.12] border-white/[0.16] shadow-liquid-sm'
                        : 'bg-transparent border-transparent hover:bg-white/[0.04] hover:border-white/[0.06]'
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h4 className={clsx(
                        'text-xs font-semibold tracking-tight leading-snug line-clamp-1',
                        isSelected ? 'text-white' : 'text-white/80 group-hover:text-white'
                      )}>
                        {l.title}
                      </h4>
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
      )}

      {/* ── Right Column: Clean Reading & Editing Sheet (flex-1) ── */}
      {selectedLecture ? (
        <div className="flex-1 flex flex-col overflow-hidden content-canvas border-l border-white/[0.06] relative">
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
                    className="p-1.5 mt-0.5 rounded-xl liquid-glass-subtle hover:bg-white/[0.10] text-white/60 hover:text-white border border-white/[0.10] transition-all cursor-pointer flex items-center gap-1.5 flex-shrink-0 shadow-liquid-sm"
                  >
                    <PanelLeftOpen size={14} strokeWidth={1.75} />
                    <span className="text-[11px] font-medium hidden sm:inline">Indice</span>
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
                      isGeneratingSummary ? (
                        <button
                          type="button"
                          onClick={handleCancelSummary}
                          className="h-8 px-3.5 rounded-full bg-red-500/20 hover:bg-red-500/30 text-xs font-semibold text-red-300 border border-red-500/40 transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm animate-pulse"
                          title="Interrompi subito l'inferenza IA e libera le risorse del computer"
                        >
                          <Square size={10} fill="currentColor" strokeWidth={0} />
                          <span>Interrompi</span>
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={handleGenerateSummary}
                          className="h-8 px-3.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/90 border border-white/[0.14] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                          title="Genera o rigenera la sintesi strutturata del Quaderno"
                        >
                          <Sparkles size={12} strokeWidth={2} className="text-amber-300" />
                          <span>Sintesi IA</span>
                        </button>
                      )
                    )}

                    <button
                      type="button"
                      onClick={startEditing}
                      className="h-8 px-3.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/90 border border-white/[0.14] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                      title="Modifica appunti o titolo"
                    >
                      <Pencil size={12} strokeWidth={2} />
                      <span>Modifica</span>
                    </button>

                    {/* Bottone Esporta con Morphing in 'X' & Pannello a Bolla Liquid Glass */}
                    <div className="relative" ref={exportMenuRef}>
                      <button
                        type="button"
                        onClick={() => setShowExportMenu(!showExportMenu)}
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
                            initial={{ opacity: 0, scale: 0.15, y: -16, filter: 'blur(10px)' }}
                            animate={{ 
                              opacity: 1, 
                              scale: 1, 
                              y: 0, 
                              filter: 'blur(0px)',
                              transition: { type: 'spring', damping: 22, stiffness: 340, mass: 0.75 } 
                            }}
                            exit={{ 
                              opacity: 0, 
                              scale: 0.25, 
                              y: -12, 
                              filter: 'blur(8px)',
                              transition: { duration: 0.16, ease: 'easeIn' } 
                            }}
                            style={{ transformOrigin: 'top right' }}
                            className="absolute right-0 top-11 w-72 rounded-3xl bg-[#13141c]/95 backdrop-blur-3xl border border-white/[0.20] shadow-[0_24px_60px_rgba(0,0,0,0.85),0_0_0_1px_rgba(255,255,255,0.08)] p-2 z-50 space-y-1.5"
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
                                  <span className="block truncate font-semibold">Solo Trascrizione (.txt)</span>
                                  <span className="block text-[10px] text-white/45">Testo integrale sbobinato</span>
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
                                  <span className="block truncate font-semibold">File Audio Registrato</span>
                                  <span className="block text-[10px] text-white/45">Traccia audio della lezione</span>
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
                                <span className="block text-[10px] text-white/45">{lectures.length} lezioni del corso</span>
                              </div>
                            </button>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <button
                      type="button"
                      onClick={() => setShowDeleteModal(true)}
                      title="Rimuovi lezione"
                      className="w-8 h-8 rounded-full flex items-center justify-center text-white/40 hover:text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-all cursor-pointer"
                    >
                      <Trash2 size={14} strokeWidth={1.75} />
                    </button>
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
                  size="sm"
                  layoutId="quaderno-reading-tab"
                  options={[
                    { id: 'notes' as const, label: 'Quaderno', icon: FileText },
                    { id: 'transcript' as const, label: 'Trascrizione', icon: AlignLeft },
                    {
                      id: 'questions' as const,
                      label: 'Domande',
                      icon: HelpCircle,
                      badge: parsedQuestions.length > 0 ? (
                        <span className="text-[10px] font-mono font-medium bg-amber-400/20 text-amber-300 px-1.5 py-0.5 rounded-full border border-amber-400/30">
                          {parsedQuestions.length}
                        </span>
                      ) : undefined,
                    },
                  ]}
                  value={activeTab}
                  onChange={setActiveTab}
                />
              </div>
            )}
          </div>

          {/* Sheet Body */}
          <div className="flex-1 overflow-y-auto px-12 py-8 pb-32">
            <div className="max-w-3xl mx-auto space-y-6">
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
                        <MarkdownContent content={editNotes} className="quaderno-markdown" />
                      ) : (
                        <p className="text-xs text-white/40 italic">Nessun contenuto da visualizzare nell'anteprima.</p>
                      )}
                    </div>
                  )}
                </div>
              ) : activeTab === 'notes' ? (
                <>
                  {isGeneratingSummary ? (
                    <div className="p-8 rounded-3xl bg-white/[0.03] border border-amber-400/25 text-center space-y-4 animate-fade-in shadow-liquid-md">
                      <div className="w-12 h-12 rounded-2xl bg-amber-400/10 border border-amber-400/30 flex items-center justify-center mx-auto text-amber-300">
                        <Loader2 size={22} className="animate-spin text-amber-400" />
                      </div>
                      <div className="space-y-1">
                        <h3 className="text-sm font-semibold text-white">Generazione Capitolo con Qwen 3.5...</h3>
                        <p className="text-xs text-white/60 max-w-md mx-auto leading-relaxed">
                          L'LLM sta sintetizzando la trascrizione e i concetti chiave della lezione per redigere il capitolo strutturato per il tuo Quaderno.
                        </p>
                      </div>
                      <div className="flex items-center justify-center gap-1.5 py-1">
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse [animation-delay:150ms]" />
                        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse [animation-delay:300ms]" />
                        <span className="text-[11px] font-mono text-amber-300/80 ml-2">Inferenza locale attiva (100% Offline)</span>
                      </div>
                      <div className="pt-2">
                        <button
                          type="button"
                          onClick={handleCancelSummary}
                          className="px-4 py-1.5 rounded-full bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-xs font-semibold text-red-300 transition-all inline-flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                          title="Interrompi subito l'elaborazione IA e libera la RAM"
                        >
                          <Square size={10} fill="currentColor" strokeWidth={0} />
                          <span>Interrompi e libera risorse</span>
                        </button>
                      </div>
                    </div>
                  ) : selectedLecture.processed_notes ? (
                    <div className="p-4 sm:p-6 rounded-3xl bg-white/[0.02] border border-white/[0.06] shadow-liquid-sm">
                      <MarkdownContent content={selectedLecture.processed_notes} className="quaderno-markdown" />
                    </div>
                  ) : (
                    <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/[0.06] text-center space-y-3">
                      <Sparkles size={24} strokeWidth={1.5} className="mx-auto text-amber-300 mb-2" />
                      <h3 className="text-sm font-semibold text-white/90">Nessun appunto nel Quaderno</h3>
                      <p className="text-xs text-white/50 max-w-sm mx-auto leading-relaxed">
                        {selectedLecture.raw_transcript 
                          ? "La trascrizione è pronta! Puoi generare automaticamente il capitolo strutturato per il Quaderno con l'LLM locale."
                          : "Non ci sono note scritte per questa lezione. Puoi scriverle o incollarle cliccando su Modifica."}
                      </p>
                      
                      {summaryError && (
                        <p className="text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-1.5 max-w-sm mx-auto">
                          {summaryError}
                        </p>
                      )}

                      <div className="flex items-center justify-center gap-2 pt-2">
                        {selectedLecture.raw_transcript && (
                          <button
                            type="button"
                            onClick={handleGenerateSummary}
                            className="px-4 py-1.5 rounded-full bg-white/[0.18] hover:bg-white/[0.26] border border-white/[0.22] text-xs font-semibold text-white transition-all inline-flex items-center gap-1.5 cursor-pointer shadow-liquid-sm"
                          >
                            <Sparkles size={13} className="text-amber-300" />
                            <span>Genera Capitolo</span>
                          </button>
                        )}
                        <button
                          onClick={startEditing}
                          className="px-4 py-1.5 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-semibold text-white transition-all inline-flex items-center gap-1.5 cursor-pointer"
                        >
                          <Pencil size={12} strokeWidth={2} />
                          <span>Modifica Manuale</span>
                        </button>
                      </div>
                    </div>
                  )}
                </>
              ) : activeTab === 'transcript' ? (
                <div className="animate-fade-in">
                  <AppleLyricsView
                    lines={karaokeLines}
                    audioRef={audioRef}
                    isPlaying={isPlaying}
                    onSeek={handleSeekAudio}
                    subjectColor={subject.color}
                    rawTranscript={selectedLecture.raw_transcript || ''}
                    onCopyTranscript={handleCopyTranscript}
                    copiedTranscript={copiedTranscript}
                    isLoadingTimestamps={loadingKaraoke}
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                    <div className="flex items-center gap-2">
                      <HelpCircle size={15} className="text-amber-400" />
                      <span className="text-xs font-semibold text-white/90">
                        Domande e Interventi del Docente
                      </span>
                    </div>
                    <span className="text-[11px] font-mono text-white/40">
                      {parsedQuestions.length} {parsedQuestions.length === 1 ? 'domanda registrata' : 'domande registrate'}
                    </span>
                  </div>

                  {parsedQuestions.length === 0 ? (
                    <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/[0.06] text-center">
                      <HelpCircle size={24} strokeWidth={1.5} className="mx-auto text-white/30 mb-3" />
                      <h3 className="text-sm font-semibold text-white/90">Nessuna domanda per questa lezione</h3>
                      <p className="text-xs text-white/50 mt-1 max-w-sm mx-auto">
                        In questa specifica registrazione non sono state rilevate domande dirette o frasi interrogative del docente.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {parsedQuestions.map((q, idx) => {
                        const hasModel = Boolean(settings.modelsDownloaded);
                        const isPending = isPlaceholderAnswer(q.suggestedAnswer);
                        // Fallback sull'indice: il backend riconosce le domande anche per posizione
                        const qKey = q.id || String(idx);
                        const isAnswering = answeringQuestionId === qKey;

                        return (
                          <div
                            key={qKey}
                            className="p-4 rounded-2xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-2.5 animate-slide-up"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <p className="text-xs sm:text-[13px] font-medium text-white/95 leading-relaxed">
                                "{q.question}"
                              </p>
                              <div className="flex items-center gap-1.5 flex-shrink-0">
                                <span className="text-[9px] font-mono text-amber-300/80 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-md">
                                  Prof
                                </span>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteSavedQuestion(qKey)}
                                  title="Elimina domanda non pertinente"
                                  className="p-1 rounded-md text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer"
                                >
                                  <Trash2 size={12} strokeWidth={1.75} />
                                </button>
                              </div>
                            </div>

                            {/* Stato risposta */}
                            {isPending ? (
                              <div className="pt-2 border-t border-white/[0.06]">
                                {!hasModel ? (
                                  <div className="flex items-center justify-between gap-3 bg-amber-500/[0.08] border border-amber-500/20 p-3 rounded-xl">
                                    <div className="flex items-center gap-2 text-xs text-amber-200/90 leading-relaxed">
                                      <AlertTriangle size={14} className="text-amber-400 flex-shrink-0" />
                                      <span>Modello LLM non installato. Vai nelle Impostazioni e scarica un modello LLM per avere le risposte alle domande.</span>
                                    </div>
                                    {onOpenSettings && (
                                      <button
                                        type="button"
                                        onClick={onOpenSettings}
                                        className="px-3 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 text-[11px] font-semibold transition-all flex-shrink-0 cursor-pointer"
                                      >
                                        Impostazioni
                                      </button>
                                    )}
                                  </div>
                                ) : (
                                  <div className="flex items-center justify-between gap-3 bg-white/[0.02] border border-white/[0.04] p-3 rounded-xl">
                                    <span className="text-xs text-white/40 italic">
                                      Nessuna risposta memorizzata per questa domanda.
                                    </span>
                                    <button
                                      type="button"
                                      disabled={isAnswering}
                                      onClick={() => handleAnswerQuestion(qKey)}
                                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white/[0.14] hover:bg-white/[0.22] text-white text-xs font-semibold border border-white/[0.18] shadow-liquid-sm transition-all cursor-pointer disabled:opacity-50"
                                    >
                                      {isAnswering ? (
                                        <>
                                          <Loader2 size={12} className="animate-spin text-amber-300" />
                                          <span>SbobinAI sta rispondendo...</span>
                                        </>
                                      ) : (
                                        <>
                                          <Sparkles size={12} className="text-amber-300" />
                                          <span>Rispondi con l'IA</span>
                                        </>
                                      )}
                                    </button>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="pt-2.5 border-t border-white/[0.06]">
                                <div className="flex items-center justify-between mb-1.5">
                                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-white/40">
                                    <Sparkles size={11} className="text-amber-400/70" />
                                    <span>Risposta SbobinAI (locale)</span>
                                  </div>
                                  {hasModel && (
                                    <button
                                      type="button"
                                      disabled={isAnswering}
                                      onClick={() => handleAnswerQuestion(qKey)}
                                      title="Rigenera la risposta a questa domanda con SbobinAI"
                                      className="text-[10px] text-white/40 hover:text-white/80 underline underline-offset-2 transition-colors cursor-pointer"
                                    >
                                      {isAnswering ? 'Rigenerazione...' : 'Rigenera'}
                                    </button>
                                  )}
                                </div>
                                <div className="text-xs text-white/80 leading-relaxed bg-white/[0.03] p-3 rounded-xl border border-white/[0.04]">
                                  <MarkdownContent content={q.suggestedAnswer!} />
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
                <div className="pointer-events-auto mb-2 flex items-center gap-1.5 overflow-x-auto max-w-xl py-1 px-3.5 rounded-full bg-[#0c0d14]/85 backdrop-blur-2xl border border-white/[0.12] shadow-[0_8px_24px_rgba(0,0,0,0.5)] no-scrollbar animate-fade-in">
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
              <div className="pointer-events-auto w-full max-w-xl h-[52px] px-4 rounded-full bg-white/[0.08] hover:bg-white/[0.11] backdrop-blur-3xl border border-white/[0.20] shadow-[0_20px_50px_rgba(0,0,0,0.65),inset_0_1px_1px_rgba(255,255,255,0.30)] flex items-center gap-3 transition-all">
                {/* HTML Audio element */}
                <audio
                  ref={audioRef}
                  src={api.lectures.getAudioUrl(selectedLecture.id)}
                  preload="metadata"
                  onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
                  onLoadedMetadata={() => setAudioDuration(audioRef.current?.duration || selectedLecture.duration_seconds || 0)}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
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
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          {isSidebarCollapsed && (
            <button
              type="button"
              onClick={() => setIsSidebarCollapsed(false)}
              className="mb-4 px-3 py-1.5 rounded-xl liquid-glass hover:bg-white/[0.12] text-xs font-medium text-white flex items-center gap-2 border border-white/[0.10] shadow-liquid-sm cursor-pointer"
            >
              <PanelLeftOpen size={14} />
              <span>Mostra indice lezioni</span>
            </button>
          )}
          <BookOpen size={32} strokeWidth={1.5} className="text-white/20 mb-3" />
          <h3 className="text-sm font-semibold text-white/80">Nessuna lezione selezionata</h3>
          <p className="text-xs text-white/40 max-w-xs mt-1">Seleziona una lezione dall'indice o importane una dai materiali.</p>
        </div>
      )}

      {/* ── Modal Finder Memoria: Importa da Materiali nel Quaderno ── */}
      <MaterialiFinderModal
        isOpen={showFinderModal}
        onClose={() => setShowFinderModal(false)}
        subject={subject}
        lecturesInQuaderno={lectures}
        onIncludeLectureInQuaderno={async (lectureId) => {
          await onRestoreToQuaderno(lectureId);
        }}
        onImportMaterialAsLecture={async (data) => {
          const created = await onAddLecture(data);
          setSelectedLectureId(created.id);
          return created;
        }}
        onOpenAddMaterialModal={onOpenAddMaterialModal}
      />

      {/* ── Modal: Scelta Eliminazione (Solo Quaderno vs Definitiva) ── */}
      {showDeleteModal && selectedLecture && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/60 backdrop-blur-md animate-fade-in"
          onClick={() => setShowDeleteModal(false)}
        >
          <div 
            className="w-full max-w-md rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg p-6 animate-slide-up space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl liquid-glass-subtle border border-red-500/20 flex items-center justify-center text-red-400">
                  <Trash2 size={15} strokeWidth={1.75} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold tracking-tight text-white">
                    Rimuovi Lezione
                  </h3>
                  <p className="text-[11px] text-white/40 truncate max-w-[240px]">
                    {selectedLecture.title}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowDeleteModal(false)}
                className="w-6 h-6 rounded-full flex items-center justify-center text-white/40 hover:text-white transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            <p className="text-xs text-white/70 leading-relaxed">
              Puoi rimuovere questa lezione solo dal quaderno di studio, oppure cancellarla per sempre dall'archivio del corso.
            </p>

            <div className="space-y-2 pt-1">
              {/* Opzione 1: Rimuovi solo dal quaderno */}
              <button
                onClick={async () => {
                  await onRemoveFromQuadernoOnly(selectedLecture.id);
                  setShowDeleteModal(false);
                }}
                className="w-full p-3.5 rounded-2xl liquid-glass-subtle hover:bg-white/[0.08] border border-white/[0.10] text-left transition-all group flex items-start gap-3"
              >
                <BookOpen size={16} className="text-white/60 mt-0.5 flex-shrink-0 group-hover:text-white transition-colors" />
                <div>
                  <h4 className="text-xs font-semibold text-white/95">
                    Rimuovi solo dal Quaderno
                  </h4>
                  <p className="text-[11px] text-white/40 leading-snug mt-0.5">
                    La registrazione e la trascrizione restano salvate nella memoria (sezione Materiali). Potrai ripristinarla quando vuoi.
                  </p>
                </div>
              </button>

              {/* Opzione 2: Elimina definitivamente */}
              <button
                onClick={async () => {
                  try {
                    await onDeletePermanently(selectedLecture.id);
                    setShowDeleteModal(false);
                  } catch (err) {
                    console.error('Impossibile eliminare la lezione:', err);
                  }
                }}
                className="w-full p-3.5 rounded-2xl bg-red-500/10 hover:bg-red-500/15 border border-red-500/20 text-left transition-all group flex items-start gap-3"
              >
                <Trash2 size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="text-xs font-semibold text-red-300">
                    Elimina definitivamente anche dalla memoria
                  </h4>
                  <p className="text-[11px] text-red-300/60 leading-snug mt-0.5">
                    Cancella definitivamente la lezione, la registrazione e gli appunti da tutto il database.
                  </p>
                </div>
              </button>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="px-4 py-1.5 rounded-full text-xs font-medium text-white/50 hover:text-white transition-colors"
              >
                Annulla
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
