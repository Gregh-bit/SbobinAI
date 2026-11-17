import { useState, useRef, useEffect, useLayoutEffect, useMemo } from 'react';
import {
  Plus,
  ArrowUp,
  Square,
  Sparkles,
  HelpCircle,
  Mic,
  BookOpen,
  FileText,
  Check,
  RotateCcw,
  Brain,
  Paperclip,
  X,
  ChevronLeft,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { ChatScope, ChatStudyMode, CourseMaterial, Lecture } from '../../types';

export interface StudyModeMeta {
  id: ChatStudyMode;
  label: string;
  shortLabel: string;
  description: string;
  icon: typeof Sparkles;
  colorClass: string;
}

export const STUDY_MODE_OPTIONS: StudyModeMeta[] = [
  {
    id: 'quiz',
    label: 'Crea un Quiz',
    shortLabel: 'Quiz',
    description: 'Genera un test a scelta multipla salvato nei Quiz',
    icon: HelpCircle,
    colorClass: 'text-sky-400',
  },
];

export function getStudyModeMeta(mode: string | undefined): StudyModeMeta | undefined {
  if (!mode || mode === 'auto') return undefined;
  return STUDY_MODE_OPTIONS.find((opt) => opt.id === mode);
}

export interface ChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
  subjectName: string;
  studyMode: ChatStudyMode;
  onStudyModeChange: (mode: ChatStudyMode) => void;
  scope: ChatScope;
  onScopeChange: (scope: ChatScope) => void;
  lectures: Lecture[];
  materials: CourseMaterial[];
}

/**
 * ChatGPT 2026 unified floating pill composer with inline active mode chips,
 * multi-action popover (+), voice dictation, Pensa reasoning toggle, and ArrowUp send.
 */
export default function ChatComposer({
  value,
  onChange,
  onSend,
  onStop,
  isStreaming,
  disabled = false,
  subjectName,
  studyMode,
  onStudyModeChange,
  scope,
  onScopeChange,
  lectures,
  materials,
}: ChatComposerProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuView, setMenuView] = useState<'main' | 'scope'>('main');
  const [isListening, setIsListening] = useState(false);
  const [thinkingMode, setThinkingMode] = useState(false);
  const [isMultiline, setIsMultiline] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const preDictationDraftRef = useRef('');
  const valueRef = useRef(value);
  valueRef.current = value;

  // Auto-grow textarea up to ~160px with optical single-line centering (layout effect avoids frame flicker)
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    if (!value.trim()) {
      el.style.height = '24px';
      setIsMultiline(false);
      return;
    }
    el.style.height = 'auto';
    const scrollH = el.scrollHeight;
    if (scrollH <= 34 && !value.includes('\n')) {
      el.style.height = '24px';
      setIsMultiline(false);
    } else {
      el.style.height = `${Math.min(scrollH, 160)}px`;
      setIsMultiline(true);
    }
  }, [value]);

  // Close popovers on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handleOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
        setMenuView('main');
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [menuOpen]);

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.onstart = null;
          recognitionRef.current.onresult = null;
          recognitionRef.current.onerror = null;
          recognitionRef.current.onend = null;
          recognitionRef.current.stop();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  const fileMaterials = useMemo(
    () => materials.filter((m) => m.type !== 'lecture'),
    [materials]
  );

  const safeLectureIds = useMemo(() => scope?.lecture_ids ?? [], [scope?.lecture_ids]);
  const safeMaterialIds = useMemo(() => scope?.material_ids ?? [], [scope?.material_ids]);

  // Filtra solo ID validi che esistono effettivamente nelle lezioni e nei materiali del corso
  const validLectureIds = useMemo(() => {
    if (lectures.length === 0) return safeLectureIds;
    const lectureSet = new Set(lectures.map((l) => l.id));
    return safeLectureIds.filter((id) => lectureSet.has(id));
  }, [lectures, safeLectureIds]);

  const validMaterialIds = useMemo(() => {
    if (materials.length === 0) return safeMaterialIds;
    const materialSet = new Set(materials.map((m) => m.id));
    materials.forEach((m) => materialSet.add(m.name));
    return safeMaterialIds.filter((id) => materialSet.has(id) || id.startsWith('infocorso-') || id.toLowerCase() === 'infocorso.md');
  }, [materials, safeMaterialIds]);

  const scopedCount = validLectureIds.length + validMaterialIds.length;
  const scopeLabel = useMemo(() => {
    if (scopedCount === 0) return 'Tutto il corso';
    const parts: string[] = [];
    if (validLectureIds.length > 0) {
      parts.push(
        `${validLectureIds.length} ${validLectureIds.length === 1 ? 'lezione' : 'lezioni'}`
      );
    }
    if (validMaterialIds.length > 0) {
      parts.push(`${validMaterialIds.length} file`);
    }
    return parts.join(' · ');
  }, [validLectureIds.length, validMaterialIds.length, scopedCount]);

  const toggleLectureScope = (lectureId: number) => {
    const exists = validLectureIds.includes(lectureId);
    const nextLectures = exists
      ? validLectureIds.filter((id) => id !== lectureId)
      : [...validLectureIds, lectureId];
    onScopeChange({ lecture_ids: nextLectures, material_ids: validMaterialIds });
  };

  const toggleMaterialScope = (materialId: string) => {
    const exists = validMaterialIds.includes(materialId);
    const nextMaterials = exists
      ? validMaterialIds.filter((id) => id !== materialId)
      : [...validMaterialIds, materialId];
    onScopeChange({ lecture_ids: validLectureIds, material_ids: nextMaterials });
  };

  const toggleListening = () => {
    if (isListening) {
      try {
        recognitionRef.current?.stop();
      } catch {
        // ignore
      }
      setIsListening(false);
      return;
    }

    const SpeechRecognitionCtor =
      (window as unknown as { SpeechRecognition?: any }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: any }).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) return;

    try {
      preDictationDraftRef.current = valueRef.current;
      const recognition = new SpeechRecognitionCtor();
      recognition.lang = 'it-IT';
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onstart = () => setIsListening(true);
      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        const dictated = transcript.trimStart();
        if (!dictated) return;
        const prefix = preDictationDraftRef.current;
        const needsSpace = prefix.length > 0 && !/\s$/.test(prefix);
        onChange(prefix ? `${prefix}${needsSpace ? ' ' : ''}${dictated}` : dictated);
      };
      recognition.onerror = () => setIsListening(false);
      recognition.onend = () => setIsListening(false);

      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  const speechSupported =
    typeof window !== 'undefined' &&
    Boolean(
      (window as unknown as { SpeechRecognition?: unknown }).SpeechRecognition ||
        (window as unknown as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition
    );

  const activeModeMeta = getStudyModeMeta(studyMode);
  const canSend = Boolean(value.trim() || studyMode !== 'auto');

  return (
    <div className="px-4 sm:px-6 pb-4 pt-2 flex-shrink-0">
      <div ref={containerRef} className="max-w-3xl mx-auto relative">
        {/* ChatGPT Actions Popover (+) */}
        {menuOpen && (
          <div className="absolute bottom-full left-0 mb-3 w-80 sm:w-[440px] max-w-[calc(100vw-2rem)] rounded-2xl bg-zinc-900/98 border border-white/10 shadow-2xl backdrop-blur-xl p-1.5 z-40 animate-fade-in">
            {menuView === 'main' ? (
              <div className="space-y-1">
                {/* 1. Filtra fonti del corso */}
                <button
                  type="button"
                  onClick={() => setMenuView('scope')}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl hover:bg-white/[0.08] transition-colors cursor-pointer text-left group"
                >
                  <div className="flex items-center gap-3.5 min-w-0">
                    <Paperclip
                      size={18}
                      className="text-zinc-400 group-hover:text-zinc-200 transition-colors flex-shrink-0"
                    />
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-sm font-medium text-zinc-100 flex-shrink-0">
                        Filtra fonti del corso
                      </span>
                      {scopedCount > 0 ? (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-normal flex-shrink-0">
                          {scopedCount} {scopedCount === 1 ? 'fonte' : 'fonti'}
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-400 truncate">
                          Scegli lezioni o slide
                        </span>
                      )}
                    </div>
                  </div>
                  <ChevronLeft
                    size={15}
                    className="rotate-180 text-zinc-500 group-hover:text-zinc-300 transition-colors flex-shrink-0"
                  />
                </button>

                {/* 2. Crea un Quiz */}
                <button
                  type="button"
                  onClick={() => {
                    const isSelected = studyMode === 'quiz';
                    onStudyModeChange(isSelected ? 'auto' : 'quiz');
                    setMenuOpen(false);
                    textareaRef.current?.focus();
                  }}
                  className={clsx(
                    'w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl transition-colors cursor-pointer text-left group',
                    studyMode === 'quiz'
                      ? 'bg-white/[0.12] text-white'
                      : 'hover:bg-white/[0.08] text-zinc-200'
                  )}
                >
                  <div className="flex items-center gap-3.5 min-w-0">
                    <HelpCircle size={18} className="text-sky-400 flex-shrink-0" />
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-sm font-medium text-zinc-100 flex-shrink-0">
                        Crea un Quiz
                      </span>
                      <span className="text-xs text-zinc-400 truncate">
                        Genera un test a scelta multipla nel corso
                      </span>
                    </div>
                  </div>
                  {studyMode === 'quiz' && <Check size={15} className="text-emerald-400 flex-shrink-0" />}
                </button>
              </div>
            ) : (
              /* Scope View */
              <div className="p-1 space-y-2">
                <div className="flex items-center justify-between px-2 py-1 border-b border-white/[0.07]">
                  <button
                    type="button"
                    onClick={() => setMenuView('main')}
                    className="inline-flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-100 transition-colors cursor-pointer"
                  >
                    <ChevronLeft size={14} />
                    <span>Tutte le opzioni</span>
                  </button>
                  {scopedCount > 0 && (
                    <button
                      type="button"
                      onClick={() => onScopeChange({ lecture_ids: [], material_ids: [] })}
                      className="inline-flex items-center gap-1 text-[11px] text-emerald-300 hover:text-emerald-200 cursor-pointer"
                    >
                      <RotateCcw size={10} />
                      <span>Tutto il corso</span>
                    </button>
                  )}
                </div>

                <div className="max-h-60 overflow-y-auto py-1 space-y-1.5 select-none">
                  <button
                    type="button"
                    onClick={() => {
                      onScopeChange({ lecture_ids: [], material_ids: [] });
                      setMenuOpen(false);
                    }}
                    className={clsx(
                      'w-full flex items-center justify-between px-2.5 py-1.5 rounded-xl text-xs transition-colors cursor-pointer',
                      scopedCount === 0
                        ? 'bg-emerald-500/15 text-emerald-200 border border-emerald-500/25'
                        : 'text-zinc-300 hover:bg-white/[0.05] border border-transparent'
                    )}
                  >
                    <span>Tutto il corso ({subjectName})</span>
                    {scopedCount === 0 && <Check size={12} className="text-emerald-400" />}
                  </button>

                  {lectures.length > 0 && (
                    <div className="space-y-0.5">
                      <div className="px-2.5 pt-1 text-[10px] font-medium text-zinc-500 flex items-center gap-1.5">
                        <BookOpen size={11} />
                        <span>Lezioni ({lectures.length})</span>
                      </div>
                      {lectures.map((lec) => {
                        const checked = scope.lecture_ids.includes(lec.id);
                        return (
                          <button
                            key={lec.id}
                            type="button"
                            onClick={() => toggleLectureScope(lec.id)}
                            className={clsx(
                              'w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-xl text-left text-xs transition-colors cursor-pointer',
                              checked
                                ? 'bg-white/[0.09] text-zinc-100'
                                : 'text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200'
                            )}
                          >
                            <span className="truncate">{lec.title || `Lezione #${lec.id}`}</span>
                            <span
                              className={clsx(
                                'w-4 h-4 rounded-md border flex items-center justify-center flex-shrink-0',
                                checked
                                  ? 'bg-emerald-500/25 border-emerald-400/60 text-emerald-300'
                                  : 'border-white/15 bg-transparent'
                              )}
                            >
                              {checked && <Check size={10} />}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {fileMaterials.length > 0 && (
                    <div className="space-y-0.5">
                      <div className="px-2.5 pt-1 text-[10px] font-medium text-zinc-500 flex items-center gap-1.5">
                        <FileText size={11} />
                        <span>Slide e Materiali ({fileMaterials.length})</span>
                      </div>
                      {fileMaterials.map((mat) => {
                        const checked = scope.material_ids.includes(mat.id);
                        return (
                          <button
                            key={mat.id}
                            type="button"
                            onClick={() => toggleMaterialScope(mat.id)}
                            className={clsx(
                              'w-full flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-xl text-left text-xs transition-colors cursor-pointer',
                              checked
                                ? 'bg-white/[0.09] text-zinc-100'
                                : 'text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200'
                            )}
                          >
                            <span className="truncate">{mat.name}</span>
                            <span
                              className={clsx(
                                'w-4 h-4 rounded-md border flex items-center justify-center flex-shrink-0',
                                checked
                                  ? 'bg-emerald-500/25 border-emerald-400/60 text-emerald-300'
                                  : 'border-white/15 bg-transparent'
                              )}
                            >
                              {checked && <Check size={10} />}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {lectures.length === 0 && fileMaterials.length === 0 && (
                    <p className="px-2.5 py-2 text-[11px] text-zinc-500">
                      Nessuna lezione o slide caricata per questo corso.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ChatGPT 2026 Unified Floating Pill Container */}
        <div
          className={clsx(
            'rounded-[28px] bg-zinc-900/95 border border-white/10 shadow-2xl backdrop-blur-md px-3 py-2 transition-all focus-within:border-white/20 flex gap-2 min-h-[50px]',
            isMultiline ? 'items-end' : 'items-center'
          )}
        >
          {/* + Actions Button */}
          <button
            type="button"
            onClick={() => {
              setMenuView('main');
              setMenuOpen((prev) => !prev);
            }}
            title="Aggiungi file o scegli modalità"
            className={clsx(
              'w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer flex-shrink-0',
              isMultiline && 'mb-0.5',
              menuOpen
                ? 'bg-white/20 text-white'
                : 'bg-white/[0.06] hover:bg-white/[0.12] text-zinc-300 hover:text-white'
            )}
          >
            <Plus size={16} />
          </button>

          {/* Center Input Area with Active Chips and Textarea */}
          <div className="flex-1 min-w-0 flex flex-wrap items-center gap-1.5 py-0.5">
            {/* Active Study Mode Chip */}
            {activeModeMeta && (
              <span className="h-[26px] inline-flex items-center gap-1.5 px-2.5 rounded-full bg-white/[0.08] border border-white/12 text-[12px] font-medium text-zinc-100 select-none flex-shrink-0 animate-fade-in shadow-sm">
                <activeModeMeta.icon size={12} className={clsx(activeModeMeta.colorClass, 'flex-shrink-0')} />
                <span className="leading-none pt-[0.5px]">{activeModeMeta.label}</span>
                <button
                  type="button"
                  onClick={() => {
                    onStudyModeChange('auto');
                    textareaRef.current?.focus();
                  }}
                  className="w-3.5 h-3.5 rounded-full hover:bg-white/20 flex items-center justify-center text-zinc-400 hover:text-white transition-colors cursor-pointer -mr-0.5"
                  title="Rimuovi modalità"
                >
                  <X size={10} />
                </button>
              </span>
            )}

            {/* Active Scoped Sources Chip */}
            {scopedCount > 0 && (
              <span className="h-[26px] inline-flex items-center gap-1.5 px-2.5 rounded-full bg-emerald-500/15 border border-emerald-400/30 text-emerald-200 text-[12px] font-medium select-none flex-shrink-0 animate-fade-in shadow-sm">
                <Paperclip size={11} className="text-emerald-300 flex-shrink-0" />
                <span className="leading-none pt-[0.5px] max-w-[130px] truncate">{scopeLabel}</span>
                <button
                  type="button"
                  onClick={() => {
                    onScopeChange({ lecture_ids: [], material_ids: [] });
                    textareaRef.current?.focus();
                  }}
                  className="w-3.5 h-3.5 rounded-full hover:bg-white/20 flex items-center justify-center text-emerald-300 hover:text-white transition-colors cursor-pointer -mr-0.5"
                  title="Ripristina tutto il corso"
                >
                  <X size={10} />
                </button>
              </span>
            )}

            {/* Prompt Textarea */}
            <textarea
              ref={textareaRef}
              rows={1}
              value={value}
              disabled={disabled}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Backspace' && !value) {
                  if (studyMode !== 'auto') {
                    onStudyModeChange('auto');
                    return;
                  }
                  if (scopedCount > 0) {
                    onScopeChange({ lecture_ids: [], material_ids: [] });
                    return;
                  }
                }
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  if (!isStreaming && canSend) {
                    onSend();
                  }
                }
              }}
              placeholder={
                activeModeMeta
                  ? `Specifica l'argomento (oppure premi Invio)...`
                  : `Chiedi qualsiasi cosa...`
              }
              className="flex-1 min-w-[120px] resize-none bg-transparent px-1 py-[1px] text-[13px] leading-[22px] text-zinc-100 placeholder-zinc-500 focus:outline-none max-h-40 select-text overflow-y-auto block"
            />
          </div>

          {/* Right Controls: Pensa, Dictation, Send / Stop */}
          <div
            className={clsx(
              'flex items-center gap-1 flex-shrink-0',
              isMultiline && 'mb-0.5'
            )}
          >
            {/* Pensa Button (ChatGPT Style Reasoning Toggle) */}
            <button
              type="button"
              onClick={() => setThinkingMode((prev) => !prev)}
              title={
                thinkingMode
                  ? 'Ragionamento approfondito attivo'
                  : 'Attiva ragionamento approfondito'
              }
              className={clsx(
                'h-8 px-2.5 rounded-full flex items-center gap-1.5 border text-xs font-medium transition-all cursor-pointer select-none',
                thinkingMode
                  ? 'bg-white/10 border-white/25 text-white shadow-sm'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.06]'
              )}
            >
              <Brain
                size={14}
                className={thinkingMode ? 'text-indigo-300' : 'text-zinc-400'}
              />
              <span className="hidden sm:inline">Pensa</span>
            </button>

            {/* Voice Dictation */}
            {speechSupported && (
              <button
                type="button"
                onClick={toggleListening}
                title={isListening ? 'Interrompi dettatura vocale' : 'Dettatura vocale'}
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer flex-shrink-0',
                  isListening
                    ? 'bg-rose-500/25 text-rose-300 border border-rose-500/40 animate-pulse'
                    : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.08]'
                )}
              >
                <Mic size={15} strokeWidth={isListening ? 2.4 : 1.8} />
              </button>
            )}

            {/* Send / Stop Circular Button (ArrowUp) */}
            {isStreaming ? (
              <button
                type="button"
                onClick={onStop}
                title="Interrompi generazione"
                className="w-8 h-8 rounded-full flex items-center justify-center bg-white text-zinc-950 hover:bg-zinc-200 transition-all cursor-pointer flex-shrink-0 shadow-md"
              >
                <Square size={10} fill="currentColor" strokeWidth={0} />
              </button>
            ) : (
              <button
                type="button"
                onClick={onSend}
                disabled={disabled || !canSend}
                title="Invia messaggio"
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center transition-all flex-shrink-0',
                  canSend
                    ? 'bg-white text-zinc-950 hover:bg-zinc-200 cursor-pointer shadow-md'
                    : 'bg-white/[0.08] text-white/30 cursor-not-allowed'
                )}
              >
                <ArrowUp size={16} strokeWidth={2.5} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
