// components/AppleLyricsView.tsx — Unified Continuous Transcript with Apple Music Synchronized Glow
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import clsx from 'clsx';
import { ArrowDownCircle, Check, Copy, Sparkles, Volume2 } from 'lucide-react';
import type { KaraokeLine } from '../types';

interface AppleLyricsViewProps {
  lines: KaraokeLine[] | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  isPlaying: boolean;
  onSeek: (seconds: number) => void;
  subjectColor?: string;
  lectureTitle?: string;
  rawTranscript?: string;
  onCopyTranscript?: (textToCopy?: string) => void;
  copiedTranscript?: boolean;
  isLoadingTimestamps?: boolean;
}

// Palette armonica per il mesh gradient basata sul colore del corso
function extractPalette(baseHex?: string) {
  const fallback = '#6366f1';
  const hex = (baseHex && baseHex.startsWith('#') && baseHex.length >= 7) ? baseHex : fallback;

  const r = parseInt(hex.slice(1, 3), 16) || 99;
  const g = parseInt(hex.slice(3, 5), 16) || 102;
  const b = parseInt(hex.slice(5, 7), 16) || 241;

  return {
    c1: `rgba(${r}, ${g}, ${b}, 0.50)`,
    c2: `rgba(${Math.min(255, b + 25)}, ${Math.max(40, r - 25)}, ${Math.min(255, g + 35)}, 0.40)`,
    c3: `rgba(${Math.max(30, g - 20)}, ${Math.min(255, b + 30)}, ${Math.max(40, r + 40)}, 0.35)`,
    c4: `rgba(${Math.min(255, r + 40)}, ${Math.max(20, g - 40)}, ${Math.min(255, b + 60)}, 0.30)`,
  };
}

function formatTimeMs(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

interface ParagraphLine extends KaraokeLine {
  lineIdx: number;
}

interface FlatWord {
  id: string;
  lineIdx: number;
  wordIdx: number;
  text: string;
  start: number;
  end: number;
}

export const AppleLyricsView: React.FC<AppleLyricsViewProps> = ({
  lines,
  audioRef,
  isPlaying,
  onSeek,
  subjectColor,
  rawTranscript,
  onCopyTranscript,
  copiedTranscript,
  isLoadingTimestamps,
}) => {
  const [currentMs, setCurrentMs] = useState<number>(0);
  const [isAutoScrollLocked, setIsAutoScrollLocked] = useState<boolean>(true);
  const [activeWordId, setActiveWordId] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const textWrapperRef = useRef<HTMLDivElement | null>(null);
  const caretRef = useRef<HTMLDivElement | null>(null);
  const wordElementsRef = useRef<Map<string, HTMLSpanElement>>(new Map());
  const lastYRef = useRef<number | null>(null);
  const activeWordIdRef = useRef<string | null>(null);
  const isAutoScrollLockedRef = useRef<boolean>(true);
  const isProgrammaticScrollRef = useRef<boolean>(false);
  const userScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    isAutoScrollLockedRef.current = isAutoScrollLocked;
  }, [isAutoScrollLocked]);

  // Palette dinamica 3-4 colori mesh
  const palette = useMemo(() => extractPalette(subjectColor), [subjectColor]);

  // Fallback anti-allucinazione Whisper sui timestamp
  const isHallucinatedTimestamps = useMemo(() => {
    if (!lines || lines.length === 0) return false;

    // Normalizziamo le righe dei timestamp
    const lineTexts = lines
      .map((l) =>
        l.words
          .map((w) => w.text)
          .join(' ')
          .trim()
          .toLowerCase()
          .replace(/[.,!?;:«»"'()\-]/g, '')
      )
      .filter((t) => t.length > 0);

    if (lineTexts.length === 0) return false;

    const allLinesWords = lines
      .flatMap((l) =>
        l.words.map((w) =>
          w.text.toLowerCase().replace(/[.,!?;:«»"'()\-]/g, '').trim()
        )
      )
      .filter(Boolean);

    const fullLinesString = lineTexts.join(' ');
    const rawLower = (rawTranscript || '').toLowerCase();
    const rawWords = rawLower
      .replace(/[.,!?;:«»"'()\-]/g, ' ')
      .split(/\s+/)
      .filter(Boolean);

    // 1. Formule tipiche di allucinazione / chiusura Whisper
    // ("grazie a tutti", "grazie per la visione", "sottotitoli", "iscriviti al canale", ecc.)
    // Se presenti in lines ma NON presenti in rawTranscript, imposta tassativamente true!
    const hallucinationKeywords = [
      'grazie a tutti',
      'grazie per la visione',
      'grazie della visione',
      'grazie per aver guardato',
      'grazie dell\'attenzione',
      'sottotitoli a cura di',
      'sottotitoli di',
      'sottotitoli',
      'amara.org',
      'iscriviti al canale',
      'iscriviti',
      'continua...',
      'alla prossima',
      'buona visione',
      'lascia un like',
    ];

    for (const kw of hallucinationKeywords) {
      if (fullLinesString.includes(kw) && !rawLower.includes(kw)) {
        return true;
      }
    }

    // 2. Loop di ripetizione di Whisper
    // a) Frequenza della riga più ripetuta
    const freqMap: Record<string, number> = {};
    for (const text of lineTexts) {
      freqMap[text] = (freqMap[text] || 0) + 1;
    }
    const maxFreq = Math.max(...Object.values(freqMap));
    const uniqueCount = Object.keys(freqMap).length;

    // Se più del 40% delle righe identiche con almeno 3 righe
    if (lineTexts.length >= 3 && (maxFreq / lineTexts.length) >= 0.4) {
      return true;
    }

    // Se rawTranscript è consistente (> 15 parole) ma lines contiene solo 1-2 frasi ripetute
    if (rawWords.length >= 15 && uniqueCount <= 2 && lineTexts.length >= 2) {
      return true;
    }

    // Loop consecutivo: la stessa riga ripetuta 3 o più volte di seguito
    let consecutiveCount = 1;
    for (let i = 1; i < lineTexts.length; i++) {
      if (lineTexts[i] === lineTexts[i - 1] && lineTexts[i].length > 3) {
        consecutiveCount++;
        if (consecutiveCount >= 3) {
          return true;
        }
      } else {
        consecutiveCount = 1;
      }
    }

    // 3. Testo non corrispondente a rawTranscript
    if (rawWords.length >= 10 && allLinesWords.length > 0) {
      const rawWordSet = new Set(rawWords);
      const matchingWords = allLinesWords.filter((w) => rawWordSet.has(w));
      const matchRatio = matchingWords.length / allLinesWords.length;

      // Se meno del 35% delle parole dei timestamp compare in rawTranscript
      if (matchRatio < 0.35) {
        return true;
      }

      // Se rawTranscript è consistente (>= 100 parole) ma lines ha pochissime parole (< 15)
      if (rawWords.length >= 100 && allLinesWords.length < 15) {
        return true;
      }
    }

    return false;
  }, [lines, rawTranscript]);

  // Se i timestamp sono corrotti o allucinati, scartali e usa rawTranscript
  const effectiveLines = isHallucinatedTimestamps ? null : lines;

  // Conteggio totale parole
  const totalWords = useMemo(() => {
    if (effectiveLines && effectiveLines.length > 0) {
      return effectiveLines.reduce((acc, l) => acc + l.words.length, 0);
    }
    if (rawTranscript) {
      return rawTranscript.split(/\s+/).filter(Boolean).length;
    }
    return 0;
  }, [effectiveLines, rawTranscript]);

  // Raggruppamento naturale in paragrafi per lettura fluida
  const paragraphs = useMemo(() => {
    if (!effectiveLines || effectiveLines.length === 0) return [];
    const result: ParagraphLine[][] = [];
    let currentPara: ParagraphLine[] = [];

    effectiveLines.forEach((line, idx) => {
      currentPara.push({ ...line, lineIdx: idx });

      const lastWord = line.words[line.words.length - 1];
      const lastText = lastWord ? lastWord.text : '';
      const hasPunctuation = /[.!?]$/.test(lastText);
      const nextLine = effectiveLines[idx + 1];
      const pauseBeforeNext = nextLine ? (nextLine.lineStart - line.lineEnd) : 0;

      // Paragrafo naturale: pausa di respiro o punteggiatura conclusiva
      if ((hasPunctuation && pauseBeforeNext > 1200) || currentPara.length >= 4) {
        result.push(currentPara);
        currentPara = [];
      }
    });

    if (currentPara.length > 0) {
      result.push(currentPara);
    }

    return result;
  }, [effectiveLines]);

  // ── Estrazione del Testo Visualizzato: cosa vedi a schermo = cosa copi! ──
  // Se sono renderizzati i segmenti sincronizzati effectiveLines, estrai il testo esatto dai paragrafi a schermo;
  // Se è renderizzato rawTranscript, estrai rawTranscript.
  const getDisplayedText = useCallback((): string => {
    if (effectiveLines && effectiveLines.length > 0 && paragraphs.length > 0) {
      return paragraphs
        .map((para) =>
          para
            .map((line) => line.words.map((w) => w.text).join(' '))
            .join(' ')
            .replace(/\s+/g, ' ')
            .trim()
        )
        .filter((p) => p.length > 0)
        .join('\n\n');
    }
    return rawTranscript ? rawTranscript.trim() : '';
  }, [effectiveLines, paragraphs, rawTranscript]);

  const [internalCopied, setInternalCopied] = useState<boolean>(false);
  const isCopied = copiedTranscript ?? internalCopied;

  const handleCopy = useCallback(async () => {
    const textToCopy = getDisplayedText();
    if (!textToCopy) return;

    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(textToCopy);
      }
      setInternalCopied(true);
      setTimeout(() => setInternalCopied(false), 2000);
      onCopyTranscript?.(textToCopy);
    } catch (err) {
      console.warn('Errore copia negli appunti:', err);
      onCopyTranscript?.(textToCopy);
    }
  }, [getDisplayedText, onCopyTranscript]);

  // ── Animation Loop ottimizzato (~60 FPS in riproduzione, 0% CPU in pausa) ──
  // Lista unificata di parole per lookup e interpolazione geometrica
  const flatWords = useMemo<FlatWord[]>(() => {
    if (!paragraphs || paragraphs.length === 0) return [];
    const list: FlatWord[] = [];
    paragraphs.forEach((para) => {
      para.forEach((line) => {
        line.words.forEach((w, wIdx) => {
          list.push({
            id: `${line.lineIdx}-${wIdx}`,
            lineIdx: line.lineIdx,
            wordIdx: wIdx,
            text: w.text,
            start: w.start,
            end: w.end,
          });
        });
      });
    });
    return list;
  }, [paragraphs]);

  // Ricerca binaria della parola attiva dato il timestamp audio
  const findActiveWord = useCallback(
    (ms: number): FlatWord | null => {
      if (flatWords.length === 0) return null;
      if (ms < flatWords[0].start) return flatWords[0];
      if (ms >= flatWords[flatWords.length - 1].end) return flatWords[flatWords.length - 1];

      let low = 0;
      let high = flatWords.length - 1;

      while (low <= high) {
        const mid = (low + high) >> 1;
        const w = flatWords[mid];
        if (ms >= w.start && ms <= w.end) {
          return w;
        }
        if (ms < w.start) {
          high = mid - 1;
        } else {
          low = mid + 1;
        }
      }

      // Se ms cade in una pausa tra parole, restituisce la parola precedente
      // così il caret attende al termine di quella parola fino all'attacco della successiva
      if (high >= 0 && high < flatWords.length) {
        return flatWords[high];
      }
      return flatWords[0];
    },
    [flatWords]
  );

  // ── Calcolo Coordinate Cursore & Interpolazione X/Y ─────────────────────────
  const updateCaretPosition = useCallback(
    (ms: number, forceSnap = false) => {
      const caretEl = caretRef.current;
      const wrapperEl = textWrapperRef.current;
      if (!caretEl || !wrapperEl || flatWords.length === 0) {
        if (caretEl) caretEl.style.opacity = '0';
        return;
      }

      const word = findActiveWord(ms);
      if (!word) {
        caretEl.style.opacity = '0';
        return;
      }

      if (word.id !== activeWordIdRef.current) {
        activeWordIdRef.current = word.id;
        setActiveWordId(word.id);
      }

      const wordEl = wordElementsRef.current.get(word.id);
      if (!wordEl) return;

      const wordRect = wordEl.getBoundingClientRect();
      const wrapRect = wrapperEl.getBoundingClientRect();

      // 1. Interpolazione X lungo la parola attiva
      const startX = wordRect.left - wrapRect.left;
      const wordWidth = wordRect.width;
      const duration = Math.max(1, word.end - word.start);
      const ratio = (ms - word.start) / duration;
      const progress = Math.min(1, Math.max(0, ratio));
      const targetX = startX + progress * wordWidth;

      // 2. Coordinata Y e altezza del cursore
      const targetY = wordRect.top - wrapRect.top;
      const targetHeight = Math.max(20, wordRect.height);

      // 3. Gestione cambio riga (line wrap)
      const lastY = lastYRef.current;
      const isLineChange = forceSnap || (lastY !== null && Math.abs(targetY - lastY) > 6);

      if (isLineChange) {
        // Disabilita temporaneamente la transizione per non far volare il cursore in diagonale
        caretEl.style.transition = 'none';
        caretEl.style.transform = `translate3d(${targetX}px, ${targetY}px, 0)`;
        caretEl.style.height = `${targetHeight}px`;
        // Forza reflow immediato per applicare lo snap
        void caretEl.offsetWidth;
        // Ripristina transizione fluida per l'avanzamento sulla nuova riga
        caretEl.style.transition = 'transform 120ms linear, opacity 150ms ease';
      } else {
        caretEl.style.transform = `translate3d(${targetX}px, ${targetY}px, 0)`;
        caretEl.style.height = `${targetHeight}px`;
      }

      lastYRef.current = targetY;
      caretEl.style.opacity = '1';

      // 4. Auto-scroll fluido al cambio riga per mantenere la lettura centrata
      if (isAutoScrollLockedRef.current && containerRef.current && isLineChange) {
        const container = containerRef.current;
        const targetScrollTop = targetY - (container.clientHeight * 0.38);
        if (Math.abs(container.scrollTop - targetScrollTop) > 30) {
          isProgrammaticScrollRef.current = true;
          container.scrollTo({
            top: Math.max(0, targetScrollTop),
            behavior: 'smooth',
          });
          setTimeout(() => {
            isProgrammaticScrollRef.current = false;
          }, 450);
        }
      }
    },
    [flatWords, findActiveWord]
  );

  const handleWordClick = useCallback(
    (word: FlatWord) => {
      onSeek(word.start / 1000);
      setCurrentMs(word.start);
      lastYRef.current = null; // force snap on seek
      requestAnimationFrame(() => {
        updateCaretPosition(word.start, true);
      });
    },
    [onSeek, updateCaretPosition]
  );

  useEffect(() => {
    if (!isPlaying) return;
    let animId: number;

    const tick = () => {
      const audio = audioRef.current;
      if (audio && !audio.paused) {
        const ms = Math.round(audio.currentTime * 1000);
        setCurrentMs(ms);
        updateCaretPosition(ms, false);
        animId = requestAnimationFrame(tick);
      }
    };

    animId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animId);
    };
  }, [audioRef, isPlaying, updateCaretPosition]);

  // Ascolto eventi audio nativi per scrubbing istantaneo
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const syncTime = () => {
      const ms = Math.round(audio.currentTime * 1000);
      setCurrentMs(ms);
      updateCaretPosition(ms, false);
    };

    const handleSeeked = () => {
      const ms = Math.round(audio.currentTime * 1000);
      setCurrentMs(ms);
      lastYRef.current = null;
      updateCaretPosition(ms, true);
    };

    audio.addEventListener('timeupdate', syncTime);
    audio.addEventListener('seeked', handleSeeked);
    audio.addEventListener('play', syncTime);
    audio.addEventListener('playing', syncTime);
    audio.addEventListener('pause', syncTime);

    return () => {
      audio.removeEventListener('timeupdate', syncTime);
      audio.removeEventListener('seeked', handleSeeked);
      audio.removeEventListener('play', syncTime);
      audio.removeEventListener('playing', syncTime);
      audio.removeEventListener('pause', syncTime);
    };
  }, [audioRef, updateCaretPosition]);

  // Sync iniziale e al ridimensionamento della finestra
  useEffect(() => {
    if (flatWords.length > 0) {
      const timer = setTimeout(() => {
        updateCaretPosition(currentMs, true);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [flatWords, updateCaretPosition]);

  useEffect(() => {
    const handleResize = () => {
      updateCaretPosition(currentMs, true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [currentMs, updateCaretPosition]);

  // ── Gestione Interruzione Manuale Utente (Scroll/Wheel) ─────────────────────
  const handleUserInteraction = () => {
    if (isProgrammaticScrollRef.current) return;

    setIsAutoScrollLocked(false);
    isAutoScrollLockedRef.current = false;
    if (userScrollTimeoutRef.current) {
      clearTimeout(userScrollTimeoutRef.current);
    }

    // Riaggancia l'auto-scroll dopo 4.5 secondi di inattività
    userScrollTimeoutRef.current = setTimeout(() => {
      setIsAutoScrollLocked(true);
      isAutoScrollLockedRef.current = true;
      updateCaretPosition(currentMs, true);
    }, 4500);
  };

  useEffect(() => {
    return () => {
      if (userScrollTimeoutRef.current) {
        clearTimeout(userScrollTimeoutRef.current);
      }
    };
  }, []);

  const centerOnActiveCaret = useCallback(() => {
    setIsAutoScrollLocked(true);
    isAutoScrollLockedRef.current = true;
    updateCaretPosition(currentMs, true);
    if (containerRef.current) {
      const caretTop = lastYRef.current ?? 0;
      const container = containerRef.current;
      const targetScrollTop = caretTop - (container.clientHeight * 0.38);
      container.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: 'smooth',
      });
    }
  }, [currentMs, updateCaretPosition]);

  const hasSyncData = Boolean(effectiveLines && effectiveLines.length > 0);

  return (
    <div className="relative w-full rounded-3xl overflow-hidden bg-[#07080d] border border-white/[0.08] shadow-[0_20px_50px_rgba(0,0,0,0.6)]">
      {/* ── 1. Dynamic Mesh Gradient / Liquid Blur Canvas ── */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0" aria-hidden="true">
        <div
          className="absolute -top-[10%] -left-[10%] w-[480px] h-[480px] rounded-full mix-blend-screen filter blur-[85px] opacity-65 animate-mesh-drift-1"
          style={{ background: `radial-gradient(circle at center, ${palette.c1} 0%, transparent 70%)` }}
        />
        <div
          className="absolute -bottom-[15%] -right-[10%] w-[520px] h-[520px] rounded-full mix-blend-screen filter blur-[90px] opacity-60 animate-mesh-drift-2"
          style={{ background: `radial-gradient(circle at center, ${palette.c2} 0%, transparent 70%)` }}
        />
        <div
          className="absolute top-[35%] right-[20%] w-[420px] h-[420px] rounded-full mix-blend-screen filter blur-[85px] opacity-55 animate-mesh-drift-3"
          style={{ background: `radial-gradient(circle at center, ${palette.c3} 0%, transparent 70%)` }}
        />
        <div
          className="absolute bottom-[25%] left-[20%] w-[400px] h-[400px] rounded-full mix-blend-screen filter blur-[80px] opacity-45 animate-mesh-drift-1"
          style={{ background: `radial-gradient(circle at center, ${palette.c4} 0%, transparent 70%)` }}
        />
        {/* Heavy Glass Overlay */}
        <div className="absolute inset-0 backdrop-blur-[90px] bg-gradient-to-b from-[#07080d]/50 via-[#07080d]/70 to-[#07080d]/85" />
      </div>

      {/* ── 2. Minimalist Apple-like Top Bar ── */}
      <div className="relative z-20 flex items-center justify-between px-6 sm:px-8 py-3.5 border-b border-white/[0.06] bg-white/[0.02]">
        <div className="flex items-center gap-2.5">
          <Volume2 size={15} className="text-amber-300 opacity-90" />
          <span className="text-xs font-semibold text-white/85 tracking-wide">
            Trascrizione
          </span>
          {totalWords > 0 && (
            <>
              <span className="text-white/20">•</span>
              <span className="text-[11px] font-mono tabular-nums text-white/40">
                {totalWords} parole
              </span>
            </>
          )}
          {isLoadingTimestamps && (
            <div className="w-3 h-3 rounded-full border border-white/20 border-t-white/80 animate-spin ml-1.5 opacity-80" />
          )}
        </div>

        {/* Pulsante Copia Trascrizione (Cosa vedi a schermo = cosa copi!) */}
        {(onCopyTranscript || rawTranscript || (effectiveLines && effectiveLines.length > 0)) && (
          <button
            type="button"
            onClick={handleCopy}
            className="h-7 px-3 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/85 border border-white/[0.14] transition-all flex items-center justify-center gap-1.5 cursor-pointer shadow-liquid-sm active:scale-95 leading-normal"
            title="Copia l'intera trascrizione visibile a schermo negli appunti"
          >
            {isCopied ? (
              <>
                <Check size={12} className="text-emerald-400" />
                <span className="text-emerald-300 font-semibold text-[11px] leading-normal">Copiato!</span>
              </>
            ) : (
              <>
                <Copy size={12} className="text-white/60" />
                <span className="text-[11px] leading-normal">Copia</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* ── 3. Floating Re-Sync Button (Se l'utente ha fatto scroll manuale) ── */}
      {!isAutoScrollLocked && hasSyncData && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 animate-fade-in pointer-events-auto">
          <button
            type="button"
            onClick={centerOnActiveCaret}
            className="px-4 py-2 rounded-full liquid-glass text-xs font-semibold text-white/95 hover:text-white hover:bg-white/[0.16] border border-white/[0.22] shadow-[0_8px_30px_rgba(0,0,0,0.5)] backdrop-blur-2xl flex items-center gap-2 cursor-pointer transition-all active:scale-95 group"
          >
            <ArrowDownCircle size={15} className="text-amber-300 group-hover:translate-y-0.5 transition-transform" />
            <span>Torna all'audio in riproduzione</span>
          </button>
        </div>
      )}

      {/* ── 4. Continuous Flowing Transcript Viewport ── */}
      <div
        ref={containerRef}
        onWheel={handleUserInteraction}
        onTouchMove={handleUserInteraction}
        onScroll={handleUserInteraction}
        className="relative z-10 max-h-[calc(100vh-280px)] min-h-[380px] overflow-y-auto px-6 sm:px-12 py-10 space-y-7 scroll-smooth focus:outline-none select-text"
        tabIndex={0}
      >
        {isLoadingTimestamps && !hasSyncData ? (
          <div className="py-24 px-8 max-w-md mx-auto flex flex-col items-center justify-center space-y-5 text-center animate-fade-in">
            <div className="w-10 h-10 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center shadow-liquid-sm">
              <div className="w-4 h-4 border-2 border-white/20 border-t-white/80 rounded-full animate-spin" />
            </div>
            <div className="space-y-1">
              <h4 className="text-sm font-semibold text-white/85 tracking-tight">Caricamento trascrizione...</h4>
              <p className="text-xs text-white/40">Sincronizzazione dei passaggi audio in corso</p>
            </div>
            <div className="w-full space-y-2.5 pt-4 opacity-40">
              <div className="h-2.5 bg-white/[0.12] rounded-full w-4/5 mx-auto animate-pulse" />
              <div className="h-2.5 bg-white/[0.08] rounded-full w-full animate-pulse" />
              <div className="h-2.5 bg-white/[0.10] rounded-full w-3/4 mx-auto animate-pulse" />
            </div>
          </div>
        ) : hasSyncData ? (
          <div ref={textWrapperRef} className="relative select-text space-y-7">
            {/* ── Playhead Caret Cursor (Barra verticale text-caret) ── */}
            <div
              ref={caretRef}
              className={clsx(
                "pointer-events-none absolute z-20 rounded-full",
                !isPlaying && "animate-caret-blink"
              )}
              style={{
                width: '2.5px',
                background: subjectColor
                  ? `linear-gradient(180deg, #ffffff 0%, ${subjectColor} 100%)`
                  : 'linear-gradient(180deg, #ffffff 0%, #38bdf8 100%)',
                boxShadow: `0 0 8px ${subjectColor || '#38bdf8'}, 0 0 16px ${subjectColor || '#38bdf8'}80`,
                top: 0,
                left: 0,
                transform: 'translate3d(0, 0, 0)',
                opacity: 0,
                transition: 'transform 120ms linear, opacity 150ms ease',
              }}
              aria-hidden="true"
            />

            {paragraphs.map((paragraph, pIdx) => (
              <p
                key={pIdx}
                className="leading-[2.2] text-lg sm:text-[1.28rem] font-normal tracking-tight text-white/90 select-text"
              >
                {paragraph.map((line) =>
                  line.words.map((word, wIdx) => {
                    const wordId = `${line.lineIdx}-${wIdx}`;
                    const isWordActive = activeWordId === wordId;
                    const isWordPast = currentMs > word.end;

                    return (
                      <React.Fragment key={wordId}>
                        <span
                          ref={(el) => {
                            if (el) wordElementsRef.current.set(wordId, el);
                            else wordElementsRef.current.delete(wordId);
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleWordClick({
                              id: wordId,
                              lineIdx: line.lineIdx,
                              wordIdx: wIdx,
                              text: word.text,
                              start: word.start,
                              end: word.end,
                            });
                          }}
                          className={clsx(
                            "inline cursor-pointer select-text transition-colors duration-150",
                            isWordActive
                              ? "text-white font-semibold drop-shadow-[0_0_10px_rgba(255,255,255,0.7)]"
                              : isWordPast
                              ? "text-white/95"
                              : "text-white/45 hover:text-white/80"
                          )}
                          title={`[${formatTimeMs(word.start)}] Clicca per ascoltare`}
                        >
                          {word.text}
                        </span>
                        {' '}
                      </React.Fragment>
                    );
                  })
                )}
              </p>
            ))}
          </div>
        ) : rawTranscript ? (
          <div className="space-y-4">
            {rawTranscript.split(/\n\n+/).map((para, idx) => (
              <p
                key={idx}
                className="leading-[2.2] text-lg sm:text-[1.28rem] font-medium tracking-tight text-white/85"
              >
                {para}
              </p>
            ))}
          </div>
        ) : (
          <div className="py-20 text-center space-y-3">
            <Sparkles size={24} className="text-amber-300 mx-auto opacity-70" />
            <h4 className="text-sm font-semibold text-white/80">Nessun testo trascritto</h4>
            <p className="text-xs text-white/40 max-w-sm mx-auto">
              Nessuna trascrizione disponibile per questa lezione.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
