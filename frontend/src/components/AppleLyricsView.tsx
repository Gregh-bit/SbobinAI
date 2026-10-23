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
  onCopyTranscript?: () => void;
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
  const containerRef = useRef<HTMLDivElement | null>(null);
  const lineRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const isProgrammaticScrollRef = useRef<boolean>(false);
  const userScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Palette dinamica 3-4 colori mesh
  const palette = useMemo(() => extractPalette(subjectColor), [subjectColor]);

  // Conteggio totale parole
  const totalWords = useMemo(() => {
    if (lines && lines.length > 0) {
      return lines.reduce((acc, l) => acc + l.words.length, 0);
    }
    if (rawTranscript) {
      return rawTranscript.split(/\s+/).filter(Boolean).length;
    }
    return 0;
  }, [lines, rawTranscript]);

  // Raggruppamento naturale in paragrafi per lettura fluida
  const paragraphs = useMemo(() => {
    if (!lines || lines.length === 0) return [];
    const result: ParagraphLine[][] = [];
    let currentPara: ParagraphLine[] = [];

    lines.forEach((line, idx) => {
      currentPara.push({ ...line, lineIdx: idx });

      const lastWord = line.words[line.words.length - 1];
      const lastText = lastWord ? lastWord.text : '';
      const hasPunctuation = /[.!?]$/.test(lastText);
      const nextLine = lines[idx + 1];
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
  }, [lines]);

  // ── Animation Loop ottimizzato (Sincronizzazione fluidissima ~60 FPS) ──
  useEffect(() => {
    let animId: number;

    const tick = () => {
      const audio = audioRef.current;
      if (audio && !audio.paused) {
        setCurrentMs(Math.round(audio.currentTime * 1000));
      }
      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(animId);
    };
  }, [audioRef]);

  // Ascolto eventi audio nativi per scrubbing istantaneo
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const syncTime = () => setCurrentMs(Math.round(audio.currentTime * 1000));
    audio.addEventListener('timeupdate', syncTime);
    audio.addEventListener('seeked', syncTime);
    audio.addEventListener('play', syncTime);
    audio.addEventListener('playing', syncTime);
    audio.addEventListener('pause', syncTime);
    return () => {
      audio.removeEventListener('timeupdate', syncTime);
      audio.removeEventListener('seeked', syncTime);
      audio.removeEventListener('play', syncTime);
      audio.removeEventListener('playing', syncTime);
      audio.removeEventListener('pause', syncTime);
    };
  }, [audioRef]);

  // Rilevamento riga attiva corrente
  const activeLineIndex = useMemo(() => {
    if (!lines || lines.length === 0) return 0;
    const exact = lines.findIndex(l => currentMs >= l.lineStart && currentMs <= l.lineEnd);
    if (exact !== -1) return exact;
    if (currentMs < lines[0].lineStart) return 0;
    for (let i = lines.length - 1; i >= 0; i--) {
      if (currentMs >= lines[i].lineStart) {
        return i;
      }
    }
    return 0;
  }, [lines, currentMs]);

  // ── Scroll Fluido alla riga attiva ──────────────────────────────────────────
  const scrollToActiveLine = useCallback((force = false) => {
    if (!isAutoScrollLocked && !force) return;
    if (activeLineIndex < 0 || !lineRefs.current[activeLineIndex] || !containerRef.current) return;

    const container = containerRef.current;
    const targetEl = lineRefs.current[activeLineIndex];
    if (targetEl && container) {
      isProgrammaticScrollRef.current = true;
      const containerRect = container.getBoundingClientRect();
      const targetRect = targetEl.getBoundingClientRect();
      const relativeTop = targetRect.top - containerRect.top + container.scrollTop;
      const targetScrollTop = relativeTop - (container.clientHeight / 2) + (targetRect.height / 2);
      container.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: 'smooth',
      });
      setTimeout(() => {
        isProgrammaticScrollRef.current = false;
      }, 500);
    }
  }, [activeLineIndex, isAutoScrollLocked]);

  useEffect(() => {
    scrollToActiveLine();
  }, [activeLineIndex, scrollToActiveLine]);

  // ── Gestione Interruzione Manuale Utente (Scroll/Wheel) ─────────────────────
  const handleUserInteraction = () => {
    if (isProgrammaticScrollRef.current) return;

    setIsAutoScrollLocked((prev) => (prev ? false : prev));
    if (userScrollTimeoutRef.current) {
      clearTimeout(userScrollTimeoutRef.current);
    }

    // Riaggancia l'auto-scroll dopo 4.5 secondi di inattività
    userScrollTimeoutRef.current = setTimeout(() => {
      setIsAutoScrollLocked(true);
      scrollToActiveLine(true);
    }, 4500);
  };

  useEffect(() => {
    return () => {
      if (userScrollTimeoutRef.current) {
        clearTimeout(userScrollTimeoutRef.current);
      }
    };
  }, []);

  const hasSyncData = lines && lines.length > 0;
  const isInitialState =
    !isPlaying &&
    (currentMs === 0 || (lines && lines.length > 0 && currentMs < lines[0].lineStart));

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

        {/* Pulsante Copia */}
        {onCopyTranscript && (
          <button
            type="button"
            onClick={onCopyTranscript}
            className="h-7 px-3 rounded-full liquid-glass-pill hover:bg-white/[0.14] text-xs font-medium text-white/85 border border-white/[0.14] transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm active:scale-95"
            title="Copia l'intera trascrizione negli appunti"
          >
            {copiedTranscript ? (
              <>
                <Check size={12} className="text-emerald-400" />
                <span className="text-emerald-300 font-semibold text-[11px]">Copiato!</span>
              </>
            ) : (
              <>
                <Copy size={12} className="text-white/60" />
                <span className="text-[11px]">Copia</span>
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
            onClick={() => {
              setIsAutoScrollLocked(true);
              scrollToActiveLine(true);
            }}
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
          paragraphs.map((paragraph, pIdx) => (
            <p
              key={pIdx}
              className="leading-[2.2] text-lg sm:text-[1.28rem] font-medium tracking-tight text-white/95"
            >
              {paragraph.map((line) => {
                const isLineActive = line.lineIdx === activeLineIndex;
                const isLinePast = line.lineIdx < activeLineIndex;

                return (
                  <span
                    key={line.lineIdx}
                    ref={(el) => {
                      lineRefs.current[line.lineIdx] = el;
                    }}
                    className={clsx(
                      "inline transition-all duration-300",
                      isLineActive
                        ? "opacity-100"
                        : isLinePast
                        ? "opacity-75"
                        : "opacity-35"
                    )}
                  >
                    {line.words.map((word, wIdx) => {
                      const isWordActive = isLineActive && currentMs >= word.start && currentMs <= word.end;
                      const isWordPastInLine = isLineActive && currentMs > word.end;
                      const isWordFutureInLine = isLineActive && currentMs < word.start;

                      let wordStyle: React.CSSProperties = {};
                      if (!isPlaying && isInitialState) {
                        wordStyle = {
                          color: 'rgba(255, 255, 255, 0.88)',
                          opacity: 1,
                        };
                      } else if (isWordActive) {
                        const dur = Math.max(1, word.end - word.start);
                        const ratio = Math.min(1, Math.max(0, (currentMs - word.start) / dur));
                        const pct = Math.round(ratio * 100);
                        const feather = Math.min(100, pct + 3);

                        wordStyle = {
                          backgroundImage: `linear-gradient(90deg, #ffffff 0%, #ffffff ${pct}%, rgba(255, 255, 255, 0.35) ${feather}%, rgba(255, 255, 255, 0.35) 100%)`,
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          transition: 'none',
                        };
                      } else if (isWordPastInLine || isLinePast) {
                        wordStyle = {
                          color: '#ffffff',
                          opacity: 0.95,
                        };
                      } else if (isWordFutureInLine) {
                        wordStyle = {
                          color: 'rgba(255, 255, 255, 0.35)',
                          opacity: 0.40,
                        };
                      } else {
                        wordStyle = {
                          color: 'rgba(255, 255, 255, 0.25)',
                          opacity: 0.30,
                        };
                      }

                      return (
                        <span
                          key={wIdx}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSeek(word.start / 1000);
                          }}
                          style={wordStyle}
                          className={clsx(
                            "inline-block cursor-pointer select-none",
                            "mr-[0.28em] my-0.5",
                            !isWordActive && "transition-all duration-200 ease-out",
                            "hover:opacity-100 hover:text-white"
                          )}
                          title={`[${formatTimeMs(word.start)}] Clicca per ascoltare`}
                        >
                          {word.text}
                        </span>
                      );
                    })}
                    {" "}
                  </span>
                );
              })}
            </p>
          ))
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
