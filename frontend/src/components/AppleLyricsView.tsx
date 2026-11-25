// components/AppleLyricsView.tsx — Continuous Borderless Transcript with Synchronized Caret
import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import clsx from 'clsx';
import { ArrowDownCircle, Check, Copy } from 'lucide-react';
import type { KaraokeLine } from '../types';
import { useDelayedLoading, TranscriptSkeleton } from './Skeleton';

interface AppleLyricsViewProps {
  lines: KaraokeLine[] | null;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  audioPath?: string;
  isPlaying: boolean;
  onSeek: (seconds: number) => void;
  lectureTitle?: string;
  rawTranscript?: string;
  onCopyTranscript?: (textToCopy?: string) => void;
  copiedTranscript?: boolean;
  isLoadingTimestamps?: boolean;
}

const TOPIC_SHIFT_RE = /^(allora|quindi\s+passiamo|passiamo\s+(?:a|ora|adesso)|adesso\s+vediamo|ora\s+vediamo|il\s+prossimo\s+argomento|un\s+altro\s+aspetto|cambiamo\s+argomento|in\s+questo\s+capitolo)/i;

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
  audioPath,
  isPlaying,
  onSeek,
  rawTranscript,
  onCopyTranscript,
  copiedTranscript,
  isLoadingTimestamps,
}) => {
  const [currentMs, setCurrentMs] = useState<number>(0);
  const [isAutoScrollLocked, setIsAutoScrollLocked] = useState<boolean>(true);
  const [activeWordId, setActiveWordId] = useState<string | null>(null);
  const showTranscriptSkeleton = useDelayedLoading(Boolean(isLoadingTimestamps), 180, 220);

  const currentMsRef = useRef<number>(0);
  currentMsRef.current = currentMs;

  const textWrapperRef = useRef<HTMLDivElement | null>(null);
  const caretRef = useRef<HTMLDivElement | null>(null);
  const wordElementsRef = useRef<Map<string, HTMLSpanElement>>(new Map());
  const lastYRef = useRef<number | null>(null);
  const activeWordIdRef = useRef<string | null>(null);
  const isAutoScrollLockedRef = useRef<boolean>(true);
  const isProgrammaticScrollRef = useRef<boolean>(false);
  const userScrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getScrollContainer = useCallback((): HTMLElement | null => {
    return (textWrapperRef.current?.closest('.overflow-y-auto') as HTMLElement | null) ?? null;
  }, []);

  useEffect(() => {
    isAutoScrollLockedRef.current = isAutoScrollLocked;
  }, [isAutoScrollLocked]);

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

    // 3. Testo non corrispondente a rawTranscript (valutato solo con campioni consistenti per evitare falsi positivi)
    if (rawWords.length >= 50 && allLinesWords.length >= 50) {
      const rawWordSet = new Set(rawWords);
      const matchingWords = allLinesWords.filter((w) => rawWordSet.has(w));
      const matchRatio = matchingWords.length / allLinesWords.length;

      // Se meno del 20% delle parole dei timestamp compare in rawTranscript
      if (matchRatio < 0.20) {
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

  // Testo continuo: niente a capo arbitrari di Whisper, solo su pause molto lunghe o cambi netti d'argomento
  const paragraphs = useMemo(() => {
    if (!effectiveLines || effectiveLines.length === 0) return [];
    const result: ParagraphLine[][] = [];
    let currentPara: ParagraphLine[] = [];
    let paraWordCount = 0;

    effectiveLines.forEach((line, idx) => {
      currentPara.push({ ...line, lineIdx: idx });
      paraWordCount += line.words.length;

      const lastWord = line.words[line.words.length - 1];
      const lastText = lastWord ? lastWord.text : '';
      const hasPunctuation = /[.!?]$/.test(lastText);
      const nextLine = effectiveLines[idx + 1];
      if (!nextLine) return;

      const pauseBeforeNext = nextLine.lineStart - line.lineEnd;
      const nextText = nextLine.words.map((w) => w.text).join(' ').trim();
      const isLongPause = pauseBeforeNext >= 7000 || (hasPunctuation && pauseBeforeNext >= 4500);
      const isTopicShift = hasPunctuation && pauseBeforeNext >= 2200 && paraWordCount >= 40 && TOPIC_SHIFT_RE.test(nextText);

      if (isLongPause || isTopicShift) {
        result.push(currentPara);
        currentPara = [];
        paraWordCount = 0;
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
      // Prima che inizi il parlato (silenzio iniziale / intro): nessun cursore attivo
      if (ms < flatWords[0].start) return null;
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
      return null;
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
        if (activeWordIdRef.current !== null) {
          activeWordIdRef.current = null;
          setActiveWordId(null);
        }
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
      const targetHeight = Math.max(18, wordRect.height);

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

      // 4. Auto-scroll fluido sul contenitore pagina al cambio riga
      const scrollContainer = getScrollContainer();
      if (isAutoScrollLockedRef.current && scrollContainer && isLineChange) {
        const containerRect = scrollContainer.getBoundingClientRect();
        const wrapperTopInScroll = wrapRect.top - containerRect.top + scrollContainer.scrollTop;
        const targetScrollTop = wrapperTopInScroll + targetY - (scrollContainer.clientHeight * 0.38);
        if (Math.abs(scrollContainer.scrollTop - targetScrollTop) > 30) {
          isProgrammaticScrollRef.current = true;
          scrollContainer.scrollTo({
            top: Math.max(0, targetScrollTop),
            behavior: 'smooth',
          });
          setTimeout(() => {
            isProgrammaticScrollRef.current = false;
          }, 450);
        }
      }
    },
    [flatWords, findActiveWord, getScrollContainer]
  );

  const handleWordClick = useCallback(
    (word: FlatWord) => {
      onSeek(word.start / 1000);
      currentMsRef.current = word.start;
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
        currentMsRef.current = ms;
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
      currentMsRef.current = ms;
      setCurrentMs(ms);
      updateCaretPosition(ms, false);
    };

    const handleSeeked = () => {
      const ms = Math.round(audio.currentTime * 1000);
      currentMsRef.current = ms;
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
  }, [audioRef, audioPath, updateCaretPosition]);

  // Sync iniziale e al ridimensionamento della finestra
  useEffect(() => {
    if (flatWords.length > 0) {
      const timer = setTimeout(() => {
        updateCaretPosition(currentMsRef.current, true);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [flatWords, updateCaretPosition]);

  useEffect(() => {
    const handleResize = () => {
      updateCaretPosition(currentMsRef.current, true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [updateCaretPosition]);

  // ── Gestione Interruzione Manuale Utente sul contenitore scrollabile della pagina ──
  const handleUserInteraction = useCallback(() => {
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
      updateCaretPosition(currentMsRef.current, true);
    }, 4500);
  }, [updateCaretPosition]);

  useEffect(() => {
    const scrollContainer = getScrollContainer();
    if (!scrollContainer) return;
    scrollContainer.addEventListener('wheel', handleUserInteraction, { passive: true });
    scrollContainer.addEventListener('touchmove', handleUserInteraction, { passive: true });
    scrollContainer.addEventListener('scroll', handleUserInteraction, { passive: true });
    return () => {
      scrollContainer.removeEventListener('wheel', handleUserInteraction);
      scrollContainer.removeEventListener('touchmove', handleUserInteraction);
      scrollContainer.removeEventListener('scroll', handleUserInteraction);
    };
  }, [getScrollContainer, handleUserInteraction]);

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
    updateCaretPosition(currentMsRef.current, true);
    const scrollContainer = getScrollContainer();
    const wrapperEl = textWrapperRef.current;
    if (scrollContainer && wrapperEl) {
      const containerRect = scrollContainer.getBoundingClientRect();
      const wrapRect = wrapperEl.getBoundingClientRect();
      const wrapperTopInScroll = wrapRect.top - containerRect.top + scrollContainer.scrollTop;
      const caretTop = lastYRef.current ?? 0;
      const targetScrollTop = wrapperTopInScroll + caretTop - (scrollContainer.clientHeight * 0.38);
      isProgrammaticScrollRef.current = true;
      scrollContainer.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: 'smooth',
      });
      setTimeout(() => {
        isProgrammaticScrollRef.current = false;
      }, 450);
    }
  }, [getScrollContainer, updateCaretPosition]);

  const hasSyncData = Boolean(effectiveLines && effectiveLines.length > 0);
  const hasRawTranscript = Boolean(rawTranscript && rawTranscript.trim().length > 0);
  const isAtBeginningIdle = !isPlaying && currentMs <= 50;

  return (
    <div className="relative w-full">
      {/* ── 1. Intestazione discreta senza box ── */}
      {(totalWords > 0 || isLoadingTimestamps || onCopyTranscript || rawTranscript || hasSyncData) && (
        <div className="flex items-center justify-between mb-5 text-xs text-zinc-400 select-none">
          <div className="flex items-center gap-2">
            {totalWords > 0 && (
              <span className="tabular-nums text-zinc-500">
                {totalWords} {totalWords === 1 ? 'parola' : 'parole'}
              </span>
            )}
            {isLoadingTimestamps && (
              <div className="w-3 h-3 rounded-full border border-white/20 border-t-white/80 animate-spin opacity-80" />
            )}
          </div>

          {(onCopyTranscript || rawTranscript || hasSyncData) && (
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] transition-colors cursor-pointer"
              title="Copia l'intera trascrizione negli appunti"
            >
              {isCopied ? (
                <>
                  <Check size={12} className="text-emerald-400" />
                  <span className="text-emerald-300 font-medium">Copiato</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copia</span>
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* ── 2. Testo continuo appoggiato direttamente sulla tela dell'app ── */}
      <div className="focus:outline-none select-text" tabIndex={0}>
        {hasSyncData ? (
          <div ref={textWrapperRef} className="relative select-text space-y-5">
            {/* ── Cursore verticale bianco (Caret) ── */}
            <div
              ref={caretRef}
              className={clsx(
                'pointer-events-none absolute z-20 rounded-full',
                !isPlaying && 'animate-caret-blink'
              )}
              style={{
                width: '2px',
                backgroundColor: '#ffffff',
                borderRadius: '1px',
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
                className="text-[15.5px] sm:text-base leading-[1.9] font-normal text-zinc-200/90 select-text"
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
                            'inline cursor-pointer select-text transition-colors duration-150',
                            isAtBeginningIdle
                              ? 'text-zinc-200/90 hover:text-white'
                              : isWordActive
                                ? 'text-white font-medium'
                                : isWordPast
                                  ? 'text-zinc-100'
                                  : 'text-zinc-400/75 hover:text-zinc-100'
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
        ) : hasRawTranscript ? (
          <div className="space-y-5">
            {rawTranscript!.split(/\n\n+/).map((para, idx) => (
              <p
                key={idx}
                className="text-[15.5px] sm:text-base leading-[1.9] font-normal text-zinc-200/90"
              >
                {para}
              </p>
            ))}
          </div>
        ) : showTranscriptSkeleton ? (
          <TranscriptSkeleton />
        ) : isLoadingTimestamps ? (
          <div className="min-h-[260px]" aria-busy="true" />
        ) : (
          <div className="py-20 text-center space-y-2">
            <h4 className="text-sm font-semibold text-white/75">Nessuna trascrizione</h4>
            <p className="text-xs text-white/40 max-w-sm mx-auto">
              Non è stato rilevato alcun parlato in questa registrazione.
            </p>
          </div>
        )}
      </div>

      {/* ── 3. Pillola discreta in basso per riallinearsi all'audio durante lo scroll manuale ── */}
      {!isAutoScrollLocked && hasSyncData && isPlaying && (
        <div className="sticky bottom-24 z-30 flex justify-center mt-6 pointer-events-none animate-fade-in">
          <button
            type="button"
            onClick={centerOnActiveCaret}
            className="pointer-events-auto px-3.5 py-1.5 rounded-full bg-[#121622]/90 hover:bg-[#1a2033] text-xs font-medium text-zinc-200 hover:text-white border border-white/[0.14] shadow-lg backdrop-blur-xl flex items-center gap-1.5 cursor-pointer transition-colors"
          >
            <ArrowDownCircle size={13} className="text-amber-300" />
            <span>Torna all&apos;audio</span>
          </button>
        </div>
      )}
    </div>
  );
};
