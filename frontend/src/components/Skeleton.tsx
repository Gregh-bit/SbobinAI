// components/Skeleton.tsx — Unified Liquid Glass Skeleton Screen System for SbobinAI
import { useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';

/**
 * Anti-flash delayed loading hook.
 *
 * - Waits `delayMs` (default 160ms) before showing a skeleton so fast local
 *   SQLite loads never flash.
 * - If `delayMs <= 0`, transitions immediately to visible.
 * - Once visible, stays mounted for at least `minDisplayMs` (default 240ms)
 *   to prevent 1-frame flicker.
 * - Cleans up all timers safely on unmount.
 */
export function useDelayedLoading(
  isLoading: boolean,
  delayMs = 160,
  minDisplayMs = 240
): boolean {
  const initialVisible = isLoading && delayMs <= 0;
  const [visible, setVisible] = useState<boolean>(initialVisible);

  const isVisibleRef = useRef<boolean>(initialVisible);
  const shownAtRef = useRef<number | null>(initialVisible ? Date.now() : null);
  const delayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const minDisplayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const clearDelayTimer = () => {
      if (delayTimerRef.current !== null) {
        clearTimeout(delayTimerRef.current);
        delayTimerRef.current = null;
      }
    };

    const clearMinDisplayTimer = () => {
      if (minDisplayTimerRef.current !== null) {
        clearTimeout(minDisplayTimerRef.current);
        minDisplayTimerRef.current = null;
      }
    };

    if (isLoading) {
      clearMinDisplayTimer();

      if (isVisibleRef.current) {
        return;
      }

      if (delayMs <= 0) {
        clearDelayTimer();
        isVisibleRef.current = true;
        shownAtRef.current = Date.now();
        setVisible(true);
      } else if (delayTimerRef.current === null) {
        delayTimerRef.current = setTimeout(() => {
          delayTimerRef.current = null;
          isVisibleRef.current = true;
          shownAtRef.current = Date.now();
          setVisible(true);
        }, delayMs);
      }
    } else {
      clearDelayTimer();

      if (!isVisibleRef.current) {
        shownAtRef.current = null;
        return;
      }

      const elapsed =
        shownAtRef.current !== null ? Date.now() - shownAtRef.current : minDisplayMs;
      const remaining = Math.max(0, minDisplayMs - elapsed);

      if (remaining === 0) {
        clearMinDisplayTimer();
        isVisibleRef.current = false;
        shownAtRef.current = null;
        setVisible(false);
      } else {
        clearMinDisplayTimer();
        minDisplayTimerRef.current = setTimeout(() => {
          minDisplayTimerRef.current = null;
          isVisibleRef.current = false;
          shownAtRef.current = null;
          setVisible(false);
        }, remaining);
      }
    }
  }, [isLoading, delayMs, minDisplayMs]);

  useEffect(() => {
    return () => {
      if (delayTimerRef.current !== null) {
        clearTimeout(delayTimerRef.current);
        delayTimerRef.current = null;
      }
      if (minDisplayTimerRef.current !== null) {
        clearTimeout(minDisplayTimerRef.current);
        minDisplayTimerRef.current = null;
      }
    };
  }, []);

  return visible;
}

/* ── Primitives ────────────────────────────────────────────────────────── */

export type SkeletonTone = 'soft' | 'default' | 'strong';

const SKELETON_TONE_CLASSES: Record<SkeletonTone, string> = {
  soft: 'bg-white/[0.04]',
  default: 'bg-white/[0.055]',
  strong: 'bg-white/[0.08]',
};

export interface SkeletonPrimitiveProps {
  tone?: SkeletonTone;
  className?: string;
}

export function SkeletonBlock({ tone = 'default', className }: SkeletonPrimitiveProps) {
  const hasCustomRadius = Boolean(className && /\brounded(?:-[^\s]+)?\b/.test(className));
  return (
    <div
      aria-hidden="true"
      className={clsx(
        !hasCustomRadius && 'rounded-xl',
        SKELETON_TONE_CLASSES[tone],
        'relative overflow-hidden skeleton-shimmer',
        className
      )}
    />
  );
}

export function SkeletonPill({ tone = 'default', className }: SkeletonPrimitiveProps) {
  return (
    <div
      aria-hidden="true"
      className={clsx(
        'rounded-full',
        SKELETON_TONE_CLASSES[tone],
        'relative overflow-hidden skeleton-shimmer',
        className
      )}
    />
  );
}

/* ── Layout-Matched Skeletons ──────────────────────────────────────────── */

export interface SidebarSkeletonProps {
  count?: number;
  className?: string;
}

const SIDEBAR_TITLE_WIDTHS = ['w-3/5', 'w-4/5', 'w-1/2', 'w-[72%]', 'w-[64%]'];

/**
 * Course list skeleton matching `Sidebar.tsx` (4–5 rounded-xl rows with color dot,
 * title bar, and compact rounded-md lecture count badge).
 */
export function SidebarSkeleton({ count = 5, className }: SidebarSkeletonProps) {
  const rows = Array.from({ length: Math.max(1, count) });
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento corsi"
      className={clsx('space-y-1 animate-fade-in select-none', className)}
    >
      {rows.map((_, idx) => (
        <div
          key={idx}
          className={clsx(
            'w-full h-9 px-3 rounded-xl flex items-center gap-2.5 border',
            idx === 0
              ? 'bg-white/[0.05] border-white/[0.09]'
              : 'bg-white/[0.02] border-transparent'
          )}
        >
          <SkeletonPill tone="strong" className="w-2 h-2 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <SkeletonPill
              className={clsx(
                'h-2.5',
                SIDEBAR_TITLE_WIDTHS[idx % SIDEBAR_TITLE_WIDTHS.length]
              )}
            />
          </div>
          <SkeletonBlock tone="soft" className="w-5 h-4 rounded-md flex-shrink-0" />
        </div>
      ))}
    </div>
  );
}

export interface QuadernoSkeletonProps {
  className?: string;
}

const QUADERNO_CARD_TITLE_WIDTHS = ['w-4/5', 'w-3/4', 'w-11/12', 'w-2/3'];

/**
 * Full 2-column Quaderno skeleton: left `w-72` chronological index sidebar
 * (search bar + 4 lecture cards) and right document reading sheet (header,
 * metadata, `h-7` segmented control pill, and centered `max-w-3xl` paper card).
 */
export function QuadernoSkeleton({ className }: QuadernoSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento Quaderno"
      className={clsx('flex-1 flex overflow-hidden select-none animate-fade-in', className)}
    >
      {/* Left Column: w-72 Index Sidebar */}
      <aside className="w-72 flex-shrink-0 flex flex-col h-full border-r border-white/[0.08] liquid-glass-subtle">
        {/* Search bar + action buttons */}
        <div className="p-3 border-b border-white/[0.06] flex items-center gap-1.5">
          <SkeletonBlock tone="soft" className="flex-1 h-7" />
          <SkeletonBlock tone="default" className="w-7 h-7 flex-shrink-0" />
          <SkeletonBlock tone="soft" className="w-7 h-7 flex-shrink-0" />
        </div>

        {/* 4 Lecture Cards */}
        <div className="flex-1 overflow-hidden p-2 space-y-1.5">
          {QUADERNO_CARD_TITLE_WIDTHS.map((widthClass, idx) => (
            <div
              key={idx}
              className={clsx(
                'p-3 rounded-2xl border space-y-2.5',
                idx === 0
                  ? 'bg-white/[0.06] border-white/[0.11] shadow-liquid-sm'
                  : 'bg-white/[0.02] border-white/[0.05]'
              )}
            >
              <SkeletonPill className={clsx('h-3', widthClass)} />
              <div className="flex items-center justify-between gap-2">
                <SkeletonPill tone="soft" className="h-2.5 w-20" />
                <SkeletonPill tone="soft" className="h-4 w-16" />
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Right Column: Document Reading Sheet */}
      <div className="flex-1 flex flex-col overflow-hidden content-canvas">
        {/* Sheet Header */}
        <div className="px-8 pt-6 pb-4 border-b border-white/[0.06] bg-white/[0.015] flex flex-col gap-4 flex-shrink-0">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0 space-y-2.5">
              <SkeletonPill tone="strong" className="h-6 w-2/5 max-w-sm" />
              <div className="flex items-center gap-2.5">
                <SkeletonPill tone="soft" className="h-3 w-24" />
                <SkeletonPill tone="soft" className="h-3 w-16" />
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <SkeletonPill className="h-8 w-24" />
              <SkeletonPill className="h-8 w-20" />
              <SkeletonPill className="w-8 h-8" />
            </div>
          </div>

          {/* Segmented Control Pill (matches SegmentedControl size="sm" h-7) */}
          <div className="h-7 w-fit p-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center gap-0.5">
            <SkeletonPill tone="strong" className="h-6 w-20" />
            <SkeletonPill tone="soft" className="h-6 w-24" />
            <SkeletonPill tone="soft" className="h-6 w-20" />
          </div>
        </div>

        {/* Document Card Body (centered max-w-3xl matching QuadernoView) */}
        <div className="flex-1 overflow-hidden px-6 sm:px-12 py-8">
          <div className="max-w-3xl mx-auto p-4 sm:p-6 rounded-3xl bg-white/[0.02] border border-white/[0.06] shadow-liquid-sm space-y-6">
            <SkeletonBlock tone="strong" className="h-6 w-1/2 max-w-md" />
            <div className="space-y-2.5">
              <SkeletonPill className="h-3.5 w-full" />
              <SkeletonPill className="h-3.5 w-[93%]" />
              <SkeletonPill className="h-3.5 w-[76%]" />
            </div>
            <SkeletonBlock tone="strong" className="h-4 w-1/3 max-w-xs rounded-lg" />
            <div className="space-y-2.5">
              <SkeletonPill className="h-3.5 w-[95%]" />
              <SkeletonPill className="h-3.5 w-[88%]" />
              <SkeletonPill className="h-3.5 w-[66%]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export interface ChapterGenerationSkeletonProps {
  statusMessage?: string;
  progress?: number;
  className?: string;
}

/**
 * Paper-matched skeleton for `QuadernoView` when "Sintesi IA" is generating
 * a structured chapter. Mirrors the real `rounded-3xl` markdown document sheet
 * with H1, intro paragraphs, H2 + bullets, H2 + paragraphs, an optional 2px
 * hairline progress bar, and a minimal status whisper.
 */
export function ChapterGenerationSkeleton({
  statusMessage,
  progress,
  className,
}: ChapterGenerationSkeletonProps) {
  const hasProgress = typeof progress === 'number';
  const clampedProgress = hasProgress ? Math.max(0, Math.min(100, progress)) : 0;

  return (
    <div
      aria-busy="true"
      aria-label={statusMessage || 'Generazione capitolo in corso'}
      className={clsx(
        'p-4 sm:p-6 rounded-3xl bg-white/[0.02] border border-white/[0.06] shadow-liquid-sm w-full min-w-0 animate-fade-in select-none',
        className
      )}
    >
      <div className="space-y-6">
        {/* H1 Title Bar */}
        <div className="space-y-3">
          <SkeletonBlock tone="strong" className="h-6 w-3/5 max-w-md" />
          <div className="space-y-2.5 pt-1">
            <SkeletonPill className="h-3.5 w-full" />
            <SkeletonPill className="h-3.5 w-[94%]" />
            <SkeletonPill className="h-3.5 w-[78%]" />
          </div>
        </div>

        {/* H2 Section + 3 Bullet Items */}
        <div className="space-y-3 pt-1">
          <SkeletonBlock tone="strong" className="h-[18px] w-2/5 max-w-xs rounded-lg" />
          <div className="space-y-2.5 pl-1">
            {['w-[86%]', 'w-[72%]', 'w-[80%]'].map((widthClass, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <span
                  aria-hidden="true"
                  className="w-1.5 h-1.5 rounded-full bg-white/[0.16] flex-shrink-0"
                />
                <SkeletonPill className={clsx('h-3', widthClass)} />
              </div>
            ))}
          </div>
        </div>

        {/* H2 Section + 3 Paragraph Lines */}
        <div className="space-y-3 pt-1">
          <SkeletonBlock tone="strong" className="h-[18px] w-1/3 max-w-[230px] rounded-lg" />
          <div className="space-y-2.5">
            <SkeletonPill className="h-3.5 w-[96%]" />
            <SkeletonPill className="h-3.5 w-[89%]" />
            <SkeletonPill className="h-3.5 w-[64%]" />
          </div>
        </div>

        {/* Hairline Progress + Subtle Apple-like Status Whisper */}
        {(hasProgress || statusMessage) && (
          <div className="pt-4 border-t border-white/[0.05] space-y-2.5">
            {hasProgress && (
              <div
                role="progressbar"
                aria-valuenow={Math.round(clampedProgress)}
                aria-valuemin={0}
                aria-valuemax={100}
                className="h-[2px] w-full max-w-xs mx-auto rounded-full bg-white/[0.07] overflow-hidden"
              >
                <div
                  className="h-full rounded-full bg-amber-300/65 transition-all duration-300"
                  style={{ width: `${clampedProgress}%` }}
                />
              </div>
            )}
            {statusMessage && (
              <div
                className={clsx(
                  'flex items-center gap-2',
                  hasProgress && 'justify-center'
                )}
              >
                <span
                  aria-hidden="true"
                  className="w-1.5 h-1.5 rounded-full bg-white/35 animate-pulse-subtle flex-shrink-0"
                />
                <span className="text-[11px] font-medium tracking-tight text-white/40 chat-shimmer-text truncate">
                  {statusMessage}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export interface MaterialiGridSkeletonProps {
  count?: number;
  className?: string;
}

const MATERIALI_TITLE_WIDTHS = [
  ['w-[88%]', 'w-[58%]'],
  ['w-[76%]', 'w-[45%]'],
  ['w-[92%]', 'w-[66%]'],
  ['w-[82%]', 'w-[52%]'],
];

/**
 * 8-card grid skeleton matching `MaterialiView.tsx`
 * (`grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4`).
 */
export function MaterialiGridSkeleton({
  count = 8,
  className,
}: MaterialiGridSkeletonProps) {
  const cards = Array.from({ length: Math.max(1, count) });
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento materiali"
      className={clsx(
        'grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 animate-fade-in select-none',
        className
      )}
    >
      {cards.map((_, idx) => {
        const [line1, line2] =
          MATERIALI_TITLE_WIDTHS[idx % MATERIALI_TITLE_WIDTHS.length];
        return (
          <div
            key={idx}
            className="p-4 rounded-2xl liquid-glass border border-white/[0.08] flex flex-col justify-between min-h-[146px]"
          >
            <div>
              {/* Icon Box + Tag Pill */}
              <div className="flex items-start justify-between mb-3">
                <SkeletonBlock className="w-10 h-10 border border-white/[0.08]" />
                <SkeletonPill tone="soft" className="h-5 w-16" />
              </div>

              {/* 2-line Title */}
              <div className="space-y-2">
                <SkeletonPill className={clsx('h-3', line1)} />
                <SkeletonPill className={clsx('h-3', line2)} />
              </div>
            </div>

            {/* Footer Metadata */}
            <div className="mt-4 pt-2.5 border-t border-white/[0.06] flex items-center justify-between gap-2">
              <SkeletonPill tone="soft" className="h-2.5 w-12" />
              <SkeletonPill tone="soft" className="h-2.5 w-14" />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export interface MaterialiFinderSkeletonProps {
  count?: number;
  className?: string;
}

const FINDER_ROW_WIDTHS = ['w-3/5', 'w-1/2', 'w-2/3', 'w-[54%]', 'w-7/12'];

/**
 * 5 horizontal rows matching `MaterialiFinderModal.tsx` (36px icon box,
 * title + metadata stack, and right-aligned action pill).
 */
export function MaterialiFinderSkeleton({
  count = 5,
  className,
}: MaterialiFinderSkeletonProps) {
  const rows = Array.from({ length: Math.max(1, count) });
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento elementi"
      className={clsx('space-y-2 animate-fade-in select-none', className)}
    >
      {rows.map((_, idx) => (
        <div
          key={idx}
          className="p-3 rounded-xl border border-white/[0.08] bg-white/[0.03] flex items-center justify-between gap-4"
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <SkeletonBlock className="w-9 h-9 flex-shrink-0 border border-white/[0.08]" />
            <div className="min-w-0 flex-1 space-y-2">
              <SkeletonPill
                className={clsx(
                  'h-3',
                  FINDER_ROW_WIDTHS[idx % FINDER_ROW_WIDTHS.length]
                )}
              />
              <SkeletonPill tone="soft" className="h-2.5 w-40 max-w-full" />
            </div>
          </div>
          <SkeletonPill tone="strong" className="h-7 w-28 flex-shrink-0" />
        </div>
      ))}
    </div>
  );
}

export interface SlidePageSkeletonProps {
  className?: string;
}

/**
 * 16:9 / A4 rounded-2xl canvas skeleton matching `max-h-[60vh]` visual
 * slide/PDF preview containers in `MaterialiView` and `LiveRecordingWorkspace`.
 */
export function SlidePageSkeleton({ className }: SlidePageSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento pagina"
      className={clsx(
        'w-full max-h-[60vh] aspect-[16/10] min-h-[260px] rounded-2xl bg-white/[0.025] border border-white/[0.08] p-6 sm:p-8 flex flex-col justify-between gap-5 relative overflow-hidden animate-fade-in select-none',
        className
      )}
    >
      {/* Slide header bar */}
      <div className="flex items-center justify-between gap-4">
        <SkeletonPill tone="strong" className="h-4 w-2/5 max-w-xs" />
        <SkeletonPill tone="soft" className="h-3.5 w-14" />
      </div>

      {/* Main visual content area */}
      <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
        <div className="col-span-7 flex flex-col justify-center space-y-3">
          <SkeletonPill tone="default" className="h-3.5 w-[92%]" />
          <SkeletonPill tone="default" className="h-3.5 w-[84%]" />
          <SkeletonPill tone="soft" className="h-3.5 w-[70%]" />
        </div>
        <SkeletonBlock tone="soft" className="col-span-5 h-full border border-white/[0.06]" />
      </div>

      {/* Slide footer */}
      <div className="flex items-center justify-between gap-4 pt-2 border-t border-white/[0.05]">
        <SkeletonPill tone="soft" className="h-2.5 w-32" />
        <SkeletonPill tone="soft" className="h-2.5 w-10" />
      </div>
    </div>
  );
}

export interface ChatStreamSkeletonProps {
  className?: string;
}

/**
 * 2 asymmetric ChatGPT-style turns: right-aligned user pill followed by
 * left-aligned assistant sources header bar + prose lines.
 */
export function ChatStreamSkeleton({ className }: ChatStreamSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento conversazione"
      className={clsx('space-y-7 py-4 animate-fade-in select-none', className)}
    >
      {/* Turn 1 */}
      <div className="space-y-5">
        <div className="flex justify-end">
          <div className="w-[62%] max-w-sm px-4 py-3 rounded-3xl bg-white/[0.05] border border-white/[0.08] space-y-2">
            <SkeletonPill className="h-3 w-full" />
            <SkeletonPill className="h-3 w-2/3" />
          </div>
        </div>

        <div className="max-w-[88%] space-y-3">
          <SkeletonPill tone="soft" className="h-5 w-36 border border-white/[0.06]" />
          <div className="space-y-2.5">
            <SkeletonPill className="h-3.5 w-full" />
            <SkeletonPill className="h-3.5 w-[95%]" />
            <SkeletonPill className="h-3.5 w-[86%]" />
            <SkeletonPill className="h-3.5 w-[60%]" />
          </div>
        </div>
      </div>

      {/* Turn 2 */}
      <div className="space-y-5">
        <div className="flex justify-end">
          <SkeletonPill tone="default" className="h-10 w-[44%] max-w-xs border border-white/[0.08]" />
        </div>

        <div className="max-w-[88%] space-y-3">
          <SkeletonPill tone="soft" className="h-5 w-28 border border-white/[0.06]" />
          <div className="space-y-2.5">
            <SkeletonPill className="h-3.5 w-[97%]" />
            <SkeletonPill className="h-3.5 w-[90%]" />
            <SkeletonPill className="h-3.5 w-[52%]" />
          </div>
        </div>
      </div>
    </div>
  );
}

export interface TranscriptSkeletonProps {
  className?: string;
}

const TRANSCRIPT_PARAGRAPHS = [
  ['w-full', 'w-[95%]', 'w-[91%]', 'w-[68%]'],
  ['w-[97%]', 'w-full', 'w-[86%]', 'w-[54%]'],
  ['w-[93%]', 'w-[98%]', 'w-[74%]'],
];

/**
 * 3 spacious paragraph blocks matching `AppleLyricsView`'s large
 * `leading-[2.2]` continuous typography inside the dark mesh canvas.
 */
export function TranscriptSkeleton({ className }: TranscriptSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento trascrizione"
      className={clsx('space-y-8 py-2 animate-fade-in select-none', className)}
    >
      {TRANSCRIPT_PARAGRAPHS.map((lines, pIdx) => (
        <div key={pIdx} className="space-y-4">
          {lines.map((widthClass, lIdx) => (
            <SkeletonPill
              key={lIdx}
              tone="default"
              className={clsx('h-4 sm:h-[18px]', widthClass)}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export interface QuizCardSkeletonProps {
  className?: string;
}

const QUIZ_OPTION_WIDTHS = ['w-[72%]', 'w-[84%]', 'w-[64%]', 'w-[78%]'];

/**
 * Centered quiz card skeleton matching `QuizRipassoView`'s active question
 * card (eyebrow + question header lines + 4 rounded-2xl option bars).
 */
export function QuizCardSkeleton({ className }: QuizCardSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento domanda quiz"
      className={clsx(
        'w-full max-w-2xl mx-auto p-6 sm:p-7 rounded-3xl bg-white/[0.025] border border-white/[0.08] space-y-6 animate-fade-in select-none',
        className
      )}
    >
      {/* Question Header */}
      <div className="space-y-3">
        <SkeletonPill tone="soft" className="h-2.5 w-36" />
        <div className="space-y-2">
          <SkeletonPill tone="strong" className="h-5 w-[92%]" />
          <SkeletonPill tone="strong" className="h-5 w-[68%]" />
        </div>
      </div>

      {/* 4 Option Bars */}
      <div className="space-y-2.5">
        {QUIZ_OPTION_WIDTHS.map((widthClass, idx) => (
          <div
            key={idx}
            className="w-full p-4 rounded-2xl liquid-glass-subtle border border-white/[0.08] flex items-center"
          >
            <SkeletonPill className={clsx('h-3.5', widthClass)} />
          </div>
        ))}
      </div>
    </div>
  );
}

export interface QuizWizardSkeletonProps {
  className?: string;
}

const QUIZ_WIZARD_ROW_WIDTHS = ['w-3/5', 'w-2/3', 'w-1/2', 'w-[58%]'];

/**
 * Centered wizard Step 1 skeleton matching `QuizRipassoView` (title + subtitle +
 * glass lecture selection card + bottom rounded-full CTA pill).
 */
export function QuizWizardSkeleton({ className }: QuizWizardSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento simulazione quiz"
      className={clsx(
        'flex-1 flex flex-col items-center justify-center overflow-hidden p-6 sm:p-8 content-canvas animate-fade-in select-none',
        className
      )}
    >
      <div className="w-full max-w-xl mx-auto flex flex-col items-center space-y-6">
        {/* Title & Subtitle */}
        <div className="w-full flex flex-col items-center space-y-2.5">
          <SkeletonPill tone="strong" className="h-6 w-64 max-w-full" />
          <SkeletonPill tone="soft" className="h-3 w-80 max-w-full" />
          <SkeletonPill tone="soft" className="h-3 w-52 max-w-full" />
        </div>

        {/* Glass Card with Lecture Selection Rows */}
        <div className="w-full p-5 rounded-3xl liquid-glass border border-white/[0.10] space-y-4">
          <div className="flex items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <SkeletonPill tone="strong" className="h-3.5 w-28" />
              <SkeletonPill tone="soft" className="h-4 w-24" />
            </div>
            <SkeletonPill tone="soft" className="h-3 w-20" />
          </div>

          <div className="space-y-2">
            {QUIZ_WIZARD_ROW_WIDTHS.map((widthClass, idx) => (
              <div
                key={idx}
                className={clsx(
                  'p-3.5 rounded-2xl border flex items-center justify-between gap-3',
                  idx === 0
                    ? 'bg-white/[0.06] border-white/[0.12]'
                    : 'bg-white/[0.02] border-white/[0.05]'
                )}
              >
                <div className="flex items-center gap-3 min-w-0 flex-1">
                  <SkeletonBlock
                    tone={idx === 0 ? 'strong' : 'soft'}
                    className="w-5 h-5 rounded-lg flex-shrink-0"
                  />
                  <div className="min-w-0 flex-1 space-y-1.5">
                    <SkeletonPill className={clsx('h-3', widthClass)} />
                    <SkeletonPill tone="soft" className="h-2.5 w-24" />
                  </div>
                </div>
                <SkeletonPill tone="soft" className="h-5 w-16 flex-shrink-0" />
              </div>
            ))}
          </div>
        </div>

        {/* Bottom Rounded-Full CTA Pill */}
        <SkeletonPill tone="strong" className="h-10 w-44" />
      </div>
    </div>
  );
}

export interface LiveWorkspaceSkeletonProps {
  className?: string;
}

const LIVE_TRANSCRIPT_LINES = [
  ['w-full', 'w-[94%]', 'w-[88%]', 'w-[62%]'],
  ['w-[96%]', 'w-full', 'w-[82%]', 'w-[48%]'],
];

/**
 * 2-column live workspace skeleton matching `LiveRecordingWorkspace`:
 * left `.content-canvas` with `max-w-2xl mx-auto` transcript lines and right
 * `w-80 xl:w-[356px]` inspector sidebar with segmented control + card skeletons.
 */
export function LiveWorkspaceSkeleton({ className }: LiveWorkspaceSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Caricamento area di registrazione"
      className={clsx(
        'flex-1 flex flex-row w-full h-full overflow-hidden select-none animate-fade-in',
        className
      )}
    >
      {/* Left Column: Live Transcript Canvas */}
      <section className="flex-1 flex flex-col h-full overflow-hidden content-canvas px-12 py-10">
        <div className="w-full max-w-2xl mx-auto space-y-7">
          {LIVE_TRANSCRIPT_LINES.map((lines, pIdx) => (
            <div key={pIdx} className="space-y-3.5">
              {lines.map((widthClass, lIdx) => (
                <SkeletonPill
                  key={lIdx}
                  tone={pIdx === 0 && lIdx === 0 ? 'strong' : 'default'}
                  className={clsx('h-4', widthClass)}
                />
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* Right Column: Inspector Sidebar */}
      <aside className="w-80 xl:w-[356px] flex-shrink-0 flex flex-col h-full border-l border-white/[0.08] bg-[#111520]/85">
        {/* Segmented Control Header */}
        <div className="p-3 border-b border-white/[0.06]">
          <div className="h-7 w-full p-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] flex items-center gap-0.5">
            <SkeletonPill tone="strong" className="h-6 flex-1" />
            <SkeletonPill tone="soft" className="h-6 flex-1" />
          </div>
        </div>

        {/* Inspector Cards */}
        <div className="flex-1 overflow-hidden p-4 space-y-3">
          <SkeletonPill tone="soft" className="h-2.5 w-32" />
          {[0, 1, 2].map((idx) => (
            <div
              key={idx}
              className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-2.5"
            >
              <div className="space-y-2">
                <SkeletonPill tone="strong" className="h-3 w-[92%]" />
                <SkeletonPill className="h-3 w-[68%]" />
              </div>
              <div className="flex items-center gap-2 pt-1">
                <SkeletonPill tone="soft" className="h-7 flex-1" />
                <SkeletonPill tone="soft" className="w-7 h-7 flex-shrink-0" />
              </div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

export interface StorageStatsSkeletonProps {
  className?: string;
}

const STORAGE_BADGE_WIDTHS = ['w-24', 'w-20', 'w-28', 'w-24', 'w-20'];

/**
 * Compact inline skeleton for `SettingsModal`'s Apple-style Storage Overview
 * card (total MB hero + segmented horizontal bar + 5 category size badges).
 */
export function StorageStatsSkeleton({ className }: StorageStatsSkeletonProps) {
  return (
    <div
      aria-busy="true"
      aria-label="Calcolo spazio su disco"
      className={clsx('space-y-4 animate-fade-in select-none', className)}
    >
      {/* Top Row: Total MB + Refresh Action Pill */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="space-y-2">
          <SkeletonPill tone="soft" className="h-2.5 w-24" />
          <div className="flex items-baseline gap-2">
            <SkeletonBlock tone="strong" className="h-8 w-28" />
            <SkeletonPill tone="soft" className="h-4 w-10" />
          </div>
        </div>
        <SkeletonBlock tone="default" className="h-9 w-36 self-start sm:self-center" />
      </div>

      {/* Segmented Bar */}
      <div className="space-y-3">
        <div className="w-full h-3 rounded-full bg-white/[0.05] p-0.5 flex gap-1 overflow-hidden border border-white/[0.10]">
          <SkeletonPill tone="strong" className="h-full w-[42%]" />
          <SkeletonPill tone="default" className="h-full w-[24%]" />
          <SkeletonPill tone="default" className="h-full w-[16%]" />
          <SkeletonPill tone="soft" className="h-full w-[10%]" />
        </div>

        {/* 5 Category Size Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2.5 pt-2 border-t border-white/[0.06]">
          {STORAGE_BADGE_WIDTHS.map((widthClass, idx) => (
            <div key={idx} className="flex items-center gap-1.5">
              <SkeletonPill tone="strong" className="w-2 h-2 flex-shrink-0" />
              <SkeletonPill className={clsx('h-2.5', widthClass)} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
