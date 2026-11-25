// components/SegmentedControl.tsx — Apple-style Segmented Control con Framer Motion sliding pill
import React, { useRef, useState, useLayoutEffect } from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

export interface SegmentedControlOption<T extends string> {
  id: T;
  label: string;
  icon?: React.ElementType;
  badge?: React.ReactNode;
}

interface Props<T extends string> {
  options: SegmentedControlOption<T>[];
  value: T;
  onChange: (value: T) => void;
  layoutId?: string;
  size?: 'sm' | 'md';
  className?: string;
  fullWidth?: boolean;
}

export default function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  className,
  fullWidth = false,
}: Props<T>) {
  const isSm = size === 'sm';
  const tabRefs = useRef<Map<T, HTMLButtonElement>>(new Map());
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);

  useLayoutEffect(() => {
    const activeEl = tabRefs.current.get(value);
    if (activeEl) {
      setIndicator({
        left: activeEl.offsetLeft,
        width: activeEl.offsetWidth,
      });
    }
  }, [value, options]);

  return (
    <div
      role="tablist"
      className={clsx(
        'relative flex items-center p-0.5 rounded-full bg-[#121622]/85 border border-white/[0.12] shadow-liquid-sm select-none',
        isSm ? 'h-7' : 'h-8',
        fullWidth ? 'w-full' : '',
        className
      )}
    >
      {/* Indicatore a pillola scorrevole: SOLO scorrimento orizzontale X relativo al contenitore, zero salti sull'asse Y */}
      {indicator && (
        <motion.div
          className="absolute inset-y-0.5 rounded-full bg-white/[0.14] border border-white/[0.22] shadow-liquid-pill pointer-events-none"
          initial={false}
          animate={{
            x: indicator.left,
            width: indicator.width,
          }}
          transition={{
            type: 'spring',
            stiffness: 480,
            damping: 34,
            mass: 0.8,
          }}
          style={{ left: 0 }}
        />
      )}

      {options.map((option) => {
        const Icon = option.icon;
        const isActive = value === option.id;

        return (
          <button
            key={option.id}
            ref={(el) => {
              if (el) tabRefs.current.set(option.id, el);
              else tabRefs.current.delete(option.id);
            }}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(option.id)}
            className={clsx(
              'relative flex items-center justify-center gap-1.5 rounded-full font-medium transition-colors duration-150 cursor-pointer z-10 leading-normal',
              fullWidth ? 'flex-1' : '',
              isSm ? 'h-6 px-2.5 text-[11px]' : 'h-7 px-3 text-xs',
              isActive
                ? 'text-white'
                : 'text-white/55 hover:text-white/85'
            )}
          >
            {Icon && (
              <Icon
                size={isSm ? 11 : 12}
                strokeWidth={1.75}
                className={clsx(
                  'flex-shrink-0 relative z-10 transition-colors',
                  isActive ? 'text-white' : 'text-white/50'
                )}
              />
            )}

            <span className="truncate relative z-10 leading-normal flex items-center justify-center">{option.label}</span>

            {option.badge && (
              <span className="relative z-10 flex-shrink-0 flex items-center justify-center">{option.badge}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
