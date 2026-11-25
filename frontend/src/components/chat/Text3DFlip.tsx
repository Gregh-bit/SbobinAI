import { memo, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';

export interface Text3DFlipProps {
  /** Testo da renderizzare con animazione 3D flip */
  text: string;
  className?: string;
  wordClassName?: string;
  /** Modalità di roll 3D: 'line' (rotazione 3D dell'intera frase, ideale per effetto shimmer continuo) o 'word' (rotazione parola per parola) */
  mode?: 'line' | 'word';
  /** Direzione di rotazione 3D: 'top' (roll dal basso verso l'alto come ChatGPT) o 'bottom' */
  rotateDirection?: 'top' | 'bottom';
  /** Ritardo a cascata tra parole successive (usato in mode='word') */
  staggerDuration?: number;
  /** Durata o spring di transizione */
  transition?: {
    type?: 'spring' | 'tween';
    damping?: number;
    stiffness?: number;
    mass?: number;
  };
}

/**
 * Text3DFlip — Roll 3D verticale fluido stile ChatGPT 2026 sul cambio di stato,
 * con accelerazione hardware GPU via CSS 3D (perspective + preserve-3d).
 * In modalità 'line' (predefinita), applica l'effetto luce (chat-shimmer-text)
 * in modo continuo e uniforme sull'intera frase, senza frammentarlo sulle singole parole.
 */
export const Text3DFlip = memo(function Text3DFlip({
  text,
  className,
  wordClassName,
  mode = 'line',
  rotateDirection = 'top',
  staggerDuration = 0.025,
  transition = { type: 'spring', damping: 24, stiffness: 190, mass: 0.8 },
}: Text3DFlipProps) {
  const words = useMemo(() => text.split(' '), [text]);
  const isTop = rotateDirection === 'top';

  return (
    <div
      className={clsx('relative inline-flex items-center min-h-[20px] overflow-hidden', className)}
      style={{ perspective: 600 }}
    >
      <AnimatePresence mode="popLayout" initial={false}>
        {mode === 'line' ? (
          <motion.span
            key={text}
            initial={{
              opacity: 0,
              y: isTop ? 14 : -14,
              rotateX: isTop ? 80 : -80,
            }}
            animate={{
              opacity: 1,
              y: 0,
              rotateX: 0,
              transition: {
                type: 'spring',
                damping: 22,
                stiffness: 180,
                mass: 0.6,
              },
            }}
            exit={{
              opacity: 0,
              y: isTop ? -14 : 14,
              rotateX: isTop ? -80 : 80,
              transition: {
                duration: 0.18,
                ease: 'easeIn',
              },
            }}
            style={{
              display: 'inline-block',
              transformOrigin: isTop ? '50% 100%' : '50% 0%',
              transformStyle: 'preserve-3d',
              WebkitFontSmoothing: 'antialiased',
              backfaceVisibility: 'hidden',
            }}
            className={clsx('inline-block whitespace-nowrap', wordClassName)}
          >
            {text}
          </motion.span>
        ) : (
          <motion.div
            key={text}
            className="inline-flex flex-wrap items-center gap-x-1"
            style={{ transformStyle: 'preserve-3d' }}
          >
            {words.map((word, idx) => (
              <motion.span
                key={`${word}-${idx}`}
                initial={{
                  opacity: 0,
                  y: isTop ? 10 : -10,
                  rotateX: isTop ? -85 : 85,
                }}
                animate={{
                  opacity: 1,
                  y: 0,
                  rotateX: 0,
                  transition,
                }}
                exit={{
                  opacity: 0,
                  y: isTop ? -10 : 10,
                  rotateX: isTop ? 85 : -85,
                  transition: {
                    duration: 0.14,
                    delay: idx * (staggerDuration * 0.5),
                    ease: 'easeIn',
                  },
                }}
                style={{
                  display: 'inline-block',
                  transformOrigin: isTop ? '50% 100%' : '50% 0%',
                  backfaceVisibility: 'hidden',
                  WebkitFontSmoothing: 'antialiased',
                }}
                className={clsx('inline-block whitespace-nowrap', wordClassName)}
              >
                {word}
              </motion.span>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

export default Text3DFlip;
