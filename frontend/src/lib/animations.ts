// frontend/src/lib/animations.ts — Standard per tutte le animazioni Liquid Glass & Popover Menu dell'app
import type { Transition, Variants } from 'framer-motion';

/**
 * Transizione Spring standard Liquid Glass (identica a Esporta e Cestino nel Quaderno)
 */
export const liquidSpringTransition: Transition = {
  type: 'spring',
  damping: 22,
  stiffness: 340,
  mass: 0.75,
};

/**
 * Transizione di layout fluida per ridimensionamento morbido quando il contenuto cambia (es. sottomenu/subviews)
 */
export const liquidLayoutTransition: Transition = {
  type: 'spring',
  damping: 26,
  stiffness: 340,
  mass: 0.8,
};

/**
 * Varianti standard per comparsa/scomparsa a bolla (Liquid Glass Popover)
 */
export const liquidMenuVariants: Variants = {
  initial: {
    opacity: 0,
    scale: 0.15,
    y: -16,
    filter: 'blur(10px)',
  },
  animate: {
    opacity: 1,
    scale: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: liquidSpringTransition,
  },
  exit: {
    opacity: 0,
    scale: 0.25,
    y: -12,
    filter: 'blur(8px)',
    transition: {
      duration: 0.16,
      ease: 'easeIn',
    },
  },
};

/**
 * Varianti per morphing 3D icona aperta (Plus/Share/Trash -> X)
 */
export const liquidIconMorphOpen: Variants = {
  initial: { rotate: 90, opacity: 0, scale: 0.6 },
  animate: { rotate: 0, opacity: 1, scale: 1, transition: { duration: 0.16 } },
  exit: { rotate: -90, opacity: 0, scale: 0.6, transition: { duration: 0.16 } },
};

/**
 * Varianti per morphing 3D icona chiusa (X -> Plus/Share/Trash)
 */
export const liquidIconMorphClose: Variants = {
  initial: { rotate: -90, opacity: 0, scale: 0.6 },
  animate: { rotate: 0, opacity: 1, scale: 1, transition: { duration: 0.16 } },
  exit: { rotate: 90, opacity: 0, scale: 0.6, transition: { duration: 0.16 } },
};

/**
 * Classi CSS standard per i contenitori Liquid Glass Popover a bolla
 */
export const liquidPopoverContainerClass =
  'rounded-3xl bg-[#13141c]/95 backdrop-blur-3xl border border-white/[0.20] shadow-[0_24px_60px_rgba(0,0,0,0.85),0_0_0_1px_rgba(255,255,255,0.08)] p-2 z-50';
