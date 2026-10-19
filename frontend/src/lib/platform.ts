// lib/platform.ts — Rilevamento piattaforma per le etichette delle scorciatoie da tastiera

export const isMac: boolean =
  typeof navigator !== 'undefined' &&
  /mac|iphone|ipad|ipod/i.test(navigator.platform || navigator.userAgent || '');

/** Etichetta del tasto modificatore principale: "⌘" su macOS, "Ctrl+" altrove. */
export const modKeyLabel: string = isMac ? '⌘' : 'Ctrl+';

/** Formatta una scorciatoia, es. shortcutLabel('B') → "⌘B" oppure "Ctrl+B". */
export function shortcutLabel(key: string): string {
  return `${modKeyLabel}${key}`;
}
