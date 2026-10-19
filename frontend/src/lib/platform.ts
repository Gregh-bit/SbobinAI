// lib/platform.ts — Rilevamento piattaforma per etichette scorciatoie e badge hardware

const platformString: string =
  typeof navigator === 'undefined'
    ? ''
    : (
        (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ||
        navigator.platform ||
        navigator.userAgent ||
        ''
      );

const uaString: string = typeof navigator === 'undefined' ? '' : navigator.userAgent || '';

export const isMac: boolean = /mac|iphone|ipad|ipod/i.test(platformString) || /mac os x/i.test(uaString);

export const isWindows: boolean = !isMac && (/win/i.test(platformString) || /windows/i.test(uaString));

/** Badge hardware/accelerazione mostrato in UI (null su altre piattaforme). */
export const osBadge: string | null = isMac
  ? 'Ottimizzato per macOS (Metal / MLX)'
  : isWindows
    ? 'Ottimizzato per Windows (Native Acceleration)'
    : null;

/** Etichetta del tasto modificatore principale: "⌘" su macOS, "Ctrl+" altrove. */
export const modKeyLabel: string = isMac ? '⌘' : 'Ctrl+';

/** Formatta una scorciatoia, es. shortcutLabel('B') → "⌘B" / "Ctrl+B"; con shift → "⌘⇧B" / "Ctrl+Shift+B". */
export function shortcutLabel(key: string, shift = false): string {
  const s = shift ? (isMac ? '⇧' : 'Shift+') : '';
  return `${modKeyLabel}${s}${key}`;
}
