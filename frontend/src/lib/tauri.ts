// lib/tauri.ts — Helper sicuri per le API native di Tauri v2 (con fallback browser)
import { open } from '@tauri-apps/plugin-dialog';
import { isWindows } from './platform';

/** Verifica se l'app è in esecuzione all'interno di Tauri desktop */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/** Apre il selettore di cartelle nativo del sistema operativo */
export async function pickFolder(defaultPath?: string): Promise<string | null> {
  if (isTauri()) {
    try {
      // Su Windows, non passare defaultPath se inizia con '~' (causa fallimento del picker nativo)
      let sanitizedDefault = defaultPath;
      if (sanitizedDefault && isWindows && sanitizedDefault.startsWith('~')) {
        sanitizedDefault = undefined;
      }

      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Seleziona cartella SbobinAI per audio e database',
        defaultPath: sanitizedDefault || undefined,
      });
      if (typeof selected === 'string') {
        return selected;
      }
      return null;
    } catch (e) {
      console.error('[Tauri] Errore pickFolder nativo:', e);
      return null;
    }
  }
  return null;
}

