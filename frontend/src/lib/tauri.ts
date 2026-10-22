// lib/tauri.ts — Helper sicuri per le API native di Tauri v2 (con fallback browser)
import { open } from '@tauri-apps/plugin-dialog';

/** Verifica se l'app è in esecuzione all'interno di Tauri desktop */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/** Apre il selettore di cartelle nativo di macOS */
export async function pickFolder(defaultPath?: string): Promise<string | null> {
  if (isTauri()) {
    try {
      const selected = await open({
        directory: true,
        multiple: false,
        title: 'Seleziona cartella SbobinAI per audio e database',
        defaultPath: defaultPath || undefined,
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

