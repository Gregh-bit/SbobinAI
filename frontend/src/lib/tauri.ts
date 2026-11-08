// lib/tauri.ts — Helper sicuri per le API native di Tauri v2 (con fallback browser)
import { open } from '@tauri-apps/plugin-dialog';
import { isWindows } from './platform';

/** Verifica se l'app è in esecuzione all'interno di Tauri desktop */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

export interface TauriBackendStatus {
  status: 'starting' | 'initializing_venv' | 'installing_deps' | 'ready' | 'error';
  message: string;
  port_ready: boolean;
  error?: string | null;
}

/** Recupera lo stato in tempo reale del bootstrap del motore locale da Tauri */
export async function getBackendStatus(): Promise<TauriBackendStatus | null> {
  if (isTauri()) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      return await invoke<TauriBackendStatus>('get_backend_status');
    } catch {
      return null;
    }
  }
  return null;
}

/** Riavvia l'applicazione nativa in modo sicuro */
export async function relaunchApp(): Promise<void> {
  if (isTauri()) {
    try {
      const { relaunch } = await import('@tauri-apps/plugin-process');
      await relaunch();
    } catch {
      window.location.reload();
    }
  } else {
    window.location.reload();
  }
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

let webWakeLockSentinel: any = null;

/**
 * Attiva l'OS Wake Lock (Anti-sleep) nativo del sistema operativo:
 * - macOS: IOKit IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep)
 * - Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
 * + Screen Wake Lock API standard nel browser/webview come livello complementare
 */
export async function acquireWakeLock(reason: string = 'SbobinAI - Registrazione lezione in corso'): Promise<boolean> {
  let acquiredNative = false;
  if (isTauri()) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('acquire_wake_lock', { reason });
      acquiredNative = true;
    } catch (e) {
      console.warn('[WakeLock] Errore chiamata nativa acquire_wake_lock:', e);
    }
  }

  // Livello di protezione opzionale per lo schermo (se supportato)
  if (typeof navigator !== 'undefined' && 'wakeLock' in navigator) {
    try {
      webWakeLockSentinel = await (navigator as any).wakeLock.request('screen');
      webWakeLockSentinel.addEventListener('release', () => {
        webWakeLockSentinel = null;
      });
    } catch {
      // Ignora silenziosamente se la policy dello schermo o il risparmio energetico lo rifiutano
    }
  }

  return acquiredNative || webWakeLockSentinel !== null;
}

/**
 * Rilascia l'OS Wake Lock nativo e lo Screen Wake Lock, ripristinando il normale risparmio energetico del sistema.
 */
export async function releaseWakeLock(): Promise<boolean> {
  let releasedNative = false;
  if (isTauri()) {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('release_wake_lock');
      releasedNative = true;
    } catch (e) {
      console.warn('[WakeLock] Errore chiamata nativa release_wake_lock:', e);
    }
  }

  if (webWakeLockSentinel) {
    try {
      await webWakeLockSentinel.release();
    } catch {
      // ignore
    }
    webWakeLockSentinel = null;
  }

  return releasedNative;
}


