// lib/useUpdater.ts — Aggiornamenti automatici (Tauri updater), no-op nel browser
import { useCallback, useEffect, useState } from 'react';
import { isTauri } from './tauri';

export type UpdaterStatus = 'idle' | 'checking' | 'uptodate' | 'available' | 'downloading' | 'installing' | 'error';

interface UpdateHandle {
  version: string;
  downloadAndInstall: (cb?: (e: any) => void) => Promise<void>;
}

// Stato condiviso tra impostazioni e boot
interface State {
  status: UpdaterStatus;
  version: string | null;
  progress: number; // 0..100
  error: string | null;
  currentVersion: string;
}

let state: State = { status: 'idle', version: null, progress: 0, error: null, currentVersion: '0.0.17' };
let handle: UpdateHandle | null = null;
let versionFetched = false;
const listeners = new Set<() => void>();

function setState(patch: Partial<State>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

async function fetchCurrentVersion(): Promise<void> {
  if (versionFetched) return;
  versionFetched = true;
  if (!isTauri()) return;
  try {
    const { getVersion } = await import('@tauri-apps/api/app');
    const v = await getVersion();
    if (v) setState({ currentVersion: v });
  } catch (e) {
    console.warn('[Updater] Errore lettura versione app:', e);
  }
}

async function runCheck(silent: boolean): Promise<void> {
  if (!isTauri()) return;
  if (state.status === 'checking' || state.status === 'downloading' || state.status === 'installing') return;
  setState({ status: 'checking', error: null });
  try {
    const { check } = await import('@tauri-apps/plugin-updater');
    const update = await check();
    if (update) {
      try {
        (handle as any)?.close?.();
      } catch {}
      handle = update as unknown as UpdateHandle;
      setState({ status: 'available', version: update.version });
    } else {
      try {
        (handle as any)?.close?.();
      } catch {}
      handle = null;
      setState({ status: 'uptodate', version: null });
    }
  } catch (e: any) {
    console.warn('[Updater] Controllo aggiornamenti fallito:', e);
    const rawMsg = String(e?.message || e || 'Errore sconosciuto');
    let friendlyMsg = rawMsg;
    if (rawMsg.includes('None of the fallback platforms') || rawMsg.includes('fallback platforms')) {
      friendlyMsg = 'Il pacchetto per questa piattaforma è in fase di finalizzazione sui server. Riprova tra un paio di minuti.';
    }
    setState({
      status: silent ? 'idle' : 'error',
      error: silent ? null : friendlyMsg,
    });
  }
}

async function stopBackendBeforeUpdate(): Promise<void> {
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('stop_backend_for_update');
  } catch (e) {
    console.warn('[Updater] stop_backend_for_update warning:', e);
  }
}

async function runInstall(): Promise<void> {
  if (!handle) return;
  setState({ status: 'downloading', progress: 0, error: null });
  let total = 0;
  let downloaded = 0;
  try {
    await stopBackendBeforeUpdate();
    await handle.downloadAndInstall((event: any) => {
      if (event.event === 'Started') {
        total = event.data?.contentLength ?? 0;
      } else if (event.event === 'Progress') {
        downloaded += event.data?.chunkLength ?? 0;
        if (total > 0) setState({ progress: Math.min(100, Math.round((downloaded / total) * 100)) });
      } else if (event.event === 'Finished') {
        setState({ progress: 100, status: 'installing' });
      }
    });
    setState({ status: 'installing' });
    const { relaunch } = await import('@tauri-apps/plugin-process');
    await relaunch();
  } catch (e: any) {
    console.error('[Updater] Installazione fallita:', e);
    setState({ status: 'error', error: String(e?.message || e || "Errore durante l'aggiornamento") });
  }
}

export type BootUpdateEvent = {
  status: 'idle' | 'checking' | 'available' | 'downloading' | 'installing' | 'relaunching' | 'uptodate' | 'error';
  version?: string;
  progress?: number;
  error?: string;
};

export type BootUpdateProgressCallback = (event: BootUpdateEvent) => void;

/**
 * Controllo e installazione automatica dell'aggiornamento durante la schermata di caricamento iniziale (boot gate).
 * Se è disponibile una nuova release, la scarica mostrando la progressione e riavvia l'app in autonomia.
 */
export async function checkAndAutoInstall(
  onProgress?: BootUpdateProgressCallback
): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    onProgress?.({ status: 'checking' });
    const { check } = await import('@tauri-apps/plugin-updater');

    // Timeout di 3.5s per evitare blocchi prolungati all'avvio in caso di connessione assente o lenta
    const checkPromise = check();
    const timeoutPromise = new Promise<null>((resolve) => setTimeout(() => resolve(null), 3500));
    const update = await Promise.race([checkPromise, timeoutPromise]);

    if (!update) {
      onProgress?.({ status: 'uptodate' });
      return false;
    }

    const version = update.version;
    onProgress?.({ status: 'available', version, progress: 0 });
    setState({ status: 'downloading', version, progress: 0 });
    await stopBackendBeforeUpdate();

    let total = 0;
    let downloaded = 0;

    await update.downloadAndInstall((event: any) => {
      if (event.event === 'Started') {
        total = event.data?.contentLength ?? 0;
        onProgress?.({ status: 'downloading', version, progress: 0 });
      } else if (event.event === 'Progress') {
        downloaded += event.data?.chunkLength ?? 0;
        const pct = total > 0 ? Math.min(100, Math.round((downloaded / total) * 100)) : 0;
        setState({ progress: pct });
        onProgress?.({ status: 'downloading', version, progress: pct });
      } else if (event.event === 'Finished') {
        setState({ progress: 100, status: 'installing' });
        onProgress?.({ status: 'installing', version, progress: 100 });
      }
    });

    setState({ status: 'installing' });
    onProgress?.({ status: 'relaunching', version, progress: 100 });
    const { relaunch } = await import('@tauri-apps/plugin-process');
    await relaunch();
    return true;
  } catch (e: any) {
    console.warn('[Updater] checkAndAutoInstall errore (probabile offline o rete assente):', e);
    setState({ status: 'idle', error: null });
    onProgress?.({ status: 'idle' });
    return false;
  }
}

export function useUpdater() {
  const [, force] = useState(0);
  useEffect(() => {
    void fetchCurrentVersion();
    const l = () => force((n) => n + 1);
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  }, []);

  const checkNow = useCallback(() => runCheck(false), []);
  const install = useCallback(() => runInstall(), []);
  return { ...state, supported: isTauri(), checkNow, install };
}
