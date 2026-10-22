// lib/useUpdater.ts — Aggiornamenti automatici (Tauri updater), no-op nel browser
import { useCallback, useEffect, useState } from 'react';
import { isTauri } from './tauri';

export type UpdaterStatus = 'idle' | 'checking' | 'uptodate' | 'available' | 'downloading' | 'installing' | 'error';

interface UpdateHandle {
  version: string;
  downloadAndInstall: (cb?: (e: any) => void) => Promise<void>;
}

// Stato condiviso tra banner e impostazioni
interface State {
  status: UpdaterStatus;
  version: string | null;
  progress: number; // 0..100
  error: string | null;
  currentVersion: string;
}

let state: State = { status: 'idle', version: null, progress: 0, error: null, currentVersion: '0.0.3' };
let handle: UpdateHandle | null = null;
let startupDone = false;
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
      handle = update as unknown as UpdateHandle;
      setState({ status: 'available', version: update.version });
    } else {
      handle = null;
      setState({ status: 'uptodate', version: null });
    }
  } catch (e: any) {
    console.warn('[Updater] Controllo aggiornamenti fallito:', e);
    setState({
      status: silent ? 'idle' : 'error',
      error: silent ? null : String(e?.message || e || 'Errore sconosciuto'),
    });
  }
}

async function runInstall(): Promise<void> {
  if (!handle) return;
  setState({ status: 'downloading', progress: 0, error: null });
  let total = 0;
  let downloaded = 0;
  try {
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

export function useUpdater(opts: { autoCheck?: boolean } = {}) {
  const [, force] = useState(0);
  useEffect(() => {
    void fetchCurrentVersion();
    const l = () => force((n) => n + 1);
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  }, []);

  useEffect(() => {
    if (!opts.autoCheck || startupDone || !isTauri()) return;
    startupDone = true;
    const t = setTimeout(() => {
      void runCheck(true);
    }, 3000);
    return () => clearTimeout(t);
  }, [opts.autoCheck]);

  const checkNow = useCallback(() => runCheck(false), []);
  const install = useCallback(() => runInstall(), []);
  return { ...state, supported: isTauri(), checkNow, install };
}
