// lib/SettingsContext.tsx — Context React per la gestione persistente delle impostazioni di SbobinAI
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useRef,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { api } from './api';
import type { ModelsStatusResponse } from '../types';

export interface AppSettings {
  setupComplete: boolean;
  modelsDownloaded: boolean;
  llmDownloaded?: boolean;
  llmSupportsThinking?: boolean;
  autoUpdate: boolean;
  storagePath: string;
  whisperModel: string;
  llmModel: string;
  embeddingModel: string;
  ollamaModel: string; // Retrocompatibilità UI
  vadSensitivity: number; // 1 (permissivo) a 3 (molto reattivo/aggressivo)
  audioFormat?: 'm4a';
}

function inferModelSupportsThinking(modelId?: string): boolean {
  const low = (modelId || '').trim().toLowerCase();
  if (!low) return true;
  return ['qwen3', 'gemma-4', 'gemma4', 'deepseek-r1', 'thinking', 'qwq'].some((tok) => low.includes(tok));
}

interface SettingsContextType {
  settings: AppSettings;
  modelsStatus: ModelsStatusResponse | null;
  setModelsStatus: Dispatch<SetStateAction<ModelsStatusResponse | null>>;
  updateSettings: (newSettings: Partial<AppSettings>) => Promise<void>;
  completeSetup: (customSettings?: Partial<AppSettings>) => Promise<void>;
  resetSetup: () => void;
  syncWithBackend: () => Promise<void>;
}

const DEFAULT_SETTINGS: AppSettings = {
  setupComplete: false,
  modelsDownloaded: false,
  llmDownloaded: false,
  llmSupportsThinking: true,
  autoUpdate: true,
  storagePath: '~/Documents/SbobinAI',
  whisperModel: 'large-v3-turbo-q4',
  llmModel: 'mlx-community/Qwen3.5-4B-MLX-4bit',
  embeddingModel: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
  ollamaModel: 'mlx-community/Qwen3.5-4B-MLX-4bit',
  vadSensitivity: 2,
  audioFormat: 'm4a',
};

const STORAGE_KEY = 'sbobinai_settings_v1';
const LEGACY_STORAGE_KEY = 'unicopilot_settings_v1';

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.storagePath === '~/Documents/UniCoPilot' || parsed.storagePath?.endsWith('/Documents/UniCoPilot')) {
          parsed.storagePath = '~/Documents/SbobinAI';
        }
        return {
          ...DEFAULT_SETTINGS,
          ...parsed,
          autoUpdate: parsed.autoUpdate ?? true,
        };
      }
    } catch (e) {
      console.warn('Errore lettura impostazioni da localStorage:', e);
    }
    return DEFAULT_SETTINGS;
  });

  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const llmThinkingMapRef = useRef<Record<string, boolean>>({});
  const inFlightSyncRef = useRef<Promise<void> | null>(null);

  // Mantieni sempre sincronizzati modelsDownloaded, llmDownloaded e llmSupportsThinking
  // ogni volta che modelsStatus viene aggiornato (via syncWithBackend o setModelsStatus esterno).
  useEffect(() => {
    if (!modelsStatus || typeof modelsStatus !== 'object') return;

    const hasRealWhisper = Boolean(
      modelsStatus.whisper?.some((m: any) => m.id !== 'mock_mode' && m.downloaded)
    );
    const hasRealLLM = Boolean(
      modelsStatus.llm?.some((m: any) => m.downloaded)
    );

    if (Array.isArray(modelsStatus.llm)) {
      const nextMap: Record<string, boolean> = {};
      for (const m of modelsStatus.llm) {
        const sup =
          m.supports_thinking !== undefined
            ? Boolean(m.supports_thinking)
            : inferModelSupportsThinking(m.id);
        if (m.id) nextMap[m.id] = sup;
        if (m.repo_id) nextMap[m.repo_id] = sup;
      }
      llmThinkingMapRef.current = nextMap;
    }

    setSettings((prev) => {
      let activeLlm = prev.llmModel;
      const isCurrentActiveDownloaded = modelsStatus.llm?.some((m: any) => (m.id === activeLlm || m.repo_id === activeLlm) && m.downloaded);
      const firstDownloadedLlm = modelsStatus.llm?.find((m: any) => m.downloaded);
      if (!isCurrentActiveDownloaded && firstDownloadedLlm) {
        activeLlm = firstDownloadedLlm.id;
      }

      const llmSupportsThinking =
        activeLlm in llmThinkingMapRef.current
          ? llmThinkingMapRef.current[activeLlm]
          : inferModelSupportsThinking(activeLlm);

      if (
        prev.modelsDownloaded === hasRealWhisper &&
        prev.llmDownloaded === hasRealLLM &&
        prev.llmSupportsThinking === llmSupportsThinking &&
        prev.llmModel === activeLlm
      ) {
        return prev;
      }

      return {
        ...prev,
        llmModel: activeLlm,
        modelsDownloaded: hasRealWhisper,
        llmDownloaded: hasRealLLM,
        llmSupportsThinking,
      };
    });
  }, [modelsStatus]);

  // Sincronizza le impostazioni reali e lo stato effettivo dei modelli dal backend.
  // Deduplica chiamate concorrenti (es. BackendLoadingGate + React StrictMode)
  // su un'unica promise condivisa per evitare raffiche duplicate all'avvio.
  // Nota: è BackendLoadingGate a invocare syncWithBackend() appena il backend risponde
  // all'healthcheck, evitando richieste premature prima che la porta sia pronta.
  const syncWithBackend = useCallback(async () => {
    if (inFlightSyncRef.current) {
      return inFlightSyncRef.current;
    }

    const syncPromise = (async () => {
      try {
        const [settingsData, modelsData] = await Promise.all([
          api.settings.get().catch(() => null),
          api.models.status().catch(() => null),
        ]);
        const patch: Partial<AppSettings> = {};
        if (settingsData && typeof settingsData === 'object') {
          if (settingsData.storage_path && typeof settingsData.storage_path === 'string') {
            patch.storagePath = settingsData.storage_path;
          }
          if (settingsData.whisper_model && typeof settingsData.whisper_model === 'string') {
            patch.whisperModel = settingsData.whisper_model;
          }
          if (settingsData.llm_model && typeof settingsData.llm_model === 'string') {
            patch.llmModel = settingsData.llm_model;
            patch.ollamaModel = settingsData.llm_model;
          } else if (settingsData.ollama_model && typeof settingsData.ollama_model === 'string') {
            patch.llmModel = settingsData.ollama_model;
            patch.ollamaModel = settingsData.ollama_model;
          }
          if (settingsData.embedding_model && typeof settingsData.embedding_model === 'string') {
            patch.embeddingModel = settingsData.embedding_model;
          }
          if (typeof settingsData.vad_sensitivity === 'number') {
            patch.vadSensitivity = settingsData.vad_sensitivity;
          }
          if (typeof settingsData.models_downloaded === 'boolean') {
            patch.modelsDownloaded = settingsData.models_downloaded;
          }
          if (settingsData.setup_complete === true) {
            patch.setupComplete = true;
          }
          if (settingsData.audio_format) {
            patch.audioFormat = 'm4a';
          }
        }

        const hasRealWhisper = Boolean(
          modelsData &&
          typeof modelsData === 'object' &&
          modelsData.whisper?.some((m: any) => m.id !== 'mock_mode' && m.downloaded)
        );

        const hasRealLLM = Boolean(
          modelsData &&
          typeof modelsData === 'object' &&
          modelsData.llm?.some((m: any) => m.downloaded)
        );

        if (modelsData && typeof modelsData === 'object') {
          setModelsStatus(modelsData);
          patch.modelsDownloaded = hasRealWhisper;
          patch.llmDownloaded = hasRealLLM;
          if (Array.isArray(modelsData.llm)) {
            const nextMap: Record<string, boolean> = {};
            for (const m of modelsData.llm) {
              const sup = m.supports_thinking !== undefined ? Boolean(m.supports_thinking) : inferModelSupportsThinking(m.id);
              if (m.id) nextMap[m.id] = sup;
              if (m.repo_id) nextMap[m.repo_id] = sup;
            }
            llmThinkingMapRef.current = nextMap;
          }
          if (hasRealWhisper) {
            patch.setupComplete = true;
            if (!patch.whisperModel || patch.whisperModel === 'mock_mode' || patch.whisperModel === 'dual-tier-whisper') {
              patch.whisperModel = 'large-v3-turbo-q4';
            }
          }
        }

        setSettings((prev) => {
          // Se il backend indica setup non completato e non c'è nessun Whisper su disco
          // (es. reinstallazione dopo residui WebView/localStorage), forza il Wizard.
          const backendExplicitlyIncomplete =
            settingsData !== null &&
            typeof settingsData === 'object' &&
            settingsData.setup_complete !== true &&
            modelsData !== null &&
            !hasRealWhisper;

          const nextSetupComplete = backendExplicitlyIncomplete
            ? false
            : (patch.setupComplete ?? prev.setupComplete) || prev.setupComplete;

          let activeLlm = patch.llmModel || prev.llmModel;
          const isLlmDownloaded = modelsData?.llm?.some((m: any) => (m.id === activeLlm || m.repo_id === activeLlm) && m.downloaded);
          const firstDownloaded = modelsData?.llm?.find((m: any) => m.downloaded);
          if (!isLlmDownloaded && firstDownloaded) {
            activeLlm = firstDownloaded.id;
            patch.llmModel = firstDownloaded.id;
          }
          const llmSupportsThinking =
            activeLlm in llmThinkingMapRef.current
              ? llmThinkingMapRef.current[activeLlm]
              : inferModelSupportsThinking(activeLlm);

          const merged = {
            ...prev,
            ...patch,
            llmSupportsThinking,
            setupComplete: nextSetupComplete,
          };
          // Sincronizza al backend solo se il modello Whisper reale esiste davvero su disco
          if (nextSetupComplete && settingsData && settingsData.setup_complete !== true && hasRealWhisper) {
            void api.settings.update({ setup_complete: true, models_downloaded: true }).catch(() => null);
          }
          return merged;
        });
      } catch (err) {
        console.warn('[SettingsContext] Impossibile contattare backend per sincronizzazione:', err);
      } finally {
        inFlightSyncRef.current = null;
      }
    })();

    inFlightSyncRef.current = syncPromise;
    return syncPromise;
  }, []);

  // Salva ogni modifica nel localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (e) {
      console.error('Errore salvataggio impostazioni in localStorage:', e);
    }
  }, [settings]);

  const updateSettings = async (partial: Partial<AppSettings>) => {
    // Non mutare l'oggetto passato dal chiamante: può essere riutilizzato dal form.
    partial = { ...partial };
    // Il percorso di archiviazione si cambia SOLO tramite POST /settings/storage (migrazione dati),
    // mai dentro il PATCH /settings. In caso di errore viene propagato al chiamante.
    if (partial.storagePath) {
      try {
        const res = await api.settings.setStorage(partial.storagePath);
        if (res?.storage_path) {
          partial.storagePath = res.storage_path;
        }
      } catch (err) {
        console.warn('[SettingsContext] Errore migrazione storage:', err);
        throw err;
      }
    }
    const nextLlm = partial.llmModel ?? partial.ollamaModel;
    if (nextLlm !== undefined) {
      partial.llmSupportsThinking =
        nextLlm in llmThinkingMapRef.current
          ? llmThinkingMapRef.current[nextLlm]
          : inferModelSupportsThinking(nextLlm);
    }
    setSettings((prev) => ({ ...prev, ...partial }));
    // Sincronizza anche con il backend
    const backendPayload: Record<string, any> = {};
    if (partial.whisperModel !== undefined) backendPayload.whisper_model = partial.whisperModel;
    if (partial.llmModel !== undefined) {
      backendPayload.llm_model = partial.llmModel;
      backendPayload.ollama_model = partial.llmModel;
    } else if (partial.ollamaModel !== undefined) {
      backendPayload.llm_model = partial.ollamaModel;
      backendPayload.ollama_model = partial.ollamaModel;
    }
    if (partial.embeddingModel !== undefined) backendPayload.embedding_model = partial.embeddingModel;
    if (partial.vadSensitivity !== undefined) backendPayload.vad_sensitivity = partial.vadSensitivity;
    if (partial.modelsDownloaded !== undefined) backendPayload.models_downloaded = partial.modelsDownloaded;
    if (partial.setupComplete !== undefined) backendPayload.setup_complete = partial.setupComplete;
    if (partial.audioFormat !== undefined) backendPayload.audio_format = partial.audioFormat;

    if (Object.keys(backendPayload).length > 0) {
      try {
        await api.settings.update(backendPayload);
      } catch (e) {
        console.warn('[SettingsContext] Errore updateSettings backend:', e);
      }
    }
  };

  const completeSetup = async (customSettings?: Partial<AppSettings>) => {
    const updated = {
      ...settings,
      ...customSettings,
      setupComplete: true,
    };
    // Prima il percorso di archiviazione (POST dedicato, atteso), poi il resto via PATCH
    if (customSettings?.storagePath) {
      try {
        const res = await api.settings.setStorage(customSettings.storagePath);
        if (res?.storage_path) {
          updated.storagePath = res.storage_path;
        }
      } catch (e) {
        console.error('[SettingsContext] Errore impostazione percorso archiviazione:', e);
        updated.storagePath = settings.storagePath;
      }
    }
    setSettings(updated);
    const backendPayload: Record<string, any> = {
      setup_complete: true,
      models_downloaded: updated.modelsDownloaded,
    };
    if (updated.whisperModel) backendPayload.whisper_model = updated.whisperModel;
    if (updated.llmModel) {
      backendPayload.llm_model = updated.llmModel;
      backendPayload.ollama_model = updated.llmModel;
    } else if (updated.ollamaModel) {
      backendPayload.llm_model = updated.ollamaModel;
      backendPayload.ollama_model = updated.ollamaModel;
    }
    if (updated.embeddingModel) backendPayload.embedding_model = updated.embeddingModel;
    if (updated.vadSensitivity) backendPayload.vad_sensitivity = updated.vadSensitivity;
    if (updated.audioFormat) backendPayload.audio_format = updated.audioFormat;
    try {
      await api.settings.update(backendPayload);
    } catch (e) {
      console.warn('[SettingsContext] Errore completeSetup backend:', e);
    }
  };

  const resetSetup = () => {
    const fresh: AppSettings = {
      ...DEFAULT_SETTINGS,
      setupComplete: false,
      modelsDownloaded: false,
    };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(fresh));
      localStorage.removeItem(LEGACY_STORAGE_KEY);
    } catch {}
    setSettings(fresh);
    api.settings.update({ models_downloaded: false, setup_complete: false }).catch(() => null);
  };

  return (
    <SettingsContext.Provider
      value={{
        settings,
        modelsStatus,
        setModelsStatus,
        updateSettings,
        completeSetup,
        resetSetup,
        syncWithBackend,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function isAutoUpdateEnabled(): boolean {
  try {
    const saved = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (typeof parsed.autoUpdate === 'boolean') {
        return parsed.autoUpdate;
      }
    }
  } catch {}
  return true; // Di default sempre attivo (ON)
}

export function useSettings(): SettingsContextType {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings deve essere usato all\'interno di un SettingsProvider');
  }
  return context;
}

