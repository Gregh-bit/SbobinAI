// lib/SettingsContext.tsx — Context React per la gestione persistente delle impostazioni di SbobinAI
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { api } from './api';

export interface AppSettings {
  setupComplete: boolean;
  modelsDownloaded: boolean;
  storagePath: string;
  whisperModel: string;
  llmModel: string;
  embeddingModel: string;
  ollamaModel: string; // Retrocompatibilità UI
  vadSensitivity: number; // 1 (permissivo) a 3 (molto reattivo/aggressivo)
}

interface SettingsContextType {
  settings: AppSettings;
  updateSettings: (newSettings: Partial<AppSettings>) => Promise<void>;
  completeSetup: (customSettings?: Partial<AppSettings>) => Promise<void>;
  resetSetup: () => void;
}

const DEFAULT_SETTINGS: AppSettings = {
  setupComplete: false,
  modelsDownloaded: false,
  storagePath: '~/Documents/SbobinAI',
  whisperModel: 'large-v3-turbo-q4',
  llmModel: 'mlx-community/Qwen3.5-4B-MLX-4bit',
  embeddingModel: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
  ollamaModel: 'mlx-community/Qwen3.5-4B-MLX-4bit',
  vadSensitivity: 2,
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
        return { ...DEFAULT_SETTINGS, ...parsed };
      }
    } catch (e) {
      console.warn('Errore lettura impostazioni da localStorage:', e);
    }
    return DEFAULT_SETTINGS;
  });

  // Sincronizza le impostazioni reali e lo stato effettivo dei modelli dal backend all'avvio
  useEffect(() => {
    Promise.all([
      api.settings.get().catch(() => null),
      api.models.status().catch(() => null),
    ]).then(([settingsData, modelsData]) => {
      const patch: Partial<AppSettings> = {};
      if (settingsData && typeof settingsData === 'object') {
        if (settingsData.storage_path && typeof settingsData.storage_path === 'string') {
          patch.storagePath = settingsData.storage_path;
          patch.setupComplete = true;
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
      }

      if (modelsData && typeof modelsData === 'object') {
        const hasRealWhisper = modelsData.whisper?.some(
          (m: any) => m.id !== 'mock_mode' && m.downloaded
        );
        if (hasRealWhisper || (modelsData.installed_count && modelsData.installed_count > 0)) {
          patch.modelsDownloaded = true;
          if (!patch.whisperModel || patch.whisperModel === 'mock_mode' || patch.whisperModel === 'dual-tier-whisper') {
            patch.whisperModel = 'large-v3-turbo-q4';
          }
        }
      }

      if (Object.keys(patch).length > 0) {
        setSettings((prev) => ({ ...prev, ...patch }));
      }
    }).catch((err) => {
      console.warn('[SettingsContext] Impossibile contattare backend per sincronizzazione iniziale:', err);
    });
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
    // Il percorso di archiviazione si cambia SOLO tramite POST /settings/storage (migrazione dati),
    // mai dentro il PATCH /settings. In caso di errore viene propagato al chiamante.
    if (partial.storagePath) {
      await api.settings.setStorage(partial.storagePath);
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
    setSettings(updated);
    // Prima il percorso di archiviazione (POST dedicato, atteso), poi il resto via PATCH
    if (customSettings?.storagePath) {
      try {
        await api.settings.setStorage(customSettings.storagePath);
      } catch (e) {
        console.warn('[SettingsContext] Errore impostazione percorso archiviazione:', e);
      }
    }
    const backendPayload: Record<string, any> = {};
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
    backendPayload.models_downloaded = updated.modelsDownloaded;
    try {
      await api.settings.update(backendPayload);
    } catch (e) {
      console.warn('[SettingsContext] Errore completeSetup backend:', e);
    }
  };

  const resetSetup = () => {
    setSettings({
      ...DEFAULT_SETTINGS,
      setupComplete: false,
    });
  };

  return (
    <SettingsContext.Provider
      value={{
        settings,
        updateSettings,
        completeSetup,
        resetSetup,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextType {
  const context = useContext(SettingsContext);
  if (!context) {
    throw new Error('useSettings deve essere usato all\'interno di un SettingsProvider');
  }
  return context;
}

