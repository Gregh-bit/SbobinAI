import { useEffect, useState, useCallback, useRef } from 'react';
import { 
  X, Folder, Cpu, Mic, HardDrive, Check, 
  RotateCcw, Sliders, Trash2, Download, Loader2,
  RefreshCw, CheckCircle2, ShieldCheck, Database, FileAudio, 
  Layers, AlertTriangle, AlertCircle, Sparkles, Volume2, Shield, Copy
} from 'lucide-react';
import { clsx } from 'clsx';
import { useSettings } from '../lib/SettingsContext';
import { pickFolder, isTauri } from '../lib/tauri';
import { isMac, fileManagerName, normalizeDisplayPath } from '../lib/platform';
import { useUpdater } from '../lib/useUpdater';
import { api } from '../lib/api';
import type { ModelsStatusResponse, ModelInfo } from '../types';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'storage' | 'models' | 'audio' | 'system';

const defaultLlms: ModelInfo[] = [
  {
    id: 'mlx-community/Qwen3.5-4B-MLX-4bit',
    name: 'Qwen 3.5 4B',
    type: 'llm',
    approx_size: '~2.8 GB',
    description: 'Compatto e veloce. Genera capitoli, sintesi ed evidenzia i concetti chiave.',
    downloaded: false,
    status: 'idle',
    disk_size_mb: null,
  },
  {
    id: 'mlx-community/Qwen3.5-2B-MLX-4bit',
    name: 'Qwen 3.5 2B (Leggero)',
    type: 'llm',
    approx_size: '~1.4 GB',
    description: 'Versione ultraleggera a basso consumo di memoria RAM. Ideale per computer con risorse contenute.',
    downloaded: false,
    status: 'idle',
    disk_size_mb: null,
  },
  {
    id: 'mlx-community/Qwen3.5-8B-MLX-4bit',
    name: 'Qwen 3.5 8B (Avanzato)',
    type: 'llm',
    approx_size: '~5.5 GB',
    description: 'Modello ad alta capacità di ragionamento per computer con 16 GB o più di RAM.',
    downloaded: false,
    status: 'idle',
    disk_size_mb: null,
  },
];

export default function SettingsModal({ isOpen, onClose }: Props) {
  const { settings, updateSettings, resetSetup } = useSettings();
  // Ref per leggere le impostazioni correnti senza far ri-eseguire gli effetti a ogni aggiornamento
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const updateSettingsRef = useRef(updateSettings);
  updateSettingsRef.current = updateSettings;

  const updater = useUpdater();
  const [activeTab, setActiveTab] = useState<TabType>('models');

  // Form State
  const [storagePath, setStoragePath] = useState(settings.storagePath);
  const [whisperModel, setWhisperModel] = useState(settings.whisperModel);
  const [llmModel, setLlmModel] = useState(settings.llmModel || 'mlx-community/Qwen3.5-4B-MLX-4bit');
  const [embeddingModel, setEmbeddingModel] = useState(settings.embeddingModel || 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ');
  const [vadSensitivity, setVadSensitivity] = useState(settings.vadSensitivity);
  const [confirmReset, setConfirmReset] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);
  const [confirmDeleteModelId, setConfirmDeleteModelId] = useState<string | null>(null);
  const [showUninstallModal, setShowUninstallModal] = useState(false);
  const [isUninstalling, setIsUninstalling] = useState(false);
  const [deleteStorageFolder, setDeleteStorageFolder] = useState(false);

  // Stato Modelli Reali dal Backend
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [actionInProgress, setActionInProgress] = useState<string | null>(null);
  const [notification, setNotification] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  // Stato Storage & File Orfani
  const [storageStats, setStorageStats] = useState<{
    audio_mb: number;
    uploads_mb: number;
    sqlite_mb: number;
    lancedb_mb: number;
    total_mb: number;
  } | null>(null);
  const [isCleaningOrphans, setIsCleaningOrphans] = useState(false);
  const [copyTooltip, setCopyTooltip] = useState<{ x: number; y: number } | null>(null);

  const handleCopyPath = async (e: React.MouseEvent, path: string) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(path);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = path;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopyTooltip({ x: e.clientX, y: e.clientY });
    setTimeout(() => {
      setCopyTooltip(null);
    }, 1400);
  };

  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setNotification({ type, message });
    setTimeout(() => {
      setNotification((prev) => (prev?.message === message ? null : prev));
    }, 4000);
  };

  const fetchStorageStats = useCallback(async () => {
    try {
      const stats = await api.settings.getStorageStats();
      setStorageStats(stats);
    } catch {
      // Ignora silenziosamente
    }
  }, []);

  const handleRecalculateAndOptimize = async () => {
    try {
      setIsCleaningOrphans(true);
      const res = await api.settings.cleanOrphans();
      await fetchStorageStats();
      await fetchModelsStatus(true);
      if (res && res.freed_mb > 0) {
        showNotification(
          `Ottimizzazione completata: liberati ${res.freed_mb} MB e spazio ricalcolato!`,
          'success'
        );
      } else {
        showNotification('Spazio su disco ricalcolato e memoria ottimizzata.', 'info');
      }
    } catch (err: any) {
      showNotification(err?.message || 'Errore durante l\'ottimizzazione dello spazio.', 'error');
    } finally {
      setIsCleaningOrphans(false);
    }
  };

  const fetchModelsStatus = useCallback(async (quiet = false) => {
    try {
      if (!quiet) setLoadingModels(true);
      const res = await api.models.status();
      setModelsStatus(res);
      const hasReal = res.whisper?.some((m: any) => m.id !== 'mock_mode' && m.downloaded);
      if (hasReal && !settingsRef.current.modelsDownloaded) {
        updateSettingsRef.current({ modelsDownloaded: true });
      }
    } catch (err) {
      console.warn('[SettingsModal] Errore verifica modelli:', err);
    } finally {
      if (!quiet) setLoadingModels(false);
    }
  }, []);

  // Pre-caricamento in background all'avvio per eliminare layout shift e attese all'apertura
  useEffect(() => {
    fetchModelsStatus(true);
    fetchStorageStats();
  }, [fetchModelsStatus, fetchStorageStats]);

  // Sincronizza lo stato solo all'apertura del modale (non a ogni modifica delle impostazioni)
  useEffect(() => {
    if (isOpen) {
      const current = settingsRef.current;
      setStoragePath(current.storagePath);
      setWhisperModel(current.whisperModel);
      setLlmModel(current.llmModel);
      setEmbeddingModel(current.embeddingModel);
      setVadSensitivity(current.vadSensitivity);
      setConfirmReset(false);
      setConfirmDeleteAll(false);
      setConfirmDeleteModelId(null);
      setNotification(null);
      fetchModelsStatus(true);
      fetchStorageStats();
    }
  }, [isOpen, fetchModelsStatus, fetchStorageStats]);

  // Polling automatico attivo se un modello è in fase di download
  useEffect(() => {
    if (!isOpen || !modelsStatus) return;
    const isWhisperDl = modelsStatus.whisper?.some((m) => m.status === 'downloading');
    const isLlmDl = modelsStatus.llm?.some((m) => m.status === 'downloading');
    const isEmbDl = modelsStatus.embedding?.some((m) => m.status === 'downloading');
    const isDownloading = isWhisperDl || isLlmDl || isEmbDl || actionInProgress !== null;
    if (!isDownloading) return;

    let isFetching = false;
    const timer = setInterval(async () => {
      if (isFetching) return;
      isFetching = true;
      try {
        await fetchModelsStatus(true);
      } finally {
        isFetching = false;
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [isOpen, modelsStatus, actionInProgress, fetchModelsStatus]);

  // Gestione tasto ESC con logica a livelli a cascata (standard macOS / desktop app)
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || isUninstalling) return;
      e.stopPropagation();

      // Livello 1: Sotto-finestra di disinstallazione aperta -> chiudi solo la sotto-finestra
      if (showUninstallModal) {
        setShowUninstallModal(false);
        return;
      }

      // Livello 2: Conferma ripristino configurazione attiva -> annulla la conferma
      if (confirmReset) {
        setConfirmReset(false);
        return;
      }

      // Livello 3: Conferma eliminazione modelli attiva -> annulla la conferma
      if (confirmDeleteAll) {
        setConfirmDeleteAll(false);
        return;
      }
      if (confirmDeleteModelId) {
        setConfirmDeleteModelId(null);
        return;
      }

      // Livello 4: Nessuna finestra interna o conferma in sospeso -> chiudi le impostazioni
      onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose, isUninstalling, showUninstallModal, confirmReset, confirmDeleteAll, confirmDeleteModelId]);

  // Se confirmReset è attivo, cliccare su qualsiasi altra parte dell'app lo annulla
  useEffect(() => {
    if (!confirmReset) return;
    const handlePointerDown = (e: MouseEvent | TouchEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest('[data-reset-wizard-btn]')) return;
      setConfirmReset(false);
    };
    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [confirmReset]);

  // Se confirmDeleteModelId è attivo, cliccare altrove annulla la conferma
  useEffect(() => {
    if (!confirmDeleteModelId) return;
    const handlePointerDown = (e: MouseEvent | TouchEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest(`[data-delete-model-btn="${confirmDeleteModelId}"]`)) return;
      setConfirmDeleteModelId(null);
    };
    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [confirmDeleteModelId]);

  // Avvio download di un modello con aggiornamento ottimistico istantaneo
  const handleDownloadModel = async (modelId: string) => {
    try {
      setActionInProgress(modelId);
      showNotification(`Avvio download del modello ${modelId}...`, 'info');

      // 1. Aggiornamento istantaneo dell'interfaccia: mostra immediatamente la barra di progresso
      setModelsStatus((prev) => {
        if (!prev) return prev;
        const updateList = (list?: ModelInfo[]) =>
          list?.map((m) =>
            m.id === modelId
              ? {
                  ...m,
                  status: 'downloading' as const,
                  progress: 1,
                  stage: 'Avvio download e connessione...',
                }
              : m
          );
        return {
          ...prev,
          whisper: updateList(prev.whisper) || prev.whisper,
          llm: updateList(prev.llm),
          embedding: updateList(prev.embedding),
        };
      });

      // 2. Chiamata al backend
      await api.models.download(modelId);
      await fetchModelsStatus(true);
    } catch (err: any) {
      showNotification(`Errore avvio download: ${err?.message || err}`, 'error');
      await fetchModelsStatus(true);
    } finally {
      setActionInProgress(null);
    }
  };

  // Interrompi download in corso
  const handleCancelDownload = async (modelId: string) => {
    try {
      setActionInProgress(`cancel-${modelId}`);
      const res = await api.models.cancelDownload(modelId);
      showNotification(`Download interrotto e file rimossi (${res.freed_mb} MB liberati).`, 'info');
      await fetchModelsStatus();
    } catch (err: any) {
      showNotification(`Errore interruzione download: ${err?.message || err}`, 'error');
    } finally {
      setActionInProgress(null);
    }
  };

  // Elimina singolo modello
  const handleDeleteModel = async (modelId: string) => {
    try {
      setActionInProgress(`delete-${modelId}`);
      const res = await api.models.delete(modelId);
      showNotification(res.message || `Modello rimosso con successo (${res.freed_mb} MB liberati).`, 'success');
      setConfirmDeleteModelId(null);
      
      // Se il modello eliminato era quello attivo, ripristina il default
      if (whisperModel === modelId) {
        setWhisperModel('mock_mode');
        updateSettings({ whisperModel: 'mock_mode' });
      }
      if (llmModel === modelId) {
        setLlmModel('mlx-community/Qwen3.5-4B-MLX-4bit');
        updateSettings({ llmModel: 'mlx-community/Qwen3.5-4B-MLX-4bit', ollamaModel: 'mlx-community/Qwen3.5-4B-MLX-4bit' });
      }
      if (embeddingModel === modelId) {
        setEmbeddingModel('mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ');
        updateSettings({ embeddingModel: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ' });
      }
      await fetchModelsStatus();
      await fetchStorageStats();
    } catch (err: any) {
      showNotification(`Errore eliminazione: ${err?.message || err}`, 'error');
    } finally {
      setActionInProgress(null);
    }
  };

  // Elimina tutti i modelli
  const handleDeleteAllModels = async () => {
    try {
      setActionInProgress('delete-all');
      const res = await api.models.deleteAll();
      showNotification(res.message || `Tutti i modelli sono stati eliminati (${res.freed_mb} MB liberati).`, 'success');
      setConfirmDeleteAll(false);
      setWhisperModel('mock_mode');
      updateSettings({ whisperModel: 'mock_mode', modelsDownloaded: false });
      await fetchModelsStatus();
      await fetchStorageStats();
    } catch (err: any) {
      showNotification(`Errore eliminazione totale: ${err?.message || err}`, 'error');
    } finally {
      setActionInProgress(null);
    }
  };

  const handleSelectActiveLlm = (id: string) => {
    setLlmModel(id);
    updateSettings({ llmModel: id, ollamaModel: id });
    showNotification(`Modello LLM impostato su: ${id}`, 'success');
  };

  const handlePickFolder = async () => {
    const selected = await pickFolder(storagePath);
    if (selected) {
      const previous = storagePath;
      setStoragePath(selected);
      try {
        // POST /settings/storage (una sola volta, atteso) gestito da updateSettings
        await updateSettings({ storagePath: selected });
        showNotification(`Cartella di archiviazione aggiornata in ${fileManagerName}.`, 'success');
        fetchStorageStats();
      } catch (err: any) {
        setStoragePath(previous);
        showNotification(`Errore aggiornamento cartella: ${err?.message || err}`, 'error');
      }
    }
  };

  const handleVadChange = (newVal: number) => {
    setVadSensitivity(newVal);
    updateSettings({ vadSensitivity: newVal });
  };

  const handleTabChange = (tab: TabType) => {
    setConfirmReset(false);
    setConfirmDeleteAll(false);
    setConfirmDeleteModelId(null);
    setActiveTab(tab);
  };

  const handleResetWizard = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (!confirmReset) {
      setConfirmReset(true);
      return;
    }
    resetSetup();
    onClose();
  };

  const handleUninstallApp = async () => {
    setIsUninstalling(true);
    try {
      if (isTauri()) {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('uninstall_sbobinai', { 
          keepStoragePath: storagePath,
          deleteStorageFolder: deleteStorageFolder,
        });
      } else {
        try {
          await api.models.deleteAll();
        } catch {}
      }

      // Pulisce le chiavi di configurazione locali dell'app
      try {
        if (deleteStorageFolder) {
          localStorage.clear();
        } else {
          localStorage.removeItem('sbobinai_settings_v1');
          localStorage.removeItem('unicopilot_settings_v1');
        }
        sessionStorage.clear();
      } catch (e) {
        console.warn('Errore pulizia storage locale:', e);
      }
    } catch (err: any) {
      console.error('Errore durante la disinstallazione:', err);
      setIsUninstalling(false);
      setShowUninstallModal(false);
      showNotification('Errore durante la disinstallazione: ' + (err?.message || String(err)), 'error');
    }
  };

  if (!isOpen) return null;

  const totalInstalledMb = modelsStatus?.total_installed_mb ?? 0;
  const audioMb = storageStats?.audio_mb ?? 0;
  const sqliteMb = storageStats?.sqlite_mb ?? 0;
  const lancedbMb = storageStats?.lancedb_mb ?? 0;
  const uploadsMb = storageStats?.uploads_mb ?? 0;
  const combinedTotalMb = Math.round((audioMb + totalInstalledMb + sqliteMb + lancedbMb + uploadsMb) * 100) / 100;

  const downloadedModelsCount =
    (modelsStatus?.whisper?.filter((m) => m.downloaded && m.id !== 'mock_mode')?.length ?? 0) +
    (modelsStatus?.llm?.filter((m) => m.downloaded)?.length ?? 0) +
    (modelsStatus?.embedding?.filter((m) => m.downloaded)?.length ?? 0);

  const formatSize = (mb: number): string => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(2)} GB`;
    }
    return `${mb.toFixed(1)} MB`;
  };

  const getBarWidth = (valMb: number) => {
    if (combinedTotalMb <= 0 || valMb <= 0) return 0;
    const pct = (valMb / combinedTotalMb) * 100;
    return Math.max(1.8, pct);
  };

  const formatPct = (valMb: number) => {
    if (combinedTotalMb <= 0 || valMb <= 0) return '0%';
    const pct = (valMb / combinedTotalMb) * 100;
    if (pct < 1) return '<1%';
    return `${Math.round(pct)}%`;
  };

  const storageCategories = [
    {
      id: 'models',
      label: 'Modelli IA',
      shortLabel: 'Modelli IA',
      sizeMb: totalInstalledMb,
      color: 'bg-[#BF5AF2]',
      dotColor: 'bg-[#BF5AF2]',
      badgeBg: 'bg-[#BF5AF2]/15',
      badgeBorder: 'border-[#BF5AF2]/30',
      badgeText: 'text-[#D087FF]',
      icon: Cpu,
      path: '~/.cache/huggingface/hub',
      desc: 'Pesi neurali locali per trascrizione, riassunti e ricerca.',
      details: `${downloadedModelsCount} ${downloadedModelsCount === 1 ? 'modello' : 'modelli'}`,
    },
    {
      id: 'audio',
      label: 'Audio Lezioni',
      shortLabel: 'Audio Lezioni',
      sizeMb: audioMb,
      color: 'bg-[#0A84FF]',
      dotColor: 'bg-[#0A84FF]',
      badgeBg: 'bg-[#0A84FF]/15',
      badgeBorder: 'border-[#0A84FF]/30',
      badgeText: 'text-[#5AC8FA]',
      icon: FileAudio,
      path: normalizeDisplayPath(`${storagePath}/audio`),
      desc: 'Registrazioni audio compresse in formato M4A.',
      details: 'Formato M4A salva-spazio',
    },
    {
      id: 'sqlite',
      label: 'Database & Appunti',
      shortLabel: 'Database',
      sizeMb: sqliteMb,
      color: 'bg-[#30D158]',
      dotColor: 'bg-[#30D158]',
      badgeBg: 'bg-[#30D158]/15',
      badgeBorder: 'border-[#30D158]/30',
      badgeText: 'text-[#6EE7B7]',
      icon: Database,
      path: normalizeDisplayPath(`${storagePath}/sqlite/sbobinai.db`),
      desc: 'Trascrizioni, sbobine, quiz, flashcard e note dei corsi.',
      details: 'Dati e appunti',
    },
    {
      id: 'lancedb',
      label: 'Memoria Semantica',
      shortLabel: 'Memoria RAG',
      sizeMb: lancedbMb,
      color: 'bg-[#FF9F0A]',
      dotColor: 'bg-[#FF9F0A]',
      badgeBg: 'bg-[#FF9F0A]/15',
      badgeBorder: 'border-[#FF9F0A]/30',
      badgeText: 'text-[#FDBA74]',
      icon: Layers,
      path: normalizeDisplayPath(`${storagePath}/lancedb`),
      desc: 'Indice concettuale per la ricerca intelligente.',
      details: 'Indice vettoriale',
    },
    {
      id: 'uploads',
      label: 'Materiali di Studio',
      shortLabel: 'Materiali',
      sizeMb: uploadsMb,
      color: 'bg-[#FF375F]',
      dotColor: 'bg-[#FF375F]',
      badgeBg: 'bg-[#FF375F]/15',
      badgeBorder: 'border-[#FF375F]/30',
      badgeText: 'text-[#FDA4AF]',
      icon: Folder,
      path: normalizeDisplayPath(`${storagePath}/uploads`),
      desc: 'Slide, dispense e documenti PDF dei corsi.',
      details: 'Documenti PDF',
    },
  ];

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in select-none"
      onClick={() => { if (!isUninstalling) onClose(); }}
    >
      <div 
        className="relative w-[880px] max-w-[94vw] h-[640px] max-h-[90vh] rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Modal Header ── */}
        <div className="h-14 px-6 flex items-center justify-between border-b border-white/[0.08] flex-shrink-0 bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-white/[0.08] border border-white/[0.12] flex items-center justify-center text-white/80">
              <Sliders size={15} strokeWidth={1.75} />
            </div>
            <div>
              <h2 className="text-xs font-semibold tracking-tight text-white">Impostazioni SbobinAI</h2>
            </div>
          </div>

          <button
            onClick={() => { if (!isUninstalling) onClose(); }}
            disabled={isUninstalling}
            className="w-7 h-7 rounded-full flex items-center justify-center text-white/40 hover:text-white/90 hover:bg-white/[0.08] transition-colors cursor-pointer disabled:opacity-30 disabled:pointer-events-none"
            title="Chiudi (Esc)"
          >
            <X size={15} strokeWidth={2} />
          </button>
        </div>

        {/* ── Modal Layout: Sidebar a sinistra + Contenuto a destra ── */}
        <div className="flex-1 flex overflow-hidden">
          
          {/* ── Sidebar Categorie (Design coerente senza arcobaleno) ── */}
          <div className="w-56 border-r border-white/[0.08] liquid-glass-subtle p-3 flex flex-col justify-start flex-shrink-0">
            <div className="space-y-1">
              {/* Tab 1: Archiviazione */}
              <button
                onClick={() => handleTabChange('storage')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'storage'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <HardDrive size={15} className="text-white/70" />
                <span>Archiviazione</span>
              </button>

              {/* Tab 2: Modelli IA */}
              <button
                onClick={() => handleTabChange('models')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'models'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <Cpu size={15} className="text-white/70" />
                <span>Modelli IA</span>
              </button>

              {/* Tab 3: Audio & Registrazione */}
              <button
                onClick={() => handleTabChange('audio')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'audio'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <Mic size={15} className="text-white/70" />
                <span>Audio & Registrazione</span>
              </button>

              {/* Tab 4: Sistema & Privacy */}
              <button
                onClick={() => handleTabChange('system')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'system'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <ShieldCheck size={15} className="text-white/70" />
                <span>Sistema & Privacy</span>
              </button>
            </div>
          </div>

          {/* ── Main Tab Content Panel con scroll fluido e spaziatura corretta ── */}
          <div className="flex-1 p-6 pb-24 overflow-y-auto space-y-6 [scrollbar-gutter:stable]">
            
            {/* Banner Notifica Azione / Toast */}
            {notification && (
              <div className={clsx(
                'p-3.5 rounded-2xl border text-xs flex items-center justify-between animate-fade-in shadow-liquid-sm',
                notification.type === 'success' && 'bg-emerald-500/15 border-emerald-500/30 text-emerald-200',
                notification.type === 'error' && 'bg-red-500/15 border-red-500/30 text-red-200',
                notification.type === 'info' && 'bg-blue-500/15 border-blue-500/30 text-blue-200'
              )}>
                <div className="flex items-center gap-2.5">
                  {notification.type === 'success' && <CheckCircle2 size={15} className="text-emerald-400 flex-shrink-0" />}
                  {notification.type === 'error' && <AlertTriangle size={15} className="text-red-400 flex-shrink-0" />}
                  {notification.type === 'info' && <Loader2 size={15} className="text-blue-400 animate-spin flex-shrink-0" />}
                  <span className="font-medium">{notification.message}</span>
                </div>
                <button 
                  type="button" 
                  onClick={() => setNotification(null)}
                  className="text-white/40 hover:text-white cursor-pointer ml-3"
                >
                  <X size={14} />
                </button>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 1: ARCHIVIAZIONE & MEMORIA LOCALE (Stile Apple Storage)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'storage' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-bold text-white">Archiviazione</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    Gestisci lo spazio su disco occupato da lezioni, trascrizioni e modelli IA.
                  </p>
                </div>

                {/* ── Apple-style Storage Overview Card ── */}
                <div className="p-5 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-white/45 block">
                        Spazio occupato
                      </span>
                      <div className="flex items-baseline gap-2 mt-0.5 whitespace-nowrap">
                        <h4 className="text-3xl font-bold font-mono text-white tracking-tight">
                          {combinedTotalMb >= 1024 ? (combinedTotalMb / 1024).toFixed(2) : combinedTotalMb.toFixed(1)}
                        </h4>
                        <span className="text-base font-semibold font-mono text-white/70">
                          {combinedTotalMb >= 1024 ? 'GB' : 'MB'}
                        </span>
                        <span className="text-xs text-white/40 ml-1 hidden sm:inline">
                          totali su disco ({isMac ? 'Mac' : 'PC'})
                        </span>
                      </div>
                    </div>

                    {/* Tasto Azione Unificato: Ottimizza e Ricalcola */}
                    <div className="flex items-center gap-2 self-start sm:self-center">
                      <button
                        type="button"
                        onClick={handleRecalculateAndOptimize}
                        disabled={isCleaningOrphans || loadingModels}
                        className={clsx(
                          "px-3.5 py-2 rounded-xl border text-xs font-semibold transition-all flex items-center gap-2 cursor-pointer shadow-liquid-sm whitespace-nowrap disabled:opacity-50",
                          isCleaningOrphans
                            ? "bg-blue-500/20 text-blue-200 border-blue-500/30 cursor-wait"
                            : "bg-white/[0.08] hover:bg-white/[0.14] border-white/14 text-white/80 hover:text-white"
                        )}
                        title="Pulisce i file temporanei, deframmenta il database ed aggiorna lo spazio su disco"
                      >
                        <RefreshCw size={13} className={clsx((isCleaningOrphans || loadingModels) && "animate-spin text-blue-400")} />
                        <span>{isCleaningOrphans ? "Ottimizzazione in corso..." : "Ricalcola spazio"}</span>
                      </button>
                    </div>
                  </div>

                  {/* ── Barra Orizzontale Multicolore Stile Apple ── */}
                  <div className="space-y-3">
                    <div className="w-full h-3 rounded-full bg-white/[0.06] p-0.5 flex gap-1 overflow-hidden border border-white/[0.12]">
                      {combinedTotalMb === 0 ? (
                        <div className="h-full w-full bg-white/10 rounded-full animate-pulse" />
                      ) : (
                        storageCategories
                          .filter((cat) => cat.sizeMb > 0)
                          .map((cat) => (
                            <div
                              key={cat.id}
                              className={clsx('h-full rounded-full transition-all duration-500', cat.color)}
                              style={{ width: `${getBarWidth(cat.sizeMb)}%` }}
                              title={`${cat.shortLabel}: ${formatSize(cat.sizeMb)} (${formatPct(cat.sizeMb)})`}
                            />
                          ))
                      )}
                    </div>

                    {/* ── Legenda Ordinata a Griglia Stile macOS Storage ── */}
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2 pt-2 border-t border-white/[0.06] text-xs">
                      {storageCategories.map((cat) => {
                        const hasData = cat.sizeMb > 0;
                        return (
                          <div
                            key={cat.id}
                            className={clsx(
                              "flex items-center gap-1.5 whitespace-nowrap transition-opacity",
                              hasData ? "opacity-100" : "opacity-45"
                            )}
                          >
                            <span className={clsx('w-2 h-2 rounded-full flex-shrink-0 shadow-sm', cat.dotColor)} />
                            <span className="text-white/80 font-medium text-[11px] truncate">{cat.shortLabel}:</span>
                            <span className="font-mono text-[11px] text-white/70">{formatSize(cat.sizeMb)}</span>
                            <span className="text-[10px] text-white/40 font-mono">({formatPct(cat.sizeMb)})</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* ── Griglia Dati: Categorie di Archiviazione ── */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                      Categorie di Archiviazione
                    </span>
                    <span className="text-[11px] text-white/40 font-mono">
                      5 categorie monitorate
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {storageCategories.map((cat) => {
                      const Icon = cat.icon;
                      const isHero = cat.id === 'models';

                      // Scheda Hero: Modelli IA con morphing in-place fluido e altezza stabile
                      if (isHero) {
                        return (
                          <div
                            key={cat.id}
                            className={clsx(
                              'sm:col-span-2 p-4 rounded-2xl border transition-colors duration-300 relative flex flex-col justify-between min-h-[160px] overflow-hidden',
                              confirmDeleteAll
                                ? 'bg-red-500/15 border-red-500/35 shadow-liquid-sm'
                                : 'group/card bg-white/[0.05] border-white/[0.12] hover:border-white/[0.18]'
                            )}
                          >
                            <AnimatePresence mode="wait" initial={false}>
                              {confirmDeleteAll ? (
                                <motion.div
                                  key="confirm"
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  transition={{ duration: 0.15, ease: 'easeOut' }}
                                  className="flex flex-col justify-between h-full flex-1 gap-3"
                                >
                                  <div className="space-y-1.5">
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="flex items-center gap-2.5 min-w-0">
                                        <div className="w-7 h-7 rounded-xl bg-red-500/20 border border-red-500/30 flex items-center justify-center flex-shrink-0 text-red-300">
                                          <Trash2 size={14} />
                                        </div>
                                        <h4 className="text-xs font-bold text-white truncate">
                                          Eliminare tutti i modelli dal disco?
                                        </h4>
                                      </div>
                                      <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-red-500/25 text-red-200 border border-red-500/35 whitespace-nowrap flex-shrink-0">
                                        {formatSize(totalInstalledMb)} da liberare
                                      </span>
                                    </div>

                                    <p className="text-[11px] font-mono text-red-300/60 truncate" title="~/.cache/huggingface/hub">
                                      ~/.cache/huggingface/hub
                                    </p>

                                    <p className="text-[11px] text-red-200/80 leading-relaxed">
                                      Verranno rimossi fisicamente dalla cache di sistema tutti i pesi scaricati. L'app tornerà in modalità base.
                                    </p>
                                  </div>

                                  <div className="pt-2.5 border-t border-red-500/20 flex items-center justify-between text-[10px] gap-2">
                                    <span className="text-red-300/60 font-medium truncate">Azione irreversibile</span>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          setConfirmDeleteAll(false);
                                        }}
                                        className="px-3.5 py-1.5 rounded-xl text-xs font-medium text-white/70 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer whitespace-nowrap"
                                      >
                                        Annulla
                                      </button>
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          handleDeleteAllModels();
                                        }}
                                        disabled={actionInProgress === 'delete-all'}
                                        className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-red-500 hover:bg-red-600 text-white transition-colors cursor-pointer shadow-liquid-sm disabled:opacity-50 whitespace-nowrap flex items-center gap-1.5"
                                      >
                                        {actionInProgress === 'delete-all' && <Loader2 size={12} className="animate-spin text-white" />}
                                        <span>{actionInProgress === 'delete-all' ? 'Eliminazione...' : 'Sì, elimina tutto'}</span>
                                      </button>
                                    </div>
                                  </div>
                                </motion.div>
                              ) : (
                                <motion.div
                                  key="normal"
                                  initial={{ opacity: 0 }}
                                  animate={{ opacity: 1 }}
                                  exit={{ opacity: 0 }}
                                  transition={{ duration: 0.15, ease: 'easeOut' }}
                                  className="flex flex-col justify-between h-full flex-1 gap-3"
                                >
                                  <div className="space-y-1.5">
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="flex items-center gap-2.5 min-w-0">
                                        <div className={clsx('w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0', cat.badgeBg, cat.badgeText)}>
                                          <Icon size={14} />
                                        </div>
                                        <span className="text-xs font-bold text-white truncate">
                                          {cat.label}
                                        </span>
                                      </div>

                                      <div className="flex items-center gap-2 flex-shrink-0">
                                        {totalInstalledMb > 0 && (
                                          <button
                                            type="button"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              setConfirmDeleteAll(true);
                                            }}
                                            disabled={actionInProgress !== null}
                                            className="opacity-0 group-hover/card:opacity-100 transition-all p-1.5 rounded-lg text-red-300 hover:text-red-200 hover:bg-red-500/20 border border-transparent hover:border-red-500/30 cursor-pointer flex items-center gap-1 text-[11px] font-medium"
                                            title="Svuota cache e libera memoria modelli"
                                          >
                                            <Trash2 size={13} className="text-red-300" />
                                            <span className="hidden sm:inline">Libera</span>
                                          </button>
                                        )}

                                        <span className={clsx(
                                          'text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border whitespace-nowrap',
                                          cat.badgeBg, cat.badgeBorder, cat.badgeText
                                        )}>
                                          {formatSize(cat.sizeMb)}
                                        </span>
                                      </div>
                                    </div>

                                    <button
                                      type="button"
                                      onClick={(e) => handleCopyPath(e, cat.path)}
                                      className="group/path flex items-center gap-1.5 px-2 py-1 -ml-2 rounded-lg hover:bg-white/[0.06] text-[11px] font-mono text-white/40 hover:text-white/90 transition-all cursor-pointer w-fit max-w-full text-left"
                                      title="Clicca per copiare il percorso"
                                    >
                                      <span className="truncate">{cat.path}</span>
                                      <Copy size={11} className="opacity-0 group-hover/path:opacity-100 transition-opacity flex-shrink-0 text-white/50" />
                                    </button>

                                    <p className="text-[11px] text-white/50 leading-relaxed">
                                      {cat.desc}
                                    </p>
                                  </div>

                                  <div className="pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[10px] gap-2">
                                    <span className="text-white/40 font-medium truncate">{cat.details}</span>
                                    <span className="font-mono text-white/50 font-bold whitespace-nowrap flex-shrink-0">{formatPct(cat.sizeMb)} del totale</span>
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>
                        );
                      }

                      return (
                        <div
                          key={cat.id}
                          className="group/card p-4 rounded-2xl bg-white/[0.03] border border-white/[0.08] hover:border-white/[0.14] transition-all flex flex-col justify-between gap-3 relative min-h-[160px]"
                        >
                          <div className="space-y-1.5">
                            <div className="flex items-center justify-between gap-3">
                              <div className="flex items-center gap-2.5 min-w-0">
                                <div className={clsx('w-7 h-7 rounded-xl flex items-center justify-center flex-shrink-0', cat.badgeBg, cat.badgeText)}>
                                  <Icon size={14} />
                                </div>
                                <span className="text-xs font-bold text-white truncate">
                                  {cat.label}
                                </span>
                              </div>

                              <span className={clsx(
                                'text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border whitespace-nowrap',
                                cat.badgeBg, cat.badgeBorder, cat.badgeText
                              )}>
                                {formatSize(cat.sizeMb)}
                              </span>
                            </div>

                            <button
                              type="button"
                              onClick={(e) => handleCopyPath(e, cat.path)}
                              className="group/path flex items-center gap-1.5 px-2 py-1 -ml-2 rounded-lg hover:bg-white/[0.06] text-[11px] font-mono text-white/40 hover:text-white/90 transition-all cursor-pointer w-fit max-w-full text-left"
                              title="Clicca per copiare il percorso"
                            >
                              <span className="truncate">{cat.path}</span>
                              <Copy size={11} className="opacity-0 group-hover/path:opacity-100 transition-opacity flex-shrink-0 text-white/50" />
                            </button>

                            <p className="text-[11px] text-white/50 leading-relaxed">
                              {cat.desc}
                            </p>
                          </div>

                          <div className="pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[10px] gap-2">
                            <span className="text-white/40 font-medium truncate">{cat.details}</span>
                            <span className="font-mono text-white/50 font-bold whitespace-nowrap flex-shrink-0">{formatPct(cat.sizeMb)} del totale</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* ── Cartella Libreria Principale (spostata in basso) ── */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-white/80 block">
                      Cartella Libreria SbobinAI
                    </label>
                    <span className="text-[10px] text-white/40 font-mono">
                      Archivio locale quaderni e corsi
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      onClick={(e) => handleCopyPath(e, storagePath)}
                      className="group/lib flex-1 px-4 py-2.5 rounded-2xl liquid-input font-mono text-xs text-white/95 truncate hover:bg-white/[0.08] transition-all cursor-pointer flex items-center justify-between text-left"
                      title="Clicca per copiare il percorso della libreria"
                    >
                      <span className="truncate">{normalizeDisplayPath(storagePath)}</span>
                      <Copy size={13} className="opacity-0 group-hover/lib:opacity-100 transition-opacity flex-shrink-0 text-white/45 ml-2" />
                    </button>
                    <button
                      type="button"
                      onClick={handlePickFolder}
                      className="px-4 py-2.5 rounded-2xl liquid-glass-pill border border-white/18 hover:border-white/28 hover:bg-white/[0.10] text-xs font-semibold text-white transition-all flex items-center gap-2 flex-shrink-0 cursor-pointer shadow-liquid-sm whitespace-nowrap"
                    >
                      <Folder size={14} className="text-white/70" />
                      <span>Sfoglia in {fileManagerName}...</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 2: MODELLI IA (DESIGN UNIFICATO E MINIMALE)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'models' && (
              <div className="space-y-6 animate-fade-in">
                
                {/* ── Header Tab Modelli ── */}
                <div className="pb-2 border-b border-white/[0.08]">
                  <h3 className="text-sm font-bold text-white">Modelli IA</h3>
                  <p className="text-xs text-white/50 mt-0.5 leading-relaxed">
                    Eseguiti al 100% in locale sul tuo computer per la massima riservatezza e velocità.
                  </p>
                </div>

                {/* ── SEZIONE 1: MODELLI DI BASE ── */}
                <div className="space-y-3">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                      Modelli di Base
                    </span>
                    <p className="text-[11px] text-white/40 mt-0.5">
                      Necessari per la trascrizione audio e la ricerca concettuale.
                    </p>
                  </div>

                  <div className="space-y-2.5">
                    {(() => {
                      const coreWhisper = modelsStatus?.whisper?.find((m) => m.id === 'large-v3-turbo-q4') || {
                        id: 'large-v3-turbo-q4',
                        name: 'Whisper Large-v3-Turbo',
                        type: 'whisper' as const,
                        approx_size: '~580 MB',
                        description: 'Trascrive le registrazioni audio parola per parola con alta precisione.',
                        downloaded: false,
                        status: 'idle' as const,
                        disk_size_mb: null,
                      };

                      const coreEmbedding = modelsStatus?.embedding?.find((m) => m.id === 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ') || {
                        id: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
                        name: 'Qwen3 Embedding 0.6B',
                        type: 'embedding' as const,
                        approx_size: '~300 MB',
                        description: 'Organizza la memoria concettuale delle lezioni per la ricerca istantanea.',
                        downloaded: false,
                        status: 'idle' as const,
                        disk_size_mb: null,
                      };

                      const coreItems = [coreWhisper, coreEmbedding];

                      return coreItems.map((m) => {
                        const isDownloading = m.status === 'downloading' || actionInProgress === m.id;
                        const isCancellingThis = actionInProgress === `cancel-${m.id}`;
                        const isMic = m.type === 'whisper';
                        const sizeDisplay = m.downloaded
                          ? `${m.disk_size_mb || m.total_mb || '—'} MB`
                          : m.approx_size;

                        return (
                          <div
                            key={m.id}
                            className={clsx(
                              'p-3.5 rounded-2xl border transition-all space-y-2.5',
                              m.downloaded
                                ? 'liquid-glass border-white/[0.12] shadow-liquid-sm'
                                : 'liquid-glass-subtle border-white/[0.08] hover:border-white/16'
                            )}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div className="flex items-center gap-3 min-w-0">
                                <div className={clsx(
                                  'w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0',
                                  m.downloaded
                                    ? 'bg-white/[0.10] text-white border border-white/[0.14]'
                                    : 'bg-white/5 text-white/40 border border-white/10'
                                )}>
                                  {isMic ? <Mic size={16} /> : <Database size={16} />}
                                </div>

                                <div className="min-w-0">
                                  <div className="flex items-center gap-2">
                                    <h4 className="text-xs font-semibold text-white truncate">{m.name}</h4>
                                    <span className="text-[10px] font-mono text-white/40 flex-shrink-0">
                                      {sizeDisplay}
                                    </span>
                                  </div>
                                  <p className="text-[11px] text-white/50 truncate">
                                    {m.description}
                                  </p>
                                </div>
                              </div>

                              <div className="flex items-center gap-2 flex-shrink-0">
                                {m.downloaded ? (
                                  <span className="text-[11px] font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full flex items-center gap-1.5">
                                    <Check size={11} strokeWidth={2.5} />
                                    <span>Pronto</span>
                                  </span>
                                ) : isDownloading ? (
                                  <button
                                    type="button"
                                    onClick={() => handleCancelDownload(m.id)}
                                    disabled={isCancellingThis}
                                    className="flex items-center gap-1 px-3 py-1 rounded-xl text-xs font-medium bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer"
                                  >
                                    <X size={12} className="text-red-300" />
                                    <span>{isCancellingThis ? 'Annullamento...' : 'Interrompi'}</span>
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    onClick={() => handleDownloadModel(m.id)}
                                    disabled={actionInProgress !== null}
                                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-medium bg-blue-500 hover:bg-blue-600 text-white shadow-liquid-sm transition-all cursor-pointer disabled:opacity-50"
                                  >
                                    <Download size={12} />
                                    <span>Installa</span>
                                  </button>
                                )}
                              </div>
                            </div>

                            {isDownloading && (
                              <div className="w-full pt-1.5 border-t border-white/[0.08] space-y-1 animate-fade-in">
                                <div className="flex justify-between items-center text-[10px] font-mono">
                                  <span className="flex items-center gap-1.5 text-blue-300 font-medium truncate max-w-[80%]">
                                    <Loader2 size={11} className="animate-spin flex-shrink-0" />
                                    <span className="truncate">{m.stage || "Download in corso..."}</span>
                                  </span>
                                  <span className="font-bold text-white">{m.progress || 0}%</span>
                                </div>
                                <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 rounded-full transition-all duration-300"
                                    style={{ width: `${Math.min(100, Math.max(4, m.progress || 0))}%` }}
                                  />
                                </div>
                              </div>
                            )}

                            {m.status === 'error' && !isDownloading && (
                              <div className="w-full pt-1.5 border-t border-red-500/20 text-xs text-red-300 flex items-center justify-between gap-2 animate-fade-in">
                                <div className="flex items-center gap-1.5 truncate max-w-[75%]">
                                  <AlertCircle size={12} className="text-red-400 shrink-0" />
                                  <span className="truncate text-[11px]">{m.error || "Errore durante il download."}</span>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => handleDownloadModel(m.id)}
                                  className="px-2 py-0.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-200 text-[10px] font-medium shrink-0 cursor-pointer"
                                >
                                  Riprova
                                </button>
                              </div>
                            )}
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>

                {/* ── SEZIONE 2: MODELLI LINGUISTICI (LLM) ── */}
                <div className="space-y-3 pt-2">
                  <div>
                    <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                      Modelli Linguistici (LLM)
                    </span>
                    <p className="text-[11px] text-white/40 mt-0.5">
                      Generano capitoli di studio, sintesi e rispondono sul quaderno delle lezioni.
                    </p>
                  </div>

                  <div className="space-y-2.5">
                    {(() => {
                      const llmList = (modelsStatus?.llm && modelsStatus.llm.length > 0) ? modelsStatus.llm : defaultLlms;
                      return llmList.map((m: ModelInfo) => {
                        const isCurrentActive = llmModel === m.id;
                        const isDownloading = m.status === 'downloading' || actionInProgress === m.id;
                        const isDeletingThis = actionInProgress === `delete-${m.id}`;
                        const isCancellingThis = actionInProgress === `cancel-${m.id}`;
                        const sizeDisplay = m.downloaded
                          ? `${m.disk_size_mb || m.total_mb || '—'} MB`
                          : m.approx_size;

                        return (
                          <div
                            key={m.id}
                            className={clsx(
                              'p-3.5 rounded-2xl border transition-all space-y-2.5',
                              m.downloaded
                                ? 'liquid-glass border-white/[0.12] shadow-liquid-sm'
                                : 'liquid-glass-subtle border-white/[0.08] hover:border-white/16'
                            )}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <div className="flex items-center gap-3 min-w-0">
                                <div className={clsx(
                                  'w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0',
                                  m.downloaded
                                    ? 'bg-white/10 text-white/90 border border-white/15'
                                    : 'bg-white/5 text-white/40 border border-white/10'
                                )}>
                                  <Cpu size={16} />
                                </div>

                                <div className="min-w-0">
                                  <div className="flex items-center gap-2">
                                    <h4 className="text-xs font-semibold text-white truncate">{m.name}</h4>
                                    <span className="text-[10px] font-mono text-white/40 flex-shrink-0">
                                      {sizeDisplay}
                                    </span>
                                  </div>
                                  <p className="text-[11px] text-white/50 truncate">
                                    {m.description}
                                  </p>
                                </div>
                              </div>

                              <div className="flex items-center gap-2 flex-shrink-0">
                                {m.downloaded ? (
                                  <>
                                    {isCurrentActive ? (
                                      <span className="text-[11px] font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full flex items-center gap-1.5">
                                        <Check size={11} strokeWidth={2.5} />
                                        <span>In uso</span>
                                      </span>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={() => handleSelectActiveLlm(m.id)}
                                        className="px-3 py-1 rounded-xl text-xs font-medium bg-white/[0.10] hover:bg-white/[0.18] text-white border border-white/12 transition-all cursor-pointer"
                                      >
                                        Attiva
                                      </button>
                                    )}

                                    {confirmDeleteModelId === m.id ? (
                                      <div className="flex items-center gap-1.5 animate-fade-in">
                                        <button
                                          type="button"
                                          onClick={() => setConfirmDeleteModelId(null)}
                                          className="px-2 py-0.5 rounded-lg text-[10px] text-white/60 hover:text-white"
                                        >
                                          Annulla
                                        </button>
                                        <button
                                          type="button"
                                          onClick={() => handleDeleteModel(m.id)}
                                          disabled={isDeletingThis}
                                          className="px-2 py-0.5 rounded-lg text-[10px] font-semibold bg-red-500 text-white hover:bg-red-600 cursor-pointer"
                                        >
                                          {isDeletingThis ? '...' : 'Elimina'}
                                        </button>
                                      </div>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={() => setConfirmDeleteModelId(m.id)}
                                        disabled={isDeletingThis}
                                        className="p-1.5 rounded-lg text-white/40 hover:text-red-300 hover:bg-red-500/15 transition-all cursor-pointer"
                                        title="Elimina modello per liberare spazio"
                                      >
                                        <Trash2 size={13} />
                                      </button>
                                    )}
                                  </>
                                ) : isDownloading ? (
                                  <button
                                    type="button"
                                    onClick={() => handleCancelDownload(m.id)}
                                    disabled={isCancellingThis}
                                    className="flex items-center gap-1 px-3 py-1 rounded-xl text-xs font-medium bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer"
                                  >
                                    <X size={12} className="text-red-300" />
                                    <span>{isCancellingThis ? 'Annullamento...' : 'Interrompi'}</span>
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    onClick={() => handleDownloadModel(m.id)}
                                    disabled={actionInProgress !== null}
                                    className="flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-medium bg-white/[0.10] hover:bg-white/[0.18] text-white border border-white/12 transition-all cursor-pointer shadow-liquid-sm disabled:opacity-50"
                                  >
                                    <Download size={12} />
                                    <span>Scarica</span>
                                  </button>
                                )}
                              </div>
                            </div>

                            {isDownloading && (
                              <div className="w-full pt-1.5 border-t border-white/[0.08] space-y-1 animate-fade-in">
                                <div className="flex justify-between items-center text-[10px] font-mono">
                                  <span className="flex items-center gap-1.5 text-blue-300 font-medium truncate max-w-[80%]">
                                    <Loader2 size={11} className="animate-spin flex-shrink-0" />
                                    <span className="truncate">{m.stage || "Download in corso..."}</span>
                                  </span>
                                  <span className="font-bold text-white">{m.progress || 0}%</span>
                                </div>
                                <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 rounded-full transition-all duration-300"
                                    style={{ width: `${Math.min(100, Math.max(4, m.progress || 0))}%` }}
                                  />
                                </div>
                              </div>
                            )}

                            {m.status === 'error' && !isDownloading && (
                              <div className="w-full pt-1.5 border-t border-red-500/20 text-xs text-red-300 flex items-center justify-between gap-2 animate-fade-in">
                                <div className="flex items-center gap-1.5 truncate max-w-[75%]">
                                  <AlertCircle size={12} className="text-red-400 shrink-0" />
                                  <span className="truncate text-[11px]">{m.error || "Errore durante il download."}</span>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => handleDownloadModel(m.id)}
                                  className="px-2 py-0.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 text-red-200 text-[10px] font-medium shrink-0 cursor-pointer"
                                >
                                  Riprova
                                </button>
                              </div>
                            )}
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 3: FILTRO VOCALE (VAD) & FORMATO AUDIO
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'audio' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-semibold tracking-tight text-white">Filtro Vocale (VAD)</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    Regola la sensibilità con cui viene isolata la voce del docente durante la lezione, escludendo silenzi e brusio d'aula.
                  </p>
                </div>

                {/* Selezione a 3 Opzioni Minimal */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {/* Opzione 1: Ampia */}
                  <button
                    type="button"
                    onClick={() => handleVadChange(1)}
                    className={clsx(
                      'p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none',
                      vadSensitivity === 1
                        ? 'bg-white/[0.09] border-white/30 shadow-liquid-sm'
                        : 'bg-white/[0.02] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/[0.14]'
                    )}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <div className={clsx(
                          'w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
                          vadSensitivity === 1 ? 'bg-white text-black' : 'bg-white/[0.06] text-white/70'
                        )}>
                          <Volume2 size={15} strokeWidth={2} />
                        </div>
                        <div className={clsx(
                          'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                          vadSensitivity === 1 ? 'border-white bg-white text-black' : 'border-white/20'
                        )}>
                          {vadSensitivity === 1 && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-xs font-semibold text-white tracking-tight block">
                          Ampia
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Sensibilità alta per cattedre distanti o aule grandi. Cattura anche spiegazioni a voce moderata.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] text-white/40 block">
                        Per aule grandi o docente distante
                      </span>
                    </div>
                  </button>

                  {/* Opzione 2: Standard */}
                  <button
                    type="button"
                    onClick={() => handleVadChange(2)}
                    className={clsx(
                      'p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none relative',
                      vadSensitivity === 2
                        ? 'bg-white/[0.09] border-white/30 shadow-liquid-sm'
                        : 'bg-white/[0.02] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/[0.14]'
                    )}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <div className={clsx(
                          'w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
                          vadSensitivity === 2 ? 'bg-white text-black' : 'bg-white/[0.06] text-white/70'
                        )}>
                          <Sparkles size={15} strokeWidth={2} />
                        </div>
                        <div className={clsx(
                          'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                          vadSensitivity === 2 ? 'border-white bg-white text-black' : 'border-white/20'
                        )}>
                          {vadSensitivity === 2 && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-xs font-semibold text-white tracking-tight block">
                          Standard
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Bilanciamento ottimale per la maggior parte delle lezioni. Isola la voce ed esclude il brusio.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] text-white/40 block">
                        Per la maggior parte delle lezioni
                      </span>
                    </div>
                  </button>

                  {/* Opzione 3: Diretta */}
                  <button
                    type="button"
                    onClick={() => handleVadChange(3)}
                    className={clsx(
                      'p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none',
                      vadSensitivity === 3
                        ? 'bg-white/[0.09] border-white/30 shadow-liquid-sm'
                        : 'bg-white/[0.02] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/[0.14]'
                    )}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <div className={clsx(
                          'w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
                          vadSensitivity === 3 ? 'bg-white text-black' : 'bg-white/[0.06] text-white/70'
                        )}>
                          <Shield size={15} strokeWidth={2} />
                        </div>
                        <div className={clsx(
                          'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                          vadSensitivity === 3 ? 'border-white bg-white text-black' : 'border-white/20'
                        )}>
                          {vadSensitivity === 3 && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-xs font-semibold text-white tracking-tight block">
                          Diretta
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Filtro rigoroso per prime file o microfoni vicini. Esclude mormorii e voci dei compagni.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] text-white/40 block">
                        Per prime file o microfoni vicini
                      </span>
                    </div>
                  </button>
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 4: SISTEMA & PRIVACY
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'system' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-bold text-white">Sistema & Privacy</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    SbobinAI opera al 100% in locale sul tuo computer. Nessun file né dato audio viene mai inviato all'esterno.
                  </p>
                </div>

                {/* Sezione Aggiornamenti */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.12] space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h4 className="text-sm font-bold text-white">Aggiornamenti</h4>
                      <p className="text-[11px] text-white/50 mt-0.5">
                        {updater.supported
                          ? "Mantieni SbobinAI aggiornato con le ultime correzioni e miglioramenti."
                          : "Gli aggiornamenti automatici sono disponibili solo nell'app desktop."}
                      </p>
                    </div>
                    <span className="text-[10px] font-mono font-medium px-2.5 py-1 rounded-full bg-white/[0.08] text-white/70 border border-white/10 flex-shrink-0">
                      v{updater.currentVersion || '0.0.14'}
                    </span>
                  </div>

                  {/* Toggle Aggiornamento Automatico */}
                  <div className="flex items-center justify-between p-3 rounded-2xl bg-white/[0.04] border border-white/[0.08]">
                    <div className="space-y-0.5 pr-4">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-white">Aggiornamento automatico</span>
                        <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          Consigliato
                        </span>
                      </div>
                      <p className="text-[11px] text-white/50 leading-relaxed">
                        Verifica e installa automaticamente gli aggiornamenti ad ogni avvio dell'app.
                      </p>
                    </div>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={settings.autoUpdate}
                      disabled={!updater.supported}
                      onClick={() => {
                        const next = !settings.autoUpdate;
                        void updateSettings({ autoUpdate: next });
                        showNotification(
                          next
                            ? 'Aggiornamento automatico attivato (predefinito).'
                            : 'Aggiornamento automatico disattivato.',
                          'info'
                        );
                      }}
                      className={clsx(
                        'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none disabled:opacity-40 disabled:cursor-not-allowed',
                        settings.autoUpdate ? 'bg-blue-500' : 'bg-white/20'
                      )}
                    >
                      <span
                        aria-hidden="true"
                        className={clsx(
                          'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-md ring-0 transition duration-200 ease-in-out',
                          settings.autoUpdate ? 'translate-x-5' : 'translate-x-0'
                        )}
                      />
                    </button>
                  </div>

                  {/* Stato / Azione Aggiornamento */}
                  {updater.status === 'available' ? (
                    <div className="flex items-center justify-between gap-3 pt-1 animate-fade-in">
                      <span className="text-xs text-blue-300 font-semibold">
                        Versione {updater.version} disponibile
                      </span>
                      <button
                        type="button"
                        onClick={() => void updater.install()}
                        className="px-4 py-2 rounded-full bg-blue-500 hover:bg-blue-400 text-xs font-semibold text-white cursor-pointer transition-all shadow-liquid-sm flex items-center gap-1.5"
                      >
                        <Download size={13} />
                        <span>Aggiorna e Riavvia</span>
                      </button>
                    </div>
                  ) : updater.status === 'downloading' || updater.status === 'installing' ? (
                    <div className="pt-1 space-y-2 animate-fade-in">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-white/80 font-medium">
                          {updater.status === 'installing'
                            ? 'Installazione e riavvio in corso...'
                            : `Download aggiornamento (${updater.version || ''})...`}
                        </span>
                        <span className="font-mono font-bold text-blue-300">
                          {updater.progress}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 rounded-full transition-all duration-200"
                          style={{ width: `${Math.max(4, updater.progress)}%` }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3 flex-wrap pt-1">
                      <button
                        type="button"
                        disabled={!updater.supported || updater.status === 'checking'}
                        onClick={() => void updater.checkNow()}
                        className="px-4 py-1.5 rounded-full bg-white/[0.10] hover:bg-white/[0.18] border border-white/[0.12] text-xs font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors flex items-center gap-2"
                      >
                        {updater.status === 'checking' && <Loader2 size={12} className="animate-spin text-blue-400" />}
                        <span>{updater.status === 'checking' ? 'Controllo in corso...' : 'Controlla aggiornamenti'}</span>
                      </button>
                      {updater.status === 'uptodate' && (
                        <span className="text-[11px] text-emerald-300 font-medium flex items-center gap-1.5">
                          <Check size={13} />
                          <span>Sei all'ultima versione</span>
                        </span>
                      )}
                      {updater.status === 'error' && (
                        <span className="text-[11px] text-red-300">
                          {updater.error?.includes('release JSON') || updater.error?.includes('404')
                            ? 'Nessun nuovo aggiornamento disponibile.'
                            : `Errore: ${updater.error}`}
                        </span>
                      )}
                    </div>
                  )}
                </div>

                {/* Sezione Reset Onboarding */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-3">
                  <div>
                    <h4 className="text-xs font-bold text-white">Riconfigurazione</h4>
                    <p className="text-[11px] text-white/50 mt-0.5 leading-relaxed">
                      Ripercorri la procedura iniziale per riconfigurare la cartella delle lezioni e i modelli.
                    </p>
                  </div>

                  <button
                    type="button"
                    data-reset-wizard-btn="true"
                    onClick={handleResetWizard}
                    className={clsx(
                      "flex items-center gap-2 px-4 h-9 rounded-2xl text-xs font-semibold border transition-all cursor-pointer whitespace-nowrap select-none",
                      confirmReset
                        ? "bg-red-500 text-white border-red-500 shadow-liquid-sm"
                        : "bg-white/[0.08] hover:bg-white/[0.16] text-white/80 border-white/14"
                    )}
                  >
                    <RotateCcw size={13} className={clsx("flex-shrink-0", confirmReset && "animate-spin text-white")} />
                    <span>{confirmReset ? 'Sei sicuro? Clicca per riavviare' : 'Riavvia configurazione guidata'}</span>
                  </button>
                </div>

                {/* ── Sezione Disinstallazione SbobinAI ── */}
                <div className="p-4 rounded-3xl bg-red-500/[0.04] border border-red-500/20 shadow-liquid-sm space-y-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <Trash2 size={14} className="text-red-400" />
                      <h4 className="text-xs font-bold text-red-300">Disinstalla SbobinAI</h4>
                    </div>
                    <p className="text-[11px] text-white/50 mt-1 leading-relaxed">
                      Rimuove completamente SbobinAI dal computer eliminando il motore locale, i modelli IA scaricati e la cache di sistema.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={() => {
                      setDeleteStorageFolder(false);
                      setShowUninstallModal(true);
                    }}
                    className="flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 text-red-200 hover:text-white transition-all shadow-liquid-sm cursor-pointer"
                  >
                    <Trash2 size={13} />
                    <span>Disinstalla SbobinAI</span>
                  </button>
                </div>
              </div>
            )}

          </div>
        </div>

        {/* ── Modal Footer (macOS Native Style: senza "Python Backend" e con autosave pulito) ── */}
        <div className="h-14 px-6 border-t border-white/[0.08] flex items-center justify-between flex-shrink-0 bg-black/20">
          <div className="text-[11px] text-white/40">
            Le modifiche vengono salvate automaticamente.
          </div>

          <div>
            <button
              onClick={onClose}
              className="px-5 py-1.5 rounded-2xl text-xs font-semibold bg-white/[0.12] hover:bg-white/[0.20] border border-white/[0.18] text-white transition-all shadow-liquid-sm cursor-pointer"
            >
              Chiudi
            </button>
          </div>
        </div>

        {/* ── Floating Tooltip Copiato sopra il puntatore ── */}
        {copyTooltip && (
          <div
            style={{ left: copyTooltip.x, top: copyTooltip.y }}
            className="fixed pointer-events-none z-[100] -translate-x-1/2 -translate-y-full -mt-2.5"
          >
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white text-black text-xs font-semibold shadow-2xl border border-white/40 animate-slide-up select-none">
              <Check size={12} strokeWidth={2.5} className="text-emerald-600" />
              <span>Copiato!</span>
            </div>
          </div>
        )}

        {/* ── Modal di Conferma Disinstallazione Completa ── */}
        <AnimatePresence>
          {showUninstallModal && (
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.18 }}
              className="absolute inset-0 z-50 bg-[#0B0F19]/95 backdrop-blur-xl flex flex-col items-center justify-center p-6 text-center select-none"
            >
              {!isUninstalling ? (
                <div className="max-w-md w-full space-y-5 animate-scale-up text-left">
                  <div className="flex items-center gap-3 border-b border-white/10 pb-3">
                    <div className="w-10 h-10 rounded-2xl bg-red-500/15 border border-red-500/30 flex items-center justify-center text-red-400 shadow-liquid-sm flex-shrink-0">
                      <Trash2 size={20} />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-white tracking-tight">
                        Disinstallazione di SbobinAI
                      </h3>
                      <p className="text-xs text-white/50 mt-0.5">
                        Scegli come gestire la cartella con le tue lezioni.
                      </p>
                    </div>
                  </div>

                  {/* ── 2 OPZIONI CHIARE E MINIMALI ── */}
                  <div className="space-y-2.5">
                    {/* Opzione 1: Conserva i file personali */}
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => setDeleteStorageFolder(false)}
                      className={clsx(
                        "p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3",
                        !deleteStorageFolder
                          ? "bg-white/[0.08] border-white/30 shadow-liquid-sm"
                          : "bg-white/[0.02] border-white/[0.08] hover:bg-white/[0.05]"
                      )}
                    >
                      <div className={clsx(
                        "w-4 h-4 rounded-full border mt-0.5 flex items-center justify-center flex-shrink-0 transition-colors",
                        !deleteStorageFolder ? "border-white bg-white" : "border-white/30 bg-transparent"
                      )}>
                        {!deleteStorageFolder && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                      </div>
                      <div className="space-y-0.5 flex-1">
                        <span className="text-xs font-semibold text-white block">
                          Conserva lezioni e registrazioni
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Rimuove l'app e i modelli IA. I tuoi audio, le trascrizioni e gli appunti rimarranno al sicuro sul disco.
                        </p>
                      </div>
                    </div>

                    {/* Opzione 2: Pulizia totale */}
                    <div
                      role="button"
                      tabIndex={0}
                      onClick={() => setDeleteStorageFolder(true)}
                      className={clsx(
                        "p-3.5 rounded-2xl border transition-all cursor-pointer flex items-start gap-3",
                        deleteStorageFolder
                          ? "bg-red-500/15 border-red-500/40 shadow-liquid-sm"
                          : "bg-white/[0.02] border-white/[0.08] hover:bg-white/[0.05]"
                      )}
                    >
                      <div className={clsx(
                        "w-4 h-4 rounded-full border mt-0.5 flex items-center justify-center flex-shrink-0 transition-colors",
                        deleteStorageFolder ? "border-red-400 bg-red-500" : "border-white/30 bg-transparent"
                      )}>
                        {deleteStorageFolder && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                      </div>
                      <div className="space-y-0.5 flex-1">
                        <span className="text-xs font-semibold text-red-300 block">
                          Elimina anche la cartella delle lezioni
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Rimuove l'app, i modelli IA e cancella definitivamente tutte le lezioni, gli audio e gli appunti.
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* ── Pulsanti Azione (Annulla fisso a sinistra, Disinstalla a destra) ── */}
                  <div className="flex items-center justify-between pt-3 border-t border-white/10">
                    <button
                      type="button"
                      onClick={() => setShowUninstallModal(false)}
                      className="px-4 py-2 rounded-2xl text-xs font-semibold bg-white/[0.08] hover:bg-white/[0.14] border border-white/12 text-white/80 transition-all cursor-pointer"
                    >
                      Annulla
                    </button>
                    <button
                      type="button"
                      onClick={handleUninstallApp}
                      className={clsx(
                        "flex items-center gap-2 px-5 py-2 rounded-2xl text-xs font-semibold text-white shadow-liquid-sm transition-all cursor-pointer",
                        deleteStorageFolder
                          ? "bg-red-600 hover:bg-red-700 border border-red-400/50"
                          : "bg-red-500 hover:bg-red-600 border border-red-400/30"
                      )}
                    >
                      {deleteStorageFolder ? <AlertTriangle size={13} /> : <Trash2 size={13} />}
                      <span>Disinstalla</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="max-w-sm w-full space-y-4 animate-fade-in flex flex-col items-center">
                  <div className="relative">
                    <div className="w-16 h-16 rounded-full border-4 border-red-500/20 border-t-red-500 animate-spin" />
                    <Trash2 size={22} className="absolute inset-0 m-auto text-red-400" />
                  </div>
                  <div className="space-y-1 text-center">
                    <h3 className="text-sm font-bold text-white">
                      {deleteStorageFolder ? 'Pulizia totale in corso...' : 'Disinstallazione in corso...'}
                    </h3>
                    <p className="text-xs text-white/50 leading-relaxed">
                      {deleteStorageFolder
                        ? 'Eliminazione lezioni, audio, modelli AI, pulizia ambiente Python e cancellazione di ogni traccia.'
                        : 'Eliminazione dei modelli AI, pulizia dell\'ambiente Python e disinstallazione dell\'applicazione.'}
                    </p>
                    <p className="text-[11px] text-white/40 pt-2">
                      L'applicazione si chiuderà automaticamente al termine.
                    </p>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
}
