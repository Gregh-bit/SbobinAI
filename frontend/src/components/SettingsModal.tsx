import { useEffect, useState, useCallback, useRef } from 'react';
import { 
  X, Folder, Cpu, Mic, HardDrive, Check, 
  RotateCcw, Sliders, Trash2, Download, Loader2,
  RefreshCw, CheckCircle2, ShieldCheck, Database, FileAudio, 
  Layers, AlertTriangle, Sparkles, Volume2, Shield, Copy
} from 'lucide-react';
import { clsx } from 'clsx';
import { useSettings } from '../lib/SettingsContext';
import { pickFolder } from '../lib/tauri';
import { osBadge, fileManagerName, normalizeDisplayPath } from '../lib/platform';
import { useUpdater } from '../lib/useUpdater';
import { api } from '../lib/api';
import type { ModelsStatusResponse, ModelInfo } from '../types';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'storage' | 'models' | 'audio' | 'system';
type ModelFilterType = 'system_core' | 'llm';

const defaultLlms: ModelInfo[] = [
  {
    id: 'mlx-community/Qwen3.5-4B-MLX-4bit',
    name: 'Qwen 3.5 4B (MLX 4-bit)',
    type: 'llm',
    approx_size: '~2.8 GB',
    description: 'Modello linguistico consigliato: compatto, veloce e ottimizzato con MLX 4-bit per Apple Silicon. Basso consumo di memoria RAM (~2.8 GB).',
    downloaded: false,
    status: 'idle',
    disk_size_mb: null,
  },
  {
    id: 'mlx-community/Qwen3.5-2B-MLX-4bit',
    name: 'Qwen 3.5 2B (MLX 4-bit ultraleggero)',
    type: 'llm',
    approx_size: '~1.4 GB',
    description: 'Versione ultraleggera a bassissimo impatto di memoria RAM (~1.4 GB). Ideale per MacBook con 8 GB di memoria unificata.',
    downloaded: false,
    status: 'idle',
    disk_size_mb: null,
  },
  {
    id: 'mlx-community/Qwen3.5-8B-MLX-4bit',
    name: 'Qwen 3.5 8B (MLX 4-bit alta capacità)',
    type: 'llm',
    approx_size: '~5.5 GB',
    description: 'Modello ad altissima capacità di ragionamento e sintesi concettuale (~5.5 GB in RAM). Consigliato per MacBook con 16 GB o più.',
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
  const [modelFilter, setModelFilter] = useState<ModelFilterType>('system_core');

  // Form State
  const [storagePath, setStoragePath] = useState(settings.storagePath);
  const [whisperModel, setWhisperModel] = useState(settings.whisperModel);
  const [llmModel, setLlmModel] = useState(settings.llmModel || 'mlx-community/Qwen3.5-4B-MLX-4bit');
  const [embeddingModel, setEmbeddingModel] = useState(settings.embeddingModel || 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ');
  const [vadSensitivity, setVadSensitivity] = useState(settings.vadSensitivity);
  const [audioFormat, setAudioFormat] = useState<'m4a' | 'wav'>(settings.audioFormat || 'm4a');
  const [isCompressingArchive, setIsCompressingArchive] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);
  const [confirmDeleteModelId, setConfirmDeleteModelId] = useState<string | null>(null);

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

  const handleCleanOrphans = async () => {
    try {
      setIsCleaningOrphans(true);
      const res = await api.settings.cleanOrphans();
      showNotification(
        `Pulizia completata: liberati ${res.freed_mb} MB (${res.deleted_audio_files.length} audio orfani rimossi)!`,
        'success'
      );
      await fetchStorageStats();
    } catch (err: any) {
      showNotification(err?.message || 'Errore durante la pulizia dei file orfani.', 'error');
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

    const timer = setInterval(() => {
      fetchModelsStatus(true);
    }, 1000);
    return () => clearInterval(timer);
  }, [isOpen, modelsStatus, actionInProgress, fetchModelsStatus]);

  // Gestione tasto ESC
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

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

  useEffect(() => {
    if (settings.audioFormat) {
      setAudioFormat(settings.audioFormat);
    }
  }, [settings.audioFormat]);

  const handleVadChange = (newVal: number) => {
    setVadSensitivity(newVal);
    updateSettings({ vadSensitivity: newVal });
  };

  const handleAudioFormatChange = async (newFormat: 'm4a' | 'wav') => {
    setAudioFormat(newFormat);
    try {
      await updateSettings({ audioFormat: newFormat });
      showNotification(
        newFormat === 'm4a'
          ? 'Formato impostato su M4A compresso AAC (~18 MB/h, -85% di spazio sul disco).'
          : 'Formato impostato su WAV originale PCM non compresso (~115 MB/h).',
        'success'
      );
    } catch (err: any) {
      showNotification(`Errore salvataggio formato audio: ${err?.message || err}`, 'error');
    }
  };

  const handleCompressArchive = async () => {
    try {
      setIsCompressingArchive(true);
      const res = await api.settings.compressArchive();
      showNotification(res.message, res.converted_count > 0 ? 'success' : 'info');
      await fetchStorageStats();
    } catch (err: any) {
      showNotification(err?.message || 'Errore durante la compressione dell\'archivio audio.', 'error');
    } finally {
      setIsCompressingArchive(false);
    }
  };

  const handleResetWizard = () => {
    if (!confirmReset) {
      setConfirmReset(true);
      setTimeout(() => setConfirmReset(false), 4000);
      return;
    }
    resetSetup();
    onClose();
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
      desc: 'Pesi neurali quantizzati 4-bit per Whisper, Qwen LLM ed Embedding RAG.',
      details: `${downloadedModelsCount} ${downloadedModelsCount === 1 ? 'modello installato' : 'modelli installati'}`,
    },
    {
      id: 'audio',
      label: 'Registrazioni Master Audio',
      shortLabel: 'Audio Lezioni',
      sizeMb: audioMb,
      color: 'bg-[#0A84FF]',
      dotColor: 'bg-[#0A84FF]',
      badgeBg: 'bg-[#0A84FF]/15',
      badgeBorder: 'border-[#0A84FF]/30',
      badgeText: 'text-[#5AC8FA]',
      icon: FileAudio,
      path: normalizeDisplayPath(`${storagePath}/audio`),
      desc: 'File master ad alta risoluzione 16-bit PCM 16kHz per trascrizione e ascolto.',
      details: 'Audio fedele non compresso',
    },
    {
      id: 'sqlite',
      label: 'Database Relazionale',
      shortLabel: 'Database SQLite',
      sizeMb: sqliteMb,
      color: 'bg-[#30D158]',
      dotColor: 'bg-[#30D158]',
      badgeBg: 'bg-[#30D158]/15',
      badgeBorder: 'border-[#30D158]/30',
      badgeText: 'text-[#6EE7B7]',
      icon: Database,
      path: normalizeDisplayPath(`${storagePath}/sqlite/sbobinai.db`),
      desc: 'Corsi, trascrizioni parola per parola con timestamp, flashcard e note.',
      details: 'Transazionale WAL / VACUUM',
    },
    {
      id: 'lancedb',
      label: 'Indice Vettoriale RAG',
      shortLabel: 'LanceDB',
      sizeMb: lancedbMb,
      color: 'bg-[#FF9F0A]',
      dotColor: 'bg-[#FF9F0A]',
      badgeBg: 'bg-[#FF9F0A]/15',
      badgeBorder: 'border-[#FF9F0A]/30',
      badgeText: 'text-[#FDBA74]',
      icon: Layers,
      path: normalizeDisplayPath(`${storagePath}/lancedb`),
      desc: 'Embeddings a 1024 dimensioni per ricerca semantica e quiz intelligente.',
      details: 'Vector DB zero-copy locale',
    },
    {
      id: 'uploads',
      label: 'Materiali & Documenti',
      shortLabel: 'Uploads',
      sizeMb: uploadsMb,
      color: 'bg-[#FF375F]',
      dotColor: 'bg-[#FF375F]',
      badgeBg: 'bg-[#FF375F]/15',
      badgeBorder: 'border-[#FF375F]/30',
      badgeText: 'text-[#FDA4AF]',
      icon: Folder,
      path: normalizeDisplayPath(`${storagePath}/uploads`),
      desc: 'Slide PDF, appunti in PDF e allegati di supporto caricati per i corsi.',
      details: 'Dispense universitarie',
    },
  ];

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in select-none"
      onClick={onClose}
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
              <p className="text-[11px] text-white/45">Preferenze locali del quaderno universitario</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center text-white/40 hover:text-white/90 hover:bg-white/[0.08] transition-colors cursor-pointer"
            title="Chiudi (Esc)"
          >
            <X size={15} strokeWidth={2} />
          </button>
        </div>

        {/* ── Modal Layout: Sidebar a sinistra + Contenuto a destra ── */}
        <div className="flex-1 flex overflow-hidden">
          
          {/* ── Sidebar Categorie (Design coerente senza arcobaleno) ── */}
          <div className="w-56 border-r border-white/[0.08] liquid-glass-subtle p-3 flex flex-col justify-between flex-shrink-0">
            <div className="space-y-1">
              {/* Tab 1: Archiviazione */}
              <button
                onClick={() => setActiveTab('storage')}
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

              {/* Tab 2: Modelli */}
              <button
                onClick={() => setActiveTab('models')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'models'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <Cpu size={15} className="text-white/70" />
                <span>Modelli</span>
              </button>

              {/* Tab 3: Motore Audio */}
              <button
                onClick={() => setActiveTab('audio')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'audio'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <Mic size={15} className="text-white/70" />
                <span>Motore Audio & VAD</span>
              </button>

              {/* Tab 4: Sistema & Privacy */}
              <button
                onClick={() => setActiveTab('system')}
                className={clsx(
                  'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-2xl text-xs font-medium transition-all text-left cursor-pointer',
                  activeTab === 'system'
                    ? 'bg-white/[0.14] text-white shadow-liquid-sm border border-white/[0.16]'
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent'
                )}
              >
                <ShieldCheck size={15} className="text-white/70" />
                <span>Privacy & Sistema</span>
              </button>
            </div>

            {/* Bottom Sidebar: Subtle macOS Version Note */}
            <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center">
              <span className="text-[10px] text-white/35 font-mono">
                SbobinAI per macOS
              </span>
            </div>
          </div>

          {/* ── Main Tab Content Panel con scroll fluido e spaziatura corretta ── */}
          <div className="flex-1 p-6 pb-12 overflow-y-auto space-y-6 [scrollbar-gutter:stable]">
            
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
                  <h3 className="text-sm font-bold text-white">Archiviazione & Memoria Locale</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    Monitora lo spazio occupato su disco da registrazioni master, database, indici vettoriali e modelli di intelligenza artificiale.
                  </p>
                </div>

                {/* ── Apple-style Storage Overview Card ── */}
                <div className="p-5 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-white/45 block">
                        Utilizzo Spazio Disco SbobinAI
                      </span>
                      <div className="flex items-baseline gap-2.5 mt-0.5">
                        <h4 className="text-2xl font-bold font-mono text-white tracking-tight">
                          {formatSize(combinedTotalMb)}
                        </h4>
                        <span className="text-xs text-white/50">
                          totali impiegati sul Mac
                        </span>
                      </div>
                    </div>

                    {/* Tasti Azione: Ottimizza & Pulisci + Comprimi Archivio + Ricalcola */}
                    <div className="flex items-center gap-2 self-start sm:self-center">
                      <button
                        type="button"
                        onClick={handleCompressArchive}
                        disabled={isCompressingArchive}
                        className={clsx(
                          "px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm whitespace-nowrap",
                          isCompressingArchive
                            ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/30 cursor-wait"
                            : "bg-white/[0.08] hover:bg-white/[0.14] border-white/14 text-white/80 hover:text-white"
                        )}
                        title="Converte tutte le registrazioni WAV non compresse in M4A AAC (-85% di spazio sul disco)"
                      >
                        {isCompressingArchive ? (
                          <>
                            <Loader2 size={13} className="animate-spin text-emerald-300" />
                            <span>Compressione...</span>
                          </>
                        ) : (
                          <>
                            <FileAudio size={13} className="text-emerald-400" />
                            <span>Comprimi Archivio</span>
                          </>
                        )}
                      </button>

                      <button
                        type="button"
                        onClick={handleCleanOrphans}
                        disabled={isCleaningOrphans}
                        className={clsx(
                          "px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm whitespace-nowrap",
                          isCleaningOrphans
                            ? "bg-amber-500/20 text-amber-200 border-amber-500/30 cursor-wait"
                            : "bg-white/[0.08] hover:bg-white/[0.14] border-white/14 text-white/80 hover:text-white"
                        )}
                        title="Elimina vecchi audio orfani e compatta il database (VACUUM)"
                      >
                        {isCleaningOrphans ? (
                          <>
                            <Loader2 size={13} className="animate-spin text-amber-300" />
                            <span>Pulizia in corso...</span>
                          </>
                        ) : (
                          <>
                            <RotateCcw size={13} className="text-amber-400" />
                            <span>Ottimizza & Pulisci</span>
                          </>
                        )}
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          fetchStorageStats();
                          fetchModelsStatus(true);
                        }}
                        disabled={loadingModels}
                        className="px-3 py-1.5 rounded-xl liquid-glass-pill border border-white/12 hover:border-white/22 text-white/70 hover:text-white transition-all cursor-pointer text-xs flex items-center gap-1.5 whitespace-nowrap"
                        title="Ricalcola spazio disco"
                      >
                        <RefreshCw size={13} className={clsx(loadingModels && "animate-spin text-blue-400")} />
                        <span>Ricalcola</span>
                      </button>
                    </div>
                  </div>

                  {/* ── Barra Orizzontale Multicolore Stile Apple ── */}
                  <div className="space-y-3">
                    <div className="w-full h-3.5 rounded-full bg-white/[0.06] p-0.5 flex gap-1 overflow-hidden border border-white/[0.12]">
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

                    {/* ── Legenda Stile macOS Storage ── */}
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-1 text-[11px]">
                      {storageCategories.map((cat) => (
                        <div key={cat.id} className="flex items-center gap-1.5 whitespace-nowrap">
                          <span className={clsx('w-2.5 h-2.5 rounded-full flex-shrink-0 shadow-sm', cat.dotColor)} />
                          <span className="text-white/80 font-medium">{cat.shortLabel}:</span>
                          <span className="font-mono text-white/60">{formatSize(cat.sizeMb)}</span>
                          <span className="text-[10px] text-white/35 font-mono">({formatPct(cat.sizeMb)})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* ── Griglia Dati: Ripartizione Dettagliata (con cestino su hover per Modelli IA) ── */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                      Ripartizione Dettagliata per Categoria
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
                TAB 2: MODELLI (DISCO & RETI NEURALI)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'models' && (
              <div className="space-y-6 animate-fade-in">
                
                {/* ── Header Tab Modelli ── */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-2 border-b border-white/[0.08]">
                  <div>
                    <h3 className="text-sm font-bold text-white">Modelli di Intelligenza Artificiale</h3>
                    <p className="text-xs text-white/50 mt-0.5 leading-relaxed">
                      Motori vocali, vettoriali e modelli linguistici eseguiti al 100% in locale sul tuo Mac con accelerazione Apple Silicon MLX.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => fetchModelsStatus()}
                    disabled={loadingModels}
                    className="self-start sm:self-center px-3 py-1.5 rounded-xl liquid-glass-pill border border-white/12 hover:border-white/22 text-white/70 hover:text-white transition-all cursor-pointer disabled:opacity-50 flex items-center gap-1.5 text-xs flex-shrink-0"
                    title="Ricarica stato da disco"
                  >
                    <RefreshCw size={13} className={clsx(loadingModels && "animate-spin text-blue-400")} />
                    <span>Ricarica stato</span>
                  </button>
                </div>

                {/* ── Barra Filtri Modelli (Motori Base vs LLM) ── */}
                <div className="flex items-center justify-between border-b border-white/[0.08] pb-2">
                  <div className="flex items-center gap-1.5 p-1 rounded-2xl liquid-glass-subtle border border-white/[0.08]">
                    <button
                      type="button"
                      onClick={() => setModelFilter('system_core')}
                      className={clsx(
                        'px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer',
                        modelFilter === 'system_core'
                          ? 'bg-white/[0.18] text-white shadow-liquid-sm border border-white/[0.15]'
                          : 'text-white/50 hover:text-white'
                      )}
                    >
                      Motori di Sistema Base (Audio & Memoria)
                    </button>
                    <button
                      type="button"
                      onClick={() => setModelFilter('llm')}
                      className={clsx(
                        'px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer',
                        modelFilter === 'llm'
                          ? 'bg-white/[0.18] text-white shadow-liquid-sm border border-white/[0.15]'
                          : 'text-white/50 hover:text-white'
                      )}
                    >
                      Modelli Linguistici LLM (Quaderno & Chat)
                    </button>
                  </div>
                </div>

                {/* ── LISTA MOTORI DI SISTEMA BASE (Whisper + Qwen3-Embedding) ── */}
                {modelFilter === 'system_core' && (
                  <div className="space-y-4">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                        Componenti di Sistema Obbligatori{osBadge ? ` — ${osBadge}` : ''}
                      </span>
                      <p className="text-[11px] text-white/45 mt-1 leading-relaxed">
                        Questi due motori costituiscono il nucleo stabile di SbobinAI: Whisper Large Turbo per la trascrizione in memoria e Qwen3-Embedding per la memoria RAG (1024 dim) su LanceDB.
                      </p>
                    </div>

                    <div className="space-y-3">
                      {(() => {
                        const coreWhisper = modelsStatus?.whisper?.find((m) => m.id === 'large-v3-turbo-q4') || {
                            id: 'large-v3-turbo-q4',
                            name: 'Whisper Large-v3-Turbo Q4 (locale)',
                            type: 'whisper' as const,
                            approx_size: '~580 MB',
                            description: 'Motore Vocale di sistema obbligatorio: trascrizione ad alta precisione 4-bit (~580 MB). Fedeltà massima e consumo nullo in pausa.',
                            downloaded: false,
                            status: 'idle' as const,
                            disk_size_mb: null,
                          };

                          const coreEmbedding = modelsStatus?.embedding?.find((m) => m.id === 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ') || {
                            id: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
                            name: 'Qwen3 Embedding 0.6B 4-bit DWQ',
                            type: 'embedding' as const,
                            approx_size: '~300 MB',
                            description: 'Motore Memoria RAG di sistema obbligatorio: architettura Qwen3 nativa, 1024 dimensioni (~300 MB in RAM). Indicizzazione ad altissima velocità e ricerca ibrida su LanceDB.',
                            downloaded: false,
                            status: 'idle' as const,
                            disk_size_mb: null,
                          };

                          const coreItems = [coreWhisper, coreEmbedding];

                          return coreItems.map((m) => {
                            const isDownloading = m.status === 'downloading' || actionInProgress === m.id;
                            const isCancellingThis = actionInProgress === `cancel-${m.id}`;
                            const isMic = m.type === 'whisper';

                            return (
                              <div
                                key={m.id}
                                className={clsx(
                                  'p-4 rounded-3xl border transition-all space-y-3',
                                  m.downloaded
                                    ? 'liquid-glass border-white/[0.14] shadow-liquid-sm'
                                    : 'liquid-glass-subtle border-white/[0.08] hover:border-white/16'
                                )}
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                  <div className="flex items-start gap-3 min-w-0">
                                    <div className={clsx(
                                      'w-9 h-9 rounded-2xl flex items-center justify-center flex-shrink-0 mt-0.5',
                                      m.downloaded
                                        ? 'bg-white/[0.10] text-white border border-white/[0.14]'
                                        : 'bg-white/5 text-white/40 border border-white/10'
                                    )}>
                                      {isMic ? <Mic size={16} /> : <Database size={16} />}
                                    </div>

                                    <div className="min-w-0">
                                      <div className="flex items-center gap-2 flex-wrap">
                                        <h4 className="text-xs font-bold text-white">{m.name}</h4>
                                        <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-white/[0.06] text-white/60 border border-white/[0.10]">
                                          Componente Base
                                        </span>
                                        {m.downloaded ? (
                                          <span className="text-[10px] font-mono text-white/40">
                                            {m.disk_size_mb || m.total_mb} MB
                                          </span>
                                        ) : (
                                          <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-amber-500/15 text-amber-300 border border-amber-500/25">
                                            Da installare ({m.approx_size})
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-[11px] text-white/50 mt-1 leading-relaxed">
                                        {m.description}
                                      </p>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-2 flex-shrink-0 self-end sm:self-center">
                                    {m.downloaded ? (
                                      <>
                                        <span className="text-[11px] font-semibold text-emerald-300 bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-1 rounded-full flex items-center gap-1.5 shadow-liquid-sm">
                                          <Check size={11} strokeWidth={2.5} />
                                          <span>Operativo</span>
                                        </span>
                                        <button
                                          type="button"
                                          onClick={() => handleDownloadModel(m.id)}
                                          disabled={actionInProgress !== null}
                                          className="px-3 py-1.5 rounded-xl text-xs font-medium text-white/60 hover:text-white bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.10] transition-all cursor-pointer"
                                          title="Reinstalla o ripara i pesi del modello"
                                        >
                                          Ripara
                                        </button>
                                      </>
                                    ) : isDownloading ? (
                                      <button
                                        type="button"
                                        onClick={() => handleCancelDownload(m.id)}
                                        disabled={isCancellingThis}
                                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer"
                                      >
                                        <X size={13} className="text-red-300" />
                                        <span>{isCancellingThis ? 'Annullamento...' : 'Interrompi'}</span>
                                      </button>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={() => handleDownloadModel(m.id)}
                                        disabled={actionInProgress !== null}
                                        className="flex items-center gap-1.5 px-4 py-1.5 rounded-xl text-xs font-semibold bg-blue-500 hover:bg-blue-600 text-white shadow-liquid-sm transition-all cursor-pointer disabled:opacity-50"
                                      >
                                        <Download size={13} />
                                        <span>Installa ({m.approx_size})</span>
                                      </button>
                                    )}
                                  </div>
                                </div>

                                {isDownloading && (
                                  <div className="w-full pt-2 border-t border-white/[0.08] space-y-1.5 animate-fade-in">
                                    <div className="flex justify-between items-center text-[11px] font-mono">
                                      <span className="flex items-center gap-1.5 text-blue-300 font-medium truncate max-w-[80%]">
                                        <Loader2 size={12} className="animate-spin flex-shrink-0" />
                                        <span className="truncate">{m.stage || "Download pesi da Hugging Face..."}</span>
                                      </span>
                                      <span className="font-bold text-white text-xs">{m.progress || 0}%</span>
                                    </div>
                                    <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden relative">
                                      <div
                                        className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-emerald-400 rounded-full transition-all duration-300"
                                        style={{ width: `${Math.max(4, m.progress || 0)}%` }}
                                      />
                                    </div>
                                    <div className="flex justify-between text-[10px] font-mono text-white/50">
                                      <span>{m.downloaded_mb || 0} MB scaricati</span>
                                      <span>Totale: ~{m.total_mb || '580'} MB</span>
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          });
                        })()}
                    </div>
                  </div>
                )}

                {/* ── LISTA MODELLI LLM (mlx-lm: Qwen3.5 & Gemma 4) ── */}
                {modelFilter === 'llm' && (
                  <div className="space-y-4">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                        Modelli Linguistici LLM (esecuzione locale)
                      </span>
                      <p className="text-[11px] text-white/45 mt-1 leading-relaxed">
                        I modelli 4-bit generano capitoli di studio e rispondono nella chat. Puoi scaricare il modello desiderato, impostarlo come attivo o eliminarlo per recuperare spazio su disco.
                      </p>
                    </div>

                    <div className="space-y-3">
                      {(() => {
                        const llmList = (modelsStatus?.llm && modelsStatus.llm.length > 0) ? modelsStatus.llm : defaultLlms;
                        return llmList.map((m: ModelInfo) => {
                          const isCurrentActive = llmModel === m.id;
                          const isDownloading = m.status === 'downloading' || actionInProgress === m.id;
                          const isDeletingThis = actionInProgress === `delete-${m.id}`;
                          const isCancellingThis = actionInProgress === `cancel-${m.id}`;

                          return (
                            <div
                              key={m.id}
                              className={clsx(
                                'p-4 rounded-3xl border transition-all space-y-3',
                                isCurrentActive
                                  ? 'bg-white/[0.08] border-white/25 shadow-liquid-sm'
                                  : 'liquid-glass-subtle border-white/[0.08] hover:border-white/16'
                              )}
                            >
                              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                <div className="flex items-start gap-3 min-w-0">
                                  <div className={clsx(
                                    'w-9 h-9 rounded-2xl flex items-center justify-center flex-shrink-0 mt-0.5',
                                    m.downloaded
                                      ? 'bg-white/10 text-white/90 border border-white/15'
                                      : 'bg-white/5 text-white/40 border border-white/10'
                                  )}>
                                    <Cpu size={16} />
                                  </div>

                                  <div className="min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap">
                                      <h4 className="text-xs font-bold text-white">{m.name}</h4>
                                      {m.downloaded ? (
                                        <span className="text-[10px] font-mono text-white/40">
                                          {m.disk_size_mb || m.total_mb} MB
                                        </span>
                                      ) : (
                                        <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-white/[0.06] text-white/45 border border-white/[0.08]">
                                          Non scaricato ({m.approx_size})
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-[11px] text-white/50 mt-1 leading-relaxed">
                                      {m.description}
                                    </p>
                                  </div>
                                </div>

                                <div className="flex items-center gap-2 flex-shrink-0 self-end sm:self-center">
                                  {m.downloaded ? (
                                    <>
                                      {isCurrentActive ? (
                                        <span className="text-[11px] font-semibold text-emerald-300 bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-1 rounded-full flex items-center gap-1.5 shadow-liquid-sm">
                                          <Check size={11} strokeWidth={2.5} />
                                          <span>In uso</span>
                                        </span>
                                      ) : (
                                        <button
                                          type="button"
                                          onClick={() => handleSelectActiveLlm(m.id)}
                                          className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.12] hover:bg-white/[0.20] text-white border border-white/[0.16] transition-all cursor-pointer shadow-liquid-sm"
                                        >
                                          Attiva
                                        </button>
                                      )}

                                      {confirmDeleteModelId === m.id ? (
                                        <div className="flex items-center gap-1.5 animate-fade-in">
                                          <button
                                            type="button"
                                            onClick={() => setConfirmDeleteModelId(null)}
                                            className="px-2 py-1 rounded-lg text-[10px] text-white/60 hover:text-white"
                                          >
                                            Annulla
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => handleDeleteModel(m.id)}
                                            disabled={isDeletingThis}
                                            className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-red-500 text-white hover:bg-red-600 cursor-pointer"
                                          >
                                            {isDeletingThis ? 'Elimino...' : 'Conferma'}
                                          </button>
                                        </div>
                                      ) : (
                                        <button
                                          type="button"
                                          onClick={() => setConfirmDeleteModelId(m.id)}
                                          disabled={isDeletingThis}
                                          className="p-2 rounded-xl text-red-300 hover:text-red-200 hover:bg-red-500/20 border border-transparent hover:border-red-500/30 transition-all cursor-pointer"
                                          title="Elimina questo modello per liberare spazio"
                                        >
                                          <Trash2 size={14} />
                                        </button>
                                      )}
                                    </>
                                  ) : isDownloading ? (
                                    <button
                                      type="button"
                                      onClick={() => handleCancelDownload(m.id)}
                                      disabled={isCancellingThis}
                                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer"
                                    >
                                      <X size={13} className="text-red-300" />
                                      <span>{isCancellingThis ? 'Annullamento...' : 'Interrompi'}</span>
                                    </button>
                                  ) : (
                                    <button
                                      type="button"
                                      onClick={() => handleDownloadModel(m.id)}
                                      disabled={actionInProgress !== null}
                                      className="flex items-center gap-1.5 px-4 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.14] hover:bg-white/[0.22] text-white border border-white/[0.20] transition-all cursor-pointer shadow-liquid-sm disabled:opacity-50"
                                    >
                                      <Download size={13} />
                                      <span>Scarica ({m.approx_size})</span>
                                    </button>
                                  )}
                                </div>
                              </div>

                              {isDownloading && (
                                <div className="w-full pt-2 border-t border-white/[0.08] space-y-1.5 animate-fade-in">
                                  <div className="flex justify-between items-center text-[11px] font-mono">
                                    <span className="flex items-center gap-1.5 text-blue-300 font-medium truncate max-w-[80%]">
                                      <Loader2 size={12} className="animate-spin flex-shrink-0" />
                                      <span className="truncate">{m.stage || "Download pesi da Hugging Face..."}</span>
                                    </span>
                                    <span className="font-bold text-white text-xs">{m.progress || 0}%</span>
                                  </div>
                                  <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden relative">
                                    <div
                                      className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-emerald-400 rounded-full transition-all duration-300"
                                      style={{ width: `${Math.max(4, m.progress || 0)}%` }}
                                    />
                                  </div>
                                  <div className="flex justify-between text-[10px] font-mono text-white/50">
                                    <span>{m.downloaded_mb || 0} MB scaricati</span>
                                    <span>Totale: ~{m.total_mb || '2300'} MB</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          );
                        });
                      })()}
                    </div>
                  </div>
                )}

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 3: FILTRO ATTIVITÀ VOCALE (VAD) — DESIGN MINIMAL APPLE
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'audio' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-semibold tracking-tight text-white">Filtro Attività Vocale (VAD)</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    Regola la sensibilità con cui SbobinAI rileva la voce del docente ed esclude le pause di silenzio e il brusio d'aula durante la registrazione.
                  </p>
                </div>

                {/* Selezione a 3 Opzioni Minimal Stile macOS */}
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
                          Sensibilità elevata per cattedre distanti o aule ampie. Cattura anche le spiegazioni a voce moderata.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] font-mono text-white/40 block">
                        Raggio: Ampio · RMS 140
                      </span>
                    </div>
                  </button>

                  {/* Opzione 2: Standard (Consigliata) */}
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
                        <div className="flex items-center gap-1.5">
                          <span className="text-[9px] font-medium px-1.5 py-0.5 rounded-full bg-white/[0.10] text-white/80 border border-white/[0.14]">
                            Consigliata
                          </span>
                          <div className={clsx(
                            'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                            vadSensitivity === 2 ? 'border-white bg-white text-black' : 'border-white/20'
                          )}>
                            {vadSensitivity === 2 && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                          </div>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-xs font-semibold text-white tracking-tight block">
                          Standard
                        </span>
                        <p className="text-[11px] text-white/50 leading-relaxed">
                          Bilanciamento ottimale per la maggior parte delle lezioni. Isola la voce ed esclude il brusio circostante.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] font-mono text-white/40 block">
                        Raggio: 3–8m · RMS 190
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
                          Filtro rigoroso per prime file o microfoni vicini (AirPods o USB). Esclude mormorii e voci dei compagni.
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[10px] font-mono text-white/40 block">
                        Raggio: 1–3m · RMS 260
                      </span>
                    </div>
                  </button>
                </div>

                {/* Parametri di Sistema Stile macOS */}
                <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2.5 text-white/70">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    <span>WebRTC VAD attivo a 16 kHz Mono (PCM 16-bit LE)</span>
                  </div>
                  <span className="text-[11px] font-mono text-white/40">
                    Ring Buffer in RAM (zero logorio SSD)
                  </span>
                </div>

                {/* ══════════════════════════════════════════════════════════════════
                    SEZIONE: FORMATO REGISTRAZIONE AUDIO & SALVA-SPAZIO (5.6)
                ══════════════════════════════════════════════════════════════════ */}
                <div className="pt-2 space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                        <FileAudio size={16} className="text-white/80" />
                        Formato Registrazione Audio & Salva-Spazio
                      </h3>
                      <p className="text-xs text-white/50 mt-0.5 leading-relaxed">
                        Configura il formato con cui salvare automaticamente le lezioni. La compressione M4A riduce lo spazio dell'85% preservando la chiarezza della voce.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={handleCompressArchive}
                      disabled={isCompressingArchive}
                      className={clsx(
                        "px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer shadow-liquid-sm self-start sm:self-center whitespace-nowrap",
                        isCompressingArchive
                          ? "bg-emerald-500/20 text-emerald-200 border-emerald-500/30 cursor-wait"
                          : "bg-white/[0.06] hover:bg-white/[0.12] border-white/12 text-white/80 hover:text-white"
                      )}
                      title="Comprime tutte le registrazioni passate ancora in formato WAV liberando spazio sul disco"
                    >
                      {isCompressingArchive ? (
                        <>
                          <Loader2 size={13} className="animate-spin text-emerald-300" />
                          <span>Compressione...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles size={13} className="text-emerald-400" />
                          <span>Comprimi archivio esistente</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {/* Opzione 1: M4A Compresso (Consigliato) */}
                    <button
                      type="button"
                      onClick={() => handleAudioFormatChange('m4a')}
                      className={clsx(
                        'p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none relative',
                        audioFormat === 'm4a'
                          ? 'bg-white/[0.09] border-white/30 shadow-liquid-sm'
                          : 'bg-white/[0.02] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/[0.14]'
                      )}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <div className={clsx(
                            'w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
                            audioFormat === 'm4a' ? 'bg-white text-black' : 'bg-white/[0.06] text-white/70'
                          )}>
                            <Sparkles size={15} strokeWidth={2} />
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[9px] font-medium px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                              Consigliato · -85% Spazio
                            </span>
                            <div className={clsx(
                              'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                              audioFormat === 'm4a' ? 'border-white bg-white text-black' : 'border-white/20'
                            )}>
                              {audioFormat === 'm4a' && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                            </div>
                          </div>
                        </div>

                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-white tracking-tight block">
                            M4A Compresso (AAC Vocale 48 kbps)
                          </span>
                          <p className="text-[11px] text-white/50 leading-relaxed">
                            Occupa solo ~18 MB per ora (risparmia circa 97 MB/h). Riproduzione istantanea, sincronizzazione karaoke parola per parola e disco protetto dalla saturazione.
                          </p>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-[10px] font-mono text-white/40">
                        <span>Codec: AAC · 48 kbps Mono</span>
                        <span className="text-emerald-400 font-medium">~18 MB / ora</span>
                      </div>
                    </button>

                    {/* Opzione 2: WAV Non Compresso */}
                    <button
                      type="button"
                      onClick={() => handleAudioFormatChange('wav')}
                      className={clsx(
                        'p-4 rounded-2xl border text-left transition-all duration-200 flex flex-col justify-between cursor-pointer select-none',
                        audioFormat === 'wav'
                          ? 'bg-white/[0.09] border-white/30 shadow-liquid-sm'
                          : 'bg-white/[0.02] border-white/[0.07] hover:bg-white/[0.05] hover:border-white/[0.14]'
                      )}
                    >
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <div className={clsx(
                            'w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
                            audioFormat === 'wav' ? 'bg-white text-black' : 'bg-white/[0.06] text-white/70'
                          )}>
                            <Volume2 size={15} strokeWidth={2} />
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[9px] font-medium px-2 py-0.5 rounded-full bg-white/[0.08] text-white/70 border border-white/[0.12]">
                              Lineare PCM
                            </span>
                            <div className={clsx(
                              'w-4 h-4 rounded-full border flex items-center justify-center transition-all',
                              audioFormat === 'wav' ? 'border-white bg-white text-black' : 'border-white/20'
                            )}>
                              {audioFormat === 'wav' && <div className="w-1.5 h-1.5 rounded-full bg-black" />}
                            </div>
                          </div>
                        </div>

                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-white tracking-tight block">
                            WAV Originale Non Compresso
                          </span>
                          <p className="text-[11px] text-white/50 leading-relaxed">
                            Copia identica al campione microfonico grezzo a 16-bit 16 kHz. Massima fedeltà d'archivio senza compressione lossy (~115 MB/ora).
                          </p>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-white/[0.06] flex items-center justify-between text-[10px] font-mono text-white/40">
                        <span>Formato: PCM 16-bit · 16 kHz</span>
                        <span className="text-white/50">~115 MB / ora</span>
                      </div>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 4: PRIVACY, SICUREZZA & RESET SISTEMA
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'system' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-bold text-white">Privacy & Controllo Sistema</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    SbobinAI opera al 100% in locale sul processore del tuo computer. Nessun audio viene mai trasmesso all'esterno.
                  </p>
                </div>

                <div className="p-4 rounded-3xl bg-emerald-500/10 border border-emerald-500/25 flex items-start gap-3">
                  <ShieldCheck size={20} className="text-emerald-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-white/80 space-y-1">
                    <h4 className="font-bold text-emerald-300">Confidenzialità Accademica Totale</h4>
                    <p className="text-[11px] text-white/60 leading-relaxed">
                      Le registrazioni e gli appunti risiedono esclusivamente sul disco del tuo computer. I modelli IA girano direttamente sul tuo computer senza connessioni a server remoti.
                    </p>
                  </div>
                </div>

                {/* Sezione Aggiornamenti */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.12] space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-sm font-bold text-white">Aggiornamenti</h4>
                        <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-white/[0.08] text-white/70 border border-white/10">
                          v{updater.currentVersion || '0.0.10'}
                        </span>
                      </div>
                      <p className="text-[11px] text-white/50 mt-0.5">
                        {updater.supported
                          ? "Mantieni SbobinAI sempre aggiornato con le ultime correzioni e modelli."
                          : "Gli aggiornamenti automatici sono disponibili solo nell'app desktop."}
                      </p>
                    </div>
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
                        Controlla e installa automaticamente le nuove versioni ad ogni apertura dell'app, risolvendo tempestivamente bug e problemi di avvio.
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

                  <div className="flex items-center gap-3 flex-wrap pt-1">
                    <button
                      type="button"
                      disabled={!updater.supported || updater.status === 'checking' || updater.status === 'downloading' || updater.status === 'installing'}
                      onClick={() => void updater.checkNow()}
                      className="px-4 py-1.5 rounded-full bg-white/[0.10] hover:bg-white/[0.18] border border-white/[0.12] text-xs font-semibold text-white disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer transition-colors"
                    >
                      {updater.status === 'checking' ? 'Controllo in corso...' : 'Controlla aggiornamenti'}
                    </button>
                    <span className="text-[11px] text-white/40 font-medium">
                      Versione installata: <strong className="font-mono text-white/80">v{updater.currentVersion || '0.0.10'}</strong>
                    </span>
                    {updater.status === 'uptodate' && <span className="text-[11px] text-emerald-300 font-medium">Sei all'ultima versione</span>}
                    {updater.status === 'available' && (
                      <>
                        <span className="text-[11px] text-blue-300">Versione {updater.version} disponibile</span>
                        <button
                          type="button"
                          onClick={() => void updater.install()}
                          className="px-4 py-1.5 rounded-full bg-blue-500 hover:bg-blue-400 text-xs font-semibold text-white cursor-pointer"
                        >
                          Aggiorna e Riavvia
                        </button>
                      </>
                    )}
                    {updater.status === 'downloading' && <span className="text-[11px] text-white/60">Download {updater.progress}%</span>}
                    {updater.status === 'installing' && <span className="text-[11px] text-white/60">Installazione e riavvio...</span>}
                    {updater.status === 'error' && <span className="text-[11px] text-red-300">Errore: {updater.error}</span>}
                  </div>
                </div>

                {/* Sezione Reset Onboarding */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-3">
                  <div>
                    <h4 className="text-xs font-bold text-white">Ripristino Configurazione Iniziale</h4>
                    <p className="text-[11px] text-white/50 mt-0.5 leading-relaxed">
                      Se vuoi ripercorrere i passaggi del Wizard (benvenuto, selezione cartella e download modelli), puoi riavviare la procedura guidata in qualsiasi momento.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={handleResetWizard}
                    className={clsx(
                      "flex items-center gap-2 px-4 py-2 rounded-2xl text-xs font-semibold transition-all cursor-pointer",
                      confirmReset
                        ? "bg-red-500 text-white shadow-liquid-sm"
                        : "bg-white/[0.08] hover:bg-white/[0.16] text-white/80 border border-white/14"
                    )}
                  >
                    <RotateCcw size={13} className={confirmReset ? "animate-spin text-white" : ""} />
                    <span>{confirmReset ? 'Sei sicuro? Clicca per riavviare il Wizard' : 'Riavvia Configurazione Wizard'}</span>
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

      </div>
    </div>
  );
}
