import { useEffect, useState, useCallback, useRef } from 'react';
import { 
  X, Folder, Cpu, Mic, HardDrive, Check, 
  RotateCcw, Sliders, Trash2, Download, Loader2,
  RefreshCw, CheckCircle2, ShieldCheck, Database, FileAudio, 
  Layers, AlertTriangle
} from 'lucide-react';
import { clsx } from 'clsx';
import { useSettings } from '../lib/SettingsContext';
import { pickFolder } from '../lib/tauri';
import { api } from '../lib/api';
import type { ModelsStatusResponse, ModelInfo } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'storage' | 'models' | 'audio' | 'system';
type ModelFilterType = 'system_core' | 'llm';

export default function SettingsModal({ isOpen, onClose }: Props) {
  const { settings, updateSettings, resetSetup } = useSettings();
  // Ref per leggere le impostazioni correnti senza far ri-eseguire gli effetti a ogni aggiornamento
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const updateSettingsRef = useRef(updateSettings);
  updateSettingsRef.current = updateSettings;

  const [activeTab, setActiveTab] = useState<TabType>('models');
  const [modelFilter, setModelFilter] = useState<ModelFilterType>('system_core');

  // Form State
  const [storagePath, setStoragePath] = useState(settings.storagePath);
  const [whisperModel, setWhisperModel] = useState(settings.whisperModel);
  const [llmModel, setLlmModel] = useState(settings.llmModel || 'mlx-community/Qwen3.5-4B-MLX-4bit');
  const [embeddingModel, setEmbeddingModel] = useState(settings.embeddingModel || 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ');
  const [vadSensitivity, setVadSensitivity] = useState(settings.vadSensitivity);
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
      fetchModelsStatus();
      fetchStorageStats();
    }
  }, [isOpen]);

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
        showNotification(`Cartella di archiviazione aggiornata nel Finder.`, 'success');
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

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in select-none"
      onClick={onClose}
    >
      <div 
        className="relative w-full max-w-4xl rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-lg overflow-hidden flex flex-col h-[640px] animate-slide-up"
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

              {/* Tab 2: Modelli & Cache IA (Senza badge numerico) */}
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
                <span>Modelli IA & Cache</span>
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
          <div className="flex-1 p-6 pb-12 overflow-y-auto space-y-6">
            
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
                TAB 1: ARCHIVIAZIONE & DATI LOCALI (Sincronizzata con Wizard)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'storage' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-bold text-white">Cartella di Archiviazione Principale</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    Questa è la cartella scelta durante il Wizard o aggiornata di seguito. Contiene i tuoi appunti, file master WAV e il database SQLite.
                  </p>
                </div>

                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-3">
                  <label className="text-xs font-semibold text-white/70 block">Percorso Libreria Corsi sul Mac</label>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 px-4 py-2.5 rounded-2xl liquid-input font-mono text-xs text-white/95 truncate">
                      {storagePath}
                    </div>
                    <button
                      type="button"
                      onClick={handlePickFolder}
                      className="px-4 py-2.5 rounded-2xl liquid-glass-pill border border-white/18 hover:border-white/28 hover:bg-white/[0.10] text-xs font-semibold text-white transition-all flex items-center gap-2 flex-shrink-0 cursor-pointer shadow-liquid-sm"
                    >
                      <Folder size={14} className="text-white/70" />
                      <span>Sfoglia nel Finder...</span>
                    </button>
                  </div>
                </div>

                {/* Sottocartelle strutturate con metriche live */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-semibold text-white">
                        <Database size={14} className="text-white/60" />
                        <span>Database SQLite</span>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-white/50">
                        {storageStats ? `${storageStats.sqlite_mb} MB` : '...'}
                      </span>
                    </div>
                    <p className="text-[11px] font-mono text-white/40 truncate">/sqlite/sbobinai.db</p>
                    <p className="text-[10px] text-white/50">Quaderni, sbobine e trascrizioni temporizzate.</p>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-semibold text-white">
                        <FileAudio size={14} className="text-white/60" />
                        <span>Registrazioni WAV</span>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-white/50">
                        {storageStats ? `${storageStats.audio_mb} MB` : '...'}
                      </span>
                    </div>
                    <p className="text-[11px] font-mono text-white/40 truncate">/audio/...</p>
                    <p className="text-[10px] text-white/50">Tracce audio native 16-bit PCM salvate localmente.</p>
                  </div>

                  <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-semibold text-white">
                        <Layers size={14} className="text-white/60" />
                        <span>Documenti & LanceDB</span>
                      </div>
                      <span className="text-[10px] font-mono font-bold text-white/50">
                        {storageStats ? `${((storageStats.uploads_mb || 0) + (storageStats.lancedb_mb || 0)).toFixed(1)} MB` : '...'}
                      </span>
                    </div>
                    <p className="text-[11px] font-mono text-white/40 truncate">/uploads & /lancedb</p>
                    <p className="text-[10px] text-white/50">Materiali caricati e vettori RAG indicizzati.</p>
                  </div>
                </div>

                {/* Sezione Manutenzione Disco & File Orfani (Garbage Collector) */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Trash2 size={15} className="text-amber-400" />
                      <h4 className="text-xs font-bold text-white">Manutenzione & Spazzino File Orfani</h4>
                    </div>
                    <p className="text-[11px] text-white/50 leading-relaxed max-w-lg">
                      Elimina immediatamente vecchie registrazioni o file temporanei rimasti sul disco senza un record corrispondente nel DB e compatta il database (VACUUM).
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={handleCleanOrphans}
                    disabled={isCleaningOrphans}
                    className={clsx(
                      'px-4 py-2.5 rounded-2xl font-semibold text-xs transition-all flex items-center justify-center gap-2 flex-shrink-0 cursor-pointer shadow-liquid-sm',
                      isCleaningOrphans
                        ? 'bg-amber-500/20 text-amber-200 border border-amber-500/30 cursor-wait'
                        : 'bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/30'
                    )}
                  >
                    {isCleaningOrphans ? (
                      <>
                        <Loader2 size={13} className="animate-spin text-amber-300" />
                        <span>Pulizia in corso...</span>
                      </>
                    ) : (
                      <>
                        <RotateCcw size={13} />
                        <span>Svuota Cache & File Orfani</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 2: GESTIONE MODELLI IA & CACHE (SENZA TESTI TAGLIATI O DUPLICATI)
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'models' && (
              <div className="space-y-6 animate-fade-in">
                
                {/* ── Header Monitoraggio Spazio & Controlli Rapidi (Risolto overflow) ── */}
                <div className="p-4 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="flex items-center gap-3.5 min-w-0">
                    <div className="w-10 h-10 rounded-2xl bg-white/[0.08] border border-white/[0.12] flex items-center justify-center text-white/80 flex-shrink-0">
                      <HardDrive size={18} />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span className="text-xs font-bold text-white">
                          Memoria Modelli IA sul Mac:
                        </span>
                        <span className="text-xs font-bold font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                          {totalInstalledMb} MB {totalInstalledMb > 1024 ? `(~${(totalInstalledMb / 1024).toFixed(2)} GB)` : ''}
                        </span>
                      </div>
                      <p className="text-[11px] text-white/45 mt-0.5 truncate">
                        Posizione cache sul Mac: ~/.cache/huggingface/hub
                      </p>
                    </div>
                  </div>

                  {/* Pulsanti Header (senza troncamento) */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {/* Ricarica stato */}
                    <button
                      type="button"
                      onClick={() => fetchModelsStatus()}
                      disabled={loadingModels}
                      className="p-2 rounded-xl liquid-glass-pill border border-white/12 hover:border-white/22 text-white/70 hover:text-white transition-all cursor-pointer disabled:opacity-50"
                      title="Ricarica stato da disco"
                    >
                      <RefreshCw size={14} className={clsx(loadingModels && "animate-spin text-blue-400")} />
                    </button>

                    {/* Svuota Tutta la Cache (senza testo tagliato) */}
                    {totalInstalledMb > 0 && (
                      <button
                        type="button"
                        onClick={() => setConfirmDeleteAll(true)}
                        className="flex items-center gap-1.5 px-3.5 py-2 rounded-2xl text-xs font-semibold bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer whitespace-nowrap shadow-liquid-sm flex-shrink-0"
                      >
                        <Trash2 size={13} className="text-red-300" />
                        <span>Svuota tutta la cache</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Banner di Conferma Eliminazione Totale */}
                {confirmDeleteAll && (
                  <div className="p-4 rounded-3xl bg-red-500/15 border border-red-500/35 text-xs text-red-200 space-y-3 animate-fade-in shadow-liquid-sm">
                    <div className="flex items-start gap-3">
                      <Trash2 size={18} className="text-red-300 flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="font-bold text-white text-sm">Vuoi davvero eliminare tutti i modelli dal disco?</h4>
                        <p className="text-xs text-red-200/80 mt-1 leading-relaxed">
                          Verranno rimossi fisicamente dalla cache di sistema tutti i file ({totalInstalledMb} MB liberati). L'app tornerà in modalità simulatore fittizia (0 MB).
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center justify-end gap-2.5 pt-1">
                      <button
                        type="button"
                        onClick={() => setConfirmDeleteAll(false)}
                        className="px-4 py-1.5 rounded-xl text-xs font-medium text-white/70 hover:text-white hover:bg-white/[0.08] transition-colors cursor-pointer"
                      >
                        Annulla
                      </button>
                      <button
                        type="button"
                        onClick={handleDeleteAllModels}
                        disabled={actionInProgress === 'delete-all'}
                        className="px-4 py-1.5 rounded-xl text-xs font-semibold bg-red-500 text-white hover:bg-red-600 transition-colors cursor-pointer shadow-liquid-sm disabled:opacity-50"
                      >
                        {actionInProgress === 'delete-all' ? 'Eliminazione in corso...' : 'Sì, Elimina Tutto e Libera Spazio'}
                      </button>
                    </div>
                  </div>
                )}

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
                        Componenti di Sistema Obbligatori (Apple Silicon Metal)
                      </span>
                      <p className="text-[11px] text-white/45 mt-1 leading-relaxed">
                        Questi due motori costituiscono il nucleo stabile di SbobinAI: Whisper Large Turbo per la trascrizione in memoria e Qwen3-Embedding per la memoria RAG (1024 dim) su LanceDB.
                      </p>
                    </div>

                    <div className="space-y-3">
                      {loadingModels && !modelsStatus ? (
                        <div className="p-8 rounded-3xl liquid-glass border border-white/10 text-center space-y-2">
                          <RefreshCw size={22} className="animate-spin text-blue-400 mx-auto" />
                          <p className="text-xs text-white/60">Verifica componenti di sistema in corso...</p>
                        </div>
                      ) : (
                        (() => {
                          const coreWhisper = modelsStatus?.whisper?.find((m) => m.id === 'large-v3-turbo-q4') || {
                            id: 'large-v3-turbo-q4',
                            name: 'Whisper Large-v3-Turbo Q4 (Apple Silicon)',
                            type: 'whisper' as const,
                            approx_size: '~580 MB',
                            description: 'Motore Vocale di sistema obbligatorio: trascrizione ad alta precisione 4-bit per Apple Silicon Metal (~580 MB). Fedeltà massima e consumo nullo in pausa.',
                            downloaded: false,
                            status: 'idle' as const,
                            disk_size_mb: null,
                          };

                          const coreEmbedding = modelsStatus?.embedding?.find((m) => m.id === 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ') || {
                            id: 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
                            name: 'Qwen3 Embedding 0.6B 4-bit DWQ (MLX)',
                            type: 'embedding' as const,
                            approx_size: '~300 MB',
                            description: 'Motore Memoria RAG di sistema obbligatorio: architettura Qwen3 nativa MLX, 1024 dimensioni (~300 MB in RAM). Indicizzazione ad altissima velocità e ricerca ibrida su LanceDB.',
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
                                    ? 'bg-emerald-500/[0.07] border-emerald-500/25 shadow-liquid-sm'
                                    : 'liquid-glass-subtle border-white/[0.08] hover:border-white/16'
                                )}
                              >
                                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                  <div className="flex items-start gap-3 min-w-0">
                                    <div className={clsx(
                                      'w-9 h-9 rounded-2xl flex items-center justify-center flex-shrink-0 mt-0.5',
                                      m.downloaded
                                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                        : 'bg-white/5 text-white/40 border border-white/10'
                                    )}>
                                      {isMic ? <Mic size={16} /> : <Database size={16} />}
                                    </div>

                                    <div className="min-w-0">
                                      <div className="flex items-center gap-2 flex-wrap">
                                        <h4 className="text-xs font-bold text-white">{m.name}</h4>
                                        <span className="text-[10px] px-2 py-0.5 rounded-full font-mono bg-blue-500/20 text-blue-300 border border-blue-500/30">
                                          Componente Base
                                        </span>
                                        {m.downloaded ? (
                                          <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                                            <Check size={11} strokeWidth={2.5} />
                                            <span>Installato e Operativo ({m.disk_size_mb || m.total_mb} MB)</span>
                                          </span>
                                        ) : (
                                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300 border border-amber-500/25">
                                            Da Installare ({m.approx_size})
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
                                        <span className="text-[11px] font-semibold text-emerald-400 px-2 py-1">
                                          Verificato ✓
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
                        })()
                      )}
                    </div>
                  </div>
                )}

                {/* ── LISTA MODELLI LLM (mlx-lm: Qwen3.5 & Gemma 4) ── */}
                {modelFilter === 'llm' && (
                  <div className="space-y-4">
                    <div>
                      <span className="text-xs font-semibold uppercase tracking-wider text-white/50 block">
                        Modelli Linguistici LLM (Accelerazione Metal Apple Silicon & mlx-lm)
                      </span>
                      <p className="text-[11px] text-white/45 mt-1 leading-relaxed">
                        I modelli 4-bit generano capitoli di studio e rispondono nella chat. Puoi scaricare il modello desiderato, impostarlo come attivo o eliminarlo per recuperare spazio su disco.
                      </p>
                    </div>

                    <div className="space-y-3">
                      {loadingModels && !modelsStatus ? (
                        <div className="p-8 rounded-3xl liquid-glass border border-white/10 text-center space-y-2">
                          <RefreshCw size={22} className="animate-spin text-blue-400 mx-auto" />
                          <p className="text-xs text-white/60">Verifica modelli LLM in corso...</p>
                        </div>
                      ) : !modelsStatus?.llm || modelsStatus.llm.length === 0 ? (
                        <div className="p-6 rounded-3xl liquid-glass border border-white/10 text-center space-y-3">
                          <p className="text-xs text-white/70">Nessun modello LLM configurato.</p>
                        </div>
                      ) : (
                        modelsStatus.llm.map((m: ModelInfo) => {
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
                                      {m.downloaded && isCurrentActive ? (
                                        <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                                          <Check size={11} strokeWidth={2.5} />
                                          <span>In uso ({m.disk_size_mb || m.total_mb} MB)</span>
                                        </span>
                                      ) : m.downloaded ? (
                                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30">
                                          Scaricato ({m.disk_size_mb || m.total_mb} MB)
                                        </span>
                                      ) : (
                                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/10 text-white/50 border border-white/10">
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
                                      {!isCurrentActive ? (
                                        <button
                                          type="button"
                                          onClick={() => handleSelectActiveLlm(m.id)}
                                          className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-white/[0.14] hover:bg-white/[0.22] text-white border border-white/[0.18] transition-all cursor-pointer shadow-liquid-sm"
                                        >
                                          Imposta come Attivo
                                        </button>
                                      ) : (
                                        <span className="text-[11px] font-semibold text-emerald-400 px-2 py-1">
                                          Attivo ✓
                                        </span>
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
                        })
                      )}
                    </div>
                  </div>
                )}

              </div>
            )}

            {/* ══════════════════════════════════════════════════════════════════
                TAB 3: MOTORE AUDIO & FILTRO VAD
            ══════════════════════════════════════════════════════════════════ */}
            {activeTab === 'audio' && (
              <div className="space-y-6 animate-fade-in">
                <div>
                  <h3 className="text-sm font-bold text-white">Motore Audio in RAM & Voice Activity Detection</h3>
                  <p className="text-xs text-white/50 mt-1 leading-relaxed">
                    L'audio della lezione viene elaborato nei banchi RAM per azzerare l'usura del disco SSD e filtrare il silenzio dell'aula.
                  </p>
                </div>

                <div className="p-5 rounded-3xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-semibold text-white/80 flex items-center gap-2">
                      <Sliders size={14} className="text-white/70" />
                      <span>Sensibilità Filtro Voce (WebRTC VAD)</span>
                    </label>
                    <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 rounded-full">
                      Livello {vadSensitivity} di 3
                    </span>
                  </div>

                  <input
                    type="range"
                    min={1}
                    max={3}
                    step={1}
                    value={vadSensitivity}
                    onChange={(e) => handleVadChange(Number(e.target.value))}
                    className="w-full accent-white h-2 bg-white/10 rounded-lg cursor-pointer"
                  />

                  <div className="grid grid-cols-3 gap-2 pt-2 text-[11px] font-mono text-center">
                    <div className={clsx("p-2 rounded-xl border transition-all", vadSensitivity === 1 ? "bg-white/[0.12] border-white/20 text-white font-semibold" : "border-transparent text-white/40")}>
                      1: Permissivo (Voce lontana)
                    </div>
                    <div className={clsx("p-2 rounded-xl border transition-all", vadSensitivity === 2 ? "bg-white/[0.12] border-white/20 text-white font-semibold" : "border-transparent text-white/40")}>
                      2: Bilanciato (Consigliato aula)
                    </div>
                    <div className={clsx("p-2 rounded-xl border transition-all", vadSensitivity === 3 ? "bg-white/[0.12] border-white/20 text-white font-semibold" : "border-transparent text-white/40")}>
                      3: Aggressivo (Solo vicino)
                    </div>
                  </div>
                </div>

                {/* Specifiche Tecniche Audio Stream */}
                <div className="p-4 rounded-3xl bg-white/[0.02] border border-white/[0.06] space-y-2.5">
                  <span className="text-xs font-semibold text-white/70 block">Parametri Flusso Audio Locale</span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono text-white/60">
                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-[10px] text-white/35 block">Frequenza</span>
                      <strong className="text-white">16.000 Hz</strong>
                    </div>
                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-[10px] text-white/35 block">Canali</span>
                      <strong className="text-white">Mono (1 ch)</strong>
                    </div>
                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-[10px] text-white/35 block">Formato PCM</span>
                      <strong className="text-white">16-bit LE</strong>
                    </div>
                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-[10px] text-white/35 block">Architettura</span>
                      <strong className="text-white">RAM Ring Buffer</strong>
                    </div>
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
                    SbobinAI opera al 100% in locale sul processore del tuo Mac. Nessun audio viene mai trasmesso all'esterno.
                  </p>
                </div>

                <div className="p-4 rounded-3xl bg-emerald-500/10 border border-emerald-500/25 flex items-start gap-3">
                  <ShieldCheck size={20} className="text-emerald-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs text-white/80 space-y-1">
                    <h4 className="font-bold text-emerald-300">Confidenzialità Accademica Totale</h4>
                    <p className="text-[11px] text-white/60 leading-relaxed">
                      Le registrazioni e gli appunti risiedono esclusivamente sul disco del tuo Mac. I modelli IA girano tramite Apple Silicon Metal e la memoria unificata senza connessioni a server remoti.
                    </p>
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

      </div>
    </div>
  );
}
