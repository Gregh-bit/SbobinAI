import { osBadge, isMac } from '../lib/platform';
import { useEffect, useState, useRef } from 'react';
import { 
  Sparkles, Folder, ArrowRight, ArrowLeft, 
  CheckCircle2, HardDrive, ShieldCheck, Zap, Laptop,
  Download, Loader2, X, AlertCircle, Bot, Check, Database, Mic, RefreshCw
} from 'lucide-react';
import { clsx } from 'clsx';
import { useSettings } from '../lib/SettingsContext';
import { pickFolder } from '../lib/tauri';
import { api } from '../lib/api';
import type { ModelsStatusResponse } from '../types';

const STEPS = [
  { id: 1, label: 'Benvenuto' },
  { id: 2, label: 'Archiviazione' },
  { id: 3, label: 'Assistente AI' },
  { id: 4, label: 'Motori di Sistema' },
];

const LLM_OPTIONS = [
  {
    id: 'mlx-community/Qwen3.5-4B-MLX-4bit',
    name: 'Qwen 3.5 4B (4-bit)',
    badge: 'Consigliato 2026',
    badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    description: 'Architettura Gated DeltaNet 2026, KV cache quantizzata 4-bit, ideale per computer con 8GB di RAM. Risposte flash istantanee live e capitoli completi del Quaderno.',
    downloadSize: '~2.4 GB',
    ram: '~2.3 GB',
  },
  {
    id: 'unsloth/gemma-4-E4B-it-UD-MLX-4bit',
    name: 'Gemma 4 E4B (4-bit)',
    badge: 'Alternativo',
    badgeColor: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    description: 'Modello compatto ad alta efficienza per sintesi accademiche e spiegazioni didattiche in locale.',
    downloadSize: '~2.5 GB',
    ram: '~2.4 GB',
  },
  {
    id: 'later',
    name: 'Scegli e scarica in seguito',
    badge: '0 GB Scaricati · Configura dopo',
    badgeColor: 'bg-white/10 text-white/70 border-white/15',
    description: 'Usa l\'app subito per registrare e trascrivere le lezioni. Potrai scegliere e scaricare il modello linguistico con 1 clic dalle Impostazioni quando ti serviranno i riassunti IA.',
    downloadSize: '0 GB',
    ram: '0 GB',
  },
];

const CORE_WHISPER_ID = 'large-v3-turbo-q4';
const CORE_EMBEDDING_ID = 'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ';

const MODEL_SIZE_ESTIMATES: Record<string, number> = {
  [CORE_WHISPER_ID]: 580,
  [CORE_EMBEDDING_ID]: 300,
  'mlx-community/Qwen3.5-4B-MLX-4bit': 2400,
  'unsloth/gemma-4-E4B-it-UD-MLX-4bit': 2500,
  'later': 0,
};

function getModelDisplayName(id: string): string {
  if (id === CORE_WHISPER_ID) return 'Whisper Large-v3-Turbo Q4';
  if (id === CORE_EMBEDDING_ID) return 'Qwen3-Embedding 0.6B RAG';
  if (id === 'mlx-community/Qwen3.5-4B-MLX-4bit') return 'Qwen 3.5 4B (4-bit)';
  if (id === 'unsloth/gemma-4-E4B-it-UD-MLX-4bit') return 'Gemma 4 E4B (4-bit)';
  return id;
}

function formatMbOrGb(mb: number): string {
  if (mb >= 1000) {
    return `~${(mb / 1024).toFixed(1)} GB`;
  }
  return `~${Math.round(mb)} MB`;
}

function formatFriendlyError(raw?: unknown): string {
  if (!raw) return 'Errore sconosciuto durante la connessione con il motore locale.';
  const str = typeof raw === 'string' ? raw : (raw as any)?.message || String(raw);
  const lower = str.toLowerCase();
  if (
    lower.includes('load failed') ||
    lower.includes('failed to fetch') ||
    lower.includes('networkerror') ||
    lower.includes('econnrefused') ||
    lower.includes('connection refused')
  ) {
    return 'Motore locale SbobinAI (porta 8000) non raggiungibile. Verifica che il servizio locale sia avviato correttamente.';
  }
  return str;
}

export default function Wizard() {
  const { settings, completeSetup } = useSettings();
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);

  // Form State
  const [storagePath, setStoragePath] = useState(settings.storagePath || '~/Documents/SbobinAI');
  const [llmModel, setLlmModel] = useState<string>('mlx-community/Qwen3.5-4B-MLX-4bit');
  const [isSelectingFolder, setIsSelectingFolder] = useState(false);

  // Stato Modelli Reale da Backend
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [downloadQueue, setDownloadQueue] = useState<string[]>([]);
  const [queueIndex, setQueueIndex] = useState<number>(0);
  const [activeDownloadModel, setActiveDownloadModel] = useState<string | null>(null);
  const [isDownloadingScreen, setIsDownloadingScreen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isTriggering, setIsTriggering] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Ref per accedere sempre ai valori correnti nel loop del polling
  const queueRef = useRef({ downloadQueue, queueIndex, activeDownloadModel, isDownloadingScreen });
  queueRef.current = { downloadQueue, queueIndex, activeDownloadModel, isDownloadingScreen };

  // Verifica continua dello stato dei modelli
  useEffect(() => {
    let isMounted = true;

    const checkStatus = async () => {
      try {
        const res = await api.models.status();
        if (!isMounted) return;
        setModelsStatus(res);

        const { downloadQueue: q, queueIndex: qIdx, activeDownloadModel: curActive, isDownloadingScreen: isDl } = queueRef.current;

        // Se siamo nella schermata di download, verifica l'avanzamento sequenziale della coda
        if (isDl && curActive) {
          const allModels = [...(res.whisper || []), ...(res.embedding || []), ...(res.llm || [])];
          const curInfo = allModels.find((m) => m.id === curActive);

          if (curInfo?.status === 'completed') {
            const nextIdx = qIdx + 1;
            if (nextIdx < q.length) {
              const nextModelId = q[nextIdx];
              queueRef.current.queueIndex = nextIdx;
              queueRef.current.activeDownloadModel = nextModelId;
              setQueueIndex(nextIdx);
              setActiveDownloadModel(nextModelId);
              setDownloadError(null);
              try {
                await api.models.download(nextModelId);
              } catch (e: any) {
                setDownloadError(formatFriendlyError(e));
              }
            } else {
              // Tutti i modelli della coda sono stati completati con successo
              queueRef.current.isDownloadingScreen = false;
              queueRef.current.activeDownloadModel = null;
              queueRef.current.downloadQueue = [];
              queueRef.current.queueIndex = 0;
              setIsDownloadingScreen(false);
              setActiveDownloadModel(null);
              setDownloadQueue([]);
              setQueueIndex(0);
              setDownloadError(null);
            }
          } else if (curInfo?.status === 'error') {
            setDownloadError(curInfo.error || 'Errore durante la connessione con il repository.');
          }
        }
      } catch (err) {
        console.warn('[Wizard] Errore verifica modelli:', err);
      }
    };

    checkStatus();
    const intervalMs = isDownloadingScreen ? 1000 : 2500;
    const timer = setInterval(checkStatus, intervalMs);
    return () => {
      isMounted = false;
      clearInterval(timer);
    };
  }, [isDownloadingScreen]);

  const handleChooseFolder = async () => {
    setIsSelectingFolder(true);
    try {
      const selected = await pickFolder(storagePath);
      if (selected) {
        setStoragePath(selected);
      }
    } finally {
      setIsSelectingFolder(false);
    }
  };

  // Verifica presenza su disco dei componenti
  const isWhisperDownloaded = Boolean(
    modelsStatus?.whisper?.find((m) => m.id === CORE_WHISPER_ID)?.downloaded
  );
  const isEmbeddingDownloaded = Boolean(
    modelsStatus?.embedding?.find((m) => m.id === CORE_EMBEDDING_ID)?.downloaded
  );
  const isLlmDownloaded =
    llmModel === 'later'
      ? true
      : Boolean(modelsStatus?.llm?.find((m) => m.id === llmModel)?.downloaded);

  // Tutto pronto per iniziare
  const isAllRequiredReady = isWhisperDownloaded && isEmbeddingDownloaded && isLlmDownloaded;

  // Calcolo dinamico dei megabyte totali da scaricare
  let totalMbNeeded = 0;
  if (!isWhisperDownloaded) totalMbNeeded += MODEL_SIZE_ESTIMATES[CORE_WHISPER_ID] || 580;
  if (!isEmbeddingDownloaded) totalMbNeeded += MODEL_SIZE_ESTIMATES[CORE_EMBEDDING_ID] || 300;
  if (llmModel !== 'later' && !isLlmDownloaded) {
    totalMbNeeded += MODEL_SIZE_ESTIMATES[llmModel] || 2400;
  }

  // Modello attualmente monitorato durante il download
  const allKnownModels = [
    ...(modelsStatus?.whisper || []),
    ...(modelsStatus?.embedding || []),
    ...(modelsStatus?.llm || []),
  ];
  const activeInfo = allKnownModels.find((m) => m.id === activeDownloadModel);
  const progressPct = activeInfo?.progress ?? 0;
  const downloadedMb = activeInfo?.downloaded_mb ?? 0;
  const totalMb = activeInfo?.total_mb ?? (MODEL_SIZE_ESTIMATES[activeDownloadModel || ''] || 500);

  // Avvio download dei componenti in coda sequenziale
  const handleStartDownloads = async () => {
    try {
      setIsTriggering(true);
      setDownloadError(null);

      // Assicurati che lo stato modelli sia disponibile, con fallback resiliente
      let currentStatus = modelsStatus;
      if (!currentStatus) {
        try {
          currentStatus = await api.models.status();
          setModelsStatus(currentStatus);
        } catch (e) {
          console.warn('[Wizard] Impossibile recuperare stato modelli, uso fallback:', e);
        }
      }

      const whisperDone = Boolean(
        currentStatus?.whisper?.find((m) => m.id === CORE_WHISPER_ID)?.downloaded
      );
      const embeddingDone = Boolean(
        currentStatus?.embedding?.find((m) => m.id === CORE_EMBEDDING_ID)?.downloaded
      );
      const llmDone =
        llmModel === 'later'
          ? true
          : Boolean(currentStatus?.llm?.find((m) => m.id === llmModel)?.downloaded);

      // Costruisci la lista dei modelli da scaricare
      const queue: string[] = [];
      if (!whisperDone) queue.push(CORE_WHISPER_ID);
      if (!embeddingDone) queue.push(CORE_EMBEDDING_ID);
      if (llmModel !== 'later' && !llmDone) queue.push(llmModel);

      if (queue.length === 0) {
        await handleFinishSetup();
        return;
      }

      queueRef.current.downloadQueue = queue;
      queueRef.current.queueIndex = 0;
      queueRef.current.activeDownloadModel = queue[0];
      queueRef.current.isDownloadingScreen = true;

      setDownloadQueue(queue);
      setQueueIndex(0);
      setActiveDownloadModel(queue[0]);
      setIsDownloadingScreen(true);

      try {
        await api.models.download(queue[0]);
      } catch (err: any) {
        console.error('[Wizard] Errore avvio download:', err);
        setDownloadError(formatFriendlyError(err));
      }
    } catch (err: any) {
      console.error('[Wizard] Errore trigger download:', err);
      setDownloadError(formatFriendlyError(err));
    } finally {
      setIsTriggering(false);
    }
  };

  const handleRetryCurrent = async () => {
    if (!activeDownloadModel) return;
    setDownloadError(null);
    try {
      await api.models.download(activeDownloadModel);
    } catch (err: any) {
      setDownloadError(formatFriendlyError(err));
    }
  };

  const handleInterruptDownload = async () => {
    if (!activeDownloadModel) {
      setIsDownloadingScreen(false);
      return;
    }
    setIsCancelling(true);
    try {
      await api.models.cancelDownload(activeDownloadModel);
    } catch (err) {
      console.warn('[Wizard] Errore stop download:', err);
    } finally {
      queueRef.current.isDownloadingScreen = false;
      queueRef.current.activeDownloadModel = null;
      queueRef.current.downloadQueue = [];
      queueRef.current.queueIndex = 0;
      setIsCancelling(false);
      setIsDownloadingScreen(false);
      setActiveDownloadModel(null);
      setDownloadQueue([]);
      setQueueIndex(0);
      setDownloadError(null);
      try {
        const res = await api.models.status();
        setModelsStatus(res);
      } catch {}
    }
  };

  // Completamento del Wizard ed ingresso all'app
  const handleFinishSetup = async () => {
    const finalWhisper = CORE_WHISPER_ID;
    const finalEmbedding = CORE_EMBEDDING_ID;
    const finalLlm = llmModel === 'later' ? 'mlx-community/Qwen3.5-4B-MLX-4bit' : llmModel;

    await completeSetup({
      storagePath,
      whisperModel: finalWhisper,
      embeddingModel: finalEmbedding,
      llmModel: finalLlm,
      ollamaModel: finalLlm,
      modelsDownloaded: true,
    });
  };

  return (
    <div className="fixed inset-0 w-screen h-screen bg-[#0d1017]/95 backdrop-blur-md text-white flex flex-col select-none overflow-hidden z-50 rounded-[12px]">
      
      {/* Decorative ambient background glows */}
      <div 
        className="absolute w-[800px] h-[800px] rounded-full pointer-events-none -top-40 -left-40" 
        style={{ background: 'radial-gradient(circle, rgba(37, 99, 235, 0.18) 0%, transparent 70%)' }}
      />
      <div 
        className="absolute w-[700px] h-[700px] rounded-full pointer-events-none -bottom-40 -right-40" 
        style={{ background: 'radial-gradient(circle, rgba(99, 102, 241, 0.16) 0%, transparent 70%)' }}
      />

      {/* ── Header Titolo / Indicatori Step ── */}
      <header
        data-tauri-drag-region
        className="h-14 px-6 flex items-center justify-between border-b border-white/[0.08] liquid-glass flex-shrink-0 z-40"
      >
        <div className="flex items-center min-w-[200px]">
          {isMac && <div className="w-[72px] h-4 pointer-events-none flex-shrink-0" />}
        </div>

        {isDownloadingScreen ? (
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full liquid-glass-subtle border border-blue-500/30 text-xs text-blue-300 font-medium">
            <Loader2 size={12} className="animate-spin text-blue-400" />
            <span>
              Installazione Componente ({queueIndex + 1}/{downloadQueue.length || 1}):{' '}
              {getModelDisplayName(activeDownloadModel || '')}
            </span>
          </div>
        ) : (
          <div className="flex items-center p-1 rounded-full liquid-glass-subtle border border-white/[0.09] shadow-liquid-sm gap-1" data-tauri-drag-region="false">
            {STEPS.map((s) => {
              const isActive = step === s.id;
              const isCompleted = step > s.id;
              return (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => setStep(s.id as 1 | 2 | 3 | 4)}
                  className={clsx(
                    'flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium transition-all duration-150 cursor-pointer',
                    isActive
                      ? 'bg-white/[0.16] text-white shadow-liquid-pill border border-white/[0.18]'
                      : isCompleted
                      ? 'text-white/80 hover:bg-white/[0.06] border border-transparent'
                      : 'text-white/40 hover:text-white/60 border border-transparent'
                  )}
                >
                  <span className={clsx(
                    'w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-mono font-semibold',
                    isActive
                      ? 'bg-white text-black'
                      : isCompleted
                      ? 'bg-white/20 text-white'
                      : 'bg-white/10 text-white/40'
                  )}>
                    {isCompleted ? '✓' : s.id}
                  </span>
                  <span>{s.label}</span>
                </button>
              );
            })}
          </div>
        )}

        <div className="min-w-[200px] flex justify-end pointer-events-none">
          <span className="text-[11px] font-mono text-white/30 px-2.5 py-1 rounded-full border border-white/[0.06] bg-white/[0.02]">
            {osBadge ?? 'SbobinAI'}
          </span>
        </div>
      </header>

      {/* ── Main Body ── */}
      <main className="flex-1 flex flex-col items-center justify-center p-8 max-w-4xl mx-auto w-full overflow-y-auto z-30">
        
        {/* ── STEP 1: BENVENUTO ── */}
        {step === 1 && (
          <div className="animate-fade-in space-y-8 text-center w-full max-w-2xl my-auto">
            <img
              src="/logo-icon.png"
              alt="SbobinAI"
              className="w-24 h-24 mx-auto object-contain select-none pointer-events-none drop-shadow-2xl"
            />

            <div className="space-y-3">
              <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Benvenuto in SbobinAI
              </h1>
              <p className="text-sm text-white/60 max-w-lg mx-auto leading-relaxed">
                Il quaderno universitario intelligente: registra e trascrivi lezioni senza allucinazioni, 
                indicizza i materiali del corso e interroga la memoria con modelli IA al 100% in locale.
              </p>
            </div>

            <div className="grid grid-cols-3 gap-4 pt-4 text-left">
              <div className="p-4 rounded-2xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-2">
                <ShieldCheck size={22} className="text-emerald-400" />
                <p className="text-xs font-semibold text-white/95">Privacy & Locale</p>
                <p className="text-[11px] text-white/50 leading-relaxed">Audio, sbobine e vettori risiedono sul tuo computer senza invio a server esterni.</p>
              </div>
              <div className="p-4 rounded-2xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-2">
                <Zap size={22} className="text-amber-400" />
                <p className="text-xs font-semibold text-white/95">Whisper Large Turbo</p>
                <p className="text-[11px] text-white/50 leading-relaxed">Fedeltà massima della trascrizione vocale su termini tecnici universitari.</p>
              </div>
              <div className="p-4 rounded-2xl liquid-glass border border-white/[0.10] shadow-liquid-sm space-y-2">
                <Laptop size={22} className="text-blue-400" />
                <p className="text-xs font-semibold text-white/95">Leggero e veloce</p>
                <p className="text-[11px] text-white/50 leading-relaxed">KV cache 4-bit, auto-unload intelligente e zero consumo di batteria.</p>
              </div>
            </div>
          </div>
        )}

        {/* ── STEP 2: ARCHIVIAZIONE ── */}
        {step === 2 && (
          <div className="animate-fade-in space-y-6 w-full max-w-2xl my-auto">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full liquid-glass-subtle border border-white/10 text-xs text-white/80">
                <HardDrive size={13} className="text-blue-400" />
                <span>Archiviazione Locale</span>
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-white">
                Dove vuoi archiviare i tuoi corsi?
              </h2>
              <p className="text-xs text-white/60 leading-relaxed">
                SbobinAI memorizzerà in questa cartella il database SQLite delle lezioni, i file audio M4A/WAV registrati e l'indice vettoriale LanceDB.
              </p>
            </div>

            <div className="p-5 rounded-3xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-4">
              <label className="text-xs font-medium text-white/70 block">
                Percorso Cartella sul tuo computer
              </label>
              
              <div className="flex items-center gap-3">
                <div className="flex-1 flex items-center gap-2.5 px-4 py-3 rounded-2xl liquid-input font-mono text-xs text-white/95 overflow-hidden">
                  <Folder size={16} className="text-white/40 flex-shrink-0" />
                  <span className="truncate">{storagePath}</span>
                </div>
                <button
                  type="button"
                  onClick={handleChooseFolder}
                  disabled={isSelectingFolder}
                  className="px-4 py-3 rounded-2xl liquid-glass-pill border border-white/18 hover:border-white/28 hover:bg-white/[0.12] text-xs font-semibold text-white transition-all flex-shrink-0 disabled:opacity-50 cursor-pointer shadow-liquid-sm"
                >
                  {isSelectingFolder ? 'Apertura Finder...' : 'Scegli cartella...'}
                </button>
              </div>

              <div className="pt-2 flex items-center gap-2 text-xs text-white/50">
                <CheckCircle2 size={14} className="text-emerald-400 flex-shrink-0" />
                <span>I file saranno sempre accessibili direttamente dal Finder di macOS.</span>
              </div>
            </div>
          </div>
        )}

        {/* ── STEP 3: MODELLO LLM PER QUADERNO & CHAT (OPZIONALE) ── */}
        {step === 3 && (
          <div className="animate-fade-in space-y-6 w-full max-w-3xl my-auto">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full liquid-glass-subtle border border-white/10 text-xs text-white/80">
                <Bot size={13} className="text-blue-400" />
                <span>Passo Opzionale: Modello Linguistico LLM</span>
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-white">
                Scegli il modello LLM per il Quaderno (o scarica in seguito)
              </h2>
              <p className="text-xs text-white/60 leading-relaxed">
                I modelli LLM generano i capitoli del Quaderno e rispondono al volo nella chat. 
                Puoi impostare il modello ora, oppure scegliere di rimandare il download e farlo in qualsiasi momento dalle Impostazioni.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
              {LLM_OPTIONS.map((opt) => {
                const isSelected = llmModel === opt.id;
                return (
                  <div
                    key={opt.id}
                    onClick={() => setLlmModel(opt.id)}
                    className={clsx(
                      'p-4 rounded-3xl cursor-pointer transition-all border flex flex-col justify-between text-left relative',
                      isSelected
                        ? 'bg-white/[0.14] border-white/35 shadow-liquid-md ring-1 ring-white/25'
                        : 'bg-white/[0.03] border-white/[0.07] hover:bg-white/[0.06] hover:border-white/[0.14]'
                    )}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className={clsx('text-[10px] px-2 py-0.5 rounded-full border font-semibold', opt.badgeColor)}>
                          {opt.badge}
                        </span>
                        {isSelected && (
                          <div className="w-4 h-4 rounded-full bg-emerald-500 text-black flex items-center justify-center">
                            <Check size={10} strokeWidth={3} />
                          </div>
                        )}
                      </div>
                      <h4 className="text-xs font-bold text-white mb-1.5">{opt.name}</h4>
                      <p className="text-[11px] text-white/50 leading-relaxed">{opt.description}</p>
                    </div>

                    <div className="pt-3 mt-3 border-t border-white/[0.06] flex items-center justify-between text-[10px] font-mono text-white/40">
                      <span>Download: {opt.downloadSize}</span>
                      <span>RAM: {opt.ram}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="p-3.5 rounded-2xl liquid-glass border border-white/[0.08] text-xs text-white/60 flex items-center gap-3">
              <Sparkles size={16} className="text-amber-400 flex-shrink-0" />
              <span>
                {llmModel === 'later'
                  ? 'Hai scelto di rimandare: potrai registrare e sbobinare subito le lezioni, e scaricare l\'LLM in seguito dalle Impostazioni.'
                  : `Hai selezionato ${LLM_OPTIONS.find((o) => o.id === llmModel)?.name}. Verrà preparato per l'uso sul tuo computer.`}
              </span>
            </div>
          </div>
        )}

        {/* ── STEP 4: COMPONENTI DI SISTEMA (AUDIO, RAG E MODELLO IA) ── */}
        {step === 4 && !isDownloadingScreen && (
          <div className="animate-fade-in space-y-6 w-full max-w-3xl my-auto">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full liquid-glass-subtle border border-emerald-500/25 text-xs text-emerald-300 font-medium">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>Riepilogo e Installazione Componenti</span>
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-white">
                Componenti per Audio, Ricerca e Assistente IA
              </h2>
              <p className="text-xs text-white/60 leading-relaxed">
                Ecco i motori che verranno installati sul tuo computer. 
                Tutti i pesi operano al 100% in locale sulla memoria del tuo dispositivo.
              </p>
            </div>

            {/* Eventuale messaggio di errore globale */}
            {downloadError && (
              <div className="p-4 rounded-2xl bg-red-500/15 border border-red-500/30 text-red-200 text-xs flex items-center justify-between gap-3 animate-fade-in">
                <div className="flex items-center gap-2.5">
                  <AlertCircle size={16} className="text-red-400 flex-shrink-0" />
                  <span>{downloadError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDownloadError(null)}
                  className="p-1 hover:bg-white/10 rounded-lg text-white/60 hover:text-white"
                >
                  <X size={14} />
                </button>
              </div>
            )}

            {/* Le Card dei Componenti */}
            <div className={clsx(
              "grid gap-3.5",
              llmModel !== 'later' ? "grid-cols-1 sm:grid-cols-3" : "grid-cols-1 sm:grid-cols-2"
            )}>
              {/* Card 1: Whisper Large-v3-Turbo Q4 */}
              <div className={clsx(
                'p-4 rounded-3xl border transition-all flex flex-col justify-between text-left space-y-2.5',
                isWhisperDownloaded
                  ? 'bg-emerald-500/10 border-emerald-500/30 shadow-liquid-sm'
                  : 'liquid-glass border-white/[0.12]'
              )}>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className={clsx(
                      'text-[10px] px-2.5 py-0.5 rounded-full font-semibold border',
                      isWhisperDownloaded
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                        : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                    )}>
                      {isWhisperDownloaded ? 'Installato ✓' : 'Da Installare'}
                    </span>
                    <div className="w-7 h-7 rounded-xl bg-white/[0.08] flex items-center justify-center text-white/80">
                      <Mic size={14} />
                    </div>
                  </div>

                  <h3 className="text-xs font-bold text-white mb-0.5">Motore Whisper</h3>
                  <p className="text-[10px] font-mono text-white/40">Whisper Large Turbo</p>
                  <p className="text-[11px] text-white/60 mt-1.5 leading-relaxed">
                    Trascrizione vocale continua accelerata in locale.
                  </p>
                </div>

                <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[10px] font-mono text-white/50">
                  <span>Stato: {isWhisperDownloaded ? 'Pronto' : 'Richiesto'}</span>
                  <span>~580 MB</span>
                </div>
              </div>

              {/* Card 2: Qwen3-Embedding-0.6B-4bit-DWQ */}
              <div className={clsx(
                'p-4 rounded-3xl border transition-all flex flex-col justify-between text-left space-y-2.5',
                isEmbeddingDownloaded
                  ? 'bg-emerald-500/10 border-emerald-500/30 shadow-liquid-sm'
                  : 'liquid-glass border-white/[0.12]'
              )}>
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className={clsx(
                      'text-[10px] px-2.5 py-0.5 rounded-full font-semibold border',
                      isEmbeddingDownloaded
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                        : 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                    )}>
                      {isEmbeddingDownloaded ? 'Installato ✓' : 'Da Installare'}
                    </span>
                    <div className="w-7 h-7 rounded-xl bg-white/[0.08] flex items-center justify-center text-white/80">
                      <Database size={14} />
                    </div>
                  </div>

                  <h3 className="text-xs font-bold text-white mb-0.5">Memoria RAG</h3>
                  <p className="text-[10px] font-mono text-white/40">Qwen3-Embedding DWQ</p>
                  <p className="text-[11px] text-white/60 mt-1.5 leading-relaxed">
                    Vettori per ricerca semantica e memoria lezioni.
                  </p>
                </div>

                <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[10px] font-mono text-white/50">
                  <span>Stato: {isEmbeddingDownloaded ? 'Pronto' : 'Richiesto'}</span>
                  <span>~300 MB</span>
                </div>
              </div>

              {/* Card 3: Modello LLM selezionato */}
              {llmModel !== 'later' && (
                <div className={clsx(
                  'p-4 rounded-3xl border transition-all flex flex-col justify-between text-left space-y-2.5',
                  isLlmDownloaded
                    ? 'bg-emerald-500/10 border-emerald-500/30 shadow-liquid-sm'
                    : 'liquid-glass border-white/[0.12]'
                )}>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className={clsx(
                        'text-[10px] px-2.5 py-0.5 rounded-full font-semibold border',
                        isLlmDownloaded
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                          : 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                      )}>
                        {isLlmDownloaded ? 'Installato ✓' : 'Selezionato allo Step 3'}
                      </span>
                      <div className="w-7 h-7 rounded-xl bg-white/[0.08] flex items-center justify-center text-white/80">
                        <Bot size={14} />
                      </div>
                    </div>

                    <h3 className="text-xs font-bold text-white mb-0.5">Assistente IA</h3>
                    <p className="text-[10px] font-mono text-white/40 truncate">
                      {getModelDisplayName(llmModel)}
                    </p>
                    <p className="text-[11px] text-white/60 mt-1.5 leading-relaxed">
                      Generazione capitoli Quaderno e risposte in chat.
                    </p>
                  </div>

                  <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[10px] font-mono text-white/50">
                    <span>Stato: {isLlmDownloaded ? 'Pronto' : 'Da scaricare'}</span>
                    <span>{formatMbOrGb(MODEL_SIZE_ESTIMATES[llmModel] || 2400)}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Box informativo di stato */}
            <div className={clsx(
              'p-4 rounded-2xl border flex items-center gap-3 text-xs leading-relaxed',
              isAllRequiredReady
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
                : 'bg-blue-500/10 border-blue-500/25 text-blue-200'
            )}>
              {isAllRequiredReady ? (
                <>
                  <CheckCircle2 size={18} className="text-emerald-400 flex-shrink-0" />
                  <span>
                    <strong>Tutti i componenti sono pronti!</strong> Il tuo computer ha tutti i modelli necessari. Clicca su "Inizia a Studiare" per accedere all'applicazione.
                  </span>
                </>
              ) : (
                <>
                  <Sparkles size={18} className="text-blue-400 flex-shrink-0" />
                  <span>
                    <strong>Download pronto:</strong> clicca su <strong>"Scarica ({formatMbOrGb(totalMbNeeded)})"</strong> per installare sul disco i modelli selezionati.
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── SCHERMATA DOWNLOAD A TUTTO SCHERMO CON PROGRESS BAR A CERCHIO ── */}
        {isDownloadingScreen && (
          <div className="flex-1 flex flex-col items-center justify-center p-8 max-w-lg mx-auto text-center animate-fade-in space-y-7 my-auto">
            <div className="space-y-1">
              <span className="text-[11px] font-mono uppercase tracking-wider text-blue-400 font-semibold px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 inline-block">
                Installazione Componenti ({queueIndex + 1} di {downloadQueue.length || 1})
              </span>
              <h2 className="text-xl font-bold tracking-tight text-white">
                Download {getModelDisplayName(activeDownloadModel || '')} ({formatMbOrGb(MODEL_SIZE_ESTIMATES[activeDownloadModel || ''] || 500)})
              </h2>
            </div>

            {/* SVG Circular Progress Ring */}
            <div className="relative w-56 h-56 flex items-center justify-center">
              <svg className="w-56 h-56 -rotate-90 transform" viewBox="0 0 160 160">
                <circle
                  cx="80"
                  cy="80"
                  r="64"
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeWidth="8"
                  fill="transparent"
                />
                <circle
                  cx="80"
                  cy="80"
                  r="64"
                  stroke="url(#download-ring-gradient)"
                  strokeWidth="8"
                  strokeDasharray={402.12}
                  strokeDashoffset={402.12 - (Math.min(100, Math.max(0, progressPct)) / 100) * 402.12}
                  strokeLinecap="round"
                  fill="transparent"
                  className="transition-all duration-300 ease-out"
                />
                <defs>
                  <linearGradient id="download-ring-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="50%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#10b981" />
                  </linearGradient>
                </defs>
              </svg>

              <div className="absolute inset-0 flex flex-col items-center justify-center text-center select-none pointer-events-none">
                {downloadError ? (
                  <div className="flex flex-col items-center animate-fade-in px-4">
                    <AlertCircle size={40} className="text-red-400 mb-1" />
                    <span className="text-xs font-semibold text-red-300">Errore Download</span>
                  </div>
                ) : progressPct >= 100 ? (
                  <div className="flex flex-col items-center animate-fade-in">
                    <CheckCircle2 size={42} className="text-emerald-400 mb-1" />
                    <span className="text-xs font-semibold text-emerald-300">Completato ✓</span>
                  </div>
                ) : (
                  <>
                    <span className="text-4xl font-bold font-mono tracking-tight text-white">
                      {progressPct}%
                    </span>
                    <span className="text-xs font-mono text-white/50 mt-1">
                      {downloadedMb} / ~{totalMb} MB
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Testi dinamici sotto la progress bar */}
            <div className="space-y-3 max-w-sm">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.06] border border-white/[0.12] text-xs font-medium text-white/90">
                {downloadError ? (
                  <AlertCircle size={13} className="text-red-400" />
                ) : progressPct >= 100 ? (
                  <CheckCircle2 size={13} className="text-emerald-400" />
                ) : (
                  <Loader2 size={13} className="text-blue-400 animate-spin" />
                )}
                <span>
                  {downloadError
                    ? 'Errore durante il download dal repository'
                    : activeInfo?.stage || (
                        progressPct < 90
                          ? `Download in corso (${queueIndex + 1}/${downloadQueue.length || 1})...`
                          : 'Ottimizzazione pesi in locale...'
                      )}
                </span>
              </div>

              <p className="text-xs text-white/50 leading-relaxed">
                {downloadError
                  ? `${downloadError}`
                  : 'I componenti risiedono interamente nella memoria locale del tuo computer.'}
              </p>

              {downloadError && (
                <div className="pt-2 flex items-center justify-center gap-3">
                  <button
                    type="button"
                    onClick={handleRetryCurrent}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-500 hover:bg-blue-600 text-white transition-colors cursor-pointer"
                  >
                    <RefreshCw size={12} />
                    <span>Riprova</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsDownloadingScreen(false);
                      setActiveDownloadModel(null);
                    }}
                    className="px-4 py-2 rounded-xl text-xs font-medium text-white/70 hover:text-white bg-white/10 hover:bg-white/15 transition-colors cursor-pointer"
                  >
                    Torna al Wizard
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

      </main>

      {/* ── Fixed Footer Navigation Bar ── */}
      <footer className={clsx(
        "h-20 px-10 border-t border-white/[0.08] liquid-glass flex items-center flex-shrink-0 z-40",
        isDownloadingScreen ? "justify-end" : "justify-between"
      )}>
        {isDownloadingScreen ? (
          <div>
            <button
              type="button"
              onClick={handleInterruptDownload}
              disabled={isCancelling}
              className="flex items-center gap-2 px-6 py-2.5 rounded-2xl text-xs font-semibold bg-red-500/20 text-red-200 border border-red-500/35 hover:bg-red-500/30 transition-all cursor-pointer shadow-liquid-sm disabled:opacity-50"
            >
              {isCancelling ? (
                <>
                  <Loader2 size={14} className="animate-spin text-red-300" />
                  <span>Interruzione in corso...</span>
                </>
              ) : (
                <>
                  <X size={14} className="text-red-300" />
                  <span>Interrompi</span>
                </>
              )}
            </button>
          </div>
        ) : (
          <>
            <div>
              {step > 1 ? (
                <button
                  type="button"
                  onClick={() => setStep((s) => (s - 1) as 1 | 2 | 3 | 4)}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
                >
                  <ArrowLeft size={14} />
                  <span>Indietro</span>
                </button>
              ) : (
                <span className="text-xs text-white/30">Passo 1 di 4</span>
              )}
            </div>

            <div className="flex items-center gap-3">
              {step < 4 ? (
                <button
                  type="button"
                  onClick={() => setStep((s) => (s + 1) as 1 | 2 | 3 | 4)}
                  className="flex items-center gap-2 px-6 py-2.5 rounded-2xl text-xs font-semibold text-white bg-white/[0.16] hover:bg-white/[0.24] border border-white/[0.22] shadow-liquid-sm transition-all cursor-pointer"
                >
                  <span>Continua</span>
                  <ArrowRight size={14} />
                </button>
              ) : isAllRequiredReady ? (
                /* ── STEP 4: TUTTO INSTALLATO -> INIZIA A STUDIARE ── */
                <button
                  type="button"
                  onClick={handleFinishSetup}
                  className="flex items-center gap-2 px-7 py-2.5 rounded-2xl text-xs font-semibold text-white bg-emerald-500 hover:bg-emerald-600 shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all cursor-pointer"
                >
                  <CheckCircle2 size={15} />
                  <span>Inizia a Studiare</span>
                </button>
              ) : (
                /* ── STEP 4: COMPONENTI DA SCARICARE -> PULSANTE DOWNLOAD DINAMICO ── */
                <button
                  type="button"
                  onClick={handleStartDownloads}
                  disabled={isTriggering}
                  className="flex items-center gap-2 px-7 py-2.5 rounded-2xl text-xs font-semibold text-white bg-blue-500 hover:bg-blue-600 shadow-[0_0_20px_rgba(59,130,246,0.35)] transition-all cursor-pointer disabled:opacity-50"
                >
                  {isTriggering ? (
                    <>
                      <Loader2 size={14} className="animate-spin text-white" />
                      <span>Avvio download...</span>
                    </>
                  ) : (
                    <>
                      <Download size={14} />
                      <span>
                        Scarica ({formatMbOrGb(totalMbNeeded)})
                      </span>
                    </>
                  )}
                </button>
              )}
            </div>
          </>
        )}
      </footer>

    </div>
  );
}
