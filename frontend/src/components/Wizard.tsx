import { osBadge } from '../lib/platform';
import { useEffect, useState } from 'react';
import { 
  Sparkles, Folder, ArrowRight, ArrowLeft, 
  CheckCircle2, HardDrive, ShieldCheck, Zap, Laptop,
  Download, Loader2, X, AlertCircle, Bot, Check, Database, Mic
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

export default function Wizard() {
  const { settings, completeSetup } = useSettings();
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);

  // Form State
  const [storagePath, setStoragePath] = useState(settings.storagePath || '~/Documents/SbobinAI');
  const [llmModel, setLlmModel] = useState<string>('mlx-community/Qwen3.5-4B-MLX-4bit');
  const [isSelectingFolder, setIsSelectingFolder] = useState(false);

  // Stato Modelli Reale da Backend
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [activeDownloadModel, setActiveDownloadModel] = useState<string | null>(null);
  const [isDownloadingScreen, setIsDownloadingScreen] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Verifica continua dello stato dei modelli
  useEffect(() => {
    let isMounted = true;

    const checkStatus = async () => {
      try {
        const res = await api.models.status();
        if (!isMounted) return;
        setModelsStatus(res);

        // Se siamo nella schermata di download, verifica l'avanzamento sequenziale dei core
        if (isDownloadingScreen && activeDownloadModel) {
          const allModels = [...(res.whisper || []), ...(res.embedding || []), ...(res.llm || [])];
          const curInfo = allModels.find((m) => m.id === activeDownloadModel);

          if (curInfo?.status === 'completed') {
            // Se Whisper è completato, verifica se serve scaricare Qwen3-Embedding
            const embInfo = res.embedding?.find((m) => m.id === CORE_EMBEDDING_ID);
            const isEmbDone = embInfo?.downloaded === true;

            if (activeDownloadModel === CORE_WHISPER_ID && !isEmbDone) {
              setActiveDownloadModel(CORE_EMBEDDING_ID);
              await api.models.download(CORE_EMBEDDING_ID);
            } else {
              // Entrambi i core completati con successo
              setIsDownloadingScreen(false);
              setActiveDownloadModel(null);
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
  }, [step, isDownloadingScreen, activeDownloadModel]);

  const handleChooseFolder = async () => {
    setIsSelectingFolder(true);
    try {
      const selected = await pickFolder(storagePath);
      if (selected) {
        // Il percorso viene inviato al backend una sola volta al completamento del setup
        setStoragePath(selected);
      }
    } finally {
      setIsSelectingFolder(false);
    }
  };

  // Verifica presenza su disco dei due motori base obbligatori
  const isWhisperDownloaded = Boolean(
    modelsStatus?.whisper?.find((m) => m.id === CORE_WHISPER_ID)?.downloaded
  );
  const isEmbeddingDownloaded = Boolean(
    modelsStatus?.embedding?.find((m) => m.id === CORE_EMBEDDING_ID)?.downloaded
  );
  const isCoreReady = isWhisperDownloaded && isEmbeddingDownloaded;

  // Modello attualmente monitorato durante il download
  const allKnownModels = [
    ...(modelsStatus?.whisper || []),
    ...(modelsStatus?.embedding || []),
    ...(modelsStatus?.llm || []),
  ];
  const activeInfo = allKnownModels.find((m) => m.id === activeDownloadModel);
  const progressPct = activeInfo?.progress ?? 0;
  const downloadedMb = activeInfo?.downloaded_mb ?? 0;
  const totalMb = activeInfo?.total_mb ?? (activeDownloadModel === CORE_EMBEDDING_ID ? 300 : 580);

  // Avvio download dei componenti di sistema base obbligatori
  const handleStartCoreDownload = async () => {
    try {
      setDownloadError(null);
      setIsDownloadingScreen(true);

      if (!isWhisperDownloaded) {
        setActiveDownloadModel(CORE_WHISPER_ID);
        await api.models.download(CORE_WHISPER_ID);
      } else if (!isEmbeddingDownloaded) {
        setActiveDownloadModel(CORE_EMBEDDING_ID);
        await api.models.download(CORE_EMBEDDING_ID);
      }
    } catch (err: any) {
      console.error('[Wizard] Errore avvio download core:', err);
      setDownloadError(err?.message || 'Impossibile avviare il download dei componenti base.');
      setIsDownloadingScreen(false);
      setActiveDownloadModel(null);
    }
  };

  const handleInterruptDownload = async () => {
    if (!activeDownloadModel) return;
    setIsCancelling(true);
    try {
      await api.models.cancelDownload(activeDownloadModel);
    } catch (err) {
      console.warn('[Wizard] Errore stop download:', err);
    } finally {
      setIsCancelling(false);
      setIsDownloadingScreen(false);
      setActiveDownloadModel(null);
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

    // completeSetup esegue in sequenza: POST /settings/storage (atteso) e poi PATCH /settings
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
        <div className="flex items-center gap-2 min-w-[200px]">
          <div className="w-[72px] h-4 pointer-events-none" />
          <span className="text-[11px] font-medium tracking-tight text-white/35 uppercase font-mono pointer-events-none">
            SbobinAI
          </span>
        </div>

        {isDownloadingScreen ? (
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full liquid-glass-subtle border border-blue-500/30 text-xs text-blue-300 font-medium">
            <Loader2 size={12} className="animate-spin text-blue-400" />
            <span>
              Installazione Componente:{' '}
              {activeDownloadModel === CORE_WHISPER_ID
                ? 'Whisper Large Turbo (1/2)'
                : 'Qwen3-Embedding RAG (2/2)'}
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
            <div className="w-20 h-20 mx-auto rounded-3xl liquid-glass flex items-center justify-center border border-white/20 shadow-liquid-lg">
              <Sparkles size={40} className="text-white drop-shadow-[0_0_16px_rgba(255,255,255,0.7)]" />
            </div>

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
                SbobinAI memorizzerà in questa cartella il database SQLite delle lezioni, i file audio WAV registrati e l'indice vettoriale LanceDB.
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
                  : `Hai selezionato ${LLM_OPTIONS.find((o) => o.id === llmModel)?.name}. Sarà pronto all'uso sul tuo computer.`}
              </span>
            </div>
          </div>
        )}

        {/* ── STEP 4: COMPONENTI DI SISTEMA OBBLIGATORI (AUDIO & RAG) ── */}
        {step === 4 && !isDownloadingScreen && (
          <div className="animate-fade-in space-y-6 w-full max-w-2xl my-auto">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full liquid-glass-subtle border border-emerald-500/25 text-xs text-emerald-300 font-medium">
                <CheckCircle2 size={13} className="text-emerald-400" />
                <span>Installazione Obbligatoria Core Engine</span>
              </div>
              <h2 className="text-2xl font-bold tracking-tight text-white">
                Componenti di Sistema Obbligatori (Audio & Ricerca Semantica)
              </h2>
              <p className="text-xs text-white/60 leading-relaxed">
                SbobinAI richiede questi due motori nativi per poter catturare le lezioni e abilitare la memoria del corso. 
                L'installazione su disco è obbligatoria per sbloccare l'applicazione.
              </p>
            </div>

            {/* Le 2 Card dei Componenti Fissi di Sistema */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Card 1: Whisper Large-v3-Turbo Q4 */}
              <div className={clsx(
                'p-5 rounded-3xl border transition-all flex flex-col justify-between text-left space-y-3',
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
                      {isWhisperDownloaded ? 'Installato sul computer ✓' : 'Da Installare'}
                    </span>
                    <div className="w-8 h-8 rounded-xl bg-white/[0.08] flex items-center justify-center text-white/80">
                      <Mic size={15} />
                    </div>
                  </div>

                  <h3 className="text-sm font-bold text-white mb-0.5">Motore Vocale Whisper</h3>
                  <p className="text-[11px] font-mono text-white/40">Whisper Large-v3-Turbo Q4 (~580 MB)</p>
                  <p className="text-xs text-white/60 mt-2 leading-relaxed">
                    Trascrizione continua, fedele e priva di allucinazioni, accelerata in locale.
                  </p>
                </div>

                <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[11px] font-mono text-white/50">
                  <span>Stato: {isWhisperDownloaded ? 'Pronto' : 'Richiesto'}</span>
                  <span>~580 MB</span>
                </div>
              </div>

              {/* Card 2: Qwen3-Embedding-0.6B-4bit-DWQ */}
              <div className={clsx(
                'p-5 rounded-3xl border transition-all flex flex-col justify-between text-left space-y-3',
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
                      {isEmbeddingDownloaded ? 'Installato sul computer ✓' : 'Da Installare'}
                    </span>
                    <div className="w-8 h-8 rounded-xl bg-white/[0.08] flex items-center justify-center text-white/80">
                      <Database size={15} />
                    </div>
                  </div>

                  <h3 className="text-sm font-bold text-white mb-0.5">Motore Memoria RAG</h3>
                  <p className="text-[11px] font-mono text-white/40">Qwen3-Embedding 0.6B DWQ (~300 MB)</p>
                  <p className="text-xs text-white/60 mt-2 leading-relaxed">
                    Vettorializzazione 1024-dim nativa per la ricerca ibrida nei materiali e lezioni.
                  </p>
                </div>

                <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[11px] font-mono text-white/50">
                  <span>Stato: {isEmbeddingDownloaded ? 'Pronto' : 'Richiesto'}</span>
                  <span>~300 MB</span>
                </div>
              </div>
            </div>

            {/* Box informativo di stato */}
            <div className={clsx(
              'p-4 rounded-2xl border flex items-center gap-3 text-xs leading-relaxed',
              isCoreReady
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-200'
                : 'bg-amber-500/10 border-amber-500/25 text-amber-200'
            )}>
              {isCoreReady ? (
                <>
                  <CheckCircle2 size={18} className="text-emerald-400 flex-shrink-0" />
                  <span>
                    <strong>Componenti di sistema base verificati!</strong> Il tuo computer è pronto per registrare le lezioni e indicizzare i contenuti. Clicca su "Inizia a Studiare".
                  </span>
                </>
              ) : (
                <>
                  <AlertCircle size={18} className="text-amber-400 flex-shrink-0" />
                  <span>
                    <strong>Installazione necessaria:</strong> per poter usare SbobinAI, clicca sul pulsante sottostante per scaricare i componenti base mancanti.
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
                Installazione Componenti di Sistema Base
              </span>
              <h2 className="text-xl font-bold tracking-tight text-white">
                {activeDownloadModel === CORE_WHISPER_ID
                  ? 'Download Whisper Large-v3-Turbo Q4 (~580 MB)'
                  : 'Download Qwen3-Embedding-0.6B-4bit-DWQ (~300 MB)'}
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
                    <span className="text-xs font-semibold text-red-300">Errore Connessione</span>
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
            <div className="space-y-2 max-w-sm">
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
                    ? 'Errore di connessione ai server Hugging Face'
                    : activeInfo?.stage || (
                        progressPct < 90
                          ? `Download ${activeDownloadModel === CORE_WHISPER_ID ? 'Whisper Large Turbo' : 'Qwen3 Embedding'}...`
                          : 'Ottimizzazione pesi su in locale...'
                      )}
                </span>
              </div>

              <p className="text-xs text-white/50 leading-relaxed">
                {downloadError
                  ? `Si è verificato un errore: ${downloadError}. Clicca su "Interrompi" per riprovare.`
                  : 'I componenti risiedono interamente nella memoria locale del tuo computer.'}
              </p>
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
              ) : isCoreReady ? (
                /* ── STEP 4: CORE INSTALLATI -> INIZIA A STUDIARE (SBLOCCATO) ── */
                <button
                  type="button"
                  onClick={handleFinishSetup}
                  className="flex items-center gap-2 px-7 py-2.5 rounded-2xl text-xs font-semibold text-white bg-emerald-500 hover:bg-emerald-600 shadow-[0_0_20px_rgba(16,185,129,0.4)] transition-all cursor-pointer"
                >
                  <CheckCircle2 size={15} />
                  <span>Inizia a Studiare</span>
                </button>
              ) : (
                /* ── STEP 4: CORE MANCANTI -> PULSANTE DOWNLOAD OBBLIGATORIO ── */
                <button
                  type="button"
                  onClick={handleStartCoreDownload}
                  className="flex items-center gap-2 px-7 py-2.5 rounded-2xl text-xs font-semibold text-white bg-blue-500 hover:bg-blue-600 shadow-[0_0_20px_rgba(59,130,246,0.35)] transition-all cursor-pointer"
                >
                  <Download size={14} />
                  <span>
                    Installa Componenti di Sistema Base (~{(!isWhisperDownloaded && !isEmbeddingDownloaded) ? '880' : !isWhisperDownloaded ? '580' : '300'} MB)
                  </span>
                </button>
              )}
            </div>
          </>
        )}
      </footer>

    </div>
  );
}
