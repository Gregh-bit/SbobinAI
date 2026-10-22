// App.tsx — SbobinAI root component (macOS Native Liquid Glass)
import { useCallback, useEffect, useState, lazy, Suspense } from 'react';
import { api, checkBackendHealth } from './lib/api';
import Sidebar from './components/Sidebar';
import SubjectDashboard from './components/SubjectDashboard';
import EmptyState from './components/EmptyState';
import UpdateBanner from './components/UpdateBanner';
import ErrorBoundary from './components/ErrorBoundary';
import { SettingsProvider, useSettings } from './lib/SettingsContext';
import type { Lecture, Subject } from './types';

// Lazy loading componenti pesanti non visibili all'avvio (<100ms startup)
const Wizard = lazy(() => import('./components/Wizard'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

function BackendLoadingGate({ children }: { children: React.ReactNode }) {
  const [isReady, setIsReady] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Avvio del motore locale SbobinAI...');
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const maxAttempts = 75; // ~60 secondi

    const check = async () => {
      if (cancelled) return;

      // 1. Controlla lo stato dettagliato da Tauri (se attivo)
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const tauriStatus: any = await invoke('get_backend_status');
        if (tauriStatus && !cancelled) {
          if (tauriStatus.message) {
            setStatusMessage(tauriStatus.message);
          }
          if (tauriStatus.status === 'error') {
            setError(tauriStatus.error || tauriStatus.message || 'Errore di avvio del motore.');
            return;
          }
          if (tauriStatus.port_ready) {
            setIsReady(true);
            return;
          }
        }
      } catch {
        // Fuori da Tauri (in browser/dev) o comando non ancora pronto
      }

      // 2. Verifica se la porta HTTP 8000 risponde al healthcheck
      const healthy = await checkBackendHealth(1200);
      if (cancelled) return;

      if (healthy) {
        setIsReady(true);
        return;
      }

      attempts++;
      if (attempts >= maxAttempts) {
        setError(
          'Il motore locale non ha risposto entro il tempo previsto. Assicurati che Python sia installato e consulta il log in ~/Library/Logs/SbobinAI/backend.log.'
        );
        return;
      }

      setTimeout(check, 800);
    };

    check();

    return () => {
      cancelled = true;
    };
  }, [retryCount]);

  if (!isReady) {
    return (
      <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0B0F19] text-white select-none px-6">
        <div className="flex flex-col items-center max-w-sm text-center">
          {/* Logo SbobinAI Glow */}
          <div className="relative mb-6">
            <div className="absolute -inset-4 bg-gradient-to-r from-blue-600/30 via-indigo-500/20 to-purple-600/30 rounded-3xl blur-2xl animate-pulse" />
            <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-b from-white/10 to-white/5 border border-white/20 shadow-2xl flex items-center justify-center backdrop-blur-xl">
              <span className="text-3xl font-extrabold tracking-tight bg-gradient-to-br from-white via-white/90 to-white/60 bg-clip-text text-transparent">
                Sb
              </span>
            </div>
          </div>

          <h1 className="text-xl font-bold tracking-tight mb-2">SbobinAI</h1>

          {error ? (
            <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/25 text-left text-xs w-full max-w-xs">
              <p className="text-red-400 font-semibold mb-1">Avvio del motore non riuscito</p>
              <p className="text-red-200/80 mb-3 break-words">{error}</p>
              <button
                onClick={() => {
                  setError(null);
                  setRetryCount((c) => c + 1);
                }}
                className="w-full py-2 px-3 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-200 text-xs font-medium transition cursor-pointer"
              >
                Riprova
              </button>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-2.5 mt-3 mb-1.5">
                <div className="w-3.5 h-3.5 rounded-full border-2 border-blue-400/30 border-t-blue-400 animate-spin" />
                <span className="text-xs text-white/75 font-medium">
                  {statusMessage}
                </span>
              </div>
              <p className="text-[11px] text-white/40">
                Elaborazione 100% offline e locale sul tuo dispositivo
              </p>
            </>
          )}
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

function MainApp() {
  const { settings } = useSettings();
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [, setActiveLecture] = useState<Lecture | null>(null);
  const [loadingSubjects, setLoadingSubjects] = useState(true);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const loadSubjects = useCallback(async (retries = 5) => {
    setLoadingSubjects(true);
    try {
      const data = await api.subjects.list();
      setSubjects(data);
      setSelectedSubject((prev) => (prev ? prev : (data[0] || null)));
    } catch {
      // Backend in fase di avvio: riprova ogni secondo per 5 volte
      if (retries > 0) {
        setTimeout(() => loadSubjects(retries - 1), 1000);
      }
    } finally {
      setLoadingSubjects(false);
    }
  }, []);

  useEffect(() => {
    loadSubjects();
  }, [loadSubjects]);

  const handleAddSubject = async (form: {
    name: string;
    description: string;
    keywords: string;
    color: string;
  }) => {
    const subject = await api.subjects.create({
      name: form.name,
      description: form.description || undefined,
      whisper_prompt_keywords: form.keywords || undefined,
      color: form.color,
    });
    setSubjects((prev) => [...prev, subject]);
    setSelectedSubject(subject);
  };

  const handleUpdateSubject = async (
    id: number,
    form: { name: string; description: string; keywords: string; color: string }
  ) => {
    try {
      // null (e non undefined) per permettere di svuotare descrizione e parole chiave
      const updated = await api.subjects.update(id, {
        name: form.name,
        description: form.description.trim() ? form.description : null,
        whisper_prompt_keywords: form.keywords.trim() ? form.keywords : null,
        color: form.color,
      });
      setSubjects((prev) => prev.map((s) => (s.id === id ? { ...s, ...updated } : s)));
      if (selectedSubject?.id === id) {
        setSelectedSubject((prev) => (prev ? { ...prev, ...updated } : updated));
      }
    } catch (e) {
      console.error('Errore modifica corso:', e);
      throw e;
    }
  };

  const handleDeleteSubject = async (id: number) => {
    // Se l'eliminazione fallisce l'errore viene propagato (mostrato nella Sidebar) e il corso resta in lista
    await api.subjects.delete(id);
    const remaining = subjects.filter((s) => s.id !== id);
    setSubjects(remaining);
    if (selectedSubject?.id === id) {
      setSelectedSubject(remaining[0] || null);
      setActiveLecture(null);
    }
  };

  const handleSelectSubject = (subject: Subject) => {
    // Durante una registrazione live non è possibile cambiare corso
    if (isRecording && subject.id !== selectedSubject?.id) return;
    setSelectedSubject(subject);
    setActiveLecture(null);
  };

  const handleStartLecture = (lecture: Lecture) => {
    setActiveLecture(lecture);
  };

  return (
    <>
      <UpdateBanner />
      {/* Onboarding Wizard a tutto schermo se setupComplete è false */}
      {!settings.setupComplete ? (
        <Suspense fallback={<div className="h-screen w-screen flex items-center justify-center bg-black/40"><div className="w-8 h-8 rounded-full border-2 border-white/20 border-t-white animate-spin" /></div>}>
          <Wizard />
        </Suspense>
      ) : (
        /* Main App Workspace */
        <div className="flex h-screen overflow-hidden bg-transparent select-none rounded-[12px]">
          <Sidebar
            subjects={subjects}
            selectedId={selectedSubject?.id ?? null}
            onSelect={handleSelectSubject}
            onAdd={handleAddSubject}
            onUpdate={handleUpdateSubject}
            onDelete={handleDeleteSubject}
            isLoading={loadingSubjects}
            isRecording={isRecording}
            onOpenSettings={() => setShowSettingsModal(true)}
          />

          <main className="flex-1 flex flex-col overflow-hidden relative">
            {selectedSubject ? (
              <SubjectDashboard
                key={selectedSubject.id}
                subject={selectedSubject}
                onRecordingChange={setIsRecording}
                onStartLecture={handleStartLecture}
                onOpenSettings={() => setShowSettingsModal(true)}
              />
            ) : (
              <EmptyState />
            )}
          </main>
        </div>
      )}

      {/* Modal Impostazioni lazy loaded */}
      {showSettingsModal && (
        <ErrorBoundary fallbackTitle="Errore nel caricamento delle impostazioni">
          <Suspense fallback={null}>
            <SettingsModal
              isOpen={showSettingsModal}
              onClose={() => setShowSettingsModal(false)}
            />
          </Suspense>
        </ErrorBoundary>
      )}
    </>
  );
}

export default function App() {
  return (
    <BackendLoadingGate>
      <SettingsProvider>
        <MainApp />
      </SettingsProvider>
    </BackendLoadingGate>
  );
}
