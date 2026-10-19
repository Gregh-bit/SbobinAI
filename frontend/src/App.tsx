// App.tsx — SbobinAI root component (macOS Native Liquid Glass)
import { useCallback, useEffect, useState, lazy, Suspense } from 'react';
import { api } from './lib/api';
import Sidebar from './components/Sidebar';
import SubjectDashboard from './components/SubjectDashboard';
import EmptyState from './components/EmptyState';
import ErrorBoundary from './components/ErrorBoundary';
import { SettingsProvider, useSettings } from './lib/SettingsContext';
import type { Lecture, Subject } from './types';

// Lazy loading componenti pesanti non visibili all'avvio (<100ms startup)
const Wizard = lazy(() => import('./components/Wizard'));
const SettingsModal = lazy(() => import('./components/SettingsModal'));

function MainApp() {
  const { settings } = useSettings();
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [, setActiveLecture] = useState<Lecture | null>(null);
  const [loadingSubjects, setLoadingSubjects] = useState(true);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const loadSubjects = useCallback(async () => {
    setLoadingSubjects(true);
    try {
      const data = await api.subjects.list();
      setSubjects(data);
      setSelectedSubject((prev) => (prev ? prev : (data[0] || null)));
    } catch {
      // Backend non disponibile
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
    <SettingsProvider>
      <MainApp />
    </SettingsProvider>
  );
}
