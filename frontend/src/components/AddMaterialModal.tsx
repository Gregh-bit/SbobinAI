// components/AddMaterialModal.tsx — Modale Liquid Glass con Selettore Tipologia Materiale
import { useState, useRef, useEffect } from 'react';
import { 
  X, FileText, BookOpen, Mic, Upload, 
  Check, Loader2, FileCode, AlignLeft
} from 'lucide-react';
import { clsx } from 'clsx';
import { api } from '../lib/api';
import type { CourseMaterial, Lecture, Subject } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  subject: Subject;
  onMaterialAdded: (material: CourseMaterial) => void;
  onLectureCreated?: (lecture: Lecture) => void;
}

type MaterialCategory = 'slide_pdf' | 'notes' | 'audio';

export default function AddMaterialModal({
  isOpen,
  onClose,
  subject,
  onMaterialAdded,
  onLectureCreated,
}: Props) {
  const [category, setCategory] = useState<MaterialCategory>('slide_pdf');
  
  // File state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customTitle, setCustomTitle] = useState('');
  
  // Notes specific state
  const [notesMode, setNotesMode] = useState<'upload' | 'write'>('upload');
  const [writtenNotes, setWrittenNotes] = useState('');
  const [includeInQuaderno, setIncludeInQuaderno] = useState(true);

  // Audio specific state
  const [audioDate, setAudioDate] = useState(new Date().toISOString().split('T')[0]);
  const [audioDurationMinutes, setAudioDurationMinutes] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset when opened
  useEffect(() => {
    if (isOpen) {
      setSelectedFile(null);
      setCustomTitle('');
      setWrittenNotes('');
      setNotesMode('upload');
      setIncludeInQuaderno(true);
      setAudioDate(new Date().toISOString().split('T')[0]);
      setAudioDurationMinutes('');
      setError(null);
      setLoading(false);
    }
  }, [isOpen]);

  // Keyboard shortcut Esc
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    if (!customTitle) {
      const cleanName = file.name.replace(/\.[^/.]+$/, '');
      setCustomTitle(cleanName);
    }
    // If it's a markdown or text file, optionally read it into writtenNotes
    if (file.name.endsWith('.md') || file.name.endsWith('.txt')) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setWrittenNotes((ev.target?.result as string) || '');
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (category === 'slide_pdf') {
        if (!selectedFile) {
          throw new Error('Seleziona un file PDF o presentazione per procedere.');
        }
        let uploadedMat: CourseMaterial;
        try {
          uploadedMat = await api.subjects.uploadMaterialFile(subject.id, selectedFile);
        } catch {
          // Fallback locale se backend non è stato riavviato
          const f_kb = selectedFile.size / 1024;
          uploadedMat = {
            id: selectedFile.name,
            name: customTitle ? `${customTitle}.${selectedFile.name.split('.').pop()}` : selectedFile.name,
            type: selectedFile.name.endsWith('.pdf') ? 'pdf' : 'slide',
            size: f_kb > 1024 ? `${(f_kb / 1024).toFixed(1)} MB` : `${f_kb.toFixed(1)} KB`,
            uploadedAt: 'Oggi',
            pageCount: 1,
            isAutoImported: false,
          };
        }
        onMaterialAdded(uploadedMat);
        onClose();
        return;
      }

      if (category === 'notes') {
        const title = (customTitle.trim() || selectedFile?.name.replace(/\.[^/.]+$/, '') || 'Appunti Lezione');
        let content = writtenNotes.trim();

        // Se ha caricato un file, proviamo comunque a salvarlo anche su disco nei materiali
        if (selectedFile) {
          try {
            await api.subjects.uploadMaterialFile(subject.id, selectedFile);
          } catch {
            // Ignora se non riavviato
          }
        }

        // Se selezionato, creiamo la lezione nel Quaderno
        const newLecture = await api.lectures.create({
          subject_id: subject.id,
          title,
          source_type: 'imported',
          processed_notes: content || `# ${title}\n\nAppunti importati nella memoria del corso.`,
          date: new Date().toISOString(),
          in_quaderno: includeInQuaderno,
        });

        if (onLectureCreated) {
          onLectureCreated(newLecture);
        }

        const newMat: CourseMaterial = {
          id: `lecture-${newLecture.id}`,
          name: newLecture.title,
          type: 'doc',
          size: 'Appunti',
          uploadedAt: 'Oggi',
          pageCount: 1,
          content: newLecture.processed_notes || '',
          isAutoImported: false,
          lecture_id: newLecture.id,
          in_quaderno: includeInQuaderno,
          source_type: 'imported',
        };
        onMaterialAdded(newMat);
        onClose();
        return;
      }

      if (category === 'audio') {
        const title = customTitle.trim() || (selectedFile?.name.replace(/\.[^/.]+$/, '') || 'Registrazione Lezione');
        const durationSec = audioDurationMinutes ? parseInt(audioDurationMinutes, 10) * 60 : undefined;

        if (selectedFile) {
          try {
            await api.subjects.uploadMaterialFile(subject.id, selectedFile);
          } catch {
            // Ignora se non riavviato
          }
        }

        const newLecture = await api.lectures.create({
          subject_id: subject.id,
          title,
          date: audioDate ? new Date(audioDate).toISOString() : new Date().toISOString(),
          duration_seconds: durationSec,
          source_type: 'imported',
          in_quaderno: includeInQuaderno,
          processed_notes: `# ${title}\n\nRegistrazione esterna importata nella memoria del corso.`,
        });

        if (onLectureCreated) {
          onLectureCreated(newLecture);
        }

        const newMat: CourseMaterial = {
          id: `lecture-${newLecture.id}`,
          name: newLecture.title,
          type: 'lecture',
          size: durationSec ? `${Math.floor(durationSec / 60)}m` : 'Audio',
          uploadedAt: 'Oggi',
          pageCount: 1,
          content: newLecture.processed_notes || '',
          isAutoImported: false,
          lecture_id: newLecture.id,
          duration_seconds: durationSec,
          in_quaderno: includeInQuaderno,
          source_type: 'imported',
        };
        onMaterialAdded(newMat);
        onClose();
        return;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Errore durante l\'aggiunta del materiale');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in select-none"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-lg rounded-2xl liquid-glass border border-white/[0.14] shadow-liquid-lg overflow-hidden animate-slide-up flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="h-12 px-5 border-b border-white/[0.08] flex items-center justify-between flex-shrink-0 bg-white/[0.02]">
          <div className="flex items-center gap-2.5">
            <Upload size={15} strokeWidth={1.75} className="text-white/60" />
            <span className="text-xs font-semibold tracking-tight text-white/90">
              Aggiungi alla Memoria del Corso
            </span>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center text-white/40 hover:text-white hover:bg-white/[0.08] transition-colors"
            title="Chiudi (Esc)"
          >
            <X size={14} strokeWidth={2} />
          </button>
        </div>

        {/* Modal Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5 overflow-y-auto max-h-[80vh]">
          {/* Selettore Tipologia di File (Le 3 Categorie) */}
          <div className="space-y-2">
            <label className="text-[11px] font-semibold text-white/50 tracking-wider uppercase">
              Tipologia Materiale
            </label>
            <div className="grid grid-cols-3 gap-2.5">
              {/* Opzione 1: Slide & PDF */}
              <button
                type="button"
                onClick={() => {
                  setCategory('slide_pdf');
                  setSelectedFile(null);
                }}
                className={clsx(
                  'p-3 rounded-xl border text-left transition-all flex flex-col justify-between h-24',
                  category === 'slide_pdf'
                    ? 'bg-rose-500/15 border-rose-500/40 shadow-liquid-sm'
                    : 'bg-white/[0.03] border-white/[0.08] hover:bg-white/[0.06] text-white/60'
                )}
              >
                <div className="flex items-center justify-between">
                  <div className={clsx(
                    'w-7 h-7 rounded-lg flex items-center justify-center',
                    category === 'slide_pdf' ? 'bg-rose-500/25 text-rose-300' : 'bg-white/[0.06] text-white/50'
                  )}>
                    <FileText size={14} strokeWidth={2} />
                  </div>
                  {category === 'slide_pdf' && <Check size={13} className="text-rose-400" />}
                </div>
                <div>
                  <div className="text-xs font-semibold text-white/90">Slide / PDF</div>
                  <div className="text-[10px] text-white/40">Dispense & file</div>
                </div>
              </button>

              {/* Opzione 2: Appunti & Dispense */}
              <button
                type="button"
                onClick={() => {
                  setCategory('notes');
                  setSelectedFile(null);
                }}
                className={clsx(
                  'p-3 rounded-xl border text-left transition-all flex flex-col justify-between h-24',
                  category === 'notes'
                    ? 'bg-blue-500/15 border-blue-500/40 shadow-liquid-sm'
                    : 'bg-white/[0.03] border-white/[0.08] hover:bg-white/[0.06] text-white/60'
                )}
              >
                <div className="flex items-center justify-between">
                  <div className={clsx(
                    'w-7 h-7 rounded-lg flex items-center justify-center',
                    category === 'notes' ? 'bg-blue-500/25 text-blue-300' : 'bg-white/[0.06] text-white/50'
                  )}>
                    <BookOpen size={14} strokeWidth={2} />
                  </div>
                  {category === 'notes' && <Check size={13} className="text-blue-400" />}
                </div>
                <div>
                  <div className="text-xs font-semibold text-white/90">Appunti</div>
                  <div className="text-[10px] text-white/40">Markdown o testo</div>
                </div>
              </button>

              {/* Opzione 3: Registrazione Audio */}
              <button
                type="button"
                onClick={() => {
                  setCategory('audio');
                  setSelectedFile(null);
                }}
                className={clsx(
                  'p-3 rounded-xl border text-left transition-all flex flex-col justify-between h-24',
                  category === 'audio'
                    ? 'bg-amber-500/15 border-amber-500/40 shadow-liquid-sm'
                    : 'bg-white/[0.03] border-white/[0.08] hover:bg-white/[0.06] text-white/60'
                )}
              >
                <div className="flex items-center justify-between">
                  <div className={clsx(
                    'w-7 h-7 rounded-lg flex items-center justify-center',
                    category === 'audio' ? 'bg-amber-500/25 text-amber-300' : 'bg-white/[0.06] text-white/50'
                  )}>
                    <Mic size={14} strokeWidth={2} />
                  </div>
                  {category === 'audio' && <Check size={13} className="text-amber-400" />}
                </div>
                <div>
                  <div className="text-xs font-semibold text-white/90">Audio Esterno</div>
                  <div className="text-[10px] text-white/40">Registrazione</div>
                </div>
              </button>
            </div>
          </div>

          {/* ── Campi Contestuali in Base alla Tipologia ── */}

          {/* CASO 1: SLIDE / PDF */}
          {category === 'slide_pdf' && (
            <div className="space-y-3 animate-fade-in">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.pptx,.ppt,.key"
                onChange={handleFileChange}
                className="hidden"
              />
              
              <div
                onClick={() => fileInputRef.current?.click()}
                className={clsx(
                  'border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all',
                  selectedFile
                    ? 'border-rose-400/40 bg-rose-500/[0.06]'
                    : 'border-white/[0.12] bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/[0.20]'
                )}
              >
                <div className="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/[0.10] flex items-center justify-center text-white/70 mb-2">
                  <Upload size={18} strokeWidth={1.75} />
                </div>
                {selectedFile ? (
                  <div className="text-center">
                    <p className="text-xs font-semibold text-white/95">{selectedFile.name}</p>
                    <p className="text-[10px] font-mono text-white/40 mt-0.5">
                      {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB · Clicca per cambiare
                    </p>
                  </div>
                ) : (
                  <div className="text-center">
                    <p className="text-xs font-medium text-white/80">Trascina o clicca per selezionare</p>
                    <p className="text-[10px] text-white/40 mt-0.5">Formati supportati: .pdf, .pptx, .ppt, .key</p>
                  </div>
                )}
              </div>

              <div>
                <label className="text-[11px] font-semibold text-white/40 block mb-1">
                  Titolo Personalizzato (opzionale)
                </label>
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Es. Slide Capitolo 3 — Reti di Calcolatori"
                  className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                />
              </div>
            </div>
          )}

          {/* CASO 2: APPUNTI & DISPENSE */}
          {category === 'notes' && (
            <div className="space-y-3 animate-fade-in">
              <div className="flex items-center p-0.5 rounded-xl liquid-glass-subtle border border-white/[0.08] w-fit mb-1">
                <button
                  type="button"
                  onClick={() => setNotesMode('upload')}
                  className={clsx(
                    'flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all',
                    notesMode === 'upload' ? 'bg-white/[0.14] text-white' : 'text-white/40 hover:text-white'
                  )}
                >
                  <FileCode size={12} />
                  <span>Carica File (.md, .txt)</span>
                </button>
                <button
                  type="button"
                  onClick={() => setNotesMode('write')}
                  className={clsx(
                    'flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium transition-all',
                    notesMode === 'write' ? 'bg-white/[0.14] text-white' : 'text-white/40 hover:text-white'
                  )}
                >
                  <AlignLeft size={12} />
                  <span>Scrivi o Incolla Testo</span>
                </button>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-white/40 block mb-1">
                  Titolo Appunti
                </label>
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Es. Appunti Lezione 5 — Sistemi Concorrenti"
                  className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                />
              </div>

              {notesMode === 'upload' ? (
                <>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".md,.txt"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className={clsx(
                      'border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer transition-all',
                      selectedFile
                        ? 'border-blue-400/40 bg-blue-500/[0.06]'
                        : 'border-white/[0.12] bg-white/[0.02] hover:bg-white/[0.05]'
                    )}
                  >
                    <Upload size={16} className="text-white/50 mb-1.5" />
                    {selectedFile ? (
                      <p className="text-xs text-white font-medium">{selectedFile.name}</p>
                    ) : (
                      <p className="text-xs text-white/70">Seleziona file .md o .txt</p>
                    )}
                  </div>
                </>
              ) : (
                <div>
                  <textarea
                    value={writtenNotes}
                    onChange={(e) => setWrittenNotes(e.target.value)}
                    rows={5}
                    placeholder="Incolla qui gli appunti presi durante la lezione o inviati dal tuo collega..."
                    className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none resize-none font-mono"
                  />
                </div>
              )}

              {/* Checkbox Includi nel Quaderno */}
              <label className="flex items-center gap-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={includeInQuaderno}
                  onChange={(e) => setIncludeInQuaderno(e.target.checked)}
                  className="rounded border-white/20 bg-white/10 text-white focus:ring-0"
                />
                <span className="text-xs text-white/80">Includi subito anche nella sezione Quaderno</span>
              </label>
            </div>
          )}

          {/* CASO 3: REGISTRAZIONE AUDIO */}
          {category === 'audio' && (
            <div className="space-y-3 animate-fade-in">
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,.aac,.ogg"
                onChange={handleFileChange}
                className="hidden"
              />

              <div
                onClick={() => fileInputRef.current?.click()}
                className={clsx(
                  'border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center cursor-pointer transition-all',
                  selectedFile
                    ? 'border-amber-400/40 bg-amber-500/[0.06]'
                    : 'border-white/[0.12] bg-white/[0.02] hover:bg-white/[0.05]'
                )}
              >
                <Mic size={16} className="text-white/50 mb-1.5" />
                {selectedFile ? (
                  <p className="text-xs text-white font-medium">{selectedFile.name}</p>
                ) : (
                  <p className="text-xs text-white/70">Seleziona traccia audio (.mp3, .wav, .m4a)</p>
                )}
              </div>

              <div>
                <label className="text-[11px] font-semibold text-white/40 block mb-1">
                  Titolo Lezione
                </label>
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  placeholder="Es. Lezione Registrata — Teoria delle Code"
                  className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-semibold text-white/40 block mb-1">
                    Data Lezione
                  </label>
                  <input
                    type="date"
                    value={audioDate}
                    onChange={(e) => setAudioDate(e.target.value)}
                    className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-semibold text-white/40 block mb-1">
                    Durata (minuti)
                  </label>
                  <input
                    type="number"
                    min="1"
                    value={audioDurationMinutes}
                    onChange={(e) => setAudioDurationMinutes(e.target.value)}
                    placeholder="Es. 90"
                    className="w-full px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                  />
                </div>
              </div>

              {/* Checkbox Includi nel Quaderno */}
              <label className="flex items-center gap-2 cursor-pointer pt-1">
                <input
                  type="checkbox"
                  checked={includeInQuaderno}
                  onChange={(e) => setIncludeInQuaderno(e.target.checked)}
                  className="rounded border-white/20 bg-white/10 text-white focus:ring-0"
                />
                <span className="text-xs text-white/80">Includi subito anche nella sezione Quaderno</span>
              </label>
            </div>
          )}

          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 px-3 py-2 rounded-xl">
              {error}
            </p>
          )}

          {/* Action buttons */}
          <div className="pt-2 flex items-center justify-end gap-2 border-t border-white/[0.08]">
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-4 py-2 rounded-xl text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.06] transition-colors"
            >
              Annulla
            </button>
            <button
              type="submit"
              disabled={loading || (category === 'slide_pdf' && !selectedFile)}
              className="flex items-center gap-1.5 px-5 py-2 rounded-xl bg-white/[0.18] hover:bg-white/[0.28] text-white text-xs font-semibold border border-white/[0.24] shadow-liquid transition-all disabled:opacity-40"
            >
              {loading ? <Loader2 size={13} className="animate-spin" /> : <Check size={13} strokeWidth={2.5} />}
              <span>Aggiungi a Materiali</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

