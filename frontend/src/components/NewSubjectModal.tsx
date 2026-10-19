// components/NewSubjectModal.tsx — Modal Liquid Glass per creare un nuovo corso
import { useEffect, useState, type FormEvent } from 'react';
import { X, Tag, BookOpen } from 'lucide-react';
import { clsx } from 'clsx';

import type { Subject } from '../types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: { name: string; description: string; keywords: string; color: string }) => Promise<void>;
  subjectToEdit?: Subject | null;
}

const APPLE_PALETTE = [
  '#0071e3', // Apple Blue
  '#34c759', // Apple Green
  '#ff9500', // Apple Orange
  '#af52de', // Apple Purple
  '#ff2d55', // Apple Pink
  '#5ac8fa', // Apple Cyan
  '#a2845e', // Apple Gold / Warm Grey
  '#8e8e93', // Apple Grey
];

export default function NewSubjectModal({ isOpen, onClose, onSubmit, subjectToEdit }: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [keywords, setKeywords] = useState('');
  const [color, setColor] = useState(APPLE_PALETTE[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      if (subjectToEdit) {
        setName(subjectToEdit.name || '');
        setDescription(subjectToEdit.description || '');
        setKeywords(subjectToEdit.whisper_prompt_keywords || '');
        setColor(subjectToEdit.color || APPLE_PALETTE[0]);
      } else {
        setName('');
        setDescription('');
        setKeywords('');
        setColor(APPLE_PALETTE[0]);
      }
      setError('');
    }
  }, [isOpen, subjectToEdit]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setLoading(true);
    setError('');
    try {
      await onSubmit({ name: name.trim(), description: description.trim(), keywords: keywords.trim(), color });
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Errore nel salvataggio del corso');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-md animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="relative w-full max-w-lg rounded-2xl liquid-glass p-6 text-white/90 shadow-liquid-lg animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
          <div className="flex items-center gap-2.5">
            <div 
              className="w-7 h-7 rounded-lg flex items-center justify-center border border-white/10"
              style={{ backgroundColor: `${color}25` }}
            >
              <BookOpen size={14} style={{ color }} strokeWidth={2} />
            </div>
            <h2 className="text-base font-semibold tracking-tight text-white/95">
              {subjectToEdit ? 'Modifica Corso' : 'Nuovo Corso'}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="w-6 h-6 rounded-full flex items-center justify-center text-white/40 hover:text-white/80 hover:bg-white/[0.08] transition-colors"
          >
            <X size={14} strokeWidth={2} />
          </button>
        </div>

        {error && (
          <div className="mt-4 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div>
            <label className="block text-xs font-medium text-white/50 mb-1.5">Nome del Corso</label>
            <input
              type="text"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="es. Sistemi Operativi, Algoritmi, Diritto Privato..."
              required
              className="w-full px-3.5 py-2.5 rounded-xl liquid-input text-sm text-white placeholder-white/20 focus:outline-none transition-all"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-white/50">
                Informazioni & Descrizione (file InfoCorso)
              </label>
              <span className="text-[10px] text-emerald-400/80 font-mono">
                Auto-importato in Materiali
              </span>
            </div>
            <textarea
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="es. Prof. Rossi · 9 CFU · Orario: Lunedì 14-16 · Programma: Architetture distribuite, microservizi, pattern..."
              className="w-full px-3.5 py-2.5 rounded-xl liquid-input text-xs text-white placeholder-white/20 focus:outline-none transition-all resize-none font-sans"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-white/50 flex items-center gap-1.5">
                <Tag size={12} strokeWidth={1.75} />
                Keywords Vocabolario Tecnico (per Whisper)
              </label>
            </div>
            <input
              type="text"
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              placeholder="es. deadlock, mutex, paging, kernel, thread (separate da virgola)"
              className="w-full px-3.5 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/20 focus:outline-none transition-all font-mono"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-white/50 mb-2">Colore Icona</label>
            <div className="flex items-center gap-2">
              {APPLE_PALETTE.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setColor(c)}
                  className={clsx(
                    'w-6 h-6 rounded-full transition-transform border',
                    color === c ? 'scale-110 border-white shadow-[0_0_8px_rgba(255,255,255,0.4)]' : 'border-transparent opacity-70 hover:opacity-100'
                  )}
                  style={{ backgroundColor: c }}
                />
              ))}
            </div>
          </div>

          <div className="pt-4 flex items-center justify-end gap-2.5 border-t border-white/[0.08]">
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
              disabled={loading || !name.trim()}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-white bg-white/[0.12] hover:bg-white/[0.18] border border-white/[0.18] shadow-liquid-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              {loading ? (subjectToEdit ? 'Salvataggio...' : 'Creazione...') : (subjectToEdit ? 'Salva Modifiche' : 'Crea Corso')}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

