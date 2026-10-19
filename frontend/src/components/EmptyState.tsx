// components/EmptyState.tsx — Stato vuoto macOS Liquid Glass
import { BookOpen } from 'lucide-react';

export default function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 select-none bg-black/15">
      <div className="w-16 h-16 rounded-3xl liquid-glass border border-white/[0.12] flex items-center justify-center mb-4 shadow-liquid">
        <BookOpen size={24} strokeWidth={1.5} className="text-white/40" />
      </div>
      <h2 className="text-base font-semibold tracking-tight text-white/90">
        Nessun corso selezionato
      </h2>
      <p className="text-xs text-white/40 mt-1 max-w-xs text-center leading-relaxed">
        Seleziona una materia dalla barra laterale per aprire il quaderno delle lezioni o creane una nuova.
      </p>
    </div>
  );
}
