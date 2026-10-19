// components/UpdateBanner.tsx — Banner Liquid Glass per aggiornamento disponibile
import { useState } from 'react';
import { Download, X } from 'lucide-react';
import { useUpdater } from '../lib/useUpdater';

export default function UpdateBanner() {
  const { status, version, progress, error, install } = useUpdater({ autoCheck: true });
  const [dismissed, setDismissed] = useState(false);

  const busy = status === 'downloading' || status === 'installing';
  const visible = (status === 'available' || busy || (status === 'error' && !!version)) && (!dismissed || busy);
  if (!visible) return null;

  return (
    <div className="fixed bottom-5 right-5 z-[100] w-[320px] p-4 rounded-3xl liquid-glass border border-white/[0.14] shadow-liquid-sm space-y-3 animate-fade-in">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-2xl bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-300 flex-shrink-0">
          <Download size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-bold text-white">Aggiornamento disponibile</p>
          <p className="text-[11px] text-white/60 mt-0.5">SbobinAI {version} è pronto per l'installazione.</p>
        </div>
        {!busy && (
          <button
            type="button"
            onClick={() => setDismissed(true)}
            className="text-white/40 hover:text-white/80 cursor-pointer"
            aria-label="Chiudi"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {busy ? (
        <div className="space-y-1.5">
          <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div className="h-full bg-blue-400 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <p className="text-[10px] text-white/50 font-mono">
            {status === 'installing' ? 'Installazione e riavvio...' : `Download ${progress}%`}
          </p>
        </div>
      ) : (
        <>
          {error && <p className="text-[11px] text-red-300">{error}</p>}
          <button
            type="button"
            onClick={() => void install()}
            className="w-full py-2 rounded-full bg-blue-500 hover:bg-blue-400 text-white text-xs font-semibold cursor-pointer transition-colors"
          >
            Aggiorna e Riavvia
          </button>
        </>
      )}
    </div>
  );
}
