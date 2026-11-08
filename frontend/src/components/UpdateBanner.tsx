// components/UpdateBanner.tsx — Banner Notifica Aggiornamento Disponibile (Sfondo solido ad alto contrasto)
import { useState } from 'react';
import { Download, X } from 'lucide-react';
import { useUpdater } from '../lib/useUpdater';
import { useSettings } from '../lib/SettingsContext';

interface UpdateBannerProps {
  hidden?: boolean;
}

export default function UpdateBanner({ hidden }: UpdateBannerProps) {
  const { settings } = useSettings();
  const { status, version, progress, error, install } = useUpdater({ autoCheck: settings.autoUpdate });
  const [dismissed, setDismissed] = useState(false);

  const busy = status === 'downloading' || status === 'installing';
  const visible = (status === 'available' || busy || (status === 'error' && !!version)) && (!dismissed || busy);
  if (!visible || hidden) return null;

  return (
    <div className="fixed bottom-8 right-8 z-40 w-[330px] p-4.5 rounded-3xl bg-[#131722] border border-white/20 shadow-2xl shadow-black/90 space-y-3 animate-fade-in select-none">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-2xl bg-blue-500/25 border border-blue-400/40 flex items-center justify-center text-blue-300 flex-shrink-0 shadow-sm">
          <Download size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-bold text-white tracking-tight">Aggiornamento disponibile</p>
          <p className="text-[11px] text-white/70 mt-0.5 leading-snug">SbobinAI {version} è pronto per l'installazione.</p>
        </div>
        {!busy && (
          <button
            type="button"
            onClick={() => setDismissed(true)}
            className="p-1 -mr-1 -mt-1 text-white/40 hover:text-white cursor-pointer transition-colors"
            aria-label="Chiudi"
          >
            <X size={14} />
          </button>
        )}
      </div>

      {busy ? (
        <div className="space-y-1.5 pt-1">
          <div className="h-2 rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-200"
              style={{ width: `${Math.max(4, progress)}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-[10px] font-mono text-white/60">
            <span>{status === 'installing' ? 'Installazione e riavvio...' : 'Download in corso'}</span>
            <span className="font-bold text-blue-300">{progress}%</span>
          </div>
        </div>
      ) : (
        <div className="pt-1">
          {error && <p className="text-[11px] text-red-300 mb-2 leading-relaxed">{error}</p>}
          <button
            type="button"
            onClick={() => void install()}
            className="w-full py-2.5 rounded-2xl bg-blue-500 hover:bg-blue-400 text-white text-xs font-semibold cursor-pointer transition-all shadow-liquid-sm active:scale-[0.98] flex items-center justify-center gap-2"
          >
            <Download size={13} />
            <span>Aggiorna e Riavvia</span>
          </button>
        </div>
      )}
    </div>
  );
}
