import React, { useRef, useEffect } from 'react';
import { Mic, HelpCircle, MessageSquare, Sparkles, Send, Trash2, Square } from 'lucide-react';
import { clsx } from 'clsx';
import type { ChatMessage, DetectedQuestion, TranscriptChunk } from '../types';
import SegmentedControl from './SegmentedControl';
import MarkdownContent from './MarkdownContent';
import { api } from '../lib/api';

interface Props {
  chunks: TranscriptChunk[];
  isPaused?: boolean;
  audioStatus?: 'idle' | 'listening' | 'transcribing';
  detectedQuestions: DetectedQuestion[];
  onDeleteQuestion?: (id: string) => void;
  chatMessages: ChatMessage[];
  chatInput: string;
  setChatInput: (input: string) => void;
  isSendingChat: boolean;
  handleChatSend: (text?: string) => void;
  handleCancelChat?: () => void;
  inspectorTab: 'domande' | 'chat';
  setInspectorTab: (tab: 'domande' | 'chat') => void;
}

// ── Isolamento Rendering Live: Contenitore Trascrizione Memoizzato ────────────
interface LiveTranscriptContainerProps {
  chunks: TranscriptChunk[];
  isPaused: boolean;
  audioStatus: 'idle' | 'listening' | 'transcribing';
}

export const LiveTranscriptContainer = React.memo(function LiveTranscriptContainer({
  chunks,
  isPaused,
  audioStatus,
}: LiveTranscriptContainerProps) {
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chunks, audioStatus]);

  return (
    <section className="flex-1 flex flex-col h-full overflow-hidden relative pb-8 content-canvas">
      <div className="flex-1 overflow-y-auto px-12 py-10">
        <div className="max-w-2xl mx-auto space-y-4">
          {isPaused && (
            <div className="flex items-center justify-center mb-2 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs shadow-liquid-sm">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="font-medium">Registrazione in Pausa · Hardware a riposo (0% CPU/GPU)</span>
              </div>
            </div>
          )}

          {chunks.length === 0 ? (
            <div className="py-24 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 rounded-3xl liquid-glass border border-white/[0.14] flex items-center justify-center mb-4 shadow-liquid">
                {audioStatus === 'transcribing' ? (
                  <div className="w-6 h-6 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Mic
                    size={24}
                    strokeWidth={1.5}
                    className={clsx(
                      audioStatus === 'listening' ? "text-emerald-400 animate-pulse" : "text-white/60 animate-pulse-subtle"
                    )}
                  />
                )}
              </div>
              <h3 className="text-sm font-semibold tracking-tight text-white/90">
                {audioStatus === 'transcribing'
                  ? "Trascrizione Whisper Large Turbo in corso..."
                  : audioStatus === 'listening'
                  ? "Docente al microfono... cattura in corso"
                  : "In ascolto della lezione..."}
              </h3>
              <p className="text-xs text-white/50 mt-1 max-w-sm">
                {audioStatus === 'listening'
                  ? "L'audio viene accumulato ad alta fedeltà e trascritto con Whisper Large appena il docente fa una pausa."
                  : "Parla o avvia la spiegazione: la trascrizione apparirà qui con massima precisione."}
              </p>
              {audioStatus === 'listening' && (
                <div className="flex items-center gap-1.5 mt-4">
                  <span className="w-1.5 h-3 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="w-1.5 h-6 rounded-full bg-emerald-400 animate-pulse [animation-delay:150ms]" />
                  <span className="w-1.5 h-2 rounded-full bg-emerald-400 animate-pulse [animation-delay:300ms]" />
                  <span className="w-1.5 h-5 rounded-full bg-emerald-400 animate-pulse [animation-delay:75ms]" />
                  <span className="w-1.5 h-3 rounded-full bg-emerald-400 animate-pulse [animation-delay:225ms]" />
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              {chunks.map((c, i) => (
                <p
                  key={i}
                  className="text-[15px] leading-relaxed font-sans selection:bg-indigo-500/30 text-white/95 font-normal"
                >
                  {c.text}
                </p>
              ))}

              {/* Feedback Visivo Streaming Audio Engine */}
              {!isPaused && audioStatus === 'listening' && (
                <div className="flex items-center gap-2.5 pt-2 text-emerald-300 text-xs animate-fade-in">
                  <div className="flex items-center gap-1">
                    <span className="w-1 h-3 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="w-1 h-5 rounded-full bg-emerald-400 animate-pulse [animation-delay:150ms]" />
                    <span className="w-1 h-2 rounded-full bg-emerald-400 animate-pulse [animation-delay:300ms]" />
                    <span className="w-1 h-4 rounded-full bg-emerald-400 animate-pulse [animation-delay:75ms]" />
                  </div>
                  <span className="font-mono text-[11px] text-emerald-300/80">In ascolto del docente...</span>
                </div>
              )}

              {!isPaused && audioStatus === 'transcribing' && (
                <div className="flex items-center gap-2 pt-2 text-indigo-300/90 text-xs animate-fade-in">
                  <div className="w-3.5 h-3.5 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  <span className="font-mono text-[11px] text-indigo-200/80">Trascrizione Whisper Large Turbo in corso...</span>
                </div>
              )}
            </div>
          )}
          <div ref={transcriptEndRef} />
        </div>
      </div>
    </section>
  );
});

// ── LiveRecordingWorkspace / LiveLecture Container ───────────────────────────
function LiveRecordingWorkspaceComponent({
  chunks,
  isPaused = false,
  audioStatus = 'idle',
  detectedQuestions,
  onDeleteQuestion,
  chatMessages,
  chatInput,
  setChatInput,
  isSendingChat,
  handleChatSend,
  handleCancelChat,
  inspectorTab,
  setInspectorTab,
}: Props) {
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  return (
    <div className="flex-1 flex flex-row w-full h-full overflow-hidden relative select-none">
      {/* ── Foglio Centrale (Isolato tramite React.memo) ── */}
      <LiveTranscriptContainer
        chunks={chunks}
        isPaused={isPaused}
        audioStatus={audioStatus}
      />

      {/* ── Inspector Laterale Destro (30% larghezza) ── */}
      <aside className="w-80 flex-shrink-0 flex flex-col h-full border-l border-white/[0.08] bg-[#111520]/85 backdrop-blur-2xl z-10">
        {/* 2 Tab a pillola in alto con SegmentedControl */}
        <div className="p-3 border-b border-white/[0.06]">
          <SegmentedControl
            size="sm"
            layoutId="inspector-tab"
            fullWidth
            options={[
              {
                id: 'domande' as const,
                label: 'Domande Prof',
                icon: HelpCircle,
                badge: detectedQuestions.length > 0 ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                ) : undefined,
              },
              {
                id: 'chat' as const,
                label: 'Assistente AI',
                icon: MessageSquare,
              },
            ]}
            value={inspectorTab}
            onChange={setInspectorTab}
          />
        </div>

        {/* ── Pannello 1: Rilevamento Domande del Prof ── */}
        {inspectorTab === 'domande' && (
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            <div className="flex items-center justify-between pb-1">
              <span className="text-[10px] uppercase font-mono tracking-wider text-white/40">
                Domande Rilevate ({detectedQuestions.length})
              </span>
              <span className="text-[10px] text-white/30 font-mono">Live RAG</span>
            </div>

            {detectedQuestions.length === 0 ? (
              <div className="py-16 text-center">
                <HelpCircle size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-2" />
                <p className="text-xs text-white/40">Nessuna domanda del professore rilevata finora.</p>
                <p className="text-[11px] text-white/25 mt-1">
                  Whisper intercetta automaticamente le domande con intonazione interrogativa (?) e Qwen3.5 genera la risposta flash.
                </p>
              </div>
            ) : (
              detectedQuestions.map((q) => (
                <div
                  key={q.id}
                  className="p-3.5 rounded-2xl liquid-glass border border-white/[0.12] shadow-liquid-sm space-y-2 animate-slide-up group"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs font-semibold text-white/90 leading-snug">
                      "{q.question}"
                    </p>
                    {onDeleteQuestion && (
                      <button
                        type="button"
                        onClick={() => onDeleteQuestion(q.id)}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded-lg text-white/30 hover:text-red-400 hover:bg-white/[0.08] transition-all cursor-pointer"
                        title="Rimuovi domanda"
                      >
                        <Trash2 size={11} />
                      </button>
                    )}
                  </div>

                  {(q.answer || q.suggestedAnswer) ? (
                    <div className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.06] text-xs text-white/80 space-y-1">
                      <div className="flex items-center gap-1.5 text-[10px] text-amber-300/80 font-mono">
                        <Sparkles size={10} />
                        <span>Risposta flash suggerita:</span>
                      </div>
                      <p className="leading-relaxed text-[11px] font-sans">{q.answer || q.suggestedAnswer}</p>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-[10px] text-white/40 font-mono">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
                      <span>Generazione risposta flash in corso...</span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {/* ── Pannello 2: Chat Live "Non ho capito!" ── */}
        {inspectorTab === 'chat' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {chatMessages.length === 0 ? (
                <div className="py-16 text-center space-y-3">
                  <MessageSquare size={20} strokeWidth={1.5} className="mx-auto text-white/20 mb-1" />
                  <p className="text-xs text-white/50">Non hai capito un concetto spiegato dal docente?</p>
                  <div className="flex flex-col gap-1.5 max-w-[200px] mx-auto pt-2">
                    <button
                      type="button"
                      onClick={() => handleChatSend("Non ho capito l'ultimo concetto, puoi rispiegarmelo con un esempio pratico?")}
                      className="px-2.5 py-1.5 rounded-xl text-[11px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/70 hover:text-white transition-all cursor-pointer"
                    >
                      "Non ho capito l'ultimo concetto..."
                    </button>
                    <button
                      type="button"
                      onClick={() => handleChatSend("Qual è la differenza principale tra i due termini appena spiegati?")}
                      className="px-2.5 py-1.5 rounded-xl text-[11px] text-left bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-white/70 hover:text-white transition-all cursor-pointer"
                    >
                      "Qual è la differenza tra i due termini?"
                    </button>
                  </div>
                </div>
              ) : (
                chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={clsx(
                      'flex flex-col max-w-[85%] text-xs leading-relaxed animate-fade-in',
                      msg.role === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'
                    )}
                  >
                    <div
                      className={clsx(
                        'p-3 rounded-2xl break-words',
                        msg.role === 'user'
                          ? 'bg-blue-600/30 text-white border border-blue-500/40 rounded-br-sm'
                          : 'liquid-glass text-white/85 border border-white/[0.10] rounded-bl-sm shadow-liquid-sm'
                      )}
                    >
                      {msg.role === 'user' ? (
                        <p className="text-xs whitespace-pre-wrap">{msg.content}</p>
                      ) : (
                        <MarkdownContent content={msg.content} />
                      )}
                    </div>
                  </div>
                ))
              )}
              {isSendingChat && (
                <div className="flex items-center gap-1.5 text-xs text-white/40 p-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  <span className="font-mono text-[11px]">SbobinAI sta formulando la spiegazione...</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input Bar */}
            <div className="p-3 border-t border-white/[0.08] bg-black/20">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleChatSend()}
                  placeholder="Chiedi al volo..."
                  className="flex-1 px-3 py-2 rounded-xl liquid-input text-xs text-white placeholder-white/30 focus:outline-none"
                />
                {isSendingChat ? (
                  <button
                    type="button"
                    onClick={() => {
                      if (handleCancelChat) {
                        handleCancelChat();
                      } else {
                        api.ai.stop(true).catch(() => {});
                      }
                    }}
                    title="Interrompi risposta IA e libera RAM"
                    className="px-2.5 h-7 rounded-xl flex items-center gap-1.5 bg-red-500/25 hover:bg-red-500/35 text-red-300 border border-red-500/40 text-[11px] font-semibold transition-all cursor-pointer animate-pulse"
                  >
                    <Square size={10} fill="currentColor" strokeWidth={0} />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleChatSend()}
                    disabled={!chatInput.trim()}
                    className="w-7 h-7 rounded-xl flex items-center justify-center bg-white/[0.14] hover:bg-white/[0.22] text-white transition-all disabled:opacity-30 border border-white/[0.10] cursor-pointer"
                  >
                    <Send size={12} strokeWidth={2} />
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

export const LiveRecordingWorkspace = React.memo(LiveRecordingWorkspaceComponent);
export const LiveLecture = LiveRecordingWorkspace;
export default LiveRecordingWorkspace;
