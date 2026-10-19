import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Mic, Square, RotateCcw, BookOpen } from 'lucide-react';
import { clsx } from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import type { ChatMessage, Subject } from '../types';

import { api } from '../lib/api';
import MarkdownContent from './MarkdownContent';

interface Props {
  subject: Subject;
}

const INITIAL_MESSAGES: Record<string, ChatMessage[]> = {
  default: [
    {
      id: '1',
      role: 'assistant',
      content: 'Ciao! Ho indicizzato le lezioni e i materiali di questo corso. Puoi farmi qualsiasi domanda sul programma, chiedere chiarimenti o simulare le domande dell\'esame.',
      timestamp: Date.now() - 60000,
    }
  ]
};

const SUGGESTED_QUERIES = [
  'Riassumi i punti salienti dell\'ultima lezione',
  'Quali sono le tipiche domande d\'esame del professore?',
  'Spiegami il concetto più complesso trattato nel corso',
];

export default function AssistenteCorsoView({ subject }: Props) {
  const chatStorageKey = `sbobinai_chat_subject_${subject.id}`;

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(chatStorageKey);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (e) {
      console.warn('Errore lettura chat da localStorage:', e);
    }
    return INITIAL_MESSAGES.default;
  });
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Salva i messaggi ogni volta che cambiano per mantenere la chat anche cambiando schermata
  useEffect(() => {
    try {
      localStorage.setItem(chatStorageKey, JSON.stringify(messages));
    } catch (e) {
      console.warn('Errore salvataggio chat in localStorage:', e);
    }
  }, [messages, chatStorageKey]);

  // Se l'utente seleziona una materia diversa, carica la chat di quella specifica materia
  useEffect(() => {
    try {
      const saved = localStorage.getItem(`sbobinai_chat_subject_${subject.id}`);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
          return;
        }
      }
    } catch {
      // ignore
    }
    setMessages(INITIAL_MESSAGES.default);
  }, [subject.id]);

  const handleClearChat = () => {
    try {
      localStorage.removeItem(chatStorageKey);
    } catch {
      // ignore
    }
    setMessages(INITIAL_MESSAGES.default);
  };

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // ignore
        }
      }
    };
  }, []);

  const toggleListening = () => {
    if (isListening) {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // ignore
        }
      }
      setIsListening(false);
      return;
    }

    const SpeechRecognition =
      (window as unknown as { SpeechRecognition?: any }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: any }).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'it-IT';
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        if (transcript) {
          setInput(transcript);
        }
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || input).trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const assistantMsgId = crypto.randomUUID();
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      contextUsed: `${subject.name} · Archivio del corso`,
    };

    const controller = new AbortController();
    abortControllerRef.current = controller;
    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setInput('');
    setLoading(true);

    try {
      let accumulated = '';
      for await (const chunk of api.subjects.chatStream(subject.id, text, undefined, { signal: controller.signal })) {
        if (chunk.context_used) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, contextUsed: chunk.context_used } : m
            )
          );
        }
        if (chunk.token) {
          accumulated += chunk.token;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantMsgId ? { ...m, content: accumulated } : m))
          );
        }
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                content: m.content
                  ? m.content + '\n\n*(Risposta interrotta su richiesta)*'
                  : `Impossibile completare la risposta: ${err.message || err}`,
              }
            : m
        )
      );
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleCancelChat = async () => {
    setLoading(false);
    if (abortControllerRef.current) {
      try {
        abortControllerRef.current.abort();
      } catch {
        // ignore
      }
      abortControllerRef.current = null;
    }
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === 'assistant') {
        return [
          ...prev.slice(0, -1),
          {
            ...last,
            content: last.content
              ? `${last.content}\n\n*(Risposta interrotta su richiesta. Memoria Metal e cache liberate)*`
              : 'Risposta interrotta su richiesta. Memoria Metal liberata e processi azzerati.',
          },
        ];
      }
      return prev;
    });
    try {
      await api.ai.stop(true);
    } catch {
      // ignore
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-black/15 select-none">
      {/* Header Info */}
      <div className="px-8 py-2.5 border-b border-white/[0.06] flex items-center justify-between text-xs text-white/50">
        <div className="flex items-center gap-2">
          <BookOpen size={13} strokeWidth={1.75} className="text-white/60" />
          <span className="text-[11px] font-medium text-white/70">Archivio del Corso · {subject.name}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-medium text-emerald-400/90 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Memoria locale attiva
          </span>
          {messages.length > 1 && (
            <button
              type="button"
              onClick={handleClearChat}
              title="Azzera la cronologia di questa materia e avvia una nuova chat"
              className="flex items-center gap-1 px-2.5 py-1 rounded-xl liquid-glass-subtle hover:bg-white/[0.08] text-[11px] text-white/50 hover:text-white transition-all cursor-pointer border border-white/[0.08]"
            >
              <RotateCcw size={11} />
              <span>Nuova chat</span>
            </button>
          )}
        </div>
      </div>

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          {messages.map((m) => {
            const isUser = m.role === 'user';
            return (
              <div
                key={m.id}
                className={clsx('flex', isUser ? 'justify-end' : 'justify-start')}
              >
                <div
                  className={clsx(
                    'max-w-[85%] rounded-2xl p-4 text-xs tracking-tight leading-relaxed animate-fade-in',
                    isUser
                      ? 'bg-white/[0.14] text-white border border-white/[0.16] shadow-liquid-sm rounded-br-sm'
                      : 'liquid-glass text-white/90 border border-white/[0.10] shadow-liquid rounded-bl-sm'
                  )}
                >
                  {!isUser && (
                    <div className="flex flex-col gap-1 mb-2">
                      <div className="flex items-center gap-1.5 text-white/50 text-[10px] font-semibold tracking-wide">
                        <Sparkles size={11} strokeWidth={2} className="text-indigo-400" />
                        <span className="text-white/80">SbobinAI</span>
                      </div>
                      {m.contextUsed && (
                        <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.08] text-[10px] text-indigo-300 font-mono max-w-full">
                          <BookOpen size={10} className="flex-shrink-0 text-indigo-400/80" />
                          <span className="truncate">Fonti: {m.contextUsed}</span>
                        </div>
                      )}
                    </div>
                  )}
                  {m.content ? (
                    isUser ? (
                      <p className="whitespace-pre-wrap">{m.content}</p>
                    ) : (
                      <MarkdownContent content={m.content} />
                    )
                  ) : (
                    <div className="flex items-center gap-2 py-1 text-white/50 text-xs">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping" />
                      <span>Consultazione memoria del corso e generazione...</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          <div ref={endRef} />
        </div>
      </div>

      {/* Suggested Queries: visibili solo se la conversazione è ancora all'inizio, con uscita fluida */}
      <AnimatePresence>
        {messages.length <= 1 && (
          <motion.div
            key="suggested-queries-bar"
            initial={{ opacity: 0, y: 10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            exit={{ opacity: 0, y: -12, height: 0, transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] } }}
            className="px-8 overflow-hidden"
          >
            <div className="max-w-3xl mx-auto flex items-center gap-2 overflow-x-auto no-scrollbar py-1 pb-3">
              {SUGGESTED_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q)}
                  className="flex-shrink-0 px-3 py-1 rounded-full liquid-glass-subtle hover:bg-white/[0.08] text-[11px] text-white/50 hover:text-white/90 border border-white/[0.08] transition-all cursor-pointer"
                >
                  {q}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Prompt Bar */}
      <div className="p-6 pt-0">
        <div className="max-w-3xl mx-auto relative liquid-glass rounded-2xl border border-white/[0.14] shadow-liquid p-1.5 flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder={`Chiedi qualsiasi cosa su ${subject.name}...`}
            className="flex-1 bg-transparent px-3 py-2 text-xs text-white placeholder-white/30 focus:outline-none"
          />

          <button
            type="button"
            onClick={toggleListening}
            title={isListening ? 'Interrompi dettatura vocale' : 'Dettatura vocale'}
            className={clsx(
              'w-7 h-7 rounded-xl flex items-center justify-center transition-all cursor-pointer',
              isListening
                ? 'bg-red-500/20 text-red-400 border border-red-500/35 animate-pulse'
                : 'text-white/40 hover:text-white hover:bg-white/[0.08]'
            )}
          >
            <Mic size={14} strokeWidth={isListening ? 2.5 : 1.75} />
          </button>

          {loading ? (
            <button
              type="button"
              onClick={handleCancelChat}
              title="Interrompi elaborazione IA e libera RAM"
              className="px-2.5 h-7 rounded-xl flex items-center gap-1.5 bg-red-500/25 hover:bg-red-500/35 text-red-300 border border-red-500/40 text-[11px] font-semibold transition-all cursor-pointer animate-pulse"
            >
              <Square size={10} fill="currentColor" strokeWidth={0} />
              <span>Stop</span>
            </button>
          ) : (
            <button
              onClick={() => handleSend()}
              disabled={!input.trim()}
              className="w-7 h-7 rounded-xl flex items-center justify-center bg-white/[0.16] hover:bg-white/[0.25] text-white transition-all disabled:opacity-30 disabled:cursor-not-allowed border border-white/[0.15]"
            >
              <Send size={12} strokeWidth={2} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
