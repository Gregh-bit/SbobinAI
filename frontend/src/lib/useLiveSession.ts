import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from './api';
import { isTauri } from './tauri';
import { useSettings } from './SettingsContext';
import type { 
  ChatMessage, DetectedQuestion, Lecture, Subject, 
  TranscriptChunk, WSMessage 
} from '../types';

export function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) {
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function normalizeQuestion(text: string): string {
  return text.toLowerCase().replace(/[\?\.\,\s]/g, '');
}

// Tempo massimo di attesa del messaggio 'stopped' dal backend dopo l'invio di 'stop'
const STOP_ACK_TIMEOUT_MS = 10000;

interface UseLiveSessionProps {
  subject: Subject;
  lecture: Lecture | null;
  onStop: () => void;
}

export function useLiveSession({ subject, lecture, onStop }: UseLiveSessionProps) {
  const { settings } = useSettings();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioStatus, setAudioStatus] = useState<'idle' | 'listening' | 'transcribing'>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [chunks, setChunks] = useState<TranscriptChunk[]>([]);
  const [bookmarks, setBookmarks] = useState<{ id: number; timestamp: number; note: string }[]>([]);
  const [lastBookmarkPinned, setLastBookmarkPinned] = useState<number | null>(null);

  // Inspector State
  const [inspectorTab, setInspectorTab] = useState<'domande' | 'chat'>('domande');
  const [detectedQuestions, setDetectedQuestions] = useState<DetectedQuestion[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isSendingChat, setIsSendingChat] = useState(false);

  // Stato connessione / errori della registrazione
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedRef = useRef(0);
  const isPausedRef = useRef(isPaused);
  // Domande eliminate dall'utente: ignorate se arrivano di nuovo (es. con la risposta flash)
  const deletedQuestionIdsRef = useRef<Set<string>>(new Set());
  const deletedQuestionTextsRef = useRef<Set<string>>(new Set());
  // Chat live: query attiva e numero di risposte di richieste annullate ancora da scartare
  const activeChatQueryRef = useRef<string | null>(null);
  const cancelledChatResponsesRef = useRef(0);
  // Attesa del messaggio 'stopped' dopo l'invio di 'stop'
  const stopResolverRef = useRef<((acknowledged: boolean) => void) | null>(null);
  const isStoppingRef = useRef(false);

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  // Reset all state when a new lecture starts or is closed
  useEffect(() => {
    setElapsed(0);
    elapsedRef.current = 0;
    setChunks([]);
    setBookmarks([]);
    setChatMessages([]);
    setDetectedQuestions([]);
    setIsPaused(false);
    isPausedRef.current = false;
    setAudioStatus('idle');
    setConnectionError(null);
    setIsStopping(false);
    setIsSendingChat(false);
    isStoppingRef.current = false;
    deletedQuestionIdsRef.current = new Set();
    deletedQuestionTextsRef.current = new Set();
    activeChatQueryRef.current = null;
    cancelledChatResponsesRef.current = 0;
  }, [lecture?.id]);

  // ── Handle incoming WS messages ─────────────────────────────────────────────
  const handleWSMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case 'started':
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          if (!isPausedRef.current) {
            elapsedRef.current += 1;
            setElapsed(elapsedRef.current);
          }
        }, 1000);
        break;

      case 'status': {
        const st = (msg.payload.status as 'idle' | 'listening' | 'transcribing') || 'idle';
        setAudioStatus(st);
        break;
      }

      case 'transcript_chunk': {
        const text = (msg.payload.text as string) || '';
        if (!text.trim()) break;
        const chunk: TranscriptChunk = {
          text: text,
          elapsed: (msg.payload.elapsed as number) || 0,
          timestamp: Date.now(),
          speaker: 'Docente',
        };
        setChunks((prev) => [...prev, chunk]);
        setAudioStatus('idle');
        break;
      }

      case 'transcript_refined': {
        const refinedText = (msg.payload.text as string) || '';
        const elapsed = (msg.payload.elapsed as number) || 0;
        if (!refinedText.trim()) break;

        const refinedChunk: TranscriptChunk = {
          text: refinedText,
          elapsed: elapsed,
          timestamp: Date.now(),
          speaker: 'Docente',
        };
        setChunks((prev) => [...prev, refinedChunk]);
        setAudioStatus('idle');
        break;
      }

      case 'bookmark_saved':
        setBookmarks((prev) => [
          ...prev,
          {
            id: msg.payload.bookmark_id as number,
            timestamp: msg.payload.timestamp as number,
            note: (msg.payload.note as string) || '',
          }
        ]);
        setLastBookmarkPinned(Date.now());
        break;

      case 'chat_chunk': {
        const chunk = (msg.payload.chunk as string) || '';
        if (!chunk) break;
        // Scarta i frammenti di richieste annullate o non corrispondenti alla richiesta attiva
        if (cancelledChatResponsesRef.current > 0 || activeChatQueryRef.current === null) break;
        const chunkQuery = msg.payload.query as string | undefined;
        if (chunkQuery !== undefined && chunkQuery !== activeChatQueryRef.current) break;
        setChatMessages((prev) => {
          if (prev.length === 0 || prev[prev.length - 1].role !== 'assistant') {
            const newAssistant: ChatMessage = {
              id: generateId(),
              role: 'assistant',
              content: chunk,
              timestamp: Date.now(),
            };
            return [...prev, newAssistant];
          }
          const last = prev[prev.length - 1];
          const updatedLast: ChatMessage = {
            ...last,
            content: last.content + chunk,
          };
          return [...prev.slice(0, -1), updatedLast];
        });
        break;
      }

      case 'chat_response': {
        // Risposta finale di una richiesta annullata (il backend ne invia sempre una per richiesta)
        if (cancelledChatResponsesRef.current > 0) {
          cancelledChatResponsesRef.current -= 1;
          break;
        }
        if (activeChatQueryRef.current === null) break;
        activeChatQueryRef.current = null;
        const fullAnswer = (msg.payload.answer as string) || '';
        const contextUsed = msg.payload.context_used as string | undefined;
        setChatMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
            const last = prev[prev.length - 1];
            const updatedLast: ChatMessage = {
              ...last,
              content: fullAnswer || last.content,
              contextUsed: contextUsed || last.contextUsed,
            };
            return [...prev.slice(0, -1), updatedLast];
          }
          const answer: ChatMessage = {
            id: generateId(),
            role: 'assistant',
            content: fullAnswer,
            timestamp: Date.now(),
            contextUsed: contextUsed,
          };
          return [...prev, answer];
        });
        setIsSendingChat(false);
        break;
      }

      case 'question_detected': {
        const qText = (msg.payload.question as string) || '';
        if (!qText.trim()) break;
        const incomingId = msg.payload.id as string | undefined;
        if (
          (incomingId && deletedQuestionIdsRef.current.has(String(incomingId))) ||
          deletedQuestionTextsRef.current.has(normalizeQuestion(qText))
        ) {
          break;
        }

        const q: DetectedQuestion = {
          id: (msg.payload.id as string) || generateId(),
          question: qText,
          confidence: (msg.payload.confidence as number) || 0.9,
          timestamp: Date.now(),
          answer: (msg.payload.answer as string) || (msg.payload.suggested_answer as string) || undefined,
          suggestedAnswer: (msg.payload.suggested_answer as string) || (msg.payload.answer as string) || 'Domanda rilevata dal parlato del docente.',
          sourceLecture: 'Quaderno del corso',
        };

        setDetectedQuestions((prev) => {
          const cleanQ = normalizeQuestion(qText);
          const existingIdx = prev.findIndex((item) => {
            const itemClean = normalizeQuestion(item.question);
            return itemClean === cleanQ || (cleanQ.length > 15 && itemClean.includes(cleanQ)) || (itemClean.length > 15 && cleanQ.includes(itemClean));
          });
          if (existingIdx >= 0) {
            const updated = [...prev];
            updated[existingIdx] = { ...updated[existingIdx], ...q, id: updated[existingIdx].id };
            return updated;
          }
          return [q, ...prev];
        });
        break;
      }

      case 'paused':
        setIsPaused(true);
        isPausedRef.current = true;
        setAudioStatus('idle');
        break;

      case 'resumed':
        setIsPaused(false);
        isPausedRef.current = false;
        break;

      case 'stopped':
        stopTimer();
        setAudioStatus('idle');
        if (stopResolverRef.current) {
          // handleStop è in attesa di questa conferma: sarà lui a chiamare onStop
          stopResolverRef.current(true);
        } else if (!isStoppingRef.current) {
          onStop();
        }
        break;

      case 'error': {
        const message = (msg.payload?.message as string) || 'Errore sconosciuto del motore di registrazione.';
        console.warn('[LiveSession] Errore dal backend:', message);
        setConnectionError(message);
        // Un errore durante la chat non deve lasciare il pulsante bloccato
        if (activeChatQueryRef.current !== null) {
          activeChatQueryRef.current = null;
          setIsSendingChat(false);
        }
        break;
      }
    }
  }, [onStop]);

  const handleWSMessageRef = useRef(handleWSMessage);
  useEffect(() => {
    handleWSMessageRef.current = handleWSMessage;
  }, [handleWSMessage]);

  // ── WebSocket Lifecycle ─────────────────────────────────────────────────────
  // Il corso della WebSocket è quello della lezione in registrazione (fissato all'avvio)
  const sessionSubjectId = lecture?.subject_id ?? subject.id;

  useEffect(() => {
    if (!lecture) return;

    const ws = api.ws.liveLecture(sessionSubjectId);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setConnectionError(null);
      ws.send(JSON.stringify({
        type: 'start',
        payload: {
          title: lecture.title,
          lecture_id: lecture.id,
          model: settings.whisperModel,
          vad_sensitivity: settings.vadSensitivity,
        }
      }));
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data as string);
        handleWSMessageRef.current(msg);
      } catch {
        // Ignora frame non validi
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      stopTimer();
      if (wsRef.current === ws) wsRef.current = null;
      // Sblocca un'eventuale attesa di 'stopped' e una richiesta chat in corso
      if (stopResolverRef.current) stopResolverRef.current(false);
      if (activeChatQueryRef.current !== null) {
        activeChatQueryRef.current = null;
        setIsSendingChat(false);
      }
      if (!isStoppingRef.current) {
        setConnectionError((prev) => prev || 'Connessione con il motore di registrazione interrotta.');
      }
    };
    ws.onerror = () => {
      setIsConnected(false);
      if (!isStoppingRef.current) {
        setConnectionError('Impossibile connettersi al motore di registrazione. Verifica che il backend sia attivo.');
      }
    };

    return () => {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close();
      if (wsRef.current === ws) wsRef.current = null;
      stopTimer();
    };
  }, [sessionSubjectId, lecture?.id]);

  // ── Actions ─────────────────────────────────────────────────────────────────
  const handleBookmark = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setLastBookmarkPinned(Date.now());
    wsRef.current.send(JSON.stringify({
      type: 'bookmark',
      payload: { note: `Segnalibro @ ${formatElapsed(elapsedRef.current)}` }
    }));
  }, []);

  const handleStop = useCallback(async () => {
    if (isStoppingRef.current) return;
    isStoppingRef.current = true;
    setIsStopping(true);
    stopTimer();

    // Attende la conferma 'stopped' (il backend ha salvato trascrizione e audio) oppure il timeout
    let acknowledged = false;
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        const ackPromise = new Promise<boolean>((resolve) => {
          const timeout = setTimeout(() => resolve(false), STOP_ACK_TIMEOUT_MS);
          stopResolverRef.current = (ack: boolean) => {
            clearTimeout(timeout);
            resolve(ack);
          };
        });
        ws.send(JSON.stringify({ type: 'stop', payload: {} }));
        acknowledged = await ackPromise;
      } catch (err) {
        console.warn('Errore invio stop WebSocket:', err);
      } finally {
        stopResolverRef.current = null;
      }
    }

    // Fallback: se il backend non ha confermato, salva almeno durata e domande lato client
    if (!acknowledged && lecture) {
      try {
        await api.lectures.update(lecture.id, {
          duration_seconds: elapsedRef.current,
          detected_questions: detectedQuestions.length > 0 ? JSON.stringify(detectedQuestions) : null,
        });
      } catch {
        // ignore
      }
    }
    onStop();
  }, [lecture, onStop, detectedQuestions]);

  const handleChatSend = useCallback((textToSend?: string) => {
    const q = (textToSend || chatInput).trim();
    if (!q || isSendingChat) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setConnectionError("Connessione non disponibile: impossibile inviare la domanda all'assistente.");
      return;
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: q,
      timestamp: Date.now(),
    };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput('');
    setIsSendingChat(true);
    activeChatQueryRef.current = q;

    try {
      ws.send(JSON.stringify({
        type: 'chat',
        payload: { query: q, context_minutes: 5 }
      }));
    } catch (err) {
      console.warn('Errore invio chat WebSocket:', err);
      activeChatQueryRef.current = null;
      setIsSendingChat(false);
    }
  }, [chatInput, isSendingChat]);

  const handleCancelChat = useCallback(async () => {
    // La risposta della richiesta annullata arriverà comunque: va scartata
    if (activeChatQueryRef.current !== null) {
      cancelledChatResponsesRef.current += 1;
      activeChatQueryRef.current = null;
    }
    setIsSendingChat(false);
    try {
      await api.ai.stop(true);
    } catch {
      // ignore
    }
  }, []);

  const handleDeleteQuestion = useCallback((questionId: string) => {
    deletedQuestionIdsRef.current.add(String(questionId));
    setDetectedQuestions((prev) => {
      const removed = prev.find((q) => q.id === questionId);
      if (removed) deletedQuestionTextsRef.current.add(normalizeQuestion(removed.question));
      return prev.filter((q) => q.id !== questionId);
    });
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'delete_question',
        payload: { question_id: questionId },
      }));
    }
  }, []);

  const togglePause = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    if (!isPausedRef.current) {
      setIsPaused(true);
      isPausedRef.current = true;
      wsRef.current.send(JSON.stringify({ type: 'pause' }));
    } else {
      setIsPaused(false);
      isPausedRef.current = false;
      wsRef.current.send(JSON.stringify({ type: 'resume' }));
    }
  }, []);

  // Keyboard shortcut ⌘B per Segnalibro
  useEffect(() => {
    if (!lecture) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
        e.preventDefault();
        handleBookmark();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [lecture, handleBookmark]);

  // Scorciatoia globale (Tauri) -> Segnalibro
  const handleBookmarkRef = useRef(handleBookmark);
  handleBookmarkRef.current = handleBookmark;
  useEffect(() => {
    if (!lecture || !isTauri()) return;
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    (async () => {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        const un = await listen('global-bookmark', () => {
          if (isPausedRef.current) return;
          handleBookmarkRef.current(); // imposta anche lastBookmarkPinned (feedback visivo sul pulsante)
        });
        if (cancelled) un(); else unlisten = un;
      } catch (e) {
        console.warn('[Tauri] listen global-bookmark fallito:', e);
      }
    })();
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [lecture]);

  return {
    isConnected,
    connectionError,
    isStopping,
    isPaused,
    setIsPaused,
    audioStatus,
    togglePause,
    elapsed,
    chunks,
    bookmarks,
    lastBookmarkPinned,
    handleBookmark,
    handleStop,
    detectedQuestions,
    handleDeleteQuestion,
    chatMessages,
    chatInput,
    setChatInput,
    isSendingChat,
    handleChatSend,
    handleCancelChat,
    inspectorTab,
    setInspectorTab,
    formatElapsed,
  };
}

