import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from './api';
import { isTauri, acquireWakeLock, releaseWakeLock } from './tauri';
import { useSettings } from './SettingsContext';
import type { 
  ChatMessage, CitationRef, DetectedQuestion, Lecture, Subject, 
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
const STOP_ACK_TIMEOUT_MS = 12000;

interface UseLiveSessionProps {
  subject: Subject;
  lecture: Lecture | null;
  onStop: () => void;
}

export function useLiveSession({ subject, lecture, onStop }: UseLiveSessionProps) {
  const { settings } = useSettings();
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioStatus, setAudioStatus] = useState<'idle' | 'listening' | 'transcribing'>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [chunks, setChunks] = useState<TranscriptChunk[]>([]);
  const [bookmarks, setBookmarks] = useState<{ id: number; timestamp: number; note: string }[]>([]);
  const [lastBookmarkPinned, setLastBookmarkPinned] = useState<number | null>(null);

  // Inspector State
  const [inspectorTab, setInspectorTab] = useState<'domande' | 'chat'>('domande');
  const [detectedQuestions, setDetectedQuestions] = useState<DetectedQuestion[]>([]);
  const [answeringQuestionIds, setAnsweringQuestionIds] = useState<Set<string>>(new Set());
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isSendingChat, setIsSendingChat] = useState(false);

  // Stato connessione / errori della registrazione
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [isStopping, setIsStopping] = useState(false);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedRef = useRef(0);
  const isPausedRef = useRef(isPaused);
  const isConnectedRef = useRef(isConnected);
  const isReconnectingRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wasAutoPausedBySleepRef = useRef(false);
  const lectureRef = useRef(lecture);
  // Domande eliminate dall'utente: ignorate se arrivano di nuovo (es. con la risposta flash)
  const deletedQuestionIdsRef = useRef<Set<string>>(new Set());
  const deletedQuestionTextsRef = useRef<Set<string>>(new Set());
  // Chat live: query attiva e numero di risposte di richieste annullate ancora da scartare
  const activeChatQueryRef = useRef<string | null>(null);
  const cancelledChatResponsesRef = useRef(0);
  const chatChunkBufferRef = useRef<string>('');
  const chatChunkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chatMessagesRef = useRef<ChatMessage[]>(chatMessages);
  chatMessagesRef.current = chatMessages;
  // Attesa del messaggio 'stopped' dopo l'invio di 'stop'
  const stopResolverRef = useRef<((acknowledged: boolean) => void) | null>(null);
  const isStoppingRef = useRef(false);
  const settingsRef = useRef(settings);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    isConnectedRef.current = isConnected;
  }, [isConnected]);

  useEffect(() => {
    lectureRef.current = lecture;
  }, [lecture]);

  // Reset all state when a new lecture starts or is closed
  useEffect(() => {
    setElapsed(0);
    elapsedRef.current = 0;
    setChunks([]);
    setBookmarks([]);
    setChatMessages([]);
    setDetectedQuestions([]);
    setAnsweringQuestionIds(new Set());
    setIsPaused(false);
    isPausedRef.current = false;
    setAudioStatus('idle');
    setConnectionError(null);
    setIsStopping(false);
    setIsSendingChat(false);
    setIsReconnecting(false);
    isStoppingRef.current = false;
    isReconnectingRef.current = false;
    reconnectAttemptsRef.current = 0;
    wasAutoPausedBySleepRef.current = false;
    deletedQuestionIdsRef.current = new Set();
    deletedQuestionTextsRef.current = new Set();
    activeChatQueryRef.current = null;
    cancelledChatResponsesRef.current = 0;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    return () => {
      void releaseWakeLock();
    };
  }, [lecture?.id]);

  // ── Handle incoming WS messages ─────────────────────────────────────────────
  const handleWSMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case 'started': {
        void acquireWakeLock(`SbobinAI - Registrazione lezione "${lectureRef.current?.title || subject.name}" in corso`);
        setElapsed(0);
        elapsedRef.current = 0;
        let lastTickMs = Date.now();
        let activeMs = 0;
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          const now = Date.now();
          if (!isPausedRef.current) {
            activeMs += Math.max(0, now - lastTickMs);
            const currentSec = Math.floor(activeMs / 1000);
            if (currentSec !== elapsedRef.current) {
              elapsedRef.current = currentSec;
              setElapsed(currentSec);
            }
          }
          lastTickMs = now;
        }, 500);
        setConnectionError(null);
        setIsConnected(true);
        isConnectedRef.current = true;
        setIsReconnecting(false);
        isReconnectingRef.current = false;
        reconnectAttemptsRef.current = 0;
        break;
      }

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

      case 'chat_meta': {
        if (cancelledChatResponsesRef.current > 0 || activeChatQueryRef.current === null) break;
        const metaQuery = msg.payload.query as string | undefined;
        if (metaQuery !== undefined && metaQuery !== activeChatQueryRef.current) break;
        const citations = Array.isArray(msg.payload.citations)
          ? (msg.payload.citations as CitationRef[])
          : undefined;
        const contextUsed = msg.payload.context_used as string | undefined;
        setChatMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
            const last = prev[prev.length - 1];
            return [
              ...prev.slice(0, -1),
              {
                ...last,
                citations: citations ?? last.citations,
                contextUsed: contextUsed ?? last.contextUsed,
              },
            ];
          }
          return [
            ...prev,
            {
              id: generateId(),
              role: 'assistant',
              content: '',
              timestamp: Date.now(),
              citations,
              contextUsed,
            },
          ];
        });
        break;
      }

      case 'chat_chunk': {
        const chunk = (msg.payload.chunk as string) || '';
        if (!chunk) break;
        // Scarta i frammenti di richieste annullate o non corrispondenti alla richiesta attiva
        if (cancelledChatResponsesRef.current > 0 || activeChatQueryRef.current === null) break;
        const chunkQuery = msg.payload.query as string | undefined;
        if (chunkQuery !== undefined && chunkQuery !== activeChatQueryRef.current) break;

        chatChunkBufferRef.current += chunk;

        if (!chatChunkTimerRef.current) {
          chatChunkTimerRef.current = setTimeout(() => {
            chatChunkTimerRef.current = null;
            const buffered = chatChunkBufferRef.current;
            if (!buffered) return;
            chatChunkBufferRef.current = '';

            setChatMessages((prev) => {
              if (prev.length === 0 || prev[prev.length - 1].role !== 'assistant') {
                const newAssistant: ChatMessage = {
                  id: generateId(),
                  role: 'assistant',
                  content: buffered,
                  timestamp: Date.now(),
                };
                return [...prev, newAssistant];
              }
              const last = prev[prev.length - 1];
              const updatedLast: ChatMessage = {
                ...last,
                content: last.content + buffered,
              };
              return [...prev.slice(0, -1), updatedLast];
            });
          }, 45);
        }
        break;
      }

      case 'chat_response': {
        if (chatChunkTimerRef.current) {
          clearTimeout(chatChunkTimerRef.current);
          chatChunkTimerRef.current = null;
        }
        chatChunkBufferRef.current = '';

        // Risposta finale di una richiesta annullata (il backend ne invia sempre una per richiesta)
        if (cancelledChatResponsesRef.current > 0) {
          cancelledChatResponsesRef.current -= 1;
          break;
        }
        if (activeChatQueryRef.current === null) break;
        activeChatQueryRef.current = null;
        const fullAnswer = (msg.payload.answer as string) || (msg.payload.text as string) || '';
        const contextUsed = msg.payload.context_used as string | undefined;
        const citations = Array.isArray(msg.payload.citations)
          ? (msg.payload.citations as CitationRef[])
          : undefined;
        setChatMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant') {
            const last = prev[prev.length - 1];
            const updatedLast: ChatMessage = {
              ...last,
              content: fullAnswer || last.content,
              contextUsed: contextUsed || last.contextUsed,
              citations: citations ?? last.citations,
            };
            return [...prev.slice(0, -1), updatedLast];
          }
          const answer: ChatMessage = {
            id: generateId(),
            role: 'assistant',
            content: fullAnswer,
            timestamp: Date.now(),
            contextUsed: contextUsed,
            citations,
          };
          return [...prev, answer];
        });
        setIsSendingChat(false);
        break;
      }

      case 'reconnected': {
        void acquireWakeLock(`SbobinAI - Registrazione lezione "${lectureRef.current?.title || subject.name}" in corso`);
        const paused = Boolean(msg.payload?.is_paused);
        setIsPaused(paused);
        isPausedRef.current = paused;
        if (paused) {
          setAudioStatus('idle');
        }

        const restoredElapsed = typeof msg.payload?.elapsed === 'number' ? msg.payload.elapsed : 0;
        setElapsed(restoredElapsed);
        elapsedRef.current = restoredElapsed;

        let lastTickMs = Date.now();
        let activeMs = restoredElapsed * 1000;
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
          const now = Date.now();
          if (!isPausedRef.current) {
            activeMs += Math.max(0, now - lastTickMs);
            const currentSec = Math.max(restoredElapsed, Math.floor(activeMs / 1000));
            if (currentSec !== elapsedRef.current) {
              elapsedRef.current = currentSec;
              setElapsed(currentSec);
            }
          }
          lastTickMs = now;
        }, 500);

        if (Array.isArray(msg.payload?.chunks)) {
          const restoredChunks: TranscriptChunk[] = (msg.payload.chunks as unknown[]).map((c: any) => {
            if (typeof c === 'string') {
              return {
                text: c,
                elapsed: restoredElapsed,
                timestamp: Date.now(),
                speaker: 'Docente',
              };
            }
            return {
              text: c.text || '',
              elapsed: typeof c.elapsed === 'number' ? c.elapsed : restoredElapsed,
              timestamp: typeof c.timestamp === 'number' ? c.timestamp : Date.now(),
              speaker: c.speaker || 'Docente',
            };
          });
          setChunks(restoredChunks);
        }

        if (Array.isArray(msg.payload?.detected_questions)) {
          const restoredQuestions: DetectedQuestion[] = (msg.payload.detected_questions as unknown[]).map((q: any) => {
            const qText = typeof q === 'string' ? q : (q.question || q.text || '');
            const rawAns = q.answer || q.suggested_answer || q.suggestedAnswer || undefined;
            return {
              id: String(q.id || generateId()),
              question: qText,
              confidence: typeof q.confidence === 'number' ? q.confidence : 0.9,
              timestamp: typeof q.timestamp === 'number' ? q.timestamp : Date.now(),
              answer: rawAns,
              suggestedAnswer: rawAns,
              sourceLecture: q.source_lecture || q.sourceLecture || 'Quaderno del corso',
            };
          });
          setDetectedQuestions(restoredQuestions);
        }

        setConnectionError(null);
        setIsConnected(true);
        isConnectedRef.current = true;
        setIsReconnecting(false);
        isReconnectingRef.current = false;
        reconnectAttemptsRef.current = 0;
        break;
      }

      case 'question_answering': {
        const qId = (msg.payload?.question_id as string) || (msg.payload?.id as string);
        if (qId) {
          setAnsweringQuestionIds((prev) => new Set(prev).add(String(qId)));
        }
        break;
      }

      case 'question_answer_chunk': {
        const qId = (msg.payload?.question_id as string) || (msg.payload?.id as string);
        const token = (msg.payload?.chunk as string) || '';
        if (!qId || !token) break;
        setDetectedQuestions((prev) =>
          prev.map((item) => {
            if (item.id !== String(qId)) return item;
            const nextText = (item.answer || item.suggestedAnswer || '') + token;
            return { ...item, answer: nextText, suggestedAnswer: nextText };
          })
        );
        break;
      }

      case 'question_detected': {
        const qText = (msg.payload?.question as string) || '';
        if (!qText.trim()) break;
        const incomingId = msg.payload?.id ? String(msg.payload.id) : undefined;
        if (
          (incomingId && deletedQuestionIdsRef.current.has(incomingId)) ||
          deletedQuestionTextsRef.current.has(normalizeQuestion(qText))
        ) {
          break;
        }

        const answerText = (msg.payload?.answer as string) || (msg.payload?.suggested_answer as string) || undefined;
        const qId = incomingId || generateId();

        // Rimuovi da answeringQuestionIds
        setAnsweringQuestionIds((prev) => {
          if (!prev.has(qId) && (!incomingId || !prev.has(incomingId))) return prev;
          const next = new Set(prev);
          next.delete(qId);
          if (incomingId) next.delete(incomingId);
          return next;
        });

        const q: DetectedQuestion = {
          id: qId,
          question: qText,
          confidence: (msg.payload?.confidence as number) || 0.9,
          timestamp: Date.now(),
          answer: answerText,
          suggestedAnswer: answerText,
          sourceLecture: 'Quaderno del corso',
        };

        setDetectedQuestions((prev) => {
          const cleanQ = normalizeQuestion(qText);
          const existingIdx = prev.findIndex((item) => {
            if (incomingId && item.id === incomingId) return true;
            const itemClean = normalizeQuestion(item.question);
            return itemClean === cleanQ || (cleanQ.length > 15 && itemClean.includes(cleanQ)) || (itemClean.length > 15 && cleanQ.includes(itemClean));
          });
          if (existingIdx >= 0) {
            const updated = [...prev];
            const existing = updated[existingIdx];
            if (existing.id !== qId) {
              setAnsweringQuestionIds((s) => {
                if (!s.has(existing.id)) return s;
                const next = new Set(s);
                next.delete(existing.id);
                return next;
              });
            }
            updated[existingIdx] = {
              ...existing,
              ...q,
              id: existing.id,
              answer: answerText || existing.answer,
              suggestedAnswer: answerText || existing.suggestedAnswer,
            };
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
        setConnectionError(null);
        break;

      case 'stopped':
        void releaseWakeLock();
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

        // Se il backend risponde con errore tipo non supportato per reconnect, fallback su start
        if (message.includes('reconnect') && lectureRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          console.info("[LiveSession] 'reconnect' non riconosciuto dal backend, fallback immediato su 'start'");
          wsRef.current.send(JSON.stringify({
            type: 'start',
            payload: {
              title: lectureRef.current.title,
              lecture_id: lectureRef.current.id,
              model: settingsRef.current.whisperModel,
              vad_sensitivity: settingsRef.current.vadSensitivity,
            }
          }));
          break;
        }

        const failedQId = (msg.payload?.question_id as string) || (msg.payload?.id as string);
        if (failedQId) {
          setAnsweringQuestionIds((prev) => {
            const next = new Set(prev);
            next.delete(String(failedQId));
            return next;
          });
          break;
        }

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

  // ── WebSocket Lifecycle & Reconnection ──────────────────────────────────────
  // Il corso della WebSocket è quello della lezione in registrazione (fissato all'avvio)
  const sessionSubjectId = lecture?.subject_id ?? subject.id;

  const connectWebSocket = useCallback(() => {
    const currentLecture = lectureRef.current;
    if (!currentLecture || isStoppingRef.current) return;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }

    const ws = api.ws.liveLecture(sessionSubjectId);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setIsConnected(true);
      isConnectedRef.current = true;
      setIsReconnecting(false);
      isReconnectingRef.current = false;
      setConnectionError(null);

      const isReconnect = reconnectAttemptsRef.current > 0;
      reconnectAttemptsRef.current = 0;

      if (isReconnect) {
        console.log(`[LiveSession] Riconnesso con successo al motore di registrazione per lezione ${currentLecture.id}`);
        try {
          ws.send(JSON.stringify({
            type: 'reconnect',
            payload: {
              lecture_id: currentLecture.id,
              title: currentLecture.title,
              model: settingsRef.current.whisperModel,
              vad_sensitivity: settingsRef.current.vadSensitivity,
            }
          }));
        } catch (err) {
          console.warn('[LiveSession] Errore invio reconnect:', err);
        }
      } else {
        try {
          ws.send(JSON.stringify({
            type: 'start',
            payload: {
              title: currentLecture.title,
              lecture_id: currentLecture.id,
              model: settingsRef.current.whisperModel,
              vad_sensitivity: settingsRef.current.vadSensitivity,
            }
          }));
        } catch (err) {
          console.warn('[LiveSession] Errore invio start:', err);
        }
      }
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
      if (wsRef.current === ws) wsRef.current = null;
      setIsConnected(false);
      isConnectedRef.current = false;
      stopTimer();

      // Sblocca un'eventuale attesa di 'stopped', richieste chat e domande in corso
      if (stopResolverRef.current) stopResolverRef.current(false);
      cancelledChatResponsesRef.current = 0;
      if (activeChatQueryRef.current !== null) {
        activeChatQueryRef.current = null;
        setIsSendingChat(false);
        setChatMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && !prev[prev.length - 1].content.trim()) {
            return prev.slice(0, -1);
          }
          return prev;
        });
      }
      setAnsweringQuestionIds((prev) => (prev.size > 0 ? new Set() : prev));

      if (!isStoppingRef.current && lectureRef.current) {
        // Mostra stato di riconnessione automatica a intervalli crescenti (1s, 2s, 3s...)
        setIsReconnecting(true);
        isReconnectingRef.current = true;
        reconnectAttemptsRef.current += 1;
        const delay = Math.min(reconnectAttemptsRef.current * 1000, 5000);
        console.warn(`[LiveSession] Connessione interrotta. Riconnessione automatica #${reconnectAttemptsRef.current} tra ${delay}ms...`);
        reconnectTimerRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      }
    };

    ws.onerror = () => {
      if (wsRef.current === ws) {
        setIsConnected(false);
        isConnectedRef.current = false;
      }
    };
  }, [sessionSubjectId]);

  // ── Aggiornamento VAD dinamico in tempo reale (senza chiudere WebSocket o fermare timer) ──
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({
          type: 'update_vad',
          payload: { vad_sensitivity: settings.vadSensitivity }
        }));
      } catch (err) {
        console.warn('[LiveSession] Errore invio update_vad in diretta:', err);
      }
    }
  }, [settings.vadSensitivity]);

  useEffect(() => {
    if (!lecture) return;

    connectWebSocket();

    return () => {
      void releaseWakeLock();
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (chatChunkTimerRef.current) {
        clearTimeout(chatChunkTimerRef.current);
        chatChunkTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
        wsRef.current.onmessage = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      stopTimer();
    };
  }, [lecture?.id, connectWebSocket]);

  // ── Fluidità di Pausa / Ripresa ───────────────────────────────────────────
  const togglePause = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    if (!isPausedRef.current) {
      // Feedback istantaneo a zero latenza
      setIsPaused(true);
      isPausedRef.current = true;
      setAudioStatus('idle');
      try {
        wsRef.current.send(JSON.stringify({ type: 'pause' }));
      } catch (err) {
        console.warn('[LiveSession] Errore invio pause WebSocket:', err);
      }
    } else {
      // Feedback istantaneo per ripresa
      setIsPaused(false);
      isPausedRef.current = false;
      wasAutoPausedBySleepRef.current = false;
      try {
        wsRef.current.send(JSON.stringify({ type: 'resume' }));
      } catch (err) {
        console.warn('[LiveSession] Errore invio resume WebSocket:', err);
      }
    }
  }, []);

  // ── Riconoscimento Ritorno da Background / Sleep (senza mai auto-mettere in pausa) ────────
  useEffect(() => {
    if (!lecture) return;

    const handleVisibilityChange = () => {
      // Quando la finestra o scheda va in background, NON mettiamo in pausa:
      // la registrazione deve continuare ininterrottamente anche mentre lo studente consulta slide/note.
      if (document.visibilityState === 'visible') {
        if (!isStoppingRef.current) {
          const isSocketOpen = wsRef.current && wsRef.current.readyState === WebSocket.OPEN;
          if (!isSocketOpen) {
            console.log('[LiveSession] Schermo riaperto / tornato visibile: avvio immediato riconnessione automatica');
            if (reconnectTimerRef.current) {
              clearTimeout(reconnectTimerRef.current);
              reconnectTimerRef.current = null;
            }
            connectWebSocket();
          }
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [lecture, connectWebSocket]);

  // ── Actions ─────────────────────────────────────────────────────────────────
  const lastBookmarkTimeRef = useRef(0);
  const handleBookmark = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const now = Date.now();
    if (now - lastBookmarkTimeRef.current < 400) return;
    lastBookmarkTimeRef.current = now;
    setLastBookmarkPinned(now);
    wsRef.current.send(JSON.stringify({
      type: 'bookmark',
      payload: { note: `Segnalibro @ ${formatElapsed(elapsedRef.current)}` }
    }));
  }, []);

  const handleStop = useCallback(async () => {
    if (isStoppingRef.current) return;
    isStoppingRef.current = true;
    setIsStopping(true);
    setIsReconnecting(false);
    isReconnectingRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    void releaseWakeLock();
    stopTimer();

    try {
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

      // Fallback: se il backend non ha confermato, salva almeno durata, testo parziale e domande lato client
      if (!acknowledged && lecture) {
        try {
          const rawText = chunks.map((c) => c.text).join(' ').trim();
          await api.lectures.update(lecture.id, {
            duration_seconds: elapsedRef.current,
            raw_transcript: rawText || undefined,
            detected_questions: detectedQuestions.length > 0 ? JSON.stringify(detectedQuestions) : null,
          });
        } catch {
          // ignore
        }
      }
      // Stacca gli handler prima di chiudere così ws.onclose non innesca banner "Riconnessione..." durante la chiusura
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.onerror = null;
      }
      onStop();
    } finally {
      // Mantiene isStoppingRef=true finché lecture?.id non si azzera nel reset effect
      setIsStopping(false);
    }
  }, [lecture, onStop, detectedQuestions, chunks]);

  const handleChatSend = useCallback((textToSend?: string, options?: { enableThinking?: boolean }) => {
    const q = (textToSend || chatInput).trim();
    if (!q || isSendingChat) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setConnectionError("Connessione non disponibile: impossibile inviare la domanda all'assistente.");
      return;
    }

    const recentHistory = chatMessagesRef.current
      .filter((m) => m.content.trim().length > 0)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: q,
      timestamp: Date.now(),
    };
    // Inserisce subito il placeholder assistente per mostrare l'animazione ChatGPT 2026 a 0ms
    const pendingAssistant: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now() + 1,
    };
    setChatMessages((prev) => [...prev, userMsg, pendingAssistant]);
    setChatInput('');
    setIsSendingChat(true);
    activeChatQueryRef.current = q;

    try {
      ws.send(JSON.stringify({
        type: 'chat',
        payload: {
          query: q,
          context_minutes: 15,
          enable_thinking: Boolean(options?.enableThinking),
          history: recentHistory,
        }
      }));
    } catch (err) {
      console.warn('Errore invio chat WebSocket:', err);
      activeChatQueryRef.current = null;
      setIsSendingChat(false);
    }
  }, [chatInput, isSendingChat]);

  const handleCancelChat = useCallback(async () => {
    if (chatChunkTimerRef.current) {
      clearTimeout(chatChunkTimerRef.current);
      chatChunkTimerRef.current = null;
    }
    chatChunkBufferRef.current = '';

    // La risposta della richiesta annullata arriverà comunque: va scartata
    if (activeChatQueryRef.current !== null) {
      cancelledChatResponsesRef.current += 1;
      activeChatQueryRef.current = null;
    }
    setChatMessages((prev) => {
      if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && !prev[prev.length - 1].content.trim()) {
        return prev.slice(0, -1);
      }
      return prev;
    });
    setIsSendingChat(false);
    try {
      await api.ai.stop(true);
    } catch {
      // ignore
    }
  }, []);

  const handleAnswerQuestion = useCallback((questionId: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setAnsweringQuestionIds((prev) => new Set(prev).add(questionId));
    wsRef.current.send(JSON.stringify({
      type: 'answer_question',
      payload: { question_id: questionId }
    }));
  }, []);

  const handleDeleteQuestion = useCallback((questionId: string) => {
    deletedQuestionIdsRef.current.add(String(questionId));
    setAnsweringQuestionIds((prev) => {
      if (!prev.has(questionId)) return prev;
      const next = new Set(prev);
      next.delete(questionId);
      return next;
    });
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


  // Keyboard shortcut ⌘B e ⌘⇧B per Segnalibro
  useEffect(() => {
    if (!lecture) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isInput = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);

      if ((e.metaKey || e.ctrlKey) && (e.key.toLowerCase() === 'b' || e.code === 'KeyB')) {
        // Se l'utente sta scrivendo in un campo di testo ed è solo Cmd+B (senza Shift), non intercettare
        if (isInput && !e.shiftKey) return;
        e.preventDefault();
        e.stopPropagation();
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
    isReconnecting,
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
    answeringQuestionIds,
    handleAnswerQuestion,
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

