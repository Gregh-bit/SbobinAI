import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from './api';
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

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedRef = useRef(0);
  const isPausedRef = useRef(isPaused);

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
          const cleanQ = qText.toLowerCase().replace(/[\?\.\,\s]/g, '');
          const existingIdx = prev.findIndex((item) => {
            const itemClean = item.question.toLowerCase().replace(/[\?\.\,\s]/g, '');
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
        if (timerRef.current) clearInterval(timerRef.current);
        setAudioStatus('idle');
        onStop();
        break;
    }
  }, [onStop]);

  const handleWSMessageRef = useRef(handleWSMessage);
  useEffect(() => {
    handleWSMessageRef.current = handleWSMessage;
  }, [handleWSMessage]);

  // ── WebSocket Lifecycle ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!lecture) return;

    const ws = api.ws.liveLecture(subject.id);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
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

    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);

    return () => {
      ws.close();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [subject.id, lecture?.id]);

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
    if (timerRef.current) clearInterval(timerRef.current);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'stop', payload: {} }));
      } catch (err) {
        console.warn('Errore invio stop WebSocket:', err);
      }
    }
    if (lecture) {
      api.lectures.update(lecture.id, {
        duration_seconds: elapsedRef.current,
        detected_questions: detectedQuestions.length > 0 ? JSON.stringify(detectedQuestions) : null,
      }).catch(() => {});
    }
    // Ritorna subito al Quaderno senza ritardi o schermate bloccate
    onStop();
  }, [lecture, onStop, detectedQuestions]);

  const handleChatSend = useCallback((textToSend?: string) => {
    const q = (textToSend || chatInput).trim();
    if (!q || isSendingChat) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: q,
      timestamp: Date.now(),
    };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput('');
    setIsSendingChat(true);

    wsRef.current?.send(JSON.stringify({
      type: 'chat',
      payload: { query: q, context_minutes: 5 }
    }));
  }, [chatInput, isSendingChat]);

  const handleCancelChat = useCallback(async () => {
    setIsSendingChat(false);
    try {
      await api.ai.stop(true);
    } catch {
      // ignore
    }
  }, []);

  const handleDeleteQuestion = useCallback((questionId: string) => {
    setDetectedQuestions((prev) => prev.filter((q) => q.id !== questionId));
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

  return {
    isConnected,
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

