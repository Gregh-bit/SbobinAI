// lib/api.ts — SbobinAI HTTP API client

export const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://127.0.0.1:8000';

// Estrae un messaggio leggibile dal campo `detail` di FastAPI (stringa o array di errori 422)
function formatErrorDetail(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((d) => (d && typeof d === 'object' && 'msg' in d ? String((d as { msg: unknown }).msg) : String(d)))
      .filter(Boolean);
    if (msgs.length > 0) return msgs.join('; ');
  }
  return `HTTP ${status}`;
}

async function throwResponseError(res: Response): Promise<never> {
  const err = await res.json().catch(() => ({ detail: res.statusText }));
  throw new Error(formatErrorDetail(err?.detail, res.status));
}

async function request<T>(
  path: string,
  options?: RequestInit,
  timeoutMs: number | null = 30000
): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  let signal = options?.signal;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let onCallerAbort: (() => void) | null = null;
  if (timeoutMs !== null && timeoutMs > 0 && typeof AbortController !== 'undefined') {
    const controller = new AbortController();
    if (signal?.aborted) controller.abort(signal.reason);
    else if (signal) {
      onCallerAbort = () => controller.abort(signal?.reason);
      signal.addEventListener('abort', onCallerAbort, { once: true });
    }
    timer = setTimeout(() => controller.abort(), timeoutMs);
    signal = controller.signal;
  }
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      headers,
      signal,
    });
    if (!res.ok) {
      await throwResponseError(res);
    }
    if (res.status === 204) return undefined as T;
    return res.json();
  } finally {
    if (timer) clearTimeout(timer);
    if (onCallerAbort) options?.signal?.removeEventListener('abort', onCallerAbort);
  }
}

async function streamNdjson(
  path: string,
  options: RequestInit,
  onEvent: (ev: import('../types').ChatStreamEvent) => void
): Promise<void> {
  const headers = new Headers(options.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    await throwResponseError(res);
  }

  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = '';

  const emitLine = (rawLine: string) => {
    const trimmed = rawLine.trim();
    if (!trimmed) return;
    const parsed = JSON.parse(trimmed) as import('../types').ChatStreamEvent;
    onEvent(parsed);
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });

      let newlineIdx = buffer.indexOf('\n');
      while (newlineIdx !== -1) {
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);
        emitLine(line);
        newlineIdx = buffer.indexOf('\n');
      }

      if (done) {
        if (buffer.trim()) {
          emitLine(buffer);
        }
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function checkBackendHealth(timeoutMs = 1500): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const origin = BASE_URL.replace(/\/api\/v1\/?$/, '');
    const res = await fetch(`${origin}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return false;
    const data = await res.json().catch(() => null);
    return data?.status === 'ok';
  } catch {
    return false;
  }
}

// ── Subjects ──────────────────────────────────────────────────────────────────
export const api = {
  subjects: {
    list: () => request<import('../types').Subject[]>('/subjects/'),
    create: (data: { name: string; description?: string; whisper_prompt_keywords?: string; color?: string }) =>
      request<import('../types').Subject>('/subjects/', { method: 'POST', body: JSON.stringify(data) }),
    update: (id: number, data: Partial<{ name: string; description: string | null; whisper_prompt_keywords: string | null; color: string }>) =>
      request<import('../types').Subject>(`/subjects/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    delete: (id: number) => request<void>(`/subjects/${id}`, { method: 'DELETE' }),
    getMaterials: (id: number) => request<import('../types').CourseMaterial[]>(`/subjects/${id}/materials`),
    getMaterialPage: (subjectId: number, filename: string, page: number) =>
      request<import('../types').MaterialPageOut>(
        `/subjects/${subjectId}/materials/${encodeURIComponent(filename)}/page/${page}`
      ),
    getMaterialFileUrl: (subjectId: number, filename: string): string =>
      `${BASE_URL}/subjects/${subjectId}/materials/${encodeURIComponent(filename)}/file`,
    deleteMaterialFile: (id: number, filename: string) =>
      request<void>(`/subjects/${id}/materials/${encodeURIComponent(filename)}`, { method: 'DELETE' }),
    chat: (subjectId: number, query: string, modelId?: string, options?: { signal?: AbortSignal }) =>
      request<{ answer: string; context_used: string }>(`/subjects/${subjectId}/chat`, {
        method: 'POST',
        body: JSON.stringify({ query, model_id: modelId }),
        signal: options?.signal,
      }),
    chatStream: async function* (
      subjectId: number,
      query: string,
      modelId?: string,
      options?: { signal?: AbortSignal }
    ): AsyncGenerator<{ token?: string; context_used?: string }, void, unknown> {
      const res = await fetch(`${BASE_URL}/subjects/${subjectId}/chat-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model_id: modelId }),
        signal: options?.signal,
      });
      if (!res.ok) {
        await throwResponseError(res);
      }
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      const HEADER = '__CONTEXT_USED__:';
      let buffer = '';
      let isHeaderParsed = false;
      while (true) {
        const { done, value } = await reader.read();
        // A fine stream svuota il decoder (eventuali byte UTF-8 residui)
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        if (!isHeaderParsed) {
          if (buffer.startsWith(HEADER)) {
            const sep = buffer.indexOf('\n\n');
            if (sep !== -1) {
              const ctx = buffer.substring(HEADER.length, sep);
              buffer = buffer.substring(sep + 2);
              isHeaderParsed = true;
              yield { context_used: ctx };
            } else if (done) {
              // Stream terminato senza separatore: solo header, nessun token
              isHeaderParsed = true;
              yield { context_used: buffer.substring(HEADER.length) };
              buffer = '';
            }
          } else if (!done && HEADER.startsWith(buffer)) {
            // Il buffer è ancora un prefisso dell'header: continua a leggere
          } else {
            isHeaderParsed = true;
          }
        }
        if (isHeaderParsed && buffer) {
          yield { token: buffer };
          buffer = '';
        }
        if (done) break;
      }
    },
    reindex: (subjectId: number) =>
      request<{ status: string; subject_id: number; indexed_chunks: number }>(
        `/subjects/${subjectId}/reindex`,
        {
          method: 'POST',
        },
        180000
      ),
    uploadMaterialFile: async (id: number, file: File): Promise<import('../types').CourseMaterial> => {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${BASE_URL}/subjects/${id}/materials/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        await throwResponseError(res);
      }
      return res.json();
    },
    updateMaterialTag: (subjectId: number, filename: string, tag: string) =>
      request<import('../types').CourseMaterial>(
        `/subjects/${subjectId}/materials/${encodeURIComponent(filename)}/tag`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tag }),
        }
      ),
  },
  lectures: {
    listBySubject: (subjectId: number) =>
      request<import('../types').Lecture[]>(`/lectures/by-subject/${subjectId}`),
    create: (data: { subject_id: number; title?: string; date?: string; raw_transcript?: string; processed_notes?: string; duration_seconds?: number; source_type?: string; in_quaderno?: boolean }) =>
      request<import('../types').Lecture>('/lectures/', { method: 'POST', body: JSON.stringify(data) }),
    update: (id: number, data: Partial<{ title: string; date: string; raw_transcript: string; processed_notes: string; duration_seconds: number; source_type: string; in_quaderno: boolean; detected_questions: string | null }>) =>
      request<import('../types').Lecture>(`/lectures/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    move: (lectureId: number, targetSubjectId: number) =>
      request<import('../types').Lecture>(`/lectures/${lectureId}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_subject_id: targetSubjectId }),
      }),
    delete: (id: number) => request<void>(`/lectures/${id}`, { method: 'DELETE' }),
    generateSummary: (lectureId: number, options?: { signal?: AbortSignal }) =>
      request<{ status: string; lecture_id: number; processed_notes: string }>(`/lectures/${lectureId}/generate-summary`, {
        method: 'POST',
        signal: options?.signal,
      }),
    generateSummaryStream: async (
      lectureId: number,
      onProgress: (ev: { stage: string; message?: string; progress?: number; processed_notes?: string; status?: string }) => void,
      options?: { signal?: AbortSignal; model_id?: string }
    ): Promise<{ status: string; lecture_id: number; processed_notes: string }> => {
      const headers = new Headers();
      headers.set('Content-Type', 'application/json');

      const res = await fetch(`${BASE_URL}/lectures/${lectureId}/generate-summary-stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model_id: options?.model_id }),
        signal: options?.signal,
      });

      if (!res.ok) {
        await throwResponseError(res);
      }

      const reader = res.body?.getReader();
      if (!reader) {
        throw new Error('Nessun flusso dati ricevuto dal server.');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let finalResult: { status: string; lecture_id: number; processed_notes: string } | null = null;

      const processLine = (rawLine: string) => {
        const trimmed = rawLine.trim();
        if (!trimmed || trimmed.startsWith(':')) return; // ignora commenti keep-alive (: keep-alive)
        let parsed: any;
        try {
          parsed = JSON.parse(trimmed);
        } catch {
          return; // ignora righe NDJSON parziali/malformate
        }
        onProgress(parsed);
        if (parsed.stage === 'done' && parsed.processed_notes) {
          finalResult = {
            status: parsed.status || 'success',
            lecture_id: parsed.lecture_id || lectureId,
            processed_notes: parsed.processed_notes,
          };
        } else if (parsed.stage === 'cancelled') {
          finalResult = {
            status: 'cancelled',
            lecture_id: lectureId,
            processed_notes: parsed.processed_notes || '',
          };
        } else if (parsed.stage === 'error') {
          throw new Error(parsed.message || 'Errore durante la generazione del capitolo');
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });

          let newlineIdx = buffer.indexOf('\n');
          while (newlineIdx !== -1) {
            const line = buffer.slice(0, newlineIdx);
            buffer = buffer.slice(newlineIdx + 1);
            processLine(line);
            newlineIdx = buffer.indexOf('\n');
          }

          if (done) {
            if (buffer.trim()) {
              processLine(buffer);
            }
            break;
          }
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {}
      }

      if (!finalResult) {
        const updatedLecture = await request<import('../types').Lecture>(`/lectures/${lectureId}`);
        return {
          status: 'success',
          lecture_id: lectureId,
          processed_notes: updatedLecture.processed_notes || '',
        };
      }

      return finalResult;
    },
    answerQuestion: (lectureId: number, questionId: string) =>
      request<{ status: string; question: import('../types').DetectedQuestion; answer: string }>(
        `/lectures/${lectureId}/questions/${encodeURIComponent(questionId)}/answer`,
        { method: 'POST' },
        180000
      ),
    getAudioUrl: (lectureId: number) => `${BASE_URL}/lectures/${lectureId}/audio`,
    getTimestamps: (lectureId: number) =>
      request<import('../types').TranscriptKaraokeResponse>(`/lectures/${lectureId}/timestamps`),
  },
  bookmarks: {
    listByLecture: (lectureId: number) =>
      request<import('../types').Bookmark[]>(`/bookmarks/by-lecture/${lectureId}`),
    create: (data: { lecture_id: number; timestamp_seconds: number; note?: string }) =>
      request<import('../types').Bookmark>('/bookmarks/', { method: 'POST', body: JSON.stringify(data) }),
    delete: (id: number) => request<void>(`/bookmarks/${id}`, { method: 'DELETE' }),
  },
  settings: {
    get: () => request<Record<string, unknown>>('/settings/'),
    setStorage: (storagePath: string) =>
      request<{ status: string; storage_path: string; dirs: unknown }>(
        '/settings/storage',
        {
          method: 'POST',
          body: JSON.stringify({ storage_path: storagePath }),
        },
        180000
      ),
    update: (data: Record<string, unknown>) =>
      request<Record<string, unknown>>('/settings/', {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    cleanOrphans: () =>
      request<{
        status: string;
        freed_bytes: number;
        freed_mb: number;
        deleted_audio_files: string[];
        deleted_upload_dirs: string[];
        deleted_db_files: string[];
        message: string;
      }>('/settings/clean-orphans', { method: 'POST' }, 180000),
    getStorageStats: () =>
      request<{
        audio_mb: number;
        uploads_mb: number;
        sqlite_mb: number;
        lancedb_mb: number;
        total_mb: number;
      }>('/settings/storage-stats'),
    compressArchive: () =>
      request<{
        status: string;
        converted_count: number;
        freed_mb: number;
        total_audio_mb: number;
        message: string;
      }>('/settings/compress-audio', { method: 'POST' }, 180000),
  },
  models: {
    status: () => request<import('../types').ModelsStatusResponse>('/models/status'),
    downloadStatus: (modelId?: string) =>
      request<any>(
        modelId
          ? `/models/download-status?model_id=${encodeURIComponent(modelId)}`
          : '/models/download-status'
      ),
    download: (modelId: string) =>
      request<{ status: string; model_id: string; model?: string; progress?: number }>('/models/download', {
        method: 'POST',
        body: JSON.stringify({ model_id: modelId }),
      }),
    delete: (modelName: string) =>
      request<{ status: string; model: string; freed_mb: number; message?: string }>(`/models/${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
      }),
    deleteAll: () =>
      request<{ status: string; freed_mb: number; deleted_count: number; deleted_items: string[]; message: string }>(
        '/models/all/delete-all',
        { method: 'DELETE' }
      ),
    cancelDownload: (modelId?: string) =>
      request<{ status: string; message: string; freed_mb: number }>('/models/cancel-download', {
        method: 'POST',
        body: JSON.stringify({ model_id: modelId || 'all' }),
      }),
  },
  ai: {
    stop: (forceUnload = true) =>
      request<{ status: string; freed_memory: boolean; message: string }>('/ai/stop', {
        method: 'POST',
        body: JSON.stringify({ force_unload: forceUnload }),
      }),
    status: () =>
      request<{ is_busy: boolean; task: string; model_loaded: boolean; model_id: string | null }>('/ai/status'),
  },
  study: {
    generateQuiz: (
      subjectId: number,
      data?: { lecture_ids?: number[]; model_id?: string; num_questions?: number },
      signal?: AbortSignal
    ) =>
      request<{
        status: string;
        subject_id: number;
        questions: import('../types').QuizQuestion[];
        warning?: string;
      }>(`/study/${subjectId}/generate-quiz`, {
        method: 'POST',
        body: JSON.stringify(data || {}),
        signal,
      }),
    getOralQuestion: (
      subjectId: number,
      data?: { lecture_ids?: number[]; model_id?: string },
      signal?: AbortSignal
    ) =>
      request<{ status: string; question: string; topic: string }>(
        `/study/${subjectId}/oral-question`,
        {
          method: 'POST',
          body: JSON.stringify(data || {}),
          signal,
        }
      ),
    evaluateOral: async (
      subjectId: number,
      audioBlob: Blob,
      question: string,
      modelId?: string,
      signal?: AbortSignal
    ): Promise<import('../types').OralEvaluationResult> => {
      const formData = new FormData();
      const mime = audioBlob.type || '';
      const ext = mime.includes('mp4') ? 'm4a' : mime.includes('ogg') ? 'ogg' : mime.includes('wav') ? 'wav' : 'webm';
      formData.append('audio', audioBlob, `oral_answer.${ext}`);
      formData.append('question', question);
      if (modelId) formData.append('model_id', modelId);

      const res = await fetch(`${BASE_URL}/study/${subjectId}/evaluate-oral`, {
        method: 'POST',
        body: formData,
        signal,
      });
      if (!res.ok) {
        await throwResponseError(res);
      }
      return res.json();
    },
    abort: (subjectId?: number) =>
      request<{ status: string; message: string }>(
        subjectId ? `/study/${subjectId}/abort` : '/study/abort',
        { method: 'POST' }
      ),
    listQuizSets: (subjectId: number) =>
      request<import('../types').QuizSetOut[]>(`/study/${subjectId}/quiz-sets`),
    getQuizSet: (quizId: number) =>
      request<import('../types').QuizSetOut>(`/study/quiz-sets/${quizId}`),
    deleteQuizSet: (quizId: number) =>
      request<{ ok: boolean; quiz_id: number }>(`/study/quiz-sets/${quizId}`, {
        method: 'DELETE',
      }),
    submitQuizResults: (quizId: number, data: { score_percent: number; results: import('../types').QuizResultItem[] }) =>
      request<import('../types').QuizSetOut>(`/study/quiz-sets/${quizId}/results`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    expandQuizSet: (quizId: number, numQuestions: number = 5) =>
      request<import('../types').QuizSetOut>(`/study/quiz-sets/${quizId}/expand?num_questions=${numQuestions}`, {
        method: 'POST',
      }),
    getMasterySummary: (subjectId: number) =>
      request<import('../types').StudentMasterySummary>(`/study/${subjectId}/mastery`),
    markConcept: (subjectId: number, data: { topic: string; status: 'consolidato' | 'da_consolidare' }) =>
      request<import('../types').StudentConcept>(`/study/${subjectId}/concepts/mark`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    deleteConcept: (subjectId: number, conceptId: number) =>
      request<{ ok: boolean; concept_id: number }>(`/study/${subjectId}/concepts/${conceptId}`, {
        method: 'DELETE',
      }),
    clearAllConcepts: (subjectId: number) =>
      request<{ ok: boolean; deleted_count: number }>(`/study/${subjectId}/concepts`, {
        method: 'DELETE',
      }),
  },
  chats: {
    list: (subjectId: number) =>
      request<import('../types').ChatSession[]>(`/chats/subject/${subjectId}`),
    create: (subjectId: number, data?: { title?: string; scope?: import('../types').ChatScope }) =>
      request<import('../types').ChatSessionDetail>(`/chats/subject/${subjectId}`, {
        method: 'POST',
        body: JSON.stringify(data || {}),
      }),
    get: (chatId: number) =>
      request<import('../types').ChatSessionDetail>(`/chats/${chatId}`),
    update: (chatId: number, data: { title?: string; scope?: import('../types').ChatScope }) =>
      request<import('../types').ChatSessionDetail>(`/chats/${chatId}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    delete: (chatId: number) =>
      request<{ ok: boolean }>(`/chats/${chatId}`, { method: 'DELETE' }),
    deleteMessage: (chatId: number, messageId: number) =>
      request<import('../types').ChatSessionDetail>(`/chats/${chatId}/messages/${messageId}`, {
        method: 'DELETE',
      }),
    compact: (chatId: number) =>
      request<import('../types').ChatSessionDetail>(
        `/chats/${chatId}/compact`,
        {
          method: 'POST',
        },
        180000
      ),
    archive: (chatId: number, data: { title?: string; mode?: 'quaderno_and_clear' | 'quaderno_keep' }) =>
      request<{ lecture_id: number; title: string; session: import('../types').ChatSessionDetail }>(
        `/chats/${chatId}/archive`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        },
        180000
      ),
    saveNote: (chatId: number, data: { message_id: number; title?: string }) =>
      request<{ lecture_id: number; title: string }>(
        `/chats/${chatId}/save-note`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        },
        180000
      ),
    importLegacy: (
      subjectId: number,
      data: { title?: string; messages: Array<{ role: string; content: string; thinking?: string; context_used?: string }> }
    ) =>
      request<import('../types').ChatSessionDetail>(`/chats/subject/${subjectId}/import-legacy`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    streamMessage: (
      chatId: number,
      body: {
        message: string;
        mode?: import('../types').ChatStudyMode;
        scope?: import('../types').ChatScope;
        enable_thinking?: boolean;
      },
      onEvent: (ev: import('../types').ChatStreamEvent) => void,
      signal?: AbortSignal
    ): Promise<void> =>
      streamNdjson(
        `/chats/${chatId}/stream`,
        {
          method: 'POST',
          body: JSON.stringify(body),
          signal,
        },
        onEvent
      ),
    regenerateStream: (
      chatId: number,
      messageId: number,
      onEvent: (ev: import('../types').ChatStreamEvent) => void,
      signal?: AbortSignal,
      enableThinking?: boolean
    ): Promise<void> =>
      streamNdjson(
        `/chats/${chatId}/messages/${messageId}/regenerate-stream`,
        {
          method: 'POST',
          ...(enableThinking !== undefined
            ? { body: JSON.stringify({ enable_thinking: enableThinking }) }
            : {}),
          signal,
        },
        onEvent
      ),
    editAndStreamMessage: (
      chatId: number,
      messageId: number,
      payload: import('../types').ChatMessageSend,
      onEvent: (ev: import('../types').ChatStreamEvent) => void,
      signal?: AbortSignal
    ): Promise<void> =>
      streamNdjson(
        `/chats/${chatId}/messages/${messageId}/edit-stream`,
        {
          method: 'POST',
          body: JSON.stringify(payload),
          signal,
        },
        onEvent
      ),
  },
  ws: {
    liveLecture: (subjectId: number) => {
      const targetUrl = (WS_URL || 'ws://127.0.0.1:8000').replace('//localhost', '//127.0.0.1');
      return new WebSocket(`${targetUrl}/ws/live-lecture/${subjectId}`);
    },
  },
};
