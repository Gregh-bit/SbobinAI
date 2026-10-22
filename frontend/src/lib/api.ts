// lib/api.ts — SbobinAI HTTP API client

const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/v1';
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

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = new Headers(options?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    await throwResponseError(res);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
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
      request<{ status: string; subject_id: number; indexed_chunks: number }>(`/subjects/${subjectId}/reindex`, {
        method: 'POST',
      }),
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
  },
  lectures: {
    listBySubject: (subjectId: number) =>
      request<import('../types').Lecture[]>(`/lectures/by-subject/${subjectId}`),
    create: (data: { subject_id: number; title?: string; date?: string; raw_transcript?: string; processed_notes?: string; duration_seconds?: number; source_type?: string; in_quaderno?: boolean }) =>
      request<import('../types').Lecture>('/lectures/', { method: 'POST', body: JSON.stringify(data) }),
    update: (id: number, data: Partial<{ title: string; date: string; raw_transcript: string; processed_notes: string; duration_seconds: number; source_type: string; in_quaderno: boolean; detected_questions: string | null }>) =>
      request<import('../types').Lecture>(`/lectures/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    delete: (id: number) => request<void>(`/lectures/${id}`, { method: 'DELETE' }),
    generateSummary: (lectureId: number, options?: { signal?: AbortSignal }) =>
      request<{ status: string; lecture_id: number; processed_notes: string }>(`/lectures/${lectureId}/generate-summary`, {
        method: 'POST',
        signal: options?.signal,
      }),
    answerQuestion: (lectureId: number, questionId: string) =>
      request<{ status: string; question: import('../types').DetectedQuestion; answer: string }>(
        `/lectures/${lectureId}/questions/${encodeURIComponent(questionId)}/answer`,
        { method: 'POST' }
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
      request<{ status: string; storage_path: string; dirs: unknown }>('/settings/storage', {
        method: 'POST',
        body: JSON.stringify({ storage_path: storagePath }),
      }),
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
      }>('/settings/clean-orphans', { method: 'POST' }),
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
      }>('/settings/compress-audio', { method: 'POST' }),
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
        body: modelId ? JSON.stringify({ model_id: modelId }) : undefined,
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
  },
  ws: {
    liveLecture: (subjectId: number) => new WebSocket(`${WS_URL}/ws/live-lecture/${subjectId}`),
  },
};
