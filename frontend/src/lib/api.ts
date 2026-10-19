// lib/api.ts — SbobinAI HTTP API client

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Subjects ──────────────────────────────────────────────────────────────────
export const api = {
  subjects: {
    list: () => request<import('../types').Subject[]>('/subjects/'),
    create: (data: { name: string; description?: string; whisper_prompt_keywords?: string; color?: string }) =>
      request<import('../types').Subject>('/subjects/', { method: 'POST', body: JSON.stringify(data) }),
    update: (id: number, data: Partial<{ name: string; description: string; whisper_prompt_keywords: string; color: string }>) =>
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
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const reader = res.body?.getReader();
      if (!reader) return;
      const decoder = new TextDecoder();
      let buffer = '';
      let isHeaderParsed = false;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        if (!isHeaderParsed) {
          if (buffer.startsWith('__CONTEXT_USED__:')) {
            const sep = buffer.indexOf('\n\n');
            if (sep !== -1) {
              const ctx = buffer.substring('__CONTEXT_USED__:'.length, sep);
              buffer = buffer.substring(sep + 2);
              isHeaderParsed = true;
              yield { context_used: ctx };
            }
          } else {
            isHeaderParsed = true;
          }
        }
        if (isHeaderParsed && buffer) {
          yield { token: buffer };
          buffer = '';
        }
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
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
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
  },
  models: {
    status: () => request<import('../types').ModelsStatusResponse>('/models/status'),
    download: (modelId: string) =>
      request<{ status: string; model_id: string }>('/models/download', {
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
      data?: { lecture_ids?: number[]; model_id?: string; num_questions?: number }
    ) =>
      request<{
        status: string;
        subject_id: number;
        questions: import('../types').QuizQuestion[];
        warning?: string;
      }>(`/study/${subjectId}/generate-quiz`, {
        method: 'POST',
        body: JSON.stringify(data || {}),
      }),
    getOralQuestion: (
      subjectId: number,
      data?: { lecture_ids?: number[]; model_id?: string }
    ) =>
      request<{ status: string; question: string; topic: string }>(
        `/study/${subjectId}/oral-question`,
        {
          method: 'POST',
          body: JSON.stringify(data || {}),
        }
      ),
    evaluateOral: async (
      subjectId: number,
      audioBlob: Blob,
      question: string,
      modelId?: string
    ): Promise<import('../types').OralEvaluationResult> => {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'oral_answer.wav');
      formData.append('question', question);
      if (modelId) formData.append('model_id', modelId);

      const res = await fetch(`${BASE_URL}/study/${subjectId}/evaluate-oral`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return res.json();
    },
  },
  ws: {
    liveLecture: (subjectId: number) => new WebSocket(`${WS_URL}/ws/live-lecture/${subjectId}`),
  },
};
