// types/index.ts — SbobinAI shared types

export interface Subject {
  id: number;
  name: string;
  description: string | null;
  whisper_prompt_keywords: string | null;
  color: string;
  created_at: string;
  lecture_count: number;
}

export interface Lecture {
  id: number;
  subject_id: number;
  title: string;
  date: string;
  raw_transcript: string | null;
  processed_notes: string | null;
  audio_path: string | null;
  duration_seconds: number | null;
  source_type?: 'recorded' | 'imported';
  in_quaderno?: boolean;
  detected_questions?: string | null;
  created_at: string;
}

export interface Bookmark {
  id: number;
  lecture_id: number;
  timestamp_seconds: number;
  note: string | null;
  created_at: string;
}

export type WSMessageType =
  | 'started'
  | 'status'
  | 'transcript_chunk'
  | 'transcript_refined'
  | 'bookmark_saved'
  | 'chat_response'
  | 'chat_chunk'
  | 'question_detected'
  | 'question_answering'
  | 'reconnected'
  | 'paused'
  | 'resumed'
  | 'stopped'
  | 'error';

export interface WSMessage {
  type: WSMessageType;
  payload: Record<string, unknown>;
}

export interface TranscriptChunk {
  id?: string;
  text: string;
  elapsed: number;
  timestamp: number;
  speaker?: string;
}

export interface KaraokeWord {
  text: string;
  start: number; // in milliseconds
  end: number;   // in milliseconds
}

export interface KaraokeLine {
  lineStart: number; // in milliseconds
  lineEnd: number;   // in milliseconds
  words: KaraokeWord[];
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  probability?: number;
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  words: TranscriptWord[];
}

export interface TranscriptKaraokeResponse {
  lines: KaraokeLine[];
  segments?: KaraokeLine[] | TranscriptSegment[];
  cached?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  contextUsed?: string;
}

export interface DetectedQuestion {
  id: string;
  question: string;
  confidence: number;
  timestamp: number;
  answer?: string;
  suggestedAnswer?: string;
  sourceLecture?: string;
}

export type CourseTab = 'quaderno' | 'materiali' | 'quiz' | 'assistente';

export interface CourseMaterial {
  id: string;
  name: string;
  type: 'pdf' | 'slide' | 'doc' | 'lecture' | 'audio';
  size: string;
  uploadedAt: string;
  pageCount?: number;
  content?: string;
  isAutoImported?: boolean;
  lecture_id?: number | null;
  duration_seconds?: number | null;
  in_quaderno?: boolean | null;
  source_type?: string | null;
}

export interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correct_index: number;
  correctIndex?: number;
  explanation: string;
  source_lecture?: string;
}

export interface OralEvaluationResult {
  status: string;
  student_transcript: string;
  grade: string;
  score_num: number;
  good_points: string;
  missing_or_errors: string;
  ideal_formulation: string;
  warning?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  repo_id?: string;
  repos?: string[];
  ollama_tag?: string;
  type: 'whisper' | 'llm' | 'embedding' | 'ollama' | 'cuda';
  approx_size: string;
  approx_bytes?: number;
  description?: string;
  downloaded: boolean;
  system_installed?: boolean;
  disk_size_mb?: number | null;
  status: 'idle' | 'downloading' | 'completed' | 'error' | 'cancelled';
  progress?: number;
  downloaded_mb?: number;
  total_mb?: number;
  stage?: string | null;
  error?: string | null;
  is_default?: boolean;
}

export interface InstalledModelSummary {
  id: string;
  name: string;
  type: string;
  size_mb: number;
}

export interface ModelsStatusResponse {
  whisper: ModelInfo[];
  llm?: ModelInfo[];
  embedding?: ModelInfo[];
  runtime?: ModelInfo[];
  nvidia_gpu_detected?: boolean;
  backend?: string;
  backend_label?: string;
  ollama?: ModelInfo[];
  ollama_running?: boolean;
  installed_count?: number;
  total_installed_mb?: number;
  installed_models?: InstalledModelSummary[];
  cache_path?: string;
}

// ── Persistent Chat & Study Sessions (ChatGPT 2026 Upgrade) ───────────────────

export type ChatStudyMode = 'auto' | 'quiz' | 'socratic' | 'simple' | 'oral' | 'exam';

export interface CitationRef {
  source_id: string;
  source_type: 'slide' | 'lecture' | 'material' | string;
  title: string;
  page?: number | null;
  label: string;
}

export interface ChatScope {
  lecture_ids: number[];
  material_ids: string[];
}

export interface ChatMessageSend {
  message: string;
  mode?: ChatStudyMode;
  scope?: ChatScope;
}

export interface ChatStudyAction {
  type: 'quiz_created' | 'oral_question' | 'saved_to_quaderno' | string;
  quiz_id?: number | null;
  title?: string | null;
  num_questions?: number | null;
  question?: string | null;
  topic?: string | null;
  lecture_id?: number | null;
}

export interface ContextStats {
  used_tokens: number;
  max_tokens: number;
  percentage: number;
  active_messages_count: number;
  compacted_messages_count: number;
  has_compacted_summary: boolean;
  needs_compaction: boolean;
}

export interface PersistentChatMessage {
  id: number;
  session_id: number;
  role: 'user' | 'assistant';
  content: string;
  thinking: string;
  mode: ChatStudyMode | string;
  context_used: string;
  citations: CitationRef[];
  action?: ChatStudyAction | null;
  followups: string[];
  token_estimate: number;
  created_at: string;
}

export interface ChatSession {
  id: number;
  subject_id: number;
  title: string;
  compacted_summary: string;
  compacted_up_to_message_id: number;
  scope: ChatScope;
  context_stats?: ContextStats | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: PersistentChatMessage[];
}

export interface QuizSetQuestion {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
}

export interface QuizSetOut {
  id: number;
  subject_id: number;
  chat_session_id?: number | null;
  title: string;
  source_label: string;
  questions: QuizSetQuestion[];
  created_at: string;
}

export interface MaterialPageOut {
  filename: string;
  page: number;
  page_count: number;
  kind: 'image' | 'text' | string;
  image_data_url?: string | null;
  text: string;
  title: string;
}

export type ChatStreamEvent =
  | {
      type: 'meta';
      user_message: PersistentChatMessage;
      context_used: string;
      citations: CitationRef[];
      context_stats: ContextStats;
    }
  | {
      type: 'token';
      text: string;
    }
  | {
      type: 'action';
      action: ChatStudyAction;
    }
  | {
      type: 'done';
      assistant_message: PersistentChatMessage;
      followups: string[];
      context_stats: ContextStats;
      session_title?: string;
    }
  | {
      type: 'error';
      detail: string;
    };

