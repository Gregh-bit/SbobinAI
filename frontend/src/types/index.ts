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
}

export interface ModelInfo {
  id: string;
  name: string;
  repo_id?: string;
  repos?: string[];
  ollama_tag?: string;
  type: 'whisper' | 'llm' | 'embedding' | 'ollama';
  approx_size: string;
  description?: string;
  downloaded: boolean;
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
  ollama?: ModelInfo[];
  ollama_running?: boolean;
  installed_count?: number;
  total_installed_mb?: number;
  installed_models?: InstalledModelSummary[];
  cache_path?: string;
}

