"""
schemas.py — SbobinAI
Schemi Pydantic per la validazione request/response delle API.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Subject ────────────────────────────────────────────────────────────────────

class SubjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    whisper_prompt_keywords: Optional[str] = None
    color: str = Field(default="#6366f1", pattern=r"^#[0-9a-fA-F]{6}$")


class SubjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    whisper_prompt_keywords: Optional[str] = None
    color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")


class SubjectOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    whisper_prompt_keywords: Optional[str]
    color: str
    created_at: datetime
    lecture_count: int = 0

    model_config = {"from_attributes": True}


# ── Lecture ────────────────────────────────────────────────────────────────────

class LectureCreate(BaseModel):
    subject_id: int
    title: str = Field(default="Lezione senza titolo", max_length=255)
    date: Optional[datetime] = None
    raw_transcript: Optional[str] = None
    processed_notes: Optional[str] = None
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    source_type: str = "recorded"
    in_quaderno: bool = True
    detected_questions: Optional[str] = None


class LectureUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    date: Optional[datetime] = None
    raw_transcript: Optional[str] = None
    processed_notes: Optional[str] = None
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    source_type: Optional[str] = None
    in_quaderno: Optional[bool] = None
    detected_questions: Optional[str] = None


class LectureMove(BaseModel):
    target_subject_id: int


class LectureOut(BaseModel):
    id: int
    subject_id: int
    title: str
    date: datetime
    raw_transcript: Optional[str]
    processed_notes: Optional[str]
    audio_path: Optional[str]
    duration_seconds: Optional[float]
    source_type: Optional[str] = "recorded"
    in_quaderno: Optional[bool] = True
    detected_questions: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Bookmark ───────────────────────────────────────────────────────────────────

class BookmarkCreate(BaseModel):
    lecture_id: int
    timestamp_seconds: float = Field(..., ge=0)
    note: Optional[str] = None


class BookmarkOut(BaseModel):
    id: int
    lecture_id: int
    timestamp_seconds: float
    note: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Materials ──────────────────────────────────────────────────────────────────

class MaterialOut(BaseModel):
    id: str
    name: str
    type: str  # 'pdf' | 'slide' | 'doc' | 'lecture' | 'audio'
    size: str
    uploadedAt: str
    pageCount: Optional[int] = 1
    content: Optional[str] = None
    isAutoImported: bool = False
    lecture_id: Optional[int] = None
    duration_seconds: Optional[float] = None
    in_quaderno: Optional[bool] = None
    source_type: Optional[str] = None


# ── Chat Assistente Corso ──────────────────────────────────────────────────────

class CourseChatRequest(BaseModel):
    query: str
    model_id: Optional[str] = None


# ── Citazioni, Sessioni Chat, Quiz & Visualizzazione Materiali ─────────────────

class CitationRef(BaseModel):
    """Riferimento puntuale a una fonte (slide, pagina, lezione o materiale)."""
    source_id: str
    source_type: str  # "slide" | "lecture" | "material"
    title: str
    page: Optional[int] = None
    label: str = ""


class ChatScope(BaseModel):
    """Filtro opzionale per restringere il contesto RAG a specifiche lezioni o materiali."""
    lecture_ids: list[int] = Field(default_factory=list)
    material_ids: list[str] = Field(default_factory=list)


class ChatStudyAction(BaseModel):
    """Azione di studio interattiva associata a una risposta dell'assistente."""
    type: str
    quiz_id: Optional[int] = None
    title: Optional[str] = None
    num_questions: Optional[int] = None
    question: Optional[str] = None
    topic: Optional[str] = None
    lecture_id: Optional[int] = None


class ContextStats(BaseModel):
    """Statistiche sull'occupazione del contesto della sessione chat."""
    used_tokens: int
    max_tokens: int
    percentage: float
    active_messages_count: int
    compacted_messages_count: int
    has_compacted_summary: bool
    needs_compaction: bool


class ChatMessageOut(BaseModel):
    """Rappresentazione API di un singolo messaggio della sessione chat."""
    id: int
    session_id: int
    role: str
    content: str
    thinking: str = ""
    mode: str = "auto"
    context_used: str = ""
    citations: list[CitationRef] = Field(default_factory=list)
    action: Optional[ChatStudyAction] = None
    followups: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    created_at: datetime


class ChatSessionOut(BaseModel):
    """Metadati principali di una sessione chat persistente."""
    id: int
    subject_id: int
    title: str
    compacted_summary: str = ""
    compacted_up_to_message_id: int = 0
    scope: ChatScope = Field(default_factory=ChatScope)
    context_stats: Optional[ContextStats] = None
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ChatSessionDetailOut(ChatSessionOut):
    """Dettaglio completo di una sessione chat inclusi i messaggi."""
    messages: list[ChatMessageOut] = Field(default_factory=list)


class ChatSessionCreate(BaseModel):
    """Payload creazione nuova sessione chat."""
    title: Optional[str] = "Nuova chat"
    scope: Optional[ChatScope] = None


class ChatSessionUpdate(BaseModel):
    """Payload aggiornamento titolo o scope della sessione chat."""
    title: Optional[str] = None
    scope: Optional[ChatScope] = None


class ChatMessageSend(BaseModel):
    """Payload invio o modifica messaggio chat."""
    message: str
    mode: Optional[str] = "auto"
    scope: Optional[ChatScope] = None
    enable_thinking: bool = False
    rollback_quiz: bool = True


class ChatRegenerateRequest(BaseModel):
    """Payload opzionale per rigenerazione risposta assistente."""
    enable_thinking: bool = False


ChatStreamRequest = ChatMessageSend


class ChatArchiveRequest(BaseModel):
    """Payload archiviazione sessione chat nel Quaderno."""
    title: Optional[str] = None
    mode: Literal["quaderno_and_clear", "quaderno_keep", "quaderno_only"] = "quaderno_and_clear"


class ChatArchiveResponse(BaseModel):
    """Risposta archiviazione sessione chat nel Quaderno."""
    lecture_id: int
    title: str
    session: ChatSessionDetailOut


class ChatSaveNoteRequest(BaseModel):
    """Payload salvataggio di un singolo messaggio come nota."""
    message_id: int
    title: Optional[str] = None


class LegacyChatMessageIn(BaseModel):
    """Messaggio legacy da importare in una nuova sessione."""
    role: str
    content: str
    thinking: str = ""
    context_used: str = ""


class LegacyChatImportRequest(BaseModel):
    """Payload importazione storico chat legacy."""
    title: Optional[str] = "Chat precedente"
    messages: list[LegacyChatMessageIn]


class QuizSetQuestion(BaseModel):
    """Singola domanda a risposta multipla di un QuizSet."""
    question: str
    concept: str = ""
    options: list[str]
    correct_index: int
    explanation: str = ""


class QuizResultItem(BaseModel):
    """Risposta fornita dallo studente a una singola domanda del quiz."""
    question: str
    concept: str = ""
    options: list[str] = Field(default_factory=list)
    selected_index: int
    correct_index: int
    is_correct: bool
    explanation: str = ""


class QuizResultsSubmitRequest(BaseModel):
    """Payload di completamento del quiz con punteggio e risposte dettagliate."""
    score_percent: int
    results: list[QuizResultItem] = Field(default_factory=list)


class QuizSetOut(BaseModel):
    """Set di quiz di verifica restituito dall'API."""
    id: int
    subject_id: int
    chat_session_id: Optional[int] = None
    title: str
    source_label: str = ""
    questions: list[QuizSetQuestion] = Field(default_factory=list)
    completed: bool = False
    score_percent: Optional[int] = None
    results_json: Optional[str] = "[]"
    created_at: datetime


class MaterialPageOut(BaseModel):
    """Contenuto renderizzato (immagine o testo) di una singola pagina/slide."""
    filename: str
    page: int
    page_count: int
    kind: str
    image_data_url: Optional[str] = None
    text: str = ""
    title: str = ""


class StudentConceptOut(BaseModel):
    """Singolo concetto o argomento monitorato nel percorso di studio dello studente."""
    id: int
    subject_id: int
    topic: str
    status: str  # "da_consolidare" | "consolidato"
    error_count: int
    correct_count: int
    last_context: str = ""
    last_tested_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class StudentMasterySummaryOut(BaseModel):
    """Quadro riassuntivo della padronanza accademica per materia."""
    subject_id: int
    total_tracked: int = 0
    consolidated_count: int
    to_consolidate_count: int
    concepts: list[StudentConceptOut] = Field(default_factory=list)


class ConceptMarkRequest(BaseModel):
    """Richiesta per aggiornare lo stato di un concetto ('da_consolidare' | 'consolidato')."""
    topic: str
    status: str






