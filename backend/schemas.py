"""
schemas.py — SbobinAI
Schemi Pydantic per la validazione request/response delle API.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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
    duration_seconds: Optional[float] = None
    source_type: Optional[str] = "recorded"
    in_quaderno: Optional[bool] = True
    detected_questions: Optional[str] = None


class LectureUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=255)
    date: Optional[datetime] = None
    raw_transcript: Optional[str] = None
    processed_notes: Optional[str] = None
    duration_seconds: Optional[float] = None
    source_type: Optional[str] = None
    in_quaderno: Optional[bool] = None
    detected_questions: Optional[str] = None


class LectureOut(BaseModel):
    id: int
    subject_id: int
    title: str
    date: datetime
    raw_transcript: Optional[str]
    processed_notes: Optional[str]
    audio_path: Optional[str]
    duration_seconds: Optional[float]
    source_type: str = "recorded"
    in_quaderno: bool = True
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


# ── WebSocket messages ─────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    """Messaggio generico in/out per la live lecture WebSocket."""
    type: str  # 'transcript_chunk' | 'bookmark' | 'question_detected' | 'chat_query' | 'chat_response' | 'error'
    payload: dict


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


CourseChatRequest.model_rebuild()



