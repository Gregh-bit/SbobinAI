"""
routers/chats.py — SbobinAI Persistent Multi-Turn Chat, Compaction & Archive Router
Gestisce le sessioni di chat persistenti per materia, lo streaming NDJSON multi-turno
con RAG filtrato per scope, compattazione del contesto e archiviazione nel Quaderno.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import re
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import ChatMessage, ChatSession, Lecture, SessionLocal, Subject, get_db
import llm_engine
import rag
from routers.study import create_and_save_quiz_set_threadsafe
from schemas import (
    ChatArchiveRequest,
    ChatArchiveResponse,
    ChatMessageOut,
    ChatMessageSend,
    ChatSaveNoteRequest,
    ChatScope,
    ChatSessionCreate,
    ChatSessionDetailOut,
    ChatSessionOut,
    ChatSessionUpdate,
    ChatStreamRequest,
    ChatStudyAction,
    CitationRef,
    ContextStats,
    LegacyChatImportRequest,
)

logger = logging.getLogger("sbobinai.chats")

router = APIRouter(prefix="/chats", tags=["chats"])

_SYSTEM_RAG_OVERHEAD_TOKENS = 450
_QUIZ_INTENT_RE = re.compile(
    r"\b(crea|genera|fammi|prepara)\b.*\b(quiz|test di verifica|domande a risposta multipla)\b",
    re.IGNORECASE,
)
_ORAL_INTENT_RE = re.compile(
    r"\b(interrogami|simulazione orale|domanda d['’]orale)\b",
    re.IGNORECASE,
)
_THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_FALLBACK_ERROR_SUBSTRINGS = (
    "il modello llm non è ancora stato scaricato",
    "nessun modello",
    "modello llm non ancora scaricato",
    "modello non scaricato",
    "non è ancora stato scaricato",
    "non è ancora stato scaricato in locale",
    "risposta non disponibile: modello llm non ancora scaricato",
)


def _is_fallback_error_message(content: str) -> bool:
    """Riconosce messaggi di errore o fallback dell'LLM che non devono pesare sul contesto."""
    if not content:
        return False
    lower = content.lower()
    return any(sub in lower for sub in _FALLBACK_ERROR_SUBSTRINGS)


def _is_llm_ready(model_id: Optional[str] = None) -> bool:
    """Verifica se il modello LLM risolto è realmente scaricato e pronto."""
    try:
        resolved = llm_engine.resolve_llm_model(model_id)
        return bool(resolved and llm_engine.is_model_downloaded(resolved))
    except Exception:
        return False


# ── Helpers comuni: Lookup, Parsing, Serializzazione & Contesto ───────────────

def _get_subject_or_404(db: Session, subject_id: int) -> Subject:
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")
    return subject


def _get_session_or_404(db: Session, chat_id: int) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessione chat non trovata.")
    return session


def _load_session_messages(db: Session, chat_id: int) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )


def _parse_scope(scope_json: Optional[str]) -> ChatScope:
    """Decodifica difensiva di scope_json in ChatScope."""
    if not scope_json:
        return ChatScope()
    try:
        raw = json.loads(scope_json)
        if not isinstance(raw, dict):
            return ChatScope()
        lecture_ids: List[int] = []
        for lid in raw.get("lecture_ids") or []:
            try:
                val = int(lid)
                if val > 0:
                    lecture_ids.append(val)
            except (TypeError, ValueError):
                continue
        material_ids = [
            str(mid).strip()
            for mid in (raw.get("material_ids") or [])
            if str(mid).strip()
        ]
        return ChatScope(lecture_ids=lecture_ids, material_ids=material_ids)
    except Exception:
        return ChatScope()


def _sanitize_scope(db: Optional[Session], subject_id: int, scope: ChatScope) -> ChatScope:
    """Filtra lecture_ids e material_ids orfani o appartenenti ad altre materie."""
    if not scope.lecture_ids and not scope.material_ids:
        return ChatScope()

    owns_db = db is None
    active_db = SessionLocal() if owns_db else db
    try:
        clean_lec_ids: List[int] = []
        if scope.lecture_ids:
            existing_lec = {
                row[0]
                for row in active_db.query(Lecture.id)
                .filter(Lecture.subject_id == subject_id, Lecture.id.in_(scope.lecture_ids))
                .all()
            }
            clean_lec_ids = [lid for lid in scope.lecture_ids if lid in existing_lec]

        clean_mat_ids: List[str] = []
        if scope.material_ids:
            import os
            from cleaner import get_all_uploads_directories
            folder_name = f"subject_{subject_id}"
            existing_files = set()
            for udir in get_all_uploads_directories():
                s_dir = os.path.join(udir, folder_name)
                if os.path.exists(s_dir):
                    for f in os.listdir(s_dir):
                        if not f.endswith(".meta.json"):
                            existing_files.add(f)
            for mid in scope.material_ids:
                clean_name = str(mid).strip()
                if (
                    clean_name in existing_files
                    or clean_name.startswith(f"infocorso-{subject_id}")
                    or clean_name.lower() == "infocorso.md"
                ):
                    clean_mat_ids.append(clean_name)

        return ChatScope(lecture_ids=clean_lec_ids, material_ids=clean_mat_ids)
    except Exception as exc:
        logger.warning(f"Errore sanitizzazione scope: {exc}")
        return scope
    finally:
        if owns_db:
            active_db.close()


def _extract_think_and_content(raw_text: str) -> Tuple[str, str]:
    """Separa i blocchi `<think>...</think>` dal testo visibile della risposta."""
    if not raw_text:
        return "", ""
    thinks: List[str] = [m.strip() for m in _THINK_BLOCK_RE.findall(raw_text) if m.strip()]
    visible = _THINK_BLOCK_RE.sub("", raw_text)
    if "<think>" in visible:
        head, tail = visible.split("<think>", 1)
        if tail.strip():
            thinks.append(tail.strip())
        visible = head
    return "\n\n".join(thinks).strip(), visible.strip()


def _build_context_stats(
    session: ChatSession,
    messages: List[ChatMessage],
    scope: ChatScope,
) -> ContextStats:
    """Calcola l'occupazione stimata del contesto rispetto a USABLE_CHAT_CONTEXT_TOKENS."""
    compacted_up_to = session.compacted_up_to_message_id or 0
    active_messages = [m for m in messages if m.id > compacted_up_to]
    compacted_count = max(0, len(messages) - len(active_messages))
    summary_text = (session.compacted_summary or "").strip()
    max_tokens = llm_engine.USABLE_CHAT_CONTEXT_TOKENS

    all_fallback = bool(active_messages) and (
        all(_is_fallback_error_message(m.content or "") for m in active_messages)
        or (
            any(m.role == "assistant" for m in active_messages)
            and all(_is_fallback_error_message(m.content or "") for m in active_messages if m.role == "assistant")
            and not summary_text
        )
    )

    if not _is_llm_ready() or all_fallback:
        return ContextStats(
            used_tokens=0,
            max_tokens=max_tokens,
            percentage=0.0,
            active_messages_count=len(active_messages),
            compacted_messages_count=compacted_count,
            has_compacted_summary=bool(summary_text),
            needs_compaction=False,
        )

    summary_tokens = llm_engine.estimate_tokens(summary_text) if summary_text else 0
    active_tokens = sum(
        (m.token_estimate if (m.token_estimate and m.token_estimate > 0) else llm_engine.estimate_tokens(m.content or ""))
        for m in active_messages
        if not _is_fallback_error_message(m.content or "")
    )

    has_active_or_summary = bool(
        summary_tokens > 0 or any(not _is_fallback_error_message(m.content or "") for m in active_messages)
    )
    scope_overhead = (
        _SYSTEM_RAG_OVERHEAD_TOKENS + (20 if (scope.lecture_ids or scope.material_ids) else 0)
        if has_active_or_summary
        else 0
    )
    used_tokens = summary_tokens + active_tokens + scope_overhead
    percentage = round(min(100.0, (used_tokens / max(1, max_tokens)) * 100.0), 1)

    return ContextStats(
        used_tokens=used_tokens,
        max_tokens=max_tokens,
        percentage=percentage,
        active_messages_count=len(active_messages),
        compacted_messages_count=compacted_count,
        has_compacted_summary=bool(summary_text),
        needs_compaction=percentage >= 75.0,
    )


def _serialize_message(m: ChatMessage) -> ChatMessageOut:
    """Converte ChatMessage ORM in ChatMessageOut parsando i campi JSON difensivamente."""
    citations: List[CitationRef] = []
    try:
        raw_cits = json.loads(m.citations_json or "[]")
        if isinstance(raw_cits, list):
            for c in raw_cits:
                if isinstance(c, dict) and c.get("source_id") is not None:
                    citations.append(
                        CitationRef(
                            source_id=str(c.get("source_id") or ""),
                            source_type=str(c.get("source_type") or "material"),
                            title=str(c.get("title") or "Materiale"),
                            page=c.get("page"),
                            label=str(c.get("label") or ""),
                        )
                    )
    except Exception:
        citations = []

    action: Optional[ChatStudyAction] = None
    if m.action_json:
        try:
            raw_action = json.loads(m.action_json)
            if isinstance(raw_action, dict) and raw_action.get("type"):
                action = ChatStudyAction(**raw_action)
        except Exception:
            action = None

    followups: List[str] = []
    try:
        raw_f = json.loads(m.followups_json or "[]")
        if isinstance(raw_f, list):
            followups = [str(x) for x in raw_f if str(x).strip()]
    except Exception:
        followups = []

    token_est = m.token_estimate if (m.token_estimate and m.token_estimate > 0) else llm_engine.estimate_tokens(m.content or "")

    return ChatMessageOut(
        id=m.id,
        session_id=m.session_id,
        role=m.role,
        content=m.content,
        thinking=m.thinking or "",
        mode=m.mode or "auto",
        context_used=m.context_used or "",
        citations=citations,
        action=action,
        followups=followups,
        token_estimate=token_est,
        created_at=m.created_at,
    )


def _build_session_base_kwargs(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
    message_count: Optional[int] = None,
) -> Dict[str, Any]:
    scope = _parse_scope(session.scope_json)
    if messages is not None:
        stats: Optional[ContextStats] = _build_context_stats(session, messages, scope)
        resolved_count = len(messages)
    elif message_count is not None:
        stats = None
        resolved_count = message_count
    else:
        msgs = list(session.messages)
        stats = _build_context_stats(session, msgs, scope)
        resolved_count = len(msgs)
    return {
        "id": session.id,
        "subject_id": session.subject_id,
        "title": session.title or "Nuova chat",
        "compacted_summary": session.compacted_summary or "",
        "compacted_up_to_message_id": session.compacted_up_to_message_id or 0,
        "scope": scope,
        "context_stats": stats,
        "message_count": resolved_count,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _serialize_session_summary(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
    message_count: Optional[int] = None,
) -> ChatSessionOut:
    return ChatSessionOut(**_build_session_base_kwargs(session, messages, message_count=message_count))


def _serialize_session_detail(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
) -> ChatSessionDetailOut:
    msgs = list(session.messages) if messages is None else messages
    base = _build_session_base_kwargs(session, msgs)
    return ChatSessionDetailOut(
        **base,
        messages=[_serialize_message(m) for m in msgs],
    )


def _index_quaderno_lecture(
    subject_id: int,
    lecture_id: int,
    lecture_title: str,
    notes: str,
) -> None:
    """Scarica l'LLM prima di indicizzare nel RAG per sicurezza RAM su Apple Silicon 8GB."""
    llm_engine.unload_llm()
    try:
        rag.index_lecture(
            subject_id=subject_id,
            lecture_id=lecture_id,
            title=lecture_title,
            transcript="",
            notes=notes,
        )
    finally:
        try:
            rag.unload_embedding_model()
        except Exception:
            pass
        llm_engine.clear_metal_cache()


def _ndjson(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


# ── CRUD Sessioni & Messaggi ──────────────────────────────────────────────────

@router.get("/subject/{subject_id}", response_model=List[ChatSessionOut])
def list_chat_sessions(subject_id: int, db: Session = Depends(get_db)):
    """Elenca tutte le sessioni di chat di una materia, ordinate dalla più recente."""
    _get_subject_or_404(db, subject_id)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.subject_id == subject_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .all()
    )
    if not sessions:
        return []
    session_ids = [s.id for s in sessions]
    counts_rows = (
        db.query(ChatMessage.session_id, func.count(ChatMessage.id))
        .filter(ChatMessage.session_id.in_(session_ids))
        .group_by(ChatMessage.session_id)
        .all()
    )
    counts_by_session = {int(sid): int(cnt) for sid, cnt in counts_rows}
    return [
        _serialize_session_summary(s, message_count=counts_by_session.get(s.id, 0))
        for s in sessions
    ]


@router.post("/subject/{subject_id}", response_model=ChatSessionDetailOut)
def create_chat_session(
    subject_id: int,
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    """Crea una nuova sessione chat persistente per la materia."""
    _get_subject_or_404(db, subject_id)
    title = (payload.title or "Nuova chat").strip() or "Nuova chat"
    scope = payload.scope or ChatScope()
    session = ChatSession(
        subject_id=subject_id,
        title=title[:120],
        scope_json=json.dumps(scope.model_dump(), ensure_ascii=False),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _serialize_session_detail(session, [])


@router.post("/subject/{subject_id}/import-legacy", response_model=ChatSessionDetailOut)
def import_legacy_chat(
    subject_id: int,
    payload: LegacyChatImportRequest,
    db: Session = Depends(get_db),
):
    """Importa una conversazione legacy in una nuova ChatSession persistente."""
    _get_subject_or_404(db, subject_id)
    title = (payload.title or "Chat precedente").strip() or "Chat precedente"
    session = ChatSession(
        subject_id=subject_id,
        title=title[:120],
        scope_json="{}",
    )
    db.add(session)
    db.flush()

    inserted: List[ChatMessage] = []
    for item in payload.messages:
        role = (item.role or "").strip().lower()
        content = (item.content or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        msg = ChatMessage(
            session_id=session.id,
            role=role,
            content=content,
            thinking=(item.thinking or "").strip(),
            mode="auto",
            context_used=(item.context_used or "").strip(),
            citations_json="[]",
            action_json="",
            followups_json="[]",
            token_estimate=llm_engine.estimate_tokens(content),
        )
        db.add(msg)
        inserted.append(msg)

    db.commit()
    db.refresh(session)
    return _serialize_session_detail(session, inserted if inserted else [])


@router.get("/{chat_id}", response_model=ChatSessionDetailOut)
def get_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Restituisce il dettaglio completo della sessione chat e dei suoi messaggi."""
    session = _get_session_or_404(db, chat_id)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


@router.patch("/{chat_id}", response_model=ChatSessionDetailOut)
def update_chat_session(
    chat_id: int,
    payload: ChatSessionUpdate,
    db: Session = Depends(get_db),
):
    """Aggiorna il titolo e/o lo scope RAG della sessione chat."""
    session = _get_session_or_404(db, chat_id)
    if payload.title is not None:
        clean_title = payload.title.strip()
        if clean_title:
            session.title = clean_title[:120]
    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


@router.delete("/{chat_id}")
def delete_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Elimina una sessione chat e tutti i suoi messaggi."""
    session = _get_session_or_404(db, chat_id)
    db.delete(session)
    db.commit()
    return {"ok": True}


@router.delete("/{chat_id}/messages/{message_id}", response_model=ChatSessionDetailOut)
def delete_chat_message(chat_id: int, message_id: int, db: Session = Depends(get_db)):
    """Elimina il messaggio e tutti i messaggi successivi nella sessione (cancellazione a cascata)."""
    session = _get_session_or_404(db, chat_id)
    msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato nella sessione.")

    prev_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id, ChatMessage.id < message_id)
        .order_by(ChatMessage.id.desc())
        .first()
    )

    compacted_val = session.compacted_up_to_message_id or 0
    if compacted_val >= message_id:
        session.compacted_up_to_message_id = prev_msg.id if prev_msg else 0
    if not prev_msg:
        session.compacted_up_to_message_id = 0
        session.compacted_summary = None

    db.query(ChatMessage).filter(
        ChatMessage.session_id == chat_id,
        ChatMessage.id >= message_id,
    ).delete(synchronize_session=False)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


# ── Compattazione & Archiviazione Quaderno ────────────────────────────────────

@router.post("/{chat_id}/compact", response_model=ChatSessionDetailOut)
async def compact_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Comprime tutti i turni attivi in `compacted_summary` fino all'ultimo messaggio attuale."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)
    messages = _load_session_messages(db, chat_id)

    if messages:
        compacted_up_to = session.compacted_up_to_message_id or 0
        active = [m for m in messages if m.id > compacted_up_to]
        to_compact = active if active else (messages if not session.compacted_summary else [])
        if to_compact:
            turns = [{"role": m.role, "content": m.content} for m in to_compact]
            try:
                await asyncio.to_thread(rag.unload_embedding_model)
            except Exception:
                pass
            summary = await asyncio.to_thread(
                llm_engine.compact_chat_history,
                subject.name,
                session.compacted_summary or "",
                turns,
            )
            session.compacted_summary = summary

        session.compacted_up_to_message_id = messages[-1].id
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    return _serialize_session_detail(session, messages)


@router.post("/{chat_id}/archive", response_model=ChatArchiveResponse)
async def archive_chat_to_quaderno(
    chat_id: int,
    payload: ChatArchiveRequest,
    db: Session = Depends(get_db),
):
    """Salva la conversazione (e la sintesi compattata) nel Quaderno e reindicizza nel RAG."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)
    messages = _load_session_messages(db, chat_id)

    note_title = (payload.title or "").strip()
    if not note_title:
        base = session.title if (session.title and session.title != "Nuova chat") else f"{subject.name} — Appunti chat"
        note_title = base
    note_title = note_title[:255]

    md_parts: List[str] = [f"# {note_title}"]
    if session.compacted_summary and session.compacted_summary.strip():
        md_parts.append(f"## Sintesi della sessione\n\n{session.compacted_summary.strip()}")

    if messages:
        conv_lines: List[str] = ["## Conversazione e Approfondimenti"]
        for m in messages:
            speaker = "Studente" if m.role == "user" else "Assistente SbobinAI"
            conv_lines.append(f"### {speaker}\n\n{m.content.strip()}")
        md_parts.append("\n\n".join(conv_lines))
    elif not (session.compacted_summary and session.compacted_summary.strip()):
        md_parts.append("Nessun messaggio presente nella sessione archiviata.")

    notes_md = "\n\n".join(md_parts)

    lecture = Lecture(
        subject_id=session.subject_id,
        title=note_title,
        raw_transcript="",
        processed_notes=notes_md,
        source_type="imported",
        in_quaderno=True,
    )
    db.add(lecture)
    db.commit()
    db.refresh(lecture)

    lecture_id = lecture.id
    lecture_title = lecture.title
    subject_id = session.subject_id
    await asyncio.to_thread(_index_quaderno_lecture, subject_id, lecture_id, lecture_title, notes_md)

    if payload.mode == "quaderno_and_clear":
        db.query(ChatMessage).filter(ChatMessage.session_id == chat_id).delete(synchronize_session=False)
        session.compacted_summary = ""
        session.compacted_up_to_message_id = 0
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
        messages = []

    return ChatArchiveResponse(
        lecture_id=lecture_id,
        title=lecture_title,
        session=_serialize_session_detail(session, messages),
    )


@router.post("/{chat_id}/save-note")
async def save_message_as_note(
    chat_id: int,
    payload: ChatSaveNoteRequest,
    db: Session = Depends(get_db),
):
    """Salva una singola risposta della chat come nota nel Quaderno e la indicizza nel RAG."""
    session = _get_session_or_404(db, chat_id)
    msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == payload.message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato.")

    note_title = (payload.title or "").strip()
    if not note_title:
        derived = llm_engine.generate_chat_title(msg.content)
        note_title = derived if derived != "Nuova chat" else f"{session.title} — Nota"
    note_title = note_title[:255]

    lecture = Lecture(
        subject_id=session.subject_id,
        title=note_title,
        raw_transcript="",
        processed_notes=msg.content,
        source_type="imported",
        in_quaderno=True,
    )
    db.add(lecture)
    db.commit()
    db.refresh(lecture)

    lecture_id = lecture.id
    lecture_title = lecture.title
    subject_id = session.subject_id
    await asyncio.to_thread(_index_quaderno_lecture, subject_id, lecture_id, lecture_title, msg.content)
    return {"lecture_id": lecture_id, "title": lecture_title}


# ── Streaming Multi-Turno & Rigenerazione ─────────────────────────────────────

async def _iter_sync_generator_async(
    gen_factory,
    stop_event: Optional[threading.Event] = None,
) -> AsyncIterator[str]:
    """Esegue un generatore sincrono in thread pool emettendo i chunk su asyncio.Queue."""
    queue: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_flag = stop_event or threading.Event()

    def _emit(kind: str, val: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, val))
        except RuntimeError:
            stop_flag.set()

    def _worker() -> None:
        gen: Optional[Iterator[str]] = None
        try:
            gen = gen_factory()
            for chunk in gen:
                if stop_flag.is_set():
                    break
                if chunk:
                    _emit("token", chunk)
        except Exception as exc:
            if not stop_flag.is_set():
                _emit("error", exc)
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:
                    pass
            _emit("end", None)

    threading.Thread(target=_worker, daemon=True, name="chat-multi-turn-stream").start()

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, val = item
            if kind == "token":
                yield val
            elif kind == "error":
                raise val
            elif kind == "end":
                break
    finally:
        stop_flag.set()


def _extract_requested_quiz_count(text: str, default: int = 5) -> int:
    m = re.search(r"\b(\d{1,2})\s+(?:domand[ae]|quesit[oi])\b", text or "", re.IGNORECASE)
    if m:
        try:
            return max(1, min(10, int(m.group(1))))
        except ValueError:
            pass
    return default


def _build_quiz_confirmation_markdown(
    quiz_title: str,
    num_questions: int,
    citations: List[CitationRef],
) -> str:
    lines = [
        f"Ho generato e salvato **{quiz_title}** con **{num_questions} domande** a scelta multipla basate sui materiali del corso.",
        "",
        "Puoi avviare subito la verifica dalla card qui sotto oppure ritrovare questo quiz nella sezione **Quiz & Ripasso**.",
    ]
    ref_links: List[str] = []
    for c in citations[:3]:
        if c.page and c.source_id:
            label = c.label or f"{c.title}, pag. {c.page}"
            ref_links.append(f"[[ref:{c.source_id}#{c.page}|{label}]]")
        elif c.title:
            ref_links.append(c.title)
    if ref_links:
        lines.append("")
        lines.append(f"**Fonti utilizzate:** {' · '.join(ref_links)}")
    return "\n".join(lines)


def _extract_oral_question(content: str, subject_name: str) -> str:
    """Estrae la domanda d'esame principale dal testo generato in modalità orale."""
    cleaned = (content or "").strip()
    if not cleaned:
        return f"Mi illustri i concetti fondamentali affrontati nel corso di {subject_name}."
    for line in cleaned.splitlines():
        stripped = line.strip().lstrip("#*- >\"'")
        if "?" in stripped and len(stripped) >= 12:
            q_part = stripped.split("?", 1)[0].strip() + "?"
            return q_part
    sentences = re.split(r"(?<=[?!.])\s+", cleaned)
    for s in sentences:
        if "?" in s and len(s.strip()) >= 12:
            return s.strip()
    first_line = cleaned.splitlines()[0].strip().lstrip("#*- >\"'")
    return first_line[:240] or f"Mi illustri i concetti fondamentali affrontati nel corso di {subject_name}."


def _build_context_summary(citations: List[CitationRef], context_text: str) -> str:
    titles = list(dict.fromkeys(c.title.strip() for c in citations if c.title and c.title.strip()))
    if titles:
        return " · ".join(titles)
    if (context_text or "").strip():
        return "Appunti e materiali del corso"
    return "Conoscenza generale"


def _persist_assistant_turn(
    *,
    session_id: int,
    replace_message_id: Optional[int],
    content: str,
    thinking: str,
    mode: str,
    context_used: str,
    citations_json: str,
    action_json: str,
    followups_json: str,
    token_estimate: int,
    scope: ChatScope,
) -> Dict[str, Any]:
    """Salva o aggiorna in-place la risposta assistente usando una SessionLocal dedicata."""
    local_db = SessionLocal()
    try:
        session = local_db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise ValueError(f"Sessione chat {session_id} non trovata durante il salvataggio.")

        now_utc = datetime.now(timezone.utc)
        target_msg: Optional[ChatMessage] = None
        if replace_message_id is not None:
            target_msg = (
                local_db.query(ChatMessage)
                .filter(ChatMessage.id == replace_message_id, ChatMessage.session_id == session_id)
                .first()
            )

        if target_msg is not None:
            if session.compacted_up_to_message_id == target_msg.id:
                prev_msg = (
                    local_db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id, ChatMessage.id < target_msg.id)
                    .order_by(ChatMessage.id.desc())
                    .first()
                )
                session.compacted_up_to_message_id = prev_msg.id if prev_msg else 0

            target_msg.content = content
            target_msg.thinking = thinking
            target_msg.mode = mode
            target_msg.context_used = context_used
            target_msg.citations_json = citations_json
            target_msg.action_json = action_json
            target_msg.followups_json = followups_json
            target_msg.token_estimate = token_estimate
            target_msg.created_at = now_utc
            assistant_msg = target_msg
        else:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                thinking=thinking,
                mode=mode,
                context_used=context_used,
                citations_json=citations_json,
                action_json=action_json,
                followups_json=followups_json,
                token_estimate=token_estimate,
            )
            local_db.add(assistant_msg)

        session.updated_at = now_utc
        local_db.commit()
        local_db.refresh(assistant_msg)
        local_db.refresh(session)

        updated_messages = _load_session_messages(local_db, session_id)
        stats_after = _build_context_stats(session, updated_messages, scope)
        return {
            "assistant_message": _serialize_message(assistant_msg).model_dump(mode="json"),
            "context_stats": stats_after.model_dump(mode="json"),
            "session_title": session.title or "Nuova chat",
        }
    except Exception:
        local_db.rollback()
        raise
    finally:
        local_db.close()


def _stream_chat_turn(
    *,
    session_id: int,
    subject_id: int,
    subject_name: str,
    compacted_summary: str,
    user_msg_dict: Dict[str, Any],
    user_msg_content: str,
    history_messages: List[Dict[str, str]],
    stats_before_dict: Dict[str, Any],
    mode: str,
    scope_lecture_ids: List[int],
    scope_material_ids: List[str],
    replace_message_id: Optional[int] = None,
) -> StreamingResponse:
    """Esegue la pipeline RAG + LLM streaming NDJSON condivisa fra `/stream` e `/regenerate-stream`."""
    effective_mode = (mode or "auto").strip().lower() or "auto"
    scope_snapshot = _sanitize_scope(
        None,
        subject_id,
        ChatScope(
            lecture_ids=list(scope_lecture_ids),
            material_ids=list(scope_material_ids),
        ),
    )
    had_scope = bool(scope_lecture_ids or scope_material_ids)
    if had_scope and not (scope_snapshot.lecture_ids or scope_snapshot.material_ids):
        # Se tutti gli elementi nello scope sono stati eliminati da disco/DB, fallback al corso intero
        logger.info(f"[Chats] Risorse nello scope eliminate: fallback al corso intero per chat {session_id}.")
        query_lec_ids: Optional[List[int]] = None
        query_mat_ids: Optional[List[str]] = None
    else:
        query_lec_ids = scope_snapshot.lecture_ids or None
        query_mat_ids = scope_snapshot.material_ids or None

    async def event_stream() -> AsyncIterator[str]:
        stream_stop_event = threading.Event()
        try:
            rag_res = await asyncio.to_thread(
                rag.query_context_structured,
                subject_id=subject_id,
                query=user_msg_content,
                top_k=5,
                scope_lecture_ids=query_lec_ids,
                scope_material_ids=query_mat_ids,
            )
            try:
                await asyncio.to_thread(rag.unload_embedding_model)
            except Exception:
                pass
            llm_engine.clear_metal_cache()

            context_text = str(rag_res.get("context_text") or "")
            raw_citations = rag_res.get("citations") or []
            citations: List[CitationRef] = []
            for c in raw_citations:
                if isinstance(c, dict) and c.get("source_id") is not None:
                    citations.append(
                        CitationRef(
                            source_id=str(c.get("source_id") or ""),
                            source_type=str(c.get("source_type") or "material"),
                            title=str(c.get("title") or "Materiale"),
                            page=c.get("page"),
                            label=str(c.get("label") or ""),
                        )
                    )

            context_used = _build_context_summary(citations, context_text)

            yield _ndjson({
                "type": "meta",
                "user_message": user_msg_dict,
                "context_used": context_used,
                "citations": [c.model_dump(mode="json") for c in citations],
                "context_stats": stats_before_dict,
            })

            is_quiz = (effective_mode == "quiz") or (
                effective_mode == "auto" and bool(_QUIZ_INTENT_RE.search(user_msg_content))
            )
            is_oral = (not is_quiz) and (
                (effective_mode == "oral")
                or (effective_mode == "auto" and bool(_ORAL_INTENT_RE.search(user_msg_content)))
            )

            resolved_mode = effective_mode
            raw_buffer: List[str] = []
            emitted_action: Optional[ChatStudyAction] = None

            if is_quiz:
                requested_n = _extract_requested_quiz_count(user_msg_content, default=5)
                topic_hint = llm_engine.generate_chat_title(user_msg_content)
                if topic_hint == "Nuova chat":
                    topic_hint = ""
                try:
                    quiz_info = await asyncio.to_thread(
                        create_and_save_quiz_set_threadsafe,
                        subject_id=subject_id,
                        chat_session_id=session_id,
                        num_questions=requested_n,
                        context_text=context_text,
                        topic_hint=topic_hint,
                        source_label=context_used,
                        scope_lecture_ids=query_lec_ids,
                        scope_material_ids=query_mat_ids,
                        subject_name=subject_name,
                    )
                    num_saved = int(quiz_info.get("num_questions") or requested_n)
                    quiz_title = str(quiz_info.get("title") or f"Quiz su {subject_name}")
                    reply_md = _build_quiz_confirmation_markdown(quiz_title, num_saved, citations)
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

                    emitted_action = ChatStudyAction(
                        type="quiz_created",
                        quiz_id=int(quiz_info["id"]),
                        title=quiz_title,
                        num_questions=num_saved,
                    )
                    resolved_mode = "quiz"
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                except Exception as quiz_exc:
                    logger.warning(f"[Chats] Generazione quiz fallita, fallback a risposta testuale: {quiz_exc}")
                    resolved_mode = "auto"
                    async for chunk in _iter_sync_generator_async(
                        lambda: llm_engine.stream_multi_turn_chat(
                            subject_name=subject_name,
                            messages=history_messages,
                            context_text=context_text,
                            compacted_summary=compacted_summary,
                            mode="auto",
                        ),
                        stop_event=stream_stop_event,
                    ):
                        raw_buffer.append(chunk)
                        yield _ndjson({"type": "token", "text": chunk})
            else:
                stream_mode = "oral" if is_oral else effective_mode
                if is_oral:
                    resolved_mode = "oral"
                async for chunk in _iter_sync_generator_async(
                    lambda: llm_engine.stream_multi_turn_chat(
                        subject_name=subject_name,
                        messages=history_messages,
                        context_text=context_text,
                        compacted_summary=compacted_summary,
                        mode=stream_mode,
                    ),
                    stop_event=stream_stop_event,
                ):
                    raw_buffer.append(chunk)
                    yield _ndjson({"type": "token", "text": chunk})

            raw_full = "".join(raw_buffer)
            thinking, clean_content = _extract_think_and_content(raw_full)
            if not clean_content:
                clean_content = "Risposta completata."

            if is_oral and emitted_action is None:
                extracted_q = _extract_oral_question(clean_content, subject_name)
                topic_str = llm_engine.generate_chat_title(user_msg_content)
                emitted_action = ChatStudyAction(
                    type="oral_question",
                    question=extracted_q,
                    topic=topic_str if topic_str != "Nuova chat" else subject_name,
                )
                yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})

            followups = llm_engine.generate_followup_suggestions(
                subject_name=subject_name,
                user_message=user_msg_content,
                assistant_reply=clean_content,
                mode=resolved_mode,
            )

            persisted = await asyncio.to_thread(
                _persist_assistant_turn,
                session_id=session_id,
                replace_message_id=replace_message_id,
                content=clean_content,
                thinking=thinking,
                mode=resolved_mode,
                context_used=context_used,
                citations_json=json.dumps([c.model_dump(mode="json") for c in citations], ensure_ascii=False),
                action_json=json.dumps(emitted_action.model_dump(mode="json"), ensure_ascii=False)
                if emitted_action
                else "",
                followups_json=json.dumps(followups, ensure_ascii=False),
                token_estimate=llm_engine.estimate_tokens(clean_content),
                scope=scope_snapshot,
            )

            yield _ndjson({
                "type": "done",
                "assistant_message": persisted["assistant_message"],
                "followups": followups,
                "context_stats": persisted["context_stats"],
                "session_title": persisted["session_title"],
            })
        except asyncio.CancelledError:
            stream_stop_event.set()
            logger.info(f"[Chats] Streaming chat {session_id} annullato (CancelledError).")
            raise
        except Exception as exc:
            stream_stop_event.set()
            logger.exception(f"[Chats] Errore durante lo streaming della chat {session_id}: {exc}")
            yield _ndjson({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{chat_id}/stream")
def stream_chat_message(
    chat_id: int,
    payload: ChatStreamRequest,
    db: Session = Depends(get_db),
):
    """Salva il messaggio utente, interroga il RAG con scope e streamma la risposta in NDJSON."""
    text = (payload.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")

    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    scope = payload.scope if payload.scope is not None else _parse_scope(session.scope_json)

    existing_messages = _load_session_messages(db, chat_id)
    has_prior_user = any(m.role == "user" for m in existing_messages)
    if (not session.title or session.title == "Nuova chat") and not has_prior_user:
        session.title = llm_engine.generate_chat_title(text)

    effective_mode = (payload.mode or "auto").strip().lower() or "auto"
    user_msg = ChatMessage(
        session_id=chat_id,
        role="user",
        content=text,
        thinking="",
        mode=effective_mode,
        context_used="",
        citations_json="[]",
        action_json="",
        followups_json="[]",
        token_estimate=llm_engine.estimate_tokens(text),
    )
    db.add(user_msg)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(session)

    all_messages = [*existing_messages, user_msg]
    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in all_messages
        if m.id > compacted_up_to
    ]

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=None,
    )


@router.post("/{chat_id}/messages/{message_id}/regenerate-stream")
def regenerate_chat_message_stream(
    chat_id: int,
    message_id: int,
    db: Session = Depends(get_db),
):
    """Rigenera in-place la risposta assistente indicata in streaming NDJSON dal turno utente precedente."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    target_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not target_msg:
        raise HTTPException(status_code=404, detail="Messaggio da rigenerare non trovato.")
    if target_msg.role != "assistant":
        raise HTTPException(status_code=400, detail="Puoi rigenerare solo un messaggio dell'assistente.")

    all_messages = _load_session_messages(db, chat_id)
    before_target = [m for m in all_messages if m.id < target_msg.id]
    user_msg = next((m for m in reversed(before_target) if m.role == "user"), None)
    if user_msg is None:
        raise HTTPException(status_code=400, detail="Nessun messaggio utente precedente trovato per rigenerare.")

    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in before_target
        if m.id > compacted_up_to
    ]
    if not history_messages:
        history_messages = [{"role": user_msg.role, "content": user_msg.content}]

    scope = _parse_scope(session.scope_json)
    effective_mode = (user_msg.mode or target_msg.mode or "auto").strip().lower() or "auto"

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=target_msg.id,
    )


@router.post("/{chat_id}/messages/{message_id}/edit-stream")
def edit_chat_message_stream(
    chat_id: int,
    message_id: int,
    payload: ChatMessageSend,
    db: Session = Depends(get_db),
):
    """Modifica un messaggio utente, elimina i messaggi successivi e avvia lo streaming della nuova risposta."""
    text = (payload.message or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")

    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    user_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not user_msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato nella sessione.")
    if user_msg.role != "user":
        raise HTTPException(status_code=400, detail="Puoi modificare solo un messaggio utente.")

    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    scope = payload.scope if payload.scope is not None else _parse_scope(session.scope_json)

    effective_mode = (payload.mode or user_msg.mode or "auto").strip().lower() or "auto"

    user_msg.content = text
    user_msg.mode = effective_mode
    user_msg.token_estimate = llm_engine.estimate_tokens(text)

    # Elimina tutti i messaggi successivi in quella sessione (ChatMessage.id > message_id)
    db.query(ChatMessage).filter(
        ChatMessage.session_id == chat_id,
        ChatMessage.id > message_id,
    ).delete(synchronize_session=False)

    compacted_val = session.compacted_up_to_message_id or 0
    if compacted_val >= message_id:
        prev_msg = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == chat_id, ChatMessage.id < message_id)
            .order_by(ChatMessage.id.desc())
            .first()
        )
        if prev_msg:
            session.compacted_up_to_message_id = prev_msg.id
        else:
            session.compacted_up_to_message_id = 0
            session.compacted_summary = None

    has_prior_user = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == chat_id,
            ChatMessage.role == "user",
            ChatMessage.id < message_id,
        )
        .first()
        is not None
    )
    if (not session.title or session.title == "Nuova chat") and not has_prior_user:
        session.title = llm_engine.generate_chat_title(text)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(session)

    all_messages = _load_session_messages(db, chat_id)
    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in all_messages
        if m.id > compacted_up_to
    ]
    if not history_messages:
        history_messages = [{"role": user_msg.role, "content": user_msg.content}]

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=None,
    )
