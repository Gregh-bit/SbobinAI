"""
main.py — SbobinAI Backend
Server FastAPI principale con:
 - API REST per Subjects, Lectures, Bookmarks
 - WebSocket /ws/live-lecture/{subject_id} per lo streaming live
 - AudioEngine in RAM (zero disk I/O durante la registrazione)
 - Mock Transcriber a zero download dati
 - CORS configurato per il frontend Vite (localhost:5173) e Tauri
"""
from __future__ import annotations

import sys
# Su Windows assicura che stdout e stderr usino UTF-8 con fallback 'replace'
# per evitare crash da UnicodeEncodeError (es. charmap cp1252) su print/logger
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
# macOS SSL Certificates & HuggingFace Transfer Configuration
try:
    import certifi
    ca_bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_bundle
    os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
except ImportError:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import asyncio
import json
import logging
import queue
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Lecture, SessionLocal, get_db, init_db
from live_session import LiveSession, get_session, has_active_sessions, start_session
from routers.subjects import router as subjects_router
from routers.lectures import router as lectures_router, bookmark_router
from routers.settings import router as settings_router
from routers.models import router as models_router
from routers.study import router as study_router
from routers.chats import router as chats_router
from schemas import CourseChatRequest
from config import load_settings


# Console Windows (cp1252): evita UnicodeEncodeError con emoji nei log/print
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sbobinai")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inizializza il DB all'avvio, bonifica i file orfani, esegue cleanup allo spegnimento."""
    logger.info("[App] SbobinAI Backend avviato")
    init_db()
    # Bonifica automatica file orfani e compattazione DB all'avvio
    try:
        from cleaner import cleanup_orphan_files
        cleanup_orphan_files()
    except Exception as exc:
        logger.warning(f"[Startup] Cleanup orfani: {exc}")

    # Su MacBook Air M1 con 8GB di RAM, Whisper viene caricato on-demand solo all'avvio del live
    yield
    logger.info("[App] SbobinAI Backend in shutdown")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SbobinAI API",
    description="Backend per l'assistente universitario locale SbobinAI",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: permetti richieste dal frontend Vite (sia localhost che 127.0.0.1) e da Tauri
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_local_origin_middleware(request: Request, call_next):
    # Protezione CSRF per client locali: se un browser invia un'intestazione Origin esterna
    # per una richiesta che modifica lo stato (POST, PUT, PATCH, DELETE), rifiutala con 403.
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("origin")
        if origin:
            allowed_origins = {
                "null",
                "tauri://localhost",
                "http://tauri.localhost",
                "https://tauri.localhost",
            }
            if origin not in allowed_origins and not re.fullmatch(r"https?://(localhost|127\.0\.0\.1)(:\d+)?/?", origin):
                return JSONResponse(status_code=403, content={"detail": "Origine non consentita."})
    return await call_next(request)

# ── Router ─────────────────────────────────────────────────────────────────────
app.include_router(subjects_router, prefix="/api/v1")
app.include_router(lectures_router, prefix="/api/v1")
app.include_router(bookmark_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(study_router, prefix="/api/v1")
app.include_router(chats_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Endpoint di healthcheck."""
    return {
        "status": "ok",
        "service": "SbobinAI",
        "version": "0.1.0",
        "app_version": os.environ.get("SBOBINAI_APP_VERSION") or "0.0.17",
    }


@app.post("/api/v1/system/shutdown", tags=["System"])
async def system_shutdown(request: Request):
    """Arresto pulito del server locale richiesto dal client desktop (es. aggiornamento app)."""
    _check_local_control_origin(request)
    import signal
    import threading

    def _terminate():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Timer(0.15, _terminate).start()
    return {"status": "shutting_down"}


def _check_local_control_origin(request: Request) -> None:
    """Blocca richieste cross-site alle API locali che controllano il processo o il disco."""
    origin = request.headers.get("origin")
    if not origin:
        return  # client nativi/CLI; i browser inviano Origin nelle richieste cross-site
    if origin in {"null", "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"}:
        return
    if re.fullmatch(r"https?://(localhost|127\.0\.0\.1)(:\d+)?/?", origin):
        return
    raise HTTPException(status_code=403, detail="Origine non consentita.")


def extract_professor_questions(text: str) -> list[str]:
    """
    Rileva le domande poste dal professore durante la spiegazione:
    1. Frasi terminate da punto di domanda '?'
    2. Frasi aperte da indicatori interrogativi accademici tipici
    """
    import re
    clean = text.strip()
    if not clean:
        return []

    results = []
    # Splitta per punteggiatura terminale (. ? !) mantenendo le frasi distinte
    sentences = re.split(r'(?<=[.?!])\s+|\n+', clean)

    interrogative_cues = (
        "chi sa dirmi", "chi mi sa dire", "qual è", "quale sarebbe",
        "cosa accade se", "cosa succede se", "come mai", "perché",
        "cosa intendiamo per", "come possiamo definire", "chi ricorda",
        "quanto vale", "dove si trova", "come funziona", "qual è la definizione",
        "chi vuole rispondere", "ci sono domande",
    )

    # Intercalari retorici di chiusura (es. "..., ok?", "..., chiaro?", "..., no?") da non scambiare per vere domande
    rhetorical_tag_re = re.compile(
        r"(?:,\s*|\s+)(?:ok|okay|chiaro|giusto|vero|capito|no|sì|si|eh|bene|d'accordo|va bene)\s*\?+$",
        re.IGNORECASE,
    )

    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 5:
            continue
        low = s_clean.lower()
        if "?" in s_clean:
            # Scarta frasi puramente affermative chiuse da un intercalare retorico (", ok?") se non iniziano con un vero pronome/cue interrogativo
            if rhetorical_tag_re.search(s_clean) and not any(low.startswith(cue) for cue in interrogative_cues):
                continue
            results.append(s_clean)
        elif any(low.startswith(cue) for cue in interrogative_cues):
            results.append(s_clean + "?")

    return results


# ── Task in background ─────────────────────────────────────────────────────────
# L'event loop tiene solo riferimenti deboli ai task: senza questo set un task
# "fire-and-forget" può essere raccolto dal GC a metà esecuzione.
_bg_tasks: Set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


def _persist_detected_questions(session: LiveSession) -> None:
    """Salva su SQLite le domande della sessione (usato quando una risposta flash arriva a sessione chiusa)."""
    db_sess = SessionLocal()
    try:
        lecture = db_sess.query(Lecture).filter(Lecture.id == session.lecture_id).first()
        if lecture:
            lecture.detected_questions = (
                json.dumps(session.detected_questions, ensure_ascii=False)
                if session.detected_questions else None
            )
            db_sess.commit()
    except Exception as exc:
        logger.warning(f"[Flash Answer] Errore salvataggio domande su DB: {exc}")
    finally:
        db_sess.close()


_ws_locks: dict[WebSocket, asyncio.Lock] = {}


async def _safe_send_json(ws: WebSocket, payload: dict) -> bool:
    """Invia un payload JSON su WebSocket serializzando le scritture con un lock per socket."""
    if getattr(ws, "client_state", None) != WebSocketState.CONNECTED:
        _ws_locks.pop(ws, None)
        return False
    lock = _ws_locks.get(ws)
    if lock is None:
        lock = asyncio.Lock()
        _ws_locks[ws] = lock
    try:
        async with lock:
            await ws.send_json(payload)
        return True
    except Exception as e:
        logger.debug(f"[WS] Errore safe_send_json: {e}")
        _ws_locks.pop(ws, None)
        return False


async def _generate_flash_answer_task(session: LiveSession, question: str, question_id: str, ws: WebSocket):
    """
    Genera la risposta flash ultra-rapida in streaming con l'LLM locale (thinking disattivato)
    usando la trascrizione live con timestamp + lookup lessicale istantaneo dei materiali del corso.
    """
    try:
        from rag import query_context_structured, is_embedding_loaded
        from llm_engine import stream_flash_answer, _deduplicate_repeated_answer

        # 1. Recupera gli ultimi 15 minuti della trascrizione live CON TIMESTAMP [MM:SS | X min fa] (0ms RAM)
        live_context = session.get_timestamped_recent_context(minutes=15, max_chars=4500)
        if not live_context:
            live_context = "(Nessuna frase ancora trascritta in questa lezione.)"

        # 2. Ricerca istantanea (<5ms) sui materiali e lezioni precedenti: usa vettoriale solo se l'embedder è già caldo in RAM
        rag_res = await asyncio.to_thread(
            query_context_structured,
            subject_id=session.subject_id,
            query=question,
            top_k=3,
            priority_previous_lecture=True,
            use_vector=is_embedding_loaded(),
        )
        rag_context = (rag_res.get("context_text") or "").strip()

        parts = [f"TRASCRIZIONE IN DIRETTA DELLA LEZIONE CORRENTE:\n{live_context}"]
        if rag_context:
            parts.append(f"MATERIALI E LEZIONI PRECEDENTI DEL CORSO:\n{rag_context}")
        combined = "\n\n".join(parts)

        settings = load_settings()
        llm_model = settings.get("llm_model")

        # 3. Streaming token-by-token (enable_thinking=False garantito) per risposta istantanea nella card
        answer_chunks: list[str] = []
        async for token in stream_flash_answer(question=question, context=combined, model_id=llm_model):
            answer_chunks.append(token)
            await _safe_send_json(ws, {
                "type": "question_answer_chunk",
                "payload": {
                    "id": question_id,
                    "chunk": token,
                }
            })

        flash_answer = _deduplicate_repeated_answer("".join(answer_chunks)) or "Risposta completata."

        # Aggiorna la risposta nella sessione in memoria (se la domanda non è stata eliminata)
        found = False
        for q_item in session.detected_questions:
            if q_item.get("id") == question_id:
                q_item["suggestedAnswer"] = flash_answer
                found = True
                break
        if not found:
            return

        # La sessione è già stata chiusa e salvata: aggiorna direttamente il DB
        if session.finalized:
            await asyncio.to_thread(_persist_detected_questions, session)

        # Invia l'aggiornamento definitivo via WebSocket
        sent = await _safe_send_json(ws, {
            "type": "question_detected",
            "payload": {
                "id": question_id,
                "question": question,
                "confidence": 0.98,
                "suggested_answer": flash_answer,
                "trigger": "TRIGGER_RAG",
            }
        })
        if sent:
            logger.info(f"[Flash Answer] Risposta generata e inviata via WS per '{question[:30]}...'")
    except Exception as exc:
        logger.warning(f"[Flash Answer] Errore generazione risposta flash: {exc}")
        await _safe_send_json(ws, {
            "type": "error",
            "payload": {
                "id": question_id,
                "message": f"Errore risposta rapida: {exc}",
            }
        })


async def _handle_live_chat_stream(
    session: LiveSession,
    ws: WebSocket,
    query: str,
    context_minutes: int,
    subject_id: int,
    enable_thinking: bool = False,
    history: Optional[list[dict]] = None,
) -> None:
    """
    Streaming asincrono in background per la chat live RAG (stile ChatGPT 2026 con fonti strutturate e opzione Pensa).
    Non blocca l'event loop né il socket e gestisce ogni errore in modo sicuro.
    """
    from rag import query_context_structured, is_embedding_loaded, unload_embedding_model
    from llm_engine import stream_chat_response, clear_metal_cache, is_model_downloaded, resolve_llm_model

    live_context = ""
    citations: list[dict] = []
    try:
        settings = load_settings()
        llm_model = settings.get("llm_model")
        has_llm = is_model_downloaded(resolve_llm_model(llm_model))

        live_context = session.get_timestamped_recent_context(minutes=context_minutes, max_chars=5000)

        rag_context = ""
        rag_citations: list[dict] = []
        # Se nessun LLM è scaricato, evita di caricare/scaricare 600MB di embedding (risposta immediata in 0ms)
        if has_llm:
            was_loaded = is_embedding_loaded()
            try:
                rag_res = await asyncio.to_thread(
                    query_context_structured,
                    subject_id=subject_id,
                    query=query,
                    top_k=3,
                    priority_previous_lecture=True,
                    use_vector=True,
                )
                rag_context = (rag_res.get("context_text") or "").strip()
                rag_citations = list(rag_res.get("citations") or [])
            finally:
                if not was_loaded:
                    await asyncio.to_thread(unload_embedding_model)
                    clear_metal_cache()

        if live_context.strip():
            citations.append({
                "source_id": f"lecture-{session.lecture_id}",
                "source_type": "live_lecture",
                "title": "Trascrizione in diretta",
                "page": None,
                "label": "Lezione in diretta",
            })
        if has_llm:
            citations.extend(rag_citations)

        combined_parts = []
        if live_context.strip():
            combined_parts.append(
                f"TRASCRIZIONE LIVE DELLA LEZIONE IN CORSO (ultimi {context_minutes} minuti):\n{live_context}"
            )
        if rag_context:
            combined_parts.append(f"MATERIALI E LEZIONI PRECEDENTI DEL CORSO:\n{rag_context}")
        combined_context = "\n\n".join(combined_parts) or "Nessun contesto disponibile per ora."

        context_preview = (live_context[:400] + "...") if len(live_context) > 400 else live_context

        # Invia subito i metadati (fonti strutturate) prima del primo token per il pannello Fonti
        await _safe_send_json(ws, {
            "type": "chat_meta",
            "payload": {
                "query": query,
                "citations": citations,
                "context_used": context_preview,
            }
        })

        if not has_llm:
            # Pausa realistica per visualizzare l'animazione di analisi fonti prima dello switch
            await asyncio.sleep(0.40)
            notice_chunks = [
                "Il modello LLM non è ancora stato scaricato in locale.\n\n",
                "Apri le **Impostazioni** e scarica il modello per abilitare l'assistente IA 100% offline."
            ]
            full_notice = "".join(notice_chunks)
            for chunk in notice_chunks:
                await _safe_send_json(ws, {
                    "type": "chat_chunk",
                    "payload": {
                        "chunk": chunk,
                        "query": query,
                    }
                })
                await asyncio.sleep(0.04)

            await _safe_send_json(ws, {
                "type": "chat_response",
                "payload": {
                    "answer": full_notice,
                    "text": full_notice,
                    "query": query,
                    "citations": citations,
                    "context_used": context_preview,
                }
            })
            return

        full_answer = ""
        try:
            async for token in stream_chat_response(
                query=query,
                context=combined_context,
                model_id=llm_model,
                enable_thinking=enable_thinking,
                history=history,
            ):
                full_answer += token
                sent = await _safe_send_json(ws, {
                    "type": "chat_chunk",
                    "payload": {
                        "chunk": token,
                        "query": query,
                    }
                })
                if not sent:
                    return
        except Exception as c_err:
            logger.error(f"[WS Chat] Errore streaming: {c_err}")
            err_msg = f"Non è stato possibile completare la risposta ({c_err})."
            full_answer = (full_answer + f"\n\n{err_msg}").strip() if full_answer else err_msg
            await _safe_send_json(ws, {
                "type": "chat_chunk",
                "payload": {
                    "chunk": f"\n\n{err_msg}",
                    "query": query,
                }
            })

        await _safe_send_json(ws, {
            "type": "chat_response",
            "payload": {
                "answer": full_answer,
                "text": full_answer,
                "query": query,
                "citations": citations,
                "context_used": context_preview,
            }
        })
    except Exception as exc:
        logger.error(f"[WS Chat] Errore non gestito in _handle_live_chat_stream: {exc}")
        await _safe_send_json(ws, {
            "type": "chat_response",
            "payload": {
                "answer": f"Si è verificato un errore durante l'elaborazione della risposta: {exc}",
                "query": query,
                "citations": citations,
                "context_used": (live_context[:400] + "...") if len(live_context) > 400 else live_context,
            }
        })


async def _post_lecture_summary_and_index(subject_id: int, lecture_id: int, full_transcript: str):
    """
    Funzione post-lezione eseguita a fine registrazione:
    1. Scarica Whisper dalla memoria per lasciare la memoria all'LLM
    2. Cerca riferimenti a materiali e lezioni passate su LanceDB
    3. Genera il capitolo completo strutturato per il Quaderno
    4. Salva il capitolo nel DB SQLite e lo indicizza automaticamente in LanceDB
    """
    from database import Bookmark
    from rag import query_context, index_lecture, unload_embedding_model
    from llm_engine import (
        cancel_epoch,
        clear_metal_cache,
        generate_lecture_chapter,
        is_cancelled,
        unload_llm,
        unload_whisper,
    )

    start_epoch = cancel_epoch()

    # Libera preventivamente la memoria da Whisper (solo se nessun'altra lezione live lo sta usando)
    if not has_active_sessions():
        await asyncio.to_thread(unload_whisper)

    db_sess = SessionLocal()
    try:
        lecture = db_sess.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture:
            return

        if is_cancelled(start_epoch):
            raise InterruptedError("Generazione capitolo interrotta dall'utente.")

        # Riferimenti da lezioni passate citati o correlati (Embedding model)
        from transcript_cleaner import clean_transcript_for_llm
        cleaned_for_query = clean_transcript_for_llm(full_transcript)
        try:
            past_refs = await asyncio.to_thread(
                query_context,
                subject_id=subject_id,
                query=cleaned_for_query[:1500],
                top_k=3,
                priority_previous_lecture=True,
            )
        finally:
            # Rilascia modello embedding prima di caricare l'LLM pesante per evitare swap su 8GB
            unload_embedding_model()
            clear_metal_cache()

        if is_cancelled(start_epoch):
            raise InterruptedError("Generazione capitolo interrotta dall'utente.")

        # Segnalibri della lezione
        bookmarks = db_sess.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
        bm_summary = (
            "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bookmarks])
            if bookmarks else "Nessun segnalibro specifico inserito."
        )

        settings = load_settings()
        llm_model = settings.get("llm_model")

        from llm_engine import is_model_downloaded, resolve_llm_model
        target_llm = resolve_llm_model(llm_model)
        if not is_model_downloaded(target_llm):
            # Non salvare mai un segnaposto "Modello LLM non installato" dentro processed_notes:
            # indicizza solo la trascrizione grezza per il RAG e lascia il Quaderno pulito.
            logger.info(
                f"[Post-Lezione] Modello LLM '{target_llm}' non scaricato: salto generazione automatica capitolo "
                f"per lezione {lecture_id} (indicizzo solo la trascrizione)."
            )
            try:
                await asyncio.to_thread(
                    index_lecture,
                    subject_id=subject_id,
                    lecture_id=lecture_id,
                    title=lecture.title,
                    transcript=full_transcript,
                    notes=None,
                )
            finally:
                unload_embedding_model()
                clear_metal_cache()
            return

        chapter = await generate_lecture_chapter(
            title=lecture.title,
            transcript=full_transcript,
            past_references=past_refs,
            bookmarks_summary=bm_summary,
            model_id=llm_model,
            lecture_id=lecture.id,
        )
        if not chapter:
            logger.warning(f"[Post-Lezione] Nessun capitolo generato per lezione {lecture_id}: note invariate.")
            return

        lecture.processed_notes = chapter
        db_sess.commit()
        logger.info(f"[Post-Lezione] Capitolo Quaderno generato e salvato su SQLite per lezione {lecture_id}.")

        # Rilascia l'LLM prima di passare all'indicizzazione vettoriale
        unload_llm()
        clear_metal_cache()

        # Auto-indicizzazione su LanceDB
        try:
            await asyncio.to_thread(
                index_lecture,
                subject_id=subject_id,
                lecture_id=lecture_id,
                title=lecture.title,
                transcript=full_transcript,
                notes=chapter,
            )
        finally:
            # Rilascia embedding post-indicizzazione
            unload_embedding_model()
            clear_metal_cache()
        logger.info(f"[Post-Lezione] Lezione {lecture_id} indicizzata con successo su LanceDB.")
    except InterruptedError:
        unload_llm()
        clear_metal_cache()
        logger.info(f"[Post-Lezione] Generazione capitolo per lezione {lecture_id} interrotta dall'utente.")
    except Exception as exc:
        logger.error(f"[Post-Lezione] Errore elaborazione riassunto/indicizzazione: {exc}")
        # Il capitolo non è stato generato: indicizza almeno la trascrizione per il RAG
        try:
            unload_llm()
            await asyncio.to_thread(
                index_lecture,
                subject_id=subject_id,
                lecture_id=lecture_id,
                title=lecture.title if lecture else "Lezione",
                transcript=full_transcript,
                notes=None,
            )
        except Exception as idx_exc:
            logger.warning(f"[Post-Lezione] Indicizzazione di ripiego non riuscita: {idx_exc}")
        finally:
            unload_embedding_model()
            clear_metal_cache()
    finally:
        db_sess.close()
        clear_metal_cache()


def _microphone_error_message() -> str:
    """Messaggio conciso e specifico per il sistema operativo corrente se il microfono non è accessibile."""
    if sys.platform == "darwin":
        return "Microfono non accessibile. Controlla Impostazioni di Sistema → Privacy e sicurezza → Microfono."
    if sys.platform == "win32":
        return "Microfono non accessibile. Controlla Impostazioni → Privacy → Microfono."
    return "Microfono non accessibile. Verifica i permessi audio del sistema."


# ── Worker di Trascrizione Live Whisper Large Turbo ────────────────────────────
async def _transcription_worker(session: LiveSession, ws: WebSocket):
    """
    Consuma i blocchi vocali accumulati (ogni 20s continuativi o su pausa VAD >500ms)
    con Whisper Large-v3-Turbo per la massima accuratezza accademica e zero allucinazioni.
    Invia lo stato audio al frontend ('listening' / 'transcribing' / 'idle') per il feedback visivo.
    """
    logger.info(f"[WS Worker] Avviato streaming trascrizione Whisper Large Turbo per lezione {session.lecture_id}")
    last_speech_time = time.time()
    cache_released = False
    current_status = "idle"

    async def send_status(new_status: str):
        nonlocal current_status
        if current_status != new_status:
            current_status = new_status
            await _safe_send_json(ws, {
                "type": "status",
                "payload": {"status": new_status}
            })

    # Verifiche diagnostiche proattive (NON segnalare errore microfono se la sessione è intenzionalmente in pausa!)
    if not session.is_paused and session.audio_engine._stream is None and not session.transcriber.mock_mode:
        logger.warning(f"[WS Worker] Microfono hardware non attivo per lezione {session.lecture_id}")
        await _safe_send_json(ws, {
            "type": "error",
            "payload": {
                "message": _microphone_error_message()
            }
        })

    if not session.transcriber.mock_mode:
        if not session.transcriber.local_model_path:
            from transcriber import _resolve_model_path
            session.transcriber.local_model_path = _resolve_model_path(session.transcriber.repo_id)
        if not session.transcriber.local_model_path:
            logger.warning(f"[WS Worker] Modello Whisper non presente in locale per lezione {session.lecture_id}")
            await _safe_send_json(ws, {
                "type": "error",
                "payload": {
                    "message": "Modello Whisper non scaricato. Apri Impostazioni -> Modelli IA e premi 'Scarica' per iniziare a trascrivere."
                }
            })

    try:
        while session.is_active:
            if session.is_paused:
                await asyncio.sleep(0.2)
                continue

            chunk_bytes = None
            try:
                chunk_bytes = session.audio_engine.speech_queue.get_nowait()
            except queue.Empty:
                chunk_bytes = None

            if chunk_bytes is None:
                # Feedback visivo in ascolto o a riposo
                if session.audio_engine.is_voice_active:
                    await send_status("listening")
                else:
                    # Deep Sleep: dopo un silenzio prolungato (>40s) rilascia UNA volta i buffer di calcolo
                    if not cache_released and (time.time() - last_speech_time) > 40.0:
                        session.transcriber.release_metal_cache()
                        cache_released = True
                    await send_status("idle")

                if session.transcriber.mock_mode:
                    await asyncio.sleep(4.0)
                    if not session.is_active or session.is_paused:
                        # Il controllo in testa al ciclo gestisce pausa e stop
                        continue
                    chunk_bytes = b"\x00" * 960
                else:
                    await asyncio.sleep(0.1)
                    continue

            # Abbiamo un blocco vocale da trascrivere con Whisper Large
            last_speech_time = time.time()
            cache_released = False
            await send_status("transcribing")

            try:
                text = await session.transcriber.transcribe(chunk_bytes)
            except Exception as t_err:
                logger.error(f"[WS Worker] Errore trascrizione: {t_err}")
                text = ""

            if text and session.is_active:
                session.push_transcript(text)
                elapsed = session.elapsed_seconds()
                sent = await _safe_send_json(ws, {
                    "type": "transcript_chunk",
                    "payload": {
                        "text": text,
                        "elapsed": elapsed,
                    }
                })
                if not sent:
                    break

                # ── TRIGGER RAG & Rilevamento Domande Docente (SOLO su testo definitivo Whisper Large) ──
                await _announce_questions(session, text, ws)

            if session.is_active and not session.is_paused:
                if session.audio_engine.is_voice_active:
                    await send_status("listening")
                else:
                    await send_status("idle")

    except asyncio.CancelledError:
        logger.info(f"[WS Worker] Worker trascrizione terminato per lezione {session.lecture_id}")
    except Exception as exc:
        logger.error(f"[WS Worker] Errore inatteso nel worker: {exc}")


async def _announce_questions(session: LiveSession, text: str, ws: WebSocket) -> None:
    """Rileva le domande del docente nel testo, le notifica al client senza avvio automatico LLM."""
    for q in extract_professor_questions(text):
        q_obj = session.add_question(
            q,
            confidence=0.95,
            suggested_answer="",
        )
        logger.info(f"TRIGGER_RAG: [Domanda del prof rilevata] {q}")
        await _safe_send_json(ws, {
            "type": "question_detected",
            "payload": {
                "id": q_obj["id"],
                "question": q_obj["question"],
                "confidence": q_obj["confidence"],
                "suggested_answer": "",
                "trigger": "TRIGGER_RAG",
            }
        })


async def _post_lecture_pipeline(
    subject_id: int,
    lecture_id: int,
    audio_path: Optional[str],
    full_transcript: str,
) -> None:
    """Pipeline asincrona post-lezione eseguita in sequenza ordinata per evitare race condition."""
    from llm_engine import clear_metal_cache

    # 1. Calcola i timestamp sincronizzati mentre Whisper è ancora in memoria
    # Se il file audio esiste, forziamo il calcolo Whisper reale (force=True) per sostituire
    # l'eventuale stima temporanea live e recuperare frasi perse se il microfono era a basso volume.
    if audio_path and os.path.exists(audio_path):
        try:
            from routers.lectures import compute_and_cache_lecture_timestamps
            await asyncio.to_thread(compute_and_cache_lecture_timestamps, lecture_id, force=True)
        except Exception as exc:
            logger.warning(f"[WS Pipeline] Errore calcolo timestamps lezione {lecture_id}: {exc}")

    # Rilascio cooperativo della cache Metal prima della compressione
    clear_metal_cache()
    await asyncio.sleep(0.3)

    # 2. Converte la traccia temporanea in formato definitivo M4A salva-spazio ed aggiorna DB
    if audio_path and audio_path.endswith(".wav") and os.path.exists(audio_path):
        try:
            from audio_engine import compress_wav_to_m4a
            m4a_path = await asyncio.to_thread(compress_wav_to_m4a, audio_path, 48000, True)
            if m4a_path and m4a_path != audio_path and os.path.exists(m4a_path):
                with SessionLocal() as db_sess:
                    lec = db_sess.query(Lecture).filter(Lecture.id == lecture_id).first()
                    if lec:
                        lec.audio_path = m4a_path
                        db_sess.commit()
                # Rinomina la cache timestamps dal wav al m4a se esiste
                old_ts = audio_path + ".timestamps.json"
                new_ts = m4a_path + ".timestamps.json"
                if os.path.exists(old_ts) and not os.path.exists(new_ts):
                    try:
                        os.replace(old_ts, new_ts)
                    except Exception:
                        pass
                logger.info(f"[WS Pipeline] Audio lezione {lecture_id} compresso M4A salvato nel DB: {m4a_path}")
        except Exception as exc:
            logger.warning(f"[WS Pipeline] Compressione M4A non riuscita per lezione {lecture_id}: {exc}")

    # Rilascio cooperativo prima del caricamento e inferenza dell'LLM
    clear_metal_cache()
    await asyncio.sleep(0.3)

    # 3. Ora che Whisper ha terminato e i file sono pronti, genera capitoli con LLM + RAG LanceDB
    # Rileggiamo il transcript aggiornato dal DB se Whisper su audio ha recuperato più parole del live
    effective_transcript = full_transcript
    try:
        with SessionLocal() as db_sess:
            lec = db_sess.query(Lecture).filter(Lecture.id == lecture_id).first()
            if lec and lec.raw_transcript and len(lec.raw_transcript.strip()) > len((effective_transcript or "").strip()):
                effective_transcript = lec.raw_transcript
    except Exception:
        pass

    cleaned_words = [w for w in (effective_transcript or "").split() if w.strip()]
    if effective_transcript and len(effective_transcript.strip()) >= 40 and len(cleaned_words) >= 6:
        try:
            await _post_lecture_summary_and_index(
                subject_id=subject_id,
                lecture_id=lecture_id,
                full_transcript=effective_transcript,
            )
        except Exception as exc:
            logger.warning(f"[WS Pipeline] Errore riassunto/RAG per lezione {lecture_id}: {exc}")


DISCONNECT_GRACE_PERIOD_SECONDS: float = 600.0  # 10 minuti di timeout per riconnessione client


async def _disconnect_grace_finalizer(sess: LiveSession, sid: int, timeout: float = DISCONNECT_GRACE_PERIOD_SECONDS):
    """Finalizza la sessione solo se nessun client si riconnette entro il timeout di grazia."""
    try:
        await asyncio.sleep(timeout)
        if sess.is_active and not sess.finalized:
            logger.info(
                f"[WS] Timeout di grazia ({timeout}s) scaduto per lezione {sess.lecture_id} "
                f"(materia {sid}) senza riconnessione: finalizzazione sessione..."
            )
            await _finalize_session(sess, sid)
    except asyncio.CancelledError:
        logger.info(f"[WS] Timer di grazia annullato per lezione {sess.lecture_id} (riconnessione avvenuta)")


async def _finalize_session(session: LiveSession, subject_id: int) -> Tuple[float, Optional[str]]:
    """
    Chiude una sessione live in modo idempotente e reattivo:
    1. Ferma subito l'audio engine ed esporta il file audio su disco (~5ms).
    2. Salva IMMEDIATAMENTE trascrizione, durata e audio_path su SQLite per rendere subito disponibile il player.
    3. Rimuove la sessione dal registry attivo e sblocca la risposta WebSocket.
    4. Sposta in background non bloccante la trascrizione dell'ultimo chunk, il rilascio della memoria e la compressione M4A.
    """
    if session.finalized:
        return session.elapsed_seconds(), None
    session.finalized = True

    grace_task = getattr(session, "_disconnect_grace_task", None)
    if grace_task and grace_task is not asyncio.current_task() and not grace_task.done():
        grace_task.cancel()
    session._disconnect_grace_task = None

    duration = session.elapsed_seconds()
    session.is_active = False

    task = session.transcription_task
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait({task}, timeout=1.0)
        except Exception:
            pass

    audio_path: Optional[str] = None
    try:
        if session.audio_engine.is_running:
            # stop(audio_format="wav") è istantaneo (scrittura PCM su file senza codec blocking)
            audio_path = await asyncio.wait_for(
                asyncio.to_thread(session.audio_engine.stop, audio_format="wav"),
                timeout=6.0,
            )
    except Exception as exc:
        logger.error(f"[WS] Errore arresto AudioEngine: {type(exc).__name__}: {exc}")

    # ── Salvataggio IMMEDIATO nel Database (priorità assoluta per il player) ──
    full_transcript = session.get_full_transcript()
    db_sess = SessionLocal()
    try:
        lecture = db_sess.query(Lecture).filter(Lecture.id == session.lecture_id).first()
        if lecture:
            lecture.raw_transcript = full_transcript
            lecture.duration_seconds = duration
            if audio_path:
                lecture.audio_path = audio_path
            lecture.detected_questions = (
                json.dumps(session.detected_questions, ensure_ascii=False)
                if session.detected_questions
                else None
            )
            db_sess.commit()
            logger.info(f"[WS] Salvataggio immediato SQLite completato: lezione={session.lecture_id}, audio={audio_path}")

            # Genera immediatamente la cache timestamps sincronizzata basata sui blocchi registrati in live
            if audio_path and hasattr(session, "_buffer") and session._buffer:
                try:
                    from routers.lectures import _build_live_buffer_timestamps
                    init_lines = _build_live_buffer_timestamps(list(session._buffer), duration)
                    if init_lines:
                        cache_ts_path = audio_path + ".timestamps.json"
                        with open(cache_ts_path, "w", encoding="utf-8") as f_ts:
                            json.dump(init_lines, f_ts, ensure_ascii=False)
                        logger.info(f"[WS] Cache timestamps live generata da ring-buffer: {cache_ts_path}")
                except Exception as ts_err:
                    logger.debug(f"[WS] Generazione cache live non riuscita: {ts_err}")
    except Exception as db_err:
        logger.error(f"[WS] Errore salvataggio immediato lezione su DB: {db_err}")
    finally:
        db_sess.close()

    # Rimuovi dal registry sessioni attive
    try:
        from live_session import _active_sessions
        if _active_sessions.get(subject_id) is session:
            _active_sessions.pop(subject_id, None)
    except Exception:
        pass

    # ── Task in background: coda residua, rilascio memoria ed elaborazione post-lezione ──
    async def _background_post_stop():
        # Pausa di differimento cooperativo (1.2s): consente alla UI WebKit / Tauri di completare
        # lo smontaggio del workspace e il ritorno fluido a 60 FPS al Quaderno senza contesa GPU/CPU.
        await asyncio.sleep(1.2)

        # Trascrive eventuali frammenti residui rimasti in coda (non scarta il parlato su CPU più lente)
        if not session.transcriber.mock_mode:
            drained_any = False
            while not session.audio_engine.speech_queue.empty():
                try:
                    chunk = session.audio_engine.speech_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    tail = await asyncio.wait_for(session.transcriber.transcribe(chunk), timeout=25.0)
                    if tail:
                        session.push_transcript(tail)
                        drained_any = True
                except Exception as exc:
                    logger.debug(f"[WS] Coda finale non completata: {exc}")
                    break

            if drained_any:
                try:
                    with SessionLocal() as s_db:
                        lec = s_db.query(Lecture).filter(Lecture.id == session.lecture_id).first()
                        if lec:
                            new_text = session.get_full_transcript().strip()
                            existing_text = (lec.raw_transcript or "").strip()
                            if new_text and (not existing_text or len(new_text) >= len(existing_text)):
                                lec.raw_transcript = new_text
                                s_db.commit()
                except Exception as exc:
                    logger.warning(f"[WS] Errore aggiornamento trascrizione finale DB: {exc}")

        # Pipeline post-lezione (calcolo timestamps con Whisper ancora caldo, compressione M4A, capitoli/RAG)
        try:
            await _post_lecture_pipeline(
                subject_id=subject_id,
                lecture_id=session.lecture_id,
                audio_path=audio_path,
                full_transcript=session.get_full_transcript(),
            )
        finally:
            if not has_active_sessions() and hasattr(session, "transcriber") and session.transcriber:
                try:
                    await asyncio.to_thread(session.transcriber.release_memory)
                except Exception as exc:
                    logger.debug(f"[WS] Errore pulizia sessione e rilascio memoria: {exc}")

    _spawn(_background_post_stop())

    logger.info(
        f"[WS] Sessione chiusa con successo: lezione={session.lecture_id}, durata={duration:.1f}s, "
        f"chunk totali={len(session.full_transcript)}, wav={audio_path}"
    )
    return duration, audio_path


# ── WebSocket: Live Lecture ────────────────────────────────────────────────────

# Origini ammesse per il WebSocket (il CORS non si applica ai WebSocket):
# dev server Vite, Tauri macOS/Linux (tauri://localhost) e Tauri Windows (http(s)://tauri.localhost)
_ALLOWED_WS_ORIGIN = re.compile(
    r"^(null|tauri://localhost|https?://tauri\.localhost(:\d+)?/?|https?://(localhost|127\.0\.0\.1)(:\d+)?/?)$"
)


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


@app.websocket("/ws/live-lecture/{subject_id}")
async def live_lecture_ws(
    websocket: WebSocket,
    subject_id: int,
):
    """
    WebSocket per la gestione di una lezione live.

    Protocollo messaggi (JSON):
    CLIENT → SERVER:
      { "type": "start",    "payload": { "title": "...", "lecture_id": null | int } }
      { "type": "chunk",    "payload": { "text": "...", "timestamp": float } }
      { "type": "bookmark", "payload": { "note": "..." } }
      { "type": "chat",     "payload": { "query": "...", "context_minutes": 5 } }
      { "type": "stop",     "payload": {} }

    SERVER → CLIENT:
      { "type": "started",          "payload": { "lecture_id": int, "session_started_at": float } }
      { "type": "transcript_chunk", "payload": { "text": "...", "elapsed": float } }
      { "type": "bookmark_saved",   "payload": { "bookmark_id": int, "timestamp": float } }
      { "type": "chat_response",    "payload": { "answer": "...", "context_used": str } }
      { "type": "question_detected","payload": { "question": "...", "confidence": float } }
      { "type": "stopped",          "payload": { "duration": float, "total_chunks": int, "audio_path": str } }
      { "type": "error",            "payload": { "message": "..." } }
    """
    origin = websocket.headers.get("origin")
    if origin and origin != "null" and not _ALLOWED_WS_ORIGIN.match(origin):
        logger.warning(f"[WS] Connessione rifiutata da origine non ammessa: {origin}")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info(f"[WS] Connessione live aperta per materia {subject_id}")

    async def send_error(message: str) -> None:
        await _safe_send_json(websocket, {"type": "error", "payload": {"message": message}})

    session: LiveSession | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_error("JSON non valido")
                continue
            if not isinstance(msg, dict):
                await send_error("Messaggio non valido: atteso un oggetto JSON")
                continue

            msg_type = msg.get("type", "")
            payload = msg.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            # ── START / RECONNECT ──────────────────────────────────────────────
            if msg_type in ("start", "reconnect"):
                title = str(payload.get("title") or "Nuova Lezione")[:255]
                raw_lecture_id = payload.get("lecture_id")
                req_lecture_id: Optional[int] = None
                if raw_lecture_id is not None:
                    try:
                        req_lecture_id = int(raw_lecture_id)
                    except (TypeError, ValueError):
                        await send_error("lecture_id non valido")
                        continue

                # Controlla se c'è già una sessione attiva per subject_id
                stale = get_session(subject_id)
                if (
                    stale is not None
                    and stale.is_active
                    and not stale.finalized
                    and (
                        (req_lecture_id is not None and stale.lecture_id == req_lecture_id)
                        or (msg_type == "reconnect" and req_lecture_id is None)
                    )
                ):
                    # RICONNETTI!
                    logger.info(
                        f"[WS] Riconnessione su sessione attiva per materia {subject_id}, "
                        f"lezione {stale.lecture_id} (msg_type={msg_type})"
                    )
                    # Annulla l'eventuale timer di grazia di disconnessione
                    grace_task = getattr(stale, "_disconnect_grace_task", None)
                    if grace_task and not grace_task.done():
                        grace_task.cancel()
                    stale._disconnect_grace_task = None
                    stale._client_disconnected_at = None

                    session = stale

                    # Associa il worker di trascrizione al nuovo websocket
                    if session.transcription_task and not session.transcription_task.done():
                        session.transcription_task.cancel()
                    session.transcription_task = asyncio.create_task(
                        _transcription_worker(session, websocket)
                    )

                    await _safe_send_json(websocket, {
                        "type": "reconnected",
                        "payload": {
                            "lecture_id": session.lecture_id,
                            "session_started_at": session.started_at,
                            "is_paused": session.is_paused,
                            "elapsed": session.elapsed_seconds(),
                            "chunks": session.full_transcript,
                            "detected_questions": session.detected_questions,
                        }
                    })
                    continue

                if msg_type == "reconnect":
                    await send_error("Nessuna sessione attiva trovata per la riconnessione.")
                    continue

                # Per "start" con lezione diversa o nuova: finalizza eventuali sessioni precedenti
                if session is not None and session is not stale:
                    await _finalize_session(session, subject_id)
                    session = None
                if stale is not None:
                    logger.warning(
                        f"[WS] Sessione precedente ancora attiva per materia {subject_id} (lezione {stale.lecture_id}): la salvo e la chiudo."
                    )
                    grace_task = getattr(stale, "_disconnect_grace_task", None)
                    if grace_task and not grace_task.done():
                        grace_task.cancel()
                    await _finalize_session(stale, subject_id)

                lecture_id = req_lecture_id
                from database import Lecture as LectureModel, Subject
                with SessionLocal() as db:
                    subject = db.query(Subject).filter(Subject.id == subject_id).first()
                    if not subject:
                        await send_error(f"Materia {subject_id} non trovata")
                        continue
                    whisper_keywords = subject.whisper_prompt_keywords

                    if lecture_id is not None:
                        owned = (
                            db.query(LectureModel)
                            .filter(LectureModel.id == lecture_id, LectureModel.subject_id == subject_id)
                            .first()
                        )
                        if not owned:
                            await send_error(f"Lezione {lecture_id} non trovata per la materia {subject_id}")
                            continue

                    if lecture_id is None:
                        # Crea una nuova Lecture nel DB se non fornita
                        lecture = LectureModel(subject_id=subject_id, title=title)
                        db.add(lecture)
                        db.commit()
                        db.refresh(lecture)
                        lecture_id = lecture.id

                # Leggi configurazione modello Whisper e sensibilità VAD
                model = payload.get("model")
                vad_sensitivity = payload.get("vad_sensitivity")
                if model is None or vad_sensitivity is None:
                    user_settings = load_settings()
                    if model is None:
                        model = user_settings.get("whisper_model", "mock_mode")
                    if vad_sensitivity is None:
                        vad_sensitivity = user_settings.get("vad_sensitivity", 2)
                vad_level = _as_int(vad_sensitivity, default=2, lo=0, hi=3)

                mock_transcriber = (model == "mock_mode")

                # Avvia sessione in thread pool per non bloccare l'event loop asyncio
                session = await asyncio.to_thread(
                    start_session,
                    subject_id=subject_id,
                    lecture_id=lecture_id,
                    whisper_keywords=whisper_keywords,
                    mock_transcriber=mock_transcriber,
                    model_name=model,
                    vad_sensitivity=vad_level,
                )
                logger.info(
                    f"[WS] Sessione e microfono avviati: materia={subject_id}, lezione={lecture_id}, "
                    f"modello={model}, VAD={vad_level}"
                )

                await _safe_send_json(websocket, {
                    "type": "started",
                    "payload": {
                        "lecture_id": lecture_id,
                        "session_started_at": session.started_at,
                    }
                })

                # Lancia il task in background per trasmettere i chunk trascritti
                session.transcription_task = asyncio.create_task(
                    _transcription_worker(session, websocket)
                )

            # ── CHUNK (ricezione manuale se inviata dal client) ────────────────
            elif msg_type == "chunk":
                if not session:
                    await send_error("Sessione non avviata. Invia prima 'start'.")
                    continue
                text = str(payload.get("text") or "")
                if text:
                    session.push_chunk(text)
                    await _safe_send_json(websocket, {
                        "type": "transcript_chunk",
                        "payload": {
                            "text": text,
                            "elapsed": session.elapsed_seconds(),
                        }
                    })

            # ── BOOKMARK ───────────────────────────────────────────────────────
            elif msg_type == "bookmark":
                if not session:
                    continue
                from database import Bookmark
                note = str(payload.get("note") or "")
                ts = session.elapsed_seconds()
                bm_id = None
                with SessionLocal() as db:
                    bookmark = Bookmark(
                        lecture_id=session.lecture_id,
                        timestamp_seconds=ts,
                        note=note,
                    )
                    db.add(bookmark)
                    db.commit()
                    db.refresh(bookmark)
                    bm_id = bookmark.id
                logger.info(f"[WS] Segnalibro salvato @ {ts:.1f}s: '{note}'")
                await _safe_send_json(websocket, {
                    "type": "bookmark_saved",
                    "payload": {"bookmark_id": bm_id, "timestamp": ts, "note": note}
                })

            # ── CHAT (domanda all'assistente live con streaming RAG) ───────────
            elif msg_type == "chat":
                if not session:
                    await send_error("Sessione non avviata: impossibile rispondere alla chat.")
                    continue
                query = str(payload.get("query") or "").strip()
                if not query:
                    continue
                context_minutes = _as_int(payload.get("context_minutes", 15), default=15, lo=1, hi=15)
                enable_thinking = bool(payload.get("enable_thinking", False))
                raw_history = payload.get("history")
                history = raw_history if isinstance(raw_history, list) else None
                _spawn(
                    _handle_live_chat_stream(
                        session,
                        websocket,
                        query,
                        context_minutes,
                        subject_id,
                        enable_thinking=enable_thinking,
                        history=history,
                    )
                )

            # ── DELETE QUESTION (elimina domanda non pertinente in live) ───────
            elif msg_type == "delete_question":
                if session:
                    q_id = payload.get("question_id") or msg.get("question_id")
                    if q_id:
                        removed = session.delete_question(str(q_id))
                        logger.info(f"[WS] Domanda rilevata {q_id} rimossa dalla sessione live (success={removed})")

            # ── ANSWER QUESTION (richiesta esplicita utente risposta domanda) ──
            elif msg_type == "answer_question":
                q_id = str(payload.get("question_id") or "").strip()
                q_text = str(payload.get("question") or "").strip()
                if session:
                    if not q_text and q_id:
                        for item in session.detected_questions:
                            if str(item.get("id")) == q_id:
                                q_text = str(item.get("question") or "").strip()
                                break
                    if q_text and q_id:
                        logger.info(f"[WS] Richiesta esplicita utente risposta domanda: '{q_text[:30]}...'")
                        await _safe_send_json(websocket, {
                            "type": "question_answering",
                            "payload": {"id": q_id, "question": q_text}
                        })
                        _spawn(_generate_flash_answer_task(session, q_text, q_id, websocket))
                    elif q_id:
                        logger.warning(f"[WS] Domanda {q_id} non trovata nella sessione corrente.")
                        await _safe_send_json(websocket, {
                            "type": "error",
                            "payload": {
                                "id": q_id,
                                "message": "Domanda non trovata nella sessione corrente.",
                            }
                        })
                elif q_id:
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "payload": {
                            "id": q_id,
                            "message": "Domanda non trovata nella sessione corrente.",
                        }
                    })

            # ── PAUSE (Consolidamento non bloccante + Stop motori + Deep sleep) ─
            elif msg_type == "pause":
                if session:
                    if not session.is_paused:
                        logger.info(f"[WS] Richiesta PAUSA per lezione {session.lecture_id}: consolidamento non bloccante...")
                        # 1. Chiude microfono ed estrae l'ultimo chunk vocale accumulato (anche se breve)
                        last_audio = session.pause()

                        # Invia SUBITO paused prima di trascrivere l'ultimo audio!
                        await _safe_send_json(websocket, {"type": "paused"})

                        # 2. Consolidamento asincrono in background via _spawn (NON blocca il loop WS)
                        async def _process_pause_last_audio(audio_chunk: bytes, current_session: LiveSession, ws: WebSocket):
                            try:
                                if audio_chunk and len(audio_chunk) >= 1600:
                                    refined_text = await current_session.transcriber.transcribe(audio_chunk)
                                    if refined_text:
                                        current_session.push_transcript(refined_text)
                                        await _safe_send_json(ws, {
                                            "type": "transcript_chunk",
                                            "payload": {
                                                "text": refined_text,
                                                "elapsed": current_session.elapsed_seconds(),
                                            }
                                        })
                                        # Rilevamento domande sull'ultimo parlato prima della pausa
                                        await _announce_questions(current_session, refined_text, ws)
                            except Exception as p_err:
                                logger.error(f"[WS Pause] Errore trascrizione ultimo pezzo: {p_err}")
                            finally:
                                try:
                                    current_session.transcriber.release_metal_cache()
                                except Exception:
                                    pass

                        if last_audio and len(last_audio) >= 1600:
                            _spawn(_process_pause_last_audio(last_audio, session, websocket))
                        else:
                            try:
                                session.transcriber.release_metal_cache()
                            except Exception:
                                pass
                    else:
                        # Se session.is_paused è già True, invia comunque {"type": "paused"} come ACK
                        await _safe_send_json(websocket, {"type": "paused"})

            # ── RESUME (Ripresa registrazione da pausa) ────────────────────────
            elif msg_type == "resume":
                if session:
                    if session.is_paused:
                        session.resume()
                        logger.info(f"[WS] Registrazione RIPRESA per lezione {session.lecture_id}: microfono e modelli riattivati.")
                    await _safe_send_json(websocket, {"type": "resumed"})

            # ── UPDATE_VAD (Aggiornamento a caldo della sensibilità microfono) ─
            elif msg_type == "update_vad":
                if session and session.is_active:
                    new_vad = int(payload.get("vad_sensitivity", 2))
                    session.update_vad_sensitivity(new_vad)
                    logger.info(f"[WS] Sensibilità VAD aggiornata a caldo per lezione {session.lecture_id}: {new_vad}")
                    await _safe_send_json(websocket, {
                        "type": "vad_updated",
                        "payload": {"vad_sensitivity": new_vad}
                    })
                continue

            # ── STOP ───────────────────────────────────────────────────────────
            elif msg_type == "stop":
                if not session:
                    continue
                grace_task = getattr(session, "_disconnect_grace_task", None)
                if grace_task and not grace_task.done():
                    grace_task.cancel()
                duration, audio_path = await asyncio.shield(_finalize_session(session, subject_id))
                total_chunks = len(session.full_transcript)
                session = None
                await _safe_send_json(websocket, {
                    "type": "stopped",
                    "payload": {
                        "duration": duration,
                        "total_chunks": total_chunks,
                        "audio_path": audio_path,
                    }
                })
                break

            else:
                await send_error(f"Tipo messaggio sconosciuto: '{msg_type}'")

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnesso (materia {subject_id})")
    except Exception as exc:
        logger.exception(f"[WS] Errore inatteso: {exc}")
        try:
            await send_error(str(exc))
        except Exception:
            pass
    finally:
        _ws_locks.pop(websocket, None)
        # Se la disconnessione avviene senza 'stop' esplicito, metti in pausa e avvia timer di grazia
        if session is not None and not session.finalized and session.is_active:
            last_audio: Optional[bytes] = None
            try:
                last_audio = session.pause()
            except Exception as e:
                logger.warning(f"[WS] Errore pausa sessione disconnessa: {e}")
            if last_audio and len(last_audio) >= 1600:
                try:
                    session.audio_engine.speech_queue.put_nowait(last_audio)
                    logger.info(
                        f"[WS] Audio residuo ({len(last_audio)} B) preservato in speech_queue "
                        f"su disconnessione per lezione {session.lecture_id}."
                    )
                except Exception as q_err:
                    logger.warning(f"[WS] Impossibile preservare audio residuo in coda: {q_err}")
            session._client_disconnected_at = time.time()
            old_grace = getattr(session, "_disconnect_grace_task", None)
            if old_grace and not old_grace.done():
                old_grace.cancel()
            session._disconnect_grace_task = _spawn(
                _disconnect_grace_finalizer(session, subject_id, DISCONNECT_GRACE_PERIOD_SECONDS)
            )
            logger.info(
                f"[WS] Client disconnesso per lezione {session.lecture_id}: "
                f"sessione in pausa, timer di grazia ({DISCONNECT_GRACE_PERIOD_SECONDS}s) avviato."
            )


# ── REST: Generazione Capitolo Quaderno su richiesta ──────────────────────────
from pydantic import BaseModel

class GenerateSummaryRequest(BaseModel):
    model_id: Optional[str] = None


@app.post("/api/v1/lectures/{lecture_id}/generate-summary", tags=["Lectures"])
async def generate_summary_for_lecture(
    lecture_id: int,
    payload: Optional[GenerateSummaryRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Genera o rigenera il capitolo completo per il Quaderno di studio.
    Scarica preventivamente Whisper per lasciare tutta la memoria unificata (8GB) a Qwen 3.5 4B.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    has_raw = bool(lecture.raw_transcript and lecture.raw_transcript.strip())
    transcript = lecture.raw_transcript if has_raw else (lecture.processed_notes or "")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Nessun testo disponibile per generare il capitolo.")

    from rag import query_context, index_lecture, unload_embedding_model
    from llm_engine import (
        cancel_epoch,
        clear_metal_cache,
        generate_lecture_chapter,
        is_cancelled,
        is_model_downloaded,
        resolve_llm_model,
        unload_llm,
        unload_whisper,
    )
    from config import load_settings
    from database import Bookmark

    settings = load_settings()
    llm_model = (payload.model_id if payload else None) or settings.get("llm_model")
    if not is_model_downloaded(resolve_llm_model(llm_model)):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non scaricato. Apri Impostazioni → Modelli IA e scarica un modello LLM per generare la sintesi.",
        )

    start_epoch = cancel_epoch()

    # Unload coordinato di Whisper prima del riassunto (non durante una lezione live)
    if not has_active_sessions():
        await asyncio.to_thread(unload_whisper)

    bms = db.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
    bm_summary = (
        "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bms])
        if bms else "Nessun segnalibro specifico inserito."
    )

    from transcript_cleaner import clean_transcript_for_llm
    cleaned_for_query = clean_transcript_for_llm(transcript) if has_raw else transcript
    try:
        past_refs = await asyncio.to_thread(
            query_context,
            subject_id=lecture.subject_id,
            query=cleaned_for_query[:1500],
            top_k=3,
            priority_previous_lecture=True,
        )
    finally:
        # Scarica embedding prima di allocare l'LLM pesante per non superare gli 8GB di memoria
        unload_embedding_model()
        clear_metal_cache()

    try:
        if is_cancelled(start_epoch):
            raise InterruptedError("Generazione interrotta dall'utente.")
        chapter = await generate_lecture_chapter(
            title=lecture.title,
            transcript=transcript,
            past_references=past_refs,
            bookmarks_summary=bm_summary,
            model_id=llm_model,
            lecture_id=lecture.id,
            clean_transcript=has_raw,
        )
    except InterruptedError:
        unload_llm()
        clear_metal_cache()
        logger.info(f"[Lectures] Generazione sintesi per lezione {lecture_id} interrotta. Nessun salvataggio.")
        return {
            "status": "cancelled",
            "lecture_id": lecture_id,
            "processed_notes": lecture.processed_notes or "",
            "message": "Generazione annullata dall'utente. Nessuna modifica apportata.",
        }
    except Exception as exc:
        # Le note esistenti NON vengono sovrascritte con un messaggio d'errore
        unload_llm()
        clear_metal_cache()
        logger.error(f"[Lectures] Generazione sintesi per lezione {lecture_id} fallita: {exc}")
        raise HTTPException(status_code=500, detail=f"Generazione del capitolo non riuscita: {exc}")

    if not chapter:
        unload_llm()
        clear_metal_cache()
        raise HTTPException(status_code=500, detail="Il modello non ha prodotto alcun capitolo.")

    lecture.processed_notes = chapter
    db.commit()

    # Rilascia l'LLM dalla RAM prima di passare all'indicizzazione con embedding
    unload_llm()
    clear_metal_cache()

    # Indicizzazione automatica in LanceDB solo se completata con successo
    try:
        await asyncio.to_thread(
            index_lecture,
            subject_id=lecture.subject_id,
            lecture_id=lecture.id,
            title=lecture.title,
            transcript=transcript,
            notes=chapter,
        )
    finally:
        unload_embedding_model()
        clear_metal_cache()

    return {
        "status": "success",
        "lecture_id": lecture_id,
        "processed_notes": chapter,
    }


@app.post("/api/v1/lectures/{lecture_id}/generate-summary-stream", tags=["Lectures"])
async def generate_summary_stream_for_lecture(
    lecture_id: int,
    payload: Optional[GenerateSummaryRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Genera il capitolo completo per il Quaderno di studio inviando eventi di progresso in streaming NDJSON.
    Mantiene aperta la connessione TCP inviando battiti di keep-alive per evitare timeout o disconnessioni socket su Windows/macOS.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    has_raw = bool(lecture.raw_transcript and lecture.raw_transcript.strip())
    transcript = lecture.raw_transcript if has_raw else (lecture.processed_notes or "")
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Nessun testo disponibile per generare il capitolo.")

    from rag import query_context, index_lecture, unload_embedding_model
    from llm_engine import (
        cancel_epoch,
        clear_metal_cache,
        generate_lecture_chapter,
        is_cancelled,
        is_model_downloaded,
        resolve_llm_model,
        unload_llm,
        unload_whisper,
    )
    from config import load_settings
    from database import Bookmark

    settings = load_settings()
    llm_model = (payload.model_id if payload else None) or settings.get("llm_model")
    if not is_model_downloaded(resolve_llm_model(llm_model)):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non scaricato. Apri Impostazioni → Modelli IA e scarica un modello LLM per generare la sintesi.",
        )

    start_epoch = cancel_epoch()
    if not has_active_sessions():
        await asyncio.to_thread(unload_whisper)

    bms = db.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
    bm_summary = (
        "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bms])
        if bms else "Nessun segnalibro specifico inserito."
    )

    from transcript_cleaner import clean_transcript_for_llm
    cleaned_for_query = clean_transcript_for_llm(transcript) if has_raw else transcript
    try:
        past_refs = await asyncio.to_thread(
            query_context,
            subject_id=lecture.subject_id,
            query=cleaned_for_query[:1500],
            top_k=3,
            priority_previous_lecture=True,
        )
    finally:
        unload_embedding_model()
        clear_metal_cache()

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _on_progress(event_dict: Dict[str, Any]) -> None:
        try:
            line = json.dumps(event_dict, ensure_ascii=False) + "\n"
            loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception:
            pass

    async def _worker():
        try:
            if is_cancelled(start_epoch):
                raise InterruptedError("Generazione interrotta dall'utente.")
            chapter = await generate_lecture_chapter(
                title=lecture.title,
                transcript=transcript,
                past_references=past_refs,
                bookmarks_summary=bm_summary,
                model_id=llm_model,
                lecture_id=lecture.id,
                clean_transcript=has_raw,
                progress_callback=_on_progress,
            )
            if not chapter:
                unload_llm()
                clear_metal_cache()
                _on_progress({"stage": "error", "message": "Il modello non ha prodotto alcun capitolo."})
                return

            lecture.processed_notes = chapter
            db.commit()

            unload_llm()
            clear_metal_cache()

            try:
                await asyncio.to_thread(
                    index_lecture,
                    subject_id=lecture.subject_id,
                    lecture_id=lecture.id,
                    title=lecture.title,
                    transcript=transcript,
                    notes=chapter,
                )
            finally:
                unload_embedding_model()
                clear_metal_cache()

            _on_progress({
                "stage": "done",
                "status": "success",
                "lecture_id": lecture_id,
                "processed_notes": chapter,
                "progress": 100,
            })
        except InterruptedError:
            unload_llm()
            clear_metal_cache()
            _on_progress({
                "stage": "cancelled",
                "status": "cancelled",
                "lecture_id": lecture_id,
                "processed_notes": lecture.processed_notes or "",
                "message": "Generazione annullata dall'utente.",
            })
        except Exception as exc:
            unload_llm()
            clear_metal_cache()
            logger.error(f"[Lectures Stream] Errore capitolo lezione {lecture_id}: {exc}")
            _on_progress({
                "stage": "error",
                "status": "error",
                "message": f"Generazione non riuscita: {exc}",
            })
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception:
                pass

    worker_task = asyncio.create_task(_worker())

    async def _event_generator():
        try:
            while True:
                try:
                    # Timeout a 2.0s per inviare battito keep-alive anti-drop su Windows
                    item = await asyncio.wait_for(queue.get(), timeout=2.0)
                    if item is None:
                        break
                    yield item.encode("utf-8")
                except asyncio.TimeoutError:
                    # Commento keep-alive NDJSON standard
                    yield b": keep-alive\n"
        except asyncio.CancelledError:
            worker_task.cancel()
            raise

    return StreamingResponse(
        _event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/lectures/{lecture_id}/questions/{question_id}/answer", tags=["Lectures"])
async def answer_lecture_question_endpoint(
    lecture_id: int,
    question_id: str,
    db: Session = Depends(get_db),
):
    """
    Risponde specificamente a una domanda posta a lezione dal docente,
    utilizzando il contesto della lezione (quaderno e trascrizione) e l'LLM locale.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    from llm_engine import generate_flash_answer, is_model_downloaded, DEFAULT_LLM_MODEL, clear_metal_cache
    from config import load_settings

    settings = load_settings()
    llm_model = settings.get("llm_model") or DEFAULT_LLM_MODEL

    if not is_model_downloaded(llm_model):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non ancora scaricato. Apri le Impostazioni per scaricarlo."
        )

    questions_list = []
    if lecture.detected_questions:
        try:
            questions_list = json.loads(lecture.detected_questions)
        except Exception:
            questions_list = []

    if not isinstance(questions_list, list):
        questions_list = []

    target_q = None
    target_idx = -1
    for i, q in enumerate(questions_list):
        if isinstance(q, dict) and str(q.get("id")) == str(question_id):
            target_q, target_idx = q, i
            break
    if target_q is None and str(question_id).isdigit():
        i = int(question_id)
        if 0 <= i < len(questions_list) and isinstance(questions_list[i], dict):
            target_q, target_idx = questions_list[i], i

    if not target_q:
        raise HTTPException(status_code=404, detail="Domanda non trovata.")

    from transcript_cleaner import clean_transcript_for_llm
    q_text = target_q.get("question", "")
    raw_fallback = clean_transcript_for_llm(lecture.raw_transcript) if lecture.raw_transcript else ""
    context = (lecture.processed_notes or raw_fallback or lecture.title or "")[:12000]

    answer = await generate_flash_answer(question=q_text, context=context, model_id=llm_model)

    target_q["suggestedAnswer"] = answer
    questions_list[target_idx] = target_q
    lecture.detected_questions = json.dumps(questions_list, ensure_ascii=False)
    db.commit()
    db.refresh(lecture)
    clear_metal_cache()

    return {
        "status": "success",
        "question_id": question_id,
        "question": target_q,
        "answer": answer,
    }


def _prepare_course_context(subject_id: int, query: str, db: Session):
    """
    Raccoglie il contesto RAG e il registro delle domande reali poste dal docente
    durante le lezioni registrate, sintetizzando fonti leggibili per il badge studente.
    """
    import re
    from rag import query_context, unload_embedding_model
    from llm_engine import clear_metal_cache

    rag_context = query_context(subject_id=subject_id, query=query, top_k=4)
    unload_embedding_model()
    clear_metal_cache()

    # Domande rilevate a lezione dal docente (Whisper Question Detector)
    lectures = db.query(Lecture.id, Lecture.title, Lecture.detected_questions).filter(Lecture.subject_id == subject_id).all()
    all_detected_questions = []
    for lec_id, lec_title, lec_dq in lectures:
        if lec_dq:
            try:
                q_list = json.loads(lec_dq)
                if isinstance(q_list, list):
                    for q_obj in q_list:
                        q_text = q_obj.get("question") if isinstance(q_obj, dict) else str(q_obj)
                        if q_text:
                            all_detected_questions.append({
                                "lecture_title": lec_title or f"Lezione #{lec_id}",
                                "question": q_text,
                                "suggested_answer": str(q_obj.get("suggestedAnswer") or "") if isinstance(q_obj, dict) else ""
                            })
            except Exception:
                pass

    questions_block = ""
    if all_detected_questions:
        lines = ["\n--- REGISTRO DOMANDE RILEVATE A LEZIONE DAL DOCENTE (Whisper Detector) ---"]
        for item in all_detected_questions:
            lines.append(f"• In '{item['lecture_title']}': \"{item['question']}\"")
            ans = str(item.get("suggested_answer") or "").strip()
            if ans and not ans.startswith("• Risposta non disponibile") and not ans.startswith("Domanda rilevata"):
                lines.append(f"  Note/Risposta: {ans}")
        lines.append("--- FINE REGISTRO DOMANDE DOCENTE ---\n")
        questions_block = "\n".join(lines)

    combined_parts = []
    if rag_context:
        combined_parts.append(f"Materiali, lezioni e appunti del corso:\n{rag_context}")
    if questions_block:
        combined_parts.append(questions_block)

    combined = (
        "\n\n".join(combined_parts)
        if combined_parts else "Nessun materiale specifico trovato nel database del corso."
    )

    # Costruzione delle fonti pulite per il badge della chat
    source_titles = []
    if questions_block and any(w in query.lower() for w in ["domand", "chied", "chiesto", "prof", "docente", "fatte", "lezione"]):
        source_titles.append("Registro Domande Docente")

    found_titles = re.findall(r"^\[(.*?)\]:", rag_context, flags=re.MULTILINE)
    for t in found_titles:
        clean_t = t.strip()
        if clean_t and clean_t not in source_titles:
            source_titles.append(clean_t)

    if not source_titles:
        if rag_context:
            source_titles.append("Appunti e materiali del corso")
        elif questions_block:
            source_titles.append("Registro Domande Docente")
        else:
            source_titles.append("Conoscenza generale")

    ctx_summary = " · ".join(source_titles)
    return combined, ctx_summary


# ── REST: Chat Assistente della Materia (AssistenteCorsoView) ─────────────────
@app.post("/api/v1/subjects/{subject_id}/chat-stream", tags=["Subjects"])
async def course_chat_stream_endpoint(
    subject_id: int,
    payload: CourseChatRequest,
    db: Session = Depends(get_db),
):
    """
    Risponde in streaming token-by-token interrogando LanceDB e generando con Qwen 3.5 MLX.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota.")

    from llm_engine import stream_chat_response, clear_metal_cache
    from config import load_settings

    combined, ctx_summary = await asyncio.to_thread(_prepare_course_context, subject_id, query, db)
    settings = load_settings()
    llm_model = payload.model_id or settings.get("llm_model")

    async def stream_generator():
        try:
            yield f"__CONTEXT_USED__:{ctx_summary}\n\n"
            async for token in stream_chat_response(query=query, context=combined, model_id=llm_model):
                yield token
        except asyncio.CancelledError:
            logger.info(f"[Chat Stream] Richiesta interrotta dal client per materia {subject_id}")
        finally:
            clear_metal_cache()

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/v1/subjects/{subject_id}/chat", tags=["Subjects"], deprecated=True)
async def course_chat_endpoint(
    subject_id: int,
    payload: CourseChatRequest,
    db: Session = Depends(get_db),
):
    """
    [DEPRECATO] Utilizzare l'endpoint di streaming '/api/v1/subjects/{subject_id}/chat-stream'.
    Risponde ai dubbi dello studente sul corso interrogando LanceDB (lezioni passate e materiali)
    e sintetizzando con Qwen 3.5 4B su Apple Silicon Metal.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota.")

    from llm_engine import stream_chat_response, clear_metal_cache
    from config import load_settings

    combined, ctx_summary = await asyncio.to_thread(_prepare_course_context, subject_id, query, db)
    settings = load_settings()
    llm_model = payload.model_id or settings.get("llm_model")

    full_answer = ""
    async for token in stream_chat_response(query=query, context=combined, model_id=llm_model):
        full_answer += token

    clear_metal_cache()
    return {
        "answer": full_answer.strip(),
        "context_used": ctx_summary,
    }


# ── REST: Re-indicizzazione LanceDB per la materia ────────────────────────────
@app.post("/api/v1/subjects/{subject_id}/reindex", tags=["Subjects"])
def reindex_subject_knowledge(subject_id: int, db: Session = Depends(get_db)):
    """Indicizza tutte le lezioni e i materiali della materia in LanceDB."""
    from pathlib import Path
    import llm_engine
    from rag import index_lecture, index_material_document, index_material_pages, unload_embedding_model
    from database import Subject, Lecture
    from routers.subjects import list_materials, _extract_pdf_pages, _extract_pptx_pages, _resolve_material_file

    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    llm_engine.unload_llm()
    try:
        lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).all()
        count_chunks = 0
        for lec in lectures:
            if lec.raw_transcript or lec.processed_notes:
                n = index_lecture(
                    subject_id=subject_id,
                    lecture_id=lec.id,
                    title=lec.title,
                    transcript=lec.raw_transcript or "",
                    notes=lec.processed_notes,
                )
                count_chunks += n

        materials = list_materials(subject_id=subject_id, db=db)
        for m in materials:
            # Le voci "lecture" sono già indicizzate sopra con index_lecture: re-indicizzarle come
            # materiale duplicherebbe i vettori (che sopravvivrebbero all'eliminazione della lezione)
            if m.type == "lecture":
                continue

            ext = Path(m.name).suffix.lower() if m.name else ""
            if not m.isAutoImported and ext in (".pdf", ".pptx"):
                try:
                    _, resolved = _resolve_material_file(subject_id, m.id, db)
                    pages = _extract_pdf_pages(resolved) if ext == ".pdf" else _extract_pptx_pages(resolved)
                    if pages:
                        n = index_material_pages(
                            subject_id=subject_id,
                            source_id=m.id,
                            title=m.name,
                            pages=pages,
                            source_type="slide",
                        )
                        count_chunks += n
                        continue
                except Exception as exc:
                    logger.warning(f"[Reindex] Estrazione pagine fallita per '{m.name}', fallback a testo: {exc}")

            if m.content:
                n = index_material_document(
                    subject_id=subject_id,
                    material_id=m.id,
                    title=m.name,
                    content=m.content,
                    source_type=m.type,
                )
                count_chunks += n

        return {"status": "success", "subject_id": subject_id, "indexed_chunks": count_chunks}
    finally:
        try:
            unload_embedding_model()
        except Exception:
            pass
        llm_engine.clear_metal_cache()


# ── REST: Arresto di Emergenza IA & Raffreddamento Mac (Zero Calore) ───────────
class StopAIRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    force_unload: bool = True


@app.post("/api/v1/ai/stop", tags=["AI"])
def stop_all_ai_endpoint(payload: Optional[StopAIRequest] = None):
    """
    Interrompe immediatamente ogni operazione IA in corso (sintesi quaderno, chat, trascrizioni, ecc.),
    svuota la cache Metal e scarica i modelli se specificato, azzerando il carico hardware (CPU/GPU 0%).
    """
    from llm_engine import stop_all_ai
    force_unload = payload.force_unload if payload else True
    return stop_all_ai(force_unload=force_unload)


@app.get("/api/v1/ai/status", tags=["AI"])
def get_ai_status_endpoint():
    """Restituisce lo stato corrente di occupazione del motore IA."""
    from llm_engine import is_ai_busy
    return is_ai_busy()


if __name__ == "__main__":
    import uvicorn
    # Solo loopback: il server non ha autenticazione e non deve essere raggiungibile dalla LAN
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
