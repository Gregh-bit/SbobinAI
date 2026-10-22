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

import asyncio
import json
import logging
import queue
import re
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import Lecture, SessionLocal, get_db, init_db
from live_session import LiveSession, end_session, get_session, has_active_sessions, start_session
from routers.subjects import router as subjects_router
from routers.lectures import router as lectures_router, bookmark_router
from routers.settings import router as settings_router
from routers.models import router as models_router
from routers.study import router as study_router
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
    logger.info("🚀 SbobinAI Backend avviato")
    init_db()
    # Bonifica automatica file orfani e compattazione DB all'avvio
    try:
        from cleaner import cleanup_orphan_files
        cleanup_orphan_files()
    except Exception as exc:
        logger.warning(f"[Startup] Cleanup orfani: {exc}")

    # Su MacBook Air M1 con 8GB di RAM, Whisper viene caricato on-demand solo all'avvio del live
    yield
    logger.info("🛑 SbobinAI Backend in shutdown")


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
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router ─────────────────────────────────────────────────────────────────────
app.include_router(subjects_router, prefix="/api/v1")
app.include_router(lectures_router, prefix="/api/v1")
app.include_router(bookmark_router, prefix="/api/v1")
app.include_router(settings_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(study_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint di healthcheck."""
    return {"status": "ok", "service": "SbobinAI", "version": "0.1.0"}


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

    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) < 5:
            continue
        if "?" in s_clean:
            results.append(s_clean)
        elif any(s_clean.lower().startswith(cue) for cue in interrogative_cues):
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


async def _generate_flash_answer_task(session: LiveSession, question: str, question_id: str, ws: WebSocket):
    """Genera la risposta flash in 3 punti elenco con l'LLM locale e aggiorna il WebSocket."""
    try:
        from rag import query_context
        from llm_engine import generate_flash_answer

        # 1. Recupera gli ultimi 3 minuti del buffer live
        live_context = session.get_recent_context(minutes=3)
        # 2. Interroga LanceDB per la materia corrente (priorità a lezioni precedenti e materiali)
        rag_context = await asyncio.to_thread(
            query_context,
            subject_id=session.subject_id,
            query=question,
            top_k=3,
            priority_previous_lecture=True,
        )
        combined = f"Ultime spiegazioni live:\n{live_context}\n\nMateriali e lezioni precedenti:\n{rag_context}"

        settings = load_settings()
        llm_model = settings.get("llm_model")

        flash_answer = await generate_flash_answer(question=question, context=combined, model_id=llm_model)

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
        try:
            await ws.send_json({
                "type": "question_detected",
                "payload": {
                    "id": question_id,
                    "question": question,
                    "confidence": 0.98,
                    "suggested_answer": flash_answer,
                    "trigger": "TRIGGER_RAG",
                }
            })
            logger.info(f"[Flash Answer] Risposta generata e inviata via WS per '{question[:30]}...'")
        except Exception:
            pass
    except Exception as exc:
        logger.warning(f"[Flash Answer] Errore generazione risposta flash: {exc}")


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
    from llm_engine import generate_lecture_chapter, clear_metal_cache, unload_whisper, unload_llm

    # Libera preventivamente la memoria da Whisper (solo se nessun'altra lezione live lo sta usando)
    if not has_active_sessions():
        await asyncio.to_thread(unload_whisper)

    db_sess = SessionLocal()
    try:
        lecture = db_sess.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture:
            return

        # Riferimenti da lezioni passate citati o correlati (Embedding model)
        past_refs = await asyncio.to_thread(
            query_context,
            subject_id=subject_id,
            query=full_transcript[:1500],
            top_k=3,
            priority_previous_lecture=True,
        )
        # Rilascia modello embedding prima di caricare l'LLM pesante per evitare swap su 8GB
        unload_embedding_model()
        clear_metal_cache()

        # Segnalibri della lezione
        bookmarks = db_sess.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
        bm_summary = (
            "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bookmarks])
            if bookmarks else "Nessun segnalibro specifico inserito."
        )

        settings = load_settings()
        llm_model = settings.get("llm_model")

        chapter = await generate_lecture_chapter(
            title=lecture.title,
            transcript=full_transcript,
            past_references=past_refs,
            bookmarks_summary=bm_summary,
            model_id=llm_model,
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
        await asyncio.to_thread(
            index_lecture,
            subject_id=subject_id,
            lecture_id=lecture_id,
            title=lecture.title,
            transcript=full_transcript,
            notes=chapter,
        )
        # Rilascia embedding post-indicizzazione
        unload_embedding_model()
        clear_metal_cache()
        logger.info(f"[Post-Lezione] Lezione {lecture_id} indicizzata con successo su LanceDB.")
    except InterruptedError:
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
        db_sess.close()
        clear_metal_cache()


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
            try:
                await ws.send_json({
                    "type": "status",
                    "payload": {"status": new_status}
                })
            except Exception:
                pass

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

            if text and session.is_active and not session.is_paused:
                session.push_transcript(text)
                elapsed = session.elapsed_seconds()
                try:
                    await ws.send_json({
                        "type": "transcript_chunk",
                        "payload": {
                            "text": text,
                            "elapsed": elapsed,
                        }
                    })
                except Exception as ws_err:
                    logger.debug(f"[WS Worker] Errore invio transcript_chunk: {ws_err}")
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
    """Rileva le domande del docente nel testo, le notifica al client e avvia le risposte flash."""
    for q in extract_professor_questions(text):
        q_obj = session.add_question(
            q,
            confidence=0.95,
            suggested_answer="Generazione risposta flash in corso...",
        )
        logger.info(f"TRIGGER_RAG: [Domanda del prof rilevata] {q}")
        try:
            await ws.send_json({
                "type": "question_detected",
                "payload": {
                    "id": q_obj["id"],
                    "question": q_obj["question"],
                    "confidence": q_obj["confidence"],
                    "suggested_answer": q_obj["suggestedAnswer"],
                    "trigger": "TRIGGER_RAG",
                }
            })
        except Exception as ws_err:
            logger.debug(f"[WS Question] Errore invio WebSocket: {ws_err}")

        # Avvia risposta flash istantanea
        _spawn(_generate_flash_answer_task(session, q, q_obj["id"], ws))


async def _finalize_session(session: LiveSession, subject_id: int) -> Tuple[float, Optional[str]]:
    """
    Chiude una sessione live in modo idempotente:
    1. ferma worker e microfono ed esporta il WAV,
    2. trascrive i blocchi vocali ancora in coda (l'ultima frase prima dello stop),
    3. rimuove la sessione dal registry e libera la memoria del transcriber,
    4. salva trascrizione, durata, audio e domande su SQLite,
    5. avvia in background la generazione del capitolo e l'indicizzazione.
    """
    if session.finalized:
        return session.elapsed_seconds(), None
    session.finalized = True

    duration = session.elapsed_seconds()
    session.is_active = False

    task = session.transcription_task
    if task and not task.done():
        task.cancel()
        await asyncio.wait({task}, timeout=10)

    audio_path: Optional[str] = None
    try:
        if session.audio_engine.is_running:
            # stop() svuota anche l'ultimo blocco vocale nella speech_queue
            audio_path = await asyncio.to_thread(session.audio_engine.stop)
    except Exception as exc:
        logger.error(f"[WS] Errore arresto AudioEngine: {exc}")

    # Trascrive ciò che è rimasto in coda (altrimenti la fine della lezione andrebbe persa)
    if not session.transcriber.mock_mode:
        while True:
            try:
                chunk = session.audio_engine.speech_queue.get_nowait()
            except queue.Empty:
                break
            try:
                tail = await session.transcriber.transcribe(chunk)
            except Exception as exc:
                logger.error(f"[WS] Errore trascrizione coda finale: {exc}")
                break
            if tail:
                session.push_transcript(tail)

    # Registry + rilascio memoria transcriber (evita doppia invocazione di stop su audio_engine)
    try:
        from live_session import _active_sessions
        if _active_sessions.get(subject_id) is session:
            _active_sessions.pop(subject_id, None)
        if hasattr(session, "transcriber") and session.transcriber:
            await asyncio.to_thread(session.transcriber.release_memory)
    except Exception as exc:
        logger.debug(f"[WS] Errore pulizia sessione e rilascio memoria: {exc}")

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
    except Exception as db_err:
        logger.error(f"[WS] Errore salvataggio lezione su DB: {db_err}")
    finally:
        db_sess.close()

    if full_transcript and len(full_transcript.strip()) > 30:
        _spawn(_post_lecture_summary_and_index(
            subject_id=subject_id,
            lecture_id=session.lecture_id,
            full_transcript=full_transcript,
        ))

    # Pre-calcola subito i timestamp sincronizzati parola per parola a fine registrazione
    if session.lecture_id:
        from routers.lectures import compute_and_cache_lecture_timestamps
        _spawn(asyncio.to_thread(compute_and_cache_lecture_timestamps, session.lecture_id))

    logger.info(
        f"[WS] Sessione chiusa: lezione={session.lecture_id}, durata={duration:.1f}s, "
        f"chunk totali={len(session.full_transcript)}, wav={audio_path}"
    )
    return duration, audio_path


# ── WebSocket: Live Lecture ────────────────────────────────────────────────────

# Origini ammesse per il WebSocket (il CORS non si applica ai WebSocket):
# dev server Vite, Tauri macOS/Linux (tauri://localhost) e Tauri Windows (http(s)://tauri.localhost)
_ALLOWED_WS_ORIGIN = re.compile(
    r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|https?://tauri\.localhost)$"
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
    if origin and not _ALLOWED_WS_ORIGIN.match(origin):
        logger.warning(f"[WS] Connessione rifiutata da origine non ammessa: {origin}")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info(f"[WS] Connessione live aperta per materia {subject_id}")

    async def send_error(message: str) -> None:
        await websocket.send_json({"type": "error", "payload": {"message": message}})

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

            # ── START ──────────────────────────────────────────────────────────
            if msg_type == "start":
                title = str(payload.get("title") or "Nuova Lezione")[:255]
                lecture_id = payload.get("lecture_id")

                from database import Lecture as LectureModel, Subject
                with SessionLocal() as db:
                    subject = db.query(Subject).filter(Subject.id == subject_id).first()
                    if not subject:
                        await send_error(f"Materia {subject_id} non trovata")
                        continue
                    whisper_keywords = subject.whisper_prompt_keywords

                    if lecture_id is not None:
                        try:
                            lecture_id = int(lecture_id)
                        except (TypeError, ValueError):
                            await send_error("lecture_id non valido")
                            continue
                        owned = (
                            db.query(LectureModel)
                            .filter(LectureModel.id == lecture_id, LectureModel.subject_id == subject_id)
                            .first()
                        )
                        if not owned:
                            await send_error(f"Lezione {lecture_id} non trovata per la materia {subject_id}")
                            continue

                    # Una sessione già aperta (su questo socket o su un socket precedente rimasto appeso)
                    # viene prima salvata e chiusa, mai scartata.
                    if session is not None:
                        await _finalize_session(session, subject_id)
                        session = None
                    stale = get_session(subject_id)
                    if stale is not None:
                        logger.warning(f"[WS] Sessione precedente ancora attiva per materia {subject_id}: la salvo e la chiudo.")
                        await _finalize_session(stale, subject_id)

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

                # Avvia sessione (AudioEngine in RAM + Transcriber)
                session = start_session(
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

                await websocket.send_json({
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
                    await websocket.send_json({
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
                await websocket.send_json({
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
                context_minutes = _as_int(payload.get("context_minutes", 5), default=5, lo=1, hi=15)

                from rag import query_context
                from llm_engine import stream_chat_response

                live_context = session.get_recent_context(minutes=context_minutes)
                rag_context = await asyncio.to_thread(query_context, subject_id=subject_id, query=query, top_k=3)
                combined_context = (
                    f"Ultime spiegazioni live (ultimi {context_minutes} minuti):\n{live_context}\n\n"
                    f"Materiali e nozioni del corso:\n{rag_context}"
                )

                settings = load_settings()
                llm_model = settings.get("llm_model")

                full_answer = ""
                try:
                    async for token in stream_chat_response(query=query, context=combined_context, model_id=llm_model):
                        full_answer += token
                        await websocket.send_json({
                            "type": "chat_chunk",
                            "payload": {
                                "chunk": token,
                                "query": query,
                            }
                        })
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception as c_err:
                    logger.error(f"[WS Chat] Errore streaming: {c_err}")
                    full_answer = f"Errore durante l'elaborazione della risposta: {c_err}"

                await websocket.send_json({
                    "type": "chat_response",
                    "payload": {
                        "answer": full_answer,
                        "query": query,
                        "context_used": (live_context[:400] + "...") if len(live_context) > 400 else live_context,
                    }
                })

            # ── DELETE QUESTION (elimina domanda non pertinente in live) ───────
            elif msg_type == "delete_question":
                if session:
                    q_id = payload.get("question_id") or msg.get("question_id")
                    if q_id:
                        removed = session.delete_question(str(q_id))
                        logger.info(f"[WS] Domanda rilevata {q_id} rimossa dalla sessione live (success={removed})")

            # ── PAUSE (Consolidamento istantaneo + Stop motori + Deep sleep) ───
            elif msg_type == "pause":
                if session and not session.is_paused:
                    logger.info(f"[WS] Richiesta PAUSA per lezione {session.lecture_id}: consolidamento immediato buffer residuo...")
                    # 1. Chiude microfono e estrae l'ultimo chunk vocale accumulato (anche se breve)
                    last_audio = session.pause()

                    # 2. Consolidamento istantaneo (Ultimo sforzo del Large)
                    if last_audio and len(last_audio) >= 1600:
                        try:
                            refined_text = await session.transcriber.transcribe(last_audio)
                            if refined_text:
                                session.push_transcript(refined_text)
                                await websocket.send_json({
                                    "type": "transcript_chunk",
                                    "payload": {
                                        "text": refined_text,
                                        "elapsed": session.elapsed_seconds(),
                                    }
                                })
                                # Rilevamento domande sull'ultimo parlato prima della pausa
                                await _announce_questions(session, refined_text, websocket)
                        except Exception as p_err:
                            logger.error(f"[WS Pause] Errore trascrizione ultimo pezzo: {p_err}")

                    # 3. Stop motori (Calore Zero) & Svuotamento cache (Deep Sleep)
                    session.transcriber.release_metal_cache()
                    logger.info("[WS] Pausa completata: microfono chiuso, modelli a riposo, cache liberata")
                    try:
                        await websocket.send_json({"type": "paused"})
                    except Exception:
                        pass

            # ── RESUME (Ripresa registrazione da pausa) ────────────────────────
            elif msg_type == "resume":
                if session and session.is_paused:
                    session.resume()
                    logger.info(f"[WS] Registrazione RIPRESA per lezione {session.lecture_id}: microfono e modelli riattivati.")
                    try:
                        await websocket.send_json({"type": "resumed"})
                    except Exception:
                        pass

            # ── STOP ───────────────────────────────────────────────────────────
            elif msg_type == "stop":
                if not session:
                    continue
                duration, audio_path = await _finalize_session(session, subject_id)
                total_chunks = len(session.full_transcript)
                session = None
                try:
                    await websocket.send_json({
                        "type": "stopped",
                        "payload": {
                            "duration": duration,
                            "total_chunks": total_chunks,
                            "audio_path": audio_path,
                        }
                    })
                except Exception:
                    pass
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
        # Qualsiasi uscita (disconnessione, errore, crash) salva la lezione e libera il microfono
        # shield: anche se questo handler viene cancellato (shutdown, client sparito) il salvataggio termina
        if session is not None and not session.finalized:
            await asyncio.shield(_spawn(_finalize_session(session, subject_id)))


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

    transcript = lecture.raw_transcript or lecture.processed_notes or ""
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Nessun testo disponibile per generare il capitolo.")

    from rag import query_context, index_lecture, unload_embedding_model
    from llm_engine import generate_lecture_chapter, clear_metal_cache, unload_whisper, unload_llm
    from config import load_settings
    from database import Bookmark

    # Unload coordinato di Whisper prima del riassunto (non durante una lezione live)
    if not has_active_sessions():
        await asyncio.to_thread(unload_whisper)

    settings = load_settings()
    llm_model = (payload.model_id if payload else None) or settings.get("llm_model")

    bms = db.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
    bm_summary = (
        "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bms])
        if bms else "Nessun segnalibro specifico inserito."
    )

    past_refs = await asyncio.to_thread(
        query_context,
        subject_id=lecture.subject_id,
        query=transcript[:1500],
        top_k=3,
        priority_previous_lecture=True,
    )
    # Scarica embedding prima di allocare l'LLM pesante per non superare gli 8GB di memoria
    unload_embedding_model()
    clear_metal_cache()

    try:
        chapter = await generate_lecture_chapter(
            title=lecture.title,
            transcript=transcript,
            past_references=past_refs,
            bookmarks_summary=bm_summary,
            model_id=llm_model,
        )
    except InterruptedError:
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
        clear_metal_cache()
        logger.error(f"[Lectures] Generazione sintesi per lezione {lecture_id} fallita: {exc}")
        raise HTTPException(status_code=500, detail=f"Generazione del capitolo non riuscita: {exc}")

    if not chapter:
        raise HTTPException(status_code=500, detail="Il modello non ha prodotto alcun capitolo.")

    lecture.processed_notes = chapter
    db.commit()

    # Rilascia l'LLM dalla RAM prima di passare all'indicizzazione con embedding
    unload_llm()
    clear_metal_cache()

    # Indicizzazione automatica in LanceDB solo se completata con successo
    await asyncio.to_thread(
        index_lecture,
        subject_id=lecture.subject_id,
        lecture_id=lecture.id,
        title=lecture.title,
        transcript=transcript,
        notes=chapter,
    )
    unload_embedding_model()
    clear_metal_cache()

    return {
        "status": "success",
        "lecture_id": lecture_id,
        "processed_notes": chapter,
    }


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

    q_text = target_q.get("question", "")
    context = lecture.processed_notes or lecture.raw_transcript or lecture.title or ""

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
    lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).all()
    all_detected_questions = []
    for lec in lectures:
        if lec.detected_questions:
            try:
                q_list = json.loads(lec.detected_questions)
                if isinstance(q_list, list):
                    for q_obj in q_list:
                        q_text = q_obj.get("question") if isinstance(q_obj, dict) else str(q_obj)
                        if q_text:
                            all_detected_questions.append({
                                "lecture_title": lec.title or f"Lezione #{lec.id}",
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
    from rag import index_lecture, index_material_document
    from database import Subject, Lecture
    from routers.subjects import list_materials

    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

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
        if m.content and m.type != "lecture":
            n = index_material_document(
                subject_id=subject_id,
                material_id=m.id,
                title=m.name,
                content=m.content,
                source_type=m.type,
            )
            count_chunks += n

    return {"status": "success", "subject_id": subject_id, "indexed_chunks": count_chunks}


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
