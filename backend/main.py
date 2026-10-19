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
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import Lecture, get_db, init_db
from live_session import LiveSession, end_session, get_session, start_session
from routers.subjects import router as subjects_router
from routers.lectures import router as lectures_router, bookmark_router
from routers.settings import router as settings_router
from routers.models import router as models_router
from routers.study import router as study_router
from schemas import CourseChatRequest
from config import load_settings


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


async def _generate_flash_answer_task(session: LiveSession, question: str, question_id: str, ws: WebSocket):
    """Genera la risposta flash in 3 punti elenco con Qwen 3.5 MLX e aggiorna il WebSocket."""
    try:
        from rag import query_context
        from llm_engine import generate_flash_answer
        from config import load_settings

        # 1. Recupera gli ultimi 3 minuti del buffer live
        live_context = session.get_recent_context(minutes=3)
        # 2. Interroga LanceDB per la materia corrente (priorità a lezioni precedenti e materiali)
        rag_context = query_context(
            subject_id=session.subject_id,
            query=question,
            top_k=3,
            priority_previous_lecture=True,
        )
        combined = f"Ultime spiegazioni live:\n{live_context}\n\nMateriali e lezioni precedenti:\n{rag_context}"

        settings = load_settings()
        llm_model = settings.get("llm_model")

        flash_answer = await generate_flash_answer(question=question, context=combined, model_id=llm_model)

        # Aggiorna la risposta nella sessione in memoria
        for q_item in session.detected_questions:
            if q_item.get("id") == question_id:
                q_item["suggestedAnswer"] = flash_answer
                break

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
    1. Scarica Whisper dalla memoria per lasciare tutti gli 8GB a Qwen 3.5 4B
    2. Cerca riferimenti a materiali e lezioni passate su LanceDB
    3. Genera il capitolo completo strutturato per il Quaderno
    4. Salva il capitolo nel DB SQLite e lo indicizza automaticamente in LanceDB
    """
    from database import SessionLocal, Lecture, Bookmark
    from rag import query_context, index_lecture, unload_embedding_model
    from llm_engine import generate_lecture_chapter, clear_metal_cache, unload_whisper, unload_llm
    from config import load_settings

    # Libera preventivamente la memoria Metal da Whisper
    unload_whisper()

    db_sess = SessionLocal()
    try:
        lecture = db_sess.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture:
            return

        # Riferimenti da lezioni passate citati o correlati (Embedding model)
        past_refs = query_context(
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

        lecture.processed_notes = chapter
        db_sess.commit()
        logger.info(f"[Post-Lezione] Capitolo Quaderno generato e salvato su SQLite per lezione {lecture_id}.")

        # Rilascia l'LLM prima di passare all'indicizzazione vettoriale
        unload_llm()
        clear_metal_cache()

        # Auto-indicizzazione su LanceDB
        index_lecture(
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
    except Exception as exc:
        logger.error(f"[Post-Lezione] Errore elaborazione riassunto/indicizzazione: {exc}")
    finally:
        db_sess.close()
        clear_metal_cache()


# ── Worker di Trascrizione Live Whisper Large Turbo ────────────────────────────
async def _transcription_worker(session: LiveSession, ws: WebSocket):
    """
    Consuma i blocchi vocali accumulati (ogni 20s continuativi o su pausa VAD >500ms)
    con Whisper Large-v3-Turbo Q4 per la massima accuratezza accademica e zero allucinazioni.
    Invia lo stato audio al frontend ('listening' / 'transcribing' / 'idle') per il feedback visivo.
    """
    logger.info(f"[WS Worker] Avviato streaming trascrizione Whisper Large Turbo per lezione {session.lecture_id}")
    last_speech_time = time.time()
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
                    # Deep Sleep: se c'è silenzio prolungato (>40s), rilascia i buffer di calcolo Metal
                    if (time.time() - last_speech_time) > 40.0:
                        session.transcriber.release_metal_cache()
                    await send_status("idle")

                if session.transcriber.mock_mode:
                    await asyncio.sleep(4.0)
                    if not session.is_active or session.is_paused:
                        break
                    chunk_bytes = b"\x00" * 960
                else:
                    await asyncio.sleep(0.1)
                    continue

            # Abbiamo un blocco vocale da trascrivere con Whisper Large
            last_speech_time = time.time()
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
                detected_questions = extract_professor_questions(text)
                for q in detected_questions:
                    q_obj = session.add_question(
                        q,
                        confidence=0.95,
                        suggested_answer="Generazione risposta flash in corso con Qwen 3.5 MLX..."
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

                    # Avvia risposta flash istantanea con Qwen 3.5 MLX
                    asyncio.create_task(_generate_flash_answer_task(session, q, q_obj["id"], ws))

            if session.is_active and not session.is_paused:
                if session.audio_engine.is_voice_active:
                    await send_status("listening")
                else:
                    await send_status("idle")

    except asyncio.CancelledError:
        logger.info(f"[WS Worker] Worker trascrizione terminato per lezione {session.lecture_id}")
    except Exception as exc:
        logger.error(f"[WS Worker] Errore inatteso nel worker: {exc}")


# ── WebSocket: Live Lecture ────────────────────────────────────────────────────

# Registry delle connessioni WebSocket attive: subject_id → WebSocket
_ws_connections: Dict[int, WebSocket] = {}


@app.websocket("/ws/live-lecture/{subject_id}")
async def live_lecture_ws(
    websocket: WebSocket,
    subject_id: int,
    db: Session = Depends(get_db),
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
    await websocket.accept()
    _ws_connections[subject_id] = websocket
    logger.info(f"[WS] Connessione live aperta per materia {subject_id}")

    session: LiveSession | None = None
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg: Dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "payload": {"message": "JSON non valido"}})
                continue

            msg_type = msg.get("type", "")
            payload = msg.get("payload", {})

            # ── START ──────────────────────────────────────────────────────────
            if msg_type == "start":
                title = payload.get("title", "Nuova Lezione")
                lecture_id = payload.get("lecture_id")

                from database import Lecture as LectureModel, Subject
                subject = db.query(Subject).filter(Subject.id == subject_id).first()
                if not subject:
                    await websocket.send_json({
                        "type": "error",
                        "payload": {"message": f"Materia {subject_id} non trovata"}
                    })
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

                mock_transcriber = (model == "mock_mode")

                # Avvia sessione (AudioEngine in RAM + Transcriber MLX/Mock)
                session = start_session(
                    subject_id=subject_id,
                    lecture_id=lecture_id,
                    whisper_keywords=subject.whisper_prompt_keywords,
                    mock_transcriber=mock_transcriber,
                    model_name=model,
                    vad_sensitivity=int(vad_sensitivity),
                )
                logger.info(
                    f"[WS] Sessione e microfono avviati: materia={subject_id}, lezione={lecture_id}, "
                    f"modello={model}, VAD={vad_sensitivity}"
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
                    await websocket.send_json({
                        "type": "error",
                        "payload": {"message": "Sessione non avviata. Invia prima 'start'."}
                    })
                    continue
                text = payload.get("text", "")
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
                note = payload.get("note", "")
                ts = session.elapsed_seconds()
                bookmark = Bookmark(
                    lecture_id=session.lecture_id,
                    timestamp_seconds=ts,
                    note=note,
                )
                db.add(bookmark)
                db.commit()
                db.refresh(bookmark)
                logger.info(f"[WS] Segnalibro salvato @ {ts:.1f}s: '{note}'")
                await websocket.send_json({
                    "type": "bookmark_saved",
                    "payload": {"bookmark_id": bookmark.id, "timestamp": ts, "note": note}
                })

            # ── CHAT (domanda all'assistente live con streaming RAG puro MLX) ──
            elif msg_type == "chat":
                if not session:
                    continue
                query = payload.get("query", "")
                context_minutes = payload.get("context_minutes", 5)

                from rag import query_context
                from llm_engine import stream_chat_response
                from config import load_settings

                live_context = session.get_recent_context(minutes=context_minutes)
                rag_context = query_context(subject_id=subject_id, query=query, top_k=3)
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
                except Exception as c_err:
                    logger.error(f"[WS Chat] Errore streaming: {c_err}")
                    full_answer = f"Errore durante l'elaborazione della risposta: {c_err}"

                await websocket.send_json({
                    "type": "chat_response",
                    "payload": {
                        "answer": full_answer,
                        "context_used": (live_context[:400] + "...") if len(live_context) > 400 else live_context,
                    }
                })

            # ── DELETE QUESTION (elimina domanda non pertinente in live) ───────
            elif msg_type == "delete_question":
                if session:
                    q_id = payload.get("question_id") or msg.get("question_id")
                    if q_id:
                        removed = session.delete_question(q_id)
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
                                elapsed = session.elapsed_seconds()
                                await websocket.send_json({
                                    "type": "transcript_chunk",
                                    "payload": {
                                        "text": refined_text,
                                        "elapsed": elapsed,
                                    }
                                })
                                # Rilevamento domande sull'ultimo parlato prima della pausa
                                detected = extract_professor_questions(refined_text)
                                for q in detected:
                                    q_obj = session.add_question(
                                        q,
                                        confidence=0.95,
                                        suggested_answer="Generazione risposta flash in corso con Qwen 3.5 MLX..."
                                    )
                                    await websocket.send_json({
                                        "type": "question_detected",
                                        "payload": {
                                            "id": q_obj["id"],
                                            "question": q_obj["question"],
                                            "confidence": q_obj["confidence"],
                                            "suggested_answer": q_obj["suggestedAnswer"],
                                            "trigger": "TRIGGER_RAG",
                                        }
                                    })
                                    asyncio.create_task(_generate_flash_answer_task(session, q, q_obj["id"], websocket))
                        except Exception as p_err:
                            logger.error(f"[WS Pause] Errore trascrizione ultimo pezzo: {p_err}")

                    # 3. Stop motori (Calore Zero) & Svuotamento cache (Deep Sleep)
                    session.transcriber.release_metal_cache()
                    logger.info(f"[WS] Pausa completata: microfono chiuso, modelli a riposo, cache Metal liberata (0% CPU/GPU)")
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
                duration = session.elapsed_seconds()

                # 1. Arresta immediatamente il worker e l'ascolto per terminare all'istante
                session.is_active = False
                if session.transcription_task:
                    session.transcription_task.cancel()

                # 2. Ferma l'AudioEngine ed esporta il WAV (senza attendere ulteriori elaborazioni lente)
                audio_path = session.audio_engine.stop()

                # Rilascia immediatamente memoria RAM e cache Metal
                try:
                    session.transcriber.release_memory()
                except Exception:
                    pass

                full_transcript = session.get_full_transcript()
                lecture_id_val = session.lecture_id

                # 3. Termina e rimuovi la sessione
                end_session(subject_id)

                # 4. Aggiorna la lezione nel database SQLite
                try:
                    lecture = db.query(Lecture).filter(Lecture.id == lecture_id_val).first()
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
                        db.commit()
                except Exception as db_err:
                    logger.error(f"[WS Stop] Errore aggiornamento DB: {db_err}")

                # 5. Generazione Capitolo Quaderno & Indicizzazione LanceDB in background
                if full_transcript and len(full_transcript.strip()) > 30:
                    asyncio.create_task(_post_lecture_summary_and_index(
                        subject_id=subject_id,
                        lecture_id=lecture_id_val,
                        full_transcript=full_transcript,
                    ))

                _ws_connections.pop(subject_id, None)
                logger.info(f"[WS] Sessione terminata istantaneamente: durata={duration:.1f}s, chunk totali={len(session.full_transcript)}, wav={audio_path}")
                try:
                    await websocket.send_json({
                        "type": "stopped",
                        "payload": {
                            "duration": duration,
                            "total_chunks": len(session.full_transcript),
                            "audio_path": audio_path,
                        }
                    })
                except Exception:
                    pass
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": f"Tipo messaggio sconosciuto: '{msg_type}'"}
                })

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnesso (materia {subject_id})")
        if session:
            duration = session.elapsed_seconds()
            full_transcript = session.get_full_transcript()
            lec_id_val = session.lecture_id
            res = end_session(subject_id)
            audio_path = res[1] if res else None
            try:
                lecture = db.query(Lecture).filter(Lecture.id == lec_id_val).first()
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
                    db.commit()
            except Exception as e:
                logger.error(f"[WS] Errore salvataggio DB su disconnessione: {e}")

            if full_transcript and len(full_transcript.strip()) > 30:
                asyncio.create_task(_post_lecture_summary_and_index(
                    subject_id=subject_id,
                    lecture_id=lec_id_val,
                    full_transcript=full_transcript,
                ))
        _ws_connections.pop(subject_id, None)
    except Exception as exc:
        logger.exception(f"[WS] Errore inatteso: {exc}")
        try:
            await websocket.send_json({"type": "error", "payload": {"message": str(exc)}})
        except Exception:
            pass


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

    # Unload coordinato di Whisper prima del riassunto
    unload_whisper()

    settings = load_settings()
    llm_model = (payload.model_id if payload else None) or settings.get("llm_model")

    bms = db.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).all()
    bm_summary = (
        "\n".join([f"- Segnalibro a {b.timestamp_seconds:.1f}s: {b.note}" for b in bms])
        if bms else "Nessun segnalibro specifico inserito."
    )

    past_refs = query_context(
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

    lecture.processed_notes = chapter
    db.commit()

    # Rilascia l'LLM dalla RAM prima di passare all'indicizzazione con embedding
    unload_llm()
    clear_metal_cache()

    # Indicizzazione automatica in LanceDB solo se completata con successo
    index_lecture(
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

    target_q = None
    target_idx = -1
    for i, q in enumerate(questions_list):
        if str(q.get("id")) == str(question_id) or str(i) == str(question_id):
            target_q = q
            target_idx = i
            break

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
                                "suggested_answer": q_obj.get("suggestedAnswer") if isinstance(q_obj, dict) else ""
                            })
            except Exception:
                pass

    questions_block = ""
    if all_detected_questions:
        lines = ["\n--- REGISTRO DOMANDE RILEVATE A LEZIONE DAL DOCENTE (Whisper Detector) ---"]
        for item in all_detected_questions:
            lines.append(f"• In '{item['lecture_title']}': \"{item['question']}\"")
            ans = item.get("suggested_answer", "").strip()
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

    combined, ctx_summary = _prepare_course_context(subject_id, query, db)
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


@app.post("/api/v1/subjects/{subject_id}/chat", tags=["Subjects"])
async def course_chat_endpoint(
    subject_id: int,
    payload: CourseChatRequest,
    db: Session = Depends(get_db),
):
    """
    Risponde ai dubbi dello studente sul corso interrogando LanceDB (lezioni passate e materiali)
    e sintetizzando con Qwen 3.5 4B su Apple Silicon Metal.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota.")

    from llm_engine import stream_chat_response, clear_metal_cache
    from config import load_settings

    combined, ctx_summary = _prepare_course_context(subject_id, query, db)
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
