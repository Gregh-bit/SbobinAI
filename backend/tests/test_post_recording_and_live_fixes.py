"""
test_post_recording_and_live_fixes.py — SbobinAI
Unit test isolati per le correzioni post-registrazione, anti-allucinazione,
contesto live con timestamp, RAG ultra-rapido solo lessicale e resilienza WebSocket.
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave

import pytest

import live_session
import llm_engine
import main
import rag
import routers.lectures as lectures_router
import transcriber
from database import Lecture, SessionLocal
from live_session import LiveSession, TranscriptChunk


def _cleanup_live_session(sess: LiveSession) -> None:
    """Chiude eventuali file temporanei/task di una LiveSession creata in test."""
    grace = getattr(sess, "_disconnect_grace_task", None)
    if grace and not grace.done():
        grace.cancel()
    try:
        if sess.audio_engine._temp_pcm_file and not sess.audio_engine._temp_pcm_file.closed:
            sess.audio_engine._temp_pcm_file.close()
        if sess.audio_engine._temp_pcm_path and os.path.exists(sess.audio_engine._temp_pcm_path):
            os.remove(sess.audio_engine._temp_pcm_path)
    except Exception:
        pass


def _make_dummy_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    n_samples = int(duration_sec * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


def _recv_until(ws, wanted: str, limit: int = 40):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == wanted:
            return msg
    raise AssertionError(f"Messaggio WebSocket '{wanted}' non ricevuto entro {limit} frame.")


# ── 1. LiveSession.get_timestamped_recent_context() ───────────────────────────

def test_live_session_timestamped_recent_context_formatting(monkeypatch):
    """Verifica header [Tempo attuale della lezione: MM:SS] e annotazioni [MM:SS | X min fa / appena detto]."""
    sess = LiveSession(subject_id=101, lecture_id=201, mock_transcriber=True)
    try:
        # Simula lezione arrivata al minuto 05:30 (330.0s)
        monkeypatch.setattr(sess, "elapsed_seconds", lambda: 330.0)
        sess._buffer.append(TranscriptChunk(text="Definizione di integrale di Lebesgue.", timestamp=150.0))  # 02:30 -> 3 min fa
        sess._buffer.append(TranscriptChunk(text="Teorema della convergenza dominata.", timestamp=290.0))    # 04:50 -> 40 sec fa
        sess._buffer.append(TranscriptChunk(text="Passiamo ora agli esercizi svolti.", timestamp=325.0))     # 05:25 -> appena detto
        sess.full_transcript = [c.text for c in sess._buffer]

        ctx = sess.get_timestamped_recent_context(minutes=15, max_chars=4500)
        lines = ctx.splitlines()

        assert lines[0] == "[Tempo attuale della lezione: 05:30]"
        assert lines[1] == "[02:30 | 3 min fa] Definizione di integrale di Lebesgue."
        assert lines[2] == "[04:50 | 40 sec fa] Teorema della convergenza dominata."
        assert lines[3] == "[05:25 | appena detto] Passiamo ora agli esercizi svolti."
    finally:
        _cleanup_live_session(sess)


# ── 2. _microphone_error_message() ────────────────────────────────────────────

def test_microphone_error_message_os_specific(monkeypatch):
    """Verifica messaggi microfono specifici per macOS (darwin) e Windows (win32)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    mac_msg = main._microphone_error_message()
    assert "Impostazioni di Sistema" in mac_msg
    assert "Privacy e sicurezza" in mac_msg
    assert "Microfono" in mac_msg

    monkeypatch.setattr(sys, "platform", "win32")
    win_msg = main._microphone_error_message()
    assert "Impostazioni → Privacy → Microfono" in win_msg
    assert "Impostazioni di Sistema" not in win_msg

    monkeypatch.setattr(sys, "platform", "linux")
    linux_msg = main._microphone_error_message()
    assert "permessi audio" in linux_msg.lower()


# ── 3. Transcriber._extract_reliable_mlx_text() ───────────────────────────────

def test_extract_reliable_mlx_text_discards_unreliable_segments_and_keeps_speech():
    """Scarta segmenti con no_speech_prob >= 0.55, avg_logprob < -0.95 o compression_ratio > 2.2."""
    mlx_result = {
        "text": "Morre come arrivo Teorema di Gauss loop loop",
        "segments": [
            # Allucinazione su silenzio (no_speech_prob >= 0.55)
            {"text": "Morre", "no_speech_prob": 0.72, "avg_logprob": -0.30, "compression_ratio": 1.0},
            # Allucinazione a bassissima confidenza acustica (avg_logprob < -0.95)
            {"text": "come arrivo", "no_speech_prob": 0.20, "avg_logprob": -1.25, "compression_ratio": 1.1},
            # Segmento incerto combinato (no_speech >= 0.40 e avg_logprob < -0.60)
            {"text": "rumore incerto", "no_speech_prob": 0.45, "avg_logprob": -0.75, "compression_ratio": 1.0},
            # Loop compresso (compression_ratio > 2.2)
            {"text": "allora allora allora allora", "no_speech_prob": 0.05, "avg_logprob": -0.20, "compression_ratio": 2.65},
            # Parlato reale affidabile
            {"text": "Il teorema di Gauss mette in relazione flusso e divergenza.", "no_speech_prob": 0.08, "avg_logprob": -0.22, "compression_ratio": 1.25},
            # Secondo segmento valido al limite inferiore di sicurezza
            {"text": "Consideriamo ora un campo vettoriale solenoidale.", "no_speech_prob": 0.12, "avg_logprob": -0.35, "compression_ratio": 1.40},
        ],
    }

    extracted = transcriber.Transcriber._extract_reliable_mlx_text(mlx_result)
    assert "Morre" not in extracted
    assert "come arrivo" not in extracted
    assert "rumore incerto" not in extracted
    assert "allora allora" not in extracted
    assert "Il teorema di Gauss mette in relazione flusso e divergenza." in extracted
    assert "Consideriamo ora un campo vettoriale solenoidale." in extracted


# ── 4. routers/lectures.py _validate_timestamp_lines() ────────────────────────

def test_validate_timestamp_lines_rejects_empty_raw_and_mismatched_short_text():
    """Rifiuta cache non vuota quando raw_transcript è vuoto e testo breve non corrispondente ('Morre' vs 'come arrivo')."""
    hallucinated_morre = [
        {
            "lineStart": 0,
            "lineEnd": 1200,
            "words": [{"text": "Morre", "start": 0, "end": 1200}],
        }
    ]

    # Caso A: raw_transcript vuoto o solo spazi -> cache con parole deve essere rifiutata
    ok_empty, reason_empty = lectures_router._validate_timestamp_lines(hallucinated_morre, "")
    assert ok_empty is False
    assert "vuoto" in reason_empty.lower()

    ok_none, _ = lectures_router._validate_timestamp_lines(hallucinated_morre, None)
    assert ok_none is False

    # Caso B: testo breve ma completamente disgiunto ("Morre" vs "come arrivo") -> rifiutato
    ok_short, reason_short = lectures_router._validate_timestamp_lines(hallucinated_morre, "come arrivo")
    assert ok_short is False
    assert "nessuna parola in comune" in reason_short.lower()

    # Caso C: testo breve coerente -> accettato
    matching_short = [
        {
            "lineStart": 0,
            "lineEnd": 1200,
            "words": [
                {"text": "come", "start": 0, "end": 500},
                {"text": "arrivo", "start": 500, "end": 1200},
            ],
        }
    ]
    ok_match, reason_match = lectures_router._validate_timestamp_lines(matching_short, "come arrivo")
    assert ok_match is True
    assert reason_match == "ok"


# ── 5. get_lecture_timestamps() non avvelena cache_path ───────────────────────

def test_get_lecture_timestamps_does_not_poison_cache_when_whisper_scheduled(client, subject, tmp_path, monkeypatch):
    """Quando esiste l'audio e Whisper è schedulato in background, la risposta proporzionale NON scrive cache_path."""
    wav_path = _make_dummy_wav(str(tmp_path / "lec_no_poison.wav"), duration_sec=2.0)
    cache_path = wav_path + ".timestamps.json"

    with SessionLocal() as db:
        lec = Lecture(
            subject_id=subject["id"],
            title="Lezione Test Cache",
            raw_transcript="Oggi studiamo le serie di potenze e il raggio di convergenza.",
            duration_seconds=12.0,
            audio_path=wav_path,
            source_type="recorded",
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lecture_id = lec.id

    bg_calls: list[tuple[int, bool]] = []

    def fake_bg_compute(lid: int):
        # Al momento in cui parte il task Whisper, cache_path NON deve essere stato creato
        bg_calls.append((lid, os.path.exists(cache_path)))
        with lectures_router._pending_lock:
            lectures_router._pending_timestamp_tasks.discard(lid)

    monkeypatch.setattr(lectures_router, "_bg_compute_timestamps", fake_bg_compute)

    res = client.get(f"/api/v1/lectures/{lecture_id}/timestamps")
    assert res.status_code == 200
    body = res.json()
    assert len(body["lines"]) > 0
    assert body["cached"] is False
    # Il task in background deve essere stato schedulato e la cache non deve essere stata scritta
    assert bg_calls == [(lecture_id, False)]
    assert not os.path.exists(cache_path)


# ── 6. compute_and_cache_lecture_timestamps() salta Whisper se raw_transcript vuoto

def test_compute_and_cache_timestamps_skips_whisper_when_recorded_transcript_empty(subject, tmp_path, monkeypatch):
    """Su una lezione registrata con raw_transcript vuoto, Whisper NON viene mai invocato."""
    wav_path = _make_dummy_wav(str(tmp_path / "lec_silent.wav"), duration_sec=2.0)

    with SessionLocal() as db:
        lec = Lecture(
            subject_id=subject["id"],
            title="Lezione Silenziosa",
            raw_transcript="   ",
            duration_seconds=5.0,
            audio_path=wav_path,
            source_type="recorded",
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lecture_id = lec.id

    whisper_invoked = {"called": False}

    def fail_if_called(*args, **kwargs):
        whisper_invoked["called"] = True
        raise AssertionError("Whisper non doveva essere invocato su raw_transcript vuoto!")

    import ai_models
    monkeypatch.setattr(ai_models, "local_model_path", fail_if_called)
    monkeypatch.setattr(transcriber, "_iter_decoded_audio_windows", fail_if_called)

    result = lectures_router.compute_and_cache_lecture_timestamps(lecture_id)
    assert result == {"lines": [], "segments": [], "cached": False}
    assert whisper_invoked["called"] is False
    assert not os.path.exists(wav_path + ".timestamps.json")


# ── 7. generate-summary e generate-summary-stream: HTTP 400 se LLM assente ────

def test_generate_summary_endpoints_return_400_when_llm_not_downloaded(client, subject, monkeypatch):
    """Entrambi gli endpoint restituiscono HTTP 400 prima di caricare embedding o toccare processed_notes."""
    with SessionLocal() as db:
        lec = Lecture(
            subject_id=subject["id"],
            title="Lezione Senza LLM",
            raw_transcript="Spiegazione completa del metodo di eliminazione di Gauss per sistemi lineari.",
            processed_notes=None,
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lecture_id = lec.id

    rag_called = {"called": False}

    def fail_query_context(**kwargs):
        rag_called["called"] = True
        raise AssertionError("query_context / embeddings non devono essere caricati se l'LLM non è scaricato!")

    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *a, **kw: False)
    monkeypatch.setattr(llm_engine, "resolve_llm_model", lambda m=None: m or "qwen3.5-4b")
    monkeypatch.setattr(rag, "query_context", fail_query_context)

    # 1. Endpoint sincrono /generate-summary
    r1 = client.post(f"/api/v1/lectures/{lecture_id}/generate-summary")
    assert r1.status_code == 400, r1.text
    assert "non scaricato" in r1.json()["detail"].lower()
    assert rag_called["called"] is False

    # 2. Endpoint streaming /generate-summary-stream
    r2 = client.post(f"/api/v1/lectures/{lecture_id}/generate-summary-stream")
    assert r2.status_code == 400, r2.text
    assert "non scaricato" in r2.json()["detail"].lower()
    assert rag_called["called"] is False

    # processed_notes deve restare intatto (None) nel database
    with SessionLocal() as db:
        refreshed = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        assert refreshed is not None
        assert refreshed.processed_notes is None


# ── 8. rag.query_context_structured(..., use_vector=False) ────────────────────

def test_rag_query_context_structured_lexical_only_never_loads_embeddings(monkeypatch):
    """use_vector=False esegue ricerca lessicale in LanceDB lasciando is_embedding_loaded() == False."""
    rag.unload_embedding_model()
    assert rag.is_embedding_loaded() is False

    def fail_load_embed(*args, **kwargs):
        raise AssertionError("_get_embedding_model non deve mai essere chiamato con use_vector=False!")

    monkeypatch.setattr(rag, "_get_embedding_model", fail_load_embed)

    test_subject_id = 98765
    tbl = rag.get_lancedb_table()
    tbl.add([
        {
            "id": "unit-test-lexical-1",
            "vector": [0.0] * rag.EMBEDDING_DIM,
            "subject_id": test_subject_id,
            "lecture_id": 77,
            "source_type": "lecture",
            "source_id": "lecture_77",
            "title": "Lezione Termodinamica",
            "text": "Il secondo principio della termodinamica definisce l'entropia di un sistema isolato.",
            "page_start": 0,
            "page_end": 0,
            "chunk_label": "",
            "created_at": time.time(),
        }
    ])

    try:
        res = rag.query_context_structured(
            subject_id=test_subject_id,
            query="Che cos'è l'entropia nella termodinamica?",
            top_k=3,
            use_vector=False,
        )
        assert rag.is_embedding_loaded() is False
        assert "entropia" in res["context_text"].lower()
        assert len(res["citations"]) == 1
        assert res["citations"][0]["title"] == "Lezione Termodinamica"
    finally:
        rag.delete_subject_vectors(test_subject_id)


# ── 9. Fix backend/main.py: WS answer_question sconosciuta & disconnect audio ─

def test_ws_answer_question_unknown_id_sends_error_frame(client, subject):
    """Se q_id non corrisponde a nessuna domanda in sessione, il server invia un frame error con lo stesso id."""
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        _recv_until(ws, "started")

        ws.send_text(json.dumps({
            "type": "answer_question",
            "payload": {"question_id": "q-deleted-999"},
        }))
        err = _recv_until(ws, "error")
        assert err["payload"]["id"] == "q-deleted-999"
        assert err["payload"]["message"] == "Domanda non trovata nella sessione corrente."

        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")


def test_ws_disconnect_preserves_last_audio_in_speech_queue(client, subject, monkeypatch):
    """Su disconnessione WS improvvisa, l'ultimo chunk vocale restituito da session.pause() finisce in speech_queue."""
    fake_tail_audio = b"\x01\x02" * 2400  # 4800 bytes (>= 1600)
    captured_session: dict[str, LiveSession] = {}

    orig_start_session = main.start_session

    def wrapped_start_session(*args, **kwargs):
        sess = orig_start_session(*args, **kwargs)
        orig_pause = sess.pause

        def pause_with_tail():
            orig_pause()
            return fake_tail_audio

        sess.pause = pause_with_tail
        captured_session["sess"] = sess
        return sess

    monkeypatch.setattr(main, "start_session", wrapped_start_session)

    try:
        with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
            ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
            _recv_until(ws, "started")
            # Chiusura senza inviare 'stop' -> attiva il blocco finally di disconnessione

        sess = captured_session["sess"]
        assert sess.is_paused is True
        assert not sess.audio_engine.speech_queue.empty()
        queued = sess.audio_engine.speech_queue.get_nowait()
        assert queued == fake_tail_audio
    finally:
        sess = captured_session.get("sess")
        if sess is not None:
            grace = getattr(sess, "_disconnect_grace_task", None)
            if grace and not grace.done():
                grace.cancel()
            live_session.end_session(subject["id"])
