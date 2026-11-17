"""
Test dedicati per le correzioni del backend:
1. Domande docente: nessuna risposta automatica durante la registrazione (_announce_questions).
2. Risposta docente su richiesta esplicita via WebSocket (answer_question -> question_answering -> flash answer).
3. Pausa immediata non bloccante e ripresa (ACK immediato, background transcription, ACK idempotente).
4. Riconnessione WebSocket su sessione esistente (con start o reconnect).
5. AudioEngine resume con recupero stream hardware dopo OS sleep.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

import audio_engine
import live_session
import main


@pytest.fixture()
def post_calls(monkeypatch):
    """Evita la generazione del capitolo (LLM) a fine lezione, registrando le chiamate."""
    calls = []

    async def fake_post(**kw):
        calls.append(kw)

    monkeypatch.setattr(main, "_post_lecture_summary_and_index", fake_post)
    return calls


def _recv_until(ws, wanted: str, limit: int = 50):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == wanted:
            return msg
    raise AssertionError(f"Messaggio '{wanted}' non ricevuto entro {limit} letture")


# ── 1. Nessuna risposta automatica su _announce_questions ─────────────────────

@pytest.mark.asyncio
async def test_announce_questions_no_automatic_llm(monkeypatch):
    """Verifica che _announce_questions rilevi la domanda ma NON avvii l'LLM, impostando suggested_answer vuota."""
    flash_called = []

    async def fake_flash(session, question, question_id, ws):
        flash_called.append((question, question_id))

    monkeypatch.setattr(main, "_generate_flash_answer_task", fake_flash)

    session = live_session.LiveSession(subject_id=99, lecture_id=123, mock_transcriber=True)

    sent_messages = []

    class DummyWS:
        async def send_json(self, payload):
            sent_messages.append(payload)

    dummy_ws = DummyWS()
    dummy_ws.client_state = main.WebSocketState.CONNECTED

    teacher_text = "Quindi ragazzi, quali sono le proprietà dei gas ideali secondo la termodinamica classica?"

    await main._announce_questions(session, teacher_text, dummy_ws)

    # 1. Nessun task LLM deve essere stato avviato
    assert len(flash_called) == 0, "L'LLM flash answer NON deve essere avviato automaticamente!"

    # 2. La domanda registrata nella sessione deve avere risposta vuota
    assert len(session.detected_questions) == 1
    q = session.detected_questions[0]
    assert "proprietà dei gas" in q["question"]
    assert q["suggestedAnswer"] == ""

    # 3. Il messaggio inviato via WS deve contenere suggested_answer vuota
    assert len(sent_messages) == 1
    assert sent_messages[0]["type"] == "question_detected"
    assert sent_messages[0]["payload"]["suggested_answer"] == ""
    assert sent_messages[0]["payload"]["id"] == q["id"]


# ── 2. Risposta generata su richiesta esplicita answer_question ────────────────

def test_ws_answer_question_generates_flash_answer(client, subject, post_calls, monkeypatch):
    """Verifica che inviando answer_question il backend invii question_answering e poi question_detected con risposta."""
    async def fake_flash(session, question, question_id, ws):
        for q_item in session.detected_questions:
            if str(q_item.get("id")) == str(question_id):
                q_item["suggestedAnswer"] = "1. Pressione proporzionale a T.\n2. Volume trascurabile.\n3. Urti elastici."
                break
        await main._safe_send_json(ws, {
            "type": "question_detected",
            "payload": {
                "id": question_id,
                "question": question,
                "confidence": 0.98,
                "suggested_answer": "1. Pressione proporzionale a T.\n2. Volume trascurabile.\n3. Urti elastici.",
                "trigger": "TRIGGER_RAG",
            }
        })

    monkeypatch.setattr(main, "_generate_flash_answer_task", fake_flash)

    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        _recv_until(ws, "started")

        sess = live_session.get_session(subject["id"])
        assert sess is not None
        # Simula domanda rilevata senza risposta iniziale (come fa _announce_questions)
        q_obj = sess.add_question("Quali sono le caratteristiche principali dei gas nobili?", suggested_answer="")
        q_id = q_obj["id"]

        # Ora l'utente clicca su "Rispondi" -> invia answer_question
        ws.send_text(json.dumps({
            "type": "answer_question",
            "payload": {"question_id": q_id}
        }))

        # Riceve immediatamente question_answering per mostrare il loader
        answering_msg = _recv_until(ws, "question_answering")
        assert answering_msg["payload"]["id"] == q_id

        # Riceve l'aggiornamento definitivo con la risposta flash
        updated_q_msg = _recv_until(ws, "question_detected")
        assert updated_q_msg["payload"]["id"] == q_id
        assert "Pressione proporzionale" in updated_q_msg["payload"]["suggested_answer"]

        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")


# ── 3. Pausa immediata non bloccante e ripresa ─────────────────────────────────

def test_ws_pause_immediate_and_resume(client, subject, post_calls):
    """Verifica che 'pause' risponda SUBITO con 'paused', che sia idempotente, e che 'resume' riprenda."""
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        _recv_until(ws, "started")

        # Invia pausa
        t0 = time.time()
        ws.send_text(json.dumps({"type": "pause", "payload": {}}))
        paused_msg = _recv_until(ws, "paused")
        t_paused = time.time() - t0
        assert paused_msg["type"] == "paused"
        assert t_paused < 0.5, f"Pausa non è stata immediata: ha impiegato {t_paused:.2f}s"

        sess = live_session.get_session(subject["id"])
        assert sess is not None
        assert sess.is_paused is True

        # Invia una seconda pausa: deve comunque ricevere ACK "paused" senza errori
        ws.send_text(json.dumps({"type": "pause", "payload": {}}))
        ack_paused = _recv_until(ws, "paused")
        assert ack_paused["type"] == "paused"

        # Invia resume: deve ricevere "resumed"
        ws.send_text(json.dumps({"type": "resume", "payload": {}}))
        resumed_msg = _recv_until(ws, "resumed")
        assert resumed_msg["type"] == "resumed"
        assert sess.is_paused is False

        # Invia un secondo resume: deve comunque ricevere ACK "resumed"
        ws.send_text(json.dumps({"type": "resume", "payload": {}}))
        ack_resumed = _recv_until(ws, "resumed")
        assert ack_resumed["type"] == "resumed"

        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")


# ── 4. Riconnessione su sessione esistente ─────────────────────────────────────

def test_ws_reconnect_existing_session_via_start(client, subject, post_calls):
    """
    Simula chiusura schermo / disconnessione WS improvvisa:
    - Client 1 si disconnette senza 'stop'
    - La sessione va in pausa e non viene finalizzata subito
    - Client 2 si riconnette con 'start' (stessa lecture_id) e riceve 'reconnected' con chunk e stato
    """
    # 1. Connessione iniziale
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws1:
        ws1.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        started = _recv_until(ws1, "started")
        lecture_id = started["payload"]["lecture_id"]

        ws1.send_text(json.dumps({"type": "chunk", "payload": {"text": "Frammento prima dello sleep."}}))
        _recv_until(ws1, "transcript_chunk")
        # ws1 si chiude qui (simula sleep PC o interruzione rete)

    # 2. Verifica che la sessione sia rimasta attiva nel registry e sia in pausa
    active = live_session.get_session(subject["id"])
    assert active is not None
    assert active.lecture_id == lecture_id
    assert active.is_active is True
    assert active.is_paused is True
    assert active._client_disconnected_at is not None

    # 3. Riconnessione da nuovo WebSocket (es. riapertura schermo)
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws2:
        ws2.send_text(json.dumps({
            "type": "start",
            "payload": {
                "lecture_id": lecture_id,
                "model": "mock_mode",
            }
        }))

        # Deve ricevere "reconnected" con lo storico e lo stato
        rec_msg = _recv_until(ws2, "reconnected")
        payload = rec_msg["payload"]
        assert payload["lecture_id"] == lecture_id
        assert payload["is_paused"] is True
        assert len(payload["chunks"]) >= 1
        assert "Frammento prima dello sleep" in payload["chunks"][0]

        # Riprende la lezione
        ws2.send_text(json.dumps({"type": "resume"}))
        _recv_until(ws2, "resumed")

        # Invia un altro frammento dopo la riconnessione
        ws2.send_text(json.dumps({"type": "chunk", "payload": {"text": "Frammento dopo la riconnessione."}}))
        _recv_until(ws2, "transcript_chunk")

        # Chiude regolarmente con stop
        ws2.send_text(json.dumps({"type": "stop"}))
        stopped = _recv_until(ws2, "stopped")
        assert stopped["payload"]["total_chunks"] >= 2

    # A questo punto la sessione deve essere definitivamente chiusa
    assert live_session.get_session(subject["id"]) is None


def test_ws_reconnect_message_type(client, subject, post_calls):
    """Verifica la riconnessione esplicita con il messaggio type='reconnect'."""
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws1:
        ws1.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        started = _recv_until(ws1, "started")
        lecture_id = started["payload"]["lecture_id"]

        ws1.send_text(json.dumps({"type": "chunk", "payload": {"text": "Trascrizione iniziale."}}))
        _recv_until(ws1, "transcript_chunk")

    # Client 2 si riconnette con type="reconnect"
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws2:
        ws2.send_text(json.dumps({
            "type": "reconnect",
            "payload": {"lecture_id": lecture_id}
        }))

        rec_msg = _recv_until(ws2, "reconnected")
        assert rec_msg["payload"]["lecture_id"] == lecture_id

        ws2.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws2, "stopped")

    assert live_session.get_session(subject["id"]) is None


# ── 5. AudioEngine resume con recupero stream hardware ─────────────────────────

def test_audio_engine_resume_reopens_stream_on_failure(monkeypatch):
    """Verifica che AudioEngine.resume() chiuda lo stream corrotto e riapra un nuovo stream con self._open_stream()."""
    engine = audio_engine.AudioEngine(lecture_id=42)
    engine.is_running = True
    engine.is_paused = True

    # Simula uno stream vecchio il cui .start() fallisce (es. CoreAudio/WASAPI invalidato dallo sleep)
    old_stream = MagicMock()
    old_stream.start.side_effect = RuntimeError("Audio device invalidated by OS sleep")
    engine._stream = old_stream

    new_stream = MagicMock()
    open_called = []

    def fake_open_stream():
        open_called.append(True)
        return new_stream

    monkeypatch.setattr(engine, "_open_stream", fake_open_stream)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # velocizza i test

    success = engine.resume()

    assert success is True
    # Vecchio stream deve essere stato chiuso preventivamente
    old_stream.close.assert_called_once()
    # Deve essere stato aperto un nuovo stream e assegnato a self._stream
    assert len(open_called) >= 1
    assert engine._stream is new_stream
    new_stream.start.assert_called_once()
    assert engine.is_paused is False
