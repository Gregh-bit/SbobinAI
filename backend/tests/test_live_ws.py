"""Test end-to-end del WebSocket di registrazione live in mock mode (nessun modello reale)."""
from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

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
        if msg["type"] == wanted:
            return msg
    raise AssertionError(f"messaggio {wanted} non ricevuto")


def _lecture(client, lecture_id):
    return client.get(f"/api/v1/lectures/{lecture_id}").json()


def test_ws_rejects_foreign_origin(client, subject):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            f"/ws/live-lecture/{subject['id']}", headers={"origin": "https://evil.example"}
        ) as ws:
            ws.receive_json()


def test_ws_accepts_tauri_origins(client, subject):
    for origin in ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://localhost:5173"):
        with client.websocket_connect(f"/ws/live-lecture/{subject['id']}", headers={"origin": origin}) as ws:
            ws.send_text(json.dumps({"type": "unknown"}))
            assert ws.receive_json()["type"] == "error"


def test_ws_invalid_messages_do_not_crash(client, subject):
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text("non json")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps([1, 2, 3]))
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "chunk", "payload": "stringa"}))
        assert ws.receive_json()["type"] == "error"  # sessione non avviata
        ws.send_text(json.dumps({"type": "start", "payload": {"lecture_id": "abc"}}))
        assert ws.receive_json()["type"] == "error"


def test_ws_rejects_lecture_of_other_subject(client, subject):
    other = client.post("/api/v1/subjects/", json={"name": "Altra"}).json()
    lec = client.post("/api/v1/lectures/", json={"subject_id": other["id"], "raw_transcript": "intatto"}).json()
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"lecture_id": lec["id"], "model": "mock_mode"}}))
        assert ws.receive_json()["type"] == "error"
    assert _lecture(client, lec["id"])["raw_transcript"] == "intatto"


def test_ws_mock_session_start_chunk_bookmark_stop(client, subject, post_calls):
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({
            "type": "start",
            "payload": {"title": "Lezione 1", "model": "mock_mode", "vad_sensitivity": "alto"},
        }))
        started = _recv_until(ws, "started")
        lecture_id = started["payload"]["lecture_id"]

        ws.send_text(json.dumps({"type": "chunk", "payload": {"text": "Oggi parliamo di integrali doppi e del teorema di Fubini."}}))
        _recv_until(ws, "transcript_chunk")

        ws.send_text(json.dumps({"type": "bookmark", "payload": {"note": "importante"}}))
        _recv_until(ws, "bookmark_saved")

        ws.send_text(json.dumps({"type": "stop", "payload": {}}))
        stopped = _recv_until(ws, "stopped")
        assert stopped["payload"]["total_chunks"] >= 1

    lec = _lecture(client, lecture_id)
    assert "Fubini" in lec["raw_transcript"]
    assert lec["duration_seconds"] is not None
    assert live_session.get_session(subject["id"]) is None
    assert post_calls and post_calls[0]["lecture_id"] == lecture_id


def test_ws_disconnect_without_stop_saves_and_releases(client, subject, post_calls):
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        lecture_id = _recv_until(ws, "started")["payload"]["lecture_id"]
        ws.send_text(json.dumps({"type": "chunk", "payload": {"text": "Testo salvato anche senza stop esplicito."}}))
        _recv_until(ws, "transcript_chunk")
    # uscita dal context manager = disconnessione del client
    import time
    for _ in range(50):
        if live_session.get_session(subject["id"]) is None and _lecture(client, lecture_id)["raw_transcript"]:
            break
        time.sleep(0.1)
    assert live_session.get_session(subject["id"]) is None
    assert "senza stop" in _lecture(client, lecture_id)["raw_transcript"]


def test_ws_second_start_persists_previous_session(client, subject, post_calls):
    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        first_id = _recv_until(ws, "started")["payload"]["lecture_id"]
        ws.send_text(json.dumps({"type": "chunk", "payload": {"text": "Prima lezione, non deve andare persa."}}))
        _recv_until(ws, "transcript_chunk")

        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        second_id = _recv_until(ws, "started")["payload"]["lecture_id"]
        assert second_id != first_id
        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")

    assert "Prima lezione" in _lecture(client, first_id)["raw_transcript"]
