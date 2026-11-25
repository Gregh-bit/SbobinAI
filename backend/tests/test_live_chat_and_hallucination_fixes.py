"""
Test per le correzioni di backend:
1. Concorrenza Whisper + Chat Live via WebSocket e gestione sicura VRAM in llm_engine.
2. Filtro anti-allucinazione Whisper ('grazie a tutti' x30, silenzi, deduplica chunk consecutivi).
3. Endpoint utility salvataggio file POST /api/v1/utils/save-file.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import numpy as np
import pytest

import live_session
import llm_engine
import main
import transcriber
from transcript_cleaner import clean_transcript_for_llm


# ── 1. Protezione Sicurezza Endpoint di Sistema ────────────────────────────────

def test_system_shutdown_blocks_cross_origin_requests(client):
    """Verifica che /api/v1/system/shutdown rifiuti richieste cross-site esterne con HTTP 403."""
    res = client.post("/api/v1/system/shutdown", headers={"Origin": "https://malicious-website.com"})
    assert res.status_code == 403
    assert "Origine non consentita" in res.json().get("detail", "")


# ── 2. Filtro Anti-Allucinazione Transcriber & Cleaner ──────────────────────────

def test_clean_transcript_discards_closing_phrases_and_loops():
    # Formule di chiusura isolate tipiche del silenzio
    assert transcriber.Transcriber._clean_transcript("Grazie a tutti.") == ""
    assert transcriber.Transcriber._clean_transcript("Grazie per la visione!") == ""
    assert transcriber.Transcriber._clean_transcript("Grazie dell'attenzione, arrivederci.") == ""
    assert transcriber.Transcriber._clean_transcript("sottotitoli a cura di") == ""
    assert transcriber.Transcriber._clean_transcript("iscriviti al canale") == ""
    assert transcriber.Transcriber._clean_transcript("Arrivederci.") == ""

    # Loop allucinatorio ripetuto di formule di chiusura
    assert transcriber.Transcriber._clean_transcript("Grazie a tutti. Grazie a tutti. Grazie a tutti.") == ""
    assert transcriber.Transcriber._clean_transcript("Grazie per la visione. Grazie per la visione.") == ""

    # Trascrizione accademica valida non deve essere scartata
    valid_text = "Oggi analizzeremo la complessità asintotica e la convergenza degli algoritmi distribuiti."
    assert transcriber.Transcriber._clean_transcript(valid_text) == valid_text


def test_push_transcript_deduplicates_consecutive_identical_chunks():
    session = live_session.LiveSession(
        subject_id=1,
        lecture_id=1,
        mock_transcriber=True,
    )
    chunk1 = "Introduzione alla trasformata di Fourier."
    session.push_transcript(chunk1)
    assert len(session.full_transcript) == 1
    assert session.full_transcript[0] == chunk1

    # Chunk duplicato consecutivo (anche con case diverso) deve essere scartato
    session.push_transcript("introduzione alla trasformata di fourier.")
    assert len(session.full_transcript) == 1

    # Chunk successivo diverso viene registrato
    chunk2 = "Vediamo ora le proprietà di linearità."
    session.push_transcript(chunk2)
    assert len(session.full_transcript) == 2
    assert session.full_transcript[1] == chunk2


def test_is_speech_spectral_rejects_sub_rms_noise():
    # Audio con RMS 0.0005 (lieve fruscio microfonico costante) deve essere scartato
    t = np.arange(16000, dtype=np.float32) / 16000.0
    # Ampiezza tale che RMS sia ~0.0006 < 0.001
    hiss = (np.sin(2 * np.pi * 1000.0 * t) * 0.00085).astype(np.float32)
    assert transcriber.is_speech_spectral(hiss) is False

    # Audio con voce chiara (RMS ~0.05 > 0.001, frequenze 300Hz in banda vocale)
    speech = (np.sin(2 * np.pi * 300.0 * t) * 0.1).astype(np.float32)
    assert transcriber.is_speech_spectral(speech) is True


def test_faster_whisper_word_segments_passes_anti_hallucination_kwargs(monkeypatch):
    captured_kwargs = {}

    def fake_call(model_path, audio, **kwargs):
        captured_kwargs.update(kwargs)
        return [], 0.0

    monkeypatch.setattr(transcriber, "_call_faster_whisper_with_cpu_fallback", fake_call)
    transcriber.faster_whisper_word_segments("/fake/path", np.zeros(1600, dtype=np.float32))

    assert captured_kwargs.get("vad_filter") is True
    assert captured_kwargs.get("condition_on_previous_text") is False
    assert captured_kwargs.get("temperature") == (0.0,)
    assert captured_kwargs.get("compression_ratio_threshold") == 1.8
    assert captured_kwargs.get("no_speech_threshold") == 0.5


def test_transcript_cleaner_removes_grazie_a_tutti_boilerplate_and_loops():
    text = "Allora, diciamo, oggi parliamo del teorema spettrale. Grazie a tutti! Grazie a tutti!"
    cleaned = clean_transcript_for_llm(text)
    assert "grazie a tutti" not in cleaned.lower()
    assert "teorema spettrale" in cleaned


# ── 3. WebSocket Live Chat Concurrency & Error Resilience ──────────────────────

def _recv_until(ws, wanted: str, limit: int = 50):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == wanted:
            return msg
    raise AssertionError(f"messaggio {wanted} non ricevuto")


def test_ws_live_chat_streaming_concurrency(client, subject, monkeypatch):
    """Verifica che la chat live invii chat_meta, chat_chunk e chat_response senza bloccare il WS né crashare."""
    async def fake_stream_chat_response(query, context, model_id=None, **kwargs):
        yield "Risposta "
        yield "di "
        yield "test."

    monkeypatch.setattr("llm_engine.stream_chat_response", fake_stream_chat_response)
    monkeypatch.setattr("llm_engine.is_model_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(
        "rag.query_context_structured",
        lambda **kw: {"context_text": "Contesto RAG fittizio", "citations": []},
    )

    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        _recv_until(ws, "started")

        # Invia messaggio di chat
        ws.send_text(json.dumps({
            "type": "chat",
            "payload": {"query": "Cos'è la trasformata?", "context_minutes": 5}
        }))

        # Riceve i chunk e la risposta finale
        chunk_msg = _recv_until(ws, "chat_chunk")
        assert "chunk" in chunk_msg["payload"]

        resp_msg = _recv_until(ws, "chat_response")
        assert "Risposta di test." in resp_msg["payload"]["answer"]

        # Verifica che il WebSocket sia ancora perfettamente attivo e risponda ad altri comandi
        ws.send_text(json.dumps({"type": "bookmark", "payload": {"note": "test"}}))
        bm = _recv_until(ws, "bookmark_saved")
        assert bm["payload"]["note"] == "test"

        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")


def test_ws_live_chat_error_does_not_kill_websocket(client, subject, monkeypatch):
    """Se lo streaming chat genera un'eccezione, il WS non deve crashare ma inviare errore amichevole."""
    async def failing_stream_chat_response(query, context, model_id=None):
        raise RuntimeError("GPU OOM simulato durante inferenza")
        yield "non arriva mai"

    monkeypatch.setattr("llm_engine.stream_chat_response", failing_stream_chat_response)
    monkeypatch.setattr("llm_engine.is_model_downloaded", lambda *a, **k: True)
    monkeypatch.setattr("rag.query_context", lambda **kw: "Contesto")

    with client.websocket_connect(f"/ws/live-lecture/{subject['id']}") as ws:
        ws.send_text(json.dumps({"type": "start", "payload": {"model": "mock_mode"}}))
        _recv_until(ws, "started")

        ws.send_text(json.dumps({"type": "chat", "payload": {"query": "Domanda"}}))

        resp_msg = _recv_until(ws, "chat_response")
        assert "Non è stato possibile completare la risposta" in resp_msg["payload"]["answer"] or "GPU OOM" in resp_msg["payload"]["answer"]

        # Il socket è ancora vivo: invia un chunk e stoppa regolarmente
        ws.send_text(json.dumps({"type": "chunk", "payload": {"text": "Trascrizione continua."}}))
        _recv_until(ws, "transcript_chunk")

        ws.send_text(json.dumps({"type": "stop"}))
        _recv_until(ws, "stopped")


# ── 4. Gestione Memoria llm_engine con Sessione Live Attiva ────────────────────

def test_llm_engine_fallback_cpu_when_live_session_active(monkeypatch):
    """Verifica che durante una sessione live attiva, il reload sicuro passi a force_cpu=True."""
    seen_calls = []

    def fake_load_llama(path, *, safe_only=False, force_cpu=False):
        seen_calls.append({"safe_only": safe_only, "force_cpu": force_cpu})
        class Dummy:
            def token_bos(self): return 1
            def token_eos(self): return 2
            def detokenize(self, t): return b""
            def close(self): pass
        d = Dummy()
        return d, llm_engine._GGUFChatTokenizer(d)

    monkeypatch.setattr(llm_engine, "USE_MLX", False)
    monkeypatch.setattr(llm_engine, "_load_llama", fake_load_llama)
    monkeypatch.setattr(llm_engine, "_live_session_active", lambda: True)

    dummy_model = object()
    monkeypatch.setattr(llm_engine, "_loaded_model", dummy_model)
    monkeypatch.setattr(llm_engine, "_loaded_model_path", "/fake/path.gguf")
    monkeypatch.setattr(llm_engine, "_loaded_model_id", "test-id")
    monkeypatch.setattr(llm_engine, "_loaded_model_cfg", (-1, True, True))

    reloaded = llm_engine._reload_llama_safe(dummy_model)
    assert reloaded is not None
    assert len(seen_calls) == 1
    # Con live session attiva, force_cpu deve essere True per prevenire contesa VRAM
    assert seen_calls[0]["force_cpu"] is True
