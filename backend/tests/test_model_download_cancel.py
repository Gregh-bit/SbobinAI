"""
Test di regressione: cancellazione download modelli e invariante download singolo.
Verifica che annullare il download di un modello:
1. Resetti immediatamente lo stato del tracker a 'idle' senza rimanere bloccato in 'downloading'.
2. /models/download-status risponda immediatamente con 'status': 'idle'.
3. L'avvio di un secondo download cancelli automaticamente e in modo pulito il precedente (Single-Active-Download).
"""
from __future__ import annotations

import threading
import time

import ai_models
from routers import models as models_router


def test_cancel_download_immediately_resets_to_idle(client, monkeypatch):
    """Annullare un download deve riportare immediatamente lo stato a idle."""
    # Simula un worker di download che attende un cancel_event
    started_event = threading.Event()
    worker_finished = threading.Event()

    def fake_download_hf_worker(model_id, spec, cancel_event):
        started_event.set()
        while not cancel_event.is_set():
            time.sleep(0.02)
        models_router._finish_cancelled(model_id, spec, 100.0)
        worker_finished.set()

    monkeypatch.setattr(models_router, "_download_hf_worker", fake_download_hf_worker)
    monkeypatch.setattr(models_router, "_delete_repos", lambda *a, **kw: 0.0)

    # 1. Avvia download del modello gemma
    res = client.post("/api/v1/models/download", json={"model_id": "gemma"})
    assert res.status_code == 200
    assert res.json()["status"] == "downloading"
    assert started_event.wait(timeout=2.0)

    # Verifica stato downloading
    status_res = client.get("/api/v1/models/download-status?model_id=gemma")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "downloading"

    # 2. Interrompi download
    cancel_res = client.post("/api/v1/models/cancel-download", json={"model_id": "gemma"})
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"

    # 3. Lo stato deve essere IMMEDIATAMENTE idle sia su download-status che su /status
    dl_status_after = client.get("/api/v1/models/download-status?model_id=gemma")
    assert dl_status_after.status_code == 200
    assert dl_status_after.json()["status"] == "idle"

    models_status = client.get("/api/v1/models/status").json()
    llm_items = [m for m in models_status["llm"] if "gemma" in m["id"].lower()]
    assert len(llm_items) == 1
    assert llm_items[0]["status"] == "idle"
    assert llm_items[0]["progress"] == 0


def test_single_active_download_invariant_cancels_previous(client, monkeypatch):
    """Avviare il download di un secondo modello deve interrompere in sicurezza il primo."""
    gemma_cancelled = threading.Event()
    qwen_started = threading.Event()

    def fake_download_hf_worker(model_id, spec, cancel_event):
        mid = model_id.lower()
        if "gemma" in mid:
            while not cancel_event.is_set():
                time.sleep(0.02)
            gemma_cancelled.set()
            models_router._finish_cancelled(model_id, spec, 100.0)
        elif "qwen" in mid:
            qwen_started.set()
            while not cancel_event.is_set():
                time.sleep(0.02)
            models_router._finish_cancelled(model_id, spec, 100.0)

    monkeypatch.setattr(models_router, "_download_hf_worker", fake_download_hf_worker)
    monkeypatch.setattr(models_router, "_delete_repos", lambda *a, **kw: 0.0)

    # 1. Avvia download di Gemma
    res1 = client.post("/api/v1/models/download", json={"model_id": "gemma"})
    assert res1.status_code == 200
    assert res1.json()["status"] == "downloading"

    # 2. Subito dopo, avvia download di Qwen (senza aspettare la cancellazione manuale)
    res2 = client.post("/api/v1/models/download", json={"model_id": "qwen"})
    assert res2.status_code == 200
    assert res2.json()["status"] == "downloading"

    # Gemma deve aver ricevuto il segnale di cancellazione ed essere interrotta
    assert gemma_cancelled.wait(timeout=2.0)
    assert qwen_started.wait(timeout=2.0)

    # Lo stato di Gemma deve essere idle, e Qwen deve essere downloading
    status_gemma = client.get("/api/v1/models/download-status?model_id=gemma").json()
    assert status_gemma["status"] == "idle"

    status_qwen = client.get("/api/v1/models/download-status?model_id=qwen").json()
    assert status_qwen["status"] == "downloading"

    # Pulizia finale
    client.post("/api/v1/models/cancel-download", json={"model_id": "all"})
