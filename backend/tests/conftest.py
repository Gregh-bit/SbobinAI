"""
Configurazione dei test: ogni sessione di test usa un settings.json e uno storage
temporanei, così i test non toccano mai i dati reali dell'utente.
I motori AI (Whisper, LLM, embedding) non vengono caricati: le parti che li usano
sono sostituite con stub tramite monkeypatch.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

_TMP_ROOT = tempfile.mkdtemp(prefix="sbobinai-tests-")
_STORAGE = os.path.join(_TMP_ROOT, "storage")
_SETTINGS = os.path.join(_TMP_ROOT, "settings.json")

with open(_SETTINGS, "w", encoding="utf-8") as f:
    json.dump({"storage_path": _STORAGE, "whisper_model": "mock_mode"}, f)

# Deve essere impostato prima di importare config/database/main
os.environ["SBOBINAI_SETTINGS_FILE"] = _SETTINGS


@pytest.fixture(scope="session")
def tmp_root() -> str:
    return _TMP_ROOT


@pytest.fixture(scope="session")
def app():
    import main
    return main.app


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_db():
    """Svuota le tabelle tra un test e l'altro."""
    yield
    import database
    with database.engine.begin() as conn:
        for table in reversed(database.Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture()
def no_rag(monkeypatch):
    """Sostituisce indicizzazione e ricerca vettoriale con stub (nessun modello di embedding)."""
    calls = {"index_material": [], "index_lecture": [], "delete_material": []}

    def fake_index_material(**kw):
        calls["index_material"].append(kw)
        return 1

    def fake_index_lecture(**kw):
        calls["index_lecture"].append(kw)
        return 1

    import rag
    import routers.subjects as subjects

    monkeypatch.setattr(subjects, "index_material_document", fake_index_material)
    monkeypatch.setattr(subjects, "delete_material_vectors", lambda **kw: calls["delete_material"].append(kw))
    monkeypatch.setattr(subjects, "delete_subject_vectors", lambda **kw: None)
    monkeypatch.setattr(subjects, "compact_lancedb", lambda *a, **kw: None)
    monkeypatch.setattr(rag, "index_lecture", fake_index_lecture)
    monkeypatch.setattr(rag, "index_material_document", fake_index_material)
    monkeypatch.setattr(rag, "query_context", lambda **kw: "")
    monkeypatch.setattr(rag, "unload_embedding_model", lambda *a, **kw: None)
    import routers.lectures as lectures
    monkeypatch.setattr(lectures, "delete_lecture_vectors", lambda **kw: None)
    return calls


@pytest.fixture()
def subject(client):
    r = client.post("/api/v1/subjects/", json={"name": "Analisi II", "description": "desc"})
    assert r.status_code == 201, r.text
    return r.json()
