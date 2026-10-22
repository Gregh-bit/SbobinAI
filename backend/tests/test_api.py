from __future__ import annotations

import json
import os

import pytest

import config


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


# ── Subjects ───────────────────────────────────────────────────────────────────

def test_subject_crud_and_duplicate_name(client, subject):
    r = client.post("/api/v1/subjects/", json={"name": "Analisi II"})
    assert r.status_code == 409

    other = client.post("/api/v1/subjects/", json={"name": "Fisica"}).json()
    # rinominare su un nome esistente → 409 (prima: IntegrityError → 500)
    r = client.patch(f"/api/v1/subjects/{other['id']}", json={"name": "Analisi II"})
    assert r.status_code == 409

    # null su campi NOT NULL viene ignorato, non scritto
    r = client.patch(f"/api/v1/subjects/{other['id']}", json={"name": None, "color": None})
    assert r.status_code == 200
    assert r.json()["name"] == "Fisica"

    # null su description la svuota
    r = client.patch(f"/api/v1/subjects/{subject['id']}", json={"description": None})
    assert r.status_code == 200
    assert r.json()["description"] is None

    assert len(client.get("/api/v1/subjects/").json()) == 2


def test_delete_subject(client, subject, no_rag):
    r = client.delete(f"/api/v1/subjects/{subject['id']}")
    assert r.status_code == 204
    assert client.get(f"/api/v1/subjects/{subject['id']}").status_code == 404


# ── Lectures ───────────────────────────────────────────────────────────────────

def test_lecture_null_fields_do_not_break_responses(client, subject):
    r = client.post("/api/v1/lectures/", json={"subject_id": subject["id"], "title": "L1", "date": None})
    assert r.status_code == 201, r.text
    lec = r.json()
    assert lec["date"] is not None

    r = client.patch(
        f"/api/v1/lectures/{lec['id']}",
        json={"title": None, "date": None, "source_type": None, "in_quaderno": None, "processed_notes": "n"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "L1"
    assert r.json()["processed_notes"] == "n"

    r = client.get(f"/api/v1/lectures/by-subject/{subject['id']}")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_bookmarks(client, subject):
    lec = client.post("/api/v1/lectures/", json={"subject_id": subject["id"]}).json()
    r = client.post("/api/v1/bookmarks/", json={"lecture_id": lec["id"], "timestamp_seconds": 12.5, "note": "esame"})
    assert r.status_code == 201
    assert client.get(f"/api/v1/bookmarks/by-lecture/{lec['id']}").json()[0]["note"] == "esame"


def test_delete_lecture(client, subject, no_rag):
    lec = client.post("/api/v1/lectures/", json={"subject_id": subject["id"]}).json()
    assert client.delete(f"/api/v1/lectures/{lec['id']}").status_code == 204
    assert client.get(f"/api/v1/lectures/{lec['id']}").status_code == 404


# ── Materials upload ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("evil", ["../../evil.txt", "..\\..\\evil.txt", "/abs/evil.txt", "C:\\x\\evil.txt"])
def test_upload_path_traversal_is_neutralized(client, subject, no_rag, evil):
    r = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": (evil, b"contenuto del file", "text/plain")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "evil.txt"
    subj_dir = os.path.join(config.get_uploads_dir(), f"subject_{subject['id']}")
    assert os.path.isfile(os.path.join(subj_dir, "evil.txt"))
    # niente file scritti fuori dalla cartella della materia
    assert not os.path.exists(os.path.join(config.get_uploads_dir(), "evil.txt"))
    assert not os.path.exists(os.path.join(config.get_storage_path(), "evil.txt"))


@pytest.mark.parametrize("bad", ["InfoCorso.md", ".hidden", "..", "x.meta.json"])
def test_upload_reserved_names_rejected(client, subject, no_rag, bad):
    r = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": (bad, b"x", "text/plain")},
    )
    assert r.status_code == 400


def test_upload_text_indexes_and_lists(client, subject, no_rag):
    r = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": ("l'analisi.md", "# Titolo\nTesto completo".encode(), "text/markdown")},
    )
    assert r.status_code == 201
    assert no_rag["index_material"][0]["material_id"] == "l'analisi.md"
    names = [m["name"] for m in client.get(f"/api/v1/subjects/{subject['id']}/materials").json()]
    assert "InfoCorso.md" in names and "l'analisi.md" in names

    r = client.delete(f"/api/v1/subjects/{subject['id']}/materials/l'analisi.md")
    assert r.status_code == 204
    assert no_rag["delete_material"][0]["material_id"] == "l'analisi.md"


def test_delete_material_linked_to_lecture_is_refused(client, subject, no_rag):
    import database
    subj_dir = os.path.join(config.get_uploads_dir(), f"subject_{subject['id']}")
    os.makedirs(subj_dir, exist_ok=True)
    audio = os.path.join(subj_dir, "lezione.mp3")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)
    with database.SessionLocal() as s:
        s.add(database.Lecture(subject_id=subject["id"], title="Imp", audio_path=os.path.abspath(audio)))
        s.commit()
    r = client.delete(f"/api/v1/subjects/{subject['id']}/materials/lezione.mp3")
    assert r.status_code == 409
    assert os.path.exists(audio)


def test_delete_material_traversal_rejected(client, subject, no_rag):
    r = client.delete(f"/api/v1/subjects/{subject['id']}/materials/..%2F..%2Fsettings.json")
    assert r.status_code in (204, 400, 404)
    assert os.path.exists(config.SETTINGS_FILE)


# ── Settings ───────────────────────────────────────────────────────────────────

def test_settings_patch_with_storage_path_persists_new_path(client, tmp_path):
    """Regressione: PATCH con storage_path riscriveva il vecchio percorso in settings.json."""
    old_root = config.get_storage_path()
    new_root = str(tmp_path / "altro_storage")
    try:
        r = client.patch("/api/v1/settings/", json={"storage_path": new_root, "vad_sensitivity": 1})
        assert r.status_code == 200
        assert config._same_path(r.json()["storage_path"], new_root)
        on_disk = json.load(open(config.SETTINGS_FILE, encoding="utf-8"))
        assert config._same_path(on_disk["storage_path"], new_root)
        assert on_disk["vad_sensitivity"] == 1
    finally:
        config.update_storage_path(old_root, migrate_data=False)


def test_settings_patch_ignores_nulls(client):
    client.patch("/api/v1/settings/", json={"llm_model": "modello-a"})
    r = client.patch("/api/v1/settings/", json={"llm_model": None})
    assert r.json()["llm_model"] == "modello-a"


def test_settings_get_does_not_rewrite_file(client):
    client.get("/api/v1/settings/")
    mtime = os.path.getmtime(config.SETTINGS_FILE)
    client.get("/api/v1/settings/")
    assert os.path.getmtime(config.SETTINGS_FILE) == mtime


def test_settings_audio_format_toggle(client):
    """Verifica che il formato di salvataggio audio sia modificabile e persistito."""
    r = client.patch("/api/v1/settings/", json={"audio_format": "wav"})
    assert r.status_code == 200
    assert r.json()["audio_format"] == "wav"

    cur = client.get("/api/v1/settings/").json()
    assert cur["audio_format"] == "wav"

    r2 = client.patch("/api/v1/settings/", json={"audio_format": "m4a"})
    assert r2.status_code == 200
    assert r2.json()["audio_format"] == "m4a"


def test_compress_archive_endpoint(client, subject):
    """Verifica l'endpoint POST /settings/compress-audio per la compressione su richiesta."""
    import wave
    import numpy as np
    import database
    from config import get_audio_dir

    audio_dir = get_audio_dir()
    os.makedirs(audio_dir, exist_ok=True)
    test_wav = os.path.join(audio_dir, "lecture_archive_test.wav")
    with wave.open(test_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000, False)) * 10000).astype(np.int16)
        wf.writeframes(samples.tobytes())

    with database.SessionLocal() as s:
        lec = database.Lecture(subject_id=subject["id"], title="Test Compressione", audio_path=test_wav)
        s.add(lec)
        s.commit()
        lec_id = lec.id

    r = client.post("/api/v1/settings/compress-audio")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["converted_count"] >= 1
    assert data["freed_mb"] > 0

    with database.SessionLocal() as s:
        updated_lec = s.query(database.Lecture).filter(database.Lecture.id == lec_id).first()
        assert updated_lec.audio_path.endswith(".m4a")
        assert os.path.isfile(updated_lec.audio_path)
        assert not os.path.exists(test_wav)
        # Cleanup
        try:
            os.remove(updated_lec.audio_path)
        except OSError:
            pass



# ── AI endpoints (body opzionali) ──────────────────────────────────────────────

def test_ai_stop_body_is_honoured(client, monkeypatch):
    """Regressione: `Optional` non importato → il body veniva sempre ignorato."""
    import llm_engine
    seen = {}
    monkeypatch.setattr(llm_engine, "stop_all_ai", lambda force_unload=True: seen.setdefault("f", force_unload) or {"ok": 1})
    client.post("/api/v1/ai/stop", json={"force_unload": False})
    assert seen["f"] is False


def test_generate_summary_uses_requested_model_and_keeps_notes_on_error(client, subject, no_rag, monkeypatch):
    import llm_engine
    lec = client.post(
        "/api/v1/lectures/",
        json={"subject_id": subject["id"], "raw_transcript": "testo della lezione", "processed_notes": "note buone"},
    ).json()

    for name in ("clear_metal_cache", "unload_whisper", "unload_llm"):
        monkeypatch.setattr(llm_engine, name, lambda *a, **kw: None)

    seen = {}

    async def ok_chapter(**kw):
        seen["model_id"] = kw.get("model_id")
        return "# Capitolo"

    monkeypatch.setattr(llm_engine, "generate_lecture_chapter", ok_chapter)
    r = client.post(f"/api/v1/lectures/{lec['id']}/generate-summary", json={"model_id": "modello-x"})
    assert r.status_code == 200, r.text
    assert seen["model_id"] == "modello-x"
    assert r.json()["processed_notes"] == "# Capitolo"

    async def failing_chapter(**kw):
        raise RuntimeError("OOM")

    monkeypatch.setattr(llm_engine, "generate_lecture_chapter", failing_chapter)
    r = client.post(f"/api/v1/lectures/{lec['id']}/generate-summary")
    assert r.status_code == 500
    assert client.get(f"/api/v1/lectures/{lec['id']}").json()["processed_notes"] == "# Capitolo"


def test_course_chat_tolerates_null_suggested_answers(client, subject, no_rag, monkeypatch):
    import llm_engine
    client.post(
        "/api/v1/lectures/",
        json={
            "subject_id": subject["id"],
            "detected_questions": json.dumps([{"question": "Cos'è un integrale?", "suggestedAnswer": None}, "testo"]),
        },
    )

    async def fake_stream(**kw):
        yield "Risposta"

    monkeypatch.setattr(llm_engine, "stream_chat_response", fake_stream)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda *a, **kw: None)
    r = client.post(f"/api/v1/subjects/{subject['id']}/chat", json={"query": "domande del prof?"})
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "Risposta"


def test_answer_question_matches_id_before_index(client, subject, monkeypatch):
    import llm_engine
    qs = [{"id": "1", "question": "A?"}, {"id": "0", "question": "B?"}]
    lec = client.post(
        "/api/v1/lectures/", json={"subject_id": subject["id"], "detected_questions": json.dumps(qs)}
    ).json()
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda *a, **kw: None)

    async def fake_flash(question, context, model_id=None):
        return f"risposta a {question}"

    monkeypatch.setattr(llm_engine, "generate_flash_answer", fake_flash)
    r = client.post(f"/api/v1/lectures/{lec['id']}/questions/0/answer")
    assert r.status_code == 200
    assert r.json()["question"]["question"] == "B?"
