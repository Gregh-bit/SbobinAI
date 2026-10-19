from __future__ import annotations

import json

import pytest

import llm_engine


@pytest.fixture()
def lecture(client, subject):
    return client.post(
        "/api/v1/lectures/",
        json={"subject_id": subject["id"], "title": "Integrali", "raw_transcript": "Gli integrali misurano aree."},
    ).json()


@pytest.fixture(autouse=True)
def _model_present(monkeypatch):
    monkeypatch.setattr("routers.study.is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.clear_metal_cache", lambda *a, **kw: None)


def test_quiz_accepts_string_correct_index_and_requested_count(client, subject, lecture, monkeypatch):
    seen = {}

    def fake_generate(system_prompt, user_prompt, **kw):
        seen.update(kw)
        seen["system"] = system_prompt
        # il testo restituito include il prefisso "```json\n["
        return '```json\n[{"id":"q1","question":"D?","options":["a","b","c","d"],"correct_index":"2",' \
               '"explanation":"e","source_lecture":"Integrali"}]'

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate)
    r = client.post(f"/api/v1/study/{subject['id']}/generate-quiz", json={"num_questions": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    assert body["questions"][0]["correct_index"] == 2
    assert "3 domande" in seen["system"]
    assert seen["assistant_prefix"] == "```json\n["


def test_quiz_fallback_is_flagged_not_success(client, subject, lecture, monkeypatch):
    monkeypatch.setattr(llm_engine, "generate_text", lambda *a, **kw: "non è json")
    r = client.post(f"/api/v1/study/{subject['id']}/generate-quiz", json={})
    body = r.json()
    assert body["status"] == "fallback"
    assert body["warning"]
    assert len(body["questions"]) >= 1


def test_quiz_uses_settings_model_when_not_requested(client, subject, lecture, monkeypatch):
    client.patch("/api/v1/settings/", json={"llm_model": "scelto-nelle-impostazioni"})
    seen = {}

    def fake_generate(system_prompt, user_prompt, **kw):
        seen["model_id"] = kw.get("model_id")
        return "[]"

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate)
    client.post(f"/api/v1/study/{subject['id']}/generate-quiz", json={})
    assert seen["model_id"] == "scelto-nelle-impostazioni"


def test_oral_question(client, subject, lecture, monkeypatch):
    monkeypatch.setattr(llm_engine, "generate_text", lambda *a, **kw: ' "Spieghi il teorema di Fubini." ')
    r = client.post(f"/api/v1/study/{subject['id']}/oral-question", json={})
    assert r.json()["question"] == "Spieghi il teorema di Fubini."


def test_evaluate_oral_success_and_fallback_flag(client, subject, lecture, no_rag, monkeypatch):
    import transcriber

    monkeypatch.setattr(transcriber, "transcribe_file", lambda path, language="it", initial_prompt=None: ("Gli integrali sono aree", 3.0))
    good = json.dumps({
        "grade": "28/30", "score_num": "28", "good_points": "ok",
        "missing_or_errors": "poco", "ideal_formulation": "ideale",
    })
    monkeypatch.setattr(llm_engine, "generate_text", lambda *a, **kw: good)

    files = {"audio": ("oral_answer.webm", b"RIFFxxxx", "audio/webm")}
    r = client.post(f"/api/v1/study/{subject['id']}/evaluate-oral", data={"question": "Cosa sono?"}, files=files)
    body = r.json()
    assert body["status"] == "success"
    assert body["score_num"] == 28

    monkeypatch.setattr(llm_engine, "generate_text", lambda *a, **kw: "boh")
    r = client.post(f"/api/v1/study/{subject['id']}/evaluate-oral", data={"question": "Cosa sono?"}, files=files)
    body = r.json()
    assert body["status"] == "fallback"
    assert body["grade"] == "N.C."
    assert body["warning"]


def test_evaluate_oral_no_speech(client, subject, lecture, no_rag, monkeypatch):
    import transcriber

    monkeypatch.setattr(transcriber, "transcribe_file", lambda *a, **kw: ("", 0.0))
    files = {"audio": ("oral_answer.wav", b"x", "audio/wav")}
    r = client.post(f"/api/v1/study/{subject['id']}/evaluate-oral", data={"question": "Q?"}, files=files)
    assert r.json()["status"] == "no_speech"
