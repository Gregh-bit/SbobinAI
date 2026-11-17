"""
Test per pulizia non distruttiva trascrizioni Whisper e temperature differenziate LLM.
"""
from __future__ import annotations

import asyncio
import json
import types

import ai_models
import llm_engine
from transcript_cleaner import clean_transcript_for_llm


def test_cleaner_removes_hesitations_fillers_and_loops():
    raw = (
        "Allora, ehm... diciamo, ragazzi, oggi vediamo il il teorema fondamentale! [Musica]\n"
        "Questa formula, praticamente, serve per calcolare l'integrale di f(x) + (a + b). "
        "E questa parte, come posso dire, ehm... è fondamentale. "
        "E questa parte è fondamentale. E questa parte è fondamentale. "
        "Grazie per la visione."
    )
    cleaned = clean_transcript_for_llm(raw)
    assert "ehm" not in cleaned.lower()
    assert "allora" not in cleaned.lower()
    assert "praticamente" not in cleaned.lower()
    assert "come posso dire" not in cleaned.lower()
    assert "il il" not in cleaned.lower()
    assert "[Musica]" not in cleaned
    assert "Grazie per la visione" not in cleaned
    assert "(a + b)" in cleaned
    assert cleaned.count("fondamentale") == 2
    assert cleaned.startswith("Oggi vediamo il teorema fondamentale!")


def test_cleaner_preserves_technical_terms_and_numbers():
    technical = (
        "Se A allora B è una implicazione logica.\n"
        "Il tipo di dato è float32 con valore 3.14 e pi greco 3,14 e matrice 1 1 0 0.\n"
        "Procediamo piano piano con X, cioè il vettore normale."
    )
    cleaned = clean_transcript_for_llm(technical)
    assert "Se A allora B" in cleaned
    assert "tipo di dato" in cleaned
    assert "3.14" in cleaned
    assert "3,14" in cleaned and "3, 14" not in cleaned
    assert "1 1 0 0" in cleaned
    assert "piano piano" in cleaned
    assert "X, cioè il vettore normale." in cleaned


def test_cleaner_preserves_booleans_single_char_tuples_lists_and_meaningful_phrases():
    raw = (
        "Le risposte possibili sono vero, no, sì, forse.\n"
        "Consideriamo la funzione f, f definita su R, mentre e e quindi convergono.\n"
        "1. 1. Primo caso!\n"
        "Come posso dire che X è continuo? Questo, come posso dire, è ovvio."
    )
    cleaned = clean_transcript_for_llm(raw)
    assert "vero, no, sì, forse" in cleaned
    assert "funzione f, f" in cleaned
    assert "e quindi convergono" in cleaned
    assert "1. 1. Primo caso!" in cleaned
    assert "Come posso dire che X è continuo?" in cleaned
    assert "Questo, è ovvio" in cleaned or "Questo è ovvio" in cleaned


def test_temperature_constants_and_mlx_sampler():
    assert llm_engine.TEMP_SUMMARY == 0.2
    assert llm_engine.TEMP_CHAT == 0.3
    assert llm_engine.TEMP_QUESTION == 0.5
    assert llm_engine.TEMP_STRUCTURED == 0.2

    sampler = llm_engine._make_mlx_sampler(0.2)
    assert callable(sampler)


def test_stream_tokens_passes_temperature_to_mlx_and_llamacpp(monkeypatch):
    seen_mlx = {}

    class FakeResp:
        text = "ok"

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.sample_utils = __import__("mlx_lm.sample_utils", fromlist=["make_sampler"])

    def fake_stream_generate(*a, **kw):
        seen_mlx.update(kw)
        yield FakeResp()

    fake_mlx_lm.stream_generate = fake_stream_generate

    import sys
    monkeypatch.setattr(llm_engine, "USE_MLX", True)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    out = "".join(llm_engine.stream_tokens(object(), object(), "hi", max_tokens=32, temperature=0.5))
    assert out == "ok"
    assert callable(seen_mlx.get("sampler"))

    seen_cpp = {}

    class FakeLlama:
        def tokenize(self, b, **kw):
            return [1, 2, 3]
        def n_ctx(self):
            return 2048
        def create_completion(self, **kw):
            seen_cpp.update(kw)
            return iter([{"choices": [{"text": "ciao"}]}])

    monkeypatch.setattr(llm_engine, "USE_MLX", False)
    out2 = "".join(llm_engine.stream_tokens(FakeLlama(), object(), "hi", max_tokens=32, temperature=0.3))
    assert out2 == "ciao"
    assert seen_cpp["temperature"] == 0.3


def test_summary_and_study_endpoints_use_expected_temperatures(client, subject, monkeypatch):
    captured_summary = {}

    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "unload_whisper", lambda *a, **kw: None)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda *a, **kw: None)

    def fake_blocking(messages, **kw):
        captured_summary["temperature"] = kw.get("temperature")
        captured_summary["prompt"] = messages[1]["content"]
        return "# Titolo\n\nContenuto pulito."

    monkeypatch.setattr(llm_engine, "_generate_blocking", fake_blocking)
    chap = asyncio.run(
        llm_engine.generate_lecture_chapter(
            title="Analisi",
            transcript="Allora, ehm... questo è il il teorema di Rolle.",
            model_id=ai_models.LLM_DEFAULT_ID,
        )
    )
    assert "Contenuto pulito" in chap
    assert captured_summary["temperature"] == 0.2
    assert "ehm" not in captured_summary["prompt"].lower()
    assert "il il" not in captured_summary["prompt"].lower()

    # Verifica temperature sugli endpoint Studio (quiz=0.2, orale domanda=0.5, valutazione=0.2)
    monkeypatch.setattr("routers.study.is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.clear_metal_cache", lambda *a, **kw: None)
    client.post(
        "/api/v1/lectures/",
        json={"subject_id": subject["id"], "title": "Lez 1", "raw_transcript": "Testo base sul teorema."},
    )

    temps = {}

    def cap_quiz(*a, **kw):
        temps["quiz"] = kw.get("temperature")
        return '```json\n[{"id":"q1","question":"Q?","options":["a","b","c","d"],"correct_index":0,"explanation":"ok","source_lecture":"Lez 1"}]'

    monkeypatch.setattr(llm_engine, "generate_text", cap_quiz)
    assert client.post(f"/api/v1/study/{subject['id']}/generate-quiz", json={"num_questions": 1}).status_code == 200
    assert temps["quiz"] == llm_engine.TEMP_STRUCTURED == 0.2

    def cap_oral(*a, **kw):
        temps["oral_q"] = kw.get("temperature")
        return '"Spieghi il teorema."'

    monkeypatch.setattr(llm_engine, "generate_text", cap_oral)
    assert client.post(f"/api/v1/study/{subject['id']}/oral-question", json={}).status_code == 200
    assert temps["oral_q"] == llm_engine.TEMP_QUESTION == 0.5

    import transcriber
    monkeypatch.setattr(transcriber, "transcribe_file", lambda *a, **kw: ("Allora, ehm... il teorema vale sempre", 2.0))

    def cap_eval(sys_p, usr_p, **kw):
        temps["oral_eval"] = kw.get("temperature")
        temps["oral_eval_prompt"] = usr_p
        return json.dumps({
            "grade": "28/30", "score_num": 28,
            "good_points": "ok", "missing_or_errors": "-", "ideal_formulation": "top",
        })

    monkeypatch.setattr(llm_engine, "generate_text", cap_eval)
    files = {"audio": ("a.wav", b"RIFF", "audio/wav")}
    r = client.post(f"/api/v1/study/{subject['id']}/evaluate-oral", data={"question": "Q?"}, files=files)
    assert r.status_code == 200
    assert temps["oral_eval"] == llm_engine.TEMP_STRUCTURED == 0.2
    assert "ehm" not in temps["oral_eval_prompt"].lower()
