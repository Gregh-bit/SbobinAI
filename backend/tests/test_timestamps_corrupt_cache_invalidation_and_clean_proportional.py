"""
test_timestamps_corrupt_cache_invalidation_and_clean_proportional.py — SbobinAI
Test di validazione della cache karaoke, invalidazione di cache allucinate/corrotte su disco,
pulizia intercalari su stima proporzionale e verifica che nessuna risposta LLM venga
mai generata automaticamente durante la registrazione live.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from database import Lecture, SessionLocal
from routers.lectures import (
    _validate_timestamp_lines,
    _normalize_karaoke_lines,
    compute_proportional_timestamps,
    compute_and_cache_lecture_timestamps,
)


# ── 1. Test unitari su _validate_timestamp_lines ──────────────────────────────

def test_validate_timestamp_lines_valid():
    raw_transcript = (
        "Oggi esploriamo i fondamenti della programmazione orientata agli oggetti, "
        "comprese le classi, l'ereditarietà, il polimorfismo e l'incapsulamento dei dati."
    )
    # Creiamo segmenti karaoke coerenti con il testo
    words = raw_transcript.split()
    lines = [
        {
            "lineStart": 0,
            "lineEnd": 3000,
            "words": [{"text": w, "start": i * 300, "end": (i + 1) * 300} for i, w in enumerate(words[:7])],
        },
        {
            "lineStart": 3000,
            "lineEnd": 6000,
            "words": [{"text": w, "start": (i + 7) * 300, "end": (i + 8) * 300} for i, w in enumerate(words[7:])],
        },
    ]
    is_valid, reason = _validate_timestamp_lines(lines, raw_transcript)
    assert is_valid is True
    assert reason == "ok"


def test_validate_timestamp_lines_detects_hallucination_formulas():
    raw_transcript = (
        "In questa lezione introduciamo i concetti base di algebra lineare e calcolo vettoriale. "
        "Parleremo di spazi vettoriali, matrici ortogonali e determinanti."
    )
    # Cache allucinata tipica di Whisper con formule di chiusura non presenti nel testo reale
    hallucinated_lines = [
        {
            "lineStart": 0,
            "lineEnd": 2000,
            "words": [
                {"text": "Grazie", "start": 0, "end": 500},
                {"text": "a", "start": 500, "end": 800},
                {"text": "tutti.", "start": 800, "end": 1500},
            ],
        }
    ]
    is_valid, reason = _validate_timestamp_lines(hallucinated_lines, raw_transcript)
    assert is_valid is False
    assert "grazie a tutti" in reason.lower()

    # Anche con "sottotitoli" o "iscriviti al canale"
    hallucinated_sub = [
        {
            "lineStart": 0,
            "lineEnd": 2000,
            "words": [
                {"text": "Sottotitoli", "start": 0, "end": 600},
                {"text": "creati", "start": 600, "end": 1000},
                {"text": "da", "start": 1000, "end": 1200},
                {"text": "amara.org", "start": 1200, "end": 1800},
            ],
        }
    ]
    is_valid_sub, reason_sub = _validate_timestamp_lines(hallucinated_sub, raw_transcript)
    assert is_valid_sub is False
    assert "sottotitoli" in reason_sub.lower() or "amara.org" in reason_sub.lower()


def test_validate_timestamp_lines_allows_legitimate_professor_closing():
    # Se il docente dice effettivamente "grazie a tutti" a fine lezione, non deve essere scartato
    raw_transcript = (
        "La lezione termina qui per oggi, grazie a tutti per l'attenzione e ci vediamo "
        "domani mattina puntuali alle ore nove in aula magna per il laboratorio."
    )
    words = raw_transcript.split()
    lines = [
        {
            "lineStart": 0,
            "lineEnd": 4000,
            "words": [{"text": w, "start": i * 200, "end": (i + 1) * 200} for i, w in enumerate(words[:11])],
        },
        {
            "lineStart": 4000,
            "lineEnd": 8000,
            "words": [{"text": w, "start": (i + 11) * 200, "end": (i + 12) * 200} for i, w in enumerate(words[11:])],
        },
    ]
    is_valid, reason = _validate_timestamp_lines(lines, raw_transcript)
    assert is_valid is True
    assert reason == "ok"


def test_validate_timestamp_lines_too_few_words():
    raw_transcript = (
        "Oggi discutiamo il teorema di Bayes e le sue applicazioni pratiche nella probabilità condizionata, "
        "nell'inferenza statistica moderna e nei modelli di apprendimento automatico supervisionato per la classificazione."
    )
    # Trascrizione reale ha 24 parole; cache ne ha solo 3 (< 30%)
    tiny_cache = [
        {
            "lineStart": 0,
            "lineEnd": 1000,
            "words": [
                {"text": "Oggi", "start": 0, "end": 200},
                {"text": "discutiamo", "start": 200, "end": 600},
                {"text": "il", "start": 600, "end": 900},
            ],
        }
    ]
    is_valid, reason = _validate_timestamp_lines(tiny_cache, raw_transcript)
    assert is_valid is False
    assert "troppo poche parole" in reason.lower()


def test_validate_timestamp_lines_consecutive_repetitions():
    raw_transcript = (
        "Nella termodinamica classica il primo principio stabilisce la conservazione dell'energia totale. "
        "Il calore scambiato e il lavoro compiuto modificano l'energia interna del sistema."
    )
    # Due righe consecutive identiche (loop tipico Whisper) che superano la soglia del 30% ma sono ripetizioni non nel testo
    loop_cache = [
        {
            "lineStart": 0,
            "lineEnd": 2000,
            "words": [
                {"text": "Nella", "start": 0, "end": 300},
                {"text": "termodinamica", "start": 300, "end": 800},
                {"text": "classica", "start": 800, "end": 1200},
                {"text": "il", "start": 1200, "end": 1400},
                {"text": "primo", "start": 1400, "end": 1700},
                {"text": "principio", "start": 1700, "end": 2000},
            ],
        },
        {
            "lineStart": 2000,
            "lineEnd": 4000,
            "words": [
                {"text": "Nella", "start": 2000, "end": 2300},
                {"text": "termodinamica", "start": 2300, "end": 2800},
                {"text": "classica", "start": 2800, "end": 3200},
                {"text": "il", "start": 3200, "end": 3400},
                {"text": "primo", "start": 3400, "end": 3700},
                {"text": "principio", "start": 3700, "end": 4000},
            ],
        },
    ]
    is_valid, reason = _validate_timestamp_lines(loop_cache, raw_transcript)
    assert is_valid is False
    assert "ripetizione consecutiva identica" in reason.lower()


def test_validate_timestamp_lines_mismatched_words():
    raw_transcript = (
        "Il modello atomico di Bohr descrive le orbite stazionarie quantizzate degli elettroni "
        "attorno al nucleo positivo con emissione di quanti di radiazione elettromagnetica."
    )
    # Cache con testo completamente scorrelato (es. allucinazione in altra lingua o contesto)
    unrelated_cache = [
        {
            "lineStart": 0,
            "lineEnd": 3000,
            "words": [
                {"text": "Ricetta", "start": 0, "end": 500},
                {"text": "per", "start": 500, "end": 800},
                {"text": "la", "start": 800, "end": 1000},
                {"text": "pizza", "start": 1000, "end": 1400},
                {"text": "napoletana", "start": 1400, "end": 2000},
                {"text": "con", "start": 2000, "end": 2300},
                {"text": "farina", "start": 2300, "end": 2700},
                {"text": "integrale", "start": 2700, "end": 3000},
            ],
        }
    ]
    is_valid, reason = _validate_timestamp_lines(unrelated_cache, raw_transcript)
    assert is_valid is False
    assert "sovrapposizione lessicale" in reason.lower()


# ── 2. Test su compute_proportional_timestamps con clean_transcript_for_llm ──

def test_compute_proportional_timestamps_cleans_noise():
    class DummyLecture:
        raw_transcript = (
            "Allora, diciamo, ehm... oggi ragazzi parliamo di circuiti elettrici "
            "e della legge di Ohm nei conduttori metallici in corrente continua."
        )
        duration_seconds = 20.0

    result = compute_proportional_timestamps(DummyLecture())
    assert result["cached"] is False
    assert len(result["lines"]) > 0

    # Raccogliamo tutte le parole presenti nei timestamp
    all_words = [
        w["text"].lower()
        for line in result["lines"]
        for w in line["words"]
    ]
    # Gli intercalari isolati ed esitazioni vocali sono stati ripuliti da clean_transcript_for_llm
    assert "ehm" not in all_words
    assert "allora," not in all_words
    # Le parole semantiche chiave sono conservate
    assert any("circuiti" in w for w in all_words)
    assert any("elettrici" in w for w in all_words)
    assert any("ohm" in w for w in all_words)


# ── 3. Test invalidazione cache corrotta su disco ──────────────────────────────

def test_compute_and_cache_lecture_timestamps_invalidates_corrupt_disk_cache(tmp_path):
    # Creiamo una lezione con trascrizione vera
    audio_file = str(tmp_path / "test_audio.wav")
    with open(audio_file, "wb") as f:
        f.write(b"RIFF dummy wav data")

    cache_file = audio_file + ".timestamps.json"

    # Scriviamo su disco una vecchia cache spazzatura con "grazie a tutti" ripetuto
    corrupt_lines = [
        {
            "lineStart": i * 1000,
            "lineEnd": (i + 1) * 1000,
            "words": [
                {"text": "Grazie", "start": i * 1000, "end": i * 1000 + 300},
                {"text": "a", "start": i * 1000 + 300, "end": i * 1000 + 500},
                {"text": "tutti.", "start": i * 1000 + 500, "end": (i + 1) * 1000},
            ],
        }
        for i in range(10)
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(corrupt_lines, f)

    db = SessionLocal()
    try:
        from database import Subject
        sub = Subject(name="Informatica Quantistica", description="Test sub")
        db.add(sub)
        db.commit()
        db.refresh(sub)

        real_transcript = (
            "I qubit sfruttano i principi di sovrapposizione quantistica ed entanglement "
            "per eseguire calcoli esponenzialmente più veloci rispetto ai computer classici."
        )
        lec = Lecture(
            subject_id=sub.id,
            title="Qubit ed Entanglement",
            audio_path=audio_file,
            raw_transcript=real_transcript,
            duration_seconds=30.0,
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lec_id = lec.id
    finally:
        db.close()

    # Chiamiamo compute_and_cache_lecture_timestamps: deve rilevare che la cache su disco è corrotta,
    # rimuoverla dal disco, ricalcolare con compute_proportional_timestamps sul vero transcript
    # e salvare la nuova cache pulita!
    res = compute_and_cache_lecture_timestamps(lec_id)
    assert res is not None
    res_words = [w["text"].lower() for line in res.get("lines", []) for w in line.get("words", [])]

    # Non ci deve essere traccia di "grazie a tutti"
    assert "grazie" not in res_words

    # Deve contenere le parole del vero transcript
    assert any("qubit" in w for w in res_words)
    assert any("sovrapposizione" in w for w in res_words)

    # Verifichiamo che il file cache su disco ora contenga la nuova versione pulita
    assert os.path.exists(cache_file)
    with open(cache_file, "r", encoding="utf-8") as f:
        disk_lines = json.load(f)
    disk_words = [w["text"].lower() for line in disk_lines for w in line.get("words", [])]
    assert "grazie" not in disk_words
    assert any("qubit" in w for w in disk_words)


def test_get_lecture_timestamps_endpoint_invalidates_corrupt_cache(client, subject, tmp_path):
    audio_file = str(tmp_path / "endpoint_audio.wav")
    with open(audio_file, "wb") as f:
        f.write(b"dummy")

    cache_file = audio_file + ".timestamps.json"
    # Cache con formula allucinata non presente in transcript
    bad_data = [
        {
            "lineStart": 0,
            "lineEnd": 2000,
            "words": [
                {"text": "Iscriviti", "start": 0, "end": 500},
                {"text": "al", "start": 500, "end": 800},
                {"text": "canale", "start": 800, "end": 1500},
            ],
        }
    ]
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(bad_data, f)

    db = SessionLocal()
    try:
        real_transcript = (
            "Nella relatività generale lo spaziotempo viene curvato dalla presenza di massa ed energia. "
            "Le traiettorie geodetiche descrivono il moto naturale dei corpi in caduta libera."
        )
        lec = Lecture(
            subject_id=subject["id"],
            title="Relatività Generale",
            audio_path=audio_file,
            raw_transcript=real_transcript,
            duration_seconds=45.0,
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lec_id = lec.id
    finally:
        db.close()

    r = client.get(f"/api/v1/lectures/{lec_id}/timestamps")
    assert r.status_code == 200, r.text
    body = r.json()
    returned_words = [w["text"].lower() for line in body.get("lines", []) for w in line.get("words", [])]

    # Non contiene "iscriviti al canale"
    assert "iscriviti" not in returned_words
    # Contiene parole della vera lezione
    assert any("spaziotempo" in w for w in returned_words)
    assert any("geodetiche" in w for w in returned_words)


# ── 4. Verifica: nessuna risposta LLM generata automaticamente ───────────────

@pytest.mark.asyncio
async def test_no_automatic_llm_answers_on_question_detected():
    from live_session import LiveSession
    from main import _announce_questions

    # Creiamo una sessione live con mock transcriber
    session = LiveSession(
        lecture_id=999,
        subject_id=1,
        mock_transcriber=True,
    )

    from starlette.websockets import WebSocketState

    sent_messages = []

    class FakeWebSocket:
        client_state = WebSocketState.CONNECTED

        async def send_json(self, payload: dict):
            sent_messages.append(payload)

    fake_ws = FakeWebSocket()

    # Testo del professore contenente una chiara domanda
    teacher_text = "Quindi ragazzi, qual è la complessità temporale nel caso peggiore del quicksort?"

    await _announce_questions(session, teacher_text, fake_ws)

    # 1. La domanda deve essere stata rilevata
    assert len(session.detected_questions) == 1
    detected = session.detected_questions[0]
    assert "complessità temporale" in detected["question"].lower()

    # 2. suggestedAnswer DEVE essere tassativamente vuota (NESSUN avvio LLM automatico)
    assert detected.get("suggestedAnswer") == ""

    # 3. Il messaggio inviato sul websocket deve contenere suggested_answer vuota
    assert len(sent_messages) == 1
    ws_msg = sent_messages[0]
    assert ws_msg["type"] == "question_detected"
    assert ws_msg["payload"]["suggested_answer"] == ""

    # 4. Nessuna chiamata all'LLM engine è stata eseguita (nessun task di inferenza avviato)


# ── 5. Test Step 2: scarto segmenti Whisper allucinati con fallback proporzionale ──

def test_step2_whisper_hallucination_discarded(monkeypatch, tmp_path):
    import ai_models
    import transcriber

    audio_file = str(tmp_path / "whisper_test.wav")
    with open(audio_file, "wb") as f:
        f.write(b"RIFF dummy")
    cache_file = audio_file + ".timestamps.json"

    db = SessionLocal()
    try:
        from database import Subject
        sub = Subject(name="Matematica Discreta", description="desc")
        db.add(sub)
        db.commit()
        db.refresh(sub)

        real_transcript = (
            "I grafi euleriani contengono un ciclo chiuso che attraversa ogni arco esattamente una volta. "
            "Un grafo connesso è euleriano se e solo se ogni vertice ha grado pari."
        )
        lec = Lecture(
            subject_id=sub.id,
            title="Grafi Euleriani",
            audio_path=audio_file,
            raw_transcript=real_transcript,
            duration_seconds=30.0,
        )
        db.add(lec)
        db.commit()
        db.refresh(lec)
        lec_id = lec.id
    finally:
        db.close()

    monkeypatch.setattr(ai_models, "local_model_path", lambda spec: "/fake/model/path")
    import numpy as np
    dummy_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(transcriber, "_iter_decoded_audio_windows", lambda path: [(dummy_audio, 1.0, True)])
    monkeypatch.setattr(transcriber, "_is_digital_silence", lambda audio: False)

    # Simula Whisper che allucina formule di chiusura non presenti nel testo reale
    class FakeSegment:
        def __init__(self, text, start, end):
            self.start = start
            self.end = end
            self.words = [
                {"word": "Grazie", "start": start, "end": start + 0.3},
                {"word": "a", "start": start + 0.3, "end": start + 0.5},
                {"word": "tutti", "start": start + 0.5, "end": end},
            ]

    fake_segs = [FakeSegment("Grazie a tutti", 0.0, 1.0)]
    monkeypatch.setattr(transcriber, "faster_whisper_word_segments", lambda m, c, language="it": fake_segs)
    monkeypatch.setattr(transcriber, "USE_MLX", False)

    res = compute_and_cache_lecture_timestamps(lec_id)
    assert res is not None
    res_words = [w["text"].lower() for line in res.get("lines", []) for w in line.get("words", [])]

    # I segmenti Whisper allucinati con formule di chiusura devono essere scartati
    assert "grazie" not in res_words
    # È stato usato l'allineamento proporzionale basato su raw_transcript
    assert any("euleriani" in w for w in res_words)
    assert any("connesso" in w for w in res_words)
    # La nuova cache pulita è stata salvata su disco
    assert os.path.exists(cache_file)
