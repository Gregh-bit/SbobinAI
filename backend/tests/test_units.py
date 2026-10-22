from __future__ import annotations

import os

import config
import live_session
from main import extract_professor_questions


def test_extract_professor_questions():
    text = "Oggi vediamo le serie. Chi sa dirmi cosa vuol dire convergere. Qual è il raggio? Fine."
    qs = extract_professor_questions(text)
    assert "Qual è il raggio?" in qs
    assert any(q.startswith("Chi sa dirmi") and q.endswith("?") for q in qs)
    assert extract_professor_questions("   ") == []


class _FakeSession:
    def __init__(self):
        self.stopped = 0
        self.is_active = True

    def stop(self):
        self.stopped += 1
        self.is_active = False
        return "audio.wav"


def test_end_session_only_ends_own_session(monkeypatch):
    old, new = _FakeSession(), _FakeSession()
    monkeypatch.setattr(live_session, "_active_sessions", {7: new})
    # il socket vecchio che si disconnette non deve chiudere la sessione nuova
    assert live_session.end_session(7, old) is None
    assert new.stopped == 0
    assert live_session.has_active_sessions()
    res = live_session.end_session(7, new)
    assert res == (new, "audio.wav")
    assert not live_session.has_active_sessions()


def test_cleaner_skips_when_db_empty(tmp_path):
    from cleaner import cleanup_orphan_files
    audio_dir = config.get_audio_dir()
    os.makedirs(audio_dir, exist_ok=True)
    f = os.path.join(audio_dir, "registrazione.wav")
    open(f, "wb").close()
    try:
        res = cleanup_orphan_files()
        assert res["status"] == "skipped"
        assert os.path.exists(f)
    finally:
        os.remove(f)


def test_cleaner_removes_only_orphans_in_active_storage(client):
    import database
    from cleaner import cleanup_orphan_files
    audio_dir = config.get_audio_dir()
    kept = os.path.join(audio_dir, "lecture_1.wav")
    orphan = os.path.join(audio_dir, "orfano.wav")
    for p in (kept, orphan):
        open(p, "wb").close()
    with database.SessionLocal() as s:
        subj = database.Subject(name="S")
        s.add(subj)
        s.commit()
        s.add(database.Lecture(subject_id=subj.id, title="L", audio_path=kept))
        s.commit()
    res = cleanup_orphan_files()
    assert res["status"] == "success"
    assert os.path.exists(kept)
    assert not os.path.exists(orphan)
    os.remove(kept)


def test_compress_wav_to_m4a_reduces_size_and_decodes(tmp_path):
    import wave
    import numpy as np
    from audio_engine import compress_wav_to_m4a
    from transcriber import _decode_audio_file

    wav_path = str(tmp_path / "lecture_test.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 2 secondi a 16kHz
        samples = (np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000, False)) * 12000).astype(np.int16)
        wf.writeframes(samples.tobytes())

    orig_size = os.path.getsize(wav_path)
    assert orig_size >= 64000

    m4a_path = compress_wav_to_m4a(wav_path, bitrate=48000, remove_original=True)
    assert m4a_path.endswith(".m4a")
    assert os.path.isfile(m4a_path)
    assert not os.path.exists(wav_path)

    m4a_size = os.path.getsize(m4a_path)
    assert m4a_size < orig_size * 0.4  # Almeno 60-80% di riduzione

    # Verifica decodifica corretta
    decoded = _decode_audio_file(m4a_path)
    assert abs(len(decoded) - 32000) < 1500

