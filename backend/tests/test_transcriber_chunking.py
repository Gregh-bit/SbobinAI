"""
Test per chunking audio in streaming di transcribe_file e deduplica ai confini.
"""
from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

import transcriber


def _write_wav(path: str, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def test_streaming_windows_match_full_decode_and_bound_memory():
    with tempfile.TemporaryDirectory(prefix="sbobinai_wav_") as tmp:
        wav_path = os.path.join(tmp, "long.wav")
        sr = 16000
        t = np.arange(int(sr * 5.3), dtype=np.float32) / sr  # 5.3s @ 16kHz
        sig = (np.sin(2 * np.pi * 440.0 * t) * 12000.0).astype(np.int16)
        _write_wav(wav_path, sig, sr)

        windows = list(
            transcriber._iter_decoded_audio_windows(
                wav_path, sample_rate=sr, window_seconds=2.0, overlap_seconds=0.5
            )
        )
        assert len(windows) == 4  # [0..2.0], [1.5..3.5], [3.0..5.0], [4.5..5.3]
        assert all(w[0].size <= int(2.0 * sr) for w in windows)
        assert windows[0][2] is True and all(not w[2] for w in windows[1:])
        total_dur = sum(w[1] for w in windows)
        assert abs(total_dur - 5.3) < 0.02

        # Verifica corrispondenza dei campioni sulla giunzione con _decode_audio_file
        full = transcriber._decode_audio_file(wav_path, sr)
        assert np.allclose(windows[0][0], full[: windows[0][0].size], atol=1e-4)
        # Il secondo chunk parte a (2.0 - 0.5)s = 1.5s dall'inizio
        offset_1 = int(1.5 * sr)
        assert np.allclose(windows[1][0], full[offset_1 : offset_1 + windows[1][0].size], atol=1e-4)


def test_overlap_seam_dedup_merges_cleanly():
    chunks = [
        "Oggi analizziamo la convergenza uniforme delle serie di funzioni.",
        "delle serie di funzioni secondo il criterio di Weierstrass,",
        "secondo il criterio di Weierstrass con un esempio pratico.",
    ]
    merged = transcriber._merge_chunk_transcripts(chunks)
    assert merged.count("delle serie di funzioni") == 1
    assert merged.count("Weierstrass") == 1
    assert "con un esempio pratico." in merged


def test_transcribe_file_multi_window_skips_silence_and_releases_cache(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="sbobinai_chunk_tf_") as tmp:
        wav_path = os.path.join(tmp, "mixed.wav")
        sr = 16000
        # 2.0s parlato + 2.0s silenzio digitale + 2.0s parlato debole (~5e-4) = 6.0s
        part1 = (np.sin(2 * np.pi * 300.0 * np.arange(sr * 2) / sr) * 8000.0).astype(np.int16)
        part2 = np.zeros(sr * 2, dtype=np.int16)
        part3 = (np.sin(2 * np.pi * 350.0 * np.arange(sr * 2) / sr) * 20.0).astype(np.int16)
        _write_wav(wav_path, np.concatenate([part1, part2, part3]), sr)

        seen_windows = []
        cache_clears = {"count": 0}

        monkeypatch.setattr(transcriber.ai_models, "local_model_path", lambda *_: "/fake/whisper")
        monkeypatch.setattr(transcriber, "FILE_CHUNK_SECONDS", 2.0)
        monkeypatch.setattr(transcriber, "FILE_CHUNK_OVERLAP_SECONDS", 0.2)
        monkeypatch.setattr(
            transcriber,
            "release_metal_cache",
            lambda: cache_clears.__setitem__("count", cache_clears["count"] + 1),
        )

        def fake_window(model_path, audio, *, language, initial_prompt):
            seen_windows.append(float(np.max(np.abs(audio))))
            return f"finestra {len(seen_windows)}"

        monkeypatch.setattr(transcriber, "_transcribe_file_window", fake_window)

        text, dur = transcriber.transcribe_file(wav_path, language="it")
        assert abs(dur - 6.0) < 0.05
        # La finestra 2 puramente silenziosa (overlap 0.2s zero-tailed dopo la prima) o
        # verificando che l'ultima finestra debole (~6e-4 > 1e-4) NON venga scartata
        assert any(peak < 0.01 for peak in seen_windows), seen_windows
        assert cache_clears["count"] >= 3
        assert "finestra 1" in text


def test_transcribe_file_does_not_dedup_across_silence_gap(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="sbobinai_gap_tf_") as tmp:
        wav_path = os.path.join(tmp, "gap.wav")
        sr = 16000
        # Finestre (2.0s, overlap 0.2s, passo 1.8s): [0..2.0], [1.8..3.8] (silenzio puro), [3.6..5.6]
        part1 = (np.sin(2 * np.pi * 300.0 * np.arange(int(sr * 1.7)) / sr) * 8000.0).astype(np.int16)
        part2 = np.zeros(sr * 3, dtype=np.int16)
        part3 = (np.sin(2 * np.pi * 400.0 * np.arange(int(sr * 0.9)) / sr) * 8000.0).astype(np.int16)
        _write_wav(wav_path, np.concatenate([part1, part2, part3]), sr)

        monkeypatch.setattr(transcriber.ai_models, "local_model_path", lambda *_: "/fake/whisper")
        monkeypatch.setattr(transcriber, "FILE_CHUNK_SECONDS", 2.0)
        monkeypatch.setattr(transcriber, "FILE_CHUNK_OVERLAP_SECONDS", 0.2)
        monkeypatch.setattr(transcriber, "release_metal_cache", lambda: None)

        outputs = iter([
            "Oggi studiamo le funzioni continue",
            "funzioni continue nel caso reale",
        ])
        monkeypatch.setattr(
            transcriber,
            "_transcribe_file_window",
            lambda *a, **kw: next(outputs),
        )

        text, _ = transcriber.transcribe_file(wav_path, language="it")
        assert text == "Oggi studiamo le funzioni continue funzioni continue nel caso reale"


def test_clean_file_window_text_and_generator_closing(monkeypatch):
    # Mantiene risposte brevi valide ma scarta boilerplate e collassa loop
    assert transcriber._clean_file_window_text("A") == "A"
    assert transcriber._clean_file_window_text("42") == "42"
    assert transcriber._clean_file_window_text("Sottotitoli creati dalla comunità Amara.org") == ""
    looped = "Parte reale della lezione. " + ("e così via e così via e così via " * 2)
    cleaned_loop = transcriber._clean_file_window_text(looped)
    assert "Parte reale della lezione." in cleaned_loop
    assert cleaned_loop.count("e così via") < 3

    # Verifica chiusura deterministica di _iter_resampled_s16_frames su uscita anticipata
    closed = {"flag": False}

    def fake_frames(path, sample_rate):
        try:
            yield np.ones(sample_rate * 2, dtype=np.int16)
            yield np.ones(sample_rate * 2, dtype=np.int16)
        finally:
            closed["flag"] = True

    monkeypatch.setattr(transcriber, "_iter_resampled_s16_frames", fake_frames)
    gen = transcriber._iter_decoded_audio_windows("ignored.wav", sample_rate=16000, window_seconds=2.0, overlap_seconds=0.5)
    next(gen)
    gen.close()
    assert closed["flag"] is True


def test_compute_and_cache_lecture_timestamps_streams_and_offsets_windows(monkeypatch):
    import json
    import database
    import routers.lectures as lectures_router

    with tempfile.TemporaryDirectory(prefix="sbobinai_ts_") as tmp:
        wav_path = os.path.join(tmp, "lecture.wav")
        sr = 16000
        # 5.0 secondi -> finestre da 2.0s con overlap 0.5s: [0..2.0], [1.5..3.5], [3.0..5.0]
        sig = (np.sin(2 * np.pi * 440.0 * np.arange(sr * 5) / sr) * 8000.0).astype(np.int16)
        _write_wav(wav_path, sig, sr)

        with database.SessionLocal() as db:
            subj = database.Subject(name="Fisica", description="Meccanica")
            db.add(subj)
            db.commit()
            db.refresh(subj)
            lec = database.Lecture(
                subject_id=subj.id,
                title="Cinematica",
                raw_transcript="Prima frase seconda frase terza frase",
                audio_path=wav_path,
                duration_seconds=5,
            )
            db.add(lec)
            db.commit()
            db.refresh(lec)
            lec_id = lec.id

        monkeypatch.setattr(transcriber.ai_models, "local_model_path", lambda *_: "/fake/whisper")
        monkeypatch.setattr(transcriber, "FILE_CHUNK_SECONDS", 2.0)
        monkeypatch.setattr(transcriber, "FILE_CHUNK_OVERLAP_SECONDS", 0.5)
        monkeypatch.setattr("platform_utils.USE_MLX", False)
        monkeypatch.setattr(transcriber, "USE_MLX", False)
        monkeypatch.setattr(transcriber, "release_transcriber_memory", lambda: None)
        monkeypatch.setattr(transcriber, "release_metal_cache", lambda: None)
        # Disabilita _decode_audio_file per garantire che venga usato solo lo streaming a finestre
        monkeypatch.setattr(transcriber, "_decode_audio_file", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("full decode called")))

        calls = {"idx": 0}

        def fake_word_segments(model_path, audio, language="it"):
            calls["idx"] += 1
            if calls["idx"] == 1:
                return [{"words": [{"word": "Prima", "start": 0.2, "end": 0.8}, {"word": "frase", "start": 1.6, "end": 1.9}]}]
            if calls["idx"] == 2:
                # finestra 2 parte a 1.5s; parola a t_local=0.2s -> t_abs=1.7s <= 2.02s va filtrata per overlap
                return [{"words": [{"word": "frase", "start": 0.2, "end": 0.4}, {"word": "seconda", "start": 0.7, "end": 1.4}]}]
            return [{"words": [{"word": "terza", "start": 0.8, "end": 1.5}]}]

        monkeypatch.setattr(transcriber, "faster_whisper_word_segments", fake_word_segments)

        res = lectures_router.compute_and_cache_lecture_timestamps(lec_id)
        assert os.path.isfile(wav_path + ".timestamps.json")
        all_words = [w for line in res["lines"] for w in line.get("words", [])]
        assert [w["text"] for w in all_words] == ["Prima", "frase", "seconda", "terza"]
        assert abs(all_words[2]["start"] - 2200) <= 50  # (1.5s + 0.7s) in ms
        assert abs(all_words[3]["start"] - 3800) <= 50  # (3.0s + 0.8s) in ms

        # Seconda chiamata legge direttamente la cache su disco senza rilanciare Whisper
        res_cached = lectures_router.compute_and_cache_lecture_timestamps(lec_id)
        assert res_cached["cached"] is True
        assert calls["idx"] == 3
