"""
Test specifici per la registrazione audio ad alte prestazioni:
1. Risoluzione dispositivo nativo WASAPI su Windows (evita MME ad alta latenza).
2. Apertura istantanea senza blocchi e blocksize flessibile per WASAPI Shared Mode.
3. Coda disaccoppiata non bloccante tra callback audio ed elaborazione worker.
4. Noise floor adattivo dinamico per cattura voce a basso volume del docente (RMS 12-25).
5. Normalizzazione dinamica (AGC Digitale) per evitare allucinazioni su Whisper.
6. Preload asincrono in background del modello Whisper all'avvio sessione.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np
import pytest

import audio_engine
import live_session
import transcriber


# ── 1. Risoluzione dispositivo WASAPI su Windows ─────────────────────────────

def test_resolve_default_wasapi_device_on_windows(monkeypatch):
    """
    Simula l'ambiente audio di Windows:
    Host API 0: MME (default MME con latenza enorme)
    Host API 1: DirectSound
    Host API 2: Windows WASAPI (con microfono Realtek a 48kHz, 2 canali, id=4)
    """
    fake_hostapis = (
        {"name": "MME", "default_input_device": 0},
        {"name": "Windows DirectSound", "default_input_device": 1},
        {"name": "Windows WASAPI", "default_input_device": 4},
    )

    monkeypatch.setattr(audio_engine.sys, "platform", "win32")
    monkeypatch.setattr(audio_engine.sd, "query_hostapis", lambda *a, **kw: fake_hostapis)

    dev_idx = audio_engine._resolve_default_wasapi_device()
    assert dev_idx == 4


# ── 2. Apertura istantanea senza blocchi e blocksize=None ──────────────────────

def test_open_stream_prioritizes_native_params(monkeypatch):
    """Verifica che _open_stream imposti blocksize=None e chiami sd.InputStream con parametri nativi."""
    captured_kwargs = []

    fake_devices = {
        4: {"name": "Microfono Realtek (WASAPI)", "hostapi": 2, "max_input_channels": 2, "default_samplerate": 48000.0}
    }

    class DummyInputStream:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

    monkeypatch.setattr(audio_engine.sys, "platform", "win32")
    monkeypatch.setattr(audio_engine.sd, "InputStream", DummyInputStream)
    monkeypatch.setattr(audio_engine.sd, "query_devices", lambda idx=None, *a, **kw: fake_devices[idx] if idx in fake_devices else fake_devices[4])
    monkeypatch.setattr(audio_engine, "_resolve_input_device", lambda: 4)

    engine = audio_engine.AudioEngine(lecture_id=10)
    engine._device_index = 4
    stream = engine._open_stream()

    assert stream is not None
    assert len(captured_kwargs) == 1
    kw = captured_kwargs[0]
    assert kw["samplerate"] == 48000
    assert kw["channels"] == 2
    assert kw["blocksize"] is None
    assert kw["latency"] == "low"
    assert kw["dtype"] == "int16"


# ── 3. Coda audio disaccoppiata non bloccante ─────────────────────────────────

def test_audio_worker_processes_raw_queue_without_blocking_callback():
    """Il callback audio reale deve solo mettere i chunk in coda; il worker li elabora in background."""
    processed_chunks = []
    engine = audio_engine.AudioEngine(lecture_id=11)
    engine.on_speech_chunk = lambda chunk, el: processed_chunks.append(chunk)

    # Avvia solo il worker di background
    engine.is_running = True
    engine._worker_thread = threading.Thread(target=engine._process_audio_worker, daemon=True)
    engine._worker_thread.start()

    # Genera 5 pacchetti PCM raw (30ms a 16kHz = 480 campioni stereo int16)
    fake_stereo_frame = (np.ones((480, 2), dtype=np.int16) * 500)
    for _ in range(5):
        engine._raw_stream_callback(fake_stereo_frame, 480, None, None)

    # Attendi che il worker svuoti la coda
    time.sleep(0.15)
    assert engine._raw_queue.empty()
    assert engine._recorded_bytes > 0

    engine.stop()


# ── 4. Noise floor adattivo per voce bassa del docente (RMS 12-25) ─────────────

def test_adaptive_noise_floor_detects_faint_speech():
    """
    In presenza di silenzio/rumore di fondo basso (RMS 8.0), il noise floor si adatta.
    Un segnale vocale tenue (RMS 24.0) supera la soglia dinamica e attiva la voce.
    """
    engine = audio_engine.AudioEngine(lecture_id=12, vad_aggressiveness=1)
    engine.is_running = True
    engine._noise_floor = 8.0

    # 1. Chunk di rumore di fondo (RMS ~ 8.0)
    noise_frame = (np.random.normal(0, 8, 480)).astype(np.int16)
    engine._process_mono_chunk(noise_frame.tobytes())
    assert engine.is_voice_active is False

    # 2. Chunk di parlato a basso volume (frequenza vocale 400Hz, ampiezza 35 -> RMS ~ 24.0)
    t = np.arange(480) / 16000.0
    voice_frame = (np.sin(2 * np.pi * 400.0 * t) * 35.0).astype(np.int16)
    engine._process_mono_chunk(voice_frame.tobytes())
    assert engine.is_voice_active is True
    assert engine._speech_frames_count >= 1

    engine.stop()


# ── 5. Normalizzazione dinamica (AGC Digitale) ────────────────────────────────

def test_digital_agc_boosts_faint_audio_to_optimal_range():
    """
    Un frammento vocale a bassa ampiezza (es. peak 2000) deve essere scalato
    fino a un livello ottimale per Whisper prima di essere inviato al transcriber.
    """
    flushed_chunks = []
    engine = audio_engine.AudioEngine(lecture_id=13)
    engine.on_speech_chunk = lambda chunk, el: flushed_chunks.append(chunk)

    # Riempi il buffer con campioni deboli (peak = 2000)
    weak_speech = (np.ones(480, dtype=np.int16) * 2000).tobytes()
    for _ in range(25):
        engine._speech_buffer.append(weak_speech)
    engine._speech_frames_count = 25

    engine._flush_speech_buffer()

    assert len(flushed_chunks) == 1
    boosted_pcm = flushed_chunks[0]
    boosted_i16 = np.frombuffer(boosted_pcm, dtype=np.int16)
    max_peak = int(np.max(np.abs(boosted_i16)))

    # Con boost limitato a 5.0x, il picco iniziale 2000 diventa 10000 (> 8000)
    assert max_peak >= 9000
    assert max_peak <= 18500


# ── 6. Preload asincrono in background del modello Whisper ─────────────────────

def test_transcriber_preload_model_background_launches_non_blocking_thread(monkeypatch):
    """Verifica che preload_model_background avvii un thread asincrono per warm-up."""
    called_holder = []

    def fake_warm():
        time.sleep(0.04)
        called_holder.append("warmed")

    monkeypatch.setattr(transcriber, "warm_up_whisper", fake_warm)
    monkeypatch.setattr(transcriber, "_resolve_model_path", lambda repo: "/fake/model/path")

    t = transcriber.Transcriber(model_name="whisper-large-v3-turbo")

    t0 = time.time()
    t.preload_model_background(delay=0.0)
    elapsed = time.time() - t0

    # Ritorno non bloccante istantaneo (< 25ms)
    assert elapsed < 0.025
    for _ in range(25):
        if "warmed" in called_holder:
            break
        time.sleep(0.02)
    assert "warmed" in called_holder


# ── 7. Aggiornamento VAD dinamico a caldo durante la registrazione ─────────────

def test_dynamic_vad_update_during_recording():
    """Verifica che la modifica della sensibilità VAD avvenga istantaneamente a caldo senza riavviare."""
    engine = audio_engine.AudioEngine(lecture_id=14, vad_aggressiveness=2)
    assert engine.vad_aggressiveness == 2
    assert engine.rms_threshold == 65.0

    # L'utente imposta VAD permissivo (livello 1) durante la lezione
    engine.set_vad_sensitivity(1)
    assert engine.vad_aggressiveness == 1
    assert engine.rms_threshold == 45.0

    # L'utente imposta VAD reattivo/anti-rumore (livello 3)
    engine.set_vad_sensitivity(3)
    assert engine.vad_aggressiveness == 3
    assert engine.rms_threshold == 105.0

    # Verifica propagazione tramite LiveSession
    sess = live_session.LiveSession(subject_id=99, lecture_id=14, mock_transcriber=True, vad_sensitivity=2)
    assert sess.audio_engine.rms_threshold == 65.0
    sess.update_vad_sensitivity(1)
    assert sess.audio_engine.vad_aggressiveness == 1
    assert sess.audio_engine.rms_threshold == 45.0


# ── 8. Normalizzazione volume di riascolto (Playback AGC su esportazione) ──────

def test_playback_normalization_boosts_quiet_file_on_stop(monkeypatch, tmp_path):
    """Verifica che un file audio registrato a volume molto basso (peak 300) venga normalizzato per l'ascolto."""
    monkeypatch.setattr(audio_engine, "get_audio_dir", lambda: str(tmp_path))

    engine = audio_engine.AudioEngine(lecture_id=15)
    # Scrivi campioni tenui (peak = 300, ~-40 dBFS)
    quiet_pcm = (np.ones(16000, dtype=np.int16) * 300).tobytes()
    engine._write_recording(quiet_pcm)

    wav_path = engine.stop(audio_format="wav")
    assert os.path.isfile(wav_path)

    import wave
    with wave.open(wav_path, "rb") as wf:
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        peak = int(np.max(np.abs(data)))
        # Con boost applicato, peak 300 deve essere scalato a ~24000 (> 20000)
        assert peak >= 20000
        assert peak <= 24500


# ── 9. Cache proprietà hardware microfono (Zero-Lag su macOS / CoreAudio) ───────

def test_cached_device_info_skips_repeated_hardware_queries(monkeypatch):
    """Verifica che _get_cached_device_info utilizzi la cache ed eviti scansioni ripetute dell'HAL CoreAudio."""
    queries_count = 0

    def fake_query_devices(*args, **kwargs):
        nonlocal queries_count
        queries_count += 1
        return {"max_input_channels": 2, "default_samplerate": 44100.0}

    monkeypatch.setattr(audio_engine.sd, "query_devices", fake_query_devices)
    audio_engine._DEVICE_INFO_CACHE.clear()

    # Prima chiamata: interroga l'hardware
    ch1, sr1 = audio_engine._get_cached_device_info(device="mic-1")
    assert queries_count == 1
    assert ch1 == 2
    assert sr1 == 44100.0

    # Seconda e terza chiamata: istantanee dalla cache senza toccare l'HAL
    ch2, sr2 = audio_engine._get_cached_device_info(device="mic-1")
    ch3, sr3 = audio_engine._get_cached_device_info(device="mic-1")
    assert queries_count == 1
    assert ch2 == 2
    assert sr3 == 44100.0


def test_pause_resume_stop_uses_abort_close_and_exports_wav_cleanly(monkeypatch, tmp_path):
    """
    Regression test macOS CoreAudio deadlock:
    - pause() deve chiudere lo stream (abort + close) evitando stop()->start()->stop() su CoreAudio;
    - resume() riapre uno stream pulito;
    - stop() usa abort() + close(), drena la coda con force=True ed esporta un WAV valido;
    - una seconda registrazione subito dopo parte e salva senza blocchi.
    """
    import audio_engine

    monkeypatch.setattr(audio_engine, "get_audio_dir", lambda: str(tmp_path))

    opened_streams = []

    class FakeStream:
        def __init__(self, idx):
            self.idx = idx
            self.events = []

        def start(self):
            self.events.append("start")

        def stop(self):
            self.events.append("stop")

        def abort(self):
            self.events.append("abort")

        def close(self):
            self.events.append("close")

    def fake_open_stream(self):
        st = FakeStream(len(opened_streams) + 1)
        opened_streams.append(st)
        self.device_samplerate = 16000.0
        self.device_channels = 1
        self._resampler = None
        return st

    monkeypatch.setattr(audio_engine.AudioEngine, "_open_stream", fake_open_stream)
    monkeypatch.setattr(audio_engine, "sd", object())

    # Sessione 1: start -> audio -> pause -> resume -> audio -> stop
    eng1 = audio_engine.AudioEngine(lecture_id=101)
    assert eng1.start() is True
    assert len(opened_streams) == 1
    assert opened_streams[0].events == ["start"]

    # Simula un blocco vocale in coda prima della pausa
    tone = (np.sin(np.linspace(0, 20 * np.pi, 480)) * 5000).astype(np.int16)
    eng1._raw_queue.put_nowait(tone)
    eng1.pause()
    assert eng1._stream is None
    assert opened_streams[0].events == ["start", "abort", "close"]

    # Ripresa apre uno stream NUOVO pulito
    assert eng1.resume() is True
    assert len(opened_streams) == 2
    assert opened_streams[1].events == ["start"]

    eng1._raw_queue.put_nowait(tone)
    wav1 = eng1.stop()
    assert opened_streams[1].events == ["start", "abort", "close"]
    assert os.path.isfile(wav1)
    assert os.path.getsize(wav1) > 44

    # Sessione 2 subito dopo: parte e salva senza alcun blocco
    eng2 = audio_engine.AudioEngine(lecture_id=102)
    assert eng2.start() is True
    assert len(opened_streams) == 3
    eng2._raw_queue.put_nowait(tone)
    wav2 = eng2.stop()
    assert opened_streams[2].events == ["start", "abort", "close"]
    assert os.path.isfile(wav2)
    assert os.path.getsize(wav2) > 44


def test_stop_exports_wav_even_when_coreaudio_close_hangs(monkeypatch, tmp_path):
    """Se il driver hardware CoreAudio si blocca su abort()/close(), stop() non supera ~1.3s ed esporta comunque il WAV."""
    import audio_engine

    monkeypatch.setattr(audio_engine, "get_audio_dir", lambda: str(tmp_path))

    release_hang = threading.Event()

    class HangingStream:
        active = True

        def start(self):
            pass

        def abort(self):
            release_hang.wait(timeout=10.0)

        def close(self):
            pass

    def fake_open_stream(self):
        self.device_samplerate = 16000.0
        self.device_channels = 1
        self._resampler = None
        return HangingStream()

    monkeypatch.setattr(audio_engine.AudioEngine, "_open_stream", fake_open_stream)
    monkeypatch.setattr(audio_engine, "sd", object())

    eng = audio_engine.AudioEngine(lecture_id=202)
    assert eng.start() is True
    tone = (np.sin(np.linspace(0, 20 * np.pi, 480)) * 4000).astype(np.int16)
    eng._raw_queue.put_nowait(tone)

    t0 = time.monotonic()
    wav_path = eng.stop()
    elapsed = time.monotonic() - t0
    release_hang.set()

    assert elapsed < 2.2, f"stop() ha impiegato troppo tempo ({elapsed:.2f}s) nonostante il timeout su closer"
    assert os.path.isfile(wav_path)
    assert os.path.getsize(wav_path) > 44




