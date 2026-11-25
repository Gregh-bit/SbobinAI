"""
audio_engine.py — SbobinAI Live Audio Engine
Gestisce la cattura audio in tempo reale dal microfono:
 - Campionamento a 16kHz, mono, PCM 16-bit (standard nativo Whisper e WebRTC VAD)
   Se il dispositivo non accetta 16 kHz (es. alcuni driver Windows), cattura alla frequenza nativa
   e ricampiona in streaming (filtro anti-aliasing con stato + interpolazione).
 - Voice Activity Detection (VAD) ibrido (WebRTC VAD + RMS energetico)
 - Registrazione integrale su SpooledTemporaryFile: in RAM fino a 64 MB (~35 min), poi su file
   temporaneo → memoria costante anche per lezioni di ore, nessuna copia completa del buffer.
 - Accumulo a blocchi per Whisper Large-v3-Turbo:
     * Finestra temporale massima di 20 secondi continui
     * Innesco naturale su pausa stabile del docente (>500ms di silenzio)
 - Esportazione del file finale .wav (scrittura atomica .part → rename) alla chiusura della sessione

Selezione microfono: variabile d'ambiente SBOBINAI_INPUT_DEVICE (indice o parte del nome).
"""
from __future__ import annotations

import collections
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:  # ImportError o PortAudio mancante
    sd = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

logger = logging.getLogger("sbobinai.audio")

# ── Percorsi Dinamici ──────────────────────────────────────────────────────────
from config import get_audio_dir

_SPOOL_MAX_RAM_BYTES = 64 * 1024 * 1024


def _resolve_default_wasapi_device() -> Optional[int]:
    """Su Windows trova l'indice del dispositivo di input predefinito WASAPI."""
    if sd is None or sys.platform != "win32":
        return None
    try:
        hostapis = sd.query_hostapis()
        for h in hostapis:
            if "wasapi" in str(h.get("name", "")).lower():
                dev_id = h.get("default_input_device")
                if dev_id is not None and int(dev_id) >= 0:
                    return int(dev_id)
    except Exception as e:
        logger.debug(f"[AudioEngine] Query hostapi WASAPI fallita: {e}")
    return None


def _resolve_input_device() -> Optional[object]:
    """Dispositivo di input da SBOBINAI_INPUT_DEVICE o dispositivo WASAPI nativo su Windows."""
    if sd is None:
        return None
    wanted = (os.environ.get("SBOBINAI_INPUT_DEVICE") or "").strip()
    if wanted:
        if wanted.isdigit():
            return int(wanted)
        try:
            devices = sd.query_devices()
            # Su Windows cerca prima se c'è un endpoint WASAPI corrispondente
            if sys.platform == "win32":
                for idx, dev in enumerate(devices):
                    if dev.get("max_input_channels", 0) > 0 and wanted.lower() in str(dev.get("name", "")).lower():
                        h_id = dev.get("hostapi")
                        if h_id is not None:
                            try:
                                h_name = str(sd.query_hostapis(h_id).get("name", "")).lower()
                                if "wasapi" in h_name:
                                    return idx
                            except Exception:
                                pass
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0 and wanted.lower() in str(dev.get("name", "")).lower():
                    return idx
        except Exception:
            pass
        logger.warning(f"[AudioEngine] Dispositivo '{wanted}' non trovato: uso il microfono predefinito.")

    if sys.platform == "win32":
        wasapi_dev = _resolve_default_wasapi_device()
        if wasapi_dev is not None:
            return wasapi_dev

    return None


_DEVICE_INFO_CACHE: dict = {}


def _get_cached_device_info(device: Optional[object]) -> tuple[int, float]:
    """
    Recupera proprietà (canali, samplerate nativo) del dispositivo di input con cache (60s).
    Evita il freeze da scansione sincrona del CoreAudio HAL su macOS e COM su Windows.
    """
    global _DEVICE_INFO_CACHE
    if sd is None:
        return 1, 48000.0
    now = time.time()
    cache_key = str(device)
    if cache_key in _DEVICE_INFO_CACHE:
        ts, cached_ch, cached_sr = _DEVICE_INFO_CACHE[cache_key]
        if (now - ts) < 60.0:
            return cached_ch, cached_sr

    max_ch = 1
    native_sr = 48000.0
    try:
        info = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
        max_ch = max(1, int(info.get("max_input_channels", 1) or 1))
        native_sr = float(info.get("default_samplerate", 48000.0) or 48000.0)
        _DEVICE_INFO_CACHE[cache_key] = (now, max_ch, native_sr)
    except Exception as q_err:
        logger.warning(f"[AudioEngine] Query dispositivo input fallita ({q_err}), uso valori standard.")

    return max_ch, native_sr


class _StreamResampler:
    """Ricampionatore in streaming (src_rate → 16 kHz) con stato continuo tra i blocchi."""

    def __init__(self, src_rate: float, dst_rate: int = 16000):
        import scipy.signal as signal
        self.ratio = float(src_rate) / float(dst_rate)
        self._sos = signal.butter(8, (dst_rate * 0.45) / (src_rate / 2.0), btype="lowpass", output="sos")
        self._zi = signal.sosfilt_zi(self._sos) * 0.0
        self._pos = 0.0                         # posizione frazionaria del prossimo campione in uscita
        self._prev = np.zeros(1, dtype=np.float32)  # ultimo campione del blocco precedente

    def process(self, block: np.ndarray) -> np.ndarray:
        import scipy.signal as signal
        x = block.astype(np.float32, copy=False)
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        y = np.concatenate([self._prev, y.astype(np.float32, copy=False)])
        # indici di uscita rispetto a y (y[0] = ultimo campione precedente)
        positions = np.arange(self._pos, len(y) - 1, self.ratio, dtype=np.float64)
        out = np.interp(positions, np.arange(len(y)), y).astype(np.float32) if len(positions) else np.zeros(0, np.float32)
        next_pos = (positions[-1] + self.ratio) if len(positions) else self._pos
        self._pos = next_pos - (len(y) - 1)
        self._prev = y[-1:].copy()
        return out


class AudioEngine:
    """
    Cattura l'audio live dal microfono, esegue VAD ed accumula i blocchi vocali per Whisper.
    """

    SAMPLE_RATE = 16000          # 16 kHz (richiesto da Whisper e VAD)
    CHANNELS = 1                 # Mono
    FRAME_MS = 30                # 30 ms per frame VAD
    SAMPLES_PER_FRAME = 480      # 16000 * 0.030 = 480 campioni (960 bytes @ 16-bit)
    BYTES_PER_FRAME = 960        # 480 campioni * 2 bytes

    def __init__(
        self,
        lecture_id: int,
        vad_aggressiveness: int = 2,
        on_speech_chunk: Optional[Callable[[bytes, float], None]] = None,
    ):
        """
        :param lecture_id: ID della lezione per il salvataggio finale
        :param vad_aggressiveness: Modalità VAD (0: permissivo, 3: aggressivo)
        :param on_speech_chunk: Callback facoltativa (chunk_bytes, timestamp)
        """
        self.lecture_id = lecture_id
        self.on_speech_chunk = on_speech_chunk
        self.started_at: float = 0.0
        self.is_running: bool = False
        self.is_paused: bool = False
        self.is_voice_active: bool = False

        # Registrazione integrale su file temporaneo PCM (scrittura sequenziale continua, zero latenza di rollover)
        self._temp_pcm_file = tempfile.NamedTemporaryFile(
            mode="w+b", prefix="sbobinai_rec_", suffix=".pcm", delete=False
        )
        self._temp_pcm_path: Optional[str] = self._temp_pcm_file.name
        self._recorded_bytes = 0
        self._lock = threading.Lock()

        # Dispositivo e canali negoziati
        self.device_channels: int = 1
        self.device_samplerate: float = float(self.SAMPLE_RATE)

        # Coda audio decuplicata dal thread real-time PortAudio
        self._raw_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1000)
        self._worker_thread: Optional[threading.Thread] = None

        # Noise floor adattivo per microfoni a basso guadagno (laptop Windows)
        self._noise_floor: float = 8.0

        # Coda vocale per Whisper Large-v3-Turbo
        self.speech_queue: queue.Queue[bytes] = queue.Queue()

        # VAD
        self.vad = None
        # Le impostazioni possono provenire da vecchi file o richieste API non
        # validate: WebRTC VAD accetta solo livelli 0..3, la UI usa 1..3.
        try:
            vad_aggressiveness = max(1, min(3, int(vad_aggressiveness)))
        except (TypeError, ValueError):
            vad_aggressiveness = 2
        self.vad_aggressiveness = vad_aggressiveness
        if webrtcvad:
            try:
                self.vad = webrtcvad.Vad(vad_aggressiveness)
            except Exception as e:
                logger.warning(f"[AudioEngine] Inizializzazione VAD fallita: {e}")

        # Soglia RMS dinamica basata su vad_aggressiveness:
        # Calibrata per catturare voce naturale anche su microfoni integrati di laptop Windows a bassa amplificazione
        # 1: permissivo (rms > 45), 2: standard (rms > 65), 3: reattivo/anti-rumore (rms > 105)
        self.rms_threshold = 45.0 if vad_aggressiveness == 1 else (105.0 if vad_aggressiveness == 3 else 65.0)

        # Buffer di pre-roll (ultimi ~240ms / 8 frame prima dell'attacco della voce)
        self._preroll_buffer: collections.deque[bytes] = collections.deque(maxlen=8)

        # Buffer accumulo parlato per Whisper Large Turbo
        self._speech_buffer: list[bytes] = []
        self._speech_frames_count: int = 0
        self._silence_frames_count: int = 0
        self._last_speech_time: float = 0.0
        self._total_speech_flushes: int = 0

        # Filtro High-Pass anti-rumble (80 Hz) con scipy.signal
        self._hp_enabled = False
        self._hp_zi = None
        self._hp_b = None
        self._hp_a = None
        try:
            import scipy.signal as signal
            self._hp_b, self._hp_a = signal.butter(2, 80.0 / (self.SAMPLE_RATE / 2.0), btype="highpass")
            self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
            self._hp_enabled = True
            logger.info("[AudioEngine] Filtro High-Pass 80Hz inizializzato con successo.")
        except Exception as e:
            logger.warning(f"[AudioEngine] Inizializzazione filtro High-Pass fallita ({e}). Uso segnale diretto.")

        # Ricampionamento (attivo se il dispositivo hardware non supporta 16 kHz nativi)
        self._resampler: Optional[_StreamResampler] = None
        self._pending = np.zeros(0, dtype=np.float32)

        # Stream sounddevice
        self._stream = None
        self._mock_mic_thread: Optional[threading.Thread] = None

    def set_vad_sensitivity(self, vad_aggressiveness: int) -> None:
        """
        Aggiorna in tempo reale la sensibilità del VAD e la soglia RMS senza interrompere la registrazione:
        1: permissivo (cattura bisbigli e voce lontana del docente, RMS > 45, soglia noise floor 1.5x)
        2: standard (parlato normale, RMS > 65, soglia noise floor 1.8x)
        3: reattivo/anti-rumore (aula rumorosa, RMS > 105, soglia noise floor 2.3x)
        """
        level = max(1, min(3, int(vad_aggressiveness)))
        self.vad_aggressiveness = level
        self.rms_threshold = 45.0 if level == 1 else (105.0 if level == 3 else 65.0)
        if webrtcvad:
            try:
                self.vad = webrtcvad.Vad(level)
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore aggiornamento WebRTC VAD: {e}")
        logger.info(f"[AudioEngine] Sensibilità VAD aggiornata a caldo: livello={level}, soglia RMS={self.rms_threshold}")

    def _reset_filter_state(self) -> None:
        """Azzera lo stato del filtro IIR per evitare discontinuità o residui tra registrazioni/pause."""
        if self._hp_enabled and self._hp_b is not None:
            try:
                import scipy.signal as signal
                self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
            except Exception:
                pass

    def _open_stream(self):
        """
        Apre lo stream audio dal microfono con priorità nativa a zero latenza:
        - Su Windows: interroga il default WASAPI endpoint e usa samplerate nativo (es. 48kHz)
          con blocksize=None (evita timeout COM e negoziazioni fallite da 20-40s).
        - Su macOS / Linux: 16 kHz Mono nativo immediato via CoreAudio / ALSA.
        - Ricampionamento garantito da _StreamResampler a 16 kHz.
        """
        device = _resolve_input_device()
        max_ch, native_sr = _get_cached_device_info(device)

        native_int_sr = int(round(native_sr))
        preferred_ch = min(2, max_ch)

        if sys.platform == "win32":
            # Su Windows WASAPI shared mode:
            # 1. Native sample rate + canali supportati (preferito stereo/array microfonico) + blocksize=None
            # 2. Native sample rate + mono (1 ch) + blocksize=None
            # 3. Dispositivo predefinito generico (device=None)
            candidates = [
                (device, native_int_sr, preferred_ch),
                (device, native_int_sr, 1),
                (None, native_int_sr, preferred_ch),
                (None, 48000, preferred_ch),
                (None, 44100, preferred_ch),
                (None, self.SAMPLE_RATE, 1),
            ]
        else:
            # Su macOS (CoreAudio) e Linux, la conversione a 16 kHz mono hardware/sistema è nativa e istantanea
            candidates = [
                (device, self.SAMPLE_RATE, 1),
                (device, self.SAMPLE_RATE, preferred_ch),
                (device, native_int_sr, 1),
                (device, native_int_sr, preferred_ch),
                (None, self.SAMPLE_RATE, 1),
            ]

        # Rimuovi candidati duplicati mantenendo l'ordine di priorità
        unique_candidates = []
        seen = set()
        for dev, sr, ch in candidates:
            key = (dev, sr, ch)
            if key not in seen:
                seen.add(key)
                unique_candidates.append((dev, sr, ch))

        last_err = None
        for dev, sr, ch in unique_candidates:
            try:
                # IMPORTANTE: blocksize=None usa il buffer periodico nativo hardware/WASAPI senza forzare
                # blocchi rigidi che causano timeout COM di 5s o errori AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED.
                stream = sd.InputStream(
                    device=dev,
                    samplerate=sr,
                    channels=ch,
                    dtype="int16",
                    blocksize=None,
                    latency="low",
                    callback=self._raw_stream_callback,
                )
                self.device_samplerate = float(sr)
                self.device_channels = ch
                if sr != self.SAMPLE_RATE:
                    logger.info(f"[AudioEngine] Dispositivo aperto a {sr:.0f} Hz ({ch} canali) con ricampionamento streaming a 16 kHz.")
                    self._resampler = _StreamResampler(sr, self.SAMPLE_RATE)
                else:
                    logger.info(f"[AudioEngine] Dispositivo aperto a 16 kHz nativi ({ch} canali).")
                    self._resampler = None
                self._pending = np.zeros(0, dtype=np.float32)
                return stream
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Nessuna configurazione audio hardware supportata dal dispositivo ({last_err})")

    def start(self) -> bool:
        """Avvia la registrazione audio."""
        self.started_at = time.time()
        self.is_running = True
        self.is_paused = False
        self.is_voice_active = False
        self._last_speech_time = self.started_at
        self._reset_filter_state()

        # Avvia thread worker asincrono per I/O e VAD (non blocca mai il callback PortAudio)
        self._worker_thread = threading.Thread(target=self._process_audio_worker, daemon=True, name="audio-engine-worker")
        self._worker_thread.start()

        # Prova ad aprire il microfono reale con sounddevice
        if sd:
            try:
                self._stream = self._open_stream()
                self._stream.start()
                self.started_at = time.time()
                self._last_speech_time = self.started_at
                logger.info(f"[AudioEngine] Microfono hardware avviato ({self.device_samplerate:.0f}Hz, {self.device_channels}ch) per lezione {self.lecture_id}")
                return True
            except Exception as e:
                self._stream = None
                logger.warning(f"[AudioEngine] Impossibile aprire stream audio hardware ({e}). Uso fallback simulato.")

        # Fallback generatore di frame silenziosi (se il mic hardware non è accessibile)
        self._mock_mic_thread = threading.Thread(target=self._run_simulated_mic, daemon=True)
        self._mock_mic_thread.start()
        self.started_at = time.time()
        logger.info(f"[AudioEngine] Fallback simulato avviato per lezione {self.lecture_id}")
        return True

    def _write_recording(self, raw_bytes: bytes) -> None:
        try:
            with self._lock:
                if self._temp_pcm_file and not self._temp_pcm_file.closed:
                    self._temp_pcm_file.write(raw_bytes)
                    self._recorded_bytes += len(raw_bytes)
        except Exception as e:
            logger.error(f"[AudioEngine] Errore scrittura PCM: {e}")

    def _raw_stream_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback universale per cattura audio: ultra-rapido (real-time safe), downmix e accodamento non bloccante."""
        if status:
            logger.debug(f"[AudioEngine] Status sounddevice: {status}")
        if not self.is_running or self.is_paused:
            # Segnala a PortAudio/CoreAudio di fermare l'AudioUnit dal proprio thread real-time
            # evitando il deadlock di AudioOutputUnitStop() chiamato da thread esterni su macOS.
            if sd is not None and hasattr(sd, "CallbackStop"):
                raise sd.CallbackStop
            return

        # 1. Downmix a mono se il flusso hardware è multicanale (stereo / array microfonico)
        if isinstance(indata, (bytes, bytearray, memoryview)):
            indata = np.frombuffer(indata, dtype=np.int16)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono_int16 = np.mean(indata.astype(np.int32), axis=1).astype(np.int16)
        else:
            mono_int16 = indata.reshape(-1).astype(np.int16, copy=False)

        # 2. Se il worker thread è attivo, delega asincronamente per zero latenza PortAudio
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                self._raw_queue.put_nowait(mono_int16.copy())
            except queue.Full:
                pass
        else:
            # Fallback sincrono per test unitari diretti o ambienti senza thread
            self._process_mono_chunk(mono_int16)

    def _process_audio_worker(self) -> None:
        """Thread worker di background per elaborazione audio, resampling e I/O senza bloccare PortAudio."""
        while self.is_running:
            try:
                mono_chunk = self._raw_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Elabora sempre il chunk già estratto anche se stop() ha appena azzerato is_running
                self._process_mono_chunk(mono_chunk, force=not self.is_running)
            except Exception as exc:
                logger.debug(f"[AudioEngine Worker] Errore elaborazione: {exc}")

    def _process_mono_chunk(self, mono_int16: np.ndarray, *, force: bool = False) -> None:
        """Elabora un blocco mono int16: ricampionamento a 16 kHz e suddivisione in frame da 30ms."""
        if not force and (not self.is_running or self.is_paused):
            return

        if isinstance(mono_int16, (bytes, bytearray, memoryview)):
            mono_int16 = np.frombuffer(mono_int16, dtype=np.int16)

        if self._resampler is not None:
            out = self._resampler.process(mono_int16)
            buf = np.concatenate([self._pending, out]) if len(self._pending) else out
            n_frames = len(buf) // self.SAMPLES_PER_FRAME
            for i in range(n_frames):
                seg = buf[i * self.SAMPLES_PER_FRAME:(i + 1) * self.SAMPLES_PER_FRAME]
                frame = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
                self._process_16k_mono_frame(frame, force=force)
            self._pending = buf[n_frames * self.SAMPLES_PER_FRAME:].copy()
        else:
            if len(mono_int16) == self.SAMPLES_PER_FRAME and len(self._pending) == 0:
                self._process_16k_mono_frame(mono_int16, force=force)
            else:
                buf = np.concatenate([self._pending, mono_int16.astype(np.float32, copy=False)]) if len(self._pending) else mono_int16.astype(np.float32, copy=False)
                n_frames = len(buf) // self.SAMPLES_PER_FRAME
                for i in range(n_frames):
                    seg = buf[i * self.SAMPLES_PER_FRAME:(i + 1) * self.SAMPLES_PER_FRAME]
                    frame = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
                    self._process_16k_mono_frame(frame, force=force)
                self._pending = buf[n_frames * self.SAMPLES_PER_FRAME:].copy()

    # Alias per compatibilità con test unitari e chiamate dirette
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._raw_stream_callback(indata, frames, time_info, status)

    def _native_rate_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._raw_stream_callback(indata, frames, time_info, status)

    def _process_16k_mono_frame(self, frame: np.ndarray, *, force: bool = False) -> None:
        """Elabora un singolo frame mono da 480 campioni int16 a 16 kHz (filtro HP, VAD con noise floor adattivo, accumulo)."""
        if not force and (not self.is_running or self.is_paused):
            return

        # 1. Filtro High-Pass anti-rumble (80 Hz): rimuove ronzii, ventole e colpi sulla scrivania
        samples_raw = frame.reshape(-1)
        if self._hp_enabled and self._hp_zi is not None:
            try:
                import scipy.signal as signal
                samples_f = samples_raw.astype(np.float32, copy=False)
                filtered_f, self._hp_zi = signal.lfilter(self._hp_b, self._hp_a, samples_f, zi=self._hp_zi)
                filtered_int16 = np.clip(np.round(filtered_f), -32768, 32767).astype(np.int16)
                raw_bytes = filtered_int16.tobytes()
                energy = float(np.dot(filtered_f, filtered_f))
                rms = float(np.sqrt(energy / len(filtered_f)))
            except Exception:
                raw_bytes = frame.tobytes()
                samples_f = samples_raw.astype(np.float32, copy=False)
                energy = float(np.dot(samples_f, samples_f))
                rms = float(np.sqrt(energy / len(samples_f)))
        else:
            raw_bytes = frame.tobytes()
            samples_f = samples_raw.astype(np.float32, copy=False)
            energy = float(np.dot(samples_f, samples_f))
            rms = float(np.sqrt(energy / len(samples_f)))

        # 2. Salva sempre tutto il flusso nella registrazione integrale
        self._write_recording(raw_bytes)

        # 3. Analisi VAD ibrida adattiva (WebRTC VAD sul segnale pulito + Soglia RMS con Noise Floor)
        vad_active = False
        if self.vad and len(raw_bytes) == self.BYTES_PER_FRAME:
            try:
                vad_active = self.vad.is_speech(raw_bytes, self.SAMPLE_RATE)
            except Exception:
                vad_active = False

        # Adattamento continuo del rumore di fondo (noise floor)
        # Se non c'è voce evidente o picchi, aggiorna dolcemente la stima del rumore ambientale
        if not vad_active and rms < max(120.0, self._noise_floor * 1.5):
            self._noise_floor = self._noise_floor * 0.96 + rms * 0.04

        effective_noise = max(4.0, self._noise_floor)

        # Soglia minima dinamica calcolata sul rumore ambientale locale:
        # Permette di rilevare la voce anche a 10-15 RMS su laptop Windows a basso guadagno,
        # respingendo fruscii costanti e rumori di fondo ambientali.
        if vad_active:
            # Se WebRTC VAD rileva fonemi vocali, basta superare di poco il rumore di fondo
            min_rms_vad = max(10.0, effective_noise * 1.15)
            is_speech = (rms >= min_rms_vad)
        else:
            # Senza WebRTC VAD (o se incerto), serve una variazione energetica significativa sopra il noise floor
            multiplier = 1.5 if self.rms_threshold <= 45.0 else (2.3 if self.rms_threshold >= 105.0 else 1.8)
            abs_floor = 16.0 if self.rms_threshold <= 45.0 else (35.0 if self.rms_threshold >= 105.0 else 22.0)
            is_speech = (rms >= max(abs_floor, effective_noise * multiplier))

        now = time.time()

        with self._lock:
            if is_speech:
                self.is_voice_active = True
                if self._speech_frames_count == 0:
                    # Prependi i frame di pre-roll per non perdere l'attacco della prima parola
                    self._speech_buffer.extend(self._preroll_buffer)
                    self._preroll_buffer.clear()

                self._speech_buffer.append(raw_bytes)
                self._speech_frames_count += 1
                self._silence_frames_count = 0
                self._last_speech_time = now

                # Condizione 1: Finestra temporale continua max 7.5s (250 frame x 30ms = 7.5s)
                # Evita di accumulare fino a 20s prima di trascrivere se il relatore non fa pause
                if self._speech_frames_count >= 250:
                    logger.info(f"[AudioEngine] Finestra 7.5s completata ({self._speech_frames_count * 30 / 1000:.1f}s) -> Invio a Whisper Large Turbo")
                    overlap = list(self._speech_buffer[-8:]) if len(self._speech_buffer) >= 8 else []
                    self._flush_speech_buffer()
                    self._preroll_buffer.extend(overlap)

            else:
                self._preroll_buffer.append(raw_bytes)

                if self._speech_frames_count > 0:
                    # Il docente ha appena smesso di parlare o sta facendo una pausa
                    self._speech_buffer.append(raw_bytes)
                    self._silence_frames_count += 1

                    # Condizione 2: Soglia di Silenzio Adattiva a Bassa Latenza
                    # - Fast-start (primo blocco della lezione): basta una micro-pausa di 270ms (9 frame) dopo >=25 frame (~750ms) di voce
                    # - Parlato lungo (>= 150 frame ~= 4.5s): basta una pausa di 330ms (11 frame)
                    # - Parlato standard: attendiamo 390ms (13 frame) invece di 510ms
                    is_fast_start = (self._total_speech_flushes == 0 and self._speech_frames_count >= 25)
                    if is_fast_start:
                        silence_limit = 9
                    elif self._speech_frames_count >= 150:
                        silence_limit = 11
                    else:
                        silence_limit = 13

                    if self._silence_frames_count >= silence_limit:
                        self.is_voice_active = False
                        if self._speech_frames_count >= 12:
                            dur_s = (len(self._speech_buffer) * 30) / 1000.0
                            logger.info(
                                f"[AudioEngine] Innesco Pausa Parlato ({self._silence_frames_count * 30}ms silenzio "
                                f"dopo ~{dur_s:.1f}s di spiegazione, soglia={silence_limit * 30}ms) -> Invio a Whisper Large Turbo"
                            )
                            self._flush_speech_buffer()
                        elif self._silence_frames_count >= 35:
                            # Se il silenzio è superiore a 1.0s e la voce era brevissima
                            if self._speech_frames_count >= 8:  # ~240ms di voce
                                self._flush_speech_buffer()
                            else:
                                # Micro-rumore isolato (<240ms) seguito da silenzio
                                self._speech_buffer.clear()
                                self._speech_frames_count = 0
                                self._silence_frames_count = 0
                else:
                    self.is_voice_active = False
                    self._silence_frames_count += 1

    def _flush_speech_buffer(self) -> None:
        """Chiude il blocco vocale accumulato, applica AGC software anti-whisper e lo accoda per Whisper."""
        if not self._speech_buffer:
            return

        self._total_speech_flushes += 1
        chunk_bytes = b"".join(self._speech_buffer)
        self._speech_buffer.clear()
        self._speech_frames_count = 0
        self._silence_frames_count = 0

        # Verifica energia globale del blocco: scarta blocchi senza energia vocale
        samples = np.frombuffer(chunk_bytes, dtype=np.int16)
        if len(samples) < 1600:
            return
        samples_f = samples.astype(np.float32)
        seg_energy = float(np.dot(samples_f, samples_f))
        seg_rms = float(np.sqrt(seg_energy / len(samples_f)))
        min_rms = 4.0 if self.vad else 8.0
        if seg_rms < min_rms:
            logger.debug(f"[AudioEngine] Blocco ignorato (RMS troppo basso: {seg_rms:.1f})")
            return

        # Normalizzazione digitale automatica del guadagno (AGC) per microfoni a basso volume
        # (es. laptop Windows/Mac integrati con il prof a distanza o slider volume basso nel sistema operativo):
        # evita che Whisper riceva audio quasi inudibile (peak < 1000) che porta ad allucinazioni o parole perse.
        peak = int(np.max(np.abs(samples)))
        if 30 <= peak < 16000:
            target_peak = 18000.0
            boost = min(60.0, target_peak / max(1.0, float(peak)))
            if boost > 1.15:
                boosted = np.clip(np.round(samples.astype(np.float32) * boost), -32768, 32767).astype(np.int16)
                chunk_bytes = boosted.tobytes()

        duration_s = len(chunk_bytes) / (self.SAMPLE_RATE * 2)
        self.speech_queue.put(chunk_bytes)
        logger.info(f"[AudioEngine] Blocco vocale pronto: {len(chunk_bytes)} bytes (~{duration_s:.2f}s, RMS={seg_rms:.1f}) accodato per Whisper Large Turbo")

        if self.on_speech_chunk:
            elapsed = time.time() - self.started_at
            try:
                self.on_speech_chunk(chunk_bytes, elapsed)
            except Exception as e:
                logger.error(f"[AudioEngine] Errore in on_speech_chunk callback: {e}")

    def _close_stream_safely(self, stream) -> None:
        """
        Chiude uno stream PortAudio/CoreAudio/WASAPI senza mai bloccare il salvataggio del WAV:
        1. Attende brevemente (fino a 250ms) che CallbackStop fermi lo stream dal thread CoreAudio.
        2. Esegue abort/close in un thread helper con timeout massimo di 1.0s, così stop() esporta
           sempre l'audio registrato su disco anche se il driver hardware macOS impiega tempo.
        """
        if stream is None:
            return

        # 1. Se il callback sta fermando lo stream via CallbackStop, attendi fino a 250ms
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            try:
                if not getattr(stream, "active", False):
                    break
            except Exception:
                break
            time.sleep(0.01)

        def _do_close() -> None:
            try:
                if getattr(stream, "active", True):
                    if hasattr(stream, "abort"):
                        stream.abort()
                    else:
                        stream.stop()
            except Exception:
                try:
                    stream.stop()
                except Exception:
                    pass
            try:
                stream.close()
            except Exception as exc:
                logger.warning(f"[AudioEngine] Errore chiusura stream: {exc}")

        closer = threading.Thread(target=_do_close, daemon=True, name="audio-stream-closer")
        closer.start()
        closer.join(timeout=1.0)
        if closer.is_alive():
            logger.warning(
                "[AudioEngine] Chiusura hardware stream lenta (>1s): procedo subito con salvataggio WAV."
            )

    def pause(self) -> Optional[bytes]:
        """
        Sospende la cattura audio del microfono e restituisce subito l'ultimo frammento:
        1. Imposta is_paused = True (callback PortAudio diventa no-op immediato).
        2. Chiude pulitamente lo stream hardware (abort + close) evitando il deadlock CoreAudio stop->start->stop.
        3. Drena in modo limitato i frammenti già in coda e restituisce il parlato residuo.
        """
        if not self.is_running or self.is_paused:
            return None

        self.is_paused = True
        self.is_voice_active = False

        stream = self._stream
        self._stream = None
        if stream is not None:
            self._was_hardware_stream = True
            self._close_stream_safely(stream)
            logger.info(f"[AudioEngine] Stream audio hardware in PAUSA (chiuso pulitamente) per lezione {self.lecture_id}")

        # Drena in modo limitato i frammenti già presenti in coda prima della pausa
        for _ in range(500):
            try:
                chunk = self._raw_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._process_mono_chunk(chunk, force=True)
            except Exception:
                break

        self._reset_filter_state()

        # Estrai l'eventuale parlato residuo rimasto nel buffer in modo thread-safe
        last_audio = None
        with self._lock:
            if self._speech_buffer and self._speech_frames_count >= 15:  # Almeno ~450ms di voce
                last_audio = b"".join(self._speech_buffer)
            self._speech_buffer.clear()
            self._speech_frames_count = 0
            self._silence_frames_count = 0

        logger.info(f"[AudioEngine] Motori in PAUSA. Ultimo chunk estratto: {len(last_audio) if last_audio else 0} bytes")
        return last_audio

    def resume(self) -> bool:
        """Riattiva la cattura audio hardware dal microfono aprendo uno stream pulito."""
        if not self.is_running or not self.is_paused:
            return False

        needs_reopen = False
        if self._stream is not None:
            try:
                self._stream.start()
                logger.info(f"[AudioEngine] Stream audio hardware RIPRESO per lezione {self.lecture_id}")
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore ripresa stream con start(): {e}, riapro...")
                self._close_stream_safely(self._stream)
                self._stream = None
                needs_reopen = True
        elif getattr(self, "_was_hardware_stream", False) and sd is not None:
            needs_reopen = True

        if needs_reopen:
            reopened = False
            for attempt in range(3):
                if attempt > 0:
                    time.sleep(0.120)
                try:
                    self._stream = self._open_stream()
                    if self._stream:
                        self._stream.start()
                        reopened = True
                        logger.info(f"[AudioEngine] Stream audio hardware riaperto con successo al tentativo {attempt + 1}")
                        break
                except Exception as e2:
                    logger.warning(f"[AudioEngine] Tentativo {attempt + 1} riapertura fallito: {e2}")
                    self._close_stream_safely(self._stream)
                    self._stream = None
            if not reopened:
                logger.error("[AudioEngine] Impossibile riaprire lo stream: tutti i tentativi falliti")
                return False

        self.is_paused = False
        self.is_voice_active = False
        self._reset_filter_state()
        with self._lock:
            self._preroll_buffer.clear()
            self._speech_buffer.clear()
            self._speech_frames_count = 0
            self._silence_frames_count = 0
            self._pending = np.zeros(0, dtype=np.float32)
        return True

    def _run_simulated_mic(self) -> None:
        """Fallback in caso di assenza microfono hardware (rispetta la pausa)."""
        blank_frame = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16).tobytes()
        while self.is_running:
            if not self.is_paused:
                self._write_recording(blank_frame)
            time.sleep(0.030)

    def stop(self, audio_format: Optional[str] = None) -> str:
        """
        Interrompe la registrazione in modo istantaneo e non bloccante (abort + close) ed esporta
        un file .wav completo nella directory audio.
        La copia avviene a blocchi da 1 MB e il file viene scritto come .part e rinominato atomicamente.
        Restituisce il percorso assoluto del file generato.
        """
        # 1. Ferma subito i flag così il callback PortAudio termina immediatamente senza accodare altro
        self.is_running = False
        self.is_paused = True
        self.is_voice_active = False

        # 2. Chiudi subito lo stream hardware con abort() + close() (evita deadlock CoreAudio su macOS)
        stream = self._stream
        self._stream = None
        self._was_hardware_stream = False
        self._close_stream_safely(stream)

        # 3. Attendi uscita del worker thread prima del drenaggio finale
        if self._worker_thread and self._worker_thread.is_alive():
            try:
                self._worker_thread.join(timeout=0.5)
            except Exception:
                pass
            self._worker_thread = None

        if self._mock_mic_thread:
            self._mock_mic_thread.join(timeout=0.5)

        # 4. Drena in modo limitato gli ultimi frammenti già in coda (force=True perché is_running=False)
        for _ in range(500):
            try:
                chunk = self._raw_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._process_mono_chunk(chunk, force=True)
            except Exception:
                break

        # 5. Svuota l'ultimo frammento residuo se ha contenuto vocale
        if self._speech_frames_count >= 20:
            self._flush_speech_buffer()
        else:
            self._speech_buffer.clear()
            self._speech_frames_count = 0

        audio_dir = get_audio_dir()
        os.makedirs(audio_dir, exist_ok=True)
        timestamp_str = int(time.time())
        output_filename = f"lecture_{self.lecture_id}_{timestamp_str}.wav"
        output_path = os.path.join(audio_dir, output_filename)
        part_path = output_path + ".part"

        with self._lock:
            total = self._recorded_bytes
            logger.info(f"[AudioEngine] Esportazione {total} bytes di audio PCM in {output_path}")
            try:
                gain = 1.0
                if self._temp_pcm_file and not self._temp_pcm_file.closed:
                    self._temp_pcm_file.flush()
                    self._temp_pcm_file.seek(0)
                    # 1. Scansione rapida del picco massimo per normalizzazione volume riascolto (Playback AGC)
                    max_abs = 0
                    while True:
                        block = self._temp_pcm_file.read(1024 * 1024)
                        if not block:
                            break
                        i16_block = np.frombuffer(block, dtype=np.int16)
                        if len(i16_block):
                            b_max = int(np.max(np.abs(i16_block)))
                            if b_max > max_abs:
                                max_abs = b_max

                    # Se la registrazione ha ampiezza ridotta (microfoni a basso guadagno o aula grande),
                    # applichiamo un guadagno trasparente fino a ~24000 peak evitando di amplificare silenzio digitale puro (< 80)
                    if 80 <= max_abs < 18000:
                        gain = min(80.0, 24000.0 / float(max_abs))
                        logger.info(f"[AudioEngine] Normalizzazione volume riascolto applicata: peak {max_abs} -> ~24000 (gain={gain:.2f}x)")

                    self._temp_pcm_file.seek(0)

                with wave.open(part_path, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(self.SAMPLE_RATE)
                    if self._temp_pcm_file and not self._temp_pcm_file.closed:
                        while True:
                            block = self._temp_pcm_file.read(1024 * 1024)
                            if not block:
                                break
                            if gain != 1.0:
                                i16_block = np.frombuffer(block, dtype=np.int16)
                                boosted = np.clip(np.round(i16_block.astype(np.float32) * gain), -32768, 32767).astype(np.int16)
                                wf.writeframesraw(boosted.tobytes())
                            else:
                                wf.writeframesraw(block)
                # Sostituzione atomica con retry difensivo per blocchi file su Windows
                replaced = False
                for _ in range(5):
                    try:
                        os.replace(part_path, output_path)
                        replaced = True
                        break
                    except PermissionError:
                        time.sleep(0.05)
                if not replaced and os.path.exists(part_path):
                    os.replace(part_path, output_path)
            finally:
                if self._temp_pcm_file:
                    try:
                        self._temp_pcm_file.close()
                    except Exception:
                        pass
                if self._temp_pcm_path and os.path.exists(self._temp_pcm_path):
                    try:
                        os.remove(self._temp_pcm_path)
                    except OSError:
                        pass
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

        logger.info(f"[AudioEngine] Registrazione WAV esportata: {output_path}")

        # Converte nel formato finale salva-spazio M4A (a meno di flush asincrono temporaneo)
        final_path = output_path
        if audio_format != "wav":
            try:
                final_path = compress_wav_to_m4a(output_path, bitrate=48000, remove_original=True)
            except Exception as e:
                logger.error(f"[AudioEngine] Fallback su WAV non compresso ({e})")
                final_path = output_path

        logger.info(f"[AudioEngine] Registrazione finalizzata con successo: {final_path}")
        return final_path


def compress_wav_to_m4a(wav_path: str, bitrate: int = 48000, remove_original: bool = True) -> str:
    """
    Converte un file WAV PCM mono/stereo in formato M4A / AAC a basso bitrate (default 48 kbps)
    usando PyAV in streaming a blocchi, riducendo lo spazio occupato di circa l'85% preservando
    la chiarezza vocale.
    
    Non carica mai l'intera lezione in memoria (consumo RAM costante O(1)).
    Scrive su file .part e rinomina atomicamente a conversione completata.
    Se remove_original è True, elimina il file .wav originale solo se la conversione ha successo.
    Aggiorna anche l'eventuale file di cache dei timestamp (.timestamps.json).
    Restituisce il percorso del file finale (.m4a se riuscita, o il .wav originario in caso di fallback).
    """
    if not wav_path or not os.path.isfile(wav_path) or os.path.getsize(wav_path) <= 44:
        return wav_path

    if wav_path.lower().endswith(".m4a"):
        return wav_path

    base, _ = os.path.splitext(wav_path)
    m4a_path = base + ".m4a"
    part_path = m4a_path + ".part"

    try:
        import av

        with av.open(wav_path, mode="r") as in_container:
            if not in_container.streams.audio:
                logger.warning(f"[AudioEngine] Nessun flusso audio trovato in {wav_path}")
                return wav_path
            in_stream = in_container.streams.audio[0]

            with av.open(part_path, mode="w", format="mp4") as out_container:
                out_stream = out_container.add_stream("aac", rate=16000)
                out_stream.format = "fltp"
                out_stream.bit_rate = int(bitrate)
                out_stream.layout = "mono"
                try:
                    out_stream.codec_context.thread_count = 1
                except Exception:
                    pass

                resampler = av.AudioResampler(
                    format=out_stream.format,
                    layout=out_stream.layout,
                    rate=out_stream.rate,
                )

                frames_encoded = 0
                for frame in in_container.decode(in_stream):
                    frames_encoded += 1
                    for resampled in resampler.resample(frame):
                        for packet in out_stream.encode(resampled):
                            out_container.mux(packet)

                if frames_encoded > 0:
                    for resampled in resampler.resample(None):
                        for packet in out_stream.encode(resampled):
                            out_container.mux(packet)

                    for packet in out_stream.encode(None):
                        out_container.mux(packet)

        # Rilascia esplicitamente i puntatori C di PyAV/FFmpeg e forza gc per sbloccare i file su Windows
        try:
            del in_stream, out_stream, in_container, out_container
            import gc
            gc.collect()
        except Exception:
            pass

        if not os.path.exists(part_path) or os.path.getsize(part_path) == 0:
            logger.warning(f"[AudioEngine] File compresso vuoto o non generato per {wav_path}")
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            return wav_path

        # Sostituzione atomica del file compresso con retry difensivo su Windows
        replaced = False
        for _ in range(5):
            try:
                os.replace(part_path, m4a_path)
                replaced = True
                break
            except PermissionError:
                time.sleep(0.05)
        if not replaced and os.path.exists(part_path):
            os.replace(part_path, m4a_path)

        orig_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
        new_size = os.path.getsize(m4a_path) if os.path.exists(m4a_path) else 0
        saved_bytes = orig_size - new_size
        ratio = (1.0 - (new_size / orig_size)) * 100 if orig_size > 0 else 0
        logger.info(
            f"[AudioEngine] Compressione salva-spazio completata: {orig_size / (1024*1024):.1f} MB -> "
            f"{new_size / (1024*1024):.1f} MB (-{ratio:.1f}%, liberati {saved_bytes / (1024*1024):.1f} MB)"
        )

        # Se richiesto, rimuovi il WAV originale con retry su Windows
        if remove_original and os.path.exists(wav_path) and os.path.abspath(wav_path) != os.path.abspath(m4a_path):
            for _ in range(5):
                try:
                    os.remove(wav_path)
                    break
                except OSError:
                    time.sleep(0.05)

        # Se esiste una cache di timestamp pre-calcolata, migrala al nuovo nome file
        old_cache = wav_path + ".timestamps.json"
        new_cache = m4a_path + ".timestamps.json"
        if os.path.isfile(old_cache) and not os.path.exists(new_cache):
            try:
                os.replace(old_cache, new_cache)
            except Exception:
                pass

        return m4a_path

    except Exception as exc:
        logger.error(f"[AudioEngine] Errore compressione M4A per {wav_path}: {exc}. Mantengo WAV originale.")
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        return wav_path
