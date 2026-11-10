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


def _resolve_input_device() -> Optional[object]:
    """Dispositivo di input da SBOBINAI_INPUT_DEVICE (indice numerico o sottostringa del nome)."""
    wanted = (os.environ.get("SBOBINAI_INPUT_DEVICE") or "").strip()
    if not wanted or sd is None:
        return None
    if wanted.isdigit():
        return int(wanted)
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0 and wanted.lower() in str(dev.get("name", "")).lower():
                return idx
    except Exception:
        pass
    logger.warning(f"[AudioEngine] Dispositivo '{wanted}' non trovato: uso il microfono predefinito.")
    return None


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

        # Coda vocale per Whisper Large-v3-Turbo
        self.speech_queue: queue.Queue[bytes] = queue.Queue()

        # VAD
        self.vad = None
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
        Apre lo stream di input con negoziazione progressiva:
        1. 16 kHz Mono (standard ideale Whisper/VAD)
        2. 16 kHz Stereo (richiesto da molti driver WASAPI / USB su Windows)
        3. Frequenza nativa (es. 48/44.1 kHz) Mono / Stereo con ricampionamento in streaming
        """
        device = _resolve_input_device()
        max_ch = 1
        native_sr = 48000.0
        try:
            info = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
            max_ch = max(1, int(info.get("max_input_channels", 1) or 1))
            native_sr = float(info.get("default_samplerate", 48000.0) or 48000.0)
        except Exception as q_err:
            logger.warning(f"[AudioEngine] Query dispositivo input fallita ({q_err}), uso valori standard.")

        import sys
        if sys.platform == "win32":
            # Su Windows, i driver hardware (WASAPI / Realtek / Intel) operano quasi sempre a 48kHz o 44.1kHz stereo.
            # Richiedere 16kHz al driver WASAPI genera timeout COM di 4-5s per ciascun tentativo fallito (~20s di freeze iniziale).
            # Diamo massima priorità al formato nativo del dispositivo (apertura istantanea in <50ms),
            # demandando il ricampionamento a 16 kHz ad alta precisione a _StreamResampler.
            candidates = [
                (int(native_sr), min(2, max_ch)),
                (int(native_sr), 1),
                (48000, min(2, max_ch)),
                (48000, 1),
                (44100, min(2, max_ch)),
                (44100, 1),
                (self.SAMPLE_RATE, min(2, max_ch)),
                (self.SAMPLE_RATE, 1),
            ]
        else:
            # Su macOS (CoreAudio) e Linux, la conversione a 16 kHz mono hardware/sistema è nativa e istantanea
            candidates = [
                (self.SAMPLE_RATE, 1),
                (self.SAMPLE_RATE, min(2, max_ch)),
                (int(native_sr), 1),
                (int(native_sr), min(2, max_ch)),
                (48000, 1),
                (48000, min(2, max_ch)),
                (44100, 1),
                (44100, min(2, max_ch)),
            ]
        # Rimuovi duplicati mantenendo l'ordine di priorità
        unique_candidates = []
        seen = set()
        for sr, ch in candidates:
            if (sr, ch) not in seen:
                seen.add((sr, ch))
                unique_candidates.append((sr, ch))

        last_err = None
        for sr, ch in unique_candidates:
            # Prova prima con blocksize calibrato a 30ms, poi con None (buffer nativo hardware consigliato da WASAPI)
            for b_size in (int(round(sr * (self.FRAME_MS / 1000.0))), None):
                try:
                    stream = sd.InputStream(
                        device=device,
                        samplerate=sr,
                        channels=ch,
                        dtype="int16",
                        blocksize=b_size,
                        callback=self._raw_stream_callback,
                    )
                    self.device_samplerate = float(sr)
                    self.device_channels = ch
                    if sr != self.SAMPLE_RATE:
                        logger.info(f"[AudioEngine] Dispositivo aperto a {sr:.0f} Hz ({ch} canali, blocksize={b_size}) con ricampionamento a 16 kHz.")
                        self._resampler = _StreamResampler(sr, self.SAMPLE_RATE)
                    else:
                        logger.info(f"[AudioEngine] Dispositivo aperto a 16 kHz nativi ({ch} canali, blocksize={b_size}).")
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

        # Prova ad aprire il microfono reale con sounddevice
        if sd:
            try:
                self._stream = self._open_stream()
                self._stream.start()
                logger.info(f"[AudioEngine] Microfono hardware avviato ({self.device_samplerate:.0f}Hz, {self.device_channels}ch) per lezione {self.lecture_id}")
                return True
            except Exception as e:
                self._stream = None
                logger.warning(f"[AudioEngine] Impossibile aprire stream audio hardware ({e}). Uso fallback simulato.")

        # Fallback generatore di frame silenziosi (se il mic hardware non è accessibile)
        self._mock_mic_thread = threading.Thread(target=self._run_simulated_mic, daemon=True)
        self._mock_mic_thread.start()
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
        """Callback universale per cattura audio: gestisce multicanale, frequenze native e ricampionamento."""
        if status:
            logger.debug(f"[AudioEngine] Status sounddevice: {status}")
        if not self.is_running or self.is_paused:
            return

        # 1. Downmix a mono se il flusso hardware è multicanale (stereo / array microfonico)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono_int16 = np.mean(indata.astype(np.int32), axis=1).astype(np.int16)
        else:
            mono_int16 = indata.reshape(-1).astype(np.int16, copy=False)

        # 2. Se la frequenza nativa è diversa da 16 kHz, ricampiona
        if self._resampler is not None:
            out = self._resampler.process(mono_int16)
            buf = np.concatenate([self._pending, out]) if len(self._pending) else out
            n_frames = len(buf) // self.SAMPLES_PER_FRAME
            for i in range(n_frames):
                seg = buf[i * self.SAMPLES_PER_FRAME:(i + 1) * self.SAMPLES_PER_FRAME]
                frame = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
                self._process_16k_mono_frame(frame)
            self._pending = buf[n_frames * self.SAMPLES_PER_FRAME:].copy()
        else:
            if len(mono_int16) == self.SAMPLES_PER_FRAME:
                self._process_16k_mono_frame(mono_int16)
            else:
                buf = np.concatenate([self._pending, mono_int16.astype(np.float32, copy=False)]) if len(self._pending) else mono_int16.astype(np.float32, copy=False)
                n_frames = len(buf) // self.SAMPLES_PER_FRAME
                for i in range(n_frames):
                    seg = buf[i * self.SAMPLES_PER_FRAME:(i + 1) * self.SAMPLES_PER_FRAME]
                    frame = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
                    self._process_16k_mono_frame(frame)
                self._pending = buf[n_frames * self.SAMPLES_PER_FRAME:].copy()

    # Alias per compatibilità con test unitari e chiamate dirette
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._raw_stream_callback(indata, frames, time_info, status)

    def _native_rate_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        self._raw_stream_callback(indata, frames, time_info, status)

    def _process_16k_mono_frame(self, frame: np.ndarray) -> None:
        """Elabora un singolo frame mono da 480 campioni int16 a 16 kHz (filtro HP, VAD, accumulo)."""
        if not self.is_running or self.is_paused:
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

        # 3. Analisi VAD ibrida (WebRTC VAD sul segnale pulito + Soglia RMS energetica)
        vad_active = False
        if self.vad and len(raw_bytes) == self.BYTES_PER_FRAME:
            try:
                vad_active = self.vad.is_speech(raw_bytes, self.SAMPLE_RATE)
            except Exception:
                vad_active = False
            # Se WebRTC VAD rileva fonemi vocali, confermiamo con una soglia minima anti-rumore (35 RMS)
            # Se WebRTC VAD è incerto, usiamo la soglia energetica rms_threshold
            is_speech = (vad_active and rms >= 35.0) or (rms >= self.rms_threshold)
        else:
            # Fallback energetico quando webrtcvad non è presente (es. Windows senza wheel C++)
            is_speech = (rms >= self.rms_threshold)

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

                # Condizione 1: Finestra temporale continua di 20 secondi (~666 frame x 30ms = 20.0s)
                if self._speech_frames_count >= 666:
                    logger.info(f"[AudioEngine] Finestra 20s completata ({self._speech_frames_count * 30 / 1000:.1f}s) -> Invio a Whisper Large Turbo")
                    overlap = list(self._speech_buffer[-6:]) if len(self._speech_buffer) >= 6 else []
                    self._flush_speech_buffer()
                    self._preroll_buffer.extend(overlap)

            else:
                self._preroll_buffer.append(raw_bytes)

                if self._speech_frames_count > 0:
                    # Il docente ha appena smesso di parlare o sta facendo una pausa
                    self._speech_buffer.append(raw_bytes)
                    self._silence_frames_count += 1

                    # Condizione 2: Soglia di Silenzio Adattiva
                    # - parlato lungo (>= 250 frame ~= 7.5s): basta una pausa di 360ms (12 frame)
                    # - parlato breve (50..249 frame): attendiamo 510ms (17 frame)
                    silence_limit = 12 if self._speech_frames_count >= 250 else 17

                    if self._silence_frames_count >= silence_limit:
                        self.is_voice_active = False
                        if self._speech_frames_count >= 50:
                            dur_s = (len(self._speech_buffer) * 30) / 1000.0
                            logger.info(
                                f"[AudioEngine] Innesco Pausa Docente ({self._silence_frames_count * 30}ms silenzio "
                                f"dopo ~{dur_s:.1f}s di spiegazione, soglia={silence_limit * 30}ms) -> Invio a Whisper Large Turbo"
                            )
                            self._flush_speech_buffer()
                        elif self._silence_frames_count >= 40:
                            # Se il silenzio è superiore a 1.2s e la voce era brevissima (<1.5s)
                            if self._speech_frames_count >= 30:  # ~1.0s di voce
                                self._flush_speech_buffer()
                            else:
                                # Rumore isolato o colpo sul tavolo: scarta senza affaticare Whisper
                                self._speech_buffer.clear()
                                self._speech_frames_count = 0
                                self._silence_frames_count = 0
                else:
                    self.is_voice_active = False
                    self._silence_frames_count += 1

    def _flush_speech_buffer(self) -> None:
        """Chiude il blocco vocale accumulato e lo accoda per Whisper."""
        if not self._speech_buffer:
            return

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
        min_rms = 15.0 if self.vad else 25.0
        if seg_rms < min_rms:
            logger.debug(f"[AudioEngine] Blocco ignorato (RMS troppo basso: {seg_rms:.1f})")
            return

        duration_s = len(chunk_bytes) / (self.SAMPLE_RATE * 2)
        self.speech_queue.put(chunk_bytes)
        logger.info(f"[AudioEngine] Blocco vocale pronto: {len(chunk_bytes)} bytes (~{duration_s:.2f}s, RMS={seg_rms:.1f}) accodato per Whisper Large Turbo")

        if self.on_speech_chunk:
            elapsed = time.time() - self.started_at
            try:
                self.on_speech_chunk(chunk_bytes, elapsed)
            except Exception as e:
                logger.error(f"[AudioEngine] Errore in on_speech_chunk callback: {e}")

    def pause(self) -> Optional[bytes]:
        """
        Sospende la cattura audio del microfono e restituisce subito l'ultimo frammento:
        1. Arresta l'ascolto hardware (sounddevice stream).
        2. Raccoglie tutto il parlato ancora nel buffer (anche solo 2-5 secondi) per Whisper.
        3. Imposta is_paused = True (0% CPU, 0% microfono).
        """
        if not self.is_running or self.is_paused:
            return None

        self.is_paused = True
        self.is_voice_active = False
        self._reset_filter_state()

        if self._stream:
            try:
                self._stream.stop()
                logger.info(f"[AudioEngine] Stream audio hardware in PAUSA per lezione {self.lecture_id}")
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore arresto stream per pausa: {e}")

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
        """Riattiva la cattura audio hardware dal microfono."""
        if not self.is_running or not self.is_paused:
            return False

        if self._stream:
            try:
                self._stream.start()
                logger.info(f"[AudioEngine] Stream audio hardware RIPRESO per lezione {self.lecture_id}")
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore ripresa stream con start(): {e}, provo riapertura...")
                try:
                    self._open_stream()
                    if self._stream:
                        self._stream.start()
                except Exception as e2:
                    logger.error(f"[AudioEngine] Impossibile riaprire lo stream: {e2}")
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
        Interrompe la registrazione ed esporta un file .wav completo nella directory audio.
        La copia avviene a blocchi da 1 MB (nessuna copia dell'intera lezione in RAM) e il file
        viene scritto come .part e poi rinominato atomicamente.
        Restituisce il percorso assoluto del file generato.
        """
        self.is_running = False
        self.is_voice_active = False

        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore chiusura stream: {e}")
            self._stream = None

        if self._mock_mic_thread:
            self._mock_mic_thread.join(timeout=0.5)

        # Svuota l'ultimo frammento residuo se ha contenuto vocale
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
                if self._temp_pcm_file and not self._temp_pcm_file.closed:
                    self._temp_pcm_file.flush()
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
