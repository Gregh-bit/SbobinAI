"""
audio_engine.py — SbobinAI Live Audio Engine (Zero I/O in RAM)
Gestisce la cattura audio in tempo reale dal microfono:
 - Campionamento a 16kHz, mono, PCM 16-bit (standard nativo Whisper e WebRTC VAD)
 - Voice Activity Detection (VAD) ibrido (WebRTC VAD + RMS energetico)
 - Buffer audio interamente in RAM (io.BytesIO) durante tutta la registrazione
 - Accumulo a blocchi per Whisper Large-v3-Turbo Q4:
     * Finestra temporale massima di 20 secondi continui
     * Innesco naturale su pausa stabile del docente (>500ms di silenzio)
 - Esportazione del file finale .wav su disco (/data/audio/) solo alla chiusura della sessione
"""
from __future__ import annotations

import collections
import io
import logging
import os
import queue
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

logger = logging.getLogger("sbobinai.audio")

# ── Percorsi Dinamici ──────────────────────────────────────────────────────────
from config import get_audio_dir


class AudioEngine:
    """
    Cattura l'audio live dal microfono, esegue VAD ed accumula i chunk in RAM.
    Zero I/O su disco durante la sessione di registrazione.
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

        # In-RAM storage per l'intera lezione (Zero I/O su disco durante il live)
        self._ram_buffer = io.BytesIO()
        self._lock = threading.Lock()

        # Coda vocale per Whisper Large-v3-Turbo Q4
        self.speech_queue: queue.Queue[bytes] = queue.Queue()

        # VAD
        self.vad = None
        if webrtcvad:
            try:
                self.vad = webrtcvad.Vad(vad_aggressiveness)
            except Exception as e:
                logger.warning(f"[AudioEngine] Inizializzazione VAD fallita: {e}")

        # Soglia RMS dinamica basata su vad_aggressiveness:
        # 1: permissivo (rms > 140)
        # 2: standard (rms > 190)
        # 3: reattivo/anti-rumore (rms > 260)
        self.rms_threshold = 140.0 if vad_aggressiveness == 1 else (260.0 if vad_aggressiveness == 3 else 190.0)

        # Buffer di pre-roll (ultimi ~240ms / 8 frame prima dell'attacco della voce)
        self._preroll_buffer: collections.deque[bytes] = collections.deque(maxlen=8)

        # Buffer accumulo parlato per Whisper Large Turbo
        self._speech_buffer: list[bytes] = []
        self._speech_frames_count: int = 0
        self._silence_frames_count: int = 0
        self._last_speech_time: float = 0.0

        # Filtro High-Pass anti-rumble (80 Hz) con Apple Accelerate / scipy.signal
        self._hp_enabled = False
        self._hp_zi = None
        self._hp_b = None
        self._hp_a = None
        try:
            import scipy.signal as signal
            # Butterworth 2° ordine passa-alto a 80 Hz su fs=16000 Hz (frequenza di taglio = 80 Hz)
            self._hp_b, self._hp_a = signal.butter(2, 80.0 / (self.SAMPLE_RATE / 2.0), btype="highpass")
            self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
            self._hp_enabled = True
            logger.info("[AudioEngine] Filtro High-Pass 80Hz (Apple Accelerate) inizializzato con successo.")
        except Exception as e:
            logger.warning(f"[AudioEngine] Inizializzazione filtro High-Pass fallita ({e}). Uso segnale diretto.")

        # Stream sounddevice
        self._stream: Optional[sd.InputStream] = None
        self._mock_mic_thread: Optional[threading.Thread] = None

    def _reset_filter_state(self) -> None:
        """Azzera lo stato del filtro IIR per evitare discontinuità o residui tra registrazioni/pause."""
        if self._hp_enabled and self._hp_b is not None:
            try:
                import scipy.signal as signal
                self._hp_zi = signal.lfilter_zi(self._hp_b, self._hp_a) * 0.0
            except Exception:
                pass

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
                self._stream = sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    dtype="int16",
                    blocksize=self.SAMPLES_PER_FRAME,
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info(f"[AudioEngine] Microfono hardware avviato (16kHz mono) per lezione {self.lecture_id}")
                return True
            except Exception as e:
                logger.warning(f"[AudioEngine] Impossibile aprire stream audio hardware ({e}). Uso fallback simulato.")

        # Fallback generatore di frame silenziosi in RAM (se il mic hardware non è accessibile)
        self._mock_mic_thread = threading.Thread(target=self._run_simulated_mic, daemon=True)
        self._mock_mic_thread.start()
        logger.info(f"[AudioEngine] Fallback simulato avviato per lezione {self.lecture_id}")
        return True

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback di sounddevice chiamata ogni 30ms con 480 campioni int16."""
        if not self.is_running or self.is_paused:
            return

        # 1. Filtro High-Pass anti-rumble (80 Hz) con Apple Accelerate / scipy
        # Rimuove ronzii di fondo, ventole del Mac e colpi sulla scrivania (<2µs overhead)
        samples_raw = indata.reshape(-1)
        if self._hp_enabled and self._hp_zi is not None:
            try:
                import scipy.signal as signal
                samples_f = samples_raw.astype(np.float32, copy=False)
                filtered_f, self._hp_zi = signal.lfilter(self._hp_b, self._hp_a, samples_f, zi=self._hp_zi)
                filtered_int16 = np.clip(np.round(filtered_f), -32768, 32767).astype(np.int16)
                raw_bytes = filtered_int16.tobytes()
                # Calcolo energetico RMS istantaneo via Apple Accelerate cblas_sdot (zero allocazione heap)
                energy = float(np.dot(filtered_f, filtered_f))
                rms = float(np.sqrt(energy / len(filtered_f)))
            except Exception:
                raw_bytes = indata.tobytes()
                samples_f = samples_raw.astype(np.float32, copy=False)
                energy = float(np.dot(samples_f, samples_f))
                rms = float(np.sqrt(energy / len(samples_f)))
        else:
            raw_bytes = indata.tobytes()
            samples_f = samples_raw.astype(np.float32, copy=False)
            energy = float(np.dot(samples_f, samples_f))
            rms = float(np.sqrt(energy / len(samples_f)))

        # 2. Salva sempre tutto il flusso nel buffer RAM (Zero I/O su disco durante il live)
        with self._lock:
            self._ram_buffer.write(raw_bytes)

        # 3. Analisi VAD ibrida (WebRTC VAD sul segnale pulito + Soglia RMS energetica)
        vad_active = False
        if self.vad and len(raw_bytes) == self.BYTES_PER_FRAME:
            try:
                vad_active = self.vad.is_speech(raw_bytes, self.SAMPLE_RATE)
            except Exception:
                vad_active = False
            is_speech = vad_active and (rms > self.rms_threshold)
        else:
            is_speech = (rms > max(260.0, self.rms_threshold))

        now = time.time()

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
            # Invia il blocco a Whisper Large senza attendere pause prolungate
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

                # Condizione 2: Soglia di Silenzio Adattiva (Adaptive Silence Trigger)
                # - Se il docente ha spiegato a lungo (>= 250 frame ~= 7.5s), una pausa di respiro di 360ms (12 frame)
                #   è sufficiente per considerare concluso il periodo ed emettere il testo senza attendere oltre.
                # - Per parlato più breve (50..249 frame), attendiamo 510ms (17 frame) per evitare frammentazioni.
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
        """Chiude il blocco vocale accumulato e lo accoda per Whisper Large-v3-Turbo Q4."""
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
        samples_f = samples.astype(np.float32, copy=False)
        seg_energy = float(np.dot(samples_f, samples_f))
        seg_rms = float(np.sqrt(seg_energy / len(samples_f)))
        min_rms = 160.0 if self.vad else 240.0
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
        2. Raccoglie tutto il parlato ancora nel buffer (anche solo 2-5 secondi) per Whisper Large.
        3. Imposta is_paused = True (Calore Zero: 0% CPU, 0% microfono).
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

        # Estrai l'eventuale parlato residuo rimasto nel buffer
        last_audio = None
        if self._speech_buffer and self._speech_frames_count >= 15:  # Almeno ~450ms di voce
            last_audio = b"".join(self._speech_buffer)

        self._speech_buffer.clear()
        self._speech_frames_count = 0
        self._silence_frames_count = 0

        logger.info(f"[AudioEngine] Motori in PAUSA (Calore Zero). Ultimo chunk estratto: {len(last_audio) if last_audio else 0} bytes")
        return last_audio

    def resume(self) -> bool:
        """Riattiva la cattura audio hardware dal microfono."""
        if not self.is_running or not self.is_paused:
            return False

        self.is_paused = False
        self.is_voice_active = False
        self._reset_filter_state()
        self._preroll_buffer.clear()
        self._speech_buffer.clear()
        self._speech_frames_count = 0
        self._silence_frames_count = 0

        if self._stream:
            try:
                self._stream.start()
                logger.info(f"[AudioEngine] Stream audio hardware RIPRESO per lezione {self.lecture_id}")
                return True
            except Exception as e:
                logger.warning(f"[AudioEngine] Errore ripresa stream: {e}")
                return False
        return True

    def _run_simulated_mic(self) -> None:
        """Fallback in caso di assenza microfono hardware."""
        blank_frame = np.zeros(self.SAMPLES_PER_FRAME, dtype=np.int16).tobytes()
        while self.is_running:
            with self._lock:
                self._ram_buffer.write(blank_frame)
            time.sleep(0.030)

    def stop(self) -> str:
        """
        Interrompe la registrazione, estrae il buffer dalla RAM ed esporta
        un file .wav completo nella directory /data/audio/.
        Restituisce il percorso assoluto del file generato.
        """
        self.is_running = False
        self.is_voice_active = False

        if self._stream:
            try:
                self._stream.stop()
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

        # Esporta il file WAV su disco
        audio_dir = get_audio_dir()
        os.makedirs(audio_dir, exist_ok=True)
        timestamp_str = int(time.time())
        output_filename = f"lecture_{self.lecture_id}_{timestamp_str}.wav"
        output_path = os.path.join(audio_dir, output_filename)

        with self._lock:
            pcm_data = self._ram_buffer.getvalue()
            self._ram_buffer.close()

        logger.info(f"[AudioEngine] Esportazione {len(pcm_data)} bytes di audio PCM in {output_path}")

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm_data)

        logger.info(f"[AudioEngine] Registrazione salvata con successo: {output_path}")
        return output_path

    def get_recorded_duration(self) -> float:
        """Calcola la durata in secondi dell'audio presente nel buffer RAM."""
        with self._lock:
            total_bytes = self._ram_buffer.tell()
        # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        return total_bytes / (self.SAMPLE_RATE * 2)
