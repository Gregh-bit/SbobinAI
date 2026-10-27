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

        # Registrazione integrale: RAM fino a 64 MB, poi spool automatico su file temporaneo
        self._recording = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_RAM_BYTES, mode="w+b",
                                                        prefix="sbobinai_rec_", suffix=".pcm")
        self._recorded_bytes = 0
        self._lock = threading.Lock()

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
        # 1: permissivo (rms > 140), 2: standard (rms > 190), 3: reattivo/anti-rumore (rms > 260)
        self.rms_threshold = 140.0 if vad_aggressiveness == 1 else (260.0 if vad_aggressiveness == 3 else 190.0)

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

        # Ricampionamento (solo se il dispositivo non supporta 16 kHz)
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
        """Apre lo stream di input: prima a 16 kHz nativi, poi alla frequenza del dispositivo con ricampionamento."""
        device = _resolve_input_device()
        try:
            stream = sd.InputStream(
                device=device,
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="int16",
                blocksize=self.SAMPLES_PER_FRAME,
                callback=self._audio_callback,
            )
            self._resampler = None
            return stream
        except Exception as e16:
            info = sd.query_devices(device, kind="input") if device is not None else sd.query_devices(kind="input")
            native = float(info.get("default_samplerate") or 48000.0)
            logger.warning(f"[AudioEngine] 16 kHz non supportato ({e16}). Cattura a {native:.0f} Hz con ricampionamento.")
            self._resampler = _StreamResampler(native, self.SAMPLE_RATE)
            self._pending = np.zeros(0, dtype=np.float32)
            return sd.InputStream(
                device=device,
                samplerate=native,
                channels=self.CHANNELS,
                dtype="int16",
                blocksize=int(round(native * self.FRAME_MS / 1000.0)),
                callback=self._native_rate_callback,
            )

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
                logger.info(f"[AudioEngine] Microfono hardware avviato (16kHz mono) per lezione {self.lecture_id}")
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
        with self._lock:
            if not self._recording.closed:
                self._recording.write(raw_bytes)
                self._recorded_bytes += len(raw_bytes)

    def _native_rate_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback a frequenza nativa: ricampiona a 16 kHz e inoltra frame da 480 campioni."""
        if not self.is_running or self.is_paused or self._resampler is None:
            return
        out = self._resampler.process(indata.reshape(-1))
        buf = np.concatenate([self._pending, out]) if len(self._pending) else out
        n_frames = len(buf) // self.SAMPLES_PER_FRAME
        for i in range(n_frames):
            seg = buf[i * self.SAMPLES_PER_FRAME:(i + 1) * self.SAMPLES_PER_FRAME]
            frame = np.clip(np.round(seg), -32768, 32767).astype(np.int16)
            self._audio_callback(frame.reshape(-1, 1), self.SAMPLES_PER_FRAME, time_info, status)
        self._pending = buf[n_frames * self.SAMPLES_PER_FRAME:].copy()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback chiamata ogni 30ms con 480 campioni int16 a 16 kHz."""
        if not self.is_running or self.is_paused:
            return

        # 1. Filtro High-Pass anti-rumble (80 Hz): rimuove ronzii, ventole e colpi sulla scrivania
        samples_raw = indata.reshape(-1)
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
                raw_bytes = indata.tobytes()
                samples_f = samples_raw.astype(np.float32, copy=False)
                energy = float(np.dot(samples_f, samples_f))
                rms = float(np.sqrt(energy / len(samples_f)))
        else:
            raw_bytes = indata.tobytes()
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

        # Estrai l'eventuale parlato residuo rimasto nel buffer
        last_audio = None
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

        self.is_paused = False
        self.is_voice_active = False
        self._reset_filter_state()
        self._preroll_buffer.clear()
        self._speech_buffer.clear()
        self._speech_frames_count = 0
        self._silence_frames_count = 0
        self._pending = np.zeros(0, dtype=np.float32)

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
                if not self._recording.closed:
                    self._recording.flush()
                    self._recording.seek(0)
                with wave.open(part_path, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)  # 16-bit = 2 bytes
                    wf.setframerate(self.SAMPLE_RATE)
                    if not self._recording.closed:
                        while True:
                            block = self._recording.read(1024 * 1024)
                            if not block:
                                break
                            wf.writeframesraw(block)
                os.replace(part_path, output_path)
            finally:
                try:
                    self._recording.close()  # elimina anche l'eventuale file temporaneo
                except Exception:
                    pass
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass

        logger.info(f"[AudioEngine] Registrazione WAV esportata: {output_path}")

        # Determina formato desiderato (da parametro o da settings.json)
        if not audio_format:
            try:
                from config import load_settings
                audio_format = load_settings().get("audio_format", "m4a")
            except Exception:
                audio_format = "m4a"

        final_path = output_path
        if audio_format == "m4a":
            try:
                final_path = compress_wav_to_m4a(output_path, bitrate=48000, remove_original=True)
            except Exception as e:
                logger.error(f"[AudioEngine] Fallback su WAV non compresso ({e})")
                final_path = output_path

        logger.info(f"[AudioEngine] Registrazione finalizzata con successo: {final_path} (formato: {audio_format})")
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
                out_stream.bit_rate = int(bitrate)
                out_stream.layout = "mono"

                resampler = av.AudioResampler(
                    format=out_stream.format,
                    layout=out_stream.layout,
                    rate=out_stream.rate,
                )

                for frame in in_container.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        for packet in out_stream.encode(resampled):
                            out_container.mux(packet)

                for resampled in resampler.resample(None):
                    for packet in out_stream.encode(resampled):
                        out_container.mux(packet)

                for packet in out_stream.encode(None):
                    out_container.mux(packet)

        if not os.path.exists(part_path) or os.path.getsize(part_path) == 0:
            logger.warning(f"[AudioEngine] File compresso vuoto o non generato per {wav_path}")
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            return wav_path

        # Sostituzione atomica del file compresso
        os.replace(part_path, m4a_path)

        orig_size = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
        new_size = os.path.getsize(m4a_path) if os.path.exists(m4a_path) else 0
        saved_bytes = orig_size - new_size
        ratio = (1.0 - (new_size / orig_size)) * 100 if orig_size > 0 else 0
        logger.info(
            f"[AudioEngine] Compressione salva-spazio completata: {orig_size / (1024*1024):.1f} MB -> "
            f"{new_size / (1024*1024):.1f} MB (-{ratio:.1f}%, liberati {saved_bytes / (1024*1024):.1f} MB)"
        )

        # Se richiesto, rimuovi il WAV originale
        if remove_original and os.path.exists(wav_path) and os.path.abspath(wav_path) != os.path.abspath(m4a_path):
            try:
                os.remove(wav_path)
            except OSError as del_err:
                logger.warning(f"[AudioEngine] Impossibile eliminare WAV temporaneo {wav_path}: {del_err}")

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
