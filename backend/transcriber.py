"""
transcriber.py — SbobinAI Transcriber Module (platform-aware)

Backend selezionato da platform_utils.AI_BACKEND:
  - "mlx"      → mlx-whisper su Apple Silicon (Metal), repo mlx-community/whisper-large-v3-turbo-q4
  - "portable" → faster-whisper (CTranslate2): CUDA float16 se disponibile, altrimenti CPU int8,
                 repo dropbox-dash/faster-whisper-large-v3-turbo

API pubblica invariata (Transcriber, release_*, warm_up_*, get_local_snapshot_dir,
TARGET_HF_REPO) + transcribe_file() per file audio completi.
Nessun import MLX / CTranslate2 a livello di modulo.
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import gc
import logging
import re
import threading
from typing import Any, Optional, Tuple

import numpy as np

import ai_models
from platform_utils import USE_MLX, cpu_threads, cuda_device_count
from platform_utils import get_local_snapshot_dir as _get_local_snapshot_dir

logger = logging.getLogger("sbobinai.transcriber")

_WHISPER_SPEC = ai_models.CATALOG[ai_models.WHISPER_CORE_ID]
TARGET_HF_REPO = _WHISPER_SPEC.repo_id  # repo della piattaforma corrente

# Mappa modelli supportati (con alias retrocompatibili)
WHISPER_REPO_MAP = {
    "large-v3-turbo-q4": TARGET_HF_REPO,
    "large-v3-turbo": TARGET_HF_REPO,
    "dual-tier-whisper": TARGET_HF_REPO,  # Retrocompatibilità impostazioni precedenti
    "mock_mode": "mock",
}

MODEL_NOT_DOWNLOADED_MSG = (
    "Il modello di trascrizione Whisper non è ancora stato scaricato in locale. "
    "Vai in Impostazioni -> Modelli per scaricarlo con un clic."
)

# Frasi simulate accademiche per modalità mock
SIMULATED_LECTURE_FRAGMENTS = [
    "Oggi analizzeremo la complessità asintotica e la convergenza degli algoritmi distribuiti.",
    "Come potete osservare su questo passaggio matematico, la derivata parziale rispetto a theta deve annullarsi.",
    "Questo principio di località della memoria spiega l'efficacia del caching gerarchico nelle moderne CPU.",
    "Ricordatevi che questa definizione teorica compare frequentemente nelle prove d'esame degli anni precedenti.",
    "Procediamo con la dimostrazione: poniamo la funzione d'onda all'interno dell'equazione differenziale.",
    "Un aspetto fondamentale della consistenza sequenziale è che tutti i processi concordano sul medesimo ordine di eventi.",
    "Prendete nota su questo schema: la separazione delle responsabilità permette di modularizzare l'architettura.",
    "Riassumendo quanto detto finora: abbiamo verificato la correttezza del protocollo in caso di partizione di rete.",
]


def get_local_snapshot_dir(repo_id: str) -> Optional[str]:
    """
    Risolve il percorso assoluto della snapshot salvata in locale per evitare qualsiasi
    chiamata di rete verso HuggingFace durante l'inferenza di Whisper.
    (La snapshot di fallback è la più recente per mtime, non l'ultima in ordine alfabetico.)
    """
    return _get_local_snapshot_dir(repo_id)


def _resolve_model_path(repo_id: str) -> Optional[str]:
    """Snapshot locale completa del modello Whisper richiesto (None se non scaricato)."""
    if repo_id == TARGET_HF_REPO:
        return ai_models.local_model_path(_WHISPER_SPEC)
    return get_local_snapshot_dir(repo_id)


# Mutex globale e ThreadPool dedicato a worker singolo: una sola inferenza Whisper alla volta
_transcription_lock = threading.RLock()
_mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-infer")
_model_warmed: bool = False


# ── Backend portable: faster-whisper con singleton thread-safe ─────────────────

class _FasterWhisperHolder:
    model: Any = None
    model_path: Optional[str] = None
    device: Optional[str] = None


def _load_faster_whisper(model_path: str, force_cpu: bool = False) -> Any:
    """Carica (una sola volta) il WhisperModel. Va chiamato con _transcription_lock acquisito."""
    holder = _FasterWhisperHolder
    want_cuda = (not force_cpu) and cuda_device_count() > 0
    if holder.model is not None and holder.model_path == model_path and (holder.device == "cuda") == want_cuda:
        return holder.model

    holder.model = None
    gc.collect()
    from faster_whisper import WhisperModel

    if want_cuda:
        for compute_type in ("float16", "int8_float16"):
            try:
                logger.info(f"[Transcriber] Caricamento faster-whisper su CUDA ({compute_type}): {model_path}")
                holder.model = WhisperModel(model_path, device="cuda", compute_type=compute_type)
                holder.model_path, holder.device = model_path, "cuda"
                return holder.model
            except Exception as exc:
                logger.warning(f"[Transcriber] CUDA {compute_type} non disponibile ({exc}).")
        logger.warning("[Transcriber] Fallback su CPU int8 (verifica cuBLAS/cuDNN 12 nel PATH per la GPU).")

    logger.info(f"[Transcriber] Caricamento faster-whisper su CPU (int8, {cpu_threads()} thread): {model_path}")
    holder.model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=cpu_threads())
    holder.model_path, holder.device = model_path, "cpu"
    return holder.model


def _decode_audio_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Decodifica qualsiasi file audio (wav/mp3/m4a/ogg/...) in float32 mono 16 kHz con PyAV.
    Implementazione propria: faster_whisper.decode_audio non è compatibile con PyAV >= 16.
    """
    import av
    from av.audio.resampler import AudioResampler

    chunks = []
    with av.open(path, mode="r") as container:
        stream = container.streams.audio[0]
        resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)
    audio *= (1.0 / 32768.0)
    return audio


def _is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cublas", "cudnn", "cufft"))


def _faster_whisper_run(model_path: str, audio: Any, *, live: bool, initial_prompt: Optional[str],
                        language: str = "it") -> Tuple[str, float]:
    """Esegue faster-whisper (blocking). Ritenta su CPU se il runtime CUDA fallisce in inferenza."""
    kwargs = dict(
        language=language,
        initial_prompt=initial_prompt,
        beam_size=1,              # Greedy come su MLX: latenza minima
        temperature=0.0 if live else (0.0, 0.2, 0.4),
        compression_ratio_threshold=1.8 if live else 2.4,
        no_speech_threshold=0.5,
        log_prob_threshold=-0.8 if live else -1.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        without_timestamps=live,
        vad_filter=not live,      # Live: VAD già fatto da AudioEngine; file: Silero VAD salta i silenzi
    )
    for attempt in range(2):
        model = _load_faster_whisper(model_path, force_cpu=attempt > 0)
        try:
            segments, info = model.transcribe(audio, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments if seg.text)
            return text.strip(), float(getattr(info, "duration", 0.0) or 0.0)
        except Exception as exc:
            if attempt == 0 and _FasterWhisperHolder.device == "cuda" and _is_cuda_runtime_error(exc):
                logger.warning(f"[Transcriber] Errore runtime CUDA ({exc}): ricarico su CPU.")
                _FasterWhisperHolder.model = None
                _FasterWhisperHolder.device = None
                continue
            raise
    return "", 0.0


def warm_up_whisper() -> None:
    """Pre-riscalda il modello eseguendo un passaggio di prova una sola volta in modo sicuro."""
    global _model_warmed
    local_dir = ai_models.local_model_path(_WHISPER_SPEC)
    if not local_dir:
        logger.info("[Transcriber] Warm-up saltato: modello Whisper non ancora scaricato.")
        return
    with _transcription_lock:
        if _model_warmed:
            return
        try:
            logger.info(f"[Transcriber] Pre-riscaldamento Whisper per {local_dir}...")
            dummy = np.zeros(8000, dtype=np.float32)
            if USE_MLX:
                import mlx_whisper
                mlx_whisper.transcribe(dummy, path_or_hf_repo=local_dir, language="it", temperature=0.0)
            else:
                _faster_whisper_run(local_dir, dummy, live=True, initial_prompt=None)
            _model_warmed = True
            logger.info("[Transcriber] Pre-riscaldamento completato: Whisper Large Turbo pronto all'uso!")
        except Exception as exc:
            logger.warning(f"[Transcriber] Pre-riscaldamento non completato: {exc}")


class Transcriber:
    """
    Motore di trascrizione audio ad alta precisione (Whisper Large-v3-Turbo):
    mlx-whisper su Apple Silicon, faster-whisper su Windows/Linux.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        whisper_keywords: Optional[str] = None,
    ):
        self.whisper_keywords = whisper_keywords.strip() if whisper_keywords else None
        self._mock_index = 0

        # Risolvi se mock o reale
        if mock_mode is True or model_name == "mock_mode":
            self.mock_mode = True
            self.repo_id = "mock"
            self.local_model_path = None
            logger.info("[Transcriber] Inizializzato in MOCK_MODE.")
        else:
            self.mock_mode = False
            if model_name in WHISPER_REPO_MAP:
                self.repo_id = WHISPER_REPO_MAP[model_name]
            elif model_name and ai_models.resolve(model_name, kind="whisper") is not None:
                self.repo_id = TARGET_HF_REPO
            elif model_name:
                self.repo_id = model_name
            else:
                self.repo_id = TARGET_HF_REPO

            # Risolvi eventuale snapshot locale
            self.local_model_path = _resolve_model_path(self.repo_id)
            if self.local_model_path:
                logger.info(f"[Transcriber] Modello locale pronto ({self.repo_id}): {self.local_model_path}")
            else:
                logger.info(f"[Transcriber] Modello non ancora scaricato in cache locale: {self.repo_id}")

    @staticmethod
    def _clean_transcript(text: str) -> str:
        """Pulisce allucinazioni tipiche di Whisper, loop di ripetizione e artefatti web."""
        text = text.strip()
        if not text or len(text) <= 1:
            return ""

        lower = text.lower().strip(" .!?:;,-\"'")
        artifacts = [
            "grazie a tutti",
            "grazie per la visione",
            "grazie dell'attenzione",
            "iscriviti al canale",
            "lascia un like",
            "alla prossima",
            "see you next time",
            "next video",
            "in the next video",
            "thank you for watching",
            "sottotitoli a cura di",
            "sottotitoli creati dalla comunità",
            "sottotitoli di",
            "sigla",
            "www.",
            ".com",
            ".org",
            "visione di",
        ]
        for h in artifacts:
            if h in lower and (len(text.split()) <= 6 or lower.count(h) >= 2):
                return ""
            if lower == h:
                return ""

        # Rileva allucinazioni da spelling di lettere singole
        if re.match(r"^([A-Za-z][\.\-\s,]+){2,}", text):
            letters_only = re.sub(r"[^A-Za-z]", "", text)
            if len(letters_only) < 8 and (len(text) / max(1, len(letters_only))) >= 1.5:
                logger.info(f"[Transcriber Filter] Scartato spelling allucinato: '{text}'")
                return ""

        # Elimina frammenti con soli caratteri non alfabetici o 1 sola lettera
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count <= 1:
            return ""

        words = [w.strip(" .,!?:;\"'()[]{}") for w in text.split()]
        words = [w for w in words if w]
        if not words:
            return ""

        # 1. Rileva loop di ripetizione di parole (es. "aria, aria, aria...")
        if len(words) >= 4:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.45:
                logger.info(f"[Transcriber Filter] Scartata ripetizione ({unique_ratio:.2f}): '{text[:45]}...'")
                return ""

        # 2. Rileva loop di n-gram ripetuti
        for n in (2, 3, 4):
            if len(words) >= n * 3:
                ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
                counts = collections.Counter(ngrams)
                if counts and counts.most_common(1)[0][1] >= 3:
                    logger.info(f"[Transcriber Filter] Scartato n-gram ripetuto: '{text[:45]}...'")
                    return ""

        return text

    def _sync_transcribe(
        self,
        audio_np: np.ndarray,
        model_target: str,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Esecuzione sincrona con Greedy Decoding puro (anti-fallback e zero calore superfluo)."""
        with _transcription_lock:
            try:
                if USE_MLX:
                    import mlx_whisper
                    result = mlx_whisper.transcribe(
                        audio_np,
                        path_or_hf_repo=model_target,
                        language="it",
                        initial_prompt=initial_prompt,
                        temperature=(0.0,),  # Greedy rigido a passaggio singolo: evita fallback ricorsivi
                        compression_ratio_threshold=1.8,
                        no_speech_threshold=0.5,
                        logprob_threshold=-0.8,
                        condition_on_previous_text=False,
                        fp16=True,
                        word_timestamps=False,
                    )
                    raw_text = result.get("text", "")
                else:
                    raw_text, _ = _faster_whisper_run(model_target, audio_np, live=True, initial_prompt=initial_prompt)
                return self._clean_transcript(raw_text)
            except Exception as exc:
                logger.error(f"[Transcriber] Errore inferenza ({model_target}): {exc}")
                return ""

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Trascrive un blocco vocale completo con Whisper Large-v3-Turbo per la massima accuratezza accademica.
        """
        if self.mock_mode:
            await asyncio.sleep(0.6)
            phrase = SIMULATED_LECTURE_FRAGMENTS[self._mock_index % len(SIMULATED_LECTURE_FRAGMENTS)]
            self._mock_index += 1
            return phrase

        if not audio_data or len(audio_data) < 1600:
            return ""

        # Il modello può essere stato scaricato dopo la creazione della sessione
        if not self.local_model_path:
            self.local_model_path = _resolve_model_path(self.repo_id)
        target = self.local_model_path
        if not target:
            if USE_MLX:
                target = self.repo_id  # comportamento storico: mlx-whisper risolve il repo
            else:
                logger.error(f"[Transcriber] {MODEL_NOT_DOWNLOADED_MSG}")
                return ""

        try:
            # int16 → float32 normalizzato: una sola allocazione, scala in-place
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np *= (1.0 / 32768.0)

            # Pre-filtro spettrale rapido: scarta rumori prima di impegnare la GPU
            if not is_speech_spectral(audio_np):
                return ""

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                _mlx_executor,
                self._sync_transcribe,
                audio_np,
                target,
                self.whisper_keywords,
            )
            if text:
                logger.info(f"[Transcriber Large] '{text}'")
            return text
        except Exception as exc:
            logger.error(f"[Transcriber] Errore trascrizione: {exc}")
            return ""

    def release_metal_cache(self) -> None:
        """Deep Sleep: libera la cache di calcolo Metal senza scaricare i pesi del modello in RAM."""
        release_metal_cache()

    def release_memory(self) -> None:
        """Libera la cache Metal e le risorse allocate in RAM."""
        release_transcriber_memory()


def transcribe_file(audio_path: str, language: str = "it", initial_prompt: Optional[str] = None) -> Tuple[str, float]:
    """
    Trascrive un file audio completo (wav/mp3/m4a/...), BLOCCANTE e thread-safe.
    Condivide il lock del trascrittore live (mai due inferenze Whisper in parallelo sulla GPU).
    Restituisce (testo_completo, durata_secondi).
    Solleva RuntimeError se il modello non è scaricato o la trascrizione fallisce.
    """
    model_path = ai_models.local_model_path(_WHISPER_SPEC)
    if not model_path:
        raise RuntimeError(MODEL_NOT_DOWNLOADED_MSG)

    with _transcription_lock:
        try:
            # Decodifica sempre con PyAV in np.ndarray float32 mono 16kHz (nessun ffmpeg CLI esterno richiesto)
            audio = _decode_audio_file(audio_path)
            if audio.size < 1600:
                return "", audio.size / 16000.0

            if USE_MLX:
                import mlx_whisper
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=model_path,
                    language=language,
                    initial_prompt=initial_prompt,
                    temperature=(0.0,),
                    condition_on_previous_text=False,
                    fp16=True,
                )
                text = (result.get("text") or "").strip()
                segments = result.get("segments") or []
                duration = float(segments[-1].get("end", 0.0)) if segments else (audio.size / 16000.0)
            else:
                text, duration = _faster_whisper_run(model_path, audio, live=False,
                                                     initial_prompt=initial_prompt, language=language)
            del audio
        except Exception as exc:
            raise RuntimeError(f"Trascrizione del file audio non riuscita: {exc}") from exc
        finally:
            release_metal_cache()
    return text, duration


def is_speech_spectral(audio_np: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Pre-filtro spettrale ultra-rapido (BLAS dot product, nessuna allocazione intermedia).
    Verifica se il segnale contiene caratteristiche di fonemi/voce umana:
    - Livello RMS minimo (esclude silenzi e fruscii impercettibili)
    - Zero-Crossing Rate (ZCR) coerente con le formanti vocali
    - Concentrazione di energia spettrale nella banda vocale umana (200Hz - 3800Hz)
    """
    total_samples = len(audio_np)
    if total_samples < 800:
        return False

    energy = float(np.dot(audio_np, audio_np))
    rms = float(np.sqrt(energy / total_samples))
    if rms < 0.003:
        return False

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2.0 * total_samples)
    if zero_crossings < 0.005 or zero_crossings > 0.60:
        return False

    # Analisi spettrale rapida FFT sui primi 4096 campioni (~256ms)
    fft_samples = audio_np[:4096]
    fft_vals = np.abs(np.fft.rfft(fft_samples))
    freqs = np.fft.rfftfreq(len(fft_samples), d=1.0 / sample_rate)
    speech_band = (freqs >= 200) & (freqs <= 3800)
    total_energy = float(np.dot(fft_vals, fft_vals))
    if total_energy <= 1e-9:
        return False
    speech_vals = fft_vals[speech_band]
    speech_energy = float(np.dot(speech_vals, speech_vals))
    return (speech_energy / total_energy) >= 0.20


def release_metal_cache() -> None:
    """
    Deep Sleep dei buffer di calcolo:
    su MLX svuota la cache di calcolo temporanea di Metal; ovunque forza la garbage collection.
    I pesi del modello restano in RAM.
    """
    if USE_MLX:
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    gc.collect()


clear_metal_cache = release_metal_cache


def release_transcriber_memory() -> None:
    """Scarica il modello Whisper dalla RAM/VRAM, svuota le cache ed esegue gc.collect()."""
    global _model_warmed
    with _transcription_lock:
        if USE_MLX:
            try:
                import mlx_whisper.transcribe as mwt
                mwt.ModelHolder.model = None
                mwt.ModelHolder.model_path = None
            except Exception:
                pass
            try:
                import mlx.core as mx
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()
            except Exception as e:
                logger.debug(f"[Transcriber] Errore pulizia memoria: {e}")
        else:
            if _FasterWhisperHolder.model is not None:
                _FasterWhisperHolder.model = None
                _FasterWhisperHolder.model_path = None
                _FasterWhisperHolder.device = None
        _model_warmed = False
        gc.collect()
        logger.info("[Transcriber] Modello Whisper e cache rilasciati con successo.")
