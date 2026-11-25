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
import contextlib
import gc
import itertools
import logging
import re
import threading
from typing import Any, Iterator, Optional, Tuple

import numpy as np

import ai_models
from platform_utils import USE_MLX, cpu_threads, cuda_device_count, disable_cuda, is_cuda_disabled
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
    want_cuda = (not force_cpu) and (not is_cuda_disabled()) and cuda_device_count() > 0
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
        disable_cuda("caricamento modello CUDA non disponibile")
        logger.warning("[Transcriber] Fallback permanente su CPU int8 (verifica cuBLAS/cuDNN 12 nel PATH per la GPU).")

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


FILE_CHUNK_SECONDS = 30.0
FILE_CHUNK_OVERLAP_SECONDS = 2.0


def _int16_to_float32(samples: np.ndarray) -> np.ndarray:
    """Converte campioni PCM int16 mono in float32 normalizzato in [-1.0, 1.0]."""
    audio = samples.astype(np.float32)
    audio *= (1.0 / 32768.0)
    return audio


def _iter_resampled_s16_frames(path: str, sample_rate: int) -> Iterator[np.ndarray]:
    """Apre un file audio con PyAV e produce frame 1D int16 mono ricampionati in streaming."""
    import av
    from av.audio.resampler import AudioResampler

    with av.open(path, mode="r") as container:
        if not container.streams.audio:
            return
        stream = container.streams.audio[0]
        resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
        for frame in itertools.chain(container.decode(stream), (None,)):
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                if arr.size:
                    yield arr


def _iter_decoded_audio_windows(
    path: str,
    sample_rate: int = 16000,
    window_seconds: Optional[float] = None,
    overlap_seconds: Optional[float] = None,
) -> Iterator[Tuple[np.ndarray, float, bool]]:
    """
    Decodifica in streaming con PyAV a finestre scorrevoli float32 mono senza mai
    accumulare l'intero file audio in memoria.

    Produce tuple `(window_audio, new_duration_seconds, is_first_chunk)` trattenendo
    solo la coda di `overlap_seconds` tra finestre adiacenti e liberando subito i buffer int16.
    """
    eff_window_s = FILE_CHUNK_SECONDS if window_seconds is None else window_seconds
    eff_overlap_s = FILE_CHUNK_OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds
    window_samples = max(1, int(eff_window_s * sample_rate))
    overlap_samples = max(0, min(int(eff_overlap_s * sample_rate), window_samples - 1))

    pending_parts: list[np.ndarray] = []
    pending_samples = 0
    carried_overlap = 0
    is_first = True

    try:
        with contextlib.closing(_iter_resampled_s16_frames(path, sample_rate)) as frame_iter:
            for arr in frame_iter:
                pending_parts.append(arr)
                pending_samples += int(arr.size)
                if pending_samples < window_samples:
                    continue

                combined = pending_parts[0] if len(pending_parts) == 1 else np.concatenate(pending_parts)
                pending_parts.clear()
                pending_samples = 0

                while combined.size >= window_samples:
                    new_duration = (window_samples - carried_overlap) / float(sample_rate)
                    tail_start = window_samples - overlap_samples
                    remainder = (
                        combined[tail_start:].copy()
                        if tail_start < combined.size
                        else np.zeros(0, dtype=np.int16)
                    )
                    # Pop prima di yield per evitare che il frame sospeso trattenga l'array float32
                    window_holder = [_int16_to_float32(combined[:window_samples])]
                    del combined

                    yield window_holder.pop(), new_duration, is_first

                    is_first = False
                    carried_overlap = overlap_samples
                    combined = remainder

                if combined.size:
                    pending_parts.append(combined)
                    pending_samples = int(combined.size)
                else:
                    del combined

        if pending_samples > carried_overlap:
            tail = pending_parts[0] if len(pending_parts) == 1 else np.concatenate(pending_parts)
            pending_parts.clear()
            pending_samples = 0

            new_duration = (int(tail.size) - carried_overlap) / float(sample_rate)
            tail_holder = [_int16_to_float32(tail)]
            del tail
            yield tail_holder.pop(), new_duration, is_first
    finally:
        pending_parts.clear()


def _is_cuda_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cuda", "cublas", "cudnn", "cufft", "curand", "nvrtc", ".dll"))


def _call_faster_whisper_with_cpu_fallback(model_path: str, audio: Any, **kwargs) -> Tuple[list[Any], float]:
    """
    Esegue model.transcribe e materializza i segmenti con fallback permanente a CPU
    al primo eventuale errore runtime CUDA (previene deadlock C++ su chiamate CUDA successive).
    Va chiamato con _transcription_lock acquisito.
    """
    for attempt in range(2):
        model = _load_faster_whisper(model_path, force_cpu=attempt > 0)
        try:
            segments, info = model.transcribe(audio, **kwargs)
            seg_list = list(segments)
            return seg_list, float(getattr(info, "duration", 0.0) or 0.0)
        except Exception as exc:
            if attempt == 0 and _FasterWhisperHolder.device == "cuda" and _is_cuda_runtime_error(exc):
                disable_cuda(str(exc))
                logger.warning(f"[Transcriber] Errore runtime CUDA ({exc}): passaggio permanente a CPU int8.")
                _FasterWhisperHolder.model = None
                _FasterWhisperHolder.model_path = None
                _FasterWhisperHolder.device = None
                gc.collect()
                continue
            raise
    return [], 0.0


def _faster_whisper_run(model_path: str, audio: Any, *, live: bool, initial_prompt: Optional[str],
                        language: str = "it") -> Tuple[str, float]:
    """Esegue faster-whisper (blocking). Ritenta su CPU se il runtime CUDA fallisce in inferenza."""
    seg_list, dur = _call_faster_whisper_with_cpu_fallback(
        model_path,
        audio,
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
    text = " ".join(seg.text.strip() for seg in seg_list if getattr(seg, "text", None))
    return text.strip(), dur


def faster_whisper_word_segments(model_path: str, audio: Any, language: str = "it") -> list[Any]:
    """Estrae i segmenti con word_timestamps=True in modo sicuro (con fallback permanente a CPU)."""
    seg_list, _ = _call_faster_whisper_with_cpu_fallback(
        model_path,
        audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=(0.0,),
        compression_ratio_threshold=1.8,
        no_speech_threshold=0.5,
    )
    return seg_list


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
        """Pulisce allucinazioni tipiche di Whisper, formule di chiusura su silenzio, loop e artefatti web."""
        text = text.strip()
        if not text or len(text) <= 1:
            return ""

        lower = text.lower().strip(" .!?:;,-\"'")
        closing_phrases = [
            "grazie a tutti",
            "grazie per la visione",
            "grazie per l'attenzione",
            "grazie dell'attenzione",
            "grazie per aver guardato",
            "grazie per averci seguito",
            "grazie mille",
            "grazie di tutto",
            "grazie a tutti voi",
            "grazie e arrivederci",
            "molte grazie",
            "tante grazie",
            "sottotitoli a cura di",
            "sottotitoli creati dalla comunità",
            "sottotitoli e revisione a cura di",
            "sottotitoli di",
            "sottotitoli",
            "iscriviti al canale",
            "iscrivetevi al canale",
            "lascia un like",
            "lasciate un like",
            "alla prossima",
            "arrivederci",
            "buona giornata",
            "buona visione",
            "buon ascolto",
            "thank you for watching",
            "thanks for watching",
            "see you next time",
            "next video",
            "in the next video",
            "sigla",
            "www.",
            ".com",
            ".org",
            "visione di",
        ]

        # 1. Se il testo consiste SOLO in formule di chiusura / saluti di fine video (anche combinate o con punteggiatura)
        stripped_test = lower
        for formula in sorted(closing_phrases, key=len, reverse=True):
            stripped_test = stripped_test.replace(formula, " ")
        remaining = re.sub(r"[\s\.,!?:;\-\"']+", "", stripped_test)
        if not remaining:
            logger.info(f"[Transcriber Filter] Scartata formula di chiusura isolata/ripetuta: '{text}'")
            return ""

        # 2. Se una formula di chiusura compare ripetuta in loop (2 o più volte)
        for formula in closing_phrases:
            if len(formula) >= 5 and lower.count(formula) >= 2:
                logger.info(f"[Transcriber Filter] Scartata formula ripetuta in loop ('{formula}'): '{text}'")
                return ""

        # 3. Se contiene una formula di chiusura ed è molto breve (frammento orfano)
        for formula in closing_phrases:
            if formula in lower and len(text.split()) <= 6:
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

        # 4. Rileva loop estremi di parole ripetute allucinate (es. "aria aria aria aria...")
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.28:
                logger.info(f"[Transcriber Filter] Scartato loop estremo ({unique_ratio:.2f}): '{text[:45]}...'")
                return ""
        elif len(words) >= 4 and len(set(words)) == 1:
            logger.info(f"[Transcriber Filter] Scartata singola parola in loop: '{text[:45]}...'")
            return ""

        # 5. Rileva loop reali di n-gram ripetuti (evitando falsi positivi su articoli/preposizioni come 'la', 'il', 'di', 'più' in frasi lunghe!)
        lower_words = [w.lower() for w in words]

        # 5a. Parola singola ripetuta consecutivamente >= 4 volte (es. "aria aria aria aria")
        consec_count = 1
        for i in range(1, len(lower_words)):
            if lower_words[i] == lower_words[i - 1]:
                consec_count += 1
                if consec_count >= 4:
                    logger.info(f"[Transcriber Filter] Scartata parola ripetuta consecutivamente ({lower_words[i]} x{consec_count}): '{text[:45]}...'")
                    return ""
            else:
                consec_count = 1

        # 5b. Frasi di 2..4 parole ripetute in loop (es. "vedete come sono vedete come sono...")
        for n in (2, 3, 4):
            if len(lower_words) >= n * 2:
                ngrams = [" ".join(lower_words[i:i+n]) for i in range(len(lower_words) - n + 1)]
                counts = collections.Counter(ngrams)
                if counts:
                    most_common_phrase, count = counts.most_common(1)[0]
                    # Controlla se è un vero loop (ripetuto >= 3 volte con alta copertura, o >= 2 volte coprendo >= 50% del blocco)
                    coverage = (count * n) / max(1, len(lower_words))
                    if (count >= 3 and coverage >= 0.30) or (count >= 2 and coverage >= 0.50):
                        logger.info(f"[Transcriber Filter] Scartato n-gram ripetuto ({most_common_phrase} x{count}): '{text[:45]}...'")
                        return ""

        return text

    @staticmethod
    def _extract_reliable_mlx_text(result: dict) -> str:
        """
        Estrae il testo dai segmenti di mlx-whisper, preservando integralmente il parlato reale
        e scartando solo allucinazioni palesi da silenzio prolungato (no_speech >= 0.75) o loop di compressione.
        """
        segments = result.get("segments")
        if not isinstance(segments, list) or not segments:
            return str(result.get("text") or "").strip()

        kept_parts: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            seg_text = str(seg.get("text") or "").strip()
            if not seg_text:
                continue
            no_speech = float(seg.get("no_speech_prob", 0.0) or 0.0)
            avg_logprob = float(seg.get("avg_logprob", 0.0) or 0.0)
            comp_ratio = float(seg.get("compression_ratio", 0.0) or 0.0)
            seg_words = [w for w in seg_text.split() if w.strip(".,!?:;\"'()[]{}")]

            # Scarta segmenti composti da 1-2 parole isolate su silenzio o confidenza pessima
            if no_speech >= 0.70 and len(seg_words) <= 2:
                logger.info(
                    f"[Transcriber Filter] Scartato segmento silenzio (no_speech={no_speech:.2f}): '{seg_text[:45]}'"
                )
                continue
            if no_speech >= 0.40 and avg_logprob < -0.70 and len(seg_words) <= 2:
                logger.info(
                    f"[Transcriber Filter] Scartato segmento incerto (no_speech={no_speech:.2f}, logprob={avg_logprob:.2f}): '{seg_text[:45]}'"
                )
                continue
            if avg_logprob < -1.15 and len(seg_words) <= 2:
                logger.info(
                    f"[Transcriber Filter] Scartato frammento bassa confidenza (logprob={avg_logprob:.2f}): '{seg_text[:45]}'"
                )
                continue
            if comp_ratio > 2.2:
                logger.info(
                    f"[Transcriber Filter] Scartato segmento compresso/loop (cr={comp_ratio:.2f}): '{seg_text[:45]}'"
                )
                continue
            kept_parts.append(seg_text)

        return " ".join(kept_parts).strip()

    def _sync_transcribe(
        self,
        audio_np: np.ndarray,
        model_target: str,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Esecuzione sincrona con parametri Whisper calibrati per non perdere parole nel parlato reale."""
        with _transcription_lock:
            try:
                if USE_MLX:
                    import mlx_whisper
                    result = mlx_whisper.transcribe(
                        audio_np,
                        path_or_hf_repo=model_target,
                        language="it",
                        initial_prompt=initial_prompt,
                        temperature=(0.0,),
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0,
                        condition_on_previous_text=False,
                        fp16=True,
                        word_timestamps=False,
                    )
                    raw_text = self._extract_reliable_mlx_text(result)
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
            logger.error(f"[Transcriber] {MODEL_NOT_DOWNLOADED_MSG}")
            return ""

        try:
            rem = len(audio_data) % 2
            if rem:
                audio_data = audio_data[:-rem]
            if not audio_data:
                return ""
            # int16 → float32 normalizzato: una sola allocazione, scala in-place
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_np = np.nan_to_num(audio_np)
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

    def preload_model_background(self, delay: float = 0.05) -> None:
        """
        Pre-riscalda Whisper in un thread daemon in background non appena si avvia la registrazione.
        """
        if self.mock_mode:
            return
        if not self.local_model_path:
            self.local_model_path = _resolve_model_path(self.repo_id)
        if not self.local_model_path:
            return

        def _worker():
            try:
                if delay > 0:
                    time.sleep(delay)
                warm_up_whisper()
            except Exception as e:
                logger.debug(f"[Transcriber] Pre-riscaldamento background non completato: {e}")

        t = threading.Thread(target=_worker, daemon=True, name="whisper-prewarm")
        t.start()


_BOUNDARY_PUNCT_STRIP = " .,!?:;\"'()[]{}«»…—–-"
_OVERLAP_SEARCH_WORDS = 10
_MIN_OVERLAP_WORDS = 2


def _normalize_boundary_word(word: str) -> str:
    """Normalizza una parola ai margini di una finestra rimuovendo punteggiatura e maiuscole."""
    return word.strip(_BOUNDARY_PUNCT_STRIP).lower()


def _append_with_overlap_dedup(existing_text: str, next_text: str) -> str:
    """
    Unisce `next_text` ad `existing_text` rimuovendo eventuali parole duplicate al confine
    (confronto case-insensitive senza punteggiatura entro le ultime/prime 10 parole, minimo 2).
    """
    next_clean = (next_text or "").strip()
    if not next_clean:
        return (existing_text or "").strip()
    existing_clean = (existing_text or "").strip()
    if not existing_clean:
        return next_clean

    existing_norms = [
        norm for w in existing_clean.split() if (norm := _normalize_boundary_word(w))
    ]
    if not existing_norms:
        return next_clean

    next_raw = next_clean.split()
    next_indexed = [
        (idx, norm) for idx, w in enumerate(next_raw) if (norm := _normalize_boundary_word(w))
    ]
    if not next_indexed:
        return existing_clean

    tail = existing_norms[-_OVERLAP_SEARCH_WORDS:]
    head_pairs = next_indexed[:_OVERLAP_SEARCH_WORDS]
    head_norms = [norm for _, norm in head_pairs]

    max_k = min(len(tail), len(head_norms))
    drop_raw_until = 0
    for k in range(max_k, _MIN_OVERLAP_WORDS - 1, -1):
        if tail[-k:] == head_norms[:k]:
            drop_raw_until = head_pairs[k - 1][0] + 1
            break

    remaining_words = next_raw[drop_raw_until:]
    if not remaining_words or not any(_normalize_boundary_word(w) for w in remaining_words):
        return existing_clean
    return f"{existing_clean} {' '.join(remaining_words)}"


def _merge_chunk_transcripts(chunks: list[str]) -> str:
    """Concatena le trascrizioni di più finestre eliminando le sovrapposizioni ai bordi."""
    merged = ""
    for chunk_text in chunks:
        merged = _append_with_overlap_dedup(merged, chunk_text)
    return merged


def _is_digital_silence(audio: np.ndarray) -> bool:
    """Rileva finestre di silenzio digitale puro per prevenire allucinazioni di Whisper."""
    return audio.size == 0 or float(np.max(np.abs(audio))) < 1e-4


_FILE_BOILERPLATE_RE = re.compile(
    r"(?:sottotitoli\s+(?:creati|a\s+cura)\s+(?:dalla\s+comunità(?:\s+amara\.org)?|di\s+[^\n.!?]+)|"
    r"grazie\s+per\s+(?:la\s+visione|l['’]ascolto|aver\s+guardato)|"
    r"iscriviti\s+al\s+canale|lascia\s+un\s+like|thank\s+you\s+for\s+watching|see\s+you\s+next\s+time)[.!?]*",
    re.IGNORECASE,
)
_FILE_LOOP_PHRASE_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ0-9'’]+(?:\s+[A-Za-zÀ-ÿ0-9'’]+){1,5})\b(?:\s*[,;.]?\s+\1\b){2,}",
    re.IGNORECASE,
)
_FILE_LOOP_WORD_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ]{2,24})\b(?:\s*[,;.]?\s+\1\b){4,}",
    re.IGNORECASE,
)


def _clean_file_window_text(text: str) -> str:
    """Rimuove allucinazioni boilerplate/loop da finestre file senza scartare risposte brevi valide (es. 'A')."""
    s = (text or "").strip()
    if not s:
        return ""
    words = [w for w in s.split() if w.strip(".,!?:;\"'()[]{}")]
    # Testo di una sola parola/lettera valido (es. risposta "A" / "Sì" / "42"): scarta solo URL amara.org/www.
    if len(words) <= 1:
        low = s.lower()
        if "amara.org" in low or "www." in low:
            return ""
        return s if any(c.isalnum() for c in s) else ""

    cleaned = _FILE_BOILERPLATE_RE.sub(" ", s)
    cleaned = _FILE_LOOP_PHRASE_RE.sub(r"\1", cleaned)
    cleaned = _FILE_LOOP_WORD_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,;-")
    if not any(c.isalnum() for c in cleaned):
        return ""
    return cleaned


def _transcribe_file_window(
    model_path: str,
    audio: np.ndarray,
    *,
    language: str,
    initial_prompt: Optional[str],
) -> str:
    """Trascrive una singola finestra audio su MLX o faster-whisper (con fallback CUDA -> CPU)."""
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
        raw_text = (result.get("text") or "").strip()
        return _clean_file_window_text(raw_text)
    text, _ = _faster_whisper_run(
        model_path,
        audio,
        live=False,
        initial_prompt=initial_prompt,
        language=language,
    )
    return _clean_file_window_text(text)


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
        merged_text = ""
        prev_was_gap = False
        total_duration = 0.0
        chunk_count = 0
        try:
            # Decodifica sempre con PyAV in streaming a finestre (nessun ffmpeg CLI esterno richiesto)
            for chunk, chunk_new_duration, is_first in _iter_decoded_audio_windows(audio_path):
                chunk_count += 1
                total_duration += chunk_new_duration
                try:
                    if is_first and chunk_new_duration < 0.1:
                        prev_was_gap = True
                        continue
                    if _is_digital_silence(chunk):
                        logger.debug(
                            f"[Transcriber File] Finestra {chunk_count} silenzio digitale, skip Whisper."
                        )
                        prev_was_gap = True
                        continue
                    if not is_first:
                        logger.info(
                            f"[Transcriber File] Trascrizione finestra {chunk_count} "
                            f"(durata accumulata: {total_duration:.1f}s)..."
                        )
                    chunk_text = _transcribe_file_window(
                        model_path,
                        chunk,
                        language=language,
                        initial_prompt=initial_prompt,
                    )
                    if chunk_text:
                        if prev_was_gap or not merged_text:
                            merged_text = f"{merged_text} {chunk_text}".strip() if merged_text else chunk_text
                        else:
                            merged_text = _append_with_overlap_dedup(merged_text, chunk_text)
                        prev_was_gap = False
                    else:
                        prev_was_gap = True
                finally:
                    del chunk
                    release_metal_cache()
        except Exception as exc:
            raise RuntimeError(f"Trascrizione del file audio non riuscita: {exc}") from exc

    if total_duration < 0.1:
        return "", total_duration
    if chunk_count > 1:
        logger.info(
            f"[Transcriber File] Trascrizione multi-finestra completata: "
            f"{chunk_count} finestre, {total_duration:.1f}s totali."
        )
    return merged_text, total_duration


def is_speech_spectral(audio_np: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Pre-filtro spettrale ultra-rapido (BLAS dot product, nessuna allocazione intermedia).
    Verifica se il segnale contiene caratteristiche di fonemi/voce umana:
    - Livello RMS minimo (esclude silenzi digitali e fruscii microscopici <0.0002)
    - Zero-Crossing Rate (ZCR) coerente con le formanti vocali
    - Concentrazione di energia spettrale nella banda vocale umana (180Hz - 4000Hz)
    """
    total_samples = len(audio_np)
    if total_samples < 800:
        return False

    energy = float(np.dot(audio_np, audio_np))
    rms = float(np.sqrt(energy / total_samples))
    if rms < 0.0007:
        return False

    # Zero Crossing Rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_np)))) / (2.0 * total_samples)
    if zero_crossings < 0.0005 or zero_crossings > 0.85:
        return False

    # Analisi spettrale rapida FFT sulla finestra a massima energia (evita falsi negativi su pause/respiro iniziale)
    if total_samples > 4096:
        mid_idx = total_samples // 2
        cand_starts = [0, max(0, mid_idx - 2048), max(0, total_samples - 4096)]
        best_start = 0
        best_eng = -1.0
        for s in cand_starts:
            seg = audio_np[s : s + 4096]
            eng = float(np.dot(seg, seg))
            if eng > best_eng:
                best_eng = eng
                best_start = s
        fft_samples = audio_np[best_start : best_start + 4096]
    else:
        fft_samples = audio_np

    fft_vals = np.abs(np.fft.rfft(fft_samples))
    freqs = np.fft.rfftfreq(len(fft_samples), d=1.0 / sample_rate)
    speech_band = (freqs >= 120) & (freqs <= 4200)
    total_energy = float(np.dot(fft_vals, fft_vals))
    if total_energy <= 1e-9:
        return False
    speech_vals = fft_vals[speech_band]
    speech_energy = float(np.dot(speech_vals, speech_vals))
    return (speech_energy / total_energy) >= 0.025


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
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
                elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
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
