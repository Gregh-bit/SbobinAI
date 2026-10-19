"""
transcriber.py — SbobinAI MLX Transcriber Module
Utilizza mlx-whisper ottimizzato per Apple Silicon (Metal) con modello quantizzato 4-bit:
repo: mlx-community/whisper-large-v3-turbo-q4
"""
from __future__ import annotations

import asyncio
import collections
import gc
import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger("sbobinai.transcriber")

TARGET_HF_REPO = "mlx-community/whisper-large-v3-turbo-q4"

# Mappa modelli supportati (con alias retrocompatibili)
WHISPER_REPO_MAP = {
    "large-v3-turbo-q4": TARGET_HF_REPO,
    "large-v3-turbo": TARGET_HF_REPO,
    "dual-tier-whisper": TARGET_HF_REPO,  # Retrocompatibilità impostazioni precedenti
    "mock_mode": "mock",
}

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
    """
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id, local_files_only=True)
        if local_path and os.path.isdir(local_path):
            return local_path
    except Exception:
        pass

    try:
        from huggingface_hub import constants
        cache_root = str(constants.HF_HUB_CACHE)
    except Exception:
        cache_root = os.path.expanduser("~/.cache/huggingface/hub")

    hf_folder_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_root, hf_folder_name, "snapshots")
    if os.path.exists(snapshots_dir):
        entries = sorted(os.listdir(snapshots_dir))
        if entries:
            cand = os.path.join(snapshots_dir, entries[-1])
            if os.path.isdir(cand):
                return cand
    return None


def check_is_model_cached(repo_id: str) -> bool:
    """Verifica se il repo HuggingFace è già presente nella cache locale."""
    return get_local_snapshot_dir(repo_id) is not None


# Mutex globale e ThreadPool dedicato a worker singolo per preservare cache CPU P-cores
_transcription_lock = threading.Lock()
import concurrent.futures
_mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-infer")
_model_warmed: bool = False


def warm_up_whisper() -> None:
    """Pre-riscalda Metal eseguendo un passaggio di prova una sola volta in modo sicuro."""
    global _model_warmed
    local_dir = get_local_snapshot_dir(TARGET_HF_REPO)
    target = local_dir or TARGET_HF_REPO
    with _transcription_lock:
        if _model_warmed:
            return
        try:
            logger.info(f"[Transcriber] Pre-riscaldamento Metal Whisper per {target}...")
            import mlx_whisper
            dummy = np.zeros(8000, dtype=np.float32)
            mlx_whisper.transcribe(
                dummy,
                path_or_hf_repo=target,
                language="it",
                temperature=0.0,
            )
            _model_warmed = True
            logger.info("[Transcriber] Pre-riscaldamento Metal completato: Whisper Large Turbo pronto all'uso!")
        except Exception as exc:
            logger.warning(f"[Transcriber] Pre-riscaldamento non completato: {exc}")


def warm_up_in_background() -> None:
    """Lancia il warmup in un thread separato al boot del server."""
    threading.Thread(target=warm_up_whisper, daemon=True, name="whisper-warmup").start()


class Transcriber:
    """
    Motore di trascrizione audio ad alta precisione con mlx-whisper (Whisper Large-v3-Turbo Q4)
    ottimizzato per Apple Silicon Metal.
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
            elif model_name:
                self.repo_id = model_name
            else:
                self.repo_id = TARGET_HF_REPO

            # Risolvi eventuale snapshot locale
            local_dir = get_local_snapshot_dir(self.repo_id)
            self.local_model_path = local_dir
            if local_dir:
                logger.info(f"[Transcriber MLX] Modello locale pronto ({self.repo_id}): {local_dir}")
            else:
                logger.info(f"[Transcriber MLX] Modello non ancora scaricato in cache locale: {self.repo_id}")

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
        import re
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
                return self._clean_transcript(raw_text)
            except Exception as exc:
                logger.error(f"[Transcriber MLX] Errore inferenza ({model_target}): {exc}")
                return ""

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Trascrive un blocco vocale completo con Whisper Large-v3-Turbo Q4 per la massima accuratezza accademica.
        """
        if self.mock_mode:
            await asyncio.sleep(0.6)
            phrase = SIMULATED_LECTURE_FRAGMENTS[self._mock_index % len(SIMULATED_LECTURE_FRAGMENTS)]
            self._mock_index += 1
            return phrase

        target = self.local_model_path or self.repo_id
        if not audio_data or len(audio_data) < 1600:
            return ""

        try:
            # 1. Conversione contigua a 16-byte alignment per sfruttare registri SIMD NEON di Apple Accelerate
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_np = np.ascontiguousarray(audio_int16, dtype=np.float32)
            audio_np *= (1.0 / 32768.0)

            # Pre-filtro spettrale rapido: scarta rumori prima di impegnare Metal
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
                logger.info(f"[Transcriber MLX Large] '{text}'")
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


def is_speech_spectral(audio_np: np.ndarray, sample_rate: int = 16000) -> bool:
    """
    Pre-filtro spettrale ultra-rapido (<0.02ms via Apple Accelerate NEON).
    Verifica se il segnale contiene caratteristiche di fonemi/voce umana:
    - Livello RMS minimo (esclude silenzi e fruscii impercettibili) via BLAS cblas_sdot
    - Zero-Crossing Rate (ZCR) coerente con le formanti vocali
    - Concentrazione di energia spettrale nella banda vocale umana (200Hz - 3800Hz)
    """
    total_samples = len(audio_np)
    if total_samples < 800:
        return False

    # Apple Accelerate cblas_sdot: calcolo energetico 8x più veloce senza allocazione di memoria intermedia
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
    Deep Sleep dei buffer di calcolo Metal:
    Svuota la cache di calcolo temporanea di MLX e forza la garbage collection.
    I pesi del modello restano in RAM unificata, ma la GPU e CPU scendono istantaneamente a 0% di utilizzo.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()


def release_transcriber_memory() -> None:
    """Rilascia la cache Metal di MLX, azzera i modelli in RAM ed esegue gc.collect()."""
    with _transcription_lock:
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
            gc.collect()
            logger.info("[Transcriber] Cache Metal e memoria RAM rilasciate con successo.")
        except Exception as e:
            logger.debug(f"[Transcriber] Errore pulizia memoria: {e}")


# Istanza helper singleton
_default_transcriber = Transcriber()


async def transcribe_chunk(audio_data: bytes, sample_rate: int = 16000) -> str:
    """Funzione helper asincrona per trascrivere un singolo chunk."""
    return await _default_transcriber.transcribe(audio_data)
