"""
llm_engine.py — SbobinAI LLM Engine (platform-aware)

Backend selezionato da platform_utils.AI_BACKEND:
  - "mlx"      → mlx-lm su Apple Silicon (Qwen3.5-4B / Gemma-4-E4B 4-bit MLX), comportamento invariato:
                 KV cache quantizzata 4-bit, mx.clear_cache() post-generazione, unload di Whisper.
  - "portable" → llama-cpp-python con GGUF Q4_K_M (Windows/Linux): offload completo su GPU se la build
                 lo supporta (CUDA/Vulkan), altrimenti CPU multi-thread.

Concorrenza:
  - un solo modello in RAM (singleton, caricamento lazy thread-safe con RLock)
  - una sola generazione alla volta (_generation_lock) → nessuna corsa sul contesto del modello
  - cancellazione con epoch: stop_all_ai() incrementa l'epoch, ogni job confronta l'epoch catturata
    all'avvio (cancel_epoch() / is_cancelled(epoch)); nessun reset temporizzato che possa perdere lo stop.
  - zero auto-download: non scarica pesi senza esplicita richiesta dell'utente
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
import logging
import os
import re
import sys
import threading
import time
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple

import ai_models
from platform_utils import USE_MLX, cpu_threads, get_local_snapshot_dir, llama_gpu_offload_supported
from transcript_cleaner import clean_transcript_for_llm

logger = logging.getLogger("sbobinai.llm")

# ── Temperature dedicate per tipologia di inferenza ───────────────────────────
TEMP_SUMMARY = 0.2         # Sintesi globale / capitolo Quaderno (rigoroso, anti-loop)
TEMP_CHAT = 0.3            # Chat assistente ("Non ho capito!": scorrevole ma fedele)
TEMP_QUESTION = 0.5        # Domande aperte / esame orale (varietà naturale)
TEMP_STRUCTURED = 0.2      # JSON rigorosi (quiz, valutazioni orali) e risposte flash

# ── Modelli Supportati (ID logici, identici su tutte le piattaforme) ──────────
DEFAULT_LLM_MODEL = ai_models.LLM_DEFAULT_ID
LIGHT_LLM_MODEL = ai_models.LLM_LIGHT_ID
ALTERNATIVE_LLM_MODEL = ai_models.LLM_ALT_ID

SUPPORTED_LLM_MODELS = {
    mid: {
        "name": ai_models.CATALOG[mid].variant.name,
        "repo_id": ai_models.CATALOG[mid].repo_id,
        "approx_ram": ai_models.human_size(ai_models.CATALOG[mid].variant.approx_bytes),
    }
    for mid in (DEFAULT_LLM_MODEL, LIGHT_LLM_MODEL, ALTERNATIVE_LLM_MODEL)
}

LLM_CONTEXT_TOKENS = int(os.environ.get("SBOBINAI_LLM_CTX", "8192"))
USABLE_CHAT_CONTEXT_TOKENS = max(2048, LLM_CONTEXT_TOKENS - 2048)


def estimate_tokens(text: str) -> int:
    """Stima rapida del numero di token di un testo (~3 caratteri per token per italiano/LaTeX)."""
    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 2) // 3)

# ── Soglie Map-Reduce per sintesi di lezioni lunghe ───────────────────────────
SINGLE_PASS_TRANSCRIPT_CHARS = 16000
MAP_CHUNK_TARGET_CHARS = 9000
MAP_CHUNK_OVERLAP_CHARS = 350
REDUCE_MAX_CHARS = 12000


class LLMGenerationError(RuntimeError):
    """Generazione LLM fallita (errore del modello o output vuoto)."""


class LLMModelNotDownloadedError(FileNotFoundError):
    """Il modello LLM richiesto non è presente in locale."""


# ── Stato Singleton del Modello in RAM & Auto-Unload Inattività (120s) ────────
INACTIVITY_TIMEOUT_SECONDS = 120.0

_model_lock = threading.RLock()
_loaded_model: Optional[Any] = None
_loaded_tokenizer: Optional[Any] = None
_loaded_model_id: Optional[str] = None
_loaded_model_path: Optional[str] = None
_loaded_model_cfg: Optional[Tuple[int, bool, bool]] = None
_last_used_timestamp: float = 0.0
_inactivity_timer: Optional[threading.Timer] = None

# ── Stato di Generazione & Cancellazione ──────────────────────────────────────
_generation_lock = threading.Lock()   # una sola generazione alla volta
_state_lock = threading.Lock()
_active_generations = 0
_is_generating = False                # mirror booleano (retrocompatibilità lettura)
_current_task: str = ""
_cancel_epoch = 0

# DEPRECATO: mantenuto solo per compatibilità con codice esterno non ancora migrato.
# stop_all_ai() lo imposta e NON lo resetta più automaticamente; usare cancel_epoch()/is_cancelled().
_cancel_generation_event = threading.Event()


def cancel_epoch() -> int:
    """Epoch di cancellazione corrente: catturala all'avvio di un job e verificala con is_cancelled()."""
    return _cancel_epoch


def is_cancelled(epoch: int) -> bool:
    """True se stop_all_ai() è stato invocato dopo che `epoch` è stata catturata."""
    return _cancel_epoch != epoch


def _begin_generation(task: str) -> None:
    global _active_generations, _is_generating, _current_task
    with _state_lock:
        _active_generations += 1
        _is_generating = True
        if task:
            _current_task = task


def _end_generation() -> None:
    global _active_generations, _is_generating, _current_task
    with _state_lock:
        _active_generations = max(0, _active_generations - 1)
        _is_generating = _active_generations > 0
        if not _is_generating:
            _current_task = ""


@contextlib.contextmanager
def generation_session(task_label: str = "") -> Iterator[int]:
    """
    Segna il motore come occupato, serializza le generazioni e restituisce l'epoch di cancellazione.
    Uso: `with generation_session("Quiz") as epoch: ... if is_cancelled(epoch): break`
    """
    epoch = cancel_epoch()
    _begin_generation(task_label)
    try:
        with _generation_lock:
            yield epoch
    finally:
        _end_generation()


def is_ai_busy() -> Dict[str, Any]:
    """Restituisce lo stato attuale di occupazione del motore IA."""
    return {
        "is_busy": _active_generations > 0,
        "task": _current_task,
        "model_loaded": _loaded_model is not None,
        "model_id": _loaded_model_id,
    }


def stop_all_ai(force_unload: bool = True) -> Dict[str, Any]:
    """
    Segnale di emergenza immediato:
    1. Interrompe qualsiasi generazione in corso (sintesi quaderno, chat, risposte flash, quiz).
    2. Scarica Whisper e (se force_unload=True) l'LLM dalla RAM.
    3. Svuota le cache di calcolo (Metal su Mac).
    """
    global _cancel_epoch
    logger.warning("[LLM Engine] INTERRUZIONE EMERGENZA IA RICHIESTA DALL'UTENTE.")
    with _state_lock:
        _cancel_epoch += 1
    _cancel_generation_event.set()

    unload_whisper(force=True)

    try:
        from rag import unload_embedding_model
        unload_embedding_model()
    except Exception:
        pass

    if force_unload:
        unload_llm()

    clear_metal_cache()

    logger.info("[LLM Engine] Processi IA interrotti e memoria liberata con successo.")
    return {
        "status": "stopped",
        "freed_memory": True,
        "message": "Tutti i processi IA sono stati interrotti e la memoria è stata liberata.",
    }


def clear_metal_cache() -> None:
    """Rilascia la memoria di calcolo temporanea (Metal su Apple Silicon) ed esegue la GC."""
    if USE_MLX:
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception as exc:
            logger.debug(f"[LLM Engine] clear_cache: {exc}")
    gc.collect()


release_metal_cache = clear_metal_cache


def _touch_model_usage() -> None:
    """Registra l'uso del modello e (ri)avvia il timer di auto-unload per inattività."""
    global _last_used_timestamp, _inactivity_timer
    with _model_lock:
        _last_used_timestamp = time.time()
        if _inactivity_timer is not None:
            _inactivity_timer.cancel()
            _inactivity_timer = None
        if _loaded_model is not None:
            _inactivity_timer = threading.Timer(INACTIVITY_TIMEOUT_SECONDS, _check_and_auto_unload)
            _inactivity_timer.daemon = True
            _inactivity_timer.start()


def _schedule_check(delay: float) -> None:
    global _inactivity_timer
    _inactivity_timer = threading.Timer(delay, _check_and_auto_unload)
    _inactivity_timer.daemon = True
    _inactivity_timer.start()


def _check_and_auto_unload() -> None:
    """Scarica il modello dalla RAM se inattivo da oltre INACTIVITY_TIMEOUT_SECONDS e non in uso."""
    global _inactivity_timer
    if not _generation_lock.acquire(blocking=False):
        with _model_lock:
            if _loaded_model is not None:
                _schedule_check(30.0)
        return
    try:
        with _model_lock:
            if _loaded_model is None:
                return
            if _active_generations > 0:
                _schedule_check(30.0)
                return
            idle_duration = time.time() - _last_used_timestamp
            if idle_duration >= (INACTIVITY_TIMEOUT_SECONDS - 1.0):
                logger.info(
                    f"[LLM Engine] Inattività rilevata ({idle_duration:.1f}s). "
                    f"Auto-unload del modello '{_loaded_model_id}' dalla RAM."
                )
                _release_loaded_model()
                _inactivity_timer = None
                clear_metal_cache()
            else:
                _schedule_check(max(5.0, INACTIVITY_TIMEOUT_SECONDS - idle_duration))
    finally:
        _generation_lock.release()


def _release_loaded_model() -> None:
    """Rilascia il modello corrente (chiamare con _model_lock acquisito)."""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _loaded_model_path, _loaded_model_cfg
    model = _loaded_model
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_model_id = None
    _loaded_model_path = None
    _loaded_model_cfg = None
    if model is not None and not USE_MLX:
        try:
            model.close()  # llama-cpp: libera subito VRAM/mmap (su Windows sblocca anche il file GGUF)
        except Exception:
            pass
    del model
    gc.collect()


def _live_session_active() -> bool:
    try:
        import live_session
        return bool(getattr(live_session, "_active_sessions", None))
    except Exception:
        return False


def unload_whisper(force: bool = False) -> None:
    """
    Libera Whisper dalla memoria prima di caricare o eseguire l'LLM.
    Se una lezione live è in corso (e force=False) Whisper resta caricato per non interrompere la trascrizione.
    """
    if not force and _live_session_active():
        logger.info("[LLM Engine] Lezione live in corso: Whisper resta in memoria.")
        return
    try:
        from transcriber import release_transcriber_memory
        release_transcriber_memory()
        clear_metal_cache()
        logger.info("[LLM Engine] Whisper scaricato dalla RAM con successo: memoria libera per LLM.")
    except Exception as exc:
        logger.warning(f"[LLM Engine] Errore durante l'unload di Whisper: {exc}")


def unload_llm(timeout: Optional[float] = 2.0) -> bool:
    """Scarica il modello LLM attivo dalla RAM in ordine sicuro rispetto alle generazioni."""
    global _inactivity_timer
    acquired = _generation_lock.acquire(timeout=timeout if timeout is not None else -1)
    if not acquired:
        logger.info("[LLM Engine] _generation_lock occupato da un'operazione in uscita: scarico differito al completamento.")
        return False
    try:
        with _model_lock:
            if _inactivity_timer is not None:
                _inactivity_timer.cancel()
                _inactivity_timer = None
            _release_loaded_model()
            clear_metal_cache()
            logger.info("[LLM Engine] Modello LLM scaricato dalla RAM.")
            return True
    finally:
        _generation_lock.release()


def is_model_downloaded(model_id: str) -> bool:
    """Verifica se il modello (ID logico o repo) è completamente presente nella cache locale."""
    spec = ai_models.resolve(model_id, kind="llm")
    if spec is not None:
        return ai_models.is_downloaded(spec)[0]
    if model_id and os.path.isfile(model_id):  # percorso diretto a un file GGUF personalizzato
        return True
    try:
        from routers.models import check_repo_downloaded
        return check_repo_downloaded(model_id)[0]
    except Exception:
        return get_local_snapshot_dir(model_id) is not None


def resolve_llm_model(model_id: Optional[str] = None) -> str:
    """Risolve il modello LLM da utilizzare: model_id esplicito -> impostazioni utente -> DEFAULT_LLM_MODEL."""
    if model_id:
        return model_id
    try:
        from config import load_settings
        settings_model = load_settings().get("llm_model")
        if settings_model:
            return str(settings_model)
    except Exception:
        pass
    return DEFAULT_LLM_MODEL


def _resolve_model_path(target_id: str) -> Tuple[str, Optional[str]]:
    """(chiave_cache, percorso_locale|None) per il backend corrente."""
    spec = ai_models.resolve(target_id, kind="llm")
    if spec is not None:
        return spec.id, ai_models.local_model_path(spec)
    if os.path.isfile(target_id):
        return target_id, target_id
    snap = get_local_snapshot_dir(target_id)
    if snap and not USE_MLX:
        ggufs = sorted(f for f in os.listdir(snap) if f.lower().endswith(".gguf") and "mmproj" not in f.lower())
        return target_id, (os.path.join(snap, ggufs[0]) if ggufs else None)
    return target_id, snap


# ── Backend portable: llama-cpp-python ────────────────────────────────────────

class _GGUFChatTokenizer:
    """Adattatore minimale con l'interfaccia apply_chat_template di HF, basato sul template del GGUF."""

    def __init__(self, llm: Any):
        self._llm = llm
        meta = getattr(llm, "metadata", None) or {}
        self._template_src: Optional[str] = meta.get("tokenizer.chat_template")
        self._add_bos = str(meta.get("tokenizer.ggml.add_bos_token", "false")).lower() == "true"
        self.bos_token = self._tok_text(llm.token_bos())
        self.eos_token = self._tok_text(llm.token_eos())
        self._compiled = None
        if self._template_src:
            try:
                import jinja2
                from jinja2.ext import loopcontrols
                from jinja2.sandbox import ImmutableSandboxedEnvironment

                def _raise(msg):
                    raise jinja2.exceptions.TemplateError(msg)

                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
                env.globals["raise_exception"] = _raise
                env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
                self._compiled = env.from_string(self._template_src)
            except Exception as exc:
                logger.warning(f"[LLM Engine] Template chat GGUF non compilabile ({exc}): uso ChatML.")

    def _tok_text(self, tid: int) -> str:
        try:
            if tid is None or tid < 0:
                return ""
            return self._llm.detokenize([tid], special=True).decode("utf-8", "ignore")
        except Exception:
            return ""

    def apply_chat_template(self, messages: List[dict], tokenize: bool = False,
                            add_generation_prompt: bool = True, **kwargs: Any) -> str:
        text = None
        if self._compiled is not None:
            try:
                text = self._compiled.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token=self.bos_token,
                    eos_token=self.eos_token,
                    **kwargs,
                )
            except Exception as exc:
                logger.debug(f"[LLM Engine] Render template GGUF fallito ({exc}): uso ChatML.")
        if not text:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
            text = "".join(parts)
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
                if kwargs.get("enable_thinking") is False:
                    text += "<think>\n\n</think>\n\n"
        return text

    def encode_prompt(self, prompt: str) -> List[int]:
        toks = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        bos = self._llm.token_bos()
        if self._add_bos and bos is not None and bos >= 0 and (not toks or toks[0] != bos):
            toks = [bos] + toks
        return toks


def _supports_llama_kv_quant(llama_cls: Any, llama_cpp_mod: Optional[Any] = None) -> bool:
    """Verifica in modo sicuro se `Llama.__init__` accetta `type_k`/`type_v` e `GGML_TYPE_Q8_0`."""
    try:
        params = inspect.signature(llama_cls).parameters
    except Exception:
        return False
    if "type_k" in params and "type_v" in params:
        return True
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        if llama_cpp_mod is None:
            return False
        return getattr(llama_cpp_mod, "GGML_TYPE_Q8_0", None) is not None
    return False


def _load_llama(model_path: str, *, safe_only: bool = False, force_cpu: bool = False) -> Tuple[Any, Any]:
    global _loaded_model_cfg
    import llama_cpp
    from llama_cpp import Llama

    q8_type = getattr(llama_cpp, "GGML_TYPE_Q8_0", None)
    supports_kv_quant = (not safe_only) and (not force_cpu) and (q8_type is not None) and _supports_llama_kv_quant(Llama, llama_cpp)
    gpu = (not force_cpu) and llama_gpu_offload_supported()
    logger.info(
        f"[LLM Engine] Caricamento GGUF con llama.cpp ({'GPU offload completo' if gpu else 'CPU'}, "
        f"n_ctx={LLM_CONTEXT_TOKENS}, kv_q8_supported={supports_kv_quant}, safe_only={safe_only}, force_cpu={force_cpu}): {model_path}"
    )

    # type_v quantizzata in llama.cpp richiede flash_attn=True
    configs: List[Tuple[int, bool, bool]] = []
    if force_cpu:
        configs.append((0, False, False))
    elif safe_only:
        if gpu and not (_live_session_active() and sys.platform == "win32"):
            configs.append((-1, False, False))
        configs.append((0, False, False))
    else:
        if gpu:
            configs.extend([
                (-1, True, True),
                (-1, True, False),
                (-1, False, False),
            ])
        configs.extend([
            (0, True, True),
            (0, False, False),
        ])

    if not supports_kv_quant and not safe_only and not force_cpu:
        logger.info("[LLM Engine] KV Cache q8_0 non supportata dalla build di llama-cpp-python: uso KV Cache standard.")

    active_configs = [cfg for cfg in configs if supports_kv_quant or not cfg[2]]
    base_kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": LLM_CONTEXT_TOKENS,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": cpu_threads(),
        "n_threads_batch": os.cpu_count() or cpu_threads(),
        "use_mmap": True,
        "verbose": False,
    }

    for idx, (n_gpu_layers, flash_attn, use_kv_q8) in enumerate(active_configs):
        kv_q8 = use_kv_q8 and supports_kv_quant
        call_kwargs = {
            **base_kwargs,
            "n_gpu_layers": n_gpu_layers,
            "flash_attn": flash_attn,
        }
        if kv_q8:
            call_kwargs["type_k"] = q8_type
            call_kwargs["type_v"] = q8_type
        try:
            model = Llama(**call_kwargs)
        except Exception as exc:
            gc.collect()
            if idx == len(active_configs) - 1:
                raise
            mode = "GPU" if n_gpu_layers != 0 else "CPU"
            logger.warning(
                f"[LLM Engine] Caricamento GGUF {mode} fallito "
                f"(flash_attn={flash_attn}, kv_q8={kv_q8}): {exc}. Fallback alla configurazione successiva..."
            )
            continue

        _loaded_model_cfg = (n_gpu_layers, flash_attn, kv_q8)
        mode = "GPU" if n_gpu_layers != 0 else "CPU"
        if kv_q8:
            logger.info(
                f"[LLM Engine] KV Cache q8_0 attiva su {mode} "
                f"(type_k={q8_type}, type_v={q8_type}, flash_attn={flash_attn})."
            )
        else:
            logger.info(
                f"[LLM Engine] Modello GGUF caricato su {mode} senza KV Cache q8_0 "
                f"(flash_attn={flash_attn}, n_gpu_layers={n_gpu_layers})."
            )
        return model, _GGUFChatTokenizer(model)

    raise RuntimeError("Nessuna configurazione llama.cpp valida disponibile.")


def _can_fallback_safe_llama(current_model: Any) -> bool:
    return (
        not USE_MLX
        and _loaded_model is current_model
        and bool(_loaded_model_path)
        and _loaded_model_cfg is not None
        and (_loaded_model_cfg[0] != 0 or _loaded_model_cfg[1] or _loaded_model_cfg[2])
    )


def _reload_llama_safe(failed_model: Any) -> Optional[Tuple[Any, Any]]:
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _loaded_model_path
    with _model_lock:
        if _loaded_model is not failed_model or not _loaded_model_path:
            return None
        path = _loaded_model_path
        cache_key = _loaded_model_id
        # Se c'è una sessione live attiva o se la configurazione precedente usava già GPU in modalità safe,
        # passa direttamente a CPU pura (n_gpu_layers=0) per eliminare ogni contesa VRAM
        force_cpu = (
            _live_session_active()
            or (_loaded_model_cfg is not None and not _loaded_model_cfg[1] and not _loaded_model_cfg[2])
        )
        _release_loaded_model()
        clear_metal_cache()
        try:
            new_model, new_tok = _load_llama(path, safe_only=True, force_cpu=force_cpu)
        except TypeError:
            new_model, new_tok = _load_llama(path, safe_only=True)
        _loaded_model, _loaded_tokenizer, _loaded_model_id, _loaded_model_path = (
            new_model, new_tok, cache_key, path
        )
        return new_model, new_tok


def get_or_load_model(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Carica il modello LLM (o il default Qwen3.5-4B) per il backend corrente.
    Se già in memoria, riutilizza l'istanza singleton. Ricarica on-demand dopo l'auto-unload.
    Lancia LLMModelNotDownloadedError (sottoclasse di FileNotFoundError) se non scaricato.
    """
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _loaded_model_path, _loaded_model_cfg

    target_id = resolve_llm_model(model_id)
    cache_key, local_path = _resolve_model_path(target_id)

    with _model_lock:
        if _loaded_model is not None and _loaded_model_id == cache_key:
            model, tokenizer = _loaded_model, _loaded_tokenizer
        else:
            if not local_path or not is_model_downloaded(target_id):
                raise LLMModelNotDownloadedError(
                    f"Il modello '{target_id}' non è ancora stato scaricato in locale. "
                    "Vai in Impostazioni -> Modelli per scaricarlo con un clic."
                )

            # Libera Whisper prima di allocare l'LLM (salvo lezione live in corso)
            unload_whisper()

            if _loaded_model is not None:
                _release_loaded_model()
                clear_metal_cache()

            if USE_MLX:
                logger.info(f"[LLM Engine] Caricamento modello MLX su Apple Silicon Metal: {target_id}")
                import mlx_lm
                model, tokenizer = mlx_lm.load(local_path)
                _loaded_model_cfg = None
            else:
                model, tokenizer = _load_llama(local_path)
            _loaded_model, _loaded_tokenizer, _loaded_model_id, _loaded_model_path = (
                model, tokenizer, cache_key, local_path
            )
            logger.info(f"[LLM Engine] Modello '{target_id}' caricato con successo.")

    _touch_model_usage()
    return model, tokenizer


# ── Pulizia output & Disattivazione Thinking per Risposte Flash ─────────────────

def _clean_thinking_artifacts(text: str) -> str:
    """Rimuove eventuali tag <think>...</think> o sezioni 'Thinking Process:'."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned)
    cleaned = re.sub(r"(?i)^Thinking Process:[\s\S]*?(?=\n# |\n## |\n---|\Z)", "", cleaned)
    return cleaned.strip()


def _format_chat_prompt(tokenizer: Any, messages: list[dict], fallback_str: str) -> str:
    """
    Formatta il prompt chat del modello (Qwen 3.5 / Gemma 4) disattivando esplicitamente
    la modalità thinking per eliminare latenza e generare subito l'output utile.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Se il template apre <think> senza chiuderlo, chiudilo per forzare la risposta immediata
        if "<think>" in prompt_str and "</think>" not in prompt_str:
            prompt_str = prompt_str.rstrip() + "\n</think>\n\n"
        return prompt_str
    return fallback_str


def _make_mlx_sampler(temperature: float, top_p: float = 0.9) -> Optional[Any]:
    """Crea il sampler MLX con la temperatura desiderata (0.0 = greedy)."""
    try:
        from mlx_lm.sample_utils import make_sampler
        temp = max(0.0, float(temperature))
        if temp == 0.0:
            return make_sampler(temp=0.0)
        return make_sampler(temp=temp, top_p=top_p, min_p=0.05)
    except Exception as exc:
        logger.debug(f"[LLM Engine] make_sampler non disponibile ({exc}), uso default.")
        return None


def stream_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = TEMP_SUMMARY,
    top_p: float = 0.9,
    epoch: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    kv_bits: Optional[int] = 4,
    kv_group_size: int = 64,
    max_kv_size: Optional[int] = None,
) -> Iterator[str]:
    """
    Generatore backend-agnostico di frammenti di testo. Si interrompe se l'epoch è cancellata
    o se stop_event viene impostato. Va consumato con generation_session() attiva.
    """
    def _stopped() -> bool:
        return (epoch is not None and is_cancelled(epoch)) or (stop_event is not None and stop_event.is_set())

    temp = max(0.0, float(temperature))

    if USE_MLX:
        import mlx_lm
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
        sampler = _make_mlx_sampler(temp, top_p=top_p)
        if sampler is not None:
            kwargs["sampler"] = sampler
        if kv_bits:
            kwargs.update(kv_bits=kv_bits, kv_group_size=kv_group_size)
        if max_kv_size:
            kwargs["max_kv_size"] = max_kv_size
        for response in mlx_lm.stream_generate(model=model, tokenizer=tokenizer, prompt=prompt, **kwargs):
            if _stopped():
                break
            yield response.text
        return

    # llama.cpp: tokenizza una volta, rispetta la finestra di contesto troncando il centro del prompt
    cur_model, cur_tokenizer = model, tokenizer
    for attempt in range(2):
        emitted_any = False
        stream = None
        try:
            if hasattr(cur_tokenizer, "encode_prompt"):
                tokens = cur_tokenizer.encode_prompt(prompt)
            else:
                tokens = cur_model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
            n_ctx = cur_model.n_ctx()
            eff_max_tokens = max(16, min(max_tokens, n_ctx // 2))
            budget = n_ctx - eff_max_tokens - 8
            if len(tokens) > budget:
                head = budget // 3
                tail = budget - head
                logger.warning(f"[LLM Engine] Prompt di {len(tokens)} token troncato a {budget} (n_ctx={n_ctx}).")
                tokens = tokens[:head] + tokens[-tail:]

            stream = cur_model.create_completion(
                prompt=tokens,
                max_tokens=eff_max_tokens,
                stream=True,
                temperature=temp,
                top_p=top_p if temp > 0 else 1.0,
                top_k=30,
                min_p=0.05 if temp > 0 else 0.0,
                repeat_penalty=1.05,
            )
            for chunk in stream:
                if _stopped():
                    return
                text = chunk["choices"][0].get("text") or ""
                if text:
                    emitted_any = True
                    yield text
            return
        except Exception as exc:
            if attempt == 0 and not emitted_any and not _stopped() and _can_fallback_safe_llama(cur_model):
                logger.warning(
                    f"[LLM Engine] Inferenza GGUF fallita al primo token ({exc}); "
                    "ricarico in modalità sicura (flash_attn=False, kv_q8=False)..."
                )
                with contextlib.suppress(Exception):
                    if stream is not None:
                        stream.close()
                stream = None
                reloaded = _reload_llama_safe(cur_model)
                if reloaded is not None:
                    cur_model, cur_tokenizer = reloaded
                    continue
            raise
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()


def _generate_blocking(
    messages: List[dict],
    *,
    model_id: Optional[str],
    max_tokens: int,
    fallback_str: str,
    epoch: int,
    temperature: float = TEMP_SUMMARY,
    assistant_prefix: str = "",
    kv_bits: Optional[int] = 4,
    kv_group_size: int = 64,
    max_kv_size: Optional[int] = None,
    raise_on_cancel: bool = False,
) -> str:
    """Genera il testo completo (chiamare dentro generation_session). Restituisce prefisso + output."""
    model, tokenizer = get_or_load_model(model_id)
    if raise_on_cancel and is_cancelled(epoch):
        raise InterruptedError("Generazione interrotta dall'utente.")
    prompt_str = _format_chat_prompt(tokenizer, messages, fallback_str=fallback_str)
    if assistant_prefix:
        prompt_str += assistant_prefix
    parts: List[str] = [assistant_prefix] if assistant_prefix else []
    try:
        for text in stream_tokens(model, tokenizer, prompt_str, max_tokens=max_tokens,
                                  temperature=temperature, epoch=epoch,
                                  kv_bits=kv_bits, kv_group_size=kv_group_size, max_kv_size=max_kv_size):
            parts.append(text)
        if raise_on_cancel and is_cancelled(epoch):
            raise InterruptedError("Generazione interrotta dall'utente.")
    finally:
        clear_metal_cache()
        _touch_model_usage()
    return "".join(parts)


def generate_text(
    system_prompt: str,
    user_prompt: str,
    *,
    model_id: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = TEMP_STRUCTURED,
    assistant_prefix: str = "",
    task_label: str = "",
    kv_bits: Optional[int] = None,
) -> str:
    """
    Generazione BLOCCANTE backend-agnostica (da chiamare via asyncio.to_thread o da un thread).
    - costruisce il prompt chat (thinking disattivato) e accoda `assistant_prefix` dopo l'apertura
      del turno assistente; il testo restituito INCLUDE il prefisso
    - tiene il lock di generazione e aggiorna lo stato busy / task corrente
    - rispetta stop_all_ai(): in caso di stop solleva InterruptedError
    - restituisce _clean_thinking_artifacts(testo).strip()
    Solleva LLMModelNotDownloadedError (FileNotFoundError) se il modello non è scaricato,
    LLMGenerationError per errori del motore.
    """
    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        raise LLMModelNotDownloadedError(
            f"Il modello '{target}' non è ancora stato scaricato in locale. "
            "Vai in Impostazioni -> Modelli per scaricarlo con un clic."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    with generation_session(task_label) as epoch:
        try:
            text = _generate_blocking(
                messages,
                model_id=target,
                max_tokens=max_tokens,
                fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n",
                epoch=epoch,
                temperature=temperature,
                assistant_prefix=assistant_prefix,
                kv_bits=kv_bits,
                raise_on_cancel=True,
            )
        except (LLMModelNotDownloadedError, InterruptedError):
            raise
        except Exception as exc:
            raise LLMGenerationError(f"Errore durante la generazione del testo: {exc}") from exc
    return _clean_thinking_artifacts(text).strip()


# ── 1. Risposta Flash alle Domande Docente (Live) ──────────────────────────────

async def generate_flash_answer(
    question: str,
    context: str,
    model_id: Optional[str] = None,
) -> str:
    """
    Genera una risposta flash in massimo 3 punti elenco chiari per la domanda rilevata.
    Modalità thinking disattivata per latenza minima.
    """
    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        return (
            "• Risposta non disponibile: modello LLM non ancora scaricato.\n"
            "• Scarica Qwen 3.5 4B dalle Impostazioni per attivare le risposte flash.\n"
            "• Domanda memorizzata per il ripasso e per il Quaderno."
        )

    system_prompt = (
        "Sei SbobinAI, assistente universitario locale. "
        "Rispondi alla domanda del professore in MODO ESTREMAMENTE CONCISO E IMMEDIATO.\n"
        "REGOLE FERREE:\n"
        "1. Rispondi con MASSIMO 3 punti elenco chiari ed essenziali (max 40 parole in totale).\n"
        "2. Nessun preambolo, nessun saluto, nessun tag di ragionamento interno.\n"
        "3. Basa la risposta sul contesto della spiegazione se pertinente."
    )

    user_prompt = f"Contesto spiegato a lezione:\n{context}\n\nDomanda del docente:\n{question}"

    def _run_inference() -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with generation_session("Risposta Flash") as epoch:
            if is_cancelled(epoch):
                raise InterruptedError("Generazione interrotta dall'utente.")
            full_text = _generate_blocking(
                messages,
                model_id=target,
                max_tokens=180,
                fallback_str=f"{system_prompt}\n\nDomanda:\n{user_prompt}\n\nRisposta Flash:",
                epoch=epoch,
                temperature=TEMP_STRUCTURED,
                kv_bits=4,
                max_kv_size=8192,
                raise_on_cancel=True,
            )
        return _clean_thinking_artifacts(full_text)

    try:
        answer = await asyncio.to_thread(_run_inference)
        return answer.strip() or "Risposta completata."
    except InterruptedError:
        logger.info("[LLM Flash] Generazione risposta flash interrotta dall'utente.")
        clear_metal_cache()
        return "Generazione interrotta dall'utente."
    except Exception as exc:
        logger.error(f"[LLM Flash] Errore generazione risposta: {exc}")
        clear_metal_cache()
        return f"Errore durante l'elaborazione della risposta: {exc}"


# ── 2. Chat Live "Non ho capito!" (Streaming Token-by-Token) ───────────────────

async def stream_chat_response(
    query: str,
    context: str,
    model_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Genera la spiegazione in streaming token-by-token per la chat live o la vista assistente.
    Il worker si ferma se il consumatore chiude lo stream (es. disconnessione WebSocket).
    """
    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        yield (
            "Il modello LLM Qwen 3.5 4B non è ancora stato scaricato in locale.\n\n"
            "Apri le **Impostazioni** (rotellina in basso a sinistra) e premi **Scarica** "
            "per abilitare l'assistente IA 100% offline."
        )
        return

    system_prompt = (
        "Sei SbobinAI, assistente accademico e tutor universitario personale dello studente.\n"
        "Rispondi alle domande dello studente basandoti RIGOROSAMENTE sul contesto del corso (trascrizioni, quaderno, materiali e registro domande).\n\n"
        "LINEE GUIDA INDISPENSABILI:\n"
        "1. Rispondi DIRETTAMENTE alla richiesta senza convenevoli, saluti o preamboli inutili (NON iniziare MAI con 'Ciao! Sono SbobinAI' o simili).\n"
        "2. Se lo studente chiede quali domande ha fatto il docente o professore a lezione, elenca ESCLUSIVAMENTE le domande esplicite registrate nel 'REGISTRO DOMANDE RILEVATE A LEZIONE'. Non inventare MAI 'domande implicite' da frasi dichiarative, saluti o presentazioni. Se non risultano domande registrate, comunicalo chiaramente.\n"
        "3. Usa una formattazione Markdown elegante e chiara (elenchi puntati, grassetti per i concetti chiave, corsivo per citazioni dirette).\n"
        "4. Se lo studente chiede spiegazioni su un argomento, spiega il concetto con accuratezza accademica e fornisci esempi pratici se utili.\n"
        "5. Rispondi sempre in lingua italiana."
    )

    user_prompt = f"Contesto del corso (materiali, appunti e registro domande):\n{context}\n\nRichiesta dello studente:\n{query}"

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_flag = threading.Event()

    def _emit(item: Optional[str]) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # event loop chiuso: il consumatore non c'è più
            stop_flag.set()

    def _sync_stream():
        try:
            with generation_session("Chat Assistente") as epoch:
                if stop_flag.is_set() or is_cancelled(epoch):
                    return
                model, tokenizer = get_or_load_model(target)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt_str = _format_chat_prompt(
                    tokenizer,
                    messages,
                    fallback_str=f"{system_prompt}\n\n{user_prompt}\n\nSpiegazione:"
                )

                in_think = False
                for token_text in stream_tokens(model, tokenizer, prompt_str, max_tokens=600,
                                                temperature=TEMP_CHAT, epoch=epoch,
                                                stop_event=stop_flag, kv_bits=4, max_kv_size=8192):
                    # Filtra tag di reasoning interno se presenti
                    if "<think>" in token_text:
                        in_think = True
                        continue
                    if "</think>" in token_text:
                        in_think = False
                        continue
                    if in_think:
                        continue
                    if token_text:
                        _emit(token_text)
                if is_cancelled(epoch):
                    logger.warning("[LLM Chat] Streaming chat interrotto dall'utente.")
        except Exception as exc:
            logger.error(f"[LLM Chat] Errore streaming chat: {exc}")
            _emit(f"\n\n[Errore generazione: {exc}]")
        finally:
            clear_metal_cache()
            _touch_model_usage()
            _emit(None)

    threading.Thread(target=_sync_stream, daemon=True, name="llm-chat-stream").start()

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    finally:
        stop_flag.set()


# ── 3. Generazione Riassunto & Unione al Quaderno (Post-Lezione) ───────────────

_CHAPTER_SYSTEM_PROMPT = (
    "Sei SbobinAI, redattore di appunti universitari accademici di eccellenza. "
    "Il tuo compito è trasformare la trascrizione orale del professore in un capitolo "
    "di quaderno universitario chiaro, approfondito, rigoroso e formattato in elegante Markdown."
)


def _find_split_pos(text: str, start: int, raw_end: int, min_split: int) -> int:
    """Individua il miglior punto di taglio entro `raw_end` privilegiando paragrafi, frasi e parole."""
    # 1. Preferisci fine paragrafo nell'ultimo quarto della finestra (greedy ma coeso)
    para_floor = max(min_split, raw_end - max(1, (raw_end - start) // 4))
    para_idx = text.rfind("\n\n", para_floor, raw_end)
    if para_idx != -1 and para_idx + 2 > start:
        return min(raw_end, para_idx + 2)

    # 2. Altrimenti cerca il confine di frase/paragrafo più vicino a raw_end
    search_hi = min(len(text), raw_end + 1)
    best_sentence_end = -1
    for sep in ("\n\n", ". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.rfind(sep, min_split, search_hi)
        if idx == -1:
            continue
        if sep == "\n\n":
            cand = idx + 2 if idx + 2 <= raw_end else idx
        else:
            cand = idx + 1
        if start < cand <= raw_end and cand > best_sentence_end:
            best_sentence_end = cand
    if best_sentence_end > start:
        return best_sentence_end

    # 3. Fallback sull'ultimo separatore utile tra a-capo e spazio
    best_ws = max(
        text.rfind("\n", min_split, search_hi),
        text.rfind(" ", min_split, search_hi),
    )
    if best_ws != -1:
        cand = best_ws + 1 if best_ws + 1 <= raw_end else best_ws
        if start < cand <= raw_end:
            return cand

    # 4. Nessun separatore (es. stringa continua senza spazi): taglio netto
    return raw_end


def _split_transcript_chunks(
    text: str,
    target_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[str]:
    """
    Suddivide una trascrizione lunga in blocchi coerenti con overlap controllato,
    evitando di tagliare a metà frase o parola quando possibile e garantendo sempre
    avanzamento stretto della finestra (`next_start > start`).
    """
    effective_target = max(1, int(MAP_CHUNK_TARGET_CHARS if target_chars is None else target_chars))
    effective_overlap = max(
        0,
        min(
            int(MAP_CHUNK_OVERLAP_CHARS if overlap_chars is None else overlap_chars),
            effective_target - 1,
        ),
    )

    if len(text) <= effective_target:
        return [text]

    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        raw_end = min(n, start + effective_target)
        if raw_end >= n:
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        min_split = min(raw_end, start + max(effective_overlap + 1, effective_target // 2))
        end = max(start + 1, _find_split_pos(text, start, raw_end, min_split))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        candidate_start = max(start + 1, end - effective_overlap)
        if effective_overlap > 0 and start + 1 < candidate_start < end:
            # Allinea l'inizio dell'overlap al confine di parola precedente per evitare parole spezzate
            if not text[candidate_start - 1].isspace() and not text[candidate_start].isspace():
                back_floor = max(start + 1, candidate_start - 40)
                ws = max(
                    text.rfind(" ", back_floor, candidate_start),
                    text.rfind("\n", back_floor, candidate_start),
                )
                if ws != -1 and start < ws + 1 < end:
                    candidate_start = ws + 1

        start = max(start + 1, candidate_start)

    return chunks


def _build_chapter_prompt(
    title: str,
    source_label: str,
    source_content: str,
    past_references: str,
    bookmarks_summary: str,
) -> str:
    """Compone il prompt utente con la struttura obbligatoria a 5 sezioni del capitolo."""
    return f"""Crea il capitolo di studio per la lezione: "{title}".

{source_label}:
\"\"\"{source_content}\"\"\"

RIFERIMENTI A MATERIALI E LEZIONI PRECEDENTI:
\"\"\"{past_references[:2500]}\"\"\"

SEGNALIBRI ESAME SALVATI DALLO STUDENTE:
\"\"\"{bookmarks_summary[:1200]}\"\"\"

STRUTTURA OBBLIGATORIA DEL CAPITOLO (in Markdown):
# {title} — Capitolo di Studio

## 1. Mappa Concettuale & Punti Chiave
(Sintesi visiva ad elenco dei concetti cardine e delle definizioni principali)

## 2. Spiegazione Approfondita dei Temi Trattati
(Trattazione rigorosa, chiara e dettagliata degli argomenti affrontati a lezione, formalizzando i passaggi orali)

## 3. Esempi Pratici e Casi di Studio del Docente
(Tutti gli esempi, dimostrazioni numeriche o metafore citati dal professore)

## 4. Focus Esame & Segnalibri
(Evidenziazione dei concetti che il docente ha indicato come tipiche domande d'esame e commento sui segnalibri)

## 5. Collegamenti & Riferimenti con le Lezioni Precedenti
(Come questa lezione si inserisce nel percorso del corso e nei materiali caricati)
"""


def _summarize_transcript_chunk(
    title: str,
    chunk: str,
    part_idx: int,
    total_parts: int,
    model_id: Optional[str],
    epoch: int,
) -> str:
    """Fase Map: estrae note accademiche fedeli e dense da un singolo blocco della trascrizione."""
    sys_prompt = (
        "Sei SbobinAI, assistente accademico universitario. "
        "Estrai appunti completi, precisi e fedeli da un estratto parziale di lezione, "
        "senza inventare nulla e conservando definizioni, formule, teoremi, esempi e spunti d'esame."
    )
    usr_prompt = (
        f'Lezione: "{title}" — Parte {part_idx} di {total_parts}.\n'
        "Estrai in punti elenco strutturati e completi:\n"
        "- Concetti, definizioni formali, teoremi e passaggi logici o formule\n"
        "- Esempi pratici, casi di studio e dimostrazioni citati dal docente\n"
        "- Eventuali avvisi o possibili domande d'esame segnalati dal docente\n\n"
        f'TRATTO DI TRASCRIZIONE:\n"""{chunk}"""'
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]
    raw_note = _generate_blocking(
        messages,
        model_id=model_id,
        max_tokens=700,
        fallback_str=f"{sys_prompt}\n\n{usr_prompt}\n\n",
        epoch=epoch,
        temperature=TEMP_SUMMARY,
        kv_bits=4,
        raise_on_cancel=True,
    )
    return _clean_thinking_artifacts(raw_note).strip()


def _condense_block_notes(
    notes: List[str],
    title: str,
    model_id: Optional[str],
    epoch: int,
    max_chars: Optional[int] = None,
) -> str:
    """
    Condensa gerarchicamente a batch gli estratti dei blocchi quando superano `max_chars`,
    evitando di sforare il contesto nella fase Reduce anche su lezioni di 3-4 ore.
    """
    eff_max_chars = REDUCE_MAX_CHARS if max_chars is None else max_chars
    current = [n.strip() for n in notes if n and n.strip()]
    joined = "\n\n".join(current)
    if len(joined) <= eff_max_chars:
        return joined

    batch_char_limit = max(200, min(eff_max_chars, max(eff_max_chars // 2, SINGLE_PASS_TRANSCRIPT_CHARS - 2000)))
    sys_prompt = (
        "Sei SbobinAI, redattore accademico universitario. "
        "Unisci e condensa fedelmente più estratti cronologici di una stessa lezione, "
        "preservando definizioni, formule, teoremi, esempi ed enfasi d'esame ed eliminando solo ridondanze."
    )

    for round_idx in range(1, 5):
        if len(joined) <= eff_max_chars:
            return joined

        batches: List[List[str]] = []
        buf: List[str] = []
        buf_len = 0
        for note in current:
            added_len = len(note) + (2 if buf else 0)
            if buf and buf_len + added_len > batch_char_limit:
                batches.append(buf)
                buf = [note]
                buf_len = len(note)
            else:
                buf.append(note)
                buf_len += added_len
        if buf:
            batches.append(buf)

        logger.info(
            f"[LLM Summary] Condensazione gerarchica round {round_idx}: "
            f"{len(joined)} char -> {len(batches)} batch (limite {eff_max_chars} char)."
        )

        next_notes: List[str] = []
        total_batches = len(batches)
        for b_idx, batch in enumerate(batches, start=1):
            if is_cancelled(epoch):
                raise InterruptedError("Generazione della sintesi interrotta dall'utente.")
            batch_text = "\n\n".join(batch)
            if len(batch) == 1 and len(batch_text) <= batch_char_limit // 3 and total_batches > 1:
                next_notes.append(batch_text)
                continue

            usr_prompt = (
                f'Lezione: "{title}" — Sintesi intermedia blocco {b_idx}/{total_batches}.\n'
                "Condensa in modo denso, fedele e ben ordinato i seguenti estratti mantenendo "
                "tutti i concetti chiave, le formule e gli esempi:\n\n"
                f'"""{batch_text}"""'
            )
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
            try:
                condensed = _clean_thinking_artifacts(
                    _generate_blocking(
                        messages,
                        model_id=model_id,
                        max_tokens=900,
                        fallback_str=f"{sys_prompt}\n\n{usr_prompt}\n\n",
                        epoch=epoch,
                        temperature=TEMP_SUMMARY,
                        kv_bits=4,
                        raise_on_cancel=True,
                    )
                ).strip()
            except InterruptedError:
                raise
            except Exception as exc:
                logger.warning(f"[LLM Summary] Errore condensazione batch {b_idx}/{total_batches}: {exc}")
                condensed = ""
            finally:
                clear_metal_cache()

            if condensed:
                next_notes.append(f"[Sintesi Blocco {b_idx}/{total_batches}]\n{condensed}")
            else:
                budget = max(500, eff_max_chars // max(1, total_batches))
                next_notes.append(batch_text[:budget])

        new_joined = "\n\n".join(next_notes)
        if len(new_joined) >= len(joined):
            logger.warning(
                f"[LLM Summary] Condensazione non ha ridotto la dimensione "
                f"({len(joined)} -> {len(new_joined)} char); tronco a {eff_max_chars} char."
            )
            return new_joined[:eff_max_chars]
        current = next_notes
        joined = new_joined

    return joined[:eff_max_chars] if len(joined) > eff_max_chars else joined


async def generate_lecture_chapter(
    title: str,
    transcript: str,
    past_references: str = "",
    bookmarks_summary: str = "",
    model_id: Optional[str] = None,
    *,
    clean_transcript: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> str:
    """
    Genera il capitolo completo e strutturato per il Quaderno di studio.
    Usa sintesi single-pass per trascrizioni fino a SINGLE_PASS_TRANSCRIPT_CHARS e
    Map-Reduce adattivo (con eventuale condensazione gerarchica) per lezioni lunghe.
    - Modello non scaricato → restituisce il capitolo segnaposto con la trascrizione grezza (come prima).
    - Stop utente → solleva InterruptedError.
    - Errore del motore o output vuoto → solleva LLMGenerationError (sottoclasse di RuntimeError):
      i chiamanti NON devono sovrascrivere appunti esistenti in questo caso.
    """
    if progress_callback:
        progress_callback({"stage": "start", "progress": 5, "message": "Analisi e pulizia trascrizione..."})

    cleaned_transcript = clean_transcript_for_llm(transcript) if clean_transcript else (transcript or "")
    if transcript and len(cleaned_transcript) < len(transcript):
        saved_pct = round((1.0 - (len(cleaned_transcript) / max(1, len(transcript)))) * 100.0, 1)
        logger.info(
            f"[LLM Summary] Trascrizione alleggerita prima dell'LLM: "
            f"{len(transcript)} -> {len(cleaned_transcript)} char (-{saved_pct}%)."
        )

    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        return (
            f"# {title} — Appunti di Lezione\n\n"
            "> [!NOTE]\n"
            "> Il modello IA non è ancora installato. Scarica **Qwen 3.5 4B** dalle Impostazioni "
            "per generare automaticamente lo schema concettuale strutturato.\n\n"
            "## Trascrizione Grezza della Lezione\n\n"
            f"{cleaned_transcript[:3000]}..."
        )

    system_prompt = _CHAPTER_SYSTEM_PROMPT

    def _notify(info: Dict[str, Any]) -> None:
        if progress_callback:
            try:
                progress_callback(info)
            except Exception:
                pass

    def _run_summary() -> str:
        with generation_session(f"Sintesi Quaderno: {title}") as epoch:
            if is_cancelled(epoch):
                raise InterruptedError("Generazione della sintesi interrotta dall'utente.")
            # Libera Whisper (salvo live in corso) ed Embedding per lasciare la memoria all'LLM
            unload_whisper()
            try:
                from rag import unload_embedding_model
                unload_embedding_model()
            except Exception:
                pass

            if len(cleaned_transcript) <= SINGLE_PASS_TRANSCRIPT_CHARS:
                _notify({"stage": "single_pass", "progress": 25, "message": "Redazione del capitolo strutturato in corso..."})
                user_prompt = _build_chapter_prompt(
                    title=title,
                    source_label="TRASCRIZIONE INTEGRALE DELLA LEZIONE",
                    source_content=cleaned_transcript[:SINGLE_PASS_TRANSCRIPT_CHARS],
                    past_references=past_references,
                    bookmarks_summary=bookmarks_summary,
                )
            else:
                chunks = _split_transcript_chunks(cleaned_transcript)
                total_chunks = len(chunks)
                logger.info(
                    f"[LLM Summary] Trascrizione lunga ({len(cleaned_transcript)} char): "
                    f"avvio Map-Reduce adattivo su {total_chunks} blocchi."
                )

                block_notes: List[str] = []
                for idx, chunk in enumerate(chunks, start=1):
                    if is_cancelled(epoch):
                        raise InterruptedError("Generazione della sintesi interrotta dall'utente.")
                    pct = int(10 + (idx / total_chunks) * 55)
                    _notify({
                        "stage": "chunk",
                        "current": idx,
                        "total": total_chunks,
                        "progress": pct,
                        "message": f"Sintesi blocco concettuale {idx} di {total_chunks}..."
                    })
                    try:
                        note = _summarize_transcript_chunk(
                            title=title,
                            chunk=chunk,
                            part_idx=idx,
                            total_parts=total_chunks,
                            model_id=target,
                            epoch=epoch,
                        )
                    except InterruptedError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            f"[LLM Summary] Errore estrazione blocco {idx}/{total_chunks}: {exc}"
                        )
                        note = ""
                    finally:
                        clear_metal_cache()
                    if note:
                        block_notes.append(f"[Parte {idx}/{total_chunks}]\n{note}")

                if not block_notes:
                    raise LLMGenerationError(
                        "Tutti i blocchi della trascrizione hanno prodotto un output vuoto o fallito."
                    )

                effective_reduce_max = min(
                    REDUCE_MAX_CHARS,
                    max(3500, int((LLM_CONTEXT_TOKENS - 2800) * 2.5)),
                )
                _notify({
                    "stage": "condense",
                    "progress": 72,
                    "message": "Armonizzazione e collegamento concetti chiave..."
                })
                combined_notes = _condense_block_notes(
                    block_notes,
                    title=title,
                    model_id=target,
                    epoch=epoch,
                    max_chars=effective_reduce_max,
                )
                if is_cancelled(epoch):
                    raise InterruptedError("Generazione della sintesi interrotta dall'utente.")

                user_prompt = _build_chapter_prompt(
                    title=title,
                    source_label="ESTRATTI CRONOLOGICI STRUTTURATI DELLA LEZIONE",
                    source_content=combined_notes,
                    past_references=past_references,
                    bookmarks_summary=bookmarks_summary,
                )

            _notify({
                "stage": "final",
                "progress": 82,
                "message": "Redazione del capitolo finale con definizioni ed enfasi d'esame..."
            })
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            full_chapter = _generate_blocking(
                messages,
                model_id=target,
                max_tokens=2500,
                fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n# {title} — Capitolo di Studio\n",
                epoch=epoch,
                temperature=TEMP_SUMMARY,
                kv_bits=4,
                raise_on_cancel=True,
            )
        return _clean_thinking_artifacts(full_chapter)

    try:
        chapter = (await asyncio.to_thread(_run_summary)).strip()
        if progress_callback:
            progress_callback({
                "stage": "completed",
                "progress": 100,
                "message": "Capitolo completato con successo!"
            })
    except InterruptedError:
        logger.info("[LLM Summary] Operazione annullata dall'utente. Rilascio memoria.")
        raise
    except LLMGenerationError:
        raise
    except Exception as exc:
        logger.error(f"[LLM Summary] Errore generazione capitolo: {exc}")
        raise LLMGenerationError(f"Errore durante la generazione del capitolo: {exc}") from exc
    finally:
        clear_metal_cache()
    if not chapter:
        raise LLMGenerationError("Il modello ha restituito un capitolo vuoto.")
    return chapter


# ── 4. Chat Multi-Turno Persistente, Compattazione & Titoli/Follow-up ─────────

_CHAT_MODE_INSTRUCTIONS: Dict[str, str] = {
    "socratic": (
        "MODALITÀ SOCRATICA: Guida lo studente al ragionamento passo dopo passo "
        "ponendo 1-2 domande mirate e brevi indizi, senza svelare subito tutta la soluzione."
    ),
    "simple": (
        "MODALITÀ SEMPLICE (FEYNMAN): Spiega con linguaggio chiaro e analogie quotidiane intuitive, "
        "collegandole poi ai termini precisi del corso."
    ),
    "oral": (
        "MODALITÀ ESAME ORALE: Comportati come un docente universitario in sede d'esame. "
        "Inizia subito formulando chiaramente UNA domanda d'esame orale realistica e stimolante "
        "(su una riga dedicata), poi elenca brevemente i 3-4 concetti chiave che lo studente deve coprire."
    ),
    "exam": (
        "MODALITÀ ESAME RIGOROSO: Rispondi con profondità accademica universitaria completa, "
        "curando definizioni formali, ipotesi, dimostrazioni essenziali ed errori tipici d'esame."
    ),
    "quiz": (
        "MODALITÀ VERIFICA: Organizza la spiegazione in punti chiari e verificabili per il ripasso attivo."
    ),
    "auto": (
        "MODALITÀ ACCADEMICA: Rispondi in modo nitido, strutturato e didatticamente efficace."
    ),
}


def _build_multi_turn_system_prompt(
    subject_name: str,
    context_text: str,
    compacted_summary: str,
    mode: str,
) -> str:
    """Compone il system prompt italiano per la chat multi-turno con citazioni e LaTeX."""
    mode_key = (mode or "auto").strip().lower()
    mode_instruction = _CHAT_MODE_INSTRUCTIONS.get(mode_key, _CHAT_MODE_INSTRUCTIONS["auto"])

    parts: List[str] = [
        f"Sei SbobinAI, tutor universitario per il corso di '{subject_name}'.",
        "REGOLE INDISPENSABILI:",
        "1. Rispondi in italiano in modo diretto, senza saluti o preamboli inutili.",
        "2. Fonda la risposta sui materiali del corso forniti nel contesto; se un dettaglio manca dai materiali, segnalalo con onestà senza inventarlo.",
        (
            "3. Quando citi una fonte contrassegnata nel contesto come "
            "[FONTE: ... | REF: SOURCE_ID#PAGE | ETICHETTA: LABEL], inserisci SEMPRE un link inline cliccabile "
            "con la sintassi ESATTA [[ref:SOURCE_ID#PAGE|LABEL]] "
            "(esempio: [[ref:Lezione3.pdf#14|Lezione 3, pag. 14]]). "
            "Non inventare MAI numeri di pagina non presenti nei tag [FONTE: ... | REF: ...]."
        ),
        "4. Usa la notazione LaTeX ($...$ inline e $$...$$ a blocco) per formule ed espressioni matematiche.",
        f"5. {mode_instruction}",
    ]

    summary_clean = (compacted_summary or "").strip()
    if summary_clean:
        max_summary_chars = min(2000, max(600, USABLE_CHAT_CONTEXT_TOKENS // 3))
        parts.append(f"\nMEMORIA COMPATTATA DELLA CONVERSAZIONE PRECEDENTE:\n{summary_clean[:max_summary_chars]}")

    ctx_clean = (context_text or "").strip()
    if ctx_clean:
        # Riserva un budget equilibrato per la cronologia anche con finestre di contesto ridotte
        max_ctx_chars = min(10000, max(1600, USABLE_CHAT_CONTEXT_TOKENS * 2))
        parts.append(f"\nMATERIALI E APPUNTI DEL CORSO:\n{ctx_clean[:max_ctx_chars]}")

    return "\n".join(parts)


def _fit_messages_to_budget(messages: List[dict], system_prompt: str) -> List[dict]:
    """Seleziona i turni dal più recente al più vecchio restando entro USABLE_CHAT_CONTEXT_TOKENS."""
    normalized: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            normalized.append({"role": role, "content": content})

    if not normalized:
        return [{"role": "system", "content": system_prompt}]

    sys_tokens = estimate_tokens(system_prompt) + 12
    remaining = max(256, USABLE_CHAT_CONTEXT_TOKENS - sys_tokens)

    selected_rev: List[dict] = []
    used = 0
    for msg in reversed(normalized):
        cost = estimate_tokens(msg["content"]) + 6
        if not selected_rev:
            # Garantisci sempre almeno l'ultimo turno utente troncandolo se eccede il budget
            if cost > remaining:
                max_chars = max(200, (remaining - 6) * 4)
                selected_rev.append({"role": msg["role"], "content": msg["content"][-max_chars:]})
            else:
                selected_rev.append(msg)
            used += min(cost, remaining)
            continue
        if used + cost > remaining:
            break
        selected_rev.append(msg)
        used += cost

    selected_rev.reverse()
    return [{"role": "system", "content": system_prompt}, *selected_rev]


def _messages_to_fallback_prompt(chat_messages: List[dict]) -> str:
    """Fallback testuale lineare se il tokenizer non espone apply_chat_template."""
    lines = [f"{m['role']}: {m['content']}" for m in chat_messages]
    lines.append("assistant:")
    return "\n\n".join(lines)


def stream_multi_turn_chat(
    subject_name: str,
    messages: List[dict],
    context_text: str = "",
    compacted_summary: str = "",
    mode: str = "auto",
    model_id: Optional[str] = None,
) -> Iterator[str]:
    """
    Generatore sincrono per la chat multi-turno persistente.
    Adatta i turni al budget di contesto, applica la modalità di studio e streamma i token a TEMP_CHAT.
    """
    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        yield (
            "Il modello LLM non è ancora stato scaricato in locale.\n\n"
            "Apri le **Impostazioni** e scarica il modello per abilitare l'assistente IA 100% offline."
        )
        return

    system_prompt = _build_multi_turn_system_prompt(
        subject_name=subject_name,
        context_text=context_text,
        compacted_summary=compacted_summary,
        mode=mode,
    )
    chat_messages = _fit_messages_to_budget(messages, system_prompt)
    fallback_str = _messages_to_fallback_prompt(chat_messages)

    with generation_session(f"Chat: {subject_name}") as epoch:
        if is_cancelled(epoch):
            return
        model, tokenizer = get_or_load_model(target)
        prompt_str = _format_chat_prompt(tokenizer, chat_messages, fallback_str=fallback_str)
        try:
            for chunk in stream_tokens(
                model,
                tokenizer,
                prompt_str,
                max_tokens=1200,
                temperature=TEMP_CHAT,
                epoch=epoch,
                kv_bits=4,
                max_kv_size=LLM_CONTEXT_TOKENS,
            ):
                if is_cancelled(epoch):
                    break
                if chunk:
                    yield chunk
        finally:
            clear_metal_cache()
            _touch_model_usage()


def _extractive_chat_summary(
    subject_name: str,
    previous_summary: str,
    messages: List[dict],
) -> str:
    """Fallback deterministico per la compattazione della chat quando l'LLM non è disponibile."""
    lines: List[str] = [f"### Sintesi Compattata — {subject_name}"]
    prev = (previous_summary or "").strip()
    if prev:
        lines.append(prev[:1200])

    total_chars = sum(len(x) for x in lines)
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = re.sub(r"\s+", " ", str(m.get("content") or "")).strip()
        if not content:
            continue
        label = "Studente" if role == "user" else "Assistente"
        refs = re.findall(r"\[\[ref:[^\]]+\]\]", content)
        snippet = content[:240] + ("…" if len(content) > 240 else "")
        if refs:
            missing_refs = [r for r in refs if r not in snippet]
            if missing_refs:
                snippet = f"{snippet} ({', '.join(missing_refs[:2])})"
        bullet = f"- **{label}**: {snippet}"
        if total_chars + len(bullet) > 2600:
            break
        lines.append(bullet)
        total_chars += len(bullet)
    return "\n".join(lines).strip()


def compact_chat_history(
    subject_name: str,
    previous_summary: str,
    messages: List[dict],
    model_id: Optional[str] = None,
) -> str:
    """
    Unisce `previous_summary` e i turni `messages` in una sintesi strutturata in italiano
    conservando definizioni, formule, dubbi aperti e riferimenti alle slide.
    Include un fallback estrattivo deterministico e non solleva eccezioni se il modello manca.
    """
    if not messages:
        return (previous_summary or "").strip()

    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        return _extractive_chat_summary(subject_name, previous_summary, messages)

    turns_text_parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = "Studente" if str(m.get("role") or "").lower() == "user" else "Assistente"
        content = str(m.get("content") or "").strip()
        if content:
            turns_text_parts.append(f"{role}: {content[:1200]}")

    if not turns_text_parts:
        return (previous_summary or "").strip()

    turns_block = "\n\n".join(turns_text_parts)[:9000]
    prev_block = (previous_summary or "").strip()[:2500] or "Nessuna sintesi precedente."

    sys_prompt = (
        "Sei SbobinAI, assistente accademico universitario. "
        "Comprimi lo storico di una sessione di studio in una memoria compatta e fedele in italiano. "
        "Preserva SEMPRE: definizioni formali, formule LaTeX ($...$), dubbi aperti dello studente "
        "e riferimenti puntuali alle fonti (es. [[ref:...]] o nomi file/pagine)."
    )
    usr_prompt = (
        f"Materia: {subject_name}\n\n"
        f"SINTESI PRECEDENTE:\n{prev_block}\n\n"
        f"NUOVI TURNI DA INTEGRARE:\n{turns_block}\n\n"
        "Restituisci una sintesi Markdown concisa e strutturata (punti elenco per concetti, formule, dubbi aperti e riferimenti):"
    )
    chat_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]

    try:
        with generation_session(f"Compattazione Chat: {subject_name}") as epoch:
            raw = _generate_blocking(
                chat_messages,
                model_id=target,
                max_tokens=700,
                fallback_str=f"{sys_prompt}\n\n{usr_prompt}\n\nSintesi:",
                epoch=epoch,
                temperature=TEMP_SUMMARY,
                kv_bits=4,
                raise_on_cancel=True,
            )
        cleaned = _clean_thinking_artifacts(raw).strip()
        return cleaned if cleaned else _extractive_chat_summary(subject_name, previous_summary, messages)
    except Exception as exc:
        logger.warning(f"[LLM Chat] Fallback estrattivo compattazione per '{subject_name}': {exc}")
        return _extractive_chat_summary(subject_name, previous_summary, messages)


def generate_chat_title(first_user_message: str) -> str:
    """Genera un titolo pulito di 4-7 parole (max 48 caratteri) dal primo messaggio utente."""
    clean = re.sub(r"\s+", " ", (first_user_message or "")).strip(" \t\r\n\"'«»?!.:;,")
    if not clean:
        return "Nuova chat"

    words = clean.split(" ")
    candidate = " ".join(words[:7])
    if len(candidate) > 48:
        truncated = candidate[:48].rsplit(" ", 1)[0].strip(" ,;:-")
        candidate = truncated if truncated else candidate[:48].strip()
    return candidate[:48].strip() or "Nuova chat"


def generate_followup_suggestions(
    subject_name: str,
    user_message: str,
    assistant_reply: str,
    mode: str,
) -> List[str]:
    """
    Restituisce 3 suggerimenti di follow-up rapidi e deterministici in italiano,
    adattati alla modalità di studio e alle parole chiave della risposta.
    """
    combined = f"{user_message or ''} {assistant_reply or ''}".lower()
    mode_key = (mode or "auto").strip().lower()

    suggestions: List[str] = []

    if mode_key == "socratic":
        suggestions.extend([
            "Dammi un altro indizio per verificare il mio ragionamento",
            "Riassumiamo i passaggi chiave fin qui",
            "Fammi un esempio pratico per controllare se ho capito",
        ])
    elif mode_key == "simple":
        suggestions.extend([
            "Ora mostrami la definizione formale da esame",
            "Fammi un altro esempio semplice applicato",
            "Qual è l'errore più comune su questo argomento?",
        ])
    elif mode_key == "oral":
        suggestions.extend([
            "Valuta la mia risposta e dimmi cosa manca",
            "Fammi una seconda domanda d'esame collegata",
            "Quali parole chiave vuole sentire il docente?",
        ])
    elif mode_key == "exam":
        suggestions.extend([
            "Evidenzia ipotesi critiche e controesempi d'esame",
            "Quali domande orali tipiche fa il professore su questo tema?",
            "Genera un quiz rapido di verifica su questo argomento",
        ])
    elif mode_key == "quiz":
        suggestions.extend([
            "Spiegami gli errori più frequenti su questi concetti",
            "Genera un altro quiz più difficile",
            "Fammi una domanda aperta in stile orale",
        ])
    else:
        if any(k in combined for k in ("teorema", "dimostraz", "lemma", "corollario")):
            suggestions.append("Mostrami i passaggi intuitivi della dimostrazione")
        if any(k in combined for k in ("formula", "equazione", "$", "integrale", "derivata", "matrice")):
            suggestions.append("Fammi un esercizio svolto passo passo con questa formula")
        if "[[ref:" in combined or "pag." in combined or "slide" in combined:
            suggestions.append("Collega questo punto con le altre lezioni del corso")
        if any(k in combined for k in ("differenza", "confront", "vs")):
            suggestions.append("Crea una tabella comparativa sintetica")
        suggestions.extend([
            f"Fammi un esempio pratico su questo tema di {subject_name}",
            "Quali domande d'esame può fare il professore qui?",
            "Genera un quiz di 5 domande per verificare se ho capito",
        ])

    unique: List[str] = []
    for item in suggestions:
        if item not in unique:
            unique.append(item)
        if len(unique) == 3:
            break
    return unique


# Alias di compatibilità
explain_quick_doubt = generate_flash_answer
