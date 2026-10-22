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
import logging
import os
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple

import ai_models
from platform_utils import USE_MLX, cpu_threads, get_local_snapshot_dir, llama_gpu_offload_supported

logger = logging.getLogger("sbobinai.llm")

# ── Modelli Supportati (ID logici, identici su tutte le piattaforme) ──────────
DEFAULT_LLM_MODEL = ai_models.LLM_DEFAULT_ID
ALTERNATIVE_LLM_MODEL = ai_models.LLM_ALT_ID

SUPPORTED_LLM_MODELS = {
    mid: {
        "name": ai_models.CATALOG[mid].variant.name,
        "repo_id": ai_models.CATALOG[mid].repo_id,
        "approx_ram": ai_models.human_size(ai_models.CATALOG[mid].variant.approx_bytes),
    }
    for mid in (DEFAULT_LLM_MODEL, ALTERNATIVE_LLM_MODEL)
}

LLM_CONTEXT_TOKENS = int(os.environ.get("SBOBINAI_LLM_CTX", "8192"))


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
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _inactivity_timer
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


def _release_loaded_model() -> None:
    """Rilascia il modello corrente (chiamare con _model_lock acquisito)."""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id
    model = _loaded_model
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_model_id = None
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


def unload_llm() -> None:
    """Scarica il modello LLM attivo dalla RAM."""
    global _inactivity_timer
    with _model_lock:
        if _inactivity_timer is not None:
            _inactivity_timer.cancel()
            _inactivity_timer = None
        _release_loaded_model()
        clear_metal_cache()
        logger.info("[LLM Engine] Modello LLM scaricato dalla RAM.")


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


def _load_llama(model_path: str) -> Tuple[Any, Any]:
    from llama_cpp import Llama

    gpu = llama_gpu_offload_supported()
    logger.info(
        f"[LLM Engine] Caricamento GGUF con llama.cpp ({'GPU offload completo' if gpu else 'CPU'}, "
        f"n_ctx={LLM_CONTEXT_TOKENS}): {model_path}"
    )
    model = Llama(
        model_path=model_path,
        n_ctx=LLM_CONTEXT_TOKENS,
        n_batch=512,
        n_ubatch=512,
        n_gpu_layers=-1 if gpu else 0,
        n_threads=cpu_threads(),
        n_threads_batch=os.cpu_count() or cpu_threads(),
        flash_attn=gpu,
        use_mmap=True,
        verbose=False,
    )
    return model, _GGUFChatTokenizer(model)


def get_or_load_model(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Carica il modello LLM (o il default Qwen3.5-4B) per il backend corrente.
    Se già in memoria, riutilizza l'istanza singleton. Ricarica on-demand dopo l'auto-unload.
    Lancia LLMModelNotDownloadedError (sottoclasse di FileNotFoundError) se non scaricato.
    """
    global _loaded_model, _loaded_tokenizer, _loaded_model_id

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
            else:
                model, tokenizer = _load_llama(local_path)
            _loaded_model, _loaded_tokenizer, _loaded_model_id = model, tokenizer, cache_key
            logger.info(f"[LLM Engine] Modello '{target_id}' caricato con successo.")

    _touch_model_usage()
    return model, tokenizer


# ── Pulizia output & Disattivazione Thinking per Risposte Flash ─────────────────

def _clean_thinking_artifacts(text: str) -> str:
    """Rimuove eventuali tag <think>...</think> o sezioni 'Thinking Process:'."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
    cleaned = re.sub(r"^<think>[\s\S]*$", "", cleaned)
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


def stream_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
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

    if USE_MLX:
        import mlx_lm
        kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
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
    if hasattr(tokenizer, "encode_prompt"):
        tokens = tokenizer.encode_prompt(prompt)
    else:
        tokens = model.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
    n_ctx = model.n_ctx()
    max_tokens = max(16, min(max_tokens, n_ctx // 2))
    budget = n_ctx - max_tokens - 8
    if len(tokens) > budget:
        head = budget // 3
        tail = budget - head
        logger.warning(f"[LLM Engine] Prompt di {len(tokens)} token troncato a {budget} (n_ctx={n_ctx}).")
        tokens = tokens[:head] + tokens[-tail:]

    stream = model.create_completion(
        prompt=tokens,
        max_tokens=max_tokens,
        stream=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        repeat_penalty=1.05,
    )
    try:
        for chunk in stream:
            if _stopped():
                break
            text = chunk["choices"][0].get("text") or ""
            if text:
                yield text
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def _generate_blocking(
    messages: List[dict],
    *,
    model_id: Optional[str],
    max_tokens: int,
    fallback_str: str,
    epoch: int,
    assistant_prefix: str = "",
    kv_bits: Optional[int] = 4,
    kv_group_size: int = 64,
    max_kv_size: Optional[int] = None,
    raise_on_cancel: bool = False,
) -> str:
    """Genera il testo completo (chiamare dentro generation_session). Restituisce prefisso + output."""
    model, tokenizer = get_or_load_model(model_id)
    prompt_str = _format_chat_prompt(tokenizer, messages, fallback_str=fallback_str)
    if assistant_prefix:
        prompt_str += assistant_prefix
    parts: List[str] = [assistant_prefix] if assistant_prefix else []
    try:
        for text in stream_tokens(model, tokenizer, prompt_str, max_tokens=max_tokens, epoch=epoch,
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
    assistant_prefix: str = "",
    task_label: str = "",
    kv_bits: Optional[int] = None,
) -> str:
    """
    Generazione BLOCCANTE backend-agnostica (da chiamare via asyncio.to_thread o da un thread).
    - costruisce il prompt chat (thinking disattivato) e accoda `assistant_prefix` dopo l'apertura
      del turno assistente; il testo restituito INCLUDE il prefisso
    - tiene il lock di generazione e aggiorna lo stato busy / task corrente
    - rispetta stop_all_ai(): in caso di stop restituisce il testo parziale
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
                assistant_prefix=assistant_prefix,
                kv_bits=kv_bits,
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
            full_text = _generate_blocking(
                messages,
                model_id=target,
                max_tokens=180,
                fallback_str=f"{system_prompt}\n\nDomanda:\n{user_prompt}\n\nRisposta Flash:",
                epoch=epoch,
                kv_bits=4,
                max_kv_size=8192,
            )
        return _clean_thinking_artifacts(full_text)

    try:
        answer = await asyncio.to_thread(_run_inference)
        return answer.strip() or "Risposta completata."
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
                for token_text in stream_tokens(model, tokenizer, prompt_str, max_tokens=600, epoch=epoch,
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

async def generate_lecture_chapter(
    title: str,
    transcript: str,
    past_references: str = "",
    bookmarks_summary: str = "",
    model_id: Optional[str] = None,
) -> str:
    """
    Genera il capitolo completo e strutturato per il Quaderno di studio.
    - Modello non scaricato → restituisce il capitolo segnaposto con la trascrizione grezza (come prima).
    - Stop utente → solleva InterruptedError.
    - Errore del motore o output vuoto → solleva LLMGenerationError (sottoclasse di RuntimeError):
      i chiamanti NON devono sovrascrivere appunti esistenti in questo caso.
    """
    target = resolve_llm_model(model_id)
    if not is_model_downloaded(target):
        return (
            f"# {title} — Appunti di Lezione\n\n"
            "> [!NOTE]\n"
            "> Il modello IA non è ancora installato. Scarica **Qwen 3.5 4B** dalle Impostazioni "
            "per generare automaticamente lo schema concettuale strutturato.\n\n"
            "## Trascrizione Grezza della Lezione\n\n"
            f"{transcript[:3000]}..."
        )

    system_prompt = (
        "Sei SbobinAI, redattore di appunti universitari accademici di eccellenza. "
        "Il tuo compito è trasformare la trascrizione orale del professore in un capitolo "
        "di quaderno universitario chiaro, approfondito, rigoroso e formattato in elegante Markdown."
    )

    user_prompt = f"""Crea il capitolo di studio per la lezione: "{title}".

TRASCRIZIONE INTEGRALE DELLA LEZIONE:
\"\"\"{transcript[:12000]}\"\"\"

RIFERIMENTI A MATERIALI E LEZIONI PRECEDENTI:
\"\"\"{past_references[:2500]}\"\"\"

SEGNALIBRI ESAME SALVATI DALLO STUDENTE:
\"\"\"{bookmarks_summary}\"\"\"

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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            full_chapter = _generate_blocking(
                messages,
                model_id=target,
                max_tokens=1200,
                fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n# {title} — Capitolo di Studio\n",
                epoch=epoch,
                kv_bits=4,
                raise_on_cancel=True,
            )
        return _clean_thinking_artifacts(full_chapter)

    try:
        chapter = (await asyncio.to_thread(_run_summary)).strip()
    except InterruptedError:
        logger.info("[LLM Summary] Operazione annullata dall'utente. Rilascio memoria.")
        raise
    except Exception as exc:
        logger.error(f"[LLM Summary] Errore generazione capitolo: {exc}")
        raise LLMGenerationError(f"Errore durante la generazione del capitolo: {exc}") from exc
    finally:
        clear_metal_cache()
    if not chapter:
        raise LLMGenerationError("Il modello ha restituito un capitolo vuoto.")
    return chapter


# Alias di compatibilità
explain_quick_doubt = generate_flash_answer
