"""
llm_engine.py — SbobinAI Pure MLX-LM Engine
Motore LLM 100% nativo Apple Silicon basato su mlx-lm e modelli 2026 4-bit
(Qwen3.5-4B / Gemma-4-E4B), ottimizzato per MacBook Air M1 con 8GB di RAM unificata.

Ottimizzazioni per M1 8GB:
  - KV Cache quantizzata 4-bit (kv_bits=4, kv_group_size=64, max_kv_size=8192 live)
  - Gestione Metal Cache: mx.clear_cache() post-generazione per liberare subito la RAM unificata
  - Unload coordinato di Whisper prima del riassunto pesante della lezione
  - Zero auto-download: non scarica pesi senza esplicita richiesta dell'utente
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

logger = logging.getLogger("sbobinai.llm")

# ── Modelli Supportati (2026 4-bit MLX) ─────────────────────────────────────────
DEFAULT_LLM_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
ALTERNATIVE_LLM_MODEL = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"

SUPPORTED_LLM_MODELS = {
    DEFAULT_LLM_MODEL: {
        "name": "Qwen 3.5 4B (4-bit MLX)",
        "repo_id": DEFAULT_LLM_MODEL,
        "approx_ram": "~2.3 GB",
    },
    ALTERNATIVE_LLM_MODEL: {
        "name": "Gemma 4 E4B (4-bit MLX)",
        "repo_id": ALTERNATIVE_LLM_MODEL,
        "approx_ram": "~2.4 GB",
    },
}

# ── Stato Singleton del Modello in RAM & Auto-Unload Inattività (120s su M1 8GB) ─
INACTIVITY_TIMEOUT_SECONDS = 120.0

_model_lock = threading.Lock()
_loaded_model: Optional[Any] = None
_loaded_tokenizer: Optional[Any] = None
_loaded_model_id: Optional[str] = None
_last_used_timestamp: float = 0.0
_inactivity_timer: Optional[threading.Timer] = None

# ── Stato di Generazione & Cancellazione di Emergenza ─────────────────────────
_cancel_generation_event = threading.Event()
_is_generating = False
_current_task: str = ""


def is_ai_busy() -> Dict[str, Any]:
    """Restituisce lo stato attuale di occupazione del motore IA."""
    return {
        "is_busy": _is_generating,
        "task": _current_task,
        "model_loaded": _loaded_model is not None,
        "model_id": _loaded_model_id,
    }


def stop_all_ai(force_unload: bool = True) -> Dict[str, Any]:
    """
    Segnale di emergenza immediato:
    1. Interrompe qualsiasi generazione in corso (sintesi quaderno, chat, risposte flash).
    2. Scarica Whisper e (se force_unload=True) l'LLM dalla RAM.
    3. Svuota la cache Metal (mx.clear_cache) azzerando all'istante l'utilizzo di CPU e GPU per raffreddare il Mac.
    """
    global _cancel_generation_event, _is_generating, _current_task
    logger.warning("[LLM Engine] 🛑 INTERRUZIONE EMERGENZA IA RICHIESTA DALL'UTENTE.")
    _cancel_generation_event.set()
    _is_generating = False
    _current_task = ""

    # Libera Whisper
    unload_whisper()

    # Libera modello embedding se presente per massimizzare la RAM libera
    try:
        from rag import unload_embedding_model
        unload_embedding_model()
    except Exception:
        pass

    # Se richiesto, scarica l'LLM dalla RAM unificata
    if force_unload:
        unload_llm()

    clear_metal_cache()

    # Ripristina l'evento di cancellazione dopo breve timeout
    def _reset_cancel():
        time.sleep(0.4)
        _cancel_generation_event.clear()

    threading.Thread(target=_reset_cancel, daemon=True).start()

    logger.info("[LLM Engine] ❄️ Processi IA interrotti e memoria Metal azzerata con successo.")
    return {
        "status": "stopped",
        "freed_memory": True,
        "message": "Tutti i processi IA sono stati interrotti e la memoria Metal è stata liberata.",
    }


def clear_metal_cache() -> None:
    """
    Rilascia immediatamente la memoria grafica/Metal allocata da MLX a macOS.
    Fondamentale sull'M1 8GB per evitare qualsiasi memory pressure o swap su disco.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception as exc:
        logger.debug(f"[LLM Engine] clear_cache: {exc}")
    gc.collect()


def _touch_model_usage() -> None:
    """Registra l'uso del modello e avvia o riavvia il timer di auto-unload dopo 180s di inattività."""
    global _last_used_timestamp, _inactivity_timer
    _last_used_timestamp = time.time()

    with _model_lock:
        if _inactivity_timer is not None:
            _inactivity_timer.cancel()
            _inactivity_timer = None

        if _loaded_model is not None:
            _inactivity_timer = threading.Timer(INACTIVITY_TIMEOUT_SECONDS, _check_and_auto_unload)
            _inactivity_timer.daemon = True
            _inactivity_timer.start()


def _check_and_auto_unload() -> None:
    """Verifica se il modello è rimasto inattivo per oltre 120s e lo scarica dalla RAM se non sta generando."""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _inactivity_timer
    with _model_lock:
        if _loaded_model is None:
            return

        if _is_generating:
            _inactivity_timer = threading.Timer(30.0, _check_and_auto_unload)
            _inactivity_timer.daemon = True
            _inactivity_timer.start()
            return

        idle_duration = time.time() - _last_used_timestamp
        if idle_duration >= (INACTIVITY_TIMEOUT_SECONDS - 1.0):
            logger.info(
                f"[LLM Engine] Inattività rilevata ({idle_duration:.1f}s >= {INACTIVITY_TIMEOUT_SECONDS}s). "
                f"Auto-unload del modello '{_loaded_model_id}' dalla RAM per azzerare il consumo."
            )
            del _loaded_model
            del _loaded_tokenizer
            _loaded_model = None
            _loaded_tokenizer = None
            _loaded_model_id = None
            _inactivity_timer = None
            clear_metal_cache()
            logger.info("[LLM Engine] Auto-unload completato con successo: memoria Metal liberata a macOS.")
        else:
            remaining = max(5.0, INACTIVITY_TIMEOUT_SECONDS - idle_duration)
            _inactivity_timer = threading.Timer(remaining, _check_and_auto_unload)
            _inactivity_timer.daemon = True
            _inactivity_timer.start()


def unload_whisper() -> None:
    """
    Libera Whisper Large dalla memoria unificata prima di caricare o eseguire
    il riassunto pesante della lezione con Qwen3.5-4B.
    """
    try:
        from transcriber import release_transcriber_memory
        release_transcriber_memory()
        clear_metal_cache()
        logger.info("[LLM Engine] Whisper scaricato dalla RAM con successo: memoria libera per LLM.")
    except Exception as exc:
        logger.warning(f"[LLM Engine] Errore durante l'unload di Whisper: {exc}")


def unload_llm() -> None:
    """Scarica il modello LLM attivo dalla RAM unificata."""
    global _loaded_model, _loaded_tokenizer, _loaded_model_id, _inactivity_timer
    with _model_lock:
        if _inactivity_timer is not None:
            _inactivity_timer.cancel()
            _inactivity_timer = None
        if _loaded_model is not None:
            del _loaded_model
            _loaded_model = None
        if _loaded_tokenizer is not None:
            del _loaded_tokenizer
            _loaded_tokenizer = None
        _loaded_model_id = None
        clear_metal_cache()
        logger.info("[LLM Engine] Modello LLM scaricato dalla RAM e cache Metal azzerata.")


def is_model_downloaded(model_id: str) -> bool:
    """Verifica se il modello HuggingFace è già presente nella cache locale."""
    try:
        from routers.models import check_repo_downloaded
        downloaded, _ = check_repo_downloaded(model_id)
        return downloaded
    except Exception:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        folder = "models--" + model_id.replace("/", "--")
        p = os.path.join(cache_dir, folder, "snapshots")
        return os.path.exists(p) and bool(os.listdir(p))


def get_or_load_model(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Carica il modello MLX-LM specificato (o il default Qwen3.5-4B) in memoria unificata.
    Se già in memoria, riutilizza l'istanza singleton.
    Ricarica on-demand dopo eventuale auto-unload e avvia il timer di inattività (180s).
    Lancia FileNotFoundError se il modello non è ancora stato scaricato dall'utente.
    """
    global _loaded_model, _loaded_tokenizer, _loaded_model_id

    target_id = model_id or DEFAULT_LLM_MODEL

    with _model_lock:
        if _loaded_model is not None and _loaded_model_id == target_id:
            _touch_model_usage()
            return _loaded_model, _loaded_tokenizer

        # Controllo presenza pesi su disco per evitare download automatici
        if not is_model_downloaded(target_id):
            raise FileNotFoundError(
                f"Il modello '{target_id}' non è ancora stato scaricato in locale. "
                "Vai in Impostazioni -> Modelli per scaricarlo con un clic."
            )

        # Su M1 8GB garantiamo sempre che Whisper non occupi la RAM prima di allocare l'LLM
        unload_whisper()

        # Se c'era un altro modello, rilascialo prima
        if _loaded_model is not None:
            del _loaded_model
            del _loaded_tokenizer
            _loaded_model = None
            _loaded_tokenizer = None
            clear_metal_cache()

        logger.info(f"[LLM Engine] Caricamento modello MLX su Apple Silicon Metal: {target_id}")
        import mlx_lm
        from transcriber import get_local_snapshot_dir

        local_path = get_local_snapshot_dir(target_id) or target_id
        model, tokenizer = mlx_lm.load(local_path)
        _loaded_model = model
        _loaded_tokenizer = tokenizer
        _loaded_model_id = target_id
        logger.info(f"[LLM Engine] Modello '{target_id}' caricato con successo in Unified Memory.")

    _touch_model_usage()
    return _loaded_model, _loaded_tokenizer


# ── Pulizia output & Disattivazione Thinking per Risposte Flash ─────────────────

def _clean_thinking_artifacts(text: str) -> str:
    """Rimuove eventuali tag <think>...</think> o sezioni 'Thinking Process:'."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
    cleaned = re.sub(r"^<think>[\s\S]*$", "", cleaned)
    cleaned = re.sub(r"(?i)^Thinking Process:[\s\S]*?(?=\n# |\n## |\n---|\Z)", "", cleaned)
    return cleaned.strip()


def _format_chat_prompt(tokenizer: Any, messages: list[dict], fallback_str: str) -> str:
    """
    Formatta il prompt per il modello MLX (es. Qwen 3.5 / Gemma 4),
    disattivando esplicitamente la modalità thinking per eliminare latenza,
    evitare surriscaldamento del Mac e generare subito l'output utile.
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


# ── 1. Risposta Flash alle Domande Docente (Live) ──────────────────────────────

async def generate_flash_answer(
    question: str,
    context: str,
    model_id: Optional[str] = None,
) -> str:
    """
    Genera una risposta flash in massimo 3 punti elenco chiari per la domanda rilevata.
    Modalità thinking disattivata per latenza minima su M1 8GB.
    """
    if not is_model_downloaded(model_id or DEFAULT_LLM_MODEL):
        return (
            "• Risposta non disponibile: modello LLM non ancora scaricato.\n"
            "• Scarica Qwen 3.5 4B dalle Impostazioni per attivare le risposte flash con MLX.\n"
            "• Domanda memorizzata per il ripasso e per il Quaderno."
        )

    system_prompt = (
        "Sei SbobinAI, assistente universitario locale ottimizzato per Apple Silicon. "
        "Rispondi alla domanda del professore in MODO ESTREMAMENTE CONCISO E IMMEDIATO.\n"
        "REGOLE FERREE:\n"
        "1. Rispondi con MASSIMO 3 punti elenco chiari ed essenziali (max 40 parole in totale).\n"
        "2. Nessun preambolo, nessun saluto, nessun tag di ragionamento interno.\n"
        "3. Basa la risposta sul contesto della spiegazione se pertinente."
    )

    user_prompt = f"Contesto spiegato a lezione:\n{context}\n\nDomanda del docente:\n{question}"

    def _run_inference() -> str:
        import mlx_lm
        model, tokenizer = get_or_load_model(model_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = _format_chat_prompt(
            tokenizer,
            messages,
            fallback_str=f"{system_prompt}\n\nDomanda:\n{user_prompt}\n\nRisposta Flash:"
        )

        full_text = ""
        # Ottimizzazione KV cache per M1 8GB: kv_bits=4, max_kv_size=8192
        for response in mlx_lm.stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_str,
            max_tokens=180,
            kv_bits=4,
            kv_group_size=64,
            max_kv_size=8192,
        ):
            if _cancel_generation_event.is_set():
                break
            full_text += response.text

        clear_metal_cache()
        _touch_model_usage()
        return _clean_thinking_artifacts(full_text)

    try:
        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(None, _run_inference)
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
    Usa la KV cache quantizzata a 4 bit per preservare la batteria dell'M1.
    """
    if not is_model_downloaded(model_id or DEFAULT_LLM_MODEL):
        yield (
            "Il modello LLM Qwen 3.5 4B non è ancora stato scaricato in locale.\n\n"
            "Apri le **Impostazioni** (rotellina in basso a sinistra) e premi **Scarica** "
            "per abilitare l'assistente IA 100% offline su Apple Silicon Metal."
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

    def _sync_stream():
        try:
            import mlx_lm
            model, tokenizer = get_or_load_model(model_id)

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
            for response in mlx_lm.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_str,
                max_tokens=600,
                kv_bits=4,
                kv_group_size=64,
                max_kv_size=8192,
            ):
                if _cancel_generation_event.is_set():
                    logger.warning("[LLM Chat] Streaming chat interrotto dall'utente.")
                    break

                token_text = response.text
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
                    loop.call_soon_threadsafe(queue.put_nowait, token_text)

        except Exception as exc:
            logger.error(f"[LLM Chat] Errore streaming chat: {exc}")
            loop.call_soon_threadsafe(queue.put_nowait, f"\n\n[Errore generazione: {exc}]")
        finally:
            clear_metal_cache()
            _touch_model_usage()
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Avvia il worker in thread pool
    threading.Thread(target=_sync_stream, daemon=True, name="llm-chat-stream").start()

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


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
    Prima dell'inferenza, scarica Whisper per lasciare tutti gli 8GB di RAM a Qwen3.5-4B.
    """
    if not is_model_downloaded(model_id or DEFAULT_LLM_MODEL):
        return (
            f"# {title} — Appunti di Lezione\n\n"
            "> [!NOTE]\n"
            "> Il modello IA non è ancora installato. Scarica **Qwen 3.5 4B** dalle Impostazioni "
            "per generare automaticamente lo schema concettuale strutturato con Apple Silicon Metal.\n\n"
            "## Trascrizione Grezza della Lezione\n\n"
            f"{transcript[:3000]}..."
        )

    global _is_generating, _current_task
    _cancel_generation_event.clear()
    _is_generating = True
    _current_task = f"Sintesi Quaderno: {title}"

    # 1. Unload preventivo di Whisper ed Embedding dalla memoria per lasciare tutti gli 8GB a Qwen
    unload_whisper()
    try:
        from rag import unload_embedding_model
        unload_embedding_model()
    except Exception:
        pass

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
        import mlx_lm
        model, tokenizer = get_or_load_model(model_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_str = _format_chat_prompt(
            tokenizer,
            messages,
            fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n# {title} — Capitolo di Studio\n"
        )

        full_chapter = ""
        # kv_bits=4 e kv_group_size=64 per elaborare contesti lunghi senza toccare la memoria swap
        for response in mlx_lm.stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_str,
            max_tokens=1200,
            kv_bits=4,
            kv_group_size=64,
        ):
            if _cancel_generation_event.is_set():
                logger.warning("[LLM Summary] Generazione interrotta su richiesta dell'utente.")
                raise InterruptedError("Generazione della sintesi interrotta dall'utente.")
            full_chapter += response.text

        clear_metal_cache()
        _touch_model_usage()
        return _clean_thinking_artifacts(full_chapter)

    try:
        loop = asyncio.get_running_loop()
        chapter = await loop.run_in_executor(None, _run_summary)
        return chapter.strip()
    except InterruptedError:
        logger.info("[LLM Summary] Operazione annullata dall'utente. Rilascio memoria.")
        clear_metal_cache()
        raise
    except Exception as exc:
        logger.error(f"[LLM Summary] Errore generazione capitolo: {exc}")
        clear_metal_cache()
        return f"# {title} — Capitolo di Studio\n\nErrore durante la generazione del capitolo: {exc}\n\n## Trascrizione\n{transcript}"
    finally:
        _is_generating = False
        _current_task = ""
        clear_metal_cache()


# Alias di compatibilità
explain_quick_doubt = generate_flash_answer

