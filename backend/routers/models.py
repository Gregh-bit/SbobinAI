"""
routers/models.py — SbobinAI Model Management API (platform-aware)

Gestisce monitoraggio, download tracciato con progress bar in tempo reale ed eliminazione
dei modelli IA locali del backend corrente (vedi ai_models.py):
  - macOS Apple Silicon: Whisper Large-v3-Turbo Q4 MLX, Qwen 3.5 4B / Gemma 4 E4B MLX 4-bit,
    Qwen3-Embedding-0.6B-4bit-DWQ MLX
  - Windows / Linux:     faster-whisper Large-v3-Turbo, Qwen 3.5 4B / Gemma 4 E4B GGUF Q4_K_M,
    Qwen3-Embedding-0.6B GGUF Q8_0
Gli ID logici dei modelli sono identici su tutte le piattaforme.
Le eliminazioni toccano SOLO i repo noti all'app (mai l'intera cache HuggingFace).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from tqdm.auto import tqdm

import ai_models
from platform_utils import AI_BACKEND, backend_label, dir_size_mb, hf_cache_dir, hf_repo_folder, safe_rmtree

logger = logging.getLogger("sbobinai.models")

router = APIRouter(prefix="/models", tags=["Models"])

# ── Configurazioni Modelli Supportati (backend corrente) ──────────────────────

def _infos(kind: str) -> Dict[str, Dict[str, Any]]:
    return {mid: ai_models.info_dict(spec) for mid, spec in ai_models.CATALOG.items() if spec.type == kind}


WHISPER_MODELS: Dict[str, Dict[str, Any]] = {**_infos("whisper"), ai_models.MOCK_WHISPER_ID: dict(ai_models.MOCK_WHISPER_INFO)}
LLM_MODELS: Dict[str, Dict[str, Any]] = _infos("llm")
EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = _infos("embedding")

# ── Stato Download in Memoria ──────────────────────────────────────────────────
_download_tracker: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_cancel_events: Dict[str, threading.Event] = {}
_download_threads: Dict[str, threading.Thread] = {}


def get_hf_cache_dir() -> str:
    """Restituisce la cartella di cache dei modelli HuggingFace."""
    return hf_cache_dir()


def _get_dir_size_mb(path: str) -> float:
    return dir_size_mb(path)


def check_repo_downloaded(repo_id: str) -> tuple[bool, float]:
    """
    Verifica se un modello (ID logico, repo id di qualsiasi backend o alias) è scaricato
    COMPLETAMENTE (config + tutti i file dei pesi) per il backend corrente.
    Un modello con download in corso non risulta scaricato.
    """
    if not repo_id or repo_id == "mock":
        return False, 0.0
    spec = ai_models.resolve(repo_id)
    if spec is not None:
        return ai_models.is_downloaded(spec)
    # Repo sconosciuto al catalogo: snapshot presente con almeno un file di pesi
    from platform_utils import get_local_snapshot_dir, snapshot_is_complete
    snap = get_local_snapshot_dir(repo_id)
    if snap and any(
        snapshot_is_complete(snap, (p,)) for p in ("*.safetensors", "*.gguf", "*.npz", "model.bin")
    ):
        return True, dir_size_mb(hf_repo_folder(repo_id))
    return False, 0.0


def is_whisper_downloaded(model_info: Dict[str, Any]) -> tuple[bool, float]:
    """Verifica se il modello Whisper del backend corrente è presente nella cache HF."""
    if model_info.get("id") == ai_models.MOCK_WHISPER_ID:
        return True, 0.0
    return ai_models.is_downloaded(ai_models.CATALOG[ai_models.WHISPER_CORE_ID])


# ── Background Worker con Tracking Percentuale ────────────────────────────────

def _download_hf_worker(model_id: str, spec: ai_models.ModelSpec, cancel_event: threading.Event):
    variant = spec.variant
    repo_id = variant.repo_id
    allow = list(variant.allow_patterns) if variant.allow_patterns else None
    logger.info(f"[Models] Avvio download HuggingFace ({model_id}): {repo_id} allow={allow}")

    def _check_cancel():
        if cancel_event.is_set():
            raise InterruptedError("Download annullato su richiesta dell'utente.")

    total_bytes = 0
    try:
        from huggingface_hub import HfApi
        import fnmatch
        _check_cancel()
        info = HfApi().model_info(repo_id, files_metadata=True)
        _check_cancel()
        for f in info.siblings or []:
            if f.size and (allow is None or any(fnmatch.fnmatch(f.rfilename, p) for p in allow)):
                total_bytes += f.size
    except InterruptedError:
        _finish_cancelled(model_id, spec, 0.0)
        return
    except Exception:
        pass
    if total_bytes <= 0:
        total_bytes = variant.approx_bytes or (580 * 1024 * 1024)
    total_mb = round(total_bytes / (1024 * 1024), 1)

    with _lock:
        _download_tracker[model_id] = {
            "status": "downloading",
            "progress": 2,
            "downloaded_bytes": 0,
            "downloaded_mb": 0.0,
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "stage": "Connessione ai server HuggingFace...",
            "error": None,
        }

    class DualProgressTracker(tqdm):
        def update(self, n=1):
            _check_cancel()
            super().update(n)
            # Solo le barre in byte (non il contatore "Fetching N files")
            if getattr(self, "unit", "") != "B" or not n:
                return
            with _lock:
                tracker = _download_tracker.get(model_id)
                if tracker is not None:
                    tracker["downloaded_bytes"] = tracker.get("downloaded_bytes", 0) + n
                    tracker["downloaded_mb"] = round(tracker["downloaded_bytes"] / (1024 * 1024), 1)
                    t_bytes = tracker.get("total_bytes") or total_bytes
                    if t_bytes > 0:
                        pct = min(99, int((tracker["downloaded_bytes"] / t_bytes) * 100))
                        tracker["progress"] = max(tracker["progress"], pct)

    try:
        from huggingface_hub import snapshot_download

        _check_cancel()
        with _lock:
            if model_id in _download_tracker:
                _download_tracker[model_id]["stage"] = f"Download {repo_id}..."
        snapshot_download(repo_id=repo_id, allow_patterns=allow, tqdm_class=DualProgressTracker)
        _check_cancel()

        ai_models.set_downloading(model_id, False)
        if ai_models.local_snapshot(spec) is None:
            raise RuntimeError("Download terminato ma alcuni file del modello risultano mancanti.")

        with _lock:
            _download_tracker[model_id] = {
                "status": "completed",
                "progress": 100,
                "downloaded_bytes": total_bytes,
                "downloaded_mb": total_mb,
                "total_bytes": total_bytes,
                "total_mb": total_mb,
                "stage": "Completato con successo!",
                "error": None,
            }
        logger.info(f"[Models] Download modello completato al 100%: {model_id}")

        # Aggiorna settings.json
        try:
            from config import load_settings, save_settings
            s = load_settings()
            s["models_downloaded"] = True
            if spec.type == "whisper":
                s["whisper_model"] = model_id
            elif spec.type == "llm":
                s["llm_model"] = model_id
            elif spec.type == "embedding":
                s["embedding_model"] = model_id
            save_settings(s)
        except Exception:
            pass

    except InterruptedError:
        _finish_cancelled(model_id, spec, total_mb)

    except Exception as exc:
        logger.error(f"[Models] Errore durante il download di {model_id}: {exc}")
        with _lock:
            _download_tracker[model_id] = {
                "status": "error",
                "progress": 0,
                "downloaded_mb": 0.0,
                "total_mb": total_mb,
                "stage": "Errore",
                "error": str(exc),
            }
    finally:
        ai_models.set_downloading(model_id, False)
        with _lock:
            if _cancel_events.get(model_id) is cancel_event:
                _cancel_events.pop(model_id, None)
            _download_threads.pop(model_id, None)


def _finish_cancelled(model_id: str, spec: ai_models.ModelSpec, total_mb: float) -> None:
    """Gestione interruzione: unico punto in cui vengono eliminati i file parziali."""
    logger.info(f"[Models] Download di {model_id} interrotto: rimozione file parziali.")
    ai_models.set_downloading(model_id, False)
    try:
        _delete_repos([spec.repo_id])
    except Exception as exc:
        logger.warning(f"[Models] Pulizia file parziali {model_id}: {exc}")
    with _lock:
        _download_tracker[model_id] = {
            "status": "cancelled",
            "progress": 0,
            "downloaded_mb": 0.0,
            "total_mb": total_mb,
            "stage": "Download interrotto dall'utente.",
            "error": "Download interrotto dall'utente.",
        }


# ── Endpoints API ─────────────────────────────────────────────────────────────

def _status_item(model_id: str, info: Dict[str, Any], is_dl: bool, size_mb: float) -> Dict[str, Any]:
    approx_mb = round(info.get("approx_bytes", 0) / (1024 * 1024), 1)
    tracker = _download_tracker.get(model_id, {
        "status": "idle",
        "progress": 100 if is_dl else 0,
        "downloaded_mb": size_mb if is_dl else 0.0,
        "total_mb": size_mb if is_dl else approx_mb,
        "stage": None,
        "error": None,
    })
    current_status = tracker["status"]
    if is_dl and current_status != "downloading":
        current_status = "completed"
        progress = 100
    else:
        progress = tracker.get("progress", 0)
    return {
        **info,
        "downloaded": is_dl,
        "disk_size_mb": size_mb,
        "status": current_status,
        "progress": progress,
        "downloaded_mb": tracker.get("downloaded_mb", size_mb if is_dl else 0.0),
        "total_mb": tracker.get("total_mb", approx_mb),
        "stage": tracker.get("stage"),
        "error": tracker.get("error"),
    }


@router.get("/status")
def get_models_status():
    """
    Restituisce lo stato dettagliato di ciascun modello (Whisper, LLM, Embeddings) del backend corrente:
    se è scaricato, la dimensione sul disco, la percentuale e i byte scaricati in tempo reale.
    """
    whisper_list: List[Dict[str, Any]] = []
    llm_list: List[Dict[str, Any]] = []
    embedding_list: List[Dict[str, Any]] = []
    installed_summary: List[Dict[str, Any]] = []

    for model_id, info in WHISPER_MODELS.items():
        is_dl, size_mb = is_whisper_downloaded(info)
        whisper_list.append(_status_item(model_id, info, is_dl, size_mb))
        if is_dl and model_id != ai_models.MOCK_WHISPER_ID:
            installed_summary.append({"id": model_id, "name": info["name"], "type": "whisper", "size_mb": size_mb})

    for target, kind, models in ((llm_list, "llm", LLM_MODELS), (embedding_list, "embedding", EMBEDDING_MODELS)):
        for model_id, info in models.items():
            is_dl, size_mb = ai_models.is_downloaded(ai_models.CATALOG[model_id])
            target.append(_status_item(model_id, info, is_dl, size_mb))
            if is_dl:
                installed_summary.append({"id": model_id, "name": info["name"], "type": kind, "size_mb": size_mb})

    total_installed_mb = sum(m["size_mb"] for m in installed_summary)

    return {
        "whisper": whisper_list,
        "llm": llm_list,
        "embedding": embedding_list,
        # Retrocompatibilità per il frontend finché usa il campo ollama
        "ollama": llm_list,
        "ollama_running": True,  # motore locale integrato, sempre disponibile
        "installed_count": len(installed_summary),
        "total_installed_mb": round(total_installed_mb, 1),
        "installed_models": installed_summary,
        "cache_path": get_hf_cache_dir(),
        "backend": AI_BACKEND,
        "backend_label": backend_label(),
    }


class DownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


@router.post("/download")
def download_model(payload: DownloadRequest):
    """
    Avvia il download in background del modello specificato con monitoraggio dei byte.
    Accetta ID logici, repo id (MLX o portable) e alias storici.
    """
    requested = payload.model_id.strip()
    if requested == ai_models.MOCK_WHISPER_ID:
        return {"status": "completed", "message": "Modalità simulatore sempre disponibile (0 MB)."}

    spec = ai_models.resolve(requested)
    if spec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modello '{requested}' non riconosciuto tra i modelli supportati."
        )
    model_id = spec.id
    repo_id = spec.repo_id

    with _lock:
        thread = _download_threads.get(model_id)
        if thread is not None and thread.is_alive():
            current = _download_tracker.get(model_id, {})
            return {
                "status": "already_downloading",
                "model_id": model_id,
                "progress": current.get("progress", 0),
            }

        approx_b = spec.variant.approx_bytes or (580 * 1024 * 1024)
        _download_tracker[model_id] = {
            "status": "downloading",
            "progress": 1,
            "downloaded_bytes": 0,
            "downloaded_mb": 0.0,
            "total_bytes": approx_b,
            "total_mb": round(approx_b / (1024 * 1024), 1),
            "stage": "Avvio download e connessione...",
            "error": None,
        }
        cancel_event = threading.Event()
        _cancel_events[model_id] = cancel_event
        ai_models.set_downloading(model_id, True)
        thread = threading.Thread(
            target=_download_hf_worker,
            args=(model_id, spec, cancel_event),
            daemon=True,
            name=f"hf-download-{model_id}",
        )
        _download_threads[model_id] = thread
    thread.start()
    return {
        "status": "started",
        "model_id": model_id,
        "repos": [repo_id],
        "progress": 1,
    }


# ── INTERROMPI DOWNLOAD ATTIVO ────────────────────────────────────────────────

class CancelDownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: Optional[str] = None


@router.post("/cancel-download")
def cancel_download(payload: Optional[CancelDownloadRequest] = None):
    """
    Interrompe il download in corso del modello indicato (model_id obbligatorio; "all" = tutti i download
    in corso). I file parziali vengono rimossi dal worker stesso al termine dell'interruzione.
    """
    target = (payload.model_id or "").strip() if payload else ""
    if not target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Specificare 'model_id' del download da interrompere (oppure 'all').",
        )

    with _lock:
        if target == "all":
            ids = list(_cancel_events.keys())
        else:
            spec = ai_models.resolve(target)
            ids = [spec.id if spec else target]
        events = [(mid, _cancel_events.get(mid), _download_threads.get(mid)) for mid in ids]

    cancelled: List[str] = []
    for mid, ev, th in events:
        if ev is None:
            continue
        logger.info(f"[Models] Segnale di interruzione download ricevuto per {mid}.")
        ev.set()
        cancelled.append(mid)
        if th is not None and th.is_alive():
            th.join(timeout=5.0)

    return {
        "status": "cancelled" if cancelled else "not_downloading",
        "cancelled": cancelled,
        "message": "Download interrotto e file parziali rimossi." if cancelled else "Nessun download in corso per il modello indicato.",
        "freed_mb": 0.0,
    }


# ── ELIMINAZIONE ───────────────────────────────────────────────────────────────

def _release_engines() -> None:
    """Scarica i modelli dalla RAM prima di eliminare i file (su Windows i file mappati sono bloccati)."""
    try:
        from transcriber import release_transcriber_memory
        release_transcriber_memory()
    except Exception:
        pass
    try:
        from llm_engine import unload_llm
        unload_llm()
    except Exception:
        pass
    try:
        from rag import unload_embedding_model
        unload_embedding_model()
    except Exception:
        pass


def _delete_repos(repo_ids: List[str]) -> float:
    """Elimina dalla cache HF solo i repo indicati. Restituisce i MB liberati."""
    freed_mb = 0.0
    for repo_id in repo_ids:
        model_dir = hf_repo_folder(repo_id)
        if not os.path.exists(model_dir):
            continue
        size_mb = dir_size_mb(model_dir)
        if safe_rmtree(model_dir):
            freed_mb += size_mb
            logger.info(f"[Models] Cartella eliminata: {model_dir} (liberati {size_mb} MB)")
        else:
            logger.error(f"[Models] Impossibile eliminare {model_dir} (file in uso?).")
    # Lock residui di huggingface_hub per questi repo
    locks_root = os.path.join(hf_cache_dir(), ".locks")
    for repo_id in repo_ids:
        safe_rmtree(os.path.join(locks_root, "models--" + repo_id.replace("/", "--")), retries=1)
    return round(freed_mb, 1)


@router.delete("/all/delete-all")
def delete_all_models():
    """
    Elimina dal disco TUTTI i modelli dell'app (entrambi i backend), senza toccare
    modelli di altre applicazioni presenti nella cache HuggingFace.
    """
    with _lock:
        for ev in _cancel_events.values():
            ev.set()
    _release_engines()

    repos = ai_models.all_known_repo_ids()
    deleted_items = [r for r in repos if os.path.exists(hf_repo_folder(r))]
    total_freed_mb = _delete_repos(repos)

    with _lock:
        _download_tracker.clear()

    try:
        from config import load_settings, save_settings
        s = load_settings()
        s["models_downloaded"] = False
        s["whisper_model"] = ai_models.MOCK_WHISPER_ID
        save_settings(s)
    except Exception:
        pass

    logger.info(f"[Models] Eliminazione modelli SbobinAI completata. Liberati {total_freed_mb:.1f} MB.")
    return {
        "status": "success",
        "freed_mb": total_freed_mb,
        "deleted_count": len(deleted_items),
        "deleted_items": deleted_items,
        "message": f"Tutti i modelli sono stati eliminati con successo dal disco ({total_freed_mb} MB liberati).",
    }


@router.delete("/{model_name:path}")
def delete_single_model(model_name: str):
    """
    Elimina i file del singolo modello specificato (entrambe le varianti di backend)
    e scarica la memoria corrispondente.
    """
    clean_name = model_name.strip()
    if clean_name in ("all/delete-all", "all"):
        return delete_all_models()

    spec = ai_models.resolve(clean_name)
    if spec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Modello '{clean_name}' non riconosciuto tra i modelli dell'app.",
        )

    with _lock:
        ev = _cancel_events.get(spec.id)
    if ev is not None:
        ev.set()

    _release_engines()
    freed_mb = _delete_repos(spec.all_repo_ids())

    with _lock:
        _download_tracker.pop(spec.id, None)
        _download_tracker.pop(clean_name, None)

    return {
        "status": "deleted",
        "model": clean_name,
        "freed_mb": freed_mb,
        "message": f"Modello {clean_name} rimosso con successo dal disco. Liberati {freed_mb} MB e memoria RAM scaricata.",
    }
