"""
routers/models.py — SbobinAI Pure MLX Model Management API
Gestisce il monitoraggio, il download tracciato con progress bar in tempo reale
e l'eliminazione dei modelli IA locali nativi Apple Silicon Metal:
  - Whisper Large-v3-Turbo Q4 (Trascrizione)
  - Qwen 3.5 4B / Gemma 4 E4B 4-bit (LLM locale)
  - Qwen3-Embedding-0.6B-4bit-DWQ (RAG Embeddings)
Zero dipendenze PyTorch o demoni Ollama esterni.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from tqdm.auto import tqdm

logger = logging.getLogger("sbobinai.models")

router = APIRouter(prefix="/models", tags=["Models"])

# ── Configurazioni Modelli Supportati (100% MLX Apple Silicon 2026) ────────────

WHISPER_MODELS: Dict[str, Dict[str, Any]] = {
    "large-v3-turbo-q4": {
        "id": "large-v3-turbo-q4",
        "name": "Whisper Large-v3-Turbo Q4 (Apple Silicon)",
        "repos": [
            "mlx-community/whisper-large-v3-turbo-q4",
        ],
        "repo_id": "mlx-community/whisper-large-v3-turbo-q4",
        "type": "whisper",
        "is_system_core": True,
        "approx_size": "~580 MB",
        "approx_bytes": 580 * 1024 * 1024,
        "description": "Motore Vocale di sistema obbligatorio: trascrizione ad alta precisione 4-bit per Apple Silicon Metal (~580 MB). Fedeltà massima, zero allucinazioni e consumo nullo in pausa.",
    },
    "mock_mode": {
        "id": "mock_mode",
        "name": "Modalità Simulatore (Mock)",
        "repos": [],
        "repo_id": "mock",
        "type": "whisper",
        "approx_size": "0 MB",
        "approx_bytes": 0,
        "description": "Nessun download, trascrizione istantanea simulata per test rapido.",
    },
}

LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "mlx-community/Qwen3.5-4B-MLX-4bit": {
        "id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "name": "Qwen 3.5 4B (4-bit MLX Metal)",
        "repos": [
            "mlx-community/Qwen3.5-4B-MLX-4bit",
        ],
        "repo_id": "mlx-community/Qwen3.5-4B-MLX-4bit",
        "type": "llm",
        "is_default": True,
        "approx_size": "~2.4 GB",
        "approx_bytes": 2400 * 1024 * 1024,
        "description": "LLM Primario (Consigliato 2026): Architettura Gated DeltaNet 2026, KV cache quantizzata 4-bit, ideale per MacBook Air M1 8GB. Risposte flash istantanee live e capitoli Quaderno.",
    },
    "unsloth/gemma-4-E4B-it-UD-MLX-4bit": {
        "id": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        "name": "Gemma 4 E4B (4-bit MLX Metal)",
        "repos": [
            "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        ],
        "repo_id": "unsloth/gemma-4-E4B-it-UD-MLX-4bit",
        "type": "llm",
        "is_default": False,
        "approx_size": "~2.5 GB",
        "approx_bytes": 2500 * 1024 * 1024,
        "description": "LLM Alternativo selezionabile basato su Gemma 4 E4B 4-bit per sintesi e spiegazioni didattiche.",
    },
}

EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": {
        "id": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "name": "Qwen3 Embedding 0.6B 4-bit DWQ (MLX)",
        "repos": [
            "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        ],
        "repo_id": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "type": "embedding",
        "is_default": True,
        "is_system_core": True,
        "approx_size": "~300 MB",
        "approx_bytes": 300 * 1024 * 1024,
        "description": "Motore Memoria RAG di sistema obbligatorio: architettura Qwen3 nativa MLX, 1024 dimensioni (~300 MB in RAM). Indicizzazione ad altissima velocità e ricerca ibrida su LanceDB.",
    },
}

# ── Stato Download in Memoria ──────────────────────────────────────────────────
_download_tracker: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_cancel_requested = threading.Event()
_active_thread: Optional[threading.Thread] = None


def get_hf_cache_dir() -> str:
    """Restituisce la cartella di cache dei modelli HuggingFace."""
    try:
        from huggingface_hub import constants
        return str(constants.HF_HUB_CACHE)
    except Exception:
        return os.path.expanduser("~/.cache/huggingface/hub")


def _get_dir_size_mb(path: str) -> float:
    """Calcola la dimensione totale in MB di una directory, risolvendo i symlink ai blob HF reali."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    seen_inodes = set()
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    real_fp = os.path.realpath(fp)
                    if os.path.exists(real_fp):
                        st = os.stat(real_fp)
                        key = (st.st_dev, st.st_ino)
                        if key not in seen_inodes:
                            seen_inodes.add(key)
                            total += st.st_size
                except Exception:
                    pass
    except Exception:
        pass
    return round(total / (1024 * 1024), 1)


def check_repo_downloaded(repo_id: str) -> tuple[bool, float]:
    """Verifica se un singolo repo HuggingFace è scaricato in locale con pesi reali."""
    if not repo_id or repo_id == "mock":
        return False, 0.0

    # Metodo 1: scan_cache_dir di huggingface_hub
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            clean_cached = repo.repo_id.replace("-mlx", "")
            clean_target = repo_id.replace("-mlx", "")
            if repo.repo_id == repo_id or clean_cached == clean_target:
                size_mb = round(repo.size_on_disk / (1024 * 1024), 1)
                if size_mb > 2.0:
                    return True, size_mb
    except Exception as e:
        logger.debug(f"[Models] Fallback scansione filesystem cache HF: {e}")

    # Metodo 2: Ispezione percorsi su filesystem
    cache_root = get_hf_cache_dir()
    possible_names = [
        "models--" + repo_id.replace("/", "--"),
    ]
    if repo_id.endswith("-mlx"):
        possible_names.append("models--" + repo_id[:-4].replace("/", "--"))
    else:
        possible_names.append("models--" + repo_id.replace("/", "--") + "-mlx")

    for hf_folder_name in possible_names:
        model_dir = os.path.join(cache_root, hf_folder_name)
        if os.path.exists(model_dir):
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
                size_mb = _get_dir_size_mb(model_dir)
                if size_mb > 2.0:
                    return True, size_mb

    return False, 0.0


def is_whisper_downloaded(model_info: Dict[str, Any]) -> tuple[bool, float]:
    """Verifica se il modello Whisper Large-v3-Turbo Q4 è presente nella cache HF."""
    if model_info.get("id") == "mock_mode":
        return True, 0.0
    return check_repo_downloaded("mlx-community/whisper-large-v3-turbo-q4")


# ── Background Worker con Tracking Percentuale Sequenziale ───────────────────

def _download_hf_worker(model_id: str, repos: List[str], expected_bytes: int):
    logger.info(f"[Models] Avvio download sequenziale HuggingFace ({model_id}): {repos}")

    from huggingface_hub import HfApi
    api = HfApi()
    total_bytes = 0
    for repo_id in repos:
        r_bytes = 0
        try:
            info = api.model_info(repo_id, files_metadata=True)
            if info.siblings:
                r_bytes = sum(f.size for f in info.siblings if f.size)
        except Exception:
            pass
        if r_bytes <= 0:
            r_bytes = expected_bytes or (580 * 1024 * 1024)
        total_bytes += r_bytes

    if total_bytes <= 0:
        total_bytes = expected_bytes or (580 * 1024 * 1024)

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
            if _cancel_requested.is_set():
                raise InterruptedError("Download annullato su richiesta dell'utente.")
            super().update(n)
            with _lock:
                if model_id in _download_tracker:
                    tracker = _download_tracker[model_id]
                    tracker["downloaded_bytes"] += n
                    d_mb = round(tracker["downloaded_bytes"] / (1024 * 1024), 1)
                    tracker["downloaded_mb"] = d_mb
                    t_bytes = tracker.get("total_bytes") or total_bytes
                    if t_bytes > 0:
                        pct = min(99, int((tracker["downloaded_bytes"] / t_bytes) * 100))
                        tracker["progress"] = max(tracker["progress"], pct)

    try:
        from huggingface_hub import snapshot_download

        for idx, repo_id in enumerate(repos):
            if _cancel_requested.is_set():
                raise InterruptedError("Download annullato su richiesta dell'utente.")

            stage_desc = f"Download {repo_id}..."
            with _lock:
                if model_id in _download_tracker:
                    _download_tracker[model_id]["stage"] = stage_desc

            logger.info(f"[Models] Snapshot download repo ({idx+1}/{len(repos)}): {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                tqdm_class=DualProgressTracker,
            )

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
            if model_id in WHISPER_MODELS:
                s["whisper_model"] = model_id
            elif model_id in LLM_MODELS:
                s["llm_model"] = model_id
            elif model_id in EMBEDDING_MODELS:
                s["embedding_model"] = model_id
            save_settings(s)
        except Exception:
            pass

    except InterruptedError:
        logger.info(f"[Models] Download di {model_id} interrotto con successo.")
        with _lock:
            _download_tracker[model_id] = {
                "status": "cancelled",
                "progress": 0,
                "downloaded_mb": 0.0,
                "total_mb": total_mb,
                "stage": "Download interrotto dall'utente.",
                "error": "Download interrotto dall'utente.",
            }
        try:
            delete_single_model(model_id)
        except Exception:
            pass

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


# ── Endpoints API ─────────────────────────────────────────────────────────────

@router.get("/status")
def get_models_status():
    """
    Restituisce lo stato dettagliato di ciascun modello nativo MLX (Whisper, LLM, Embeddings):
    se è scaricato, la dimensione sul disco, la percentuale e i byte scaricati in tempo reale.
    """
    whisper_list = []
    llm_list = []
    embedding_list = []
    installed_summary = []

    # 1. Modelli Whisper
    for model_id, info in WHISPER_MODELS.items():
        is_dl, size_mb = is_whisper_downloaded(info)
        tracker = _download_tracker.get(model_id, {
            "status": "idle",
            "progress": 100 if is_dl else 0,
            "downloaded_mb": size_mb if is_dl else 0.0,
            "total_mb": size_mb if is_dl else round(info.get("approx_bytes", 0) / (1024 * 1024), 1),
            "error": None,
        })
        current_status = tracker["status"]
        if is_dl and current_status != "downloading":
            current_status = "completed"
            progress = 100
        else:
            progress = tracker.get("progress", 0)

        item = {
            **info,
            "downloaded": is_dl,
            "disk_size_mb": size_mb,
            "status": current_status,
            "progress": progress,
            "downloaded_mb": tracker.get("downloaded_mb", size_mb if is_dl else 0.0),
            "total_mb": tracker.get("total_mb", round(info.get("approx_bytes", 0) / (1024 * 1024), 1)),
            "stage": tracker.get("stage"),
            "error": tracker.get("error"),
        }
        whisper_list.append(item)
        if is_dl and model_id != "mock_mode":
            installed_summary.append({
                "id": model_id,
                "name": info["name"],
                "type": "whisper",
                "size_mb": size_mb,
            })

    # 2. Modelli LLM (Qwen 3.5 4B / Gemma 4 E4B)
    for model_id, info in LLM_MODELS.items():
        is_dl, size_mb = check_repo_downloaded(info["repo_id"])
        tracker = _download_tracker.get(model_id, {
            "status": "idle",
            "progress": 100 if is_dl else 0,
            "downloaded_mb": size_mb if is_dl else 0.0,
            "total_mb": size_mb if is_dl else round(info.get("approx_bytes", 0) / (1024 * 1024), 1),
            "stage": None,
            "error": None,
        })
        current_status = tracker["status"]
        if is_dl and current_status != "downloading":
            current_status = "completed"
            progress = 100
        else:
            progress = tracker.get("progress", 0)

        item = {
            **info,
            "downloaded": is_dl,
            "disk_size_mb": size_mb,
            "status": current_status,
            "progress": progress,
            "downloaded_mb": tracker.get("downloaded_mb", size_mb if is_dl else 0.0),
            "total_mb": tracker.get("total_mb", round(info.get("approx_bytes", 0) / (1024 * 1024), 1)),
            "stage": tracker.get("stage"),
            "error": tracker.get("error"),
        }
        llm_list.append(item)
        if is_dl:
            installed_summary.append({
                "id": model_id,
                "name": info["name"],
                "type": "llm",
                "size_mb": size_mb,
            })

    # 3. Modelli Embedding (BGE Small / Multilingual E5)
    for model_id, info in EMBEDDING_MODELS.items():
        is_dl, size_mb = check_repo_downloaded(info["repo_id"])
        tracker = _download_tracker.get(model_id, {
            "status": "idle",
            "progress": 100 if is_dl else 0,
            "downloaded_mb": size_mb if is_dl else 0.0,
            "total_mb": size_mb if is_dl else round(info.get("approx_bytes", 0) / (1024 * 1024), 1),
            "stage": None,
            "error": None,
        })
        current_status = tracker["status"]
        if is_dl and current_status != "downloading":
            current_status = "completed"
            progress = 100
        else:
            progress = tracker.get("progress", 0)

        item = {
            **info,
            "downloaded": is_dl,
            "disk_size_mb": size_mb,
            "status": current_status,
            "progress": progress,
            "downloaded_mb": tracker.get("downloaded_mb", size_mb if is_dl else 0.0),
            "total_mb": tracker.get("total_mb", round(info.get("approx_bytes", 0) / (1024 * 1024), 1)),
            "stage": tracker.get("stage"),
            "error": tracker.get("error"),
        }
        embedding_list.append(item)
        if is_dl:
            installed_summary.append({
                "id": model_id,
                "name": info["name"],
                "type": "embedding",
                "size_mb": size_mb,
            })

    total_installed_mb = sum(m["size_mb"] for m in installed_summary)

    return {
        "whisper": whisper_list,
        "llm": llm_list,
        "embedding": embedding_list,
        # Retrocompatibilità per il frontend finché usa il campo ollama
        "ollama": llm_list,
        "ollama_running": True,  # MLX Metal è sempre attivo nativamente su macOS
        "installed_count": len(installed_summary),
        "total_installed_mb": round(total_installed_mb, 1),
        "installed_models": installed_summary,
        "cache_path": get_hf_cache_dir(),
    }


class DownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


@router.post("/download")
def download_model(payload: DownloadRequest):
    """
    Avvia il download in background del modello specificato con monitoraggio dei byte.
    Supporta modelli Whisper, LLM MLX (Qwen 3.5 4B) ed Embedding MLX (BGE / E5).
    """
    model_id = payload.model_id.strip()

    # Mappatura alias
    if model_id in ("dual-tier-whisper", "large-v3-turbo-q4", "large-v3-turbo", "whisper") or (
        "whisper" in model_id.lower() or "turbo" in model_id.lower()
    ):
        model_id = "large-v3-turbo-q4"
    elif model_id in (
        "qwen3-embedding",
        "embedding",
        "mlx-community/bge-small-en-v1.5-mlx",
        "mlx-community/multilingual-e5-small-mlx",
    ) or ("qwen3" in model_id.lower() and "embed" in model_id.lower()):
        model_id = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    # Cerca tra i modelli Whisper
    target_info = WHISPER_MODELS.get(model_id)

    # Cerca tra i modelli LLM
    if not target_info:
        target_info = LLM_MODELS.get(model_id)
        if not target_info:
            for k, v in LLM_MODELS.items():
                if v.get("repo_id") == model_id or model_id in v.get("repos", []):
                    target_info = v
                    model_id = k
                    break

    # Cerca tra i modelli Embedding
    if not target_info:
        target_info = EMBEDDING_MODELS.get(model_id)
        if not target_info:
            for k, v in EMBEDDING_MODELS.items():
                if v.get("repo_id") == model_id or model_id in v.get("repos", []):
                    target_info = v
                    model_id = k
                    break

    if target_info:
        if target_info.get("id") == "mock_mode":
            return {"status": "completed", "message": "Modalità simulatore sempre disponibile (0 MB)."}

        with _lock:
            current = _download_tracker.get(model_id, {})
            if current.get("status") == "downloading":
                return {
                    "status": "already_downloading",
                    "model_id": model_id,
                    "progress": current.get("progress", 0),
                }

            approx_b = target_info.get("approx_bytes", 0) or (580 * 1024 * 1024)
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

        _cancel_requested.clear()
        global _active_thread
        repos_to_download = target_info.get("repos") or [target_info["repo_id"]]
        _active_thread = threading.Thread(
            target=_download_hf_worker,
            args=(model_id, repos_to_download, target_info.get("approx_bytes", 0)),
            daemon=True,
        )
        _active_thread.start()
        return {
            "status": "started",
            "model_id": model_id,
            "repos": repos_to_download,
            "progress": 1,
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Modello '{model_id}' non riconosciuto tra i modelli MLX supportati."
    )


# ── INTERROMPI DOWNLOAD ATTIVO & PULISCI DISCO ─────────────────────────────────

class CancelDownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: Optional[str] = None


@router.post("/cancel-download")
def cancel_download(payload: Optional[CancelDownloadRequest] = None):
    """
    Interrompe immediatamente qualsiasi download in corso,
    attende che il worker termini in sicurezza,
    cancella i file parziali scaricati dal disco e reimposta lo stato.
    """
    global _active_thread
    logger.info("[Models] Segnale di interruzione download ricevuto.")
    _cancel_requested.set()

    if _active_thread and _active_thread.is_alive():
        _active_thread.join(timeout=2.0)
        _active_thread = None

    import time
    time.sleep(0.3)
    target = payload.model_id.strip() if payload and payload.model_id else None
    if target and target != "all":
        res = delete_single_model(target)
        freed = res.get("freed_mb", 0.0)
    else:
        res = delete_all_models()
        freed = res.get("freed_mb", 0.0)

    _cancel_requested.clear()
    with _lock:
        if target and target != "all":
            _download_tracker.pop(target, None)
        else:
            _download_tracker.clear()
    return {
        "status": "cancelled",
        "message": "Download interrotto e file rimossi con successo.",
        "freed_mb": freed,
    }


# ── ELIMINA TUTTI I MODELLI (WHISPER + LLM + EMBEDDINGS) ──────────────────────

@router.delete("/all/delete-all")
def delete_all_models():
    """
    Elimina fisicamente TUTTI i modelli MLX (Whisper, LLM, Embeddings) scaricati su disco,
    libera completamente la cache locale e azzera la RAM.
    """
    total_freed_mb = 0.0
    deleted_items: List[str] = []

    # 1. Elimina revisioni tramite scan_cache_dir
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        all_revs = []
        for repo in info.repos:
            all_revs.extend([rev.commit_hash for rev in repo.revisions])
        if all_revs:
            del_strat = info.delete_revisions(*all_revs)
            freed_from_hf = round(del_strat.expected_freed_size / (1024 * 1024), 1)
            del_strat.execute()
            total_freed_mb += freed_from_hf
            deleted_items.append(f"Cache HF liberata: {freed_from_hf} MB")
            logger.info(f"[Models] scan_cache_dir ha eliminato tutte le revisioni HF ({freed_from_hf} MB).")
    except Exception as e:
        logger.warning(f"[Models] Errore eliminazione HF scan_cache_dir: {e}")

    # 2. Elimina qualsiasi cartella rimasta nella cache HuggingFace
    cache_root = get_hf_cache_dir()
    if os.path.exists(cache_root):
        try:
            for item in os.listdir(cache_root):
                if item.startswith("models--") or item.startswith(".tmp") or item.startswith("tmp") or item in ("blobs", ".locks"):
                    item_path = os.path.join(cache_root, item)
                    if os.path.isdir(item_path):
                        size_mb = _get_dir_size_mb(item_path)
                        try:
                            shutil.rmtree(item_path)
                            total_freed_mb += size_mb
                            deleted_items.append(f"Cartella rimossa ({item}): {size_mb} MB")
                            logger.info(f"[Models] Cartella rimossa: {item_path} ({size_mb} MB)")
                        except Exception as e:
                            logger.error(f"[Models] Errore eliminazione {item}: {e}")
                    elif os.path.isfile(item_path):
                        try:
                            os.remove(item_path)
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"[Models] Errore ispezione cache HF: {e}")

    # 3. Resetta lo stato in memoria e scarica la RAM
    with _lock:
        _download_tracker.clear()

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

    # 4. Aggiorna settings.json
    try:
        from config import load_settings, save_settings
        s = load_settings()
        s["models_downloaded"] = False
        s["whisper_model"] = "mock_mode"
        save_settings(s)
    except Exception:
        pass

    logger.info(f"[Models] Eliminazione totale completata. Liberati {total_freed_mb:.1f} MB.")
    return {
        "status": "success",
        "freed_mb": round(total_freed_mb, 1),
        "deleted_count": len(deleted_items),
        "deleted_items": deleted_items,
        "message": f"Tutti i modelli sono stati eliminati con successo dal disco ({round(total_freed_mb, 1)} MB liberati).",
    }


# ── ELIMINA SINGOLO MODELLO ────────────────────────────────────────────────────

@router.delete("/{model_name:path}")
def delete_single_model(model_name: str):
    """
    Elimina fisicamente i file del singolo modello specificato dalla cache locale
    e scarica la memoria RAM corrispondente.
    """
    clean_name = model_name.strip()

    if clean_name in ("all/delete-all", "all"):
        return delete_all_models()

    repos_to_delete: List[str] = []

    # Cerca tra Whisper
    if clean_name in ("dual-tier-whisper", "whisper", "large-v3-turbo-q4", "large-v3-turbo"):
        repos_to_delete = ["mlx-community/whisper-large-v3-turbo-q4"]
    else:
        for cat in (WHISPER_MODELS, LLM_MODELS, EMBEDDING_MODELS):
            for k, v in cat.items():
                if k == clean_name or v.get("repo_id") == clean_name or clean_name in v.get("repos", []):
                    repos_to_delete = list(v.get("repos") or [v.get("repo_id")])
                    break
            if repos_to_delete:
                break

    if not repos_to_delete:
        repos_to_delete = [clean_name]

    freed_mb = 0.0

    # 1. Usa scan_cache_dir per eliminare revisioni e blobs
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        for repo in info.repos:
            for target_repo in repos_to_delete:
                clean_r = repo.repo_id.replace("-mlx", "")
                clean_t = target_repo.replace("-mlx", "")
                if repo.repo_id == target_repo or clean_r == clean_t:
                    revs = [rev.commit_hash for rev in repo.revisions]
                    if revs:
                        strat = info.delete_revisions(*revs)
                        freed_from_hf = round(strat.expected_freed_size / (1024 * 1024), 1)
                        strat.execute()
                        freed_mb += freed_from_hf
                        logger.info(f"[Models] Eliminato tramite HF scan_cache_dir: {repo.repo_id} ({freed_from_hf} MB)")
    except Exception as e:
        logger.warning(f"[Models] Errore scan_cache_dir delete: {e}")

    # 2. Rimuovi cartelle su disco
    cache_root = get_hf_cache_dir()
    seen_dirs = set()
    for target_repo in repos_to_delete:
        candidates = [
            "models--" + target_repo.replace("/", "--"),
        ]
        if target_repo.endswith("-mlx"):
            candidates.append("models--" + target_repo[:-4].replace("/", "--"))
        else:
            candidates.append("models--" + target_repo.replace("/", "--") + "-mlx")

        for folder_name in candidates:
            model_dir = os.path.join(cache_root, folder_name)
            if model_dir not in seen_dirs and os.path.exists(model_dir):
                seen_dirs.add(model_dir)
                size_mb = _get_dir_size_mb(model_dir)
                try:
                    shutil.rmtree(model_dir)
                    freed_mb += size_mb
                    logger.info(f"[Models] Cartella eliminata: {model_dir} (liberati {size_mb} MB)")
                except Exception as e:
                    logger.error(f"[Models] Errore eliminazione directory {model_dir}: {e}")

    with _lock:
        _download_tracker.pop(clean_name, None)

    # Rilascia la memoria
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

    return {
        "status": "deleted",
        "model": clean_name,
        "freed_mb": round(freed_mb, 1),
        "message": f"Modello {clean_name} rimosso con successo dal disco. Liberati {round(freed_mb, 1)} MB e memoria RAM scaricata.",
    }
