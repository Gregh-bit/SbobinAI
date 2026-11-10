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

import datetime
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

# macOS SSL Certificates & HuggingFace Transfer Configuration
try:
    import certifi
    ca_bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_bundle
    os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle
except ImportError:
    pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from tqdm.auto import tqdm

import ai_models
from platform_utils import (
    AI_BACKEND,
    IS_WINDOWS,
    backend_label,
    cuda_runtime_status,
    dir_size_mb,
    get_cuda_runtime_bin_dir,
    get_cuda_runtime_dir,
    has_nvidia_gpu,
    hf_cache_dir,
    hf_repo_folder,
    invalidate_dir_size_cache,
    remove_cuda_runtime,
    reset_cuda_state,
    safe_rmtree,
)

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
DOWNLOAD_STATUS = _download_tracker  # Alias pubblico per accesso rapido
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
                    if tracker["downloaded_bytes"] > t_bytes:
                        tracker["total_bytes"] = tracker["downloaded_bytes"]
                        tracker["total_mb"] = tracker["downloaded_mb"]
                        t_bytes = tracker["downloaded_bytes"]
                    if t_bytes > 0:
                        pct = min(99, max(1, int((tracker["downloaded_bytes"] / t_bytes) * 100)))
                        tracker["progress"] = max(tracker.get("progress", 1), pct)

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

        invalidate_dir_size_cache(hf_repo_folder(repo_id))
        with _lock:
            actual_bytes = max(total_bytes, _download_tracker.get(model_id, {}).get("downloaded_bytes", total_bytes))
            actual_mb = round(actual_bytes / (1024 * 1024), 1)
            _download_tracker[model_id] = {
                "status": "completed",
                "progress": 100,
                "downloaded_bytes": actual_bytes,
                "downloaded_mb": actual_mb,
                "total_bytes": actual_bytes,
                "total_mb": actual_mb,
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
        err_msg = str(exc)
        logger.error(f"[Models] Errore durante il download di {model_id}: {err_msg}", exc_info=True)
        try:
            if sys.platform == "win32":
                log_dir = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local"), "SbobinAI", "logs")
            elif sys.platform == "darwin":
                log_dir = os.path.expanduser("~/Library/Logs/SbobinAI")
            else:
                log_dir = os.path.expanduser("~/.sbobinai/logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "backend.log"), "a", encoding="utf-8") as f:
                f.write(f"\n[DOWNLOAD ERROR] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Modello {model_id}: {err_msg}\n")
        except Exception:
            pass

        # Pulisce eventuali file parziali per non lasciare sporcizia sul disco
        try:
            if ai_models.local_snapshot(spec) is None:
                _delete_repos(spec.all_repo_ids())
        except Exception:
            pass

        with _lock:
            _download_tracker[model_id] = {
                "status": "error",
                "progress": 0,
                "downloaded_mb": 0.0,
                "total_mb": total_mb,
                "stage": "Errore",
                "error": err_msg,
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
        _delete_repos(spec.all_repo_ids())
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


def _finish_cuda_cancelled(total_mb: float) -> None:
    model_id = ai_models.CUDA_RUNTIME_ID
    ai_models.set_downloading(model_id, False)
    safe_rmtree(get_cuda_runtime_dir() + ".staging", retries=2, delay=0.15)
    with _lock:
        _download_tracker[model_id] = {
            "status": "cancelled",
            "progress": 0,
            "downloaded_mb": 0.0,
            "total_mb": total_mb,
            "stage": "Download interrotto dall'utente.",
            "error": "Download interrotto dall'utente.",
        }


def _resolve_pypi_win_wheel(pkg: str, major_prefix: str) -> tuple[str, int, str]:
    """Risolve versione, dimensione e URL del wheel win_amd64 compatibile su PyPI."""
    import re
    import requests

    def _ver_key(v: str) -> list[int]:
        return [int(x) if x.isdigit() else 0 for x in re.split(r"[.\-a-zA-Z]+", v) if x]

    resp = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=15)
    resp.raise_for_status()
    releases = resp.json().get("releases") or {}
    for ver in sorted(releases.keys(), key=_ver_key, reverse=True):
        if major_prefix and not ver.startswith(major_prefix):
            continue
        for asset in releases[ver]:
            fname = str(asset.get("filename") or "")
            if "win_amd64" in fname and fname.endswith(".whl") and not asset.get("yanked"):
                return ver, int(asset.get("size") or 0), str(asset["url"])
    raise RuntimeError(f"Nessun pacchetto Windows compatibile trovato per {pkg} ({major_prefix}*).")


def _download_cuda_worker(cancel_event: threading.Event) -> None:
    """
    Scarica e estrae in modo atomico i binari CUDA 12 + cuDNN 9 per Windows x64
    in %LOCALAPPDATA%\\SbobinAI\\cuda_runtime\\bin senza richiedere pip o CUDA Toolkit.
    """
    import shutil
    import zipfile
    import requests

    model_id = ai_models.CUDA_RUNTIME_ID
    staging_dir = get_cuda_runtime_dir() + ".staging"
    staging_bin = os.path.join(staging_dir, "bin")

    def _check_cancel() -> None:
        if cancel_event.is_set():
            raise InterruptedError("Download annullato su richiesta dell'utente.")

    total_bytes = ai_models.CUDA_RUNTIME_APPROX_BYTES
    total_mb = round(total_bytes / (1024 * 1024), 1)

    try:
        packages = [
            ("nvidia-cuda-runtime-cu12", "12.", "CUDA Runtime 12"),
            ("nvidia-cublas-cu12", "12.", "cuBLAS 12"),
            ("nvidia-cudnn-cu12", "9.", "cuDNN 9"),
        ]
        resolved: list[tuple[str, str, int, str]] = []
        _check_cancel()
        for pkg, prefix, label in packages:
            ver, sz, url = _resolve_pypi_win_wheel(pkg, prefix)
            resolved.append((pkg, label, sz, url))
            _check_cancel()

        exact_total = sum(sz for _, _, sz, _ in resolved)
        if exact_total > 0:
            total_bytes = exact_total
            total_mb = round(total_bytes / (1024 * 1024), 1)

        with _lock:
            _download_tracker[model_id] = {
                "status": "downloading",
                "progress": 2,
                "downloaded_bytes": 0,
                "downloaded_mb": 0.0,
                "total_bytes": total_bytes,
                "total_mb": total_mb,
                "stage": "Avvio download librerie CUDA...",
                "error": None,
            }

        safe_rmtree(staging_dir, retries=2, delay=0.15)
        os.makedirs(staging_bin, exist_ok=True)

        downloaded_bytes = 0
        for idx, (pkg, label, _sz, url) in enumerate(resolved, start=1):
            _check_cancel()
            whl_path = os.path.join(staging_dir, f"{pkg}.whl")
            with _lock:
                if model_id in _download_tracker:
                    _download_tracker[model_id]["stage"] = f"Download {label} ({idx}/{len(resolved)})..."

            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(whl_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=262144):
                        _check_cancel()
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded_bytes += len(chunk)
                        with _lock:
                            tr = _download_tracker.get(model_id)
                            if tr is not None:
                                tr["downloaded_bytes"] = downloaded_bytes
                                tr["downloaded_mb"] = round(downloaded_bytes / (1024 * 1024), 1)
                                t_b = tr.get("total_bytes") or total_bytes
                                if downloaded_bytes > t_b:
                                    tr["total_bytes"] = downloaded_bytes
                                    tr["total_mb"] = tr["downloaded_mb"]
                                    t_b = downloaded_bytes
                                pct = min(92, max(2, int((downloaded_bytes / t_b) * 92)))
                                tr["progress"] = max(tr.get("progress", 2), pct)

            with _lock:
                if model_id in _download_tracker:
                    _download_tracker[model_id]["stage"] = f"Estrazione DLL {label}..."
                    _download_tracker[model_id]["progress"] = max(_download_tracker[model_id].get("progress", 92), 92 + idx * 2)

            with zipfile.ZipFile(whl_path, "r") as zf:
                for member in zf.infolist():
                    _check_cancel()
                    if member.is_dir() or not member.filename.lower().endswith(".dll"):
                        continue
                    dll_name = os.path.basename(member.filename)
                    if not dll_name:
                        continue
                    dest = os.path.join(staging_bin, dll_name)
                    with zf.open(member, "r") as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            try:
                os.remove(whl_path)
            except OSError:
                pass

        _check_cancel()
        target_dir = get_cuda_runtime_dir()
        target_bin = get_cuda_runtime_bin_dir()

        # Se era presente una vecchia cartella o .pending_delete, rilascia eventuali motori prima di sostituire
        _release_engines()
        safe_rmtree(target_dir, retries=3, delay=0.2)
        pending_marker = os.path.join(target_dir, ".pending_delete")
        if os.path.exists(pending_marker):
            try:
                os.remove(pending_marker)
            except OSError:
                pass
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(target_bin):
            for entry_name in os.listdir(staging_bin):
                src_p = os.path.join(staging_bin, entry_name)
                dst_p = os.path.join(target_bin, entry_name)
                try:
                    os.replace(src_p, dst_p)
                except OSError:
                    shutil.copy2(src_p, dst_p)
        else:
            shutil.move(staging_bin, target_bin)

        # Attiva subito CUDA nel processo corrente senza bisogno di riavviare
        reset_cuda_state()
        ai_models.set_downloading(model_id, False)
        invalidate_dir_size_cache(target_dir)

        actual_mb = dir_size_mb(target_dir) or total_mb
        with _lock:
            _download_tracker[model_id] = {
                "status": "completed",
                "progress": 100,
                "downloaded_bytes": total_bytes,
                "downloaded_mb": actual_mb,
                "total_bytes": total_bytes,
                "total_mb": actual_mb,
                "stage": "Accelerazione GPU attiva!",
                "error": None,
            }
        logger.info(f"[Models] Librerie CUDA installate con successo in {target_dir} ({actual_mb} MB).")

    except InterruptedError:
        _finish_cuda_cancelled(total_mb)

    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"[Models] Errore download librerie CUDA: {err_msg}", exc_info=True)
        with _lock:
            _download_tracker[model_id] = {
                "status": "error",
                "progress": 0,
                "downloaded_mb": 0.0,
                "total_mb": total_mb,
                "stage": "Errore",
                "error": err_msg,
            }
    finally:
        ai_models.set_downloading(model_id, False)
        safe_rmtree(staging_dir, retries=2, delay=0.15)
        with _lock:
            if _cancel_events.get(model_id) is cancel_event:
                _cancel_events.pop(model_id, None)
            _download_threads.pop(model_id, None)


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
    Restituisce lo stato dettagliato di ciascun modello (Whisper, LLM, Embeddings) del backend corrente
    e, SOLO su Windows con GPU NVIDIA rilevata, del pacchetto opzionale CUDA.
    """
    whisper_list: List[Dict[str, Any]] = []
    llm_list: List[Dict[str, Any]] = []
    embedding_list: List[Dict[str, Any]] = []
    runtime_list: List[Dict[str, Any]] = []
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

    nvidia_detected = bool(IS_WINDOWS and has_nvidia_gpu())
    if nvidia_detected:
        cuda_info = ai_models.cuda_runtime_info()
        is_cuda_dl, is_sys_cuda, cuda_size_mb = cuda_runtime_status()
        cuda_downloading = ai_models.is_downloading(ai_models.CUDA_RUNTIME_ID)
        effective_dl = is_cuda_dl and not cuda_downloading
        item = _status_item(ai_models.CUDA_RUNTIME_ID, cuda_info, effective_dl, cuda_size_mb)
        item["system_installed"] = is_sys_cuda
        runtime_list.append(item)
        if effective_dl and not is_sys_cuda and cuda_size_mb > 0:
            installed_summary.append({
                "id": ai_models.CUDA_RUNTIME_ID,
                "name": str(cuda_info["name"]),
                "type": "cuda",
                "size_mb": cuda_size_mb,
            })

    total_installed_mb = sum(m["size_mb"] for m in installed_summary)

    return {
        "whisper": whisper_list,
        "llm": llm_list,
        "embedding": embedding_list,
        "runtime": runtime_list,
        "nvidia_gpu_detected": nvidia_detected,
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


@router.get("/download-status")
def get_download_status(model_id: Optional[str] = None):
    """
    Endpoint veloce per interrogare lo stato del download in corso (polling ogni 1-2 secondi)
    senza calcolare le dimensioni su disco di tutti i modelli del sistema.
    """
    with _lock:
        if model_id:
            raw_id = model_id.strip()
            spec = ai_models.resolve(raw_id)
            resolved_id = spec.id if spec else raw_id
            item = _download_tracker.get(resolved_id)
            if item is None:
                if resolved_id == ai_models.CUDA_RUNTIME_ID:
                    is_dl, _sys, size_mb = cuda_runtime_status()
                    if is_dl:
                        return {
                            "status": "completed",
                            "model_id": resolved_id,
                            "progress": 100,
                            "downloaded_mb": size_mb,
                            "total_mb": size_mb,
                            "stage": "Completato",
                            "error": None,
                        }
                elif spec is not None:
                    is_dl, size_mb = ai_models.is_downloaded(spec)
                    if is_dl:
                        return {
                            "status": "completed",
                            "model_id": resolved_id,
                            "progress": 100,
                            "downloaded_mb": size_mb,
                            "total_mb": size_mb,
                            "stage": "Completato",
                            "error": None,
                        }
                return {
                    "status": "idle",
                    "model_id": resolved_id,
                    "progress": 0,
                    "downloaded_mb": 0.0,
                    "total_mb": 0.0,
                    "stage": None,
                    "error": None,
                }
            return {"model_id": resolved_id, **item}
        return {mid: dict(data) for mid, data in _download_tracker.items()}


class DownloadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str


@router.post("/download")
def download_model(payload: DownloadRequest):
    """
    Avvia il download in background del modello o componente specificato con monitoraggio dei byte.
    Risponde IMMEDIATAMENTE (in ~50ms) per evitare timeout su WebKit/Tauri.
    """
    requested = payload.model_id.strip()
    if requested == ai_models.MOCK_WHISPER_ID:
        return {"status": "completed", "model": requested, "model_id": requested, "message": "Modalità simulatore sempre disponibile (0 MB)."}

    if requested == ai_models.CUDA_RUNTIME_ID:
        if not (IS_WINDOWS and has_nvidia_gpu()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="L'accelerazione GPU NVIDIA (CUDA) è disponibile solo su Windows con GPU NVIDIA rilevata.",
            )
        model_id = ai_models.CUDA_RUNTIME_ID
        with _lock:
            thread = _download_threads.get(model_id)
            if thread is not None and thread.is_alive():
                current = _download_tracker.get(model_id, {})
                return {
                    "status": "downloading",
                    "model": model_id,
                    "model_id": model_id,
                    "progress": current.get("progress", 1),
                    "downloaded_mb": current.get("downloaded_mb", 0.0),
                    "total_mb": current.get("total_mb", 0.0),
                }

            approx_b = ai_models.CUDA_RUNTIME_APPROX_BYTES
            approx_mb = round(approx_b / (1024 * 1024), 1)
            _download_tracker[model_id] = {
                "status": "downloading",
                "progress": 1,
                "downloaded_bytes": 0,
                "downloaded_mb": 0.0,
                "total_bytes": approx_b,
                "total_mb": approx_mb,
                "stage": "Ricerca librerie CUDA compatibili...",
                "error": None,
            }
            cancel_event = threading.Event()
            _cancel_events[model_id] = cancel_event
            ai_models.set_downloading(model_id, True)
            thread = threading.Thread(
                target=_download_cuda_worker,
                args=(cancel_event,),
                daemon=True,
                name="cuda-download-worker",
            )
            _download_threads[model_id] = thread
        thread.start()
        return {
            "status": "downloading",
            "model": model_id,
            "model_id": model_id,
            "repos": ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
            "progress": 1,
        }

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
                "status": "downloading",
                "model": model_id,
                "model_id": model_id,
                "progress": current.get("progress", 1),
                "downloaded_mb": current.get("downloaded_mb", 0.0),
                "total_mb": current.get("total_mb", 0.0),
            }

        approx_b = spec.variant.approx_bytes or (580 * 1024 * 1024)
        approx_mb = round(approx_b / (1024 * 1024), 1)
        _download_tracker[model_id] = {
            "status": "downloading",
            "progress": 1,
            "downloaded_bytes": 0,
            "downloaded_mb": 0.0,
            "total_bytes": approx_b,
            "total_mb": approx_mb,
            "stage": "Connessione a HuggingFace...",
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
        "status": "downloading",
        "model": model_id,
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
    invalidate_dir_size_cache()
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

    if IS_WINDOWS and os.path.exists(get_cuda_runtime_dir()):
        cuda_freed = remove_cuda_runtime()
        if cuda_freed > 0:
            total_freed_mb = round(total_freed_mb + cuda_freed, 1)
            deleted_items.append(ai_models.CUDA_RUNTIME_ID)

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
    Elimina i file del singolo modello o runtime specificato (entrambe le varianti di backend)
    e scarica la memoria corrispondente.
    """
    clean_name = model_name.strip()
    if clean_name in ("all/delete-all", "all"):
        return delete_all_models()

    if clean_name == ai_models.CUDA_RUNTIME_ID:
        with _lock:
            ev = _cancel_events.get(ai_models.CUDA_RUNTIME_ID)
            th = _download_threads.get(ai_models.CUDA_RUNTIME_ID)
        if ev is not None:
            ev.set()
        if th is not None and th.is_alive():
            th.join(timeout=4.0)
        _release_engines()
        freed_mb = remove_cuda_runtime()
        with _lock:
            _download_tracker.pop(ai_models.CUDA_RUNTIME_ID, None)
        return {
            "status": "deleted",
            "model": clean_name,
            "freed_mb": freed_mb,
            "message": f"Librerie CUDA rimosse con successo ({freed_mb} MB liberati). Trascrizione impostata su CPU.",
        }

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

    try:
        from config import update_settings
        if spec.type == "whisper":
            update_settings({"models_downloaded": False, "whisper_model": ai_models.MOCK_WHISPER_ID})
        elif spec.type == "llm":
            update_settings({"llm_model": ai_models.LLM_DEFAULT_ID})
        elif spec.type == "embedding":
            update_settings({"embedding_model": ai_models.EMBEDDING_DEFAULT_ID})
    except Exception:
        pass

    return {
        "status": "deleted",
        "model": clean_name,
        "freed_mb": freed_mb,
        "message": f"Modello {clean_name} rimosso con successo dal disco. Liberati {freed_mb} MB e memoria RAM scaricata.",
    }
