"""
platform_utils.py — SbobinAI platform detection & filesystem helpers.

Seleziona il backend IA in base alla piattaforma:
  - "mlx"      → macOS su Apple Silicon (mlx-whisper, mlx-lm, mlx-embeddings)
  - "portable" → Windows / Linux / Mac Intel (faster-whisper + llama-cpp-python)

Override manuale: variabile d'ambiente SBOBINAI_AI_BACKEND=mlx|portable.

Nessun import pesante a livello di modulo: questo file deve essere importabile ovunque.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import platform
import shutil
import stat
import sys
import time
from typing import Iterable, List, Optional

logger = logging.getLogger("sbobinai.platform")

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"

_env_backend = (os.environ.get("SBOBINAI_AI_BACKEND") or "").strip().lower()
if _env_backend in ("mlx", "portable"):
    AI_BACKEND = _env_backend
else:
    AI_BACKEND = "mlx" if IS_APPLE_SILICON else "portable"

USE_MLX = AI_BACKEND == "mlx"

# Su Windows huggingface_hub non può creare symlink senza Developer Mode: copia i file.
# Silenzia il warning ripetuto (comportamento invariato).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _register_nvidia_dll_dirs() -> None:
    """
    Windows: rende visibili a CTranslate2 le DLL CUDA installate via pip
    (nvidia-cublas-cu12, nvidia-cudnn-cu12 → site-packages/nvidia/*/bin).
    """
    if not IS_WINDOWS:
        return
    try:
        import site
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        roots = [p for p in sys.path if p.endswith("site-packages")]
    added = []
    for root in roots:
        nvidia_dir = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in added:
                added.append(bin_dir)
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])


if not USE_MLX:
    _register_nvidia_dll_dirs()


def backend_label() -> str:
    if USE_MLX:
        return "Apple Silicon (MLX Metal)"
    if cuda_device_count() > 0:
        return "NVIDIA CUDA"
    return "CPU"


# ── Accelerazione hardware (portable) ─────────────────────────────────────────

_cuda_count_cache: Optional[int] = None


def cuda_device_count() -> int:
    """Numero di GPU CUDA visibili a CTranslate2 (0 se assente / non installato)."""
    global _cuda_count_cache
    if _cuda_count_cache is not None:
        return _cuda_count_cache
    count = 0
    if not USE_MLX:
        try:
            import ctranslate2
            count = int(ctranslate2.get_cuda_device_count())
        except Exception:
            count = 0
    _cuda_count_cache = count
    return count


def llama_gpu_offload_supported() -> bool:
    """True se la build installata di llama-cpp-python supporta l'offload su GPU (CUDA/Vulkan)."""
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def cpu_threads() -> int:
    """Thread di calcolo consigliati: core fisici stimati (logici / 2), minimo 2, massimo 16."""
    logical = os.cpu_count() or 4
    return max(2, min(16, logical // 2 if logical >= 8 else logical))


# ── Cache HuggingFace ─────────────────────────────────────────────────────────

def hf_cache_dir() -> str:
    """Cartella di cache dei modelli HuggingFace (rispetta HF_HOME / HF_HUB_CACHE)."""
    try:
        from huggingface_hub import constants
        return str(constants.HF_HUB_CACHE)
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def hf_repo_folder(repo_id: str) -> str:
    return os.path.join(hf_cache_dir(), "models--" + repo_id.replace("/", "--"))


def _latest_snapshot(repo_id: str) -> Optional[str]:
    snapshots_dir = os.path.join(hf_repo_folder(repo_id), "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    cands = [os.path.join(snapshots_dir, e) for e in os.listdir(snapshots_dir)]
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        return None
    # La snapshot più recente su disco (non l'ordine alfabetico dell'hash di commit)
    return max(cands, key=os.path.getmtime)


def get_local_snapshot_dir(repo_id: str) -> Optional[str]:
    """
    Risolve la snapshot locale di un repo HF senza alcuna chiamata di rete.
    Restituisce None se il repo non è in cache.
    """
    if not repo_id or repo_id == "mock":
        return None
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id, local_files_only=True)
        if local_path and os.path.isdir(local_path):
            return str(local_path)
    except Exception:
        pass
    return _latest_snapshot(repo_id)


def _file_ok(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def snapshot_is_complete(snapshot_dir: Optional[str], required: Iterable[str]) -> bool:
    """
    Verifica che la snapshot contenga TUTTI i file richiesti (glob ammessi) non vuoti e,
    se presente un indice safetensors, tutti gli shard dei pesi elencati.
    """
    if not snapshot_dir or not os.path.isdir(snapshot_dir):
        return False
    try:
        names: List[str] = []
        for dirpath, _, filenames in os.walk(snapshot_dir):
            rel = os.path.relpath(dirpath, snapshot_dir)
            for f in filenames:
                names.append(f if rel == "." else os.path.join(rel, f).replace("\\", "/"))
    except OSError:
        return False

    for pattern in required:
        matches = [n for n in names if fnmatch.fnmatch(n, pattern)]
        if not matches or not all(_file_ok(os.path.join(snapshot_dir, m)) for m in matches):
            return False

    # Pesi sharded: verifica che ogni shard nel weight_map esista
    for n in names:
        if n.endswith(".safetensors.index.json"):
            try:
                import json
                with open(os.path.join(snapshot_dir, n), "r", encoding="utf-8") as fh:
                    weight_map = json.load(fh).get("weight_map", {})
                for shard in set(weight_map.values()):
                    if not _file_ok(os.path.join(snapshot_dir, os.path.dirname(n), shard)):
                        return False
            except Exception:
                return False
    return True


def dir_size_mb(path: str) -> float:
    """Dimensione su disco di una directory (dedup per inode: blob HF + symlink contati una volta)."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    seen = set()
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                st = os.stat(os.path.realpath(fp))
                key = (st.st_dev, st.st_ino) if st.st_ino else fp
                if key not in seen:
                    seen.add(key)
                    total += st.st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 1)


# ── Rimozione file robusta (Windows: file bloccati / read-only / mmap) ───────

def _on_rm_error(func, path, exc_info=None):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def safe_rmtree(path: str, retries: int = 5, delay: float = 0.3) -> bool:
    """
    Elimina una directory riprovando: su Windows un file ancora mappato in memoria
    (es. GGUF con mmap) o aperto da un antivirus genera PermissionError temporanei.
    """
    for attempt in range(retries):
        if not os.path.exists(path):
            return True
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_on_rm_error)
            else:  # pragma: no cover
                shutil.rmtree(path, onerror=_on_rm_error)
        except Exception as exc:
            logger.debug(f"[Platform] rmtree tentativo {attempt + 1} fallito per {path}: {exc}")
        if not os.path.exists(path):
            return True
        import gc
        gc.collect()
        time.sleep(delay * (attempt + 1))
    return not os.path.exists(path)


def safe_remove(path: str, retries: int = 5, delay: float = 0.2) -> bool:
    """Rimuove un file riprovando in caso di lock temporanei (Windows)."""
    for attempt in range(retries):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            try:
                os.chmod(path, stat.S_IWRITE)
            except Exception:
                pass
            time.sleep(delay * (attempt + 1))
        except OSError:
            time.sleep(delay * (attempt + 1))
    return not os.path.exists(path)
