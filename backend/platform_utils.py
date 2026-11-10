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
import threading
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


_dll_dir_handles: list = []
_registered_dll_dirs: set[str] = set()


def get_cuda_runtime_dir() -> str:
    """Cartella locale dedicata alle librerie CUDA opzionali su Windows."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return os.path.join(base, "SbobinAI", "cuda_runtime")
    return os.path.expanduser("~/.sbobinai/cuda_runtime")


def get_cuda_runtime_bin_dir() -> str:
    return os.path.join(get_cuda_runtime_dir(), "bin")


def _cuda_pending_delete_marker() -> str:
    return os.path.join(get_cuda_runtime_dir(), ".pending_delete")


def cleanup_pending_cuda_runtime() -> None:
    """Elimina eventuali file CUDA marcati per la rimozione alla chiusura precedente (lock DLL Windows)."""
    if not IS_WINDOWS:
        return
    marker = _cuda_pending_delete_marker()
    cuda_dir = get_cuda_runtime_dir()
    if os.path.isfile(marker):
        try:
            shutil.rmtree(cuda_dir, ignore_errors=True)
        except Exception:
            pass


def _register_nvidia_dll_dirs() -> None:
    """
    Windows: rende visibili a CTranslate2 le DLL CUDA scaricate in cuda_runtime/bin,
    eventuali wheel nvidia in site-packages e il CUDA Toolkit di sistema.
    """
    if not IS_WINDOWS:
        return
    cleanup_pending_cuda_runtime()

    candidate_dirs: list[str] = []
    # 1. Runtime CUDA gestito da SbobinAI (%LOCALAPPDATA%\SbobinAI\cuda_runtime\bin)
    if not os.path.isfile(_cuda_pending_delete_marker()):
        managed_bin = get_cuda_runtime_bin_dir()
        if os.path.isdir(managed_bin):
            candidate_dirs.append(managed_bin)

    # 2. Eventuali pacchetti nvidia in site-packages
    try:
        import site
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        roots = [p for p in sys.path if p.endswith("site-packages")]

    for root in roots:
        nvidia_dir = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir):
                candidate_dirs.append(bin_dir)

    # 3. CUDA Toolkit di sistema (CUDA_PATH / CUDA_PATH_V12_*)
    for env_key, env_val in os.environ.items():
        if env_val and (env_key == "CUDA_PATH" or env_key.startswith("CUDA_PATH_V12")):
            bin_dir = os.path.join(env_val, "bin")
            if os.path.isdir(bin_dir):
                candidate_dirs.append(bin_dir)

    added: list[str] = []
    for bin_dir in candidate_dirs:
        norm = os.path.normcase(os.path.abspath(bin_dir))
        if norm not in _registered_dll_dirs:
            _registered_dll_dirs.add(norm)
            added.append(bin_dir)
            try:
                _dll_dir_handles.append(os.add_dll_directory(bin_dir))
            except Exception:
                pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])


if not USE_MLX:
    _register_nvidia_dll_dirs()


def _can_load_dll(name: str) -> bool:
    """Su Windows verifica se una DLL è caricabile e rilascia subito l'handle di probe."""
    if not IS_WINDOWS:
        return True
    try:
        import ctypes
        lib = ctypes.WinDLL(name)
        handle = getattr(lib, "_handle", None)
        del lib
        if handle:
            try:
                ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            except Exception:
                pass
        return True
    except Exception:
        return False


def _cuda_runtime_dlls_available() -> bool:
    """
    Su Windows verifica che le DLL CUDA 12 essenziali (cublas64_12.dll, cublasLt64_12.dll)
    siano effettivamente caricabili prima di inizializzare CTranslate2 su GPU.
    """
    if not IS_WINDOWS:
        return True
    if _can_load_dll("cublas64_12.dll") and _can_load_dll("cublasLt64_12.dll"):
        return True
    logger.info(
        "[Platform] GPU NVIDIA rilevata ma runtime CUDA 12 non ancora installato: "
        "utilizzo automatico CPU int8."
    )
    return False


_has_nvidia_cache: Optional[bool] = None


def has_nvidia_gpu() -> bool:
    """
    True SOLO su Windows e SOLO se il sistema rileva una GPU NVIDIA.
    Su macOS e Linux restituisce sempre False.
    """
    global _has_nvidia_cache
    if not IS_WINDOWS or USE_MLX:
        return False
    if _has_nvidia_cache is not None:
        return _has_nvidia_cache
    detected = False
    try:
        import ctranslate2
        if int(ctranslate2.get_cuda_device_count()) > 0:
            detected = True
    except Exception:
        pass
    if not detected:
        try:
            if shutil.which("nvidia-smi"):
                detected = True
        except Exception:
            pass
    _has_nvidia_cache = detected
    return detected


def cuda_runtime_status() -> tuple[bool, bool, float]:
    """
    Restituisce (scaricato_o_pronto, fornito_dal_sistema, dimensione_mb).
    Sempre (False, False, 0.0) al di fuori di Windows o senza GPU NVIDIA.
    """
    if not IS_WINDOWS or not has_nvidia_gpu():
        return False, False, 0.0
    if os.path.isfile(_cuda_pending_delete_marker()):
        return False, False, 0.0

    managed_dir = get_cuda_runtime_dir()
    managed_bin = get_cuda_runtime_bin_dir()
    has_managed_cublas = (
        os.path.isfile(os.path.join(managed_bin, "cublas64_12.dll"))
        and os.path.isfile(os.path.join(managed_bin, "cublasLt64_12.dll"))
    )
    has_managed_cudnn = any(
        os.path.isfile(os.path.join(managed_bin, dll))
        for dll in ("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn64_8.dll")
    )
    if has_managed_cublas and has_managed_cudnn:
        return True, False, dir_size_mb(managed_dir)

    # Verifica se già disponibile a livello di sistema (CUDA Toolkit globale o site-packages)
    has_sys_cublas = _can_load_dll("cublas64_12.dll") and _can_load_dll("cublasLt64_12.dll")
    has_sys_cudnn = any(
        _can_load_dll(dll)
        for dll in ("cudnn_ops64_9.dll", "cudnn64_9.dll", "cudnn_ops_infer64_8.dll", "cudnn64_8.dll")
    )
    if has_sys_cublas and has_sys_cudnn:
        return True, True, 0.0

    return False, False, 0.0


def reset_cuda_state() -> None:
    """Riattiva il rilevamento CUDA dopo l'installazione del runtime GPU in-app."""
    global _cuda_count_cache, _cuda_disabled
    _register_nvidia_dll_dirs()
    _cuda_disabled = False
    _cuda_count_cache = None
    invalidate_dir_size_cache(get_cuda_runtime_dir())


def remove_cuda_runtime() -> float:
    """
    Rimuove la cartella cuda_runtime scaricata da SbobinAI.
    Se Windows mantiene una DLL bloccata in RAM fino al riavvio, marca .pending_delete
    così viene disabilitata subito e rimossa al prossimo avvio.
    """
    disable_cuda("librerie CUDA rimosse dall'utente")
    cuda_dir = get_cuda_runtime_dir()
    if not os.path.exists(cuda_dir):
        return 0.0
    size_before = dir_size_mb(cuda_dir)
    if not safe_rmtree(cuda_dir, retries=3, delay=0.2):
        try:
            os.makedirs(cuda_dir, exist_ok=True)
            with open(_cuda_pending_delete_marker(), "w", encoding="utf-8") as fh:
                fh.write("pending\n")
        except Exception:
            pass
    invalidate_dir_size_cache(cuda_dir)
    return size_before


def backend_label() -> str:
    if USE_MLX:
        return "Apple Silicon (MLX Metal)"
    if cuda_device_count() > 0:
        return "NVIDIA CUDA"
    return "CPU"


# ── Accelerazione hardware (portable) ─────────────────────────────────────────

_cuda_count_cache: Optional[int] = None
_cuda_disabled: bool = False


def disable_cuda(reason: str = "") -> None:
    """Disabilita permanentemente CUDA per il processo corrente dopo un errore runtime."""
    global _cuda_disabled, _cuda_count_cache
    _cuda_disabled = True
    _cuda_count_cache = 0
    if reason:
        logger.warning(f"[Platform] CUDA disabilitato permanentemente per questa sessione ({reason}).")


def is_cuda_disabled() -> bool:
    return _cuda_disabled


def cuda_device_count() -> int:
    """Numero di GPU CUDA realmente utilizzabili da CTranslate2 (0 se assente o DLL mancanti)."""
    global _cuda_count_cache
    if _cuda_disabled:
        return 0
    if _cuda_count_cache is not None:
        return _cuda_count_cache
    count = 0
    if not USE_MLX:
        try:
            if IS_WINDOWS and os.path.isfile(_cuda_pending_delete_marker()):
                count = 0
            else:
                import ctranslate2
                raw_count = int(ctranslate2.get_cuda_device_count())
                if raw_count > 0 and _cuda_runtime_dlls_available():
                    count = raw_count
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


_size_cache: dict[str, tuple[float, float]] = {}
_size_cache_lock = threading.Lock()


def invalidate_dir_size_cache(path: Optional[str] = None) -> None:
    """Invalida la cache delle dimensioni su disco."""
    with _size_cache_lock:
        if path:
            _size_cache.pop(path, None)
        else:
            _size_cache.clear()


def dir_size_mb(path: str, max_age_seconds: float = 6.0) -> float:
    """Dimensione su disco di una directory (dedup per inode: blob HF + symlink contati una volta). Cache breve per evitare disk thrashing."""
    if not os.path.exists(path):
        invalidate_dir_size_cache(path)
        return 0.0
    now = time.time()
    with _size_cache_lock:
        if path in _size_cache:
            ts, sz = _size_cache[path]
            if now - ts < max_age_seconds:
                return sz

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
    sz = round(total / (1024 * 1024), 1)
    with _size_cache_lock:
        _size_cache[path] = (now, sz)
    return sz


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
            invalidate_dir_size_cache(path)
            return True
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_on_rm_error)
            else:  # pragma: no cover
                shutil.rmtree(path, onerror=_on_rm_error)
        except Exception as exc:
            logger.debug(f"[Platform] rmtree tentativo {attempt + 1} fallito per {path}: {exc}")
        if not os.path.exists(path):
            invalidate_dir_size_cache(path)
            return True
        import gc
        gc.collect()
        time.sleep(delay * (attempt + 1))
    success = not os.path.exists(path)
    if success:
        invalidate_dir_size_cache(path)
    return success

