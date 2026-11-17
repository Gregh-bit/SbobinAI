"""
Test per rilevamento GPU NVIDIA / runtime CUDA 12 + cuDNN 9 e isolamento multipiattaforma.
"""
from __future__ import annotations

import os
import tempfile
import threading
import zipfile

import ai_models
import platform_utils
from routers import models as models_router


def test_macos_hides_cuda_and_blocks_download(client):
    r = client.get("/api/v1/models/status")
    assert r.status_code == 200
    data = r.json()
    assert data["nvidia_gpu_detected"] is False
    assert data["runtime"] == []

    dl = client.post("/api/v1/models/download", json={"model_id": "cuda-runtime"})
    assert dl.status_code == 400


def test_windows_without_nvidia_gpu_hides_cuda(client, monkeypatch):
    monkeypatch.setattr(platform_utils, "IS_WINDOWS", True)
    monkeypatch.setattr(platform_utils, "USE_MLX", False)
    monkeypatch.setattr(platform_utils, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(models_router, "IS_WINDOWS", True)
    monkeypatch.setattr(models_router, "has_nvidia_gpu", lambda: False)

    r = client.get("/api/v1/models/status")
    assert r.status_code == 200
    data = r.json()
    assert data["nvidia_gpu_detected"] is False
    assert data["runtime"] == []

    dl = client.post("/api/v1/models/download", json={"model_id": "cuda-runtime"})
    assert dl.status_code == 400


def test_windows_nvidia_system_drivers_already_installed_are_counted_and_skipped(client, monkeypatch):
    """
    Se il PC Windows ha già CUDA 12 + cuDNN 9 installati a livello di sistema:
    - /models/status segnala downloaded=True, system_installed=True, status='completed'
    - /models/download-status segnala completed (100%)
    - POST /models/download ritorna 'completed' senza avviare alcun download
    """
    with tempfile.TemporaryDirectory(prefix="sbobinai_cuda_empty_") as empty_dir:
        monkeypatch.setattr(platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(platform_utils, "USE_MLX", False)
        monkeypatch.setattr(platform_utils, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_dir", lambda: empty_dir)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_bin_dir", lambda: os.path.join(empty_dir, "bin"))
        monkeypatch.setattr(
            platform_utils,
            "_can_load_dll",
            lambda name: name in {"cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops64_9.dll"},
        )
        monkeypatch.setattr(models_router, "IS_WINDOWS", True)
        monkeypatch.setattr(models_router, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(models_router, "get_cuda_runtime_dir", lambda: empty_dir)
        monkeypatch.setattr(models_router, "get_cuda_runtime_bin_dir", lambda: os.path.join(empty_dir, "bin"))

        worker_called = {"count": 0}
        monkeypatch.setattr(
            models_router,
            "_download_cuda_worker",
            lambda *a, **kw: worker_called.__setitem__("count", worker_called["count"] + 1),
        )

        st = client.get("/api/v1/models/status").json()
        assert st["nvidia_gpu_detected"] is True
        assert len(st["runtime"]) == 1
        rt = st["runtime"][0]
        assert rt["id"] == "cuda-runtime"
        assert rt["downloaded"] is True
        assert rt["system_installed"] is True
        assert rt["status"] == "completed"
        assert rt["progress"] == 100

        ds = client.get("/api/v1/models/download-status?model_id=cuda-runtime").json()
        assert ds["status"] == "completed"
        assert ds["progress"] == 100

        dl = client.post("/api/v1/models/download", json={"model_id": "cuda-runtime"})
        assert dl.status_code == 200
        assert dl.json()["status"] == "completed"
        assert dl.json()["progress"] == 100
        assert worker_called["count"] == 0


def test_windows_nvidia_partial_system_drivers_without_cudnn_are_not_counted(client, monkeypatch):
    """Se c'è solo cublas64_12 ma manca cuDNN 9, non deve risultare già installato."""
    with tempfile.TemporaryDirectory(prefix="sbobinai_cuda_partial_") as empty_dir:
        monkeypatch.setattr(platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(platform_utils, "USE_MLX", False)
        monkeypatch.setattr(platform_utils, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_dir", lambda: empty_dir)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_bin_dir", lambda: os.path.join(empty_dir, "bin"))
        monkeypatch.setattr(
            platform_utils,
            "_can_load_dll",
            lambda name: name in {"cublas64_12.dll", "cublasLt64_12.dll"},
        )
        monkeypatch.setattr(models_router, "IS_WINDOWS", True)
        monkeypatch.setattr(models_router, "has_nvidia_gpu", lambda: True)

        st = client.get("/api/v1/models/status").json()
        assert st["nvidia_gpu_detected"] is True
        rt = st["runtime"][0]
        assert rt["downloaded"] is False
        assert rt["system_installed"] is False


def test_windows_nvidia_local_cuda_download_and_delete(client, monkeypatch):
    """Simula download atomico locale dei pacchetti CUDA + rimozione dalle impostazioni."""
    with tempfile.TemporaryDirectory(prefix="sbobinai_cuda_local_") as tmp_dir:
        cuda_dir = os.path.join(tmp_dir, "cuda_runtime")
        bin_dir = os.path.join(cuda_dir, "bin")

        fake_whls = {}
        specs = {
            "nvidia-cuda-runtime-cu12": [("nvidia/cuda_runtime/bin/cudart64_12.dll", b"MZ" + b"\x00" * 1024)],
            "nvidia-cublas-cu12": [
                ("nvidia/cublas/bin/cublas64_12.dll", b"MZ" + b"\x00" * 1024),
                ("nvidia/cublas/bin/cublasLt64_12.dll", b"MZ" + b"\x00" * 1024),
            ],
            "nvidia-cudnn-cu12": [("nvidia/cudnn/bin/cudnn_ops64_9.dll", b"MZ" + b"\x00" * 1024)],
        }
        for pkg, entries in specs.items():
            whl_path = os.path.join(tmp_dir, f"{pkg}.whl")
            with zipfile.ZipFile(whl_path, "w") as zf:
                for arcname, payload in entries:
                    zf.writestr(arcname, payload)
            with open(whl_path, "rb") as fh:
                fake_whls[pkg] = fh.read()

        class FakeResp:
            def __init__(self, data: bytes):
                self._data = data
            def raise_for_status(self):
                return None
            def iter_content(self, chunk_size=262144):
                yield self._data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        import requests

        monkeypatch.setattr(platform_utils, "IS_WINDOWS", True)
        monkeypatch.setattr(platform_utils, "USE_MLX", False)
        monkeypatch.setattr(platform_utils, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_dir", lambda: cuda_dir)
        monkeypatch.setattr(platform_utils, "get_cuda_runtime_bin_dir", lambda: bin_dir)
        monkeypatch.setattr(platform_utils, "_can_load_dll", lambda _name: False)
        monkeypatch.setattr(models_router, "IS_WINDOWS", True)
        monkeypatch.setattr(models_router, "has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(models_router, "get_cuda_runtime_dir", lambda: cuda_dir)
        monkeypatch.setattr(models_router, "get_cuda_runtime_bin_dir", lambda: bin_dir)
        monkeypatch.setattr(models_router, "reset_cuda_state", lambda: None)
        monkeypatch.setattr(
            models_router,
            "_resolve_pypi_win_wheel",
            lambda pkg, _m: ("12.0.0", len(fake_whls[pkg]), f"https://example.invalid/{pkg}.whl"),
        )
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, stream=True, timeout=30: next(
                FakeResp(data) for pkg, data in fake_whls.items() if f"/{pkg}.whl" in url
            ),
        )

        try:
            models_router._download_cuda_worker(threading.Event())
            st = client.get("/api/v1/models/status").json()
            rt = st["runtime"][0]
            assert rt["downloaded"] is True
            assert rt["system_installed"] is False

            res = client.delete("/api/v1/models/cuda-runtime")
            assert res.status_code == 200
            assert not os.path.exists(cuda_dir)

            st_after = client.get("/api/v1/models/status").json()
            assert st_after["runtime"][0]["downloaded"] is False
        finally:
            models_router._download_tracker.pop(ai_models.CUDA_RUNTIME_ID, None)
            platform_utils._cuda_disabled = False
            platform_utils._cuda_count_cache = None
