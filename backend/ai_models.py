"""
ai_models.py — SbobinAI model catalog (platform-aware).

Gli ID logici dei modelli (salvati in settings.json e usati dal frontend) sono IDENTICI su
tutte le piattaforme; ogni ID ha una variante per backend:
  - "mlx"      → repo MLX 4-bit per Apple Silicon (comportamento macOS invariato)
  - "portable" → faster-whisper (CTranslate2) + GGUF per llama-cpp-python (Windows/Linux)

Qualsiasi repo id (MLX o portable) o alias storico viene risolto allo stesso modello logico.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from platform_utils import (
    AI_BACKEND,
    dir_size_mb,
    get_local_snapshot_dir,
    hf_repo_folder,
    snapshot_is_complete,
)

MB = 1024 * 1024

# ── ID logici (stabili, condivisi con config.DEFAULT_SETTINGS e frontend) ─────
WHISPER_CORE_ID = "large-v3-turbo-q4"
LLM_DEFAULT_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"
LLM_ALT_ID = "unsloth/gemma-4-E4B-it-UD-MLX-4bit"
EMBEDDING_DEFAULT_ID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
MOCK_WHISPER_ID = "mock_mode"


@dataclass(frozen=True)
class Variant:
    repo_id: str
    required: Tuple[str, ...]                  # glob dei file obbligatori nella snapshot
    allow_patterns: Optional[Tuple[str, ...]]  # None = scarica tutto il repo
    approx_bytes: int
    name: str
    description: str
    model_file: Optional[str] = None           # file GGUF da caricare (backend portable)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    type: str  # whisper | llm | embedding
    variants: Dict[str, Variant]
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    is_default: bool = False
    is_system_core: bool = False

    @property
    def variant(self) -> Variant:
        return self.variants.get(AI_BACKEND) or self.variants["portable"]

    @property
    def repo_id(self) -> str:
        return self.variant.repo_id

    def all_repo_ids(self) -> List[str]:
        return [v.repo_id for v in self.variants.values()]


CATALOG: Dict[str, ModelSpec] = {
    WHISPER_CORE_ID: ModelSpec(
        id=WHISPER_CORE_ID,
        type="whisper",
        is_system_core=True,
        aliases=("large-v3-turbo", "dual-tier-whisper", "whisper", "turbo"),
        variants={
            "mlx": Variant(
                repo_id="mlx-community/whisper-large-v3-turbo-q4",
                required=("config.json", "weights.*"),
                allow_patterns=None,
                approx_bytes=465 * MB,
                name="Whisper Large-v3-Turbo Q4 (Apple Silicon)",
                description="Motore Vocale di sistema obbligatorio: trascrizione ad alta precisione 4-bit per Apple Silicon Metal (~470 MB). Fedeltà massima, zero allucinazioni e consumo nullo in pausa.",
            ),
            "portable": Variant(
                repo_id="dropbox-dash/faster-whisper-large-v3-turbo",
                required=("config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.json"),
                allow_patterns=("*.json", "model.bin", "*.txt"),
                approx_bytes=1550 * MB,
                name="Whisper Large-v3-Turbo (faster-whisper, CUDA/CPU)",
                description="Motore Vocale di sistema obbligatorio: Whisper Large-v3-Turbo su CTranslate2 (~1.6 GB). GPU NVIDIA in float16 se disponibile, altrimenti CPU int8.",
            ),
        },
    ),
    LLM_DEFAULT_ID: ModelSpec(
        id=LLM_DEFAULT_ID,
        type="llm",
        is_default=True,
        aliases=("qwen3.5-4b", "qwen"),
        variants={
            "mlx": Variant(
                repo_id="mlx-community/Qwen3.5-4B-MLX-4bit",
                required=("config.json", "tokenizer*", "*.safetensors"),
                allow_patterns=None,
                approx_bytes=3050 * MB,
                name="Qwen 3.5 4B (4-bit MLX Metal)",
                description="LLM Primario (Consigliato 2026): Architettura Gated DeltaNet 2026, KV cache quantizzata 4-bit, ideale per MacBook Air M1 8GB. Risposte flash istantanee live e capitoli Quaderno.",
            ),
            "portable": Variant(
                repo_id="unsloth/Qwen3.5-4B-GGUF",
                required=("Qwen3.5-4B-Q4_K_M.gguf",),
                allow_patterns=("Qwen3.5-4B-Q4_K_M.gguf",),
                approx_bytes=2615 * MB,
                model_file="Qwen3.5-4B-Q4_K_M.gguf",
                name="Qwen 3.5 4B (GGUF Q4_K_M, llama.cpp)",
                description="LLM Primario: Qwen 3.5 4B quantizzato Q4_K_M via llama.cpp (~2.6 GB). Offload completo su GPU (CUDA/Vulkan) se la build lo supporta, altrimenti CPU.",
            ),
        },
    ),
    LLM_ALT_ID: ModelSpec(
        id=LLM_ALT_ID,
        type="llm",
        aliases=("gemma-4-e4b", "gemma"),
        variants={
            "mlx": Variant(
                repo_id="unsloth/gemma-4-E4B-it-UD-MLX-4bit",
                required=("config.json", "tokenizer*", "*.safetensors"),
                allow_patterns=None,
                approx_bytes=6290 * MB,
                name="Gemma 4 E4B (4-bit MLX Metal)",
                description="LLM Alternativo selezionabile basato su Gemma 4 E4B 4-bit per sintesi e spiegazioni didattiche.",
            ),
            "portable": Variant(
                repo_id="unsloth/gemma-4-E4B-it-GGUF",
                required=("gemma-4-E4B-it-Q4_K_M.gguf",),
                allow_patterns=("gemma-4-E4B-it-Q4_K_M.gguf",),
                approx_bytes=4750 * MB,
                model_file="gemma-4-E4B-it-Q4_K_M.gguf",
                name="Gemma 4 E4B (GGUF Q4_K_M, llama.cpp)",
                description="LLM Alternativo: Gemma 4 E4B quantizzato Q4_K_M via llama.cpp (~4.8 GB).",
            ),
        },
    ),
    EMBEDDING_DEFAULT_ID: ModelSpec(
        id=EMBEDDING_DEFAULT_ID,
        type="embedding",
        is_default=True,
        is_system_core=True,
        aliases=(
            "qwen3-embedding",
            "embedding",
            "mlx-community/bge-small-en-v1.5-mlx",
            "mlx-community/multilingual-e5-small-mlx",
            "bge-small-en",
            "multilingual-e5-small",
        ),
        variants={
            "mlx": Variant(
                repo_id="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                required=("config.json", "tokenizer*", "*.safetensors"),
                allow_patterns=None,
                approx_bytes=335 * MB,
                name="Qwen3 Embedding 0.6B 4-bit DWQ (MLX)",
                description="Motore Memoria RAG di sistema obbligatorio: architettura Qwen3 nativa MLX, 1024 dimensioni (~300 MB in RAM). Indicizzazione ad altissima velocità e ricerca ibrida su LanceDB.",
            ),
            "portable": Variant(
                repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
                required=("Qwen3-Embedding-0.6B-Q8_0.gguf",),
                allow_patterns=("Qwen3-Embedding-0.6B-Q8_0.gguf",),
                approx_bytes=610 * MB,
                model_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
                name="Qwen3 Embedding 0.6B (GGUF Q8_0, llama.cpp)",
                description="Motore Memoria RAG di sistema obbligatorio: Qwen3-Embedding 0.6B via llama.cpp, 1024 dimensioni (~610 MB). Stesso schema LanceDB di macOS.",
            ),
        },
    ),
}

MOCK_WHISPER_INFO = {
    "id": MOCK_WHISPER_ID,
    "name": "Modalità Simulatore (Mock)",
    "repos": [],
    "repo_id": "mock",
    "type": "whisper",
    "approx_size": "0 MB",
    "approx_bytes": 0,
    "description": "Nessun download, trascrizione istantanea simulata per test rapido.",
}


def resolve(model_id: Optional[str], kind: Optional[str] = None) -> Optional[ModelSpec]:
    """Risolve ID logico, repo id (di qualsiasi backend) o alias al ModelSpec."""
    if not model_id:
        if kind == "whisper":
            return CATALOG[WHISPER_CORE_ID]
        if kind == "llm":
            return CATALOG[LLM_DEFAULT_ID]
        if kind == "embedding":
            return CATALOG[EMBEDDING_DEFAULT_ID]
        return None
    mid = model_id.strip()
    low = mid.lower()
    for spec in CATALOG.values():
        if kind and spec.type != kind:
            continue
        if mid == spec.id or mid in spec.all_repo_ids() or low in (a.lower() for a in spec.aliases):
            return spec
    # Euristiche storiche (es. "whisper-large-v3-turbo", "...qwen3...embed...")
    if kind in (None, "whisper") and ("whisper" in low or "turbo" in low):
        return CATALOG[WHISPER_CORE_ID]
    if kind in (None, "embedding") and "qwen3" in low and "embed" in low:
        return CATALOG[EMBEDDING_DEFAULT_ID]
    return None


def human_size(nbytes: int) -> str:
    if nbytes >= 1024 * MB:
        return f"~{nbytes / (1024 * MB):.1f} GB"
    return f"~{int(round(nbytes / MB))} MB"


def info_dict(spec: ModelSpec) -> Dict[str, object]:
    """Descrizione JSON (formato storico di routers/models.py) per il backend corrente."""
    v = spec.variant
    return {
        "id": spec.id,
        "name": v.name,
        "repos": [v.repo_id],
        "repo_id": v.repo_id,
        "type": spec.type,
        "is_default": spec.is_default,
        "is_system_core": spec.is_system_core,
        "approx_size": human_size(v.approx_bytes),
        "approx_bytes": v.approx_bytes,
        "description": v.description,
        "backend": AI_BACKEND,
        "model_file": v.model_file,
    }


# ── Stato download in corso (un modello in download non è "scaricato") ────────
_downloading: set = set()
_downloading_lock = threading.Lock()


def set_downloading(model_id: str, active: bool) -> None:
    with _downloading_lock:
        if active:
            _downloading.add(model_id)
        else:
            _downloading.discard(model_id)


def is_downloading(model_id: str) -> bool:
    with _downloading_lock:
        return model_id in _downloading


def local_snapshot(spec: ModelSpec) -> Optional[str]:
    """Snapshot locale COMPLETA della variante corrente, oppure None."""
    v = spec.variant
    snap = get_local_snapshot_dir(v.repo_id)
    if snap and snapshot_is_complete(snap, v.required):
        return snap
    return None


def local_model_path(spec: ModelSpec) -> Optional[str]:
    """Percorso da passare al motore: file GGUF (portable) o cartella snapshot."""
    snap = local_snapshot(spec)
    if not snap:
        return None
    v = spec.variant
    if v.model_file:
        p = os.path.join(snap, v.model_file)
        return p if os.path.isfile(p) else None
    return snap


def is_downloaded(spec: ModelSpec) -> Tuple[bool, float]:
    """(scaricato_completo, dimensione_MB) per la variante del backend corrente."""
    if is_downloading(spec.id):
        return False, 0.0
    if local_snapshot(spec) is None:
        return False, 0.0
    return True, dir_size_mb(hf_repo_folder(spec.repo_id))


def all_known_repo_ids() -> List[str]:
    repos: List[str] = []
    for spec in CATALOG.values():
        for r in spec.all_repo_ids():
            if r not in repos:
                repos.append(r)
    return repos
