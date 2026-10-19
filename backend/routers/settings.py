"""
routers/settings.py — SbobinAI Settings & Storage API
Gestisce la configurazione del percorso di archiviazione dati
e la sincronizzazione delle preferenze tra frontend e backend.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from config import get_storage_path, load_settings, save_settings, update_storage_path

router = APIRouter(prefix="/settings", tags=["Settings"])


class StoragePathRequest(BaseModel):
    storage_path: str


class SettingsUpdateRequest(BaseModel):
    storage_path: Optional[str] = None
    whisper_model: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    ollama_model: Optional[str] = None  # Retrocompatibilità
    vad_sensitivity: Optional[int] = None
    models_downloaded: Optional[bool] = None


@router.get("/")
def get_current_settings():
    """Restituisce le impostazioni correnti del backend verificando lo stato reale su disco."""
    current = load_settings()
    try:
        from routers.models import WHISPER_MODELS, is_whisper_downloaded
        curr_whisper = current.get("whisper_model", "large-v3-turbo-q4")
        if curr_whisper in ("dual-tier-whisper", "large-v3-turbo", "base-mlx", "small-mlx"):
            curr_whisper = "large-v3-turbo-q4"
            current["whisper_model"] = "large-v3-turbo-q4"
        if curr_whisper in WHISPER_MODELS and curr_whisper != "mock_mode":
            is_dl, _ = is_whisper_downloaded(WHISPER_MODELS[curr_whisper])
            if is_dl:
                current["models_downloaded"] = True
            save_settings(current)
    except Exception:
        pass
    return current


@router.post("/storage")
def set_storage_path(payload: StoragePathRequest):
    """
    Riceve il percorso di archiviazione scelto dall'utente,
    aggiorna settings.json, migra il DB e crea le sottocartelle (sqlite, audio, uploads).
    """
    path = payload.storage_path.strip()
    if not path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Il percorso di archiviazione non può essere vuoto."
        )

    try:
        result = update_storage_path(path, migrate_data=True)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante l'aggiornamento del percorso di archiviazione: {exc}"
        )


@router.patch("/")
def update_settings(payload: SettingsUpdateRequest):
    """Aggiorna parzialmente le impostazioni nel backend."""
    current = load_settings()
    data = payload.model_dump(exclude_unset=True)

    if "storage_path" in data and data["storage_path"]:
        update_storage_path(data["storage_path"], migrate_data=True)

    for k, v in data.items():
        if k != "storage_path":
            current[k] = v

    # Se un modello Whisper reale è fisicamente presente su disco, models_downloaded deve rimanere True
    try:
        from routers.models import WHISPER_MODELS, is_whisper_downloaded
        curr_whisper = current.get("whisper_model", "large-v3-turbo-q4")
        if curr_whisper in ("dual-tier-whisper", "large-v3-turbo", "base-mlx", "small-mlx"):
            curr_whisper = "large-v3-turbo-q4"
            current["whisper_model"] = "large-v3-turbo-q4"
        if curr_whisper in WHISPER_MODELS and curr_whisper != "mock_mode":
            is_dl, _ = is_whisper_downloaded(WHISPER_MODELS[curr_whisper])
            if is_dl:
                current["models_downloaded"] = True
    except Exception:
        pass

    saved = save_settings(current)
    return saved


@router.post("/clean-orphans")
def trigger_clean_orphans():
    """
    Esegue manualmente la bonifica dei file orfani dal disco,
    eliminando registrazioni non collegate al DB, cartelle obsolete e compattando SQLite e LanceDB.
    """
    try:
        from cleaner import cleanup_orphan_files
        res = cleanup_orphan_files()
        return res
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore durante la pulizia dei file orfani: {exc}"
        )


@router.get("/storage-stats")
def get_storage_stats():
    """Restituisce il dettaglio dello spazio su disco occupato da audio, uploads, SQLite e vettori."""
    try:
        from cleaner import get_storage_breakdown
        return get_storage_breakdown()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore calcolo statistiche storage: {exc}"
        )

