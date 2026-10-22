"""
config.py — SbobinAI Configuration & Storage Management
Gestisce la persistenza delle impostazioni locali in settings.json
e calcola dinamicamente i percorsi di archiviazione per SQLite, Audio e Uploads.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("sbobinai.config")


def _documents_dir() -> str:
    """
    Cartella Documenti dell'utente. Su Windows usa la Known Folder di sistema,
    che può essere reindirizzata (es. OneDrive/Documents) e non coincidere con ~/Documents.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            import uuid

            class _GUID(ctypes.Structure):
                _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                            ("Data3", wintypes.WORD), ("Data4", wintypes.BYTE * 8)]

            u = uuid.UUID("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")  # FOLDERID_Documents
            guid = _GUID(u.fields[0], u.fields[1], u.fields[2], (wintypes.BYTE * 8)(*u.bytes[8:]))
            out = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(out)) == 0:
                path = out.value
                ctypes.windll.ole32.CoTaskMemFree(out)
                if path:
                    return path
        except Exception:
            pass
    return os.path.expanduser(os.path.join("~", "Documents"))


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_default_settings_file() -> str:
    env_file = os.environ.get("SBOBINAI_SETTINGS_FILE")
    if env_file:
        return env_file
    local_file = os.path.join(BASE_DIR, "settings.json")
    if os.path.exists(local_file) and os.access(local_file, os.W_OK):
        return local_file
    if not os.path.exists(local_file) and os.access(BASE_DIR, os.W_OK):
        return local_file
    # Fallback su cartella dati applicazione utente (evita PermissionError in /Applications)
    if sys.platform == "darwin":
        app_dir = os.path.expanduser("~/Library/Application Support/com.sbobinai.app")
    elif sys.platform == "win32":
        app_dir = os.path.expandvars(r"%LOCALAPPDATA%\SbobinAI")
    else:
        app_dir = os.path.expanduser("~/.sbobinai")
    try:
        os.makedirs(app_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(app_dir, "settings.json")

SETTINGS_FILE = _get_default_settings_file()
DOCUMENTS_DIR = _documents_dir()
DEFAULT_STORAGE_ROOT = os.path.join(DOCUMENTS_DIR, "SbobinAI")
LEGACY_STORAGE_ROOT = os.path.join(DOCUMENTS_DIR, "UniCoPilot")
REPO_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

DEFAULT_SETTINGS: Dict[str, Any] = {
    "storage_path": DEFAULT_STORAGE_ROOT,
    "whisper_model": "large-v3-turbo-q4",
    "llm_model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "embedding_model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    "vad_sensitivity": 2,
    "models_downloaded": False,
    "audio_format": "m4a",
}


# Serializza letture/scritture di settings.json tra i thread del server
SETTINGS_LOCK = threading.RLock()


def load_settings() -> Dict[str, Any]:
    """Carica le impostazioni da settings.json o crea il file con valori di default."""
    with SETTINGS_LOCK:
        if not os.path.exists(SETTINGS_FILE):
            save_settings(DEFAULT_SETTINGS)
            return dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"[Config] Errore lettura {SETTINGS_FILE}: {e}. Uso defaults.")
            return dict(DEFAULT_SETTINGS)

    try:
        if not isinstance(data, dict):
            raise ValueError("settings.json non contiene un oggetto JSON")
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        # Migrazione da vecchio ollama_model a llm_model puro MLX
        if "ollama_model" in merged:
            if "llm_model" not in data:
                merged["llm_model"] = "mlx-community/Qwen3.5-4B-MLX-4bit"
            merged.pop("ollama_model", None)
        # Migrazione modello embedding a Qwen3-Embedding-0.6B-4bit-DWQ (1024 dim)
        if merged.get("embedding_model") in (
            "mlx-community/bge-small-en-v1.5-mlx",
            "mlx-community/multilingual-e5-small-mlx",
            "bge-small-en",
            "multilingual-e5-small",
        ):
            merged["embedding_model"] = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        # Normalizza storage_path e migra da UniCoPilot a SbobinAI
        if "storage_path" in merged and merged["storage_path"]:
            sp = os.path.abspath(os.path.expanduser(merged["storage_path"]))
            if _same_path(sp, LEGACY_STORAGE_ROOT):
                sp = DEFAULT_STORAGE_ROOT
            merged["storage_path"] = sp
        # Validazione formato di registrazione audio (default "m4a" compresso AAC)
        if merged.get("audio_format") not in ("m4a", "wav"):
            merged["audio_format"] = "m4a"
        return merged
    except Exception as e:
        logger.error(f"[Config] Errore lettura {SETTINGS_FILE}: {e}. Uso defaults.")
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Salva le impostazioni nel file settings.json (scrittura atomica protetta da lock)."""
    with SETTINGS_LOCK:
        tmp = SETTINGS_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, SETTINGS_FILE)
            logger.info(f"[Config] Impostazioni salvate in {SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"[Config] Impossibile salvare {SETTINGS_FILE}: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return settings


def update_settings(changes: Dict[str, Any]) -> Dict[str, Any]:
    """Applica `changes` a settings.json in modo atomico (load → merge → save sotto lock)."""
    with SETTINGS_LOCK:
        current = load_settings()
        current.update(changes)
        save_settings(current)
        return current


_active_storage_root: Optional[str] = None


def get_storage_path() -> str:
    """Restituisce il percorso root di archiviazione configurato dall'utente."""
    global _active_storage_root
    if _active_storage_root:
        return _active_storage_root

    settings = load_settings()
    path = settings.get("storage_path") or DEFAULT_STORAGE_ROOT
    expanded = os.path.abspath(os.path.expanduser(path))

    # Se punta a SbobinAI ma non esiste e UniCoPilot esiste, migra la directory
    if _same_path(expanded, DEFAULT_STORAGE_ROOT) and not os.path.exists(expanded):
        if os.path.exists(LEGACY_STORAGE_ROOT):
            try:
                shutil.copytree(LEGACY_STORAGE_ROOT, expanded)
                logger.info(f"[Config] Cartella legacy migrata con successo: {LEGACY_STORAGE_ROOT} -> {expanded}")
            except Exception as e:
                logger.warning(f"[Config] Errore migrazione automatica cartella legacy {LEGACY_STORAGE_ROOT}: {e}")

    # Prova di accesso
    try:
        os.makedirs(expanded, exist_ok=True)
        _active_storage_root = expanded
        return expanded
    except (PermissionError, OSError) as e:
        logger.warning(f"[Config] Cartella '{expanded}' non accessibile ({e}). Fallback in /data/storage.")
        fallback = os.path.join(REPO_DATA_DIR, "storage")
        try:
            os.makedirs(fallback, exist_ok=True)
        except Exception:
            pass
        _active_storage_root = fallback
        return fallback


def get_sqlite_dir() -> str:
    """Directory contenente il database sbobinai.db."""
    return os.path.join(get_storage_path(), "sqlite")


def get_audio_dir() -> str:
    """Directory contenente le registrazioni audio .wav."""
    return os.path.join(get_storage_path(), "audio")


def get_uploads_dir() -> str:
    """Directory contenente i materiali, PDF e appunti del corso."""
    return os.path.join(get_storage_path(), "uploads")


def get_lancedb_dir() -> str:
    """Directory contenente il vector database LanceDB per il RAG."""
    return os.path.join(get_storage_path(), "lancedb")


def get_db_path() -> str:
    """Percorso assoluto del file SQLite sbobinai.db."""
    return os.path.join(get_sqlite_dir(), "sbobinai.db")


def ensure_storage_dirs() -> Dict[str, str]:
    """
    Assicura che la root di archiviazione e tutte le sottocartelle
    (sqlite, audio, uploads, lancedb) esistano fisicamente sul disco.
    """
    root = get_storage_path()
    sqlite_dir = get_sqlite_dir()
    audio_dir = get_audio_dir()
    uploads_dir = get_uploads_dir()
    lancedb_dir = get_lancedb_dir()

    os.makedirs(root, exist_ok=True)
    os.makedirs(sqlite_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(lancedb_dir, exist_ok=True)

    # Gestione file database sbobinai.db con migrazione trasparente da unicopilot.db
    target_db = os.path.join(sqlite_dir, "sbobinai.db")
    if not os.path.exists(target_db):
        old_local_db = os.path.join(sqlite_dir, "unicopilot.db")
        if os.path.exists(old_local_db):
            try:
                shutil.copy2(old_local_db, target_db)
                logger.info(f"[Config] DB migrato da {old_local_db} -> {target_db}")
            except Exception as e:
                logger.warning(f"[Config] Impossibile copiare DB locale: {e}")
        else:
            legacy_doc_db = os.path.join(LEGACY_STORAGE_ROOT, "sqlite", "unicopilot.db")
            if os.path.exists(legacy_doc_db):
                try:
                    shutil.copy2(legacy_doc_db, target_db)
                    logger.info(f"[Config] DB migrato da {legacy_doc_db} -> {target_db}")
                except Exception as e:
                    logger.warning(f"[Config] Impossibile copiare DB legacy: {e}")
            else:
                old_repo_db = os.path.join(REPO_DATA_DIR, "sqlite", "unicopilot.db")
                if os.path.exists(old_repo_db):
                    try:
                        shutil.copy2(old_repo_db, target_db)
                        logger.info(f"[Config] DB iniziale copiato da {old_repo_db} -> {target_db}")
                    except Exception as e:
                        logger.warning(f"[Config] Impossibile copiare DB iniziale: {e}")

    return {
        "root": root,
        "sqlite": sqlite_dir,
        "audio": audio_dir,
        "uploads": uploads_dir,
        "lancedb": lancedb_dir,
        "db_path": target_db,
    }


def update_storage_path(new_path: str, migrate_data: bool = True) -> Dict[str, Any]:
    """
    Aggiorna il percorso di archiviazione scelto dall'utente.
    Se richiesto, sposta il database e i file audio/uploads esistenti nella nuova directory.
    """
    global _active_storage_root
    old_root = get_storage_path()
    expanded_new = os.path.abspath(os.path.expanduser(new_path))

    if _same_path(old_root, expanded_new):
        dirs = ensure_storage_dirs()
        return {"status": "unchanged", "storage_path": expanded_new, "dirs": dirs}

    logger.info(f"[Config] Migrazione storage: {old_root} -> {expanded_new}")

    # Assicura le nuove directory con gestione permessi
    try:
        new_sqlite = os.path.join(expanded_new, "sqlite")
        new_audio = os.path.join(expanded_new, "audio")
        new_uploads = os.path.join(expanded_new, "uploads")
        new_lancedb = os.path.join(expanded_new, "lancedb")
        os.makedirs(new_sqlite, exist_ok=True)
        os.makedirs(new_audio, exist_ok=True)
        os.makedirs(new_uploads, exist_ok=True)
        os.makedirs(new_lancedb, exist_ok=True)
        _active_storage_root = expanded_new
    except (PermissionError, OSError) as e:
        logger.warning(f"[Config] Cartella '{expanded_new}' non accessibile ({e}). Uso fallback in data/storage.")
        expanded_new = os.path.join(REPO_DATA_DIR, "storage")
        new_sqlite = os.path.join(expanded_new, "sqlite")
        new_audio = os.path.join(expanded_new, "audio")
        new_uploads = os.path.join(expanded_new, "uploads")
        new_lancedb = os.path.join(expanded_new, "lancedb")
        os.makedirs(new_sqlite, exist_ok=True)
        os.makedirs(new_audio, exist_ok=True)
        os.makedirs(new_uploads, exist_ok=True)
        os.makedirs(new_lancedb, exist_ok=True)
        _active_storage_root = expanded_new

    # Migrazione se abilitata
    if migrate_data and os.path.exists(old_root):
        # 1. DB SQLite
        old_db = os.path.join(old_root, "sqlite", "sbobinai.db")
        if not os.path.exists(old_db):
            old_db = os.path.join(old_root, "sqlite", "unicopilot.db")
        new_db = os.path.join(new_sqlite, "sbobinai.db")
        if os.path.exists(old_db) and not os.path.exists(new_db):
            try:
                shutil.copy2(old_db, new_db)
                logger.info(f"[Config] Database spostato in {new_db}")
            except Exception as e:
                logger.error(f"[Config] Errore copia DB: {e}")

        # 2. Cartella Audio
        old_audio = os.path.join(old_root, "audio")
        if os.path.exists(old_audio):
            for fname in os.listdir(old_audio):
                src = os.path.join(old_audio, fname)
                dst = os.path.join(new_audio, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        logger.warning(f"[Config] Errore copia file audio {fname}: {e}")

        # 3. Cartella Uploads
        old_uploads = os.path.join(old_root, "uploads")
        if os.path.exists(old_uploads):
            for item in os.listdir(old_uploads):
                src = os.path.join(old_uploads, item)
                dst = os.path.join(new_uploads, item)
                if not os.path.exists(dst):
                    try:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                    except Exception as e:
                        logger.warning(f"[Config] Errore copia uploads {item}: {e}")

        # 4. Cartella LanceDB
        old_lancedb = os.path.join(old_root, "lancedb")
        if os.path.exists(old_lancedb):
            for item in os.listdir(old_lancedb):
                src = os.path.join(old_lancedb, item)
                dst = os.path.join(new_lancedb, item)
                if not os.path.exists(dst):
                    try:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                    except Exception as e:
                        logger.warning(f"[Config] Errore copia lancedb {item}: {e}")

    # Salva le nuove impostazioni
    update_settings({"storage_path": expanded_new})

    # Re-bind del database SQLAlchemy
    try:
        from database import rebind_database, SessionLocal, Lecture
        rebind_database()
        # Aggiorna audio_path nel database per puntare alla nuova cartella
        if migrate_data:
            with SessionLocal() as db_mig:
                lectures = db_mig.query(Lecture).all()
                updated_count = 0
                old_audio_norm = os.path.normcase(os.path.abspath(old_audio))
                for lec in lectures:
                    if not lec.audio_path:
                        continue
                    fname = os.path.basename(lec.audio_path)
                    potential_new = os.path.join(new_audio, fname)
                    lec_norm = os.path.normcase(os.path.abspath(lec.audio_path))
                    # Aggiorna tutti i record che puntavano a old_audio o il cui file si trova nella nuova cartella
                    if (lec_norm.startswith(old_audio_norm) or os.path.exists(potential_new)) and lec.audio_path != potential_new:
                        lec.audio_path = potential_new
                        updated_count += 1
                if updated_count > 0:
                    db_mig.commit()
                    logger.info(f"[Config] Aggiornati {updated_count} percorsi audio nel database.")
    except Exception as e:
        logger.warning(f"[Config] Rebind database / migrazione audio_path: {e}")

    # Riapre LanceDB sulla nuova cartella
    try:
        import rag
        if hasattr(rag, "reset_connection"):
            rag.reset_connection()
    except Exception as e:
        logger.warning(f"[Config] Reset connessione LanceDB: {e}")

    return {
        "status": "success",
        "storage_path": expanded_new,
        "dirs": {
            "root": expanded_new,
            "sqlite": new_sqlite,
            "audio": new_audio,
            "uploads": new_uploads,
            "lancedb": new_lancedb,
            "db_path": os.path.join(new_sqlite, "sbobinai.db"),
        }
    }
