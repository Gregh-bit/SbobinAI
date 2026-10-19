"""
cleaner.py — SbobinAI Garbage Collector & Storage Optimizer
Scansione e bonifica fisica dei file orfani (audio, documenti, vettori e vecchie versioni del DB)
con compattazione SQLite (VACUUM) e LanceDB (optimize).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any, Dict, List, Set

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import (
    REPO_DATA_DIR,
    get_audio_dir,
    get_lancedb_dir,
    get_storage_path,
    get_uploads_dir,
)
from database import Lecture, Subject, engine

logger = logging.getLogger("sbobinai.cleaner")


def get_all_audio_directories() -> List[str]:
    """Restituisce tutte le cartelle note in cui possono trovarsi file audio."""
    dirs = [
        get_audio_dir(),
        os.path.join(REPO_DATA_DIR, "audio"),
        os.path.join(REPO_DATA_DIR, "storage", "audio"),
        os.path.expanduser("~/Documents/UniCoPilot/audio"),
        os.path.expanduser("~/Documents/SbobinAI/audio"),
    ]
    # Restituisce solo cartelle esistenti e uniche
    seen = set()
    valid_dirs = []
    for d in dirs:
        norm = os.path.abspath(os.path.normpath(d))
        if norm not in seen and os.path.exists(norm) and os.path.isdir(norm):
            seen.add(norm)
            valid_dirs.append(norm)
    return valid_dirs


def get_all_uploads_directories() -> List[str]:
    """Restituisce tutte le cartelle note in cui possono trovarsi materiali e upload."""
    dirs = [
        get_uploads_dir(),
        os.path.join(REPO_DATA_DIR, "uploads"),
        os.path.join(REPO_DATA_DIR, "storage", "uploads"),
        os.path.expanduser("~/Documents/UniCoPilot/uploads"),
        os.path.expanduser("~/Documents/SbobinAI/uploads"),
    ]
    seen = set()
    valid_dirs = []
    for d in dirs:
        norm = os.path.abspath(os.path.normpath(d))
        if norm not in seen and os.path.exists(norm) and os.path.isdir(norm):
            seen.add(norm)
            valid_dirs.append(norm)
    return valid_dirs


def vacuum_sqlite() -> bool:
    """Esegue VACUUM su SQLite per deframmentare il file e restituire i blocchi al filesystem."""
    try:
        with engine.connect() as conn:
            conn.execute(text("VACUUM;"))
            conn.commit()
        logger.info("[Cleaner] VACUUM SQLite eseguito con successo.")
        return True
    except Exception as exc:
        logger.warning(f"[Cleaner] Errore durante VACUUM SQLite: {exc}")
        return False


def optimize_lancedb() -> bool:
    """Compatta i file e le transazioni storiche in LanceDB."""
    try:
        from rag import get_lancedb_table
        tbl = get_lancedb_table()
        if hasattr(tbl, "optimize"):
            tbl.optimize()
            logger.info("[Cleaner] Ottimizzazione e compattazione LanceDB completata.")
            return True
    except Exception as exc:
        logger.warning(f"[Cleaner] LanceDB optimize non disponibile o non necessario: {exc}")
    return False


def cleanup_orphan_files(db: Session = None) -> Dict[str, Any]:
    """
    Spazzino principale del sistema:
    1. Interroga SQLite per tutti i percorsi audio validi e ID di soggetti/lezioni.
    2. Elimina tutti i file audio non referenziati o relativi a lezioni rimosse.
    3. Elimina le sottocartelle upload relative a materie inesistenti.
    4. Rimuove file .tmp, .part e vecchi database 'unicopilot.db' orfani.
    5. Esegue VACUUM SQLite e compattazione LanceDB.
    """
    from database import SessionLocal

    owns_session = False
    if db is None:
        db = SessionLocal()
        owns_session = True

    deleted_audio_files: List[str] = []
    deleted_upload_dirs: List[str] = []
    deleted_db_files: List[str] = []
    total_freed_bytes = 0

    try:
        # 1. Raccogli dati dal DB SQLite
        lectures = db.query(Lecture).all()
        subjects = db.query(Subject).all()

        valid_audio_paths: Set[str] = set()
        valid_audio_basenames: Set[str] = set()
        valid_lecture_ids: Set[int] = set()

        for lec in lectures:
            valid_lecture_ids.add(lec.id)
            if lec.audio_path:
                norm_p = os.path.abspath(os.path.normpath(lec.audio_path))
                valid_audio_paths.add(norm_p)
                valid_audio_basenames.add(os.path.basename(norm_p))

        valid_subject_ids: Set[int] = {s.id for s in subjects}

        logger.info(
            f"[Cleaner] Inizio scansione file orfani: {len(valid_lecture_ids)} lezioni attive, "
            f"{len(valid_subject_ids)} materie attive."
        )

        # 2. Bonifica file audio orfani
        audio_extensions = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".webm", ".flac"}
        audio_dirs = get_all_audio_directories()

        for adir in audio_dirs:
            try:
                for fname in os.listdir(adir):
                    fpath = os.path.join(adir, fname)
                    if not os.path.isfile(fpath):
                        continue

                    ext = os.path.splitext(fname)[1].lower()
                    if ext in audio_extensions or ext in {".tmp", ".part"}:
                        norm_fpath = os.path.abspath(os.path.normpath(fpath))
                        is_orphan = False

                        # Se è un file temporaneo/parziale, è orfano
                        if ext in {".tmp", ".part"}:
                            is_orphan = True
                        # Se il percorso e il nome non corrispondono ad alcuna lezione registrata
                        elif norm_fpath not in valid_audio_paths and fname not in valid_audio_basenames:
                            is_orphan = True

                        if is_orphan:
                            try:
                                sz = os.path.getsize(fpath)
                                os.remove(fpath)
                                total_freed_bytes += sz
                                deleted_audio_files.append(fpath)
                                logger.info(f"[Cleaner] 🗑️ File audio orfano rimosso: {fpath} ({sz / (1024*1024):.2f} MB)")
                            except Exception as rm_err:
                                logger.warning(f"[Cleaner] Impossibile rimuovere {fpath}: {rm_err}")
            except Exception as adir_err:
                logger.warning(f"[Cleaner] Errore lettura cartella audio {adir}: {adir_err}")

        # 3. Bonifica cartelle uploads orfane
        upload_dirs = get_all_uploads_directories()
        for udir in upload_dirs:
            try:
                for item in os.listdir(udir):
                    ipath = os.path.join(udir, item)
                    if not os.path.isdir(ipath):
                        # Pulisci file temporanei sparsi
                        if item.endswith(".tmp") or item.endswith(".part"):
                            try:
                                sz = os.path.getsize(ipath)
                                os.remove(ipath)
                                total_freed_bytes += sz
                            except Exception:
                                pass
                        continue

                    # Verifica se è una cartella di tipo subject_{id}
                    m = re.match(r"^subject_(\d+)$", item)
                    if m:
                        subj_id = int(m.group(1))
                        if subj_id not in valid_subject_ids:
                            try:
                                sz = sum(
                                    os.path.getsize(os.path.join(root, f))
                                    for root, _, files in os.walk(ipath)
                                    for f in files
                                    if os.path.isfile(os.path.join(root, f))
                                )
                                shutil.rmtree(ipath, ignore_errors=True)
                                total_freed_bytes += sz
                                deleted_upload_dirs.append(ipath)
                                logger.info(f"[Cleaner] 🗑️ Cartella uploads orfana rimossa: {ipath} ({sz / 1024:.1f} KB)")
                            except Exception as u_err:
                                logger.warning(f"[Cleaner] Errore rimozione cartella uploads {ipath}: {u_err}")
            except Exception as udir_err:
                logger.warning(f"[Cleaner] Errore scansione cartella uploads {udir}: {udir_err}")

        # 4. Bonifica vecchi database 'unicopilot.db' rimasti
        old_db_candidates = [
            os.path.join(get_storage_path(), "sqlite", "unicopilot.db"),
            os.path.join(REPO_DATA_DIR, "sqlite", "unicopilot.db"),
            os.path.expanduser("~/Documents/UniCoPilot/sqlite/unicopilot.db"),
        ]
        target_db = os.path.join(get_storage_path(), "sqlite", "sbobinai.db")
        if os.path.exists(target_db) and os.path.getsize(target_db) > 1024:
            for old_db in old_db_candidates:
                if os.path.exists(old_db) and os.path.isfile(old_db):
                    try:
                        sz = os.path.getsize(old_db)
                        os.remove(old_db)
                        total_freed_bytes += sz
                        deleted_db_files.append(old_db)
                        logger.info(f"[Cleaner] 🗑️ Vecchio DB orfano rimosso: {old_db} ({sz / 1024:.1f} KB)")
                    except Exception as db_err:
                        logger.warning(f"[Cleaner] Errore rimozione {old_db}: {db_err}")

        # 5. Compattazione database e vector store
        vac_ok = vacuum_sqlite()
        opt_ok = optimize_lancedb()

    finally:
        if owns_session:
            db.close()

    freed_mb = round(total_freed_bytes / (1024 * 1024), 2)
    logger.info(
        f"[Cleaner] ✅ Pulizia completata: {len(deleted_audio_files)} audio rimossi, "
        f"{len(deleted_upload_dirs)} cartelle rimosse, {freed_mb} MB liberati sul disco."
    )

    return {
        "status": "success",
        "freed_bytes": total_freed_bytes,
        "freed_mb": freed_mb,
        "deleted_audio_files": deleted_audio_files,
        "deleted_upload_dirs": deleted_upload_dirs,
        "deleted_db_files": deleted_db_files,
        "vacuum_executed": vac_ok,
        "lancedb_optimized": opt_ok,
        "message": f"Pulizia completata con successo. Liberati {freed_mb} MB di spazio sul disco.",
    }


def get_storage_breakdown() -> Dict[str, Any]:
    """Calcola lo spazio occupato in MB da audio, uploads, database e vettori."""
    def _dir_size(d: str) -> float:
        if not os.path.exists(d):
            return 0.0
        total = 0
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if os.path.isfile(p):
                        total += os.path.getsize(p)
                except Exception:
                    pass
        return round(total / (1024 * 1024), 2)

    audio_mb = sum(_dir_size(d) for d in get_all_audio_directories())
    uploads_mb = sum(_dir_size(d) for d in get_all_uploads_directories())
    sqlite_mb = _dir_size(os.path.join(get_storage_path(), "sqlite"))
    lancedb_mb = _dir_size(get_lancedb_dir())
    total_mb = round(audio_mb + uploads_mb + sqlite_mb + lancedb_mb, 2)

    return {
        "audio_mb": round(audio_mb, 2),
        "uploads_mb": round(uploads_mb, 2),
        "sqlite_mb": round(sqlite_mb, 2),
        "lancedb_mb": round(lancedb_mb, 2),
        "total_mb": total_mb,
    }

