"""
routers/subjects.py — SbobinAI
Endpoint REST per la gestione delle Materie e Materiali del corso.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import Bookmark, Lecture, Subject, get_db
from schemas import MaterialOut, SubjectCreate, SubjectOut, SubjectUpdate
from rag import (
    index_material_document,
    delete_material_vectors,
    delete_subject_vectors,
    compact_lancedb,
    clear_metal_cache,
)
from config import get_uploads_dir

logger = logging.getLogger("sbobinai.subjects")

router = APIRouter(prefix="/subjects", tags=["Subjects"])

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".ogg")


def _safe_filename(raw: str | None) -> str:
    """
    Riduce un nome file fornito dal client al solo basename (niente '../', percorsi
    assoluti o separatori Windows/Unix), rifiutando null-byte e nomi riservati.
    """
    if raw is None or "\x00" in raw:
        raise HTTPException(status_code=400, detail="Nome file non valido.")
    name = os.path.basename(raw.replace("\\", "/")).strip()
    if not name or name in (".", "..") or name.startswith(".") or name == "InfoCorso.md" or name.endswith(".meta.json"):
        raise HTTPException(status_code=400, detail="Nome file non valido.")
    return name



def ensure_infocorso_file(subject: Subject) -> str:
    """Crea o aggiorna automaticamente il file InfoCorso.md su disco."""
    subj_dir = os.path.join(get_uploads_dir(), f"subject_{subject.id}")
    os.makedirs(subj_dir, exist_ok=True)
    file_path = os.path.join(subj_dir, "InfoCorso.md")

    date_str = subject.created_at.strftime('%d/%m/%Y') if subject.created_at else ''
    content = f"""# {subject.name} — Informazioni del Corso

## Informazioni Generali & Descrizione
{subject.description or "Nessuna descrizione o informazione aggiuntiva specificata."}

## Parole Chiave Riconoscimento Vocale (Whisper)
{subject.whisper_prompt_keywords or "Nessuna parola chiave definita."}

## Dettagli di Registrazione
- **Materia:** {subject.name}
- **Data creazione:** {date_str}
- **Stato:** Attivo nel sistema SbobinAI
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


@router.get("/", response_model=List[SubjectOut])
def list_subjects(db: Session = Depends(get_db)):
    """Restituisce tutte le materie ordinate per nome (ottimizzata con singola query SQL aggregata)."""
    counts = (
        db.query(Lecture.subject_id, func.count(Lecture.id).label("count"))
        .group_by(Lecture.subject_id)
        .all()
    )
    count_map = {c[0]: c[1] for c in counts}

    subjects = db.query(Subject).order_by(Subject.name).all()
    result = []
    for s in subjects:
        out = SubjectOut.model_validate(s)
        out.lecture_count = count_map.get(s.id, 0)
        result.append(out)
    return result


@router.post("/", response_model=SubjectOut, status_code=status.HTTP_201_CREATED)
def create_subject(payload: SubjectCreate, db: Session = Depends(get_db)):
    """Crea una nuova materia e genera subito InfoCorso.md."""
    existing = db.query(Subject).filter(Subject.name == payload.name).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Materia '{payload.name}' già esistente.")
    subject = Subject(**payload.model_dump())
    db.add(subject)
    db.commit()
    db.refresh(subject)

    # Genera automaticamente il file InfoCorso.md
    ensure_infocorso_file(subject)

    out = SubjectOut.model_validate(subject)
    out.lecture_count = 0
    return out


@router.get("/{subject_id}", response_model=SubjectOut)
def get_subject(subject_id: int, db: Session = Depends(get_db)):
    """Dettaglio di una materia."""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")
    out = SubjectOut.model_validate(subject)
    out.lecture_count = len(subject.lectures)
    return out


@router.patch("/{subject_id}", response_model=SubjectOut)
def update_subject(subject_id: int, payload: SubjectUpdate, db: Session = Depends(get_db)):
    """Aggiorna i campi di una materia e aggiorna InfoCorso.md."""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")
    changes = payload.model_dump(exclude_unset=True)
    # name e color non possono essere null; description/keywords null = svuota il campo
    for required in ("name", "color"):
        if required in changes and changes[required] is None:
            changes.pop(required)
    if "name" in changes:
        clash = db.query(Subject).filter(Subject.name == changes["name"], Subject.id != subject_id).first()
        if clash:
            raise HTTPException(status_code=409, detail=f"Materia '{changes['name']}' già esistente.")
    for field, value in changes.items():
        setattr(subject, field, value)
    db.commit()
    db.refresh(subject)

    # Aggiorna InfoCorso.md con i nuovi dati
    ensure_infocorso_file(subject)

    out = SubjectOut.model_validate(subject)
    out.lecture_count = len(subject.lectures)
    return out


@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(subject_id: int, db: Session = Depends(get_db)):
    """
    Elimina fisicamente una materia con cascata totale:
    1. Tutti i file audio (.wav) delle lezioni dal disco.
    2. Tutti i materiali e la cartella uploads/subject_{id}.
    3. Tutti i vettori LanceDB associati alla materia.
    4. Tutti i record SQLite (segnalibri, lezioni, materia) con VACUUM finale.
    """
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    # 1. Elimina file audio delle lezioni associate
    from cleaner import get_all_audio_directories, get_all_uploads_directories, vacuum_sqlite
    lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).all()
    audio_dirs = get_all_audio_directories()

    for lec in lectures:
        # File audio registrato
        if lec.audio_path and os.path.exists(lec.audio_path) and os.path.isfile(lec.audio_path):
            try:
                os.remove(lec.audio_path)
            except Exception:
                pass

        # Cerca file associati a questo ID in tutte le directory audio
        prefix = f"lecture_{lec.id}_"
        prefix_alt = f"lecture_{lec.id}."
        for adir in audio_dirs:
            if os.path.exists(adir):
                for fname in os.listdir(adir):
                    if fname.startswith(prefix) or fname.startswith(prefix_alt):
                        fp = os.path.join(adir, fname)
                        if os.path.isfile(fp):
                            try:
                                os.remove(fp)
                            except Exception:
                                pass

        # Elimina segnalibri
        db.query(Bookmark).filter(Bookmark.lecture_id == lec.id).delete()

    # 2. Elimina tutte le cartelle uploads/subject_{id}
    folder_name = f"subject_{subject_id}"
    for udir in get_all_uploads_directories():
        sdir = os.path.join(udir, folder_name)
        if os.path.exists(sdir):
            try:
                shutil.rmtree(sdir, ignore_errors=True)
            except Exception:
                pass

    # 3. Elimina tutti i vettori da LanceDB
    delete_subject_vectors(subject_id=subject_id)
    compact_lancedb()

    # 4. Elimina la materia da SQLite
    db.delete(subject)
    db.commit()

    # 5. Compattazione SQLite
    vacuum_sqlite()


@router.get("/{subject_id}/materials", response_model=List[MaterialOut])
def list_materials(subject_id: int, db: Session = Depends(get_db)):
    """Restituisce l'elenco dei materiali del corso, con InfoCorso.md auto-importato."""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    content = ensure_infocorso_file(subject)
    subj_dir = os.path.join(get_uploads_dir(), f"subject_{subject.id}")

    materials: List[MaterialOut] = []
    # 1. InfoCorso.md sempre presente in cima
    size_kb = len(content.encode("utf-8")) / 1024
    materials.append(MaterialOut(
        id=f"infocorso-{subject.id}",
        name="InfoCorso.md",
        type="doc",
        size=f"{size_kb:.1f} KB",
        uploadedAt="Auto-importato",
        pageCount=1,
        content=content,
        isAutoImported=True,
    ))

    # 2. Tutte le lezioni / registrazioni della memoria del corso
    from database import Lecture
    lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).order_by(Lecture.date.desc()).all()
    for lec in lectures:
        dur_str = f"{int(lec.duration_seconds // 60)}m" if lec.duration_seconds else "Audio / Appunti"
        date_str = lec.date.strftime('%d %b %Y') if lec.date else "Lezione"
        materials.append(MaterialOut(
            id=f"lecture-{lec.id}",
            name=lec.title,
            type="lecture",
            size=dur_str,
            uploadedAt=date_str,
            pageCount=1,
            content=lec.processed_notes or lec.raw_transcript,
            isAutoImported=False,
            lecture_id=lec.id,
            duration_seconds=lec.duration_seconds,
            in_quaderno=lec.in_quaderno,
            source_type=lec.source_type,
        ))

    # 3. Eventuali altri file fisici caricati (PDF, slide, note, audio)
    if os.path.exists(subj_dir):
        for fname in sorted(os.listdir(subj_dir)):
            if fname == "InfoCorso.md" or fname.startswith(".") or fname.endswith(".meta.json"):
                continue
            fpath = os.path.join(subj_dir, fname)
            if os.path.isfile(fpath):
                f_kb = os.path.getsize(fpath) / 1024
                size_str = f"{f_kb / 1024:.1f} MB" if f_kb > 1024 else f"{f_kb:.1f} KB"
                ftype = (
                    "slide" if fname.lower().endswith((".ppt", ".pptx", ".key"))
                    else "pdf" if fname.lower().endswith(".pdf")
                    else "audio" if fname.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg"))
                    else "doc"
                )

                page_count = 1
                doc_content = None

                # Prova a leggere metadati salvati
                meta_path = f"{fpath}.meta.json"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            m_data = json.load(mf)
                            page_count = m_data.get("page_count", 1)
                            doc_content = m_data.get("content")
                    except Exception:
                        pass
                elif fname.lower().endswith(".pdf"):
                    try:
                        import pymupdf
                        pdoc = pymupdf.open(fpath)
                        page_count = len(pdoc)
                        pdoc.close()
                    except Exception:
                        pass
                elif fname.lower().endswith((".txt", ".md")):
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                            doc_content = tf.read(4000)
                    except Exception:
                        pass

                materials.append(MaterialOut(
                    id=fname,
                    name=fname,
                    type=ftype,
                    size=size_str,
                    uploadedAt="Caricato",
                    pageCount=page_count,
                    content=doc_content,
                    isAutoImported=False,
                    lecture_id=None,
                    duration_seconds=None,
                    in_quaderno=None,
                    source_type=None,
                ))

    return materials


def _process_imported_audio_bg(subject_id: int, audio_path: str, orig_filename: str):
    """
    Task in background su thread separato:
    1. Trascrive l'audio esterno con mlx-whisper (Large-v3-Turbo Q4 su Metal).
    2. Calcola la durata effettiva.
    3. Genera il capitolo di studio con Qwen 3.5 4B (se scaricato).
    4. Crea la Lezione nel database SQLite ('imported', in_quaderno=True).
    5. Indicizza trascrizione e appunti in LanceDB.
    """
    try:
        # Non contendere GPU/memoria con una lezione live in corso: attendi che finisca
        from live_session import has_active_sessions
        while has_active_sessions():
            time.sleep(10)

        logger.info(f"[Audio Ingest] Inizio trascrizione Whisper per: {orig_filename} ({audio_path})")
        # Recupera eventuali keywords tecniche configurate per la materia
        from database import SessionLocal, Subject, Lecture
        keywords = None
        with SessionLocal() as db_tmp:
            subj = db_tmp.query(Subject).filter(Subject.id == subject_id).first()
            if subj and subj.whisper_prompt_keywords:
                keywords = subj.whisper_prompt_keywords.strip()

        from transcriber import transcribe_file
        transcript, duration_sec = transcribe_file(audio_path, language="it", initial_prompt=keywords)
        transcript = (transcript or "").strip()

        clean_title = os.path.splitext(orig_filename)[0].replace("_", " ").strip()
        clean_title = clean_title[0].upper() + clean_title[1:] if clean_title else "Lezione Importata"

        # Genera schema e appunti strutturati con Qwen 3.5 se presente
        notes = None
        from llm_engine import generate_lecture_chapter, is_model_downloaded, DEFAULT_LLM_MODEL
        from config import load_settings
        llm_model = load_settings().get("llm_model") or DEFAULT_LLM_MODEL
        if is_model_downloaded(llm_model) and transcript:
            try:
                notes = asyncio.run(
                    generate_lecture_chapter(title=clean_title, transcript=transcript, model_id=llm_model)
                )
            except Exception as e:
                logger.warning(f"[Audio Ingest] Sintesi automatica non riuscita: {e}")

        if not notes:
            notes = (
                f"# {clean_title} — Appunti di Lezione\n\n"
                f"> [!NOTE]\n> Registrazione esterna importata nella memoria del corso.\n\n"
                f"## Trascrizione Integrale\n{transcript if transcript else 'Trascrizione non disponibile o audio privo di parlato.'}"
            )

        # Salva in SQLite
        from database import SessionLocal, Lecture
        lec_id = None
        with SessionLocal() as session:
            new_lec = Lecture(
                subject_id=subject_id,
                title=clean_title,
                raw_transcript=transcript,
                processed_notes=notes,
                audio_path=audio_path,
                duration_seconds=duration_sec,
                source_type="imported",
                in_quaderno=True,
            )
            session.add(new_lec)
            session.commit()
            session.refresh(new_lec)
            lec_id = new_lec.id

        # Indicizza in LanceDB
        if lec_id:
            from rag import index_lecture
            index_lecture(
                subject_id=subject_id,
                lecture_id=lec_id,
                title=clean_title,
                transcript=transcript,
                notes=notes
            )
            # Pre-calcola e salva in cache i timestamp parola per parola
            try:
                from routers.lectures import compute_and_cache_lecture_timestamps
                compute_and_cache_lecture_timestamps(lec_id)
            except Exception as e_ts:
                logger.warning(f"[Audio Ingest] Pre-calcolo timestamp fallito: {e_ts}")

        clear_metal_cache()
        logger.info(f"[Audio Ingest] [OK] Completato con successo! Lezione '{clean_title}' (ID {lec_id}) salvata e indicizzata.")
    except Exception as exc:
        logger.error(f"[Audio Ingest] [ERRORE] Errore durante l'elaborazione dell'audio '{orig_filename}': {exc}", exc_info=True)
        clear_metal_cache()


@router.post("/{subject_id}/materials/upload", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
def upload_material_file(
    subject_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Carica un file (PDF, Slide, Appunti, Audio) nella memoria della materia:
    - Per PDF/Testi: estrae il testo pagina per pagina con PyMuPDF e indicizza subito in LanceDB.
    - Per Audio (.wav, .mp3, .m4a): lancia un worker in background che trascrive con Whisper e genera il Quaderno.
    Handler sincrono: FastAPI lo esegue nel threadpool, così copia, estrazione PDF ed embedding
    non bloccano l'event loop (e quindi la lezione live via WebSocket).
    """
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    upload_root = Path(get_uploads_dir()).resolve()
    subj_dir_path = (upload_root / f"subject_{subject_id}").resolve()
    subj_dir_path.mkdir(parents=True, exist_ok=True)

    filename = _safe_filename(file.filename)
    fpath_obj = (subj_dir_path / filename).resolve()
    if not fpath_obj.is_relative_to(subj_dir_path):
        raise HTTPException(status_code=400, detail="Percorso non consentito.")
    fpath = str(fpath_obj)

    with open(fpath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    f_kb = os.path.getsize(fpath) / 1024
    size_str = f"{f_kb / 1024:.1f} MB" if f_kb > 1024 else f"{f_kb:.1f} KB"
    ext = os.path.splitext(filename)[1].lower()

    ftype = (
        "slide" if ext in (".ppt", ".pptx", ".key")
        else "pdf" if ext == ".pdf"
        else "audio" if ext in AUDIO_EXTENSIONS
        else "doc"
    )

    extracted_content = None
    page_count = 1

    # 1. PDF / Slide / Testi: Estrazione rapida con PyMuPDF & Indicizzazione LanceDB
    if ext == ".pdf":
        try:
            import pymupdf
            doc = pymupdf.open(fpath)
            page_count = len(doc)
            pages_text = []
            for idx in range(page_count):
                ptxt = doc[idx].get_text().strip()
                if ptxt:
                    pages_text.append(f"--- Pagina {idx + 1} ---\n{ptxt}")
            doc.close()

            extracted_content = "\n\n".join(pages_text) if pages_text else f"Documento PDF: {filename}"

            # Salva metadati per cache rapida
            meta_path = f"{fpath}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({"page_count": page_count, "content": extracted_content}, mf, ensure_ascii=False)

            # Indicizza subito in LanceDB tramite Qwen3-Embedding
            indexed_chunks = index_material_document(
                subject_id=subject_id,
                material_id=filename,
                title=filename,
                content=extracted_content,
                source_type="material",
            )
            logger.info(f"[Material Upload] PDF '{filename}' indicizzato con successo in LanceDB ({indexed_chunks} chunk).")
        except Exception as exc:
            logger.error(f"[Material Upload] Errore estrazione PyMuPDF su '{filename}': {exc}")

    elif ext in (".txt", ".md"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                extracted_content = tf.read()
            page_count = 1

            meta_path = f"{fpath}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump({"page_count": 1, "content": extracted_content}, mf, ensure_ascii=False)

            indexed_chunks = index_material_document(
                subject_id=subject_id,
                material_id=filename,
                title=filename,
                content=extracted_content,
                source_type="notes" if ext == ".md" else "material",
            )
            logger.info(f"[Material Upload] Testo '{filename}' indicizzato in LanceDB ({indexed_chunks} chunk).")
        except Exception as exc:
            logger.error(f"[Material Upload] Errore lettura testo '{filename}': {exc}")

    # 2. Registrazioni Audio Esterne (.wav, .mp3, .m4a): Background Task Whisper + Qwen3.5
    elif ext in AUDIO_EXTENSIONS:
        logger.info(f"[Material Upload] Rilevato audio esterno '{filename}'. Avvio worker Whisper in background...")
        threading.Thread(
            target=_process_imported_audio_bg,
            args=(subject_id, fpath, filename),
            daemon=True,
            name=f"audio-import-{filename[:12]}"
        ).start()

    return MaterialOut(
        id=filename,
        name=filename,
        type=ftype,
        size=size_str,
        uploadedAt="Oggi",
        pageCount=page_count,
        content=extracted_content,
        isAutoImported=False,
        lecture_id=None,
        duration_seconds=None,
        in_quaderno=None,
        source_type=None,
    )


@router.delete("/{subject_id}/materials/{filename}", status_code=status.HTTP_204_NO_CONTENT)
def delete_material_file(subject_id: int, filename: str, db: Session = Depends(get_db)):
    """
    Elimina un file caricato da disco e rimuove a cascata i relativi vettori da LanceDB.
    """
    from cleaner import get_all_uploads_directories

    filename = _safe_filename(filename)
    folder_name = f"subject_{subject_id}"

    # Un audio importato è anche il file della lezione generata: non rompere il player
    for udir in get_all_uploads_directories():
        udir_root = Path(udir).resolve()
        base_dir = (udir_root / folder_name).resolve()
        candidate_obj = (base_dir / filename).resolve()
        if not candidate_obj.is_relative_to(base_dir):
            raise HTTPException(status_code=400, detail="Percorso non consentito.")
        candidate = str(candidate_obj)
        candidate_unresolved = os.path.abspath(os.path.join(udir, folder_name, filename))
        linked = db.query(Lecture).filter(
            (Lecture.audio_path == candidate) | (Lecture.audio_path == candidate_unresolved)
        ).first()
        if linked:
            raise HTTPException(
                status_code=409,
                detail=f"Il file è l'audio della lezione '{linked.title}'. Elimina la lezione per rimuoverlo.",
            )

    for udir in get_all_uploads_directories():
        udir_root = Path(udir).resolve()
        base_dir = (udir_root / folder_name).resolve()
        fpath_obj = (base_dir / filename).resolve()
        if not fpath_obj.is_relative_to(base_dir):
            continue
        fpath = str(fpath_obj)
        if os.path.exists(fpath) and os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                logger.warning(f"Errore rimozione file {fpath}: {e}")

        meta_path = f"{fpath}.meta.json"
        if os.path.exists(meta_path) and os.path.isfile(meta_path):
            try:
                os.remove(meta_path)
            except Exception:
                pass

    # Eliminazione a cascata vettori da LanceDB e compattazione
    delete_material_vectors(subject_id=subject_id, material_id=filename)
    compact_lancedb()



