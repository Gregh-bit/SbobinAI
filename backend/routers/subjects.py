"""
routers/subjects.py — SbobinAI
Endpoint REST per la gestione delle Materie e Materiali del corso.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import List, Optional
import xml.etree.ElementTree as ET
import zipfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import Bookmark, Lecture, SessionLocal, Subject, get_db
from schemas import MaterialOut, MaterialPageOut, SubjectCreate, SubjectOut, SubjectUpdate
import rag
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
_PREVIEWABLE_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json",
    ".py", ".ts", ".tsx", ".js", ".jsx", ".html", ".css",
    ".xml", ".yaml", ".yml", ".tex", ".log",
})


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


_PPTX_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


def _extract_pptx_pages(file_path: Path) -> list[dict]:
    """Estrae il testo per ciascuna slide di un file .pptx ordinato numericamente."""
    pages: list[dict] = []
    with zipfile.ZipFile(file_path, "r") as zf:
        slide_entries: list[tuple[int, str]] = []
        for entry_name in zf.namelist():
            match = _PPTX_SLIDE_RE.match(entry_name)
            if match:
                slide_entries.append((int(match.group(1)), entry_name))
        slide_entries.sort(key=lambda item: item[0])

        for slide_num, entry_name in slide_entries:
            xml_bytes = zf.read(entry_name)
            root = ET.fromstring(xml_bytes)
            paragraphs: list[str] = []
            for elem in root.iter():
                if elem.tag.endswith("}p") or elem.tag == "p":
                    runs = [
                        t_elem.text
                        for t_elem in elem.iter()
                        if (t_elem.tag.endswith("}t") or t_elem.tag == "t") and t_elem.text
                    ]
                    para_text = "".join(runs).strip()
                    if para_text:
                        paragraphs.append(para_text)
            if not paragraphs:
                fallback_runs = [
                    elem.text.strip()
                    for elem in root.iter()
                    if (elem.tag.endswith("}t") or elem.tag == "t") and elem.text and elem.text.strip()
                ]
                slide_text = " ".join(fallback_runs)
            else:
                slide_text = "\n".join(paragraphs)
            pages.append({
                "page": slide_num,
                "text": slide_text,
                "label": f"Slide {slide_num}",
            })
    return pages


def _extract_pdf_pages(file_path: Path) -> list[dict]:
    """Estrae il testo pagina per pagina di un file PDF tramite PyMuPDF."""
    import pymupdf

    pages: list[dict] = []
    with pymupdf.open(str(file_path)) as doc:
        for idx in range(len(doc)):
            text = doc.load_page(idx).get_text("text").strip()
            pages.append({
                "page": idx + 1,
                "text": text,
                "label": f"Pag. {idx + 1}",
            })
    return pages


def _split_text_pages(text: str, page_chars: int = 2500) -> list[str]:
    """Divide un testo lungo in pagine logiche di circa 2500 caratteri."""
    if not text:
        return [""]
    pages: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + page_chars, length)
        if end < length:
            split_at = text.rfind("\n", start, end)
            if split_at <= start + page_chars // 2:
                split_at = text.rfind(" ", start, end)
            if split_at > start + page_chars // 2:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            pages.append(chunk)
        start = end
    return pages or [""]


def _save_material_meta(fpath: str, page_count: int, content: str) -> None:
    """Salva i metadati estratti in cache .meta.json accanto al materiale."""
    try:
        with open(f"{fpath}.meta.json", "w", encoding="utf-8") as mf:
            json.dump({"page_count": page_count, "content": content}, mf, ensure_ascii=False)
    except Exception:
        pass


def _resolve_material_file(subject_id: int, filename: str, db: Session) -> tuple[str, Path]:
    """Valida materia, nome file e contenimento del percorso, restituendo il file su disco."""
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    safe_name = _safe_filename(filename)
    uploads_root = Path(get_uploads_dir()).resolve()
    for folder_name in (f"subject_{subject_id}", str(subject_id)):
        base_dir = (uploads_root / folder_name).resolve()
        resolved = (base_dir / safe_name).resolve()
        if not resolved.is_relative_to(base_dir):
            raise HTTPException(status_code=403, detail="Percorso non consentito.")
        if resolved.is_file():
            return safe_name, resolved

    raise HTTPException(status_code=404, detail="File non trovato.")


def _material_and_subject_exist(subject_id: int, target_path: Path) -> bool:
    """Verifica che il file su disco e la materia in SQLite esistano ancora."""
    if not target_path.is_file():
        return False
    local_db = SessionLocal()
    try:
        return local_db.query(Subject.id).filter(Subject.id == subject_id).first() is not None
    finally:
        local_db.close()


def _index_material_pages_bg(
    subject_id: int,
    source_id: str,
    title: str,
    pages: list[dict],
    source_type: str = "slide",
    file_path: Optional[str] = None,
) -> None:
    """Esegue in background l'indicizzazione per pagina/slide in LanceDB."""
    target_path = Path(file_path) if file_path else (Path(get_uploads_dir()) / f"subject_{subject_id}" / source_id)
    try:
        if not _material_and_subject_exist(subject_id, target_path):
            logger.info(f"[Material Upload] '{title}' rimosso prima dell'indicizzazione, skip.")
            return
        indexed = rag.index_material_pages(
            subject_id=subject_id,
            source_id=source_id,
            title=title,
            pages=pages,
            source_type=source_type,
        )
        if not _material_and_subject_exist(subject_id, target_path):
            delete_material_vectors(subject_id=subject_id, material_id=source_id)
            logger.info(f"[Material Upload] '{title}' eliminato durante l'indicizzazione, vettori ripuliti.")
            return
        logger.info(f"[Material Upload] '{title}' indicizzato in LanceDB ({indexed} chunk, source_type={source_type}).")
    except Exception as exc:
        logger.error(f"[Material Upload] Errore indicizzazione pagine '{title}': {exc}")
    finally:
        try:
            rag.unload_embedding_model()
        except Exception:
            pass
        clear_metal_cache()



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

    # Genera automaticamente il file InfoCorso.md e indicizzalo per il RAG
    info_content = ensure_infocorso_file(subject)
    try:
        from rag import index_material_document
        index_material_document(
            subject_id=subject.id,
            material_id=f"infocorso-{subject.id}",
            title="InfoCorso.md",
            content=info_content,
            source_type="infocorso",
        )
    except Exception as exc:
        logger.debug(f"[Subjects] Indicizzazione InfoCorso: {exc}")

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
    out.lecture_count = db.query(func.count(Lecture.id)).filter(Lecture.subject_id == subject_id).scalar() or 0
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

    # Aggiorna InfoCorso.md con i nuovi dati e aggiorna i vettori
    info_content = ensure_infocorso_file(subject)
    try:
        from rag import index_material_document
        index_material_document(
            subject_id=subject.id,
            material_id=f"infocorso-{subject.id}",
            title="InfoCorso.md",
            content=info_content,
            source_type="infocorso",
        )
    except Exception as exc:
        logger.debug(f"[Subjects] Indicizzazione InfoCorso update: {exc}")

    out = SubjectOut.model_validate(subject)
    out.lecture_count = db.query(func.count(Lecture.id)).filter(Lecture.subject_id == subject_id).scalar() or 0
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
                        pages = _extract_pdf_pages(Path(fpath))
                        page_count = len(pages) or 1
                        text_chunks = [p["text"] for p in pages if p["text"]]
                        if text_chunks:
                            doc_content = "\n".join(text_chunks)
                            _save_material_meta(fpath, page_count, doc_content)
                    except Exception:
                        pass
                elif fname.lower().endswith(".pptx"):
                    try:
                        pages = _extract_pptx_pages(Path(fpath))
                        page_count = len(pages) or 1
                        text_chunks = [p["text"] for p in pages if p["text"]]
                        if text_chunks:
                            doc_content = "\n".join(text_chunks)
                            _save_material_meta(fpath, page_count, doc_content)
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
        from llm_engine import generate_lecture_chapter, unload_llm, is_model_downloaded, DEFAULT_LLM_MODEL
        from config import load_settings
        llm_model = load_settings().get("llm_model") or DEFAULT_LLM_MODEL
        if is_model_downloaded(llm_model) and transcript:
            try:
                notes = asyncio.run(
                    generate_lecture_chapter(title=clean_title, transcript=transcript, model_id=llm_model)
                )
            except InterruptedError:
                logger.info(f"[Audio Ingest] Sintesi interrotta dall'utente per '{orig_filename}'.")
                unload_llm()
                clear_metal_cache()
                return
            except Exception as e:
                logger.warning(f"[Audio Ingest] Sintesi automatica non riuscita: {e}")
            finally:
                # Scarica sempre l'LLM prima di caricare embedding o Whisper su M1 8GB
                unload_llm()
                clear_metal_cache()

        if not notes:
            notes = (
                f"# {clean_title} — Appunti di Lezione\n\n"
                f"> [!NOTE]\n> Registrazione esterna importata nella memoria del corso.\n\n"
                f"## Trascrizione Integrale\n{transcript if transcript else 'Trascrizione non disponibile o audio privo di parlato.'}"
            )

        # Salva in SQLite solo se la materia esiste ancora
        from database import SessionLocal, Subject, Lecture
        lec_id = None
        with SessionLocal() as session:
            if session.query(Subject).filter(Subject.id == subject_id).first() is None:
                logger.info(f"[Audio Ingest] Materia {subject_id} eliminata nel frattempo: skip salvataggio '{clean_title}'.")
                return
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

        # Indicizza in LanceDB (con ciclo di memoria disaccoppiato dall'LLM e da Whisper)
        if lec_id:
            from rag import index_lecture, unload_embedding_model
            try:
                index_lecture(
                    subject_id=subject_id,
                    lecture_id=lec_id,
                    title=clean_title,
                    transcript=transcript,
                    notes=notes
                )
            finally:
                unload_embedding_model()
                clear_metal_cache()

            # Pre-calcola e salva in cache i timestamp parola per parola, poi rilascia Whisper
            try:
                from routers.lectures import compute_and_cache_lecture_timestamps
                from transcriber import release_transcriber_memory
                compute_and_cache_lecture_timestamps(lec_id)
                release_transcriber_memory()
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
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """
    Carica un file (PDF, Slide, Appunti, Audio) nella memoria della materia:
    - Per PDF/PPTX: estrae il testo pagina/slide per pagina e indicizza in LanceDB in background.
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

    # 1. PDF / Slide PPTX: Estrazione per pagina/slide & Indicizzazione LanceDB in background
    if ext in (".pdf", ".pptx"):
        try:
            pages = _extract_pdf_pages(fpath_obj) if ext == ".pdf" else _extract_pptx_pages(fpath_obj)
            page_count = len(pages) or 1
            section_prefix = "Pagina" if ext == ".pdf" else "Slide"
            pages_text = [f"--- {section_prefix} {p['page']} ---\n{p['text']}" for p in pages if p["text"]]
            fallback_label = f"Documento PDF: {filename}" if ext == ".pdf" else f"Presentazione: {filename}"
            extracted_content = "\n\n".join(pages_text) if pages_text else fallback_label

            _save_material_meta(fpath, page_count, extracted_content)

            if background_tasks is not None:
                background_tasks.add_task(_index_material_pages_bg, subject_id, filename, filename, pages, "slide", fpath)
            else:
                _index_material_pages_bg(subject_id, filename, filename, pages, "slide", fpath)
        except Exception as exc:
            logger.error(f"[Material Upload] Errore estrazione {ext} su '{filename}': {exc}")

    elif ext in (".txt", ".md"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                extracted_content = tf.read()
            page_count = 1

            _save_material_meta(fpath, 1, extracted_content)

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

    # Pulizia a cascata dello scope in tutte le sessioni di chat della materia
    try:
        from database import ChatSession
        sessions = db.query(ChatSession).filter(ChatSession.subject_id == subject_id).all()
        for s in sessions:
            if not s.scope_json:
                continue
            try:
                s_scope = json.loads(s.scope_json)
                if isinstance(s_scope, dict):
                    mats = s_scope.get("material_ids") or []
                    if filename in mats:
                        s_scope["material_ids"] = [m for m in mats if m != filename]
                        s.scope_json = json.dumps(s_scope, ensure_ascii=False)
            except Exception:
                pass
        db.commit()
    except Exception as e:
        logger.warning(f"Errore pulizia scope chat per materiale '{filename}': {e}")


@router.get("/{subject_id}/materials/{filename}/file")
def get_material_file(subject_id: int, filename: str, db: Session = Depends(get_db)):
    """Restituisce il file originale caricato nella memoria della materia."""
    safe_name, resolved = _resolve_material_file(subject_id, filename, db)
    return FileResponse(path=str(resolved), filename=safe_name)


@router.get("/{subject_id}/materials/{filename}/page/{page_num}", response_model=MaterialPageOut)
def get_material_page(subject_id: int, filename: str, page_num: int, db: Session = Depends(get_db)):
    """Restituisce una singola pagina/slide renderizzata (immagine PNG per PDF o testo per PPTX/testi)."""
    if page_num < 1:
        raise HTTPException(status_code=400, detail="Numero di pagina non valido.")

    safe_name, resolved = _resolve_material_file(subject_id, filename, db)
    ext = resolved.suffix.lower()

    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(str(resolved)) as doc:
            page_count = len(doc)
            if page_count == 0 or page_num > page_count:
                raise HTTPException(status_code=404, detail="Pagina non trovata.")
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6))
            png_b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            text = page.get_text("text").strip()

        return MaterialPageOut(
            filename=safe_name,
            page=page_num,
            page_count=page_count,
            kind="image",
            image_data_url=f"data:image/png;base64,{png_b64}",
            text=text,
            title=f"{safe_name} — Pag. {page_num}",
        )

    if ext == ".pptx":
        pages = _extract_pptx_pages(resolved)
        page_count = len(pages)
        if page_count == 0:
            raise HTTPException(status_code=404, detail="Slide non trovata.")
        target = next((p for p in pages if p["page"] == page_num), None)
        if target is None and page_num <= page_count and all(p["page"] == idx + 1 for idx, p in enumerate(pages)):
            target = pages[page_num - 1]
        if target is None:
            raise HTTPException(status_code=404, detail="Slide non trovata.")
        return MaterialPageOut(
            filename=safe_name,
            page=page_num,
            page_count=page_count,
            kind="text",
            image_data_url=None,
            text=target["text"],
            title=f"{safe_name} — {target['label']}",
        )

    if ext not in _PREVIEWABLE_TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Anteprima testuale non supportata per questo tipo di file binario.",
        )

    with open(resolved, "r", encoding="utf-8", errors="replace") as tf:
        raw_text = tf.read(200_000)
    text_pages = _split_text_pages(raw_text, page_chars=2500)
    page_count = len(text_pages)
    if page_num > page_count:
        raise HTTPException(status_code=404, detail="Pagina non trovata.")

    return MaterialPageOut(
        filename=safe_name,
        page=page_num,
        page_count=page_count,
        kind="text",
        image_data_url=None,
        text=text_pages[page_num - 1],
        title=f"{safe_name} — Pag. {page_num}" if page_count > 1 else safe_name,
    )




