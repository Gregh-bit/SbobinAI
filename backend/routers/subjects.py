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
from pydantic import BaseModel
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
MAX_MATERIAL_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_PREVIEWABLE_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json",
    ".py", ".ts", ".tsx", ".js", ".jsx", ".html", ".css",
    ".xml", ".yaml", ".yml", ".tex", ".log", ".docx",
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


def _extract_docx_text(file_path: Path) -> str:
    """Estrae testo, intestazioni, tabelle e sezioni da un file Word .docx preservando la struttura Markdown."""
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            names = set(zf.namelist())
            if "word/document.xml" not in names:
                return ""

            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

            def _parse_xml_elements(xml_bytes: bytes) -> list[str]:
                root = ET.fromstring(xml_bytes)
                elements: list[str] = []
                for child in root.iter():
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "p":
                        runs = [t.text for t in child.findall(".//w:t", ns) if t.text]
                        line = "".join(runs).strip()
                        if not line:
                            continue
                        pPr = child.find("w:pPr", ns)
                        style = ""
                        if pPr is not None:
                            pStyle = pPr.find("w:pStyle", ns)
                            if pStyle is not None:
                                style = pStyle.attrib.get(f"{{{ns['w']}}}val", "").lower()
                        if "heading1" in style or style == "title":
                            elements.append(f"# {line}")
                        elif "heading2" in style or style == "subtitle":
                            elements.append(f"## {line}")
                        elif "heading3" in style:
                            elements.append(f"### {line}")
                        elif "heading" in style:
                            elements.append(f"#### {line}")
                        else:
                            elements.append(line)
                    elif tag == "tbl":
                        rows: list[list[str]] = []
                        for tr in child.findall(".//w:tr", ns):
                            row: list[str] = []
                            for tc in tr.findall(".//w:tc", ns):
                                tc_runs = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                                cell_text = " ".join("".join(tc_runs).split())
                                row.append(cell_text)
                            if row and any(c.strip() for c in row):
                                rows.append(row)
                        if rows:
                            max_cols = max(len(r) for r in rows)
                            normalized_rows = [r + [""] * (max_cols - len(r)) for r in rows]
                            header = "| " + " | ".join(normalized_rows[0]) + " |"
                            sep = "| " + " | ".join(["---"] * max_cols) + " |"
                            tbl_lines = [header, sep]
                            for r in normalized_rows[1:]:
                                tbl_lines.append("| " + " | ".join(r) + " |")
                            elements.append("\n".join(tbl_lines))
                return elements

            all_parts: list[str] = []
            # 1. Intestazioni
            for h in sorted(n for n in names if n.startswith("word/header") and n.endswith(".xml")):
                all_parts.extend(_parse_xml_elements(zf.read(h)))
            # 2. Documento principale (testo + tabelle)
            all_parts.extend(_parse_xml_elements(zf.read("word/document.xml")))
            # 3. Note
            if "word/footnotes.xml" in names:
                all_parts.extend(_parse_xml_elements(zf.read("word/footnotes.xml")))
            # 4. Piè di pagina
            for f in sorted(n for n in names if n.startswith("word/footer") and n.endswith(".xml")):
                all_parts.extend(_parse_xml_elements(zf.read(f)))

            return "\n\n".join(dict.fromkeys(all_parts))
    except Exception as exc:
        logger.error(f"[DOCX Extraction] Errore su {file_path}: {exc}")
        return ""


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


def _save_material_meta(
    fpath: str,
    page_count: int,
    content: str,
    tag: Optional[str] = None,
    lecture_id: Optional[int] = None,
) -> None:
    """Salva i metadati estratti in cache .meta.json accanto al materiale."""
    meta_path = f"{fpath}.meta.json"
    data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                data = json.load(mf)
        except Exception:
            pass
    data["page_count"] = page_count
    data["content"] = content
    if tag is not None:
        data["tag"] = tag
        data["source_type"] = tag
    if lecture_id is not None:
        data["lecture_id"] = lecture_id
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(data, mf, ensure_ascii=False)
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
    keywords_section = ""
    if subject.whisper_prompt_keywords and subject.whisper_prompt_keywords.strip():
        keywords_section = f"\n## Vocabolario e Termini Chiave\n{subject.whisper_prompt_keywords.strip()}\n"

    desc = subject.description.strip() if subject.description and subject.description.strip() else "Nessuna descrizione o informazione aggiuntiva specificata."
    content = f"""# {subject.name} — Informazioni del Corso

## Informazioni Generali & Descrizione
{desc}
{keywords_section}
## Dettagli Corso
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
        # audio_path può essere stato valorizzato via API: elimina solo file posseduti
        # dall'app nelle directory audio note, mai un percorso arbitrario esterno.
        managed_audio_path = False
        if lec.audio_path:
            try:
                candidate = Path(lec.audio_path).resolve()
                managed_audio_path = any(
                    candidate.is_relative_to(Path(audio_dir).resolve())
                    for audio_dir in audio_dirs
                )
            except (OSError, ValueError):
                managed_audio_path = False
        if managed_audio_path and os.path.isfile(lec.audio_path):
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

    # 2. Tutte le lezioni / registrazioni e file caricati, ordinati dal più recente
    import unicodedata
    from datetime import timezone as _tz
    from database import Lecture
    lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).order_by(Lecture.date.desc()).all()

    # Mappa lezioni per ID, audio_path (filename collegato) e stem/titolo normalizzati
    lecture_by_id: Dict[int, Lecture] = {lec.id: lec for lec in lectures}
    lecture_by_audio: Dict[str, Lecture] = {}
    lecture_by_key: Dict[str, Lecture] = {}
    for lec in lectures:
        if lec.audio_path:
            norm_audio = unicodedata.normalize("NFC", os.path.basename(lec.audio_path)).strip().lower()
            lecture_by_audio[norm_audio] = lec
        lec_key = unicodedata.normalize("NFC", lec.title or "").strip().lower()
        if lec_key:
            lecture_by_key[lec_key] = lec

    covered_lecture_ids = set()
    dated_materials: List[tuple[float, MaterialOut]] = []

    # 2. File fisici caricati in uploads (PDF, docx, slide, note, audio)
    if os.path.exists(subj_dir):
        for fname in os.listdir(subj_dir):
            if fname == "InfoCorso.md" or fname.startswith(".") or fname.endswith(".meta.json"):
                continue
            fpath = os.path.join(subj_dir, fname)
            if os.path.isfile(fpath):
                f_kb = os.path.getsize(fpath) / 1024
                size_str = f"{f_kb / 1024:.1f} MB" if f_kb > 1024 else f"{f_kb:.1f} KB"
                norm_fname = unicodedata.normalize("NFC", fname).strip()
                lower_name = norm_fname.lower()
                stem = unicodedata.normalize("NFC", os.path.splitext(norm_fname)[0]).strip().lower()

                ftype = (
                    "slide" if lower_name.endswith((".ppt", ".pptx", ".key", ".odp"))
                    else "pdf" if lower_name.endswith(".pdf")
                    else "audio" if lower_name.endswith(AUDIO_EXTENSIONS)
                    else "doc"
                )

                page_count = 1
                doc_content = None
                saved_tag = None
                saved_lecture_id = None

                # Prova a leggere metadati salvati
                meta_path = f"{fpath}.meta.json"
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            m_data = json.load(mf)
                            page_count = m_data.get("page_count", 1)
                            doc_content = m_data.get("content")
                            saved_tag = m_data.get("tag") or m_data.get("source_type")
                            saved_lecture_id = m_data.get("lecture_id")
                    except Exception:
                        pass

                if not doc_content:
                    if lower_name.endswith(".pdf"):
                        try:
                            pages = _extract_pdf_pages(Path(fpath))
                            page_count = len(pages) or 1
                            text_chunks = [p["text"] for p in pages if p["text"]]
                            if text_chunks:
                                doc_content = "\n".join(text_chunks)
                                _save_material_meta(fpath, page_count, doc_content, tag="slide", lecture_id=saved_lecture_id)
                        except Exception:
                            pass
                    elif lower_name.endswith(".pptx"):
                        try:
                            pages = _extract_pptx_pages(Path(fpath))
                            page_count = len(pages) or 1
                            text_chunks = [p["text"] for p in pages if p["text"]]
                            if text_chunks:
                                doc_content = "\n".join(text_chunks)
                                _save_material_meta(fpath, page_count, doc_content, tag="slide", lecture_id=saved_lecture_id)
                        except Exception:
                            pass
                    elif lower_name.endswith(".docx"):
                        try:
                            doc_content = _extract_docx_text(Path(fpath))
                            page_count = 1
                            _save_material_meta(fpath, 1, doc_content, tag="notes", lecture_id=saved_lecture_id)
                        except Exception:
                            pass
                    elif lower_name.endswith((".txt", ".md")):
                        try:
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                                doc_content = tf.read()
                                _save_material_meta(fpath, 1, doc_content, tag="notes", lecture_id=saved_lecture_id)
                        except Exception:
                            pass

                tag = saved_tag or (
                    "slide" if ftype in ("slide", "pdf")
                    else "notes" if lower_name.endswith((".md", ".txt", ".docx", ".doc"))
                    else "audio" if ftype == "audio"
                    else "doc"
                )

                # Verifica se esiste una Lezione collegata (per saved_lecture_id, audio_path, titolo o stem)
                linked_lec = (
                    (lecture_by_id.get(saved_lecture_id) if saved_lecture_id else None)
                    or lecture_by_audio.get(lower_name)
                    or lecture_by_key.get(stem)
                    or lecture_by_key.get(lower_name)
                )
                if not linked_lec:
                    for lk, lobj in lecture_by_key.items():
                        if len(lk) >= 5 and (stem.startswith(lk) or lk.startswith(stem)):
                            linked_lec = lobj
                            break

                linked_lec_id = None
                linked_in_quaderno = None
                if linked_lec:
                    covered_lecture_ids.add(linked_lec.id)
                    linked_lec_id = linked_lec.id
                    linked_in_quaderno = linked_lec.in_quaderno

                    # Auto-heal: se la lezione nel quaderno era salvata con solo lo stub segnaposto
                    # ("Appunti importati da Materiali") ma ora abbiamo il contenuto vero estratto dal file,
                    # aggiorna le note complete della lezione nel database!
                    if doc_content:
                        placeholder_indicators = ("appunti importati da", "documento importato da")
                        cur_notes = (linked_lec.processed_notes or "").strip().lower()
                        if not cur_notes or any(p in cur_notes for p in placeholder_indicators) or (len(cur_notes) < 200 and len(doc_content) > 300):
                            linked_lec.processed_notes = doc_content
                            try:
                                db.commit()
                                db.refresh(linked_lec)
                                logger.info(f"[Materials] Aggiornato testo completo ({len(doc_content)} caratteri) per la lezione {linked_lec.id} da '{fname}'")
                            except Exception as commit_err:
                                db.rollback()
                                logger.warning(f"[Materials] Errore aggiornamento note lezione {linked_lec.id}: {commit_err}")

                    if not doc_content and linked_lec.processed_notes:
                        doc_content = linked_lec.processed_notes

                    if not linked_lec.audio_path:
                        linked_lec.audio_path = fname
                        try:
                            db.commit()
                        except Exception:
                            db.rollback()

                    if saved_lecture_id != linked_lec.id:
                        _save_material_meta(fpath, page_count, doc_content, tag=tag, lecture_id=linked_lec.id)

                try:
                    file_ts = os.path.getmtime(fpath)
                    file_date_str = datetime.fromtimestamp(file_ts, tz=_tz.utc).strftime('%d %b %Y')
                except Exception:
                    file_ts = 0.0
                    file_date_str = ""

                dated_materials.append((file_ts, MaterialOut(
                    id=fname,
                    name=fname,
                    type=ftype,
                    size=size_str,
                    uploadedAt=file_date_str,
                    pageCount=page_count,
                    content=doc_content,
                    isAutoImported=False,
                    lecture_id=linked_lec_id,
                    duration_seconds=linked_lec.duration_seconds if linked_lec else None,
                    in_quaderno=linked_in_quaderno,
                    source_type=tag,
                )))

    # 3. Lezioni registrate o create nel Quaderno non coperte da file fisici in uploads
    for lec in lectures:
        if lec.id in covered_lecture_ids:
            continue
        is_imported = (lec.source_type == "imported")
        # Se la lezione era di tipo 'imported' e ha un file corrispondente nella cartella uploads,
        # non duplicarla come card fantasma nei materiali
        if is_imported and os.path.exists(subj_dir):
            lec_norm = unicodedata.normalize("NFC", lec.title or "").strip().lower()
            file_stems = [
                unicodedata.normalize("NFC", os.path.splitext(f)[0]).strip().lower()
                for f in os.listdir(subj_dir)
                if not f.startswith(".")
            ]
            if lec_norm in file_stems or any(len(fs) >= 5 and (lec_norm.startswith(fs) or fs.startswith(lec_norm)) for fs in file_stems):
                continue

        dur_str = f"{int(lec.duration_seconds // 60)}m" if lec.duration_seconds else ("Appunti" if is_imported else "Audio / Appunti")
        date_str = lec.date.strftime('%d %b %Y') if lec.date else "Lezione"
        if lec.date:
            dt_utc = lec.date if lec.date.tzinfo is not None else lec.date.replace(tzinfo=_tz.utc)
            lec_ts = dt_utc.timestamp()
        else:
            lec_ts = 0.0
        dated_materials.append((lec_ts, MaterialOut(
            id=f"lecture-{lec.id}",
            name=lec.title,
            type="doc" if is_imported else "lecture",
            size=dur_str,
            uploadedAt=date_str,
            pageCount=1,
            content=lec.processed_notes or lec.raw_transcript,
            isAutoImported=False,
            lecture_id=lec.id,
            duration_seconds=lec.duration_seconds,
            in_quaderno=lec.in_quaderno,
            source_type="notes" if is_imported else "audio",
        )))

    # Ordina tutti i materiali (dopo InfoCorso.md) dal più recente al meno recente
    dated_materials.sort(key=lambda pair: pair[0], reverse=True)
    materials.extend(item for _, item in dated_materials)

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
                f"# {clean_title}\n\n"
                "> **Audio importato** — Trascrizione aggiunta alla memoria del corso.\n\n"
                f"## Trascrizione\n\n{transcript if transcript else 'Trascrizione non disponibile o audio privo di parlato.'}"
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

    copied = 0
    try:
        with open(fpath, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                copied += len(chunk)
                if copied > MAX_MATERIAL_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Il file supera il limite di 2 GB.")
                buffer.write(chunk)
    except Exception:
        try:
            os.remove(fpath)
        except OSError:
            pass
        raise

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
    tag = (
        "slide" if ext in (".pdf", ".ppt", ".pptx", ".key", ".odp")
        else "notes" if ext in (".md", ".txt", ".docx", ".doc")
        else "audio" if ext in AUDIO_EXTENSIONS
        else "doc"
    )

    if ext in (".pdf", ".pptx"):
        try:
            pages = _extract_pdf_pages(fpath_obj) if ext == ".pdf" else _extract_pptx_pages(fpath_obj)
            page_count = len(pages) or 1
            section_prefix = "Pagina" if ext == ".pdf" else "Slide"
            pages_text = [f"--- {section_prefix} {p['page']} ---\n{p['text']}" for p in pages if p["text"]]
            fallback_label = f"Documento PDF: {filename}" if ext == ".pdf" else f"Presentazione: {filename}"
            extracted_content = "\n\n".join(pages_text) if pages_text else fallback_label

            _save_material_meta(fpath, page_count, extracted_content, tag="slide")

            if background_tasks is not None:
                background_tasks.add_task(_index_material_pages_bg, subject_id, filename, filename, pages, "slide", fpath)
            else:
                threading.Thread(
                    target=_index_material_pages_bg,
                    args=(subject_id, filename, filename, pages, "slide", fpath),
                    daemon=True,
                ).start()
        except Exception as exc:
            logger.error(f"[Material Upload] Errore estrazione {ext} su '{filename}': {exc}")

    elif ext in (".docx", ".doc"):
        try:
            if ext == ".docx":
                extracted_content = _extract_docx_text(fpath_obj)
            else:
                extracted_content = ""
            page_count = 1

            _save_material_meta(fpath, 1, extracted_content, tag="notes")

            def _index_docx_bg():
                try:
                    indexed_chunks = index_material_document(
                        subject_id=subject_id,
                        material_id=filename,
                        title=filename,
                        content=extracted_content,
                        source_type="notes",
                    )
                    logger.info(f"[Material Upload] Docx '{filename}' indicizzato in LanceDB ({indexed_chunks} chunk).")
                except Exception as exc:
                    logger.error(f"[Material Upload] Errore indicizzazione docx '{filename}': {exc}")
                finally:
                    from rag import unload_embedding_model
                    unload_embedding_model()

            if background_tasks is not None:
                background_tasks.add_task(_index_docx_bg)
            else:
                threading.Thread(target=_index_docx_bg, daemon=True).start()
        except Exception as exc:
            logger.error(f"[Material Upload] Errore lettura docx '{filename}': {exc}")

    elif ext in (".txt", ".md"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                extracted_content = tf.read()
            page_count = 1

            _save_material_meta(fpath, 1, extracted_content, tag="notes")

            def _index_text_bg():
                try:
                    indexed_chunks = index_material_document(
                        subject_id=subject_id,
                        material_id=filename,
                        title=filename,
                        content=extracted_content,
                        source_type="notes",
                    )
                    logger.info(f"[Material Upload] Testo '{filename}' indicizzato in LanceDB ({indexed_chunks} chunk).")
                except Exception as exc:
                    logger.error(f"[Material Upload] Errore indicizzazione testo '{filename}': {exc}")
                finally:
                    from rag import unload_embedding_model
                    unload_embedding_model()

            if background_tasks is not None:
                background_tasks.add_task(_index_text_bg)
            else:
                threading.Thread(target=_index_text_bg, daemon=True).start()
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
        source_type=tag,
    )


class MaterialTagUpdate(BaseModel):
    tag: str


@router.patch("/{subject_id}/materials/{filename}/tag", response_model=MaterialOut)
def update_material_tag(
    subject_id: int,
    filename: str,
    payload: MaterialTagUpdate,
    db: Session = Depends(get_db)
):
    """Aggiorna il tag/classificazione di un materiale (Appunti vs Slide vs Doc)."""
    valid_tags = {"notes", "slide", "doc", "audio"}
    new_tag = payload.tag.lower().strip()
    if new_tag not in valid_tags:
        raise HTTPException(status_code=400, detail=f"Tag non valido. Usa uno tra: {', '.join(valid_tags)}")

    safe_name, resolved = _resolve_material_file(subject_id, filename, db)
    meta_path = f"{str(resolved)}.meta.json"

    meta_data = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
        except Exception:
            pass

    meta_data["tag"] = new_tag
    meta_data["source_type"] = new_tag
    try:
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"[Material Tag] Errore scrittura meta: {exc}")

    # Aggiorna anche i vettori LanceDB per riflettere il nuovo source_type
    try:
        from rag import get_lancedb_table
        tbl = get_lancedb_table()
        tbl.update(where=f"subject_id = {subject_id} AND source_id = '{safe_name}'", values={"source_type": new_tag})
    except Exception as exc:
        logger.debug(f"[Material Tag] Aggiornamento LanceDB source_type saltato: {exc}")

    f_kb = os.path.getsize(str(resolved)) / 1024
    size_str = f"{f_kb / 1024:.1f} MB" if f_kb > 1024 else f"{f_kb:.1f} KB"
    ext = resolved.suffix.lower()
    ftype = (
        "slide" if ext in (".ppt", ".pptx", ".key", ".odp")
        else "pdf" if ext == ".pdf"
        else "audio" if ext in AUDIO_EXTENSIONS
        else "doc"
    )

    return MaterialOut(
        id=safe_name,
        name=safe_name,
        type=ftype,
        size=size_str,
        uploadedAt="Modificato",
        pageCount=meta_data.get("page_count", 1),
        content=meta_data.get("content"),
        isAutoImported=False,
        lecture_id=None,
        duration_seconds=None,
        in_quaderno=None,
        source_type=new_tag,
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

    # Rimuovi anche l'eventuale lezione importata corrispondente nel database
    try:
        stem = os.path.splitext(filename)[0].strip()
        linked_lecs = db.query(Lecture).filter(
            Lecture.subject_id == subject_id,
            (Lecture.title == stem) | (Lecture.title == filename) | (Lecture.audio_path == filename)
        ).all()
        for l_item in linked_lecs:
            if l_item.source_type == "imported":
                db.delete(l_item)
        db.commit()
    except Exception as e:
        logger.warning(f"Errore rimozione lezione importata collegata a '{filename}': {e}")

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

    if ext == ".docx":
        docx_text = _extract_docx_text(resolved)
        return MaterialPageOut(
            filename=safe_name,
            page=1,
            page_count=1,
            kind="markdown",
            image_data_url=None,
            text=docx_text,
            title=safe_name,
        )

    if ext not in _PREVIEWABLE_TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Anteprima testuale non supportata per questo tipo di file binario.",
        )

    with open(resolved, "r", encoding="utf-8", errors="replace") as tf:
        raw_text = tf.read(500_000)

    if ext == ".md":
        return MaterialPageOut(
            filename=safe_name,
            page=1,
            page_count=1,
            kind="markdown",
            image_data_url=None,
            text=raw_text,
            title=safe_name,
        )

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




