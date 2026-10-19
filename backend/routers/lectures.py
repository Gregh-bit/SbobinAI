"""
routers/lectures.py — SbobinAI
Endpoint REST per la gestione delle Lezioni e Segnalibri.
"""
from typing import List
import os

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import Bookmark, Lecture, Subject, get_db
from schemas import BookmarkCreate, BookmarkOut, LectureCreate, LectureOut, LectureUpdate
from rag import delete_lecture_vectors

router = APIRouter(prefix="/lectures", tags=["Lectures"])

# Colonne NOT NULL / con default: un null esplicito dal client va ignorato, non scritto
_NON_NULLABLE_FIELDS = ("title", "date", "source_type", "in_quaderno")


# ── Lectures ───────────────────────────────────────────────────────────────────

@router.get("/by-subject/{subject_id}", response_model=List[LectureOut])
def list_lectures(subject_id: int, quaderno_only: bool = False, db: Session = Depends(get_db)):
    """Restituisce le lezioni di una materia, ordinate per data."""
    query = db.query(Lecture).filter(Lecture.subject_id == subject_id)
    if quaderno_only:
        query = query.filter(Lecture.in_quaderno == True)
    return query.order_by(Lecture.date.desc()).all()


@router.post("/", response_model=LectureOut, status_code=status.HTTP_201_CREATED)
def create_lecture(payload: LectureCreate, db: Session = Depends(get_db)):
    """Crea una nuova sessione di lezione."""
    subject = db.query(Subject).filter(Subject.id == payload.subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")
    data = payload.model_dump()
    for f in _NON_NULLABLE_FIELDS:
        if data.get(f) is None:
            data.pop(f, None)
    lecture = Lecture(**data)
    db.add(lecture)
    db.commit()
    db.refresh(lecture)
    return lecture


@router.get("/{lecture_id}", response_model=LectureOut)
def get_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")
    return lecture


@router.patch("/{lecture_id}", response_model=LectureOut)
def update_lecture(lecture_id: int, payload: LectureUpdate, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is None and field in _NON_NULLABLE_FIELDS:
            continue
        setattr(lecture, field, value)
    db.commit()
    db.refresh(lecture)
    return lecture


@router.get("/{lecture_id}/audio")
def get_lecture_audio(lecture_id: int, db: Session = Depends(get_db)):
    """Serve il file audio originale della lezione (.wav / .mp3 / .m4a)."""
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture or not lecture.audio_path:
        raise HTTPException(status_code=404, detail="File audio non registrato per questa lezione.")
    if not os.path.exists(lecture.audio_path):
        raise HTTPException(status_code=404, detail="File audio non trovato sul disco.")

    media_type = "audio/wav"
    if lecture.audio_path.endswith(".mp3"):
        media_type = "audio/mpeg"
    elif lecture.audio_path.endswith(".m4a"):
        media_type = "audio/mp4"
    elif lecture.audio_path.endswith(".aac"):
        media_type = "audio/aac"
    elif lecture.audio_path.endswith(".ogg"):
        media_type = "audio/ogg"

    return FileResponse(lecture.audio_path, media_type=media_type, filename=os.path.basename(lecture.audio_path))


@router.delete("/{lecture_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    subject_id = lecture.subject_id
    audio_path = lecture.audio_path

    # 1. Eliminazione a cascata vettori LanceDB
    delete_lecture_vectors(subject_id=subject_id, lecture_id=lecture_id)

    # 2. Eliminazione file audio fisico registrato
    if audio_path and os.path.exists(audio_path) and os.path.isfile(audio_path):
        try:
            os.remove(audio_path)
        except Exception:
            pass

    # Ricerca ed eliminazione approfondita in tutte le cartelle audio per lecture_{id}_*
    try:
        from cleaner import get_all_audio_directories
        prefix = f"lecture_{lecture_id}_"
        prefix_alt = f"lecture_{lecture_id}."
        for adir in get_all_audio_directories():
            if os.path.exists(adir):
                for fname in os.listdir(adir):
                    if fname.startswith(prefix) or fname.startswith(prefix_alt):
                        fp = os.path.join(adir, fname)
                        if os.path.isfile(fp):
                            try:
                                os.remove(fp)
                            except Exception:
                                pass
    except Exception:
        pass

    # 3. Eliminazione segnalibri associati
    db.query(Bookmark).filter(Bookmark.lecture_id == lecture_id).delete()

    # 4. Eliminazione da SQLite
    db.delete(lecture)
    db.commit()

    # 5. Compattazione automatica
    try:
        from cleaner import vacuum_sqlite
        vacuum_sqlite()
    except Exception:
        pass



# ── Bookmarks ──────────────────────────────────────────────────────────────────

bookmark_router = APIRouter(prefix="/bookmarks", tags=["Bookmarks"])


@bookmark_router.get("/by-lecture/{lecture_id}", response_model=List[BookmarkOut])
def list_bookmarks(lecture_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Bookmark)
        .filter(Bookmark.lecture_id == lecture_id)
        .order_by(Bookmark.timestamp_seconds)
        .all()
    )


@bookmark_router.post("/", response_model=BookmarkOut, status_code=status.HTTP_201_CREATED)
def create_bookmark(payload: BookmarkCreate, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == payload.lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")
    bookmark = Bookmark(**payload.model_dump())
    db.add(bookmark)
    db.commit()
    db.refresh(bookmark)
    return bookmark


@bookmark_router.delete("/{bookmark_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_bookmark(bookmark_id: int, db: Session = Depends(get_db)):
    bookmark = db.query(Bookmark).filter(Bookmark.id == bookmark_id).first()
    if not bookmark:
        raise HTTPException(status_code=404, detail="Segnalibro non trovato.")
    db.delete(bookmark)
    db.commit()
