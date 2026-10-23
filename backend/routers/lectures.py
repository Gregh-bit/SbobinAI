"""
routers/lectures.py — SbobinAI
Endpoint REST per la gestione delle Lezioni e Segnalibri.
"""
from typing import List
import os
import json
import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, defer, defer

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
    query = (
        db.query(Lecture)
        .options(defer(Lecture.raw_transcript), defer(Lecture.processed_notes))
        .filter(Lecture.subject_id == subject_id)
    )
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


def _normalize_karaoke_lines(raw_data) -> List[dict]:
    """Converte i dati grezzi nella struttura standard Apple Music (millisecondi)."""
    lines = []
    if not isinstance(raw_data, list):
        return []

    # Rilevamento robusto se i timestamp sono già in millisecondi analizzando la durata di parole o segmenti
    is_ms = False
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        for w in item.get("words", []):
            try:
                dur = float(w.get("end", 0.0)) - float(w.get("start", 0.0))
                if dur >= 15.0:
                    is_ms = True
                    break
            except Exception:
                pass
        if is_ms:
            break
        try:
            seg_dur = float(item.get("lineEnd", item.get("end", 0.0))) - float(item.get("lineStart", item.get("start", 0.0)))
            if seg_dur >= 30.0:
                is_ms = True
                break
        except Exception:
            pass

    scale = 1.0 if is_ms else 1000.0

    for item in raw_data:
        if not isinstance(item, dict):
            continue
        raw_start = max(0.0, float(item.get("lineStart", item.get("start", 0.0))))
        raw_end = max(raw_start, float(item.get("lineEnd", item.get("end", 0.0))))

        words_src = item.get("words", [])
        words_list = []
        for w in words_src:
            w_text = w.get("text") or w.get("word") or ""
            w_text = str(w_text).strip()
            if not w_text:
                continue
            w_s = float(w.get("start", 0.0)) * scale
            w_e = float(w.get("end", 0.0)) * scale
            words_list.append({
                "text": w_text,
                "start": int(round(w_s)),
                "end": int(round(w_e))
            })

        line_start = int(round(raw_start * scale))
        line_end = int(round(raw_end * scale))
        if words_list:
            line_start = min(line_start, words_list[0]["start"])
            line_end = max(line_end, words_list[-1]["end"])

        lines.append({
            "lineStart": line_start,
            "lineEnd": line_end,
            "words": words_list
        })
    return lines


def compute_proportional_timestamps(lecture) -> dict:
    """
    Allineamento proporzionale su raw_transcript e duration_seconds.
    Restituisce una stima istantanea adatta come fallback o durante l'elaborazione in background.
    """
    raw_text = (lecture.raw_transcript or "").strip()
    if not raw_text:
        return {"lines": [], "segments": [], "cached": False}

    duration_ms = int(round(float(lecture.duration_seconds or 30.0) * 1000))
    words = [w for w in raw_text.split() if w]
    if not words:
        return {"lines": [], "segments": [], "cached": False}

    time_per_word = max(250, duration_ms // max(1, len(words)))
    lines = []
    seg_size = 7
    for i in range(0, len(words), seg_size):
        chunk = words[i:i + seg_size]
        seg_start = i * time_per_word
        seg_end = min(duration_ms, (i + len(chunk)) * time_per_word)
        c_words = []
        for j, w in enumerate(chunk):
            w_start = (i + j) * time_per_word
            w_end = min(duration_ms, (i + j + 1) * time_per_word)
            c_words.append({"text": w, "start": w_start, "end": w_end})
        lines.append({
            "lineStart": seg_start,
            "lineEnd": seg_end,
            "words": c_words
        })
    return {"lines": lines, "segments": lines, "cached": False}


def compute_and_cache_lecture_timestamps(lecture_id: int) -> dict:
    """
    Calcola (tramite Whisper o allineamento proporzionale) e salva su cache su disco
    i timestamp parola per parola per una lezione.
    Può essere richiamata sia all'arrivo della richiesta GET, sia in background a fine registrazione.
    """
    from database import SessionLocal
    db = SessionLocal()
    try:
        lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        if not lecture:
            return {"lines": [], "segments": [], "cached": False}

        audio_path = lecture.audio_path
        cache_path = (audio_path + ".timestamps.json") if audio_path else None

        # 1. Controlla cache su disco esistente
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    lines = _normalize_karaoke_lines(data)
                    return {"lines": lines, "segments": lines, "cached": True}
            except Exception:
                pass

        # 2. Se abbiamo il file audio e Whisper è disponibile, calcola con Whisper
        if audio_path and os.path.isfile(audio_path):
            try:
                import ai_models
                from transcriber import _WHISPER_SPEC, USE_MLX, release_metal_cache, _transcription_lock, _decode_audio_file
                model_path = ai_models.local_model_path(_WHISPER_SPEC)
                if model_path:
                    with _transcription_lock:
                        if USE_MLX:
                            import mlx_whisper
                            # Decodifica con PyAV: supporta sia WAV che M4A/AAC senza richiedere l'eseguibile ffmpeg
                            audio_input = _decode_audio_file(audio_path)
                            res = mlx_whisper.transcribe(
                                audio_input,
                                path_or_hf_repo=model_path,
                                word_timestamps=True,
                                language="it",
                                temperature=(0.0,),
                                condition_on_previous_text=False,
                                fp16=True,
                            )
                            raw_segs = list(res.get("segments") or [])
                        else:
                            from faster_whisper import WhisperModel
                            fw_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                            segs_gen, _ = fw_model.transcribe(audio_path, language="it", word_timestamps=True)
                            raw_segs = list(segs_gen)

                    raw_converted = []
                    for s_idx, seg in enumerate(raw_segs):
                        w_list = []
                        words_src = seg.words if hasattr(seg, "words") else seg.get("words", [])
                        for w in words_src:
                            w_word = getattr(w, "word", None) if hasattr(w, "word") else w.get("word", "")
                            w_start = getattr(w, "start", 0.0) if hasattr(w, "start") else w.get("start", 0.0)
                            w_end = getattr(w, "end", 0.0) if hasattr(w, "end") else w.get("end", 0.0)
                            w_list.append({
                                "text": str(w_word).strip(),
                                "start": round(float(w_start), 3),
                                "end": round(float(w_end), 3),
                            })
                        seg_start = getattr(seg, "start", 0.0) if hasattr(seg, "start") else seg.get("start", 0.0)
                        seg_end = getattr(seg, "end", 0.0) if hasattr(seg, "end") else seg.get("end", 0.0)
                        raw_converted.append({
                            "id": s_idx,
                            "start": round(float(seg_start), 3),
                            "end": round(float(seg_end), 3),
                            "words": w_list,
                        })

                    lines = _normalize_karaoke_lines(raw_converted)

                    # Salva in cache su disco
                    if cache_path and lines:
                        try:
                            with open(cache_path, "w", encoding="utf-8") as f:
                                json.dump(lines, f, ensure_ascii=False)
                        except Exception:
                            pass

                    return {"lines": lines, "segments": lines, "cached": False}
            except Exception as exc:
                print(f"[Timestamps] Whisper alignment fallito: {exc}")

        # 3. Fallback: allineamento proporzionale su raw_transcript e duration_seconds
        result = compute_proportional_timestamps(lecture)
        lines = result.get("lines", [])

        if cache_path and lines:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(lines, f, ensure_ascii=False)
            except Exception:
                pass

        return result
    finally:
        db.close()


_pending_timestamp_tasks = set()
_pending_lock = threading.Lock()


def _bg_compute_timestamps(lecture_id: int):
    try:
        compute_and_cache_lecture_timestamps(lecture_id)
    finally:
        with _pending_lock:
            _pending_timestamp_tasks.discard(lecture_id)


@router.get("/{lecture_id}/timestamps")
def get_lecture_timestamps(
    lecture_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Restituisce la trascrizione sincronizzata parola per parola nel formato standard Apple Music.
    Tempi espressi rigorosamente in millisecondi (lineStart, lineEnd, words[{text, start, end}]).
    Se il file cache non esiste, restituisce subito la stima proporzionale e schedula il calcolo
    completo con Whisper in background, evitando di bloccare la chiamata HTTP per minuti.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    audio_path = lecture.audio_path
    cache_path = (audio_path + ".timestamps.json") if audio_path else None

    # Se esiste la cache su disco, restituiscila subito
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                lines = _normalize_karaoke_lines(data)
                return {"lines": lines, "segments": lines, "cached": True}
        except Exception:
            pass

    # Schedula il calcolo pesante con Whisper in background se non già in corso
    if audio_path and os.path.isfile(audio_path):
        with _pending_lock:
            if lecture_id not in _pending_timestamp_tasks:
                _pending_timestamp_tasks.add(lecture_id)
                background_tasks.add_task(_bg_compute_timestamps, lecture_id)

    # Restituisce immediatamente la stima proporzionale
    return compute_proportional_timestamps(lecture)


@router.delete("/{lecture_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    subject_id = lecture.subject_id
    audio_path = lecture.audio_path

    # 1. Eliminazione a cascata vettori LanceDB
    delete_lecture_vectors(subject_id=subject_id, lecture_id=lecture_id)

    # 2. Eliminazione file audio fisico registrato e relativa cache
    if audio_path:
        cache_ts = audio_path + ".timestamps.json"
        if os.path.isfile(cache_ts):
            try:
                os.remove(cache_ts)
            except Exception:
                pass
        if os.path.isfile(audio_path):
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
