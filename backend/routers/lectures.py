"""
routers/lectures.py — SbobinAI
Endpoint REST per la gestione delle Lezioni e Segnalibri.
"""
from typing import List
import os
import json
import threading
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import logging
import shutil

from database import Bookmark, Lecture, Subject, get_db
from schemas import BookmarkCreate, BookmarkOut, LectureCreate, LectureOut, LectureUpdate, LectureMove
from rag import delete_lecture_vectors

logger = logging.getLogger("sbobinai.lectures")
from transcript_cleaner import clean_transcript_for_llm

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
    updates = payload.model_dump(exclude_unset=True)
    needs_reindex = any(k in updates for k in ("title", "processed_notes", "raw_transcript"))
    for field, value in updates.items():
        if value is None and field in _NON_NULLABLE_FIELDS:
            continue
        setattr(lecture, field, value)
    db.commit()
    db.refresh(lecture)

    if needs_reindex:
        try:
            from rag import index_lecture
            index_lecture(
                subject_id=lecture.subject_id,
                lecture_id=lecture.id,
                title=lecture.title,
                transcript=lecture.raw_transcript or "",
                notes=lecture.processed_notes,
            )
        except Exception as reindex_err:
            logger.debug(f"[Lectures] Reindex LanceDB post-update: {reindex_err}")

    return lecture


@router.post("/{lecture_id}/move", response_model=LectureOut)
def move_lecture(lecture_id: int, payload: LectureMove, db: Session = Depends(get_db)):
    """Sposta una lezione da un corso a un altro.

    1. Verifica esistenza della lezione e del corso di destinazione.
    2. Rimuove i vettori semantici LanceDB dal vecchio corso e reindicizza per il nuovo corso.
    3. Sposta eventuali file fisici presenti nella cartella uploads del vecchio corso.
    4. Pulisce lo scope delle sessioni di chat del vecchio corso (rimuove la lezione dai filtri RAG).
    5. Aggiorna `lecture.subject_id` su SQLite.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    target_subject = db.query(Subject).filter(Subject.id == payload.target_subject_id).first()
    if not target_subject:
        raise HTTPException(status_code=404, detail="Corso di destinazione non trovato.")

    old_subject_id = lecture.subject_id
    new_subject_id = payload.target_subject_id

    if old_subject_id == new_subject_id:
        return lecture

    # 1. Rimuove vettori LanceDB dal vecchio corso
    try:
        from rag import delete_lecture_vectors
        delete_lecture_vectors(subject_id=old_subject_id, lecture_id=lecture.id)
    except Exception as e:
        logger.warning(f"[Move Lecture] Rimozione vettori LanceDB per materia {old_subject_id}: {e}")

    # 2. Sposta file fisici se presenti nella cartella uploads del vecchio corso
    try:
        from config import get_uploads_dir
        uploads_root = os.path.abspath(get_uploads_dir())
        old_dir = os.path.join(uploads_root, f"subject_{old_subject_id}")
        new_dir = os.path.join(uploads_root, f"subject_{new_subject_id}")

        if lecture.audio_path:
            abs_audio = os.path.abspath(lecture.audio_path)
            # startswith() considera erroneamente subject_10 contenuta in subject_1.
            # Confrontiamo il path normalizzato per componenti, non per prefisso testuale.
            try:
                abs_audio_relative = os.path.relpath(abs_audio, old_dir)
                is_in_old_dir = (
                    abs_audio_relative != os.pardir
                    and not abs_audio_relative.startswith(os.pardir + os.sep)
                    and not os.path.isabs(abs_audio_relative)
                )
            except ValueError:
                is_in_old_dir = False
            if is_in_old_dir:
                os.makedirs(new_dir, exist_ok=True)
                fname = os.path.basename(abs_audio)
                new_audio_path = os.path.join(new_dir, fname)
                if os.path.exists(abs_audio) and abs_audio != new_audio_path:
                    shutil.move(abs_audio, new_audio_path)
                    lecture.audio_path = new_audio_path

                # Sposta anche i timestamp e metadati collegati se presenti
                for extra_ext in (".timestamps.json", ".meta.json"):
                    old_extra = abs_audio + extra_ext
                    new_extra = new_audio_path + extra_ext
                    if os.path.exists(old_extra):
                        try:
                            shutil.move(old_extra, new_extra)
                        except Exception:
                            pass
    except Exception as e:
        logger.error(f"[Move Lecture] Errore spostamento file su disco: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Impossibile spostare i file audio della lezione sul disco: {e}",
        )

    # 3. Pulisce lo scope delle sessioni di chat del vecchio corso
    try:
        from database import ChatSession
        sessions = db.query(ChatSession).filter(ChatSession.subject_id == old_subject_id).all()
        for s in sessions:
            if not s.scope_json:
                continue
            try:
                s_scope = json.loads(s.scope_json)
                if isinstance(s_scope, dict):
                    lecs = s_scope.get("lecture_ids") or []
                    if lecture.id in lecs:
                        s_scope["lecture_ids"] = [lid for lid in lecs if lid != lecture.id]
                        s.scope_json = json.dumps(s_scope, ensure_ascii=False)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[Move Lecture] Errore pulizia scope chat vecchio corso: {e}")

    # 4. Aggiorna la lezione su SQLite
    lecture.subject_id = new_subject_id
    db.commit()
    db.refresh(lecture)

    # 5. Reindicizza i vettori nel nuovo corso se ci sono contenuti
    if lecture.raw_transcript or lecture.processed_notes or lecture.title:
        try:
            from rag import index_lecture, unload_embedding_model, clear_metal_cache
            try:
                index_lecture(
                    subject_id=new_subject_id,
                    lecture_id=lecture.id,
                    title=lecture.title,
                    transcript=lecture.raw_transcript or "",
                    notes=lecture.processed_notes,
                )
            finally:
                unload_embedding_model()
                clear_metal_cache()
        except Exception as e:
            logger.warning(f"[Move Lecture] Errore indicizzazione LanceDB nuovo corso {new_subject_id}: {e}")

    return lecture


@router.get("/{lecture_id}/audio")
def get_lecture_audio(lecture_id: int, db: Session = Depends(get_db)):
    """Serve il file audio originale della lezione (.wav / .mp3 / .m4a)."""
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture or not lecture.audio_path:
        raise HTTPException(status_code=404, detail="File audio non registrato per questa lezione.")

    audio_path = lecture.audio_path
    if not os.path.exists(audio_path):
        # Fallback difensivo nel caso in cui la compressione M4A in background abbia sostituito il file
        base, _ = os.path.splitext(audio_path)
        if os.path.exists(base + ".m4a"):
            audio_path = base + ".m4a"
            if lecture.audio_path != audio_path:
                try:
                    lecture.audio_path = audio_path
                    db.commit()
                except Exception:
                    pass
        elif os.path.exists(base + ".wav"):
            audio_path = base + ".wav"
        else:
            raise HTTPException(status_code=404, detail="File audio non trovato sul disco.")

    media_type = "audio/wav"
    if audio_path.endswith(".mp3"):
        media_type = "audio/mpeg"
    elif audio_path.endswith(".m4a"):
        media_type = "audio/mp4"
    elif audio_path.endswith(".aac"):
        media_type = "audio/aac"
    elif audio_path.endswith(".ogg"):
        media_type = "audio/ogg"

    return FileResponse(
        audio_path,
        media_type=media_type,
        filename=os.path.basename(audio_path),
        headers={"Accept-Ranges": "bytes"},
    )


def _normalize_karaoke_lines(raw_data) -> List[dict]:
    """Converte i dati grezzi nella struttura standard Apple Music (millisecondi)."""
    lines = []
    if isinstance(raw_data, dict):
        raw_data = raw_data.get("lines") or raw_data.get("segments") or []
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


_HALLUCINATION_FORMULAS = [
    "grazie a tutti",
    "grazie per la visione",
    "grazie per l'attenzione",
    "grazie dell'attenzione",
    "grazie per l'ascolto",
    "grazie dell'ascolto",
    "grazie per aver guardato",
    "grazie per averci seguito",
    "grazie di tutto",
    "grazie mille",
    "sottotitoli a cura di",
    "sottotitoli",
    "iscriviti al canale",
    "iscrivetevi al canale",
    "amara.org",
    "alla prossima puntata",
    "al prossimo video",
    "buona visione",
    "sigla iniziale",
    "sigla finale",
]


def _validate_timestamp_lines(lines: list, raw_transcript: str | None) -> tuple[bool, str]:
    """
    Valida la coerenza e l'integrità dei segmenti karaoke rispetto a lecture.raw_transcript.
    Rileva cache corrotte, formule allucinatorie Whisper ("grazie a tutti", "sottotitoli", ecc.),
    loop di ripetizione consecutivi o discrepanze lessicali rispetto alla vera trascrizione.
    Restituisce (True, "ok") se valido, oppure (False, motivo_del_fallimento).
    """
    if not lines or not isinstance(lines, list):
        return False, "lista linee vuota o non valida"

    line_texts = [
        " ".join(w.get("text", "") for w in line.get("words", [])).strip()
        for line in lines
        if isinstance(line, dict)
    ]
    line_texts = [lt for lt in line_texts if lt]
    if not line_texts:
        return False, "nessun testo estratto dalle linee"

    cache_full_text = " ".join(line_texts).strip()
    cache_lower = cache_full_text.lower().replace("’", "'")
    raw_text = (raw_transcript or "").strip()
    raw_lower = raw_text.lower().replace("’", "'")

    cache_words = [w for w in re.findall(r"\b\w+\b", cache_lower) if w]
    raw_words = [w for w in re.findall(r"\b\w+\b", raw_lower) if w]
    raw_word_count = len(raw_words)

    # 1. Controllo formule allucinatorie tipiche di Whisper se NON presenti nel testo reale raw_transcript
    for needle in _HALLUCINATION_FORMULAS:
        needle_norm = needle.lower().replace("’", "'")
        if needle_norm in cache_lower and needle_norm not in raw_lower:
            return False, f"formula allucinatoria '{needle}' non presente in raw_transcript"

    # 2. Se raw_transcript ha testo valido (>= 15 parole)
    if raw_word_count >= 15:
        # A. Rapporto parole cache vs raw_transcript (< 30%)
        if len(cache_words) < 0.3 * raw_word_count:
            return False, f"cache ha troppo poche parole ({len(cache_words)}) rispetto a raw_transcript ({raw_word_count})"

        # B. Ripetizioni consecutive identiche (>= 2)
        norm_raw_compact = " ".join(re.sub(r"[^\w\s]", " ", raw_lower).split())
        for idx in range(1, len(line_texts)):
            prev = " ".join(re.sub(r"[^\w\s]", " ", line_texts[idx - 1].lower()).split())
            curr = " ".join(re.sub(r"[^\w\s]", " ", line_texts[idx].lower()).split())
            if prev and curr and prev == curr and len(curr) >= 3:
                double_curr = f"{curr} {curr}"
                if double_curr not in norm_raw_compact or (idx >= 2 and curr == " ".join(re.sub(r"[^\w\s]", " ", line_texts[idx - 2].lower()).split())):
                    return False, f"ripetizione consecutiva identica rilevata: '{line_texts[idx]}'"

        # C. Sovrapposizione lessicale rispetto alle parole della vera trascrizione
        raw_words_set = set(w for w in raw_words if len(w) >= 3)
        if raw_words_set:
            cache_sig_words = [w for w in cache_words if len(w) >= 3]
            if cache_sig_words:
                overlap = sum(1 for w in cache_sig_words if w in raw_words_set) / len(cache_sig_words)
                if overlap < 0.35:
                    return False, f"sovrapposizione lessicale troppo bassa ({overlap:.1%}) rispetto a raw_transcript"
    else:
        # Se raw_transcript ha < 15 parole, controlla comunque ripetizioni identiche consecutive (>= 2)
        for idx in range(1, len(line_texts)):
            prev = " ".join(re.sub(r"[^\w\s]", " ", line_texts[idx - 1].lower()).split())
            curr = " ".join(re.sub(r"[^\w\s]", " ", line_texts[idx].lower()).split())
            if prev and curr and prev == curr and len(curr) >= 3:
                return False, f"ripetizione consecutiva identica rilevata in testo breve: '{line_texts[idx]}'"

    return True, "ok"


def compute_proportional_timestamps(lecture) -> dict:
    """
    Allineamento proporzionale su raw_transcript e duration_seconds.
    Restituisce una stima istantanea adatta come fallback o durante l'elaborazione in background.
    """
    raw_text = (lecture.raw_transcript or "").strip()
    if not raw_text:
        return {"lines": [], "segments": [], "cached": False}

    cleaned_text = clean_transcript_for_llm(raw_text).strip()
    words = [w for w in cleaned_text.split() if w]
    if not words:
        words = [w for w in raw_text.split() if w]
    if not words:
        return {"lines": [], "segments": [], "cached": False}

    duration_ms = int(round(float(lecture.duration_seconds or 30.0) * 1000))
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

        # 1. Controlla cache su disco esistente con validazione rigorosa
        if cache_path and os.path.isfile(cache_path):
            cache_is_corrupt = False
            lines = []
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    lines = _normalize_karaoke_lines(data)
                is_valid, reason = _validate_timestamp_lines(lines, lecture.raw_transcript)
                if not is_valid:
                    cache_is_corrupt = True
                    print(
                        f"[Timestamps] Cache su disco corrotta per lezione {lecture_id} ({reason}). "
                        "Rimuovo file e ricalcolo con allineamento proporzionale su raw_transcript."
                    )
                else:
                    return {"lines": lines, "segments": lines, "cached": True}
            except Exception as exc:
                cache_is_corrupt = True
                print(f"[Timestamps] Errore lettura cache su disco per lezione {lecture_id}: {exc}")

            if cache_is_corrupt:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
                clean_result = compute_proportional_timestamps(lecture)
                clean_lines = clean_result.get("lines", [])
                if cache_path and clean_lines:
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(clean_lines, f, ensure_ascii=False)
                    except Exception:
                        pass
                return clean_result

        # 2. Se abbiamo il file audio e Whisper è disponibile, calcola con Whisper in streaming a finestre
        if audio_path and os.path.isfile(audio_path):
            try:
                import ai_models
                from transcriber import (
                    _WHISPER_SPEC,
                    USE_MLX,
                    release_metal_cache,
                    _transcription_lock,
                    _iter_decoded_audio_windows,
                    _is_digital_silence,
                )
                model_path = ai_models.local_model_path(_WHISPER_SPEC)
                if model_path:
                    raw_converted = []
                    seg_counter = 0
                    cumulative_sec = 0.0
                    with _transcription_lock:
                        for chunk_audio, new_dur_sec, is_first in _iter_decoded_audio_windows(audio_path):
                            new_end_sec = cumulative_sec + new_dur_sec
                            chunk_dur = chunk_audio.size / 16000.0
                            window_start_sec = max(0.0, new_end_sec - chunk_dur)
                            cutoff_sec = 0.0 if is_first else cumulative_sec
                            try:
                                if _is_digital_silence(chunk_audio):
                                    continue
                                if USE_MLX:
                                    import mlx_whisper
                                    res = mlx_whisper.transcribe(
                                        chunk_audio,
                                        path_or_hf_repo=model_path,
                                        word_timestamps=True,
                                        language="it",
                                        temperature=(0.0,),
                                        condition_on_previous_text=False,
                                        fp16=True,
                                    )
                                    raw_segs = list(res.get("segments") or [])
                                else:
                                    from transcriber import faster_whisper_word_segments
                                    raw_segs = faster_whisper_word_segments(model_path, chunk_audio, language="it")

                                for seg in raw_segs:
                                    words_src = (seg.words if hasattr(seg, "words") else seg.get("words")) or []
                                    raw_s = getattr(seg, "start", None) if hasattr(seg, "start") else seg.get("start")
                                    raw_e = getattr(seg, "end", None) if hasattr(seg, "end") else seg.get("end")
                                    if raw_s is None and words_src:
                                        w0 = words_src[0]
                                        raw_s = getattr(w0, "start", 0.0) if hasattr(w0, "start") else w0.get("start", 0.0)
                                    if raw_e is None and words_src:
                                        wl = words_src[-1]
                                        raw_e = getattr(wl, "end", 0.0) if hasattr(wl, "end") else wl.get("end", 0.0)
                                    seg_start = float(raw_s or 0.0) + window_start_sec
                                    seg_end = float(raw_e or 0.0) + window_start_sec
                                    if seg_end <= cutoff_sec + 0.02:
                                        continue
                                    w_list = []
                                    for w in words_src:
                                        w_word = getattr(w, "word", None) if hasattr(w, "word") else w.get("word", "")
                                        w_start = float(getattr(w, "start", 0.0) if hasattr(w, "start") else w.get("start", 0.0)) + window_start_sec
                                        w_end = float(getattr(w, "end", 0.0) if hasattr(w, "end") else w.get("end", 0.0)) + window_start_sec
                                        if w_end <= cutoff_sec + 0.02:
                                            continue
                                        w_list.append({
                                            "text": str(w_word).strip(),
                                            "start": round(max(cutoff_sec, w_start), 3),
                                            "end": round(w_end, 3),
                                        })
                                    if w_list:
                                        raw_converted.append({
                                            "id": seg_counter,
                                            "start": round(max(cutoff_sec, seg_start), 3),
                                            "end": round(seg_end, 3),
                                            "words": w_list,
                                        })
                                        seg_counter += 1
                            finally:
                                cumulative_sec = new_end_sec
                                del chunk_audio
                                release_metal_cache()

                    lines = _normalize_karaoke_lines(raw_converted)

                    # Validazione rigorosa segmenti Whisper estratti
                    is_valid, reason = _validate_timestamp_lines(lines, lecture.raw_transcript)
                    if not is_valid and bool((lecture.raw_transcript or "").strip()):
                        print(
                            f"[Timestamps] Segmenti Whisper scartati per lezione {lecture_id} ({reason}). "
                            "Uso allineamento proporzionale basato su vera trascrizione raw_transcript."
                        )
                        clean_result = compute_proportional_timestamps(lecture)
                        clean_lines = clean_result.get("lines", [])
                        if cache_path and clean_lines:
                            try:
                                with open(cache_path, "w", encoding="utf-8") as f:
                                    json.dump(clean_lines, f, ensure_ascii=False)
                            except Exception:
                                pass
                        return clean_result

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
    Se il file cache non esiste o è corrotto, restituisce subito la stima proporzionale pulita
    e schedula il calcolo completo con Whisper in background se disponibile, evitando blocchi.
    """
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    audio_path = lecture.audio_path
    cache_path = (audio_path + ".timestamps.json") if audio_path else None

    # Se esiste la cache su disco, validala prima di restituirla
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                lines = _normalize_karaoke_lines(data)
                is_valid, reason = _validate_timestamp_lines(lines, lecture.raw_transcript)
                if is_valid:
                    return {"lines": lines, "segments": lines, "cached": True}
                else:
                    print(f"[Timestamps] Cache su disco non valida per lezione {lecture_id} ({reason}): eliminazione...")
                    try:
                        os.remove(cache_path)
                    except Exception:
                        pass
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass

    # Schedula il calcolo con Whisper in background se audio disponibile e non già in corso
    if audio_path and os.path.isfile(audio_path):
        with _pending_lock:
            if lecture_id not in _pending_timestamp_tasks:
                _pending_timestamp_tasks.add(lecture_id)
                background_tasks.add_task(_bg_compute_timestamps, lecture_id)

    # Restituisce immediatamente la stima proporzionale pulita a zero latenza.
    # Non salviamo la stima temporanea in cache_path se l'audio è presente,
    # altrimenti il task Whisper in background la scambierebbe per cache definitiva.
    result = compute_proportional_timestamps(lecture)
    if not (audio_path and os.path.isfile(audio_path)) and cache_path and result.get("lines"):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result["lines"], f, ensure_ascii=False)
        except Exception:
            pass
    return result


@router.delete("/{lecture_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lecture(lecture_id: int, db: Session = Depends(get_db)):
    lecture = db.query(Lecture).filter(Lecture.id == lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lezione non trovata.")

    subject_id = lecture.subject_id
    audio_path = lecture.audio_path

    # 1. Eliminazione a cascata vettori LanceDB
    delete_lecture_vectors(subject_id=subject_id, lecture_id=lecture_id)
    try:
        from rag import compact_lancedb
        compact_lancedb()
    except Exception:
        pass

    # 2. Elimina file e cache solo se appartengono allo storage dell'app.
    # audio_path è modificabile via API e non va trattato come trusted path.
    managed_path = False
    if audio_path:
        try:
            from cleaner import get_all_audio_directories, get_all_uploads_directories
            candidate = Path(audio_path).resolve()
            managed_roots = get_all_audio_directories() + get_all_uploads_directories()
            managed_path = any(candidate.is_relative_to(Path(root).resolve()) for root in managed_roots)
        except (OSError, ValueError):
            managed_path = False
    if audio_path and managed_path:
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

    # 4. Pulizia a cascata dello scope in tutte le sessioni di chat della materia
    try:
        from database import ChatSession
        sessions = db.query(ChatSession).filter(ChatSession.subject_id == subject_id).all()
        for s in sessions:
            if not s.scope_json:
                continue
            try:
                s_scope = json.loads(s.scope_json)
                if isinstance(s_scope, dict):
                    lecs = s_scope.get("lecture_ids") or []
                    if lecture_id in lecs:
                        s_scope["lecture_ids"] = [lid for lid in lecs if lid != lecture_id]
                        s.scope_json = json.dumps(s_scope, ensure_ascii=False)
            except Exception:
                pass
    except Exception as e:
        import logging
        logging.getLogger("sbobinai.lectures").warning(
            f"Errore pulizia scope chat per lezione {lecture_id}: {e}"
        )

    # 5. Eliminazione da SQLite
    db.delete(lecture)
    db.commit()



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
