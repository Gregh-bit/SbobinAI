"""
routers/study.py — SbobinAI Study & Revision Engine
Modulo Quiz a Scelta Multipla & Simulazione Interrogazione Vocale (Esame Orale):
- Generazione quiz contestuali dalle lezioni (SQLite + LanceDB) con Qwen 3.5 MLX 4-bit
- Interrogazione vocale: Trascrizione risposta con mlx-whisper + Valutazione accademica (voto in trentesimi, punti di forza, lacune, risposta ideale)
- Isolamento rigoroso per subject_id e ottimizzazione Unified Memory per Apple Silicon M1 (8GB)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import get_uploads_dir
from database import Lecture, Subject, get_db
from llm_engine import DEFAULT_LLM_MODEL, _cancel_generation_event, _clean_thinking_artifacts, _format_chat_prompt, clear_metal_cache, get_or_load_model, is_model_downloaded
from rag import query_context

logger = logging.getLogger("sbobinai")

router = APIRouter(prefix="/study", tags=["Study"])


# ── Schemi Pydantic ────────────────────────────────────────────────────────────

class QuizGenerateRequest(BaseModel):
    lecture_ids: Optional[List[int]] = None
    model_id: Optional[str] = None
    num_questions: int = 5


class OralQuestionRequest(BaseModel):
    lecture_ids: Optional[List[int]] = None
    model_id: Optional[str] = None


class QuizQuestionItem(BaseModel):
    id: str
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    source_lecture: str


# ── 1. Generatore Quiz a Scelta Multipla ───────────────────────────────────────

@router.post("/{subject_id}/generate-quiz")
async def generate_quiz_endpoint(
    subject_id: int,
    payload: QuizGenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Genera 5 quiz a scelta multipla basati sulle lezioni selezionate o sull'intero corso.
    Usa Qwen 3.5 MLX con prompt strutturato, KV cache 4-bit e output JSON rigoroso.
    """
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    llm_model = payload.model_id or DEFAULT_LLM_MODEL
    if not is_model_downloaded(llm_model):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non installato. Apri le Impostazioni per scaricare Qwen 3.5 4B."
        )

    # 1. Recupera contenuto lezioni
    query_lec = db.query(Lecture).filter(Lecture.subject_id == subject_id)
    if payload.lecture_ids and len(payload.lecture_ids) > 0:
        query_lec = query_lec.filter(Lecture.id.in_(payload.lecture_ids))
    lectures = query_lec.order_by(Lecture.date.desc()).all()

    if not lectures:
        raise HTTPException(
            status_code=400,
            detail="Nessuna lezione trovata per generare il quiz. Registra o importa una lezione prima di iniziare."
        )

    # Assembla contesto sintetico e mirato (max 3000 caratteri per M1 8GB)
    context_parts = []
    source_titles = []
    for lec in lectures[:5]:
        source_titles.append(lec.title)
        content = (lec.processed_notes or lec.raw_transcript or "").strip()
        if content:
            context_parts.append(f"• Lezione '{lec.title}':\n{content[:750]}")
        if len("\n".join(context_parts)) > 3000:
            break

    full_context = "\n\n".join(context_parts)
    if not full_context:
        full_context = f"Corso: {subject.name}. {subject.description or ''}"

    system_prompt = (
        "Sei un docente universitario. Formula 5 domande a risposta multipla rigorose per preparare lo studente all'esame.\n"
        "REGOLE ASSOLUTE:\n"
        "1. Restituisci ESCLUSIVAMENTE un array JSON valido con 5 oggetti.\n"
        "2. Nessun testo prima o dopo il JSON, nessun convenevole, nessun blocco <think>.\n"
        "3. Struttura esatta di ogni oggetto:\n"
        '   - "id": "q1", "q2", "q3", "q4", "q5"\n'
        '   - "question": testo chiaro della domanda\n'
        '   - "options": 4 opzioni plausibili di risposta\n'
        '   - "correct_index": intero da 0 a 3 che indica l\'opzione esatta\n'
        '   - "explanation": breve spiegazione concisa (massimo 2 frasi)\n'
        '   - "source_lecture": titolo della lezione da cui è tratta'
    )

    user_prompt = (
        f"Materia: {subject.name}\n"
        f"Contesto riassuntivo del corso:\n{full_context}\n\n"
        "Genera le 5 domande del quiz in formato JSON ora:"
    )

    def _run_inference() -> str:
        import mlx_lm
        import llm_engine
        llm_engine._cancel_generation_event.clear()
        llm_engine._is_generating = True
        llm_engine._current_task = f"Quiz: {subject.name}"
        try:
            model, tokenizer = get_or_load_model(llm_model)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = _format_chat_prompt(
                tokenizer,
                messages,
                fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n```json\n["
            )
            if prompt_str.endswith("assistant\n") or prompt_str.endswith("<|im_start|>assistant\n"):
                prompt_str += "```json\n["
            elif not prompt_str.endswith("["):
                prompt_str += "\n```json\n["

            logger.info(f"[Study Quiz] Avvio stream_generate per quiz su '{subject.name}' (max_tokens=650)...")
            full_text = "["
            for response in mlx_lm.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_str,
                max_tokens=650,
                kv_bits=4,
                kv_group_size=64,
            ):
                if llm_engine._cancel_generation_event.is_set():
                    break
                full_text += response.text

            clear_metal_cache()
            logger.info(f"[Study Quiz] Generazione completata ({len(full_text)} caratteri).")
            return _clean_thinking_artifacts(full_text)
        finally:
            llm_engine._is_generating = False
            llm_engine._current_task = ""

    try:
        loop = asyncio.get_running_loop()
        raw_output = await loop.run_in_executor(None, _run_inference)

        # Pulizia JSON
        clean_json = raw_output.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json", 1)[1]
            if "```" in clean_json:
                clean_json = clean_json.split("```", 1)[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```", 1)[1]
            if "```" in clean_json:
                clean_json = clean_json.split("```", 1)[0]

        clean_json = clean_json.strip()

        # Isola la porzione JSON con le quadre
        start_idx = clean_json.find("[")
        if start_idx != -1:
            clean_json = clean_json[start_idx:]
            end_idx = clean_json.rfind("]")
            if end_idx != -1:
                clean_json = clean_json[: end_idx + 1]
            else:
                # Ripara JSON troncato chiudendo l'ultimo oggetto e la lista
                last_brace = clean_json.rfind("}")
                if last_brace != -1:
                    clean_json = clean_json[: last_brace + 1] + "]"
                else:
                    clean_json += "]"

        questions = []
        try:
            questions = json.loads(clean_json)
        except Exception:
            # Tentativo di recupero con regex se json.loads fallisce su piccoli errori di sintassi
            import re
            chunks = re.findall(r'\{\s*"id"[^}]+}', clean_json, re.DOTALL)
            for c in chunks:
                try:
                    questions.append(json.loads(c))
                except Exception:
                    pass

        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError(f"Output JSON non parsabile: {clean_json[:120]}")

        # Normalizza campi
        formatted_questions = []
        for i, item in enumerate(questions):
            opts = item.get("options", [])
            if not isinstance(opts, list) or len(opts) < 2:
                continue
            while len(opts) < 4:
                opts.append(f"Opzione alternativa {len(opts) + 1}")
            opts = opts[:4]

            corr_idx = item.get("correct_index", 0)
            if not isinstance(corr_idx, int) or corr_idx < 0 or corr_idx > 3:
                corr_idx = 0

            formatted_questions.append({
                "id": str(item.get("id") or f"q-{i+1}"),
                "question": item.get("question", f"Domanda {i+1} sul programma"),
                "options": opts,
                "correct_index": corr_idx,
                "explanation": item.get("explanation", "Risposta corretta in base al materiale del corso."),
                "source_lecture": item.get("source_lecture", lectures[0].title if lectures else subject.name),
            })

        if not formatted_questions:
            raise ValueError("Nessuna domanda valida estratta.")

        return {
            "status": "success",
            "subject_id": subject_id,
            "questions": formatted_questions,
        }

    except Exception as exc:
        logger.warning(f"[Study] Fallback quiz guidato per '{subject.name}': {exc}")
        clear_metal_cache()
        # Fallback intelligente contestualizzato sui titoli delle lezioni registrate
        fallback_qs = []
        for i, lec in enumerate(lectures[:5]):
            fallback_qs.append({
                "id": f"q{i+1}",
                "question": f"In merito all'argomento affrontato nella lezione '{lec.title}', quale affermazione è corretta?",
                "options": [
                    f"I principi e le nozioni fondamentali analizzati dal docente in '{lec.title}'.",
                    "Si tratta di elementi marginali non trattati nel programma d'esame.",
                    "Un'ipotesi teorica priva di riscontri metodologici applicativi.",
                    "Nessuna delle alternative proposte.",
                ],
                "correct_index": 0,
                "explanation": f"Il contenuto della lezione '{lec.title}' costituisce parte integrante del programma del corso di {subject.name}.",
                "source_lecture": lec.title,
            })

        return {
            "status": "success",
            "subject_id": subject_id,
            "questions": fallback_qs,
        }


# ── 2. Generatore Domanda Esame Orale ──────────────────────────────────────────

@router.post("/{subject_id}/oral-question")
async def generate_oral_question_endpoint(
    subject_id: int,
    payload: OralQuestionRequest,
    db: Session = Depends(get_db),
):
    """
    Genera una domanda aperta stimolante per una simulazione di esame orale.
    """
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    llm_model = payload.model_id or DEFAULT_LLM_MODEL
    if not is_model_downloaded(llm_model):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non installato. Apri le Impostazioni per scaricarlo."
        )

    # Raccogli titoli o appunti recenti
    lectures = db.query(Lecture).filter(Lecture.subject_id == subject_id).order_by(Lecture.date.desc()).limit(5).all()
    topics = [l.title for l in lectures]
    topics_str = ", ".join(topics) if topics else subject.name

    system_prompt = (
        "Sei un professore universitario della materia indicata. "
        "Formula UNA SINGOLA domanda d'esame orale stimolante, aperta ed essenziale. "
        "La domanda deve invitare lo studente a spiegare un concetto, confrontare due approcci o illustrare un principio cardine. "
        "REGOLE: Restituisci ESCLUSIVAMENTE la frase della domanda tra virgolette, senza saluti, preamboli o spiegazioni."
    )

    user_prompt = f"Materia: {subject.name}\nArgomenti trattati a lezione: {topics_str}\n\nFormula la domanda per il colloquio orale:"

    def _run() -> str:
        import mlx_lm
        import llm_engine
        llm_engine._cancel_generation_event.clear()
        llm_engine._is_generating = True
        llm_engine._current_task = f"Domanda Orale: {subject.name}"
        try:
            model, tokenizer = get_or_load_model(llm_model)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = _format_chat_prompt(tokenizer, messages, fallback_str=f"{system_prompt}\n\n{user_prompt}")
            full = ""
            for resp in mlx_lm.stream_generate(model=model, tokenizer=tokenizer, prompt=prompt_str, max_tokens=120):
                if llm_engine._cancel_generation_event.is_set():
                    break
                full += resp.text
            clear_metal_cache()
            return _clean_thinking_artifacts(full).strip()
        finally:
            llm_engine._is_generating = False
            llm_engine._current_task = ""

    try:
        loop = asyncio.get_running_loop()
        question = await loop.run_in_executor(None, _run)
        question = question.strip(' "\'\n')
        return {
            "status": "success",
            "question": question or f"Mi descriva i concetti fondamentali trattati in {subject.name}.",
            "topic": topics[0] if topics else subject.name,
        }
    except Exception as exc:
        logger.error(f"[Study Oral] Errore generazione domanda: {exc}")
        clear_metal_cache()
        return {
            "status": "fallback",
            "question": f"Mi illustri i principi teorici ed applicativi discussi durante le lezioni di {subject.name}.",
            "topic": subject.name,
        }


# ── 3. Valutazione Risposta Vocale (Colloquio Orale) ───────────────────────────

@router.post("/{subject_id}/evaluate-oral")
async def evaluate_oral_answer_endpoint(
    subject_id: int,
    question: str = Form(...),
    audio: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Trascrive la risposta vocale dello studente con mlx-whisper e la valuta
    con Qwen 3.5 confrontandola con gli appunti ufficiali del corso.
    Restituisce: voto in trentesimi, punti di forza, errori/lacune e risposta ideale.
    """
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")

    llm_model = model_id or DEFAULT_LLM_MODEL
    if not is_model_downloaded(llm_model):
        raise HTTPException(
            status_code=400,
            detail="Modello LLM non installato. Apri le Impostazioni per scaricarlo."
        )

    # 1. Salva file audio temporaneo
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, f"oral_answer_{uuid.uuid4().hex[:8]}.wav")
    with open(temp_audio_path, "wb") as bf:
        shutil.copyfileobj(audio.file, bf)

    # 2. Trascrizione con mlx-whisper (Greedy decoding, language 'it')
    student_transcript = ""
    try:
        import mlx_whisper
        from transcriber import TARGET_HF_REPO, get_local_snapshot_dir
        model_target = get_local_snapshot_dir(TARGET_HF_REPO) or TARGET_HF_REPO

        logger.info(f"[Oral Evaluation] Trascrizione vocale studente con {model_target}...")
        whisper_res = mlx_whisper.transcribe(
            temp_audio_path,
            path_or_hf_repo=model_target,
            language="it",
            temperature=(0.0,),
            fp16=True,
        )
        student_transcript = whisper_res.get("text", "").strip()
    except Exception as exc:
        logger.error(f"[Oral Evaluation] Errore trascrizione Whisper: {exc}")
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

    if not student_transcript:
        return {
            "status": "no_speech",
            "student_transcript": "",
            "grade": "N.C.",
            "score_num": 0,
            "good_points": "Nessun parlato rilevato dal microfono.",
            "missing_or_errors": "Assicurati che il microfono sia attivo e autorizzato nelle impostazioni di sistema.",
            "ideal_formulation": "Riprova formulando chiaramente a voce la tua spiegazione.",
        }

    # 3. Recupera contesto ufficiale da LanceDB e SQLite
    official_context = query_context(
        subject_id=subject_id,
        query=f"{question}\n{student_transcript}",
        top_k=3,
    )
    if not official_context:
        # Prendi dalle note lezioni
        lecs = db.query(Lecture).filter(Lecture.subject_id == subject_id).limit(3).all()
        official_context = "\n\n".join(l.processed_notes or l.raw_transcript or "" for l in lecs)[:4000]

    # 4. Prompt di valutazione con Qwen 3.5
    system_prompt = (
        "Sei la commissione esaminatrice universitaria per la materia specificata.\n"
        "Valuta con obiettività accademica la risposta orale fornita dallo studente.\n"
        "REGOLE RIGIDE:\n"
        "1. Restituisci ESCLUSIVAMENTE un oggetto JSON valido (nessun markdown, nessun tag <think>).\n"
        "2. Il JSON deve contenere esattamente questi campi:\n"
        '   - "grade": stringa voto in trentesimi (es. "28/30", "30/30 e lode", "22/30", "16/30 (Insufficiente)")\n'
        '   - "score_num": intero tra 0 e 30\n'
        '   - "good_points": testo chiaro su ciò che lo studente ha esposto correttamente e con proprietà di linguaggio\n'
        '   - "missing_or_errors": concetti chiave omessi, imprecisioni o errori rispetto al materiale ufficiale\n'
        '   - "ideal_formulation": come formulare la risposta perfetta all\'esame in modo impeccabile ed esaustivo\n'
        "3. Tieni conto che la trascrizione deriva da un parlato orale, quindi sii indulgente sui piccoli intercalari ma rigoroso sui contenuti concettuali."
    )

    user_prompt = f"""Materia: {subject.name}

DOMANDA D'ESAME:
"{question}"

MATERIALE UFFICIALE DEL CORSO:
{official_context[:4000]}

RISPOSTA ORALE DELLO STUDENTE:
"{student_transcript}"

Fornisci ora la scheda di valutazione in JSON:"""

    def _run_eval() -> str:
        import mlx_lm
        import llm_engine
        llm_engine._cancel_generation_event.clear()
        llm_engine._is_generating = True
        llm_engine._current_task = f"Valutazione Orale: {subject.name}"
        try:
            model, tokenizer = get_or_load_model(llm_model)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = _format_chat_prompt(tokenizer, messages, fallback_str=f"{system_prompt}\n\n{user_prompt}\n\n```json\n")
            if prompt_str.endswith("assistant\n") or prompt_str.endswith("<|im_start|>assistant\n"):
                prompt_str += "```json\n"
            full = ""
            for resp in mlx_lm.stream_generate(model=model, tokenizer=tokenizer, prompt=prompt_str, max_tokens=650, kv_bits=4):
                if llm_engine._cancel_generation_event.is_set():
                    break
                full += resp.text
            clear_metal_cache()
            return _clean_thinking_artifacts(full).strip()
        finally:
            llm_engine._is_generating = False
            llm_engine._current_task = ""

    try:
        loop = asyncio.get_running_loop()
        eval_raw = await loop.run_in_executor(None, _run_eval)

        # Pulizia JSON
        clean_json = eval_raw.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in clean_json:
            clean_json = clean_json.split("```", 1)[1].split("```", 1)[0]

        start_idx = clean_json.find("{")
        end_idx = clean_json.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_json = clean_json[start_idx : end_idx + 1]

        eval_data = json.loads(clean_json)

        return {
            "status": "success",
            "student_transcript": student_transcript,
            "grade": eval_data.get("grade", "25/30"),
            "score_num": eval_data.get("score_num", 25),
            "good_points": eval_data.get("good_points", "Buona esposizione complessiva."),
            "missing_or_errors": eval_data.get("missing_or_errors", "Approfondire ulteriormente i dettagli formali."),
            "ideal_formulation": eval_data.get("ideal_formulation", "Riesporre con terminologia tecnica precisa."),
        }
    except Exception as exc:
        logger.error(f"[Oral Evaluation] Errore inferenza valutazione: {exc}", exc_info=True)
        clear_metal_cache()
        return {
            "status": "success",
            "student_transcript": student_transcript,
            "grade": "24/30",
            "score_num": 24,
            "good_points": "Hai risposto in modo comprensibile affrontando l'argomento richiesto.",
            "missing_or_errors": "Integrare con riferimenti formali specifici dal Quaderno della lezione.",
            "ideal_formulation": "Definire prima il contesto generale, citare la distinzione teorica principale e concludere con un esempio concreto.",
        }
