"""
routers/chats.py — SbobinAI Persistent Multi-Turn Chat, Compaction & Archive Router
Gestisce le sessioni di chat persistenti per materia, lo streaming NDJSON multi-turno
con RAG filtrato per scope, compattazione del contesto e archiviazione nel Quaderno.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import re
import threading
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import ChatMessage, ChatSession, Lecture, QuizSet, SessionLocal, StudentConcept, Subject, get_db
import llm_engine
import rag
from routers.study import (
    create_and_save_quiz_set_threadsafe,
    expand_and_save_quiz_set_threadsafe,
)
from schemas import (
    ChatArchiveRequest,
    ChatArchiveResponse,
    ChatMessageOut,
    ChatMessageSend,
    ChatRegenerateRequest,
    ChatSaveNoteRequest,
    ChatScope,
    ChatSessionCreate,
    ChatSessionDetailOut,
    ChatSessionOut,
    ChatSessionUpdate,
    ChatStreamRequest,
    ChatStudyAction,
    CitationRef,
    ContextStats,
    LegacyChatImportRequest,
)

logger = logging.getLogger("sbobinai.chats")

router = APIRouter(prefix="/chats", tags=["chats"])

_SYSTEM_RAG_OVERHEAD_TOKENS = 450
_QUIZ_INTENT_RE = re.compile(
    r"\b(quiz|crocett[ea]?|test\s+a\s+(?:risposta|scelta)\s+multipla|mettimi\s+alla\s+prova)\b|"
    r"\b(crea|crei|creami|crearmi|genera|generami|fammi|fai|faresti|facciamo|fammelo|prepara|prepari|prepararmi|voglio|vorrei|dammi|avvia|inizia|fai\s+un)\b.*\b(test|verific|domand[ea])\b",
    re.IGNORECASE,
)
_ORDINAL_WORD_MAP = {
    "prima": 1, "primo": 1, "1a": 1, "1ª": 1, "1°": 1, "1": 1,
    "seconda": 2, "secondo": 2, "2a": 2, "2ª": 2, "2°": 2, "2": 2,
    "terza": 3, "terzo": 3, "3a": 3, "3ª": 3, "3°": 3, "3": 3,
    "quarta": 4, "quarto": 4, "4a": 4, "4ª": 4, "4°": 4, "4": 4,
    "quinta": 5, "quinto": 5, "5a": 5, "5ª": 5, "5°": 5, "5": 5,
    "sesta": 6, "sesto": 6, "6a": 6, "6ª": 6, "6°": 6, "6": 6,
    "settima": 7, "settimo": 7, "7a": 7, "7ª": 7, "7°": 7, "7": 7,
    "ottava": 8, "ottavo": 8, "8a": 8, "8ª": 8, "8°": 8, "8": 8,
    "nona": 9, "nono": 9, "9a": 9, "9ª": 9, "9°": 9, "9": 9,
    "decima": 10, "decimo": 10, "10a": 10, "10ª": 10, "10°": 10, "10": 10,
    "ultima": -1, "ultimo": -1,
    "penultima": -2, "penultimo": -2,
}


def _clean_ordinal_token(raw_tok: str) -> str:
    t = raw_tok.strip().lower()
    if t.endswith(("ª", "°")):
        return t[:-1]
    if t.endswith("a") and t[:-1].isdigit():
        return t[:-1]
    return t


def _parse_quiz_target_question_num(text: str) -> Optional[int]:
    """Estrae l'indice 1-based (o -1 per ultima) della domanda di un quiz a cui si riferisce l'utente."""
    lowered = text.lower()
    # 1. 'prima/seconda/1a/1° domanda'
    m1 = re.search(r"\b(prima|seconda|terza|quarta|quinta|sesta|settima|ottava|nona|decima|ultima|penultima|\d+[ªa°]?)\s+domand[ae]\b", lowered)
    if m1:
        token = _clean_ordinal_token(m1.group(1))
        return _ORDINAL_WORD_MAP.get(token) or (int(token) if token.isdigit() else None)

    # 2. 'domanda 1', 'domanda numero 2', 'domanda prima'
    m2 = re.search(r"\bdomand[ae]\s+(?:numero\s+)?(prima|seconda|terza|quarta|quinta|sesta|settima|ottava|nona|decima|ultima|penultima|\d+[ªa°]?)\b", lowered)
    if m2:
        token = _clean_ordinal_token(m2.group(1))
        return _ORDINAL_WORD_MAP.get(token) or (int(token) if token.isdigit() else None)

    # 3. 'solo la prima', 'la prima non mi piace', 'la 1 falla semplice'
    m3 = re.search(r"\b(?:la|il)\s+(prima|seconda|terza|quarta|quinta|sesta|settima|ottava|nona|decima|ultima|penultima|\d+)(?!\s+(?:lezione|lezioni|volta|settimana|sbobina|registrazione|materia|parte|slide|pagina))\b", lowered)
    if m3:
        token = _clean_ordinal_token(m3.group(1))
        return _ORDINAL_WORD_MAP.get(token) or (int(token) if token.isdigit() else None)

    return None


_MODIFY_VERBS_RE = re.compile(
    r"\b(modifica|modificare|cambia|cambiare|aggiorna|aggiornare|riscrivi|riscrivere|rifai|rifarla|rifallo|rifare|rigenera|rigenerare|sostituisci|sostituire|semplifica|semplificare|più\s+semplice|più\s+facile|più\s+difficile|troppo\s+(?:difficile|facile|complicata|complessa|ostica)|non\s+mi\s+piace|falla\s+più|scrivila\s+più)\b",
    re.IGNORECASE,
)
_REMOVE_VERBS_RE = re.compile(
    r"\b(elimina|eliminare|cancella|cancellare|rimuovi|rimuovere|togli|togliere|leva|levare)\b",
    re.IGNORECASE,
)
_QUIZ_EXPAND_INTENT_RE = re.compile(
    r"\b(aggiungi|espandi|integra|metti|inserisci)\b.*\b(domand[ea]|quesiti|quiz)\b|"
    r"\b(altre\s+\d+\s+domande)\b|"
    r"\b(continua|prosegui)\b.*\b(quiz|domande)\b",
    re.IGNORECASE,
)
_QUIZ_DELETE_INTENT_RE = re.compile(
    r"\b(elimina|cancella|rimuovi|eliminarmi|cancellarmi)\b.*\b(quiz|test|crocette)\b|"
    r"\b(non\s+(?:lo\s+)?voglio\s+(?:più\s+)?questo\s+quiz)\b",
    re.IGNORECASE,
)
_QUIZ_REMOVE_QUESTION_INTENT_RE = re.compile(
    r"\b(elimina|cancella|rimuovi|togli|leva)\b.*\b(?:la\s+)?(?:domanda\s+(\d+|prima|seconda|terza|quarta|quinta|ultima)|(\d+|prima|seconda|terza|quarta|quinta|ultima)[ªa°]?\s+domanda)\b",
    re.IGNORECASE,
)
_QUIZ_MODIFY_QUESTION_INTENT_RE = re.compile(
    r"\b(modifica|cambia|aggiorna|riscrivi|semplifica|rifai|rifare|rigenera|sostituisci)\b.*\b(?:la\s+)?(?:domanda\s+(\d+|prima|seconda|terza|quarta|quinta|ultima)|(\d+|prima|seconda|terza|quarta|quinta|ultima)[ªa°]?\s+domanda)\b|"
    r"\b(?:la\s+)?(?:domanda\s+(\d+|prima|seconda|terza|quarta|quinta|ultima)|(\d+|prima|seconda|terza|quarta|quinta|ultima)[ªa°]?\s+domanda)\b.*\b(non\s+mi\s+piace|cambia|modifica|riscrivi|semplice|facile|difficile|rifai|rifarla|rifare|rigenera)\b",
    re.IGNORECASE,
)
_QUADERNO_SAVE_INTENT_RE = re.compile(
    r"\b(mett(?:i|ila|ilo|i pure|ila pure|ilo pure)|salva(?:lo|la|li)?|aggiungi(?:lo|la)?|scrivi(?:lo|la)?|esporta(?:lo|la)?|archivia(?:lo|la)?)\b.*\b(quaderno|appunti)\b|"
    r"\b(quaderno)\b.*\b(salva|metti|aggiungi)\b",
    re.IGNORECASE,
)
_QUIZ_REVIEW_INTENT_RE = re.compile(
    r"\b(ripassa|ripassare|rivediamo|rivedere|ho\s+sbagliato|perch[eé]|spiegami\s+il\s+quiz|risultat[io]|punteggio|correzione)\b",
    re.IGNORECASE,
)
_QUIZ_COMPLETED_INTENT_RE = re.compile(
    r"\b(l['’]ho\s+fatto|l['’]ho\s+finito|ho\s+fatto|ho\s+finito|fatto|finito|completato|risolto|fatto\s+il\s+quiz|finito\s+il\s+quiz|fatto\s+il\s+test|come\s+sono\s+andato|esito|risultati?)\b",
    re.IGNORECASE,
)
_UNDERSTOOD_INTENT_RE = re.compile(
    r"\b(ho\s+capito|tutto\s+chiaro|adesso\s+(?:ho\s+capito|è\s+chiaro)|ora\s+(?:ho\s+capito|è\s+chiaro)|chiarissimo|perfetto\s+grazie|ok\s+capito|ora\s+mi\s+è\s+chiaro|ho\s+afferrato|adesso\s+ho\s+afferrato|capito\s+tutto)\b",
    re.IGNORECASE,
)
_ORAL_INTENT_RE = re.compile(
    r"\b(interrogami|simulazione orale|domanda d['’]orale)\b",
    re.IGNORECASE,
)
_GREETING_INTENT_RE = re.compile(
    r"^(ciao|salve|buongiorno|buonasera|ehi|hey|hello|hi|buondì)(\s+(a tutti|sbobinai|tutor|tutti))?[!\?\.\s]*$",
    re.IGNORECASE,
)
_CITATION_LINK_RE = re.compile(r"\[\[(?:slide|ref|audio):([^\|#]+)(?:#([^\|]+))?\|([^\]]+)\]\]")


def _sanitize_hallucinated_citations(content: str, valid_sources: Set[str]) -> str:
    """Rimuove link interattivi verso sorgenti allucinate dall'LLM non presenti tra i materiali reali."""
    if not content:
        return ""
    clean_sources = {s.lower().strip() for s in valid_sources if s}
    def _repl(match):
        raw_target = match.group(1).strip().lower()
        label = match.group(3) or raw_target
        if not clean_sources or raw_target not in clean_sources:
            return label
        return match.group(0)
    return _CITATION_LINK_RE.sub(_repl, content)


_THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_FALLBACK_ERROR_SUBSTRINGS = (
    "il modello llm non è ancora stato scaricato",
    "nessun modello",
    "modello llm non ancora scaricato",
    "modello non scaricato",
    "non è ancora stato scaricato",
    "non è ancora stato scaricato in locale",
    "risposta non disponibile: modello llm non ancora scaricato",
)


def _is_fallback_error_message(content: str) -> bool:
    """Riconosce messaggi di errore o fallback dell'LLM che non devono pesare sul contesto."""
    if not content:
        return False
    lower = content.lower()
    return any(sub in lower for sub in _FALLBACK_ERROR_SUBSTRINGS)


def _clean_history_for_llm(messages: List[ChatMessage], compacted_up_to: int) -> List[Dict[str, str]]:
    """Costruisce la cronologia multi-turno per l'LLM escludendo i messaggi di fallback o notice del sistema."""
    clean: List[Dict[str, str]] = []
    for m in messages:
        if m.id <= compacted_up_to:
            continue
        if m.role == "assistant" and _is_fallback_error_message(m.content or ""):
            continue
        clean.append({"role": m.role, "content": m.content})
    return clean


def _is_llm_ready(model_id: Optional[str] = None) -> bool:
    """Verifica se il modello LLM risolto è realmente scaricato e pronto."""
    try:
        resolved = llm_engine.resolve_llm_model(model_id)
        return bool(resolved and llm_engine.is_model_downloaded(resolved))
    except Exception:
        return False


def _extract_explicit_new_question_text(user_text: str) -> Optional[str]:
    """Estrae l'eventuale formulazione letterale della nuova domanda richiesta dall'utente.
    Esempi:
      - 'cambia la prima domanda in: ti piace samuel?' -> 'ti piace samuel?'
      - 'sostituisci la prima domanda con "Qual è lo scopo del singleton?"' -> 'Qual è lo scopo del singleton?'
    """
    if not user_text:
        return None
    # 1. Con due punti dopo in / con / come / a / domanda
    m1 = re.search(
        r'(?:cambia|sostituisci|metti|fai|riformula|imposta|scrivi).*?\b(?:in|con|come|a|domanda)\s*:\s*["\'«]?(.*?)["\'»]?$',
        user_text,
        re.IGNORECASE | re.DOTALL,
    )
    if m1:
        cand = m1.group(1).strip()
        if len(cand) >= 2 and not cand.lower().startswith("modalità"):
            return cand

    # 2. Con virgolette dopo in / con / come
    m2 = re.search(
        r'\b(?:in|con|come)\s+["\'«]([^"\'»]+)["\'»]',
        user_text,
        re.IGNORECASE,
    )
    if m2:
        cand = m2.group(1).strip()
        if len(cand) >= 2:
            return cand

    return None


def _rewrite_single_quiz_question(
    *,
    old_item: dict,
    user_instruction: str,
    subject_name: str,
    context_text: str,
    llm_model: str,
    display_num: int,
    default_source_lecture: str = "",
    explicit_text_override: Optional[str] = None,
) -> Optional[dict]:
    """Riformula o sostituisce puntualmente una singola domanda del quiz seguendo l'istruzione dell'utente."""
    explicit_q = (explicit_text_override or "").strip() or _extract_explicit_new_question_text(user_instruction)
    old_q_text = str(old_item.get("question") or "") if isinstance(old_item, dict) else ""
    old_opts = old_item.get("options", []) if isinstance(old_item, dict) else []

    if explicit_q:
        sys_p = (
            f"Sei il modulo di aggiornamento quiz di SbobinAI per il corso di '{subject_name}'.\n"
            f"L'utente ha espressamente indicato il testo della nuova domanda {display_num}: \"{explicit_q}\".\n"
            "Il tuo compito è completare questa domanda generando 4 opzioni di risposta plausibili coerenti con essa, "
            "l'indice della risposta corretta (0-3) e una spiegazione concisa.\n"
            "REGOLE ASSOLUTE:\n"
            f"1. Il campo 'question' DEVE essere: \"{explicit_q}\".\n"
            "2. 'options': array di 4 stringhe con opzioni di risposta plausibili.\n"
            "3. 'correct_index': intero da 0 a 3 che indica l'opzione corretta.\n"
            "4. Restituisci ESCLUSIVAMENTE un singolo oggetto JSON valido (senza testo o markdown prima o dopo):\n"
            "   {\n"
            f'     "question": "{explicit_q}",\n'
            '     "concept": "concetto chiave in 1-3 parole",\n'
            '     "options": ["opzione 0", "opzione 1", "opzione 2", "opzione 3"],\n'
            '     "correct_index": 0,\n'
            '     "explanation": "spiegazione breve"\n'
            "   }"
        )
        user_p = (
            f"Materia: {subject_name}\n"
            f"Testo domanda: {explicit_q}\n"
            "Genera ora l'oggetto JSON per questa domanda:"
        )
    else:
        sys_p = (
            f"Sei il modulo di revisione quiz di SbobinAI per il corso di '{subject_name}'.\n"
            f"Devi modificare la domanda {display_num} di un quiz esistente in base alla richiesta dello studente.\n\n"
            "Domanda precedente:\n"
            f"{json.dumps(old_item, ensure_ascii=False, indent=2)}\n\n"
            f"Istruzione dello studente: \"{user_instruction}\"\n\n"
            "REGOLE TASSATIVE:\n"
            "1. Applica direttamente l'istruzione richiesta (es. semplifica la formulazione, rendila più diretta, adatta la difficoltà o il focus concettuale).\n"
            "2. Fornisci 4 opzioni di risposta plausibili, l'indice della risposta corretta (0-3) e una spiegazione concisa.\n"
            "3. Restituisci ESCLUSIVAMENTE un singolo oggetto JSON valido con la struttura:\n"
            "   {\n"
            '     "question": "testo aggiornato della domanda",\n'
            '     "concept": "concetto chiave specifico in 1-3 parole",\n'
            '     "options": ["opzione 0", "opzione 1", "opzione 2", "opzione 3"],\n'
            '     "correct_index": 0,\n'
            '     "explanation": "spiegazione concisa della risposta corretta"\n'
            "   }\n"
            "4. Nessun testo prima o dopo il JSON."
        )
        user_p = (
            f"Materia: {subject_name}\n"
            f"Domanda attuale: {old_q_text}\n"
            f"Istruzione: {user_instruction}\n"
            "Genera ora la domanda aggiornata in JSON:"
        )

    try:
        prefix_q = f'```json\n{{\n  "question": {json.dumps(explicit_q)},\n' if explicit_q else '```json\n{\n  "question": "'
        raw_output = llm_engine.generate_text(
            sys_p,
            user_p,
            model_id=llm_model,
            max_tokens=600,
            temperature=0.2,
            assistant_prefix=prefix_q,
            task_label=f"Modifica domanda {display_num}: {subject_name}",
        )
        raw_output = raw_output.strip()
        if "```" in raw_output:
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:].strip()
            raw_output = raw_output.rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_output)
        if isinstance(parsed, dict) and (parsed.get("question") or explicit_q):
            final_q = explicit_q if explicit_q else str(parsed.get("question")).strip()
            opts = parsed.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                opts = list(old_opts) if len(old_opts) >= 2 else ["Vero", "Falso", "Nessuna delle precedenti", "Tutte le precedenti"]
            opts = [str(o) for o in opts]
            while len(opts) < 4:
                opts.append(f"Opzione alternativa {len(opts) + 1}")
            opts = opts[:4]

            corr_idx = parsed.get("correct_index", 0)
            try:
                corr_idx = max(0, min(3, int(corr_idx)))
            except Exception:
                corr_idx = 0

            return {
                "id": str(old_item.get("id") or f"q-{display_num}"),
                "question": final_q,
                "concept": str(parsed.get("concept") or old_item.get("concept") or subject_name).strip(),
                "options": opts,
                "correct_index": corr_idx,
                "explanation": str(parsed.get("explanation") or "Risposta corretta in base all'argomento.").strip(),
                "source_lecture": str(old_item.get("source_lecture") or default_source_lecture),
            }
    except Exception as exc:
        logger.warning(f"[Chats] Riscrittura domanda {display_num} fallita: {exc}")

    if explicit_q:
        fallback_opts = list(old_opts) if len(old_opts) >= 2 else ["Opzione 1", "Opzione 2", "Opzione 3", "Opzione 4"]
        while len(fallback_opts) < 4:
            fallback_opts.append(f"Opzione {len(fallback_opts) + 1}")
        return {
            "id": str(old_item.get("id") or f"q-{display_num}"),
            "question": explicit_q,
            "concept": str(old_item.get("concept") or subject_name).strip(),
            "options": fallback_opts[:4],
            "correct_index": int(old_item.get("correct_index", 0)),
            "explanation": f"Domanda aggiornata a: {explicit_q}",
            "source_lecture": str(old_item.get("source_lecture") or default_source_lecture),
        }

    return None


_FAST_ACTION_TRIGGER_RE = re.compile(
    r"\b(quiz|test|crocett[ea]?|domand[ea]|verific[ah]|mettimi\s+alla\s+prova|esercit|orale|interrog|"
    r"quaderno|appunt[io]|salva|archivia|esporta|elimina|cancella|rimuovi|togli|leva|"
    r"modifica|cambia|correggi|sostituisc|riscriv|semplific|rifa|rigenera|aggiung|espandi|"
    r"pi[ùu]\s+(?:difficil|semplic|facil)|troppo\s+(?:difficil|facil)|finito|fatto|sbagliat|errore|punteggio|risultat)\b",
    re.IGNORECASE,
)
_ACTIVE_QUIZ_REF_RE = re.compile(
    r"\b(quella|questa|prima|seconda|terza|quarta|quinta|sesta|settima|ottava|nona|decima|ultima|penultima|\d+[ªa°]?)\b",
    re.IGNORECASE,
)


def _needs_llm_intent_classification(
    user_msg: str,
    effective_mode: str,
    session_id: int,
) -> bool:
    """
    Fast-path deterministico: evita una chiamata LLM preliminare su saluti ('ciao', 'buongiorno')
    e normali domande di studio senza intenti speciali (quiz, orale, quaderno, modifica quiz).
    """
    clean = (user_msg or "").strip()
    if effective_mode == "auto" and (not clean or _GREETING_INTENT_RE.match(clean)):
        return False
    if effective_mode != "auto":
        return True
    if _FAST_ACTION_TRIGGER_RE.search(clean):
        return True
    if _ACTIVE_QUIZ_REF_RE.search(clean):
        try:
            with SessionLocal() as db_q:
                has_quiz = (
                    db_q.query(QuizSet.id)
                    .filter(QuizSet.chat_session_id == session_id)
                    .first()
                )
                return has_quiz is not None
        except Exception:
            return False
    return False


def _classify_intent_with_llm(
    user_msg: str,
    subject_name: str,
    session_id: int,
    subject_id: int,
) -> Optional[Dict[str, Any]]:
    """Usa l'LLM per determinare semanticamente l'azione richiesta fornendo il contesto completo delle domande del quiz attivo."""
    try:
        resolved_model = llm_engine.resolve_llm_model(None)
        if not llm_engine.is_model_downloaded(resolved_model):
            # Fallback euristico se il modello LLM locale non è ancora scaricato
            lowered = user_msg.lower()
            if "quiz" in lowered or "test" in lowered:
                return {"action": "create_quiz", "num_questions": 5, "topic": user_msg.strip()}
            if "quaderno" in lowered or "appunt" in lowered:
                return {"action": "save_to_quaderno", "title": f"Appunti {subject_name}"}
            return {"action": "chat"}

        active_quiz: Optional[QuizSet] = None
        with SessionLocal() as db_q:
            active_quiz = (
                db_q.query(QuizSet)
                .filter(QuizSet.chat_session_id == session_id)
                .order_by(QuizSet.id.desc())
                .first()
            )

        quiz_context = "Nessun quiz attivo in questa chat."
        if active_quiz and active_quiz.questions_json:
            try:
                raw_qs = json.loads(active_quiz.questions_json)
                if isinstance(raw_qs, list) and raw_qs:
                    q_lines = [
                        f"{i+1}. \"{q.get('question', '') if isinstance(q, dict) else str(q)}\""
                        for i, q in enumerate(raw_qs)
                    ]
                    quiz_context = f"Quiz attivo '{active_quiz.title}' ({len(raw_qs)} domande):\n" + "\n".join(q_lines)
            except Exception:
                pass

        sys_p = (
            f"Sei il modulo decisionale e semantico di SbobinAI per il corso di '{subject_name}'.\n"
            "Comprendi la richiesta dell'utente ed emetti ESCLUSIVAMENTE un singolo oggetto JSON valido (senza testo o markdown prima o dopo).\n\n"
            "REGOLE CRITICHE:\n"
            "- Se l'utente chiede di fare, creare o ricevere un quiz o test (es. 'fai un quiz', 'mi fai un quiz', 'creami un quiz', 'quiz sull'argomento X', 'voglio fare delle domande'), l'azione è SEMPRE 'create_quiz'.\n"
            "- L'azione 'modify_quiz_question' si usa SOLO ED ESCLUSIVAMENTE se l'utente richiede di correggere, modificare o riscrivere una domanda ESISTENTE del quiz attivo (es. 'cambia la domanda 1', 'questa domanda è troppo difficile', 'quella sul cavo rifalla'). Non usarla MAI per richieste di fare un quiz.\n\n"
            "Azioni disponibili:\n"
            '- {"action": "chat"}\n'
            '  (per domande di teoria, spiegazioni, dubbi, saluti o conversazione normale)\n'
            '- {"action": "create_quiz", "num_questions": 5, "topic": "..."}\n'
            '  (se l\'utente chiede di creare o generare un nuovo quiz/test)\n'
            '- {"action": "modify_quiz_question", "question_index": 0, "new_question_text": "...", "instruction": "..."}\n'
            '  (se l\'utente vuole cambiare, modificare, semplificare, rifare o riscrivere una domanda del quiz attivo.\n'
            '   question_index: intero 0-based della domanda citata dall\'utente, es. 0 per la prima, 1 per la seconda, oppure l\'indice della domanda che corrisponde al tema citato dall\'utente.\n'
            '   new_question_text: se l\'utente specifica un nuovo testo esatto come "in: ti piace samuel?" o tra virgolette, inseriscilo qui, altrimenti null.\n'
            '   instruction: indicazione dell\'utente, es. "semplifica", "falla sul tema X", oppure null)\n'
            '- {"action": "remove_quiz_question", "question_index": 0}\n'
            '  (se l\'utente vuole eliminare o togliere una specifica domanda dal quiz attivo)\n'
            '- {"action": "delete_quiz"}\n'
            '  (se l\'utente vuole eliminare l\'intero quiz attivo)\n'
            '- {"action": "expand_quiz", "add_count": 3, "topic": "..."}\n'
            '  (se l\'utente vuole aggiungere altre domande al quiz attivo)\n'
            '- {"action": "save_to_quaderno", "title": "..."}\n'
            '  (se l\'utente chiede di salvare la spiegazione nel quaderno)\n'
            '- {"action": "oral", "topic": "..."}\n'
            '  (se l\'utente chiede una simulazione d\'orale)'
        )
        user_p = (
            f"Stato attuale:\n{quiz_context}\n\n"
            f'Richiesta utente: "{user_msg}"\n'
            "JSON:"
        )
        raw = llm_engine.generate_text(sys_p, user_p, model_id=resolved_model, max_tokens=100, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    except Exception as exc:
        logger.debug(f"[Chats Intent] Classificazione LLM non riuscita: {exc}")

    # Fallback sicuro se il modello LLM non ha risposto o output non parsabile
    lowered = user_msg.lower()
    if "quiz" in lowered or "test" in lowered:
        return {"action": "create_quiz", "num_questions": 5, "topic": user_msg.strip()}
    return {"action": "chat"}


def _execute_action_rollback(db: Session, action_json_str: str) -> Optional[str]:
    """Annulla e ripristina le modifiche a quiz/materiali se il turno associato viene eliminato o modificato."""
    if not action_json_str:
        return None
    try:
        data = json.loads(action_json_str) if isinstance(action_json_str, str) else action_json_str
        if not isinstance(data, dict):
            return None
        quiz_id = data.get("quiz_id")
        q_raw = data.get("question")
        if not quiz_id or not q_raw:
            return None

        try:
            q_payload = json.loads(q_raw) if isinstance(q_raw, str) else q_raw
        except Exception:
            return None

        if not isinstance(q_payload, dict):
            return None

        rb_type = q_payload.get("rollback_type")
        q_cand = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        if not q_cand:
            return None

        try:
            qs_list = json.loads(q_cand.questions_json or "[]")
        except Exception:
            qs_list = []

        if rb_type == "restore_question":
            q_idx = q_payload.get("question_index")
            old_q = q_payload.get("old_question")
            if q_idx is not None and 0 <= q_idx < len(qs_list) and old_q:
                qs_list[q_idx] = old_q
                q_cand.questions_json = json.dumps(qs_list, ensure_ascii=False)
                db.commit()
                logger.info(f"[Rollback] Ripristinata domanda {q_idx + 1} in quiz {quiz_id}")
                return f"Ripristinata la domanda {q_idx + 1} in **{q_cand.title}** allo stato precedente."
        elif rb_type == "restore_removed_question":
            q_idx = q_payload.get("question_index")
            old_q = q_payload.get("old_question")
            if q_idx is not None and old_q:
                qs_list.insert(q_idx, old_q)
                q_cand.questions_json = json.dumps(qs_list, ensure_ascii=False)
                db.commit()
                logger.info(f"[Rollback] Reinserita domanda rimossa al posto {q_idx + 1} in quiz {quiz_id}")
                return f"Reinserita la domanda rimossa in **{q_cand.title}**."
        elif rb_type == "delete_created_quiz":
            db.delete(q_cand)
            db.commit()
            logger.info(f"[Rollback] Eliminato quiz generato orfano {quiz_id}")
            return f"Annullata la creazione di **{q_cand.title}**."
    except Exception as exc:
        logger.warning(f"[Rollback] Errore durante rollback azione: {exc}")
    return None


# ── Helpers comuni: Lookup, Parsing, Serializzazione & Contesto ───────────────

def _get_subject_or_404(db: Session, subject_id: int) -> Subject:
    subject = db.query(Subject).filter(Subject.id == subject_id).first()
    if not subject:
        raise HTTPException(status_code=404, detail="Materia non trovata.")
    return subject


def _get_session_or_404(db: Session, chat_id: int) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessione chat non trovata.")
    return session


def _load_session_messages(db: Session, chat_id: int) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )


def _parse_scope(scope_json: Optional[str]) -> ChatScope:
    """Decodifica difensiva di scope_json in ChatScope."""
    if not scope_json:
        return ChatScope()
    try:
        raw = json.loads(scope_json)
        if not isinstance(raw, dict):
            return ChatScope()
        lecture_ids: List[int] = []
        for lid in raw.get("lecture_ids") or []:
            try:
                val = int(lid)
                if val > 0:
                    lecture_ids.append(val)
            except (TypeError, ValueError):
                continue
        material_ids = [
            str(mid).strip()
            for mid in (raw.get("material_ids") or [])
            if str(mid).strip()
        ]
        return ChatScope(lecture_ids=lecture_ids, material_ids=material_ids)
    except Exception:
        return ChatScope()


def _sanitize_scope(db: Optional[Session], subject_id: int, scope: ChatScope) -> ChatScope:
    """Filtra lecture_ids e material_ids orfani o appartenenti ad altre materie."""
    if not scope.lecture_ids and not scope.material_ids:
        return ChatScope()

    owns_db = db is None
    active_db = SessionLocal() if owns_db else db
    try:
        clean_lec_ids: List[int] = []
        if scope.lecture_ids:
            existing_lec = {
                row[0]
                for row in active_db.query(Lecture.id)
                .filter(Lecture.subject_id == subject_id, Lecture.id.in_(scope.lecture_ids))
                .all()
            }
            clean_lec_ids = [lid for lid in scope.lecture_ids if lid in existing_lec]

        clean_mat_ids: List[str] = []
        if scope.material_ids:
            import os
            from cleaner import get_all_uploads_directories
            folder_name = f"subject_{subject_id}"
            existing_files = set()
            for udir in get_all_uploads_directories():
                s_dir = os.path.join(udir, folder_name)
                if os.path.exists(s_dir):
                    for f in os.listdir(s_dir):
                        if not f.endswith(".meta.json"):
                            existing_files.add(f)
            for mid in scope.material_ids:
                clean_name = str(mid).strip()
                if (
                    clean_name in existing_files
                    or clean_name.startswith(f"infocorso-{subject_id}")
                    or clean_name.lower() == "infocorso.md"
                ):
                    clean_mat_ids.append(clean_name)

        return ChatScope(lecture_ids=clean_lec_ids, material_ids=clean_mat_ids)
    except Exception as exc:
        logger.warning(f"Errore sanitizzazione scope: {exc}")
        return scope
    finally:
        if owns_db:
            active_db.close()


def _extract_think_and_content(raw_text: str) -> Tuple[str, str]:
    """Separa i blocchi `<think>...</think>` dal testo visibile della risposta."""
    if not raw_text:
        return "", ""
    thinks: List[str] = [m.strip() for m in _THINK_BLOCK_RE.findall(raw_text) if m.strip()]
    visible = _THINK_BLOCK_RE.sub("", raw_text)
    if "<think>" in visible:
        head, tail = visible.split("<think>", 1)
        if tail.strip():
            thinks.append(tail.strip())
        visible = head
    return "\n\n".join(thinks).strip(), visible.strip()


def _build_context_stats(
    session: ChatSession,
    messages: List[ChatMessage],
    scope: ChatScope,
) -> ContextStats:
    """Calcola l'occupazione stimata del contesto rispetto a USABLE_CHAT_CONTEXT_TOKENS."""
    compacted_up_to = session.compacted_up_to_message_id or 0
    active_messages = [m for m in messages if m.id > compacted_up_to]
    compacted_count = max(0, len(messages) - len(active_messages))
    summary_text = (session.compacted_summary or "").strip()
    max_tokens = llm_engine.USABLE_CHAT_CONTEXT_TOKENS

    all_fallback = bool(active_messages) and (
        all(_is_fallback_error_message(m.content or "") for m in active_messages)
        or (
            any(m.role == "assistant" for m in active_messages)
            and all(_is_fallback_error_message(m.content or "") for m in active_messages if m.role == "assistant")
            and not summary_text
        )
    )

    if not _is_llm_ready() or all_fallback:
        return ContextStats(
            used_tokens=0,
            max_tokens=max_tokens,
            percentage=0.0,
            active_messages_count=len(active_messages),
            compacted_messages_count=compacted_count,
            has_compacted_summary=bool(summary_text),
            needs_compaction=False,
        )

    summary_tokens = llm_engine.estimate_tokens(summary_text) if summary_text else 0
    active_tokens = sum(
        (m.token_estimate if (m.token_estimate and m.token_estimate > 0) else llm_engine.estimate_tokens(m.content or ""))
        for m in active_messages
        if not _is_fallback_error_message(m.content or "")
    )

    has_active_or_summary = bool(
        summary_tokens > 0 or any(not _is_fallback_error_message(m.content or "") for m in active_messages)
    )
    scope_overhead = (
        _SYSTEM_RAG_OVERHEAD_TOKENS + (20 if (scope.lecture_ids or scope.material_ids) else 0)
        if has_active_or_summary
        else 0
    )
    used_tokens = summary_tokens + active_tokens + scope_overhead
    percentage = round(min(100.0, (used_tokens / max(1, max_tokens)) * 100.0), 1)

    return ContextStats(
        used_tokens=used_tokens,
        max_tokens=max_tokens,
        percentage=percentage,
        active_messages_count=len(active_messages),
        compacted_messages_count=compacted_count,
        has_compacted_summary=bool(summary_text),
        needs_compaction=percentage >= 75.0,
    )


def _serialize_message(m: ChatMessage) -> ChatMessageOut:
    """Converte ChatMessage ORM in ChatMessageOut parsando i campi JSON difensivamente."""
    citations: List[CitationRef] = []
    try:
        raw_cits = json.loads(m.citations_json or "[]")
        if isinstance(raw_cits, list):
            for c in raw_cits:
                if isinstance(c, dict) and c.get("source_id") is not None:
                    citations.append(
                        CitationRef(
                            source_id=str(c.get("source_id") or ""),
                            source_type=str(c.get("source_type") or "material"),
                            title=str(c.get("title") or "Materiale"),
                            page=c.get("page"),
                            label=str(c.get("label") or ""),
                        )
                    )
    except Exception:
        citations = []

    action: Optional[ChatStudyAction] = None
    if m.action_json:
        try:
            raw_action = json.loads(m.action_json)
            if isinstance(raw_action, dict) and raw_action.get("type"):
                action = ChatStudyAction(**raw_action)
        except Exception:
            action = None

    followups: List[str] = []
    try:
        raw_f = json.loads(m.followups_json or "[]")
        if isinstance(raw_f, list):
            followups = [str(x) for x in raw_f if str(x).strip()]
    except Exception:
        followups = []

    token_est = m.token_estimate if (m.token_estimate and m.token_estimate > 0) else llm_engine.estimate_tokens(m.content or "")

    return ChatMessageOut(
        id=m.id,
        session_id=m.session_id,
        role=m.role,
        content=m.content,
        thinking=m.thinking or "",
        mode=m.mode or "auto",
        context_used=m.context_used or "",
        citations=citations,
        action=action,
        followups=followups,
        token_estimate=token_est,
        created_at=m.created_at,
    )


def _build_session_base_kwargs(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
    message_count: Optional[int] = None,
) -> Dict[str, Any]:
    scope = _parse_scope(session.scope_json)
    if messages is not None:
        stats: Optional[ContextStats] = _build_context_stats(session, messages, scope)
        resolved_count = len(messages)
    elif message_count is not None:
        stats = None
        resolved_count = message_count
    else:
        msgs = list(session.messages)
        stats = _build_context_stats(session, msgs, scope)
        resolved_count = len(msgs)
    return {
        "id": session.id,
        "subject_id": session.subject_id,
        "title": session.title or "Nuova chat",
        "compacted_summary": session.compacted_summary or "",
        "compacted_up_to_message_id": session.compacted_up_to_message_id or 0,
        "scope": scope,
        "context_stats": stats,
        "message_count": resolved_count,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _serialize_session_summary(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
    message_count: Optional[int] = None,
) -> ChatSessionOut:
    return ChatSessionOut(**_build_session_base_kwargs(session, messages, message_count=message_count))


def _serialize_session_detail(
    session: ChatSession,
    messages: Optional[List[ChatMessage]] = None,
) -> ChatSessionDetailOut:
    msgs = list(session.messages) if messages is None else messages
    base = _build_session_base_kwargs(session, msgs)
    return ChatSessionDetailOut(
        **base,
        messages=[_serialize_message(m) for m in msgs],
    )


def _index_quaderno_lecture(
    subject_id: int,
    lecture_id: int,
    lecture_title: str,
    notes: str,
) -> None:
    """Scarica l'LLM prima di indicizzare nel RAG per sicurezza RAM su Apple Silicon 8GB."""
    llm_engine.unload_llm()
    try:
        rag.index_lecture(
            subject_id=subject_id,
            lecture_id=lecture_id,
            title=lecture_title,
            transcript="",
            notes=notes,
        )
    finally:
        try:
            rag.unload_embedding_model()
        except Exception:
            pass
        llm_engine.clear_metal_cache()


def _ndjson(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


# ── CRUD Sessioni & Messaggi ──────────────────────────────────────────────────

@router.get("/subject/{subject_id}", response_model=List[ChatSessionOut])
def list_chat_sessions(subject_id: int, db: Session = Depends(get_db)):
    """Elenca tutte le sessioni di chat di una materia, ordinate dalla più recente."""
    _get_subject_or_404(db, subject_id)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.subject_id == subject_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .all()
    )
    if not sessions:
        return []
    session_ids = [s.id for s in sessions]
    counts_rows = (
        db.query(ChatMessage.session_id, func.count(ChatMessage.id))
        .filter(ChatMessage.session_id.in_(session_ids))
        .group_by(ChatMessage.session_id)
        .all()
    )
    counts_by_session = {int(sid): int(cnt) for sid, cnt in counts_rows}
    return [
        _serialize_session_summary(s, message_count=counts_by_session.get(s.id, 0))
        for s in sessions
    ]


@router.post("/subject/{subject_id}", response_model=ChatSessionDetailOut)
def create_chat_session(
    subject_id: int,
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    """Crea una nuova sessione chat persistente per la materia."""
    _get_subject_or_404(db, subject_id)
    title = (payload.title or "Nuova chat").strip() or "Nuova chat"
    scope = payload.scope or ChatScope()
    session = ChatSession(
        subject_id=subject_id,
        title=title[:120],
        scope_json=json.dumps(scope.model_dump(), ensure_ascii=False),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _serialize_session_detail(session, [])


@router.post("/subject/{subject_id}/import-legacy", response_model=ChatSessionDetailOut)
def import_legacy_chat(
    subject_id: int,
    payload: LegacyChatImportRequest,
    db: Session = Depends(get_db),
):
    """Importa una conversazione legacy in una nuova ChatSession persistente."""
    _get_subject_or_404(db, subject_id)
    title = (payload.title or "Chat precedente").strip() or "Chat precedente"
    session = ChatSession(
        subject_id=subject_id,
        title=title[:120],
        scope_json="{}",
    )
    db.add(session)
    db.flush()

    inserted: List[ChatMessage] = []
    for item in payload.messages:
        role = (item.role or "").strip().lower()
        content = (item.content or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        msg = ChatMessage(
            session_id=session.id,
            role=role,
            content=content,
            thinking=(item.thinking or "").strip(),
            mode="auto",
            context_used=(item.context_used or "").strip(),
            citations_json="[]",
            action_json="",
            followups_json="[]",
            token_estimate=llm_engine.estimate_tokens(content),
        )
        db.add(msg)
        inserted.append(msg)

    db.commit()
    db.refresh(session)
    return _serialize_session_detail(session, inserted if inserted else [])


@router.get("/{chat_id}", response_model=ChatSessionDetailOut)
def get_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Restituisce il dettaglio completo della sessione chat e dei suoi messaggi."""
    session = _get_session_or_404(db, chat_id)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


@router.patch("/{chat_id}", response_model=ChatSessionDetailOut)
def update_chat_session(
    chat_id: int,
    payload: ChatSessionUpdate,
    db: Session = Depends(get_db),
):
    """Aggiorna il titolo e/o lo scope RAG della sessione chat."""
    session = _get_session_or_404(db, chat_id)
    if payload.title is not None:
        clean_title = payload.title.strip()
        if clean_title:
            session.title = clean_title[:120]
    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


@router.delete("/{chat_id}")
def delete_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Elimina una sessione chat e scollega in sicurezza i quiz (che rimangono nella materia)."""
    session = _get_session_or_404(db, chat_id)
    # Approccio A: i quiz generati in questa chat rimangono nella materia come quiz autonomi
    db.query(QuizSet).filter(QuizSet.chat_session_id == chat_id).update(
        {QuizSet.chat_session_id: None},
        synchronize_session=False,
    )
    db.delete(session)
    db.commit()
    return {"ok": True}


@router.delete("/{chat_id}/messages/{message_id}", response_model=ChatSessionDetailOut)
def delete_chat_message(chat_id: int, message_id: int, db: Session = Depends(get_db)):
    """Elimina il messaggio e tutti i messaggi successivi nella sessione (cancellazione a cascata)."""
    session = _get_session_or_404(db, chat_id)
    msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato nella sessione.")

    prev_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id, ChatMessage.id < message_id)
        .order_by(ChatMessage.id.desc())
        .first()
    )

    compacted_val = session.compacted_up_to_message_id or 0
    if compacted_val >= message_id:
        session.compacted_up_to_message_id = prev_msg.id if prev_msg else 0
    if not prev_msg:
        session.compacted_up_to_message_id = 0
        session.compacted_summary = None

    msgs_to_delete = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == chat_id, ChatMessage.id >= message_id)
        .order_by(ChatMessage.id.desc())
        .all()
    )
    for m in msgs_to_delete:
        if m.action_json:
            _execute_action_rollback(db, m.action_json)

    db.query(ChatMessage).filter(
        ChatMessage.session_id == chat_id,
        ChatMessage.id >= message_id,
    ).delete(synchronize_session=False)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    messages = _load_session_messages(db, chat_id)
    return _serialize_session_detail(session, messages)


# ── Compattazione & Archiviazione Quaderno ────────────────────────────────────

@router.post("/{chat_id}/compact", response_model=ChatSessionDetailOut)
async def compact_chat_session(chat_id: int, db: Session = Depends(get_db)):
    """Comprime tutti i turni attivi in `compacted_summary` fino all'ultimo messaggio attuale."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)
    messages = _load_session_messages(db, chat_id)

    if messages:
        compacted_up_to = session.compacted_up_to_message_id or 0
        active = [m for m in messages if m.id > compacted_up_to]
        to_compact = active if active else (messages if not session.compacted_summary else [])
        if to_compact:
            turns = [{"role": m.role, "content": m.content} for m in to_compact]
            try:
                await asyncio.to_thread(rag.unload_embedding_model)
            except Exception:
                pass
            summary = await asyncio.to_thread(
                llm_engine.compact_chat_history,
                subject.name,
                session.compacted_summary or "",
                turns,
            )
            session.compacted_summary = summary

        session.compacted_up_to_message_id = messages[-1].id
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)

    return _serialize_session_detail(session, messages)


@router.post("/{chat_id}/archive", response_model=ChatArchiveResponse)
async def archive_chat_to_quaderno(
    chat_id: int,
    payload: ChatArchiveRequest,
    db: Session = Depends(get_db),
):
    """Salva la conversazione (e la sintesi compattata) nel Quaderno e reindicizza nel RAG."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)
    messages = _load_session_messages(db, chat_id)

    note_title = (payload.title or "").strip()
    if not note_title:
        base = session.title if (session.title and session.title != "Nuova chat") else f"{subject.name} — Appunti chat"
        note_title = base
    note_title = note_title[:255]

    md_parts: List[str] = [f"# {note_title}"]
    if session.compacted_summary and session.compacted_summary.strip():
        md_parts.append(f"## Sintesi della sessione\n\n{session.compacted_summary.strip()}")

    if messages:
        conv_lines: List[str] = ["## Conversazione e Approfondimenti"]
        for m in messages:
            speaker = "Studente" if m.role == "user" else "Assistente SbobinAI"
            conv_lines.append(f"### {speaker}\n\n{m.content.strip()}")
        md_parts.append("\n\n".join(conv_lines))
    elif not (session.compacted_summary and session.compacted_summary.strip()):
        md_parts.append("Nessun messaggio presente nella sessione archiviata.")

    notes_md = "\n\n".join(md_parts)

    lecture = Lecture(
        subject_id=session.subject_id,
        title=note_title,
        raw_transcript="",
        processed_notes=notes_md,
        source_type="imported",
        in_quaderno=True,
    )
    db.add(lecture)
    db.commit()
    db.refresh(lecture)

    lecture_id = lecture.id
    lecture_title = lecture.title
    subject_id = session.subject_id
    await asyncio.to_thread(_index_quaderno_lecture, subject_id, lecture_id, lecture_title, notes_md)

    if payload.mode == "quaderno_and_clear":
        db.query(ChatMessage).filter(ChatMessage.session_id == chat_id).delete(synchronize_session=False)
        session.compacted_summary = ""
        session.compacted_up_to_message_id = 0
        session.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(session)
        messages = []

    return ChatArchiveResponse(
        lecture_id=lecture_id,
        title=lecture_title,
        session=_serialize_session_detail(session, messages),
    )


@router.post("/{chat_id}/save-note")
async def save_message_as_note(
    chat_id: int,
    payload: ChatSaveNoteRequest,
    db: Session = Depends(get_db),
):
    """Salva una singola risposta della chat come nota nel Quaderno e la indicizza nel RAG."""
    session = _get_session_or_404(db, chat_id)
    msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == payload.message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato.")

    note_title = (payload.title or "").strip()
    if not note_title:
        derived = llm_engine.generate_chat_title(msg.content)
        note_title = derived if derived != "Nuova chat" else f"{session.title} — Nota"
    note_title = note_title[:255]

    lecture = Lecture(
        subject_id=session.subject_id,
        title=note_title,
        raw_transcript="",
        processed_notes=msg.content,
        source_type="imported",
        in_quaderno=True,
    )
    db.add(lecture)
    db.commit()
    db.refresh(lecture)

    lecture_id = lecture.id
    lecture_title = lecture.title
    subject_id = session.subject_id
    await asyncio.to_thread(_index_quaderno_lecture, subject_id, lecture_id, lecture_title, msg.content)
    return {"lecture_id": lecture_id, "title": lecture_title}


# ── Streaming Multi-Turno & Rigenerazione ─────────────────────────────────────

async def _iter_sync_generator_async(
    gen_factory,
    stop_event: Optional[threading.Event] = None,
) -> AsyncIterator[str]:
    """Esegue un generatore sincrono in thread pool emettendo i chunk su asyncio.Queue."""
    queue: asyncio.Queue[Optional[Tuple[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop_flag = stop_event or threading.Event()

    def _emit(kind: str, val: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, val))
        except RuntimeError:
            stop_flag.set()

    def _worker() -> None:
        gen: Optional[Iterator[str]] = None
        try:
            gen = gen_factory()
            for chunk in gen:
                if stop_flag.is_set():
                    break
                if chunk:
                    _emit("token", chunk)
        except Exception as exc:
            if not stop_flag.is_set():
                _emit("error", exc)
        finally:
            if gen is not None and hasattr(gen, "close"):
                try:
                    gen.close()
                except Exception:
                    pass
            _emit("end", None)

    threading.Thread(target=_worker, daemon=True, name="chat-multi-turn-stream").start()

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, val = item
            if kind == "token":
                yield val
            elif kind == "error":
                raise val
            elif kind == "end":
                break
    finally:
        stop_flag.set()


def _extract_requested_quiz_count(text: str, default: int = 5) -> int:
    m = re.search(r"\b(\d{1,2})\s+(?:domand[ae]|quesit[oi])\b", text or "", re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:aggiungine|fanne|creane|metti|fai|inserisci)\s+(\d{1,2})\b", text or "", re.IGNORECASE)
    if m:
        try:
            return max(1, min(10, int(m.group(1))))
        except ValueError:
            pass
    return default


def _build_quiz_confirmation_markdown(
    quiz_title: str,
    num_questions: int,
    citations: List[CitationRef],
) -> str:
    lines = [
        f"Ho generato e salvato **{quiz_title}** con **{num_questions} domande** a scelta multipla basate sui materiali del corso.",
        "",
        "Puoi avviare subito la verifica dalla card qui sotto oppure ritrovarlo nella sezione **Ripasso**.",
    ]
    ref_links: List[str] = []
    for c in citations[:3]:
        if c.source_type == "lecture":
            sec = c.page or 0
            label = c.label or c.title
            ref_links.append(f"[[audio:{c.source_id}|{sec}|{label}]]")
        elif c.page and c.source_id:
            label = c.label or f"{c.title}, pag. {c.page}"
            ref_links.append(f"[[slide:{c.source_id}#{c.page}|{label}]]")
        elif c.title:
            ref_links.append(c.title)
    if ref_links:
        lines.append("")
        lines.append(f"**Fonti utilizzate:** {' · '.join(ref_links)}")
    return "\n".join(lines)


def _extract_oral_question(content: str, subject_name: str) -> str:
    """Estrae la domanda d'esame principale dal testo generato in modalità orale."""
    cleaned = (content or "").strip()
    if not cleaned:
        return f"Mi illustri i concetti fondamentali affrontati nel corso di {subject_name}."
    for line in cleaned.splitlines():
        stripped = line.strip().lstrip("#*- >\"'")
        if "?" in stripped and len(stripped) >= 12:
            q_part = stripped.split("?", 1)[0].strip() + "?"
            return q_part
    sentences = re.split(r"(?<=[?!.])\s+", cleaned)
    for s in sentences:
        if "?" in s and len(s.strip()) >= 12:
            return s.strip()
    first_line = cleaned.splitlines()[0].strip().lstrip("#*- >\"'")
    return first_line[:240] or f"Mi illustri i concetti fondamentali affrontati nel corso di {subject_name}."


def _build_context_summary(citations: List[CitationRef], context_text: str) -> str:
    titles = list(dict.fromkeys(c.title.strip() for c in citations if c.title and c.title.strip()))
    if titles:
        return " · ".join(titles)
    if (context_text or "").strip():
        return "Appunti e materiali del corso"
    return "Conoscenza generale"


def _format_quiz_context_for_llm(quiz: QuizSet) -> str:
    """Formatta i dettagli pedagogici di un quiz recente o completato per istruire l'LLM senza prompt leaking."""
    clean_title = (quiz.title or "Quiz").replace("Quiz:", "").strip()
    parts = [f"QUADRO ESITI QUIZ: '{clean_title}' (Punteggio: {quiz.score_percent or 0}%)"]
    if quiz.completed:
        parts.append(f"Stato: Completato | Punteggio: {quiz.score_percent or 0}%")
        try:
            results = json.loads(quiz.results_json or "[]")
            if results:
                wrong_items = [r for r in results if not r.get("is_correct")]
                correct_items = [r for r in results if r.get("is_correct")]
                parts.append(f"Bilancio: {len(correct_items)} corrette e {len(wrong_items)} errate su {len(results)} totali.")
                if wrong_items:
                    parts.append("CONCETTI SBAGLIATI DALLO STUDENTE (DA RISPIEGARE):")
                    for w in wrong_items:
                        q_txt = str(w.get("question") or "").strip()
                        sel_idx = w.get("selected_index", -1)
                        corr_idx = w.get("correct_index", -1)
                        opts = w.get("options") or []
                        sel_txt = opts[sel_idx] if 0 <= sel_idx < len(opts) else "Nessuna risposta"
                        corr_txt = opts[corr_idx] if 0 <= corr_idx < len(opts) else ""
                        expl = str(w.get("explanation") or "").strip()
                        parts.append(
                            f"- Concetto: \"{q_txt}\"\n"
                            f"  Risposta scelta dallo studente (ERRATA): \"{sel_txt}\"\n"
                            f"  Risposta CORRETTA: \"{corr_txt}\"\n"
                            f"  Spiegazione: {expl}"
                        )
                if correct_items:
                    parts.append("CONCETTI RISPOSTI CORRETTAMENTE:")
                    for c in correct_items:
                        parts.append(f"- \"{str(c.get('question') or '')[:60]}\"")
        except Exception:
            pass
    else:
        parts.append("Stato: Attivo / In corso.")
    return "\n".join(parts)


def _sanitize_quiz_feedback_leaks(content: str) -> str:
    """Rimuove eventuali allucinazioni di etichette interne o conclusioni fittizie da modelli compatti."""
    if not content:
        return content
    cleaned = re.sub(r"(?im)^#+\s*(?:📄\s*)?Quiz:\s*.*?-?\s*Verifica\s*Finale.*$", "", content)
    cleaned = re.sub(r"(?im)^\*?\*?Stato:\*?\*?\s*Completato.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^#+\s*(?:Istruzione\s+Tutor|Istruzione\s+di\s+risposta|Linee\s+guida).*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\*?\*?Nota:\*?\*?\s*(?:Ho\s+notato\s+che\s+)?la\s+risposta\s+corretta\s+è\s+sempre.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\*?\*?Nota:\*?\*?\s*Ho\s+notato\s+che\s+la\s+risposta\s+corretta.*$", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _persist_assistant_turn(
    *,
    session_id: int,
    replace_message_id: Optional[int],
    content: str,
    thinking: str,
    mode: str,
    context_used: str,
    citations_json: str,
    action_json: str,
    followups_json: str,
    token_estimate: int,
    scope: ChatScope,
) -> Dict[str, Any]:
    """Salva o aggiorna in-place la risposta assistente usando una SessionLocal dedicata."""
    local_db = SessionLocal()
    try:
        session = local_db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            raise ValueError(f"Sessione chat {session_id} non trovata durante il salvataggio.")

        now_utc = datetime.now(timezone.utc)
        target_msg: Optional[ChatMessage] = None
        if replace_message_id is not None:
            target_msg = (
                local_db.query(ChatMessage)
                .filter(ChatMessage.id == replace_message_id, ChatMessage.session_id == session_id)
                .first()
            )

        if target_msg is not None:
            if session.compacted_up_to_message_id == target_msg.id:
                prev_msg = (
                    local_db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session_id, ChatMessage.id < target_msg.id)
                    .order_by(ChatMessage.id.desc())
                    .first()
                )
                session.compacted_up_to_message_id = prev_msg.id if prev_msg else 0

            target_msg.content = content
            target_msg.thinking = thinking
            target_msg.mode = mode
            target_msg.context_used = context_used
            target_msg.citations_json = citations_json
            target_msg.action_json = action_json
            target_msg.followups_json = followups_json
            target_msg.token_estimate = token_estimate
            target_msg.created_at = now_utc
            assistant_msg = target_msg
        else:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                thinking=thinking,
                mode=mode,
                context_used=context_used,
                citations_json=citations_json,
                action_json=action_json,
                followups_json=followups_json,
                token_estimate=token_estimate,
            )
            local_db.add(assistant_msg)

        session.updated_at = now_utc
        local_db.commit()
        local_db.refresh(assistant_msg)
        local_db.refresh(session)

        updated_messages = _load_session_messages(local_db, session_id)
        stats_after = _build_context_stats(session, updated_messages, scope)
        return {
            "assistant_message": _serialize_message(assistant_msg).model_dump(mode="json"),
            "context_stats": stats_after.model_dump(mode="json"),
            "session_title": session.title or "Nuova chat",
        }
    except Exception:
        local_db.rollback()
        raise
    finally:
        local_db.close()


def _stream_chat_turn(
    *,
    session_id: int,
    subject_id: int,
    subject_name: str,
    compacted_summary: str,
    user_msg_dict: Dict[str, Any],
    user_msg_content: str,
    history_messages: List[Dict[str, str]],
    stats_before_dict: Dict[str, Any],
    mode: str,
    scope_lecture_ids: List[int],
    scope_material_ids: List[str],
    replace_message_id: Optional[int] = None,
    enable_thinking: bool = False,
    rollback_notice: Optional[str] = None,
) -> StreamingResponse:
    """Esegue la pipeline RAG + LLM streaming NDJSON condivisa fra `/stream` e `/regenerate-stream`."""
    effective_mode = (mode or "auto").strip().lower() or "auto"
    scope_snapshot = _sanitize_scope(
        None,
        subject_id,
        ChatScope(
            lecture_ids=list(scope_lecture_ids),
            material_ids=list(scope_material_ids),
        ),
    )
    had_scope = bool(scope_lecture_ids or scope_material_ids)
    if had_scope and not (scope_snapshot.lecture_ids or scope_snapshot.material_ids):
        # Se tutti gli elementi nello scope sono stati eliminati da disco/DB, fallback al corso intero
        logger.info(f"[Chats] Risorse nello scope eliminate: fallback al corso intero per chat {session_id}.")
        query_lec_ids: Optional[List[int]] = None
        query_mat_ids: Optional[List[str]] = None
    else:
        query_lec_ids = scope_snapshot.lecture_ids or None
        query_mat_ids = scope_snapshot.material_ids or None

    async def event_stream() -> AsyncIterator[str]:
        stream_stop_event = threading.Event()
        raw_buffer: List[str] = []
        citations: List[CitationRef] = []
        context_used: str = ""
        try:
            # 1. Risoluzione semantica unificata dell'intento (salta l'LLM per saluti e chat standard)
            if await asyncio.to_thread(
                _needs_llm_intent_classification,
                user_msg_content,
                effective_mode,
                session_id,
            ):
                action_plan = await asyncio.to_thread(
                    _classify_intent_with_llm,
                    user_msg=user_msg_content,
                    subject_name=subject_name,
                    session_id=session_id,
                    subject_id=subject_id,
                )
            else:
                action_plan = {"action": "chat"}
            if not action_plan or not isinstance(action_plan, dict):
                action_plan = {"action": "chat"}

            act = str(action_plan.get("action") or "chat").strip().lower()
            if effective_mode == "quiz" and act == "chat":
                act = "create_quiz"
            elif effective_mode == "oral" and act == "chat":
                act = "oral"

            # Salvaguardia: se l'azione è modify_quiz_question ma la richiesta è chiaramente di creare/fare un quiz senza citare modifiche o domande
            if act == "modify_quiz_question":
                lowered_msg = user_msg_content.lower()
                is_create_phrase = any(w in lowered_msg for w in ["fai un quiz", "fammi un quiz", "crea un quiz", "creami un quiz", "generami un quiz", "genera un quiz", "voglio un quiz", "preparami un quiz"])
                has_modify_explicit = any(w in lowered_msg for w in ["modifica", "cambia", "correggi", "sostituisci", "riscrivi", "domanda"])
                if is_create_phrase and not has_modify_explicit:
                    act = "create_quiz"

            is_quiz = (act == "create_quiz")
            is_oral = (act == "oral")
            is_greeting = (act == "chat") and bool(_GREETING_INTENT_RE.match(user_msg_content.strip()))
            is_understood = bool(_UNDERSTOOD_INTENT_RE.search(user_msg_content))
            is_pure_save = (act == "save_to_quaderno") and len(user_msg_content.strip()) < 40
            is_quiz_delete = (act == "delete_quiz")
            is_quiz_remove_q = (act == "remove_quiz_question")
            is_quiz_modify_q = (act == "modify_quiz_question")
            is_quiz_expand = (act == "expand_quiz")
            is_quiz_modify_whole = False
            is_quiz_review = False
            is_quiz_completed = False
            is_quaderno_save = (act == "save_to_quaderno")

            target_q_idx = action_plan.get("question_index")
            if target_q_idx is None and action_plan.get("question_number") is not None:
                try:
                    target_q_idx = int(action_plan["question_number"]) - 1
                except Exception:
                    target_q_idx = 0
            if target_q_idx is None:
                parsed_num = _parse_quiz_target_question_num(user_msg_content)
                target_q_idx = (parsed_num - 1) if (parsed_num is not None and parsed_num > 0) else 0

            new_q_text_explicit = action_plan.get("new_question_text") or _extract_explicit_new_question_text(user_msg_content)
            instruction_override = action_plan.get("instruction")
            is_greeting = (
                (not is_quiz)
                and (not is_quiz_delete)
                and (not is_quiz_remove_q)
                and (not is_quiz_modify_q)
                and (not is_quiz_modify_whole)
                and (not is_quiz_expand)
                and (not is_oral)
                and (not is_quiz_review)
                and (not is_understood)
                and (not is_pure_save)
                and bool(_GREETING_INTENT_RE.match(user_msg_content.strip()))
            )

            # Rilevamento automatico lezioni specifiche menzionate se lo scope non è stato impostato a mano
            active_lec_ids = list(query_lec_ids) if query_lec_ids else None
            if not active_lec_ids and user_msg_content:
                m_lec = re.search(r"\blezione\s+(\d+)\b", user_msg_content, re.IGNORECASE)
                if m_lec:
                    lec_num = int(m_lec.group(1))
                    with SessionLocal() as db_check:
                        matched_lec = (
                            db_check.query(Lecture)
                            .filter(
                                Lecture.subject_id == subject_id,
                                (Lecture.id == lec_num) | (Lecture.title.ilike(f"%lezione {lec_num}%"))
                            )
                            .first()
                        )
                        if matched_lec:
                            active_lec_ids = [matched_lec.id]
                            logger.info(f"[Chats Quiz] Rilevata lezione specifica {matched_lec.id} ('{matched_lec.title}') dal messaggio utente")

            current_target = llm_engine.resolve_llm_model(None)
            has_llm = llm_engine.is_model_downloaded(current_target)
            if not has_llm:
                # Risposta istantanea (<50ms): evita di caricare 300+ MB di embedding se nessun LLM è pronto
                yield _ndjson({
                    "type": "meta",
                    "user_message": user_msg_dict,
                    "context_used": "",
                    "citations": [],
                    "context_stats": stats_before_dict,
                })
                notice = (
                    "Il modello LLM non è ancora stato scaricato in locale.\n\n"
                    "Apri le **Impostazioni** e scarica il modello per abilitare l'assistente IA 100% offline."
                )
                yield _ndjson({"type": "token", "text": notice})
                persisted = await asyncio.to_thread(
                    _persist_assistant_turn,
                    session_id=session_id,
                    replace_message_id=replace_message_id,
                    content=notice,
                    thinking="",
                    mode=effective_mode,
                    context_used="",
                    citations_json="[]",
                    action_json="",
                    followups_json="[]",
                    token_estimate=llm_engine.estimate_tokens(notice),
                    scope=scope_snapshot,
                )
                yield _ndjson({
                    "type": "done",
                    "assistant_message": persisted["assistant_message"],
                    "followups": [],
                    "context_stats": persisted["context_stats"],
                    "session_title": persisted["session_title"],
                })
                return

            citations: List[CitationRef] = []
            if is_greeting:
                context_text = ""
                raw_citations = []
            else:
                rag_query = user_msg_content.strip()
                if is_quiz:
                    if rag_query:
                        rag_query = f"{subject_name} {rag_query}"
                    else:
                        rag_query = f"Argomenti e concetti d'esame principali di {subject_name}"

                rag_res = await asyncio.to_thread(
                    rag.query_context_structured,
                    subject_id=subject_id,
                    query=rag_query,
                    top_k=5,
                    scope_lecture_ids=active_lec_ids,
                    scope_material_ids=query_mat_ids,
                )
                try:
                    await asyncio.to_thread(rag.unload_embedding_model)
                except Exception:
                    pass
                llm_engine.clear_metal_cache()

                context_text = str(rag_res.get("context_text") or "")
                raw_citations = rag_res.get("citations") or []
                for c in raw_citations:
                    if isinstance(c, dict) and c.get("source_id") is not None:
                        citations.append(
                            CitationRef(
                                source_id=str(c.get("source_id") or ""),
                                source_type=str(c.get("source_type") or "material"),
                                title=str(c.get("title") or "Materiale"),
                                page=c.get("page"),
                                label=str(c.get("label") or ""),
                            )
                        )

                # Arricchimento contesto se lo studente sta discutendo o ripassando un quiz svolto
                try:
                    with SessionLocal() as db_q:
                        latest_completed_quiz = (
                            db_q.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id, QuizSet.chat_session_id == session_id)
                            .filter(QuizSet.completed.is_(True))
                            .order_by(QuizSet.created_at.desc(), QuizSet.id.desc())
                            .first()
                        )
                        if not latest_completed_quiz:
                            latest_completed_quiz = (
                                db_q.query(QuizSet)
                                .filter(QuizSet.subject_id == subject_id)
                                .filter(QuizSet.completed.is_(True))
                                .order_by(QuizSet.created_at.desc(), QuizSet.id.desc())
                                .first()
                            )
                        if latest_completed_quiz and (
                            is_quiz_review
                            or is_quiz_completed
                            or "quiz" in user_msg_content.lower()
                            or "punteggio" in user_msg_content.lower()
                            or "sbagliat" in user_msg_content.lower()
                            or "errore" in user_msg_content.lower()
                            or "risultat" in user_msg_content.lower()
                            or "fatto" in user_msg_content.lower()
                            or "finito" in user_msg_content.lower()
                        ):
                            quiz_ctx = _format_quiz_context_for_llm(latest_completed_quiz)
                            if quiz_ctx:
                                context_text = f"{context_text}\n\n{quiz_ctx}" if context_text else quiz_ctx
                except Exception as q_exc:
                    logger.debug(f"[Chats] Iniezione contesto quiz completato fallita: {q_exc}")

                # Via 2 (Conversazionale / Metodo Feynman): promozione concetto da_consolidare a consolidato
                if is_understood:
                    try:
                        with SessionLocal() as db_und:
                            cand_c = (
                                db_und.query(StudentConcept)
                                .filter(StudentConcept.subject_id == subject_id)
                                .filter(StudentConcept.status == "da_consolidare")
                                .order_by(StudentConcept.updated_at.desc())
                                .first()
                            )
                            if cand_c:
                                cand_c.status = "consolidato"
                                cand_c.correct_count += 1
                                cand_c.last_tested_at = datetime.now(timezone.utc)
                                db_und.commit()
                                concept_note = (
                                    f"=== REGISTRO DI PADRONANZA ACCADEMICA (AGGIORNATO) ===\n"
                                    f"Lo studente ha compreso e consolidato il concetto: '{cand_c.topic}'.\n"
                                    f"Nuovo stato nel database: CONSOLIDATO (Padronanza 100%).\n"
                                    f"[ISTRUZIONE: Conferma con entusiasmo e in poche parole che il concetto '{cand_c.topic}' "
                                    f"è stato registrato come consolidato nel suo profilo e che al prossimo quiz riceverà una domanda "
                                    f"nuova di verifica per consolidarlo definitivamente.]"
                                )
                                context_text = f"{context_text}\n\n{concept_note}" if context_text else concept_note
                    except Exception as und_exc:
                        logger.debug(f"[Chats] Promozione concetto Feynman fallita: {und_exc}")

            context_used = _build_context_summary(citations, context_text)

            yield _ndjson({
                "type": "meta",
                "user_message": user_msg_dict,
                "context_used": context_used,
                "citations": [c.model_dump(mode="json") for c in citations],
                "context_stats": stats_before_dict,
            })

            resolved_mode = effective_mode
            emitted_action: Optional[ChatStudyAction] = None

            if is_quiz_delete:
                target_quiz_id: Optional[int] = None
                target_quiz_title: str = f"Quiz su {subject_name}"
                with SessionLocal() as db_del:
                    q_cand = (
                        db_del.query(QuizSet)
                        .filter(QuizSet.chat_session_id == session_id)
                        .order_by(QuizSet.id.desc())
                        .first()
                    )
                    if not q_cand:
                        q_cand = (
                            db_del.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id)
                            .order_by(QuizSet.id.desc())
                            .first()
                        )
                    if q_cand:
                        target_quiz_id = q_cand.id
                        target_quiz_title = q_cand.title or target_quiz_title
                        db_del.delete(q_cand)
                        db_del.commit()

                if target_quiz_id is not None:
                    reply_md = f"Ho eliminato **{target_quiz_title}** dall'archivio."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})
                    emitted_action = ChatStudyAction(
                        type="quiz_deleted",
                        quiz_id=target_quiz_id,
                        title=target_quiz_title,
                    )
                    resolved_mode = "auto"
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                else:
                    reply_md = "Nessun quiz recente trovato da eliminare per questo corso."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

            elif is_quiz_remove_q:
                q_idx = target_q_idx
                display_num = q_idx + 1
                target_quiz_id: Optional[int] = None
                target_quiz_title: str = f"Quiz su {subject_name}"
                remaining_count = 0
                q_removed = False

                with SessionLocal() as db_rm:
                    q_cand = (
                        db_rm.query(QuizSet)
                        .filter(QuizSet.chat_session_id == session_id)
                        .order_by(QuizSet.id.desc())
                        .first()
                    )
                    if not q_cand:
                        q_cand = (
                            db_rm.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id)
                            .order_by(QuizSet.id.desc())
                            .first()
                        )
                    if q_cand:
                        target_quiz_id = q_cand.id
                        target_quiz_title = q_cand.title or target_quiz_title
                        try:
                            qs_list = json.loads(q_cand.questions_json or "[]")
                        except Exception:
                            qs_list = []
                        if 0 <= q_idx < len(qs_list):
                            removed_item = qs_list.pop(q_idx)
                            q_cand.questions_json = json.dumps(qs_list, ensure_ascii=False)
                            db_rm.commit()
                            remaining_count = len(qs_list)
                            q_removed = True

                if q_removed and target_quiz_id is not None:
                    reply_md = f"Ho rimosso la domanda {display_num} da **{target_quiz_title}**. Il quiz ora contiene **{remaining_count} domande**."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})
                    rollback_snapshot = json.dumps({
                        "rollback_type": "restore_removed_question",
                        "question_index": q_idx,
                        "old_question": removed_item,
                    }, ensure_ascii=False) if 'removed_item' in locals() and removed_item else None
                    emitted_action = ChatStudyAction(
                        type="quiz_created",
                        quiz_id=target_quiz_id,
                        title=target_quiz_title,
                        num_questions=remaining_count,
                        question=rollback_snapshot,
                    )
                    resolved_mode = "quiz"
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                else:
                    reply_md = f"Non è stato possibile individuare la domanda {display_num} da rimuovere dal quiz."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

            elif is_quiz_modify_q:
                q_idx = target_q_idx
                display_num = q_idx + 1
                target_quiz_id: Optional[int] = None
                target_quiz_title: str = f"Quiz su {subject_name}"
                q_modified = False
                new_q_text = ""
                remaining_count = 0

                with SessionLocal() as db_mod:
                    q_cand = (
                        db_mod.query(QuizSet)
                        .filter(QuizSet.chat_session_id == session_id)
                        .order_by(QuizSet.id.desc())
                        .first()
                    )
                    if not q_cand:
                        q_cand = (
                            db_mod.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id)
                            .order_by(QuizSet.id.desc())
                            .first()
                        )
                    if q_cand:
                        target_quiz_id = q_cand.id
                        target_quiz_title = q_cand.title or target_quiz_title
                        try:
                            qs_list = json.loads(q_cand.questions_json or "[]")
                        except Exception:
                            qs_list = []
                        if 0 <= q_idx < len(qs_list):
                            old_item = qs_list[q_idx]
                            from routers.study import _resolve_llm_model
                            llm_model = _resolve_llm_model(None)
                            new_item = _rewrite_single_quiz_question(
                                old_item=old_item,
                                user_instruction=instruction_override or user_msg_content,
                                subject_name=subject_name,
                                context_text=context_text,
                                llm_model=llm_model,
                                display_num=display_num,
                                default_source_lecture=target_quiz_title,
                                explicit_text_override=new_q_text_explicit,
                            )
                            if new_item and isinstance(new_item, dict):
                                qs_list[q_idx] = new_item
                                q_cand.questions_json = json.dumps(qs_list, ensure_ascii=False)
                                db_mod.commit()
                                q_modified = True
                                new_q_text = new_item.get("question", "")
                                remaining_count = len(qs_list)

                if q_modified and target_quiz_id is not None:
                    reply_md = (
                        f"Ho modificato la domanda {display_num} in **{target_quiz_title}**:\n\n"
                        f"> **{new_q_text}**\n\n"
                        f"Puoi avviare subito la verifica aggiornata dalla card qui sotto oppure nella sezione **Ripasso**."
                    )
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})
                    rollback_snapshot = json.dumps({
                        "rollback_type": "restore_question",
                        "question_index": q_idx,
                        "old_question": old_item,
                    }, ensure_ascii=False) if 'old_item' in locals() and old_item else None
                    emitted_action = ChatStudyAction(
                        type="quiz_created",
                        quiz_id=target_quiz_id,
                        title=target_quiz_title,
                        num_questions=remaining_count,
                        question=rollback_snapshot,
                    )
                    resolved_mode = "quiz"
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                else:
                    reply_md = f"Non è stato possibile modificare la domanda {display_num or ''} del quiz."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

            elif is_quiz_modify_whole:
                target_quiz_id: Optional[int] = None
                target_quiz_title: str = f"Quiz su {subject_name}"
                q_modified = False
                remaining_count = 0
                with SessionLocal() as db_mod_all:
                    q_cand = (
                        db_mod_all.query(QuizSet)
                        .filter(QuizSet.chat_session_id == session_id)
                        .order_by(QuizSet.id.desc())
                        .first()
                    )
                    if not q_cand:
                        q_cand = (
                            db_mod_all.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id)
                            .order_by(QuizSet.id.desc())
                            .first()
                        )
                    if q_cand:
                        target_quiz_id = q_cand.id
                        target_quiz_title = q_cand.title or target_quiz_title
                        from routers.study import _generate_quiz_items, _resolve_llm_model
                        llm_model = _resolve_llm_model(None)
                        try:
                            curr_list = json.loads(q_cand.questions_json or "[]")
                        except Exception:
                            curr_list = []
                        req_n = _extract_requested_quiz_count(user_msg_content, default=len(curr_list) or 5)
                        ai_new_title, new_items = _generate_quiz_items(
                            subject_name=subject_name,
                            full_context=context_text,
                            num_questions=req_n,
                            llm_model=llm_model,
                            default_source_lecture=target_quiz_title,
                            topic_hint=user_msg_content,
                        )
                        if new_items:
                            if ai_new_title:
                                q_cand.title = ai_new_title
                                target_quiz_title = ai_new_title
                            q_cand.questions_json = json.dumps(new_items, ensure_ascii=False)
                            q_cand.num_questions = len(new_items)
                            db_mod_all.commit()
                            q_modified = True
                            remaining_count = len(new_items)

                if q_modified and target_quiz_id is not None:
                    reply_md = (
                        f"Ho aggiornato **{target_quiz_title}** con **{remaining_count} nuove domande** seguendo le tue istruzioni.\n\n"
                        f"Puoi avviare subito la verifica dalla card qui sotto oppure nella sezione **Ripasso**."
                    )
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})
                    emitted_action = ChatStudyAction(
                        type="quiz_created",
                        quiz_id=target_quiz_id,
                        title=target_quiz_title,
                        num_questions=remaining_count,
                    )
                    resolved_mode = "quiz"
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                else:
                    reply_md = "Non è stato possibile modificare il quiz."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

            elif is_pure_save:
                last_content = None
                with SessionLocal() as db_find:
                    last_msg = (
                        db_find.query(ChatMessage)
                        .filter(ChatMessage.session_id == session_id, ChatMessage.role == "assistant")
                        .order_by(ChatMessage.id.desc())
                        .first()
                    )
                    if last_msg and last_msg.content:
                        last_content = last_msg.content

                if last_content:
                    derived_title = llm_engine.generate_chat_title(last_content[:200])
                    if not derived_title or derived_title == "Nuova chat":
                        derived_title = f"{subject_name} — Appunti"
                    derived_title = derived_title[:255]

                    with SessionLocal() as db_q:
                        saved_lec = Lecture(
                            subject_id=subject_id,
                            title=derived_title,
                            raw_transcript="",
                            processed_notes=last_content,
                            source_type="imported",
                            in_quaderno=True,
                        )
                        db_q.add(saved_lec)
                        db_q.commit()
                        db_q.refresh(saved_lec)
                        lec_id = saved_lec.id
                        lec_title = saved_lec.title

                    await asyncio.to_thread(_index_quaderno_lecture, subject_id, lec_id, lec_title, last_content)
                    reply_md = f"Ho salvato la spiegazione precedente nel tuo **Quaderno** con il titolo **{lec_title}**."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})
                    emitted_action = ChatStudyAction(
                        type="saved_to_quaderno",
                        lecture_id=lec_id,
                        title=lec_title,
                    )
                    yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                else:
                    reply_md = "Nessuna spiegazione precedente trovata da salvare nel Quaderno."
                    raw_buffer.append(reply_md)
                    yield _ndjson({"type": "token", "text": reply_md})

            elif is_quiz_expand:
                target_quiz_id: Optional[int] = None
                with SessionLocal() as db_search:
                    q_cand = (
                        db_search.query(QuizSet)
                        .filter(QuizSet.chat_session_id == session_id)
                        .order_by(QuizSet.id.desc())
                        .first()
                    )
                    if not q_cand:
                        q_cand = (
                            db_search.query(QuizSet)
                            .filter(QuizSet.subject_id == subject_id)
                            .order_by(QuizSet.id.desc())
                            .first()
                        )
                    if q_cand:
                        target_quiz_id = q_cand.id

                if target_quiz_id is not None:
                    add_cnt = action_plan.get("add_count")
                    try:
                        requested_n = int(add_cnt) if add_cnt else _extract_requested_quiz_count(user_msg_content, default=3)
                    except Exception:
                        requested_n = _extract_requested_quiz_count(user_msg_content, default=3)
                    try:
                        quiz_info = await asyncio.to_thread(
                            expand_and_save_quiz_set_threadsafe,
                            subject_id=subject_id,
                            quiz_id=target_quiz_id,
                            num_new_questions=requested_n,
                            context_text=context_text,
                            topic_hint=str(action_plan.get("topic") or user_msg_content).strip(),
                            subject_name=subject_name,
                        )
                        new_total = int(quiz_info.get("num_questions") or requested_n)
                        new_added = int(quiz_info.get("new_added") or requested_n)
                        quiz_title = str(quiz_info.get("title") or f"Quiz su {subject_name}")
                        reply_md = (
                            f"Ho aggiunto **{new_added} nuove domande** a **{quiz_title}**!\n\n"
                            f"Il quiz ora contiene in totale **{new_total} domande** di verifica basate sui materiali del corso.\n"
                            f"Puoi continuare subito il ripasso dalla scheda qui sotto."
                        )
                        raw_buffer.append(reply_md)
                        yield _ndjson({"type": "token", "text": reply_md})

                        emitted_action = ChatStudyAction(
                            type="quiz_created",
                            quiz_id=int(quiz_info["id"]),
                            title=quiz_title,
                            num_questions=new_total,
                        )
                        resolved_mode = "quiz"
                        yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                    except Exception as exp_exc:
                        logger.warning(f"[Chats] Espansione quiz {target_quiz_id} fallita: {exp_exc}")
                        is_quiz_expand = False
                else:
                    is_quiz_expand = False

            if not (is_quiz_delete or is_quiz_remove_q or is_quiz_modify_q or is_pure_save or is_quiz_expand):
                if is_quiz:
                    req_cnt = action_plan.get("num_questions")
                    try:
                        requested_n = int(req_cnt) if req_cnt else _extract_requested_quiz_count(user_msg_content, default=5)
                    except Exception:
                        requested_n = _extract_requested_quiz_count(user_msg_content, default=5)
                    topic_hint = str(action_plan.get("topic") or user_msg_content).strip()
                    try:
                        quiz_info = await asyncio.to_thread(
                            create_and_save_quiz_set_threadsafe,
                            subject_id=subject_id,
                            chat_session_id=session_id,
                            num_questions=requested_n,
                            context_text=context_text,
                            topic_hint=topic_hint,
                            source_label=context_used,
                            scope_lecture_ids=active_lec_ids,
                            scope_material_ids=query_mat_ids,
                            subject_name=subject_name,
                        )
                        num_saved = int(quiz_info.get("num_questions") or requested_n)
                        quiz_title = str(quiz_info.get("title") or f"Quiz su {subject_name}")
                        reply_md = _build_quiz_confirmation_markdown(quiz_title, num_saved, citations)
                        raw_buffer.append(reply_md)
                        yield _ndjson({"type": "token", "text": reply_md})

                        rollback_snapshot = json.dumps({
                            "rollback_type": "delete_created_quiz",
                            "quiz_id": int(quiz_info["id"]),
                        }, ensure_ascii=False)
                        emitted_action = ChatStudyAction(
                            type="quiz_created",
                            quiz_id=int(quiz_info["id"]),
                            title=quiz_title,
                            num_questions=num_saved,
                            question=rollback_snapshot,
                        )
                        resolved_mode = "quiz"
                        yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})
                    except Exception as quiz_exc:
                        logger.warning(f"[Chats] Generazione quiz fallita, fallback a risposta testuale: {quiz_exc}")
                        resolved_mode = "auto"
                        async for chunk in _iter_sync_generator_async(
                            lambda: llm_engine.stream_multi_turn_chat(
                                subject_name=subject_name,
                                messages=history_messages,
                                context_text=context_text,
                                compacted_summary=compacted_summary,
                                mode="auto",
                                enable_thinking=enable_thinking,
                            ),
                            stop_event=stream_stop_event,
                        ):
                            raw_buffer.append(chunk)
                            yield _ndjson({"type": "token", "text": chunk})
                else:
                    stream_mode = "oral" if is_oral else effective_mode
                    if is_oral:
                        resolved_mode = "oral"

                    adapted_history = list(history_messages)
                    if is_quiz_completed and adapted_history:
                        last_m = dict(adapted_history[-1])
                        last_m["content"] = (
                            f"{user_msg_content.strip()} (Ho completato il quiz: dimmi con calore come sono andato e spiegami con parole tue e in modo semplice i concetti errati, senza ristampare le domande)"
                        )
                        adapted_history[-1] = last_m

                    async for chunk in _iter_sync_generator_async(
                        lambda: llm_engine.stream_multi_turn_chat(
                            subject_name=subject_name,
                            messages=adapted_history,
                            context_text=context_text,
                            compacted_summary=compacted_summary,
                            mode=stream_mode,
                            enable_thinking=enable_thinking,
                        ),
                        stop_event=stream_stop_event,
                    ):
                        raw_buffer.append(chunk)
                        yield _ndjson({"type": "token", "text": chunk})

            raw_full = "".join(raw_buffer)
            thinking, clean_content = _extract_think_and_content(raw_full)
            if not clean_content:
                clean_content = "Risposta completata."

            valid_sources = {c.source_id for c in citations}
            if active_lec_ids:
                valid_sources |= {str(lid) for lid in active_lec_ids}
            if query_mat_ids:
                valid_sources |= {str(mid) for mid in query_mat_ids}

            # Includi tutti i materiali fisici reali presenti per la materia per consentire riferimenti corretti
            try:
                from routers.subjects import list_materials as _get_real_materials
                real_mats = _get_real_materials(subject_id, db=db)
                for rm in real_mats:
                    valid_sources.add(rm.name)
                    valid_sources.add(rm.id)
                    if rm.lecture_id:
                        valid_sources.add(str(rm.lecture_id))
            except Exception:
                pass

            clean_content = _sanitize_hallucinated_citations(clean_content, valid_sources)
            clean_content = _sanitize_quiz_feedback_leaks(clean_content)

            if is_oral and emitted_action is None:
                extracted_q = _extract_oral_question(clean_content, subject_name)
                topic_str = llm_engine.generate_chat_title(user_msg_content)
                emitted_action = ChatStudyAction(
                    type="oral_question",
                    question=extracted_q,
                    topic=topic_str if topic_str != "Nuova chat" else subject_name,
                )
                yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})

            if is_quaderno_save and not is_pure_save and clean_content and emitted_action is None:
                derived_title = llm_engine.generate_chat_title(clean_content[:300])
                if not derived_title or derived_title == "Nuova chat" or any(w in derived_title.lower() for w in ["spiega", "metti", "salva"]):
                    derived_title = f"{subject_name} — Concetto"
                derived_title = derived_title[:255]

                with SessionLocal() as db_q:
                    saved_lec = Lecture(
                        subject_id=subject_id,
                        title=derived_title,
                        raw_transcript="",
                        processed_notes=clean_content,
                        source_type="imported",
                        in_quaderno=True,
                    )
                    db_q.add(saved_lec)
                    db_q.commit()
                    db_q.refresh(saved_lec)
                    lec_id = saved_lec.id
                    lec_title = saved_lec.title

                await asyncio.to_thread(_index_quaderno_lecture, subject_id, lec_id, lec_title, clean_content)
                emitted_action = ChatStudyAction(
                    type="saved_to_quaderno",
                    lecture_id=lec_id,
                    title=lec_title,
                )
                yield _ndjson({"type": "action", "action": emitted_action.model_dump(mode="json")})

            followups = llm_engine.generate_followup_suggestions(
                subject_name=subject_name,
                user_message=user_msg_content,
                assistant_reply=clean_content,
                mode=resolved_mode,
            )

            persisted = await asyncio.to_thread(
                _persist_assistant_turn,
                session_id=session_id,
                replace_message_id=replace_message_id,
                content=clean_content,
                thinking=thinking,
                mode=resolved_mode,
                context_used=context_used,
                citations_json=json.dumps([c.model_dump(mode="json") for c in citations], ensure_ascii=False),
                action_json=json.dumps(emitted_action.model_dump(mode="json"), ensure_ascii=False)
                if emitted_action
                else "",
                followups_json=json.dumps(followups, ensure_ascii=False),
                token_estimate=llm_engine.estimate_tokens(clean_content),
                scope=scope_snapshot,
            )

            yield _ndjson({
                "type": "done",
                "assistant_message": persisted["assistant_message"],
                "followups": followups,
                "context_stats": persisted["context_stats"],
                "session_title": persisted["session_title"],
            })
        except asyncio.CancelledError:
            stream_stop_event.set()
            llm_engine.cancel_active_generations()
            llm_engine.clear_metal_cache()
            logger.info(f"[Chats] Streaming chat {session_id} annullato (CancelledError). Salvataggio stato interrotto.")
            try:
                partial_raw = "".join(raw_buffer).strip()
                _, clean_partial = _extract_think_and_content(partial_raw)
                if clean_partial:
                    final_msg = f"{clean_partial}\n\n*(Risposta interrotta)*"
                else:
                    final_msg = "Risposta interrotta."
                await asyncio.to_thread(
                    _persist_assistant_turn,
                    session_id=session_id,
                    replace_message_id=replace_message_id,
                    content=final_msg,
                    thinking="",
                    mode=effective_mode,
                    context_used=context_used if "context_used" in locals() else "",
                    citations_json=json.dumps([c.model_dump(mode="json") for c in citations], ensure_ascii=False) if "citations" in locals() else "[]",
                    action_json="",
                    followups_json="[]",
                    token_estimate=llm_engine.estimate_tokens(final_msg),
                    scope=scope_snapshot,
                )
            except Exception as persist_err:
                logger.warning(f"[Chats] Impossibile salvare turno interrotto per chat {session_id}: {persist_err}")
            raise
        except Exception as exc:
            stream_stop_event.set()
            logger.exception(f"[Chats] Errore durante lo streaming della chat {session_id}: {exc}")
            yield _ndjson({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{chat_id}/stream")
def stream_chat_message(
    chat_id: int,
    payload: ChatStreamRequest,
    db: Session = Depends(get_db),
):
    """Salva il messaggio utente, interroga il RAG con scope e streamma la risposta in NDJSON."""
    text = (payload.message or "").strip()
    effective_mode = (payload.mode or "auto").strip().lower() or "auto"
    if not text and effective_mode == "auto":
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")

    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    scope = payload.scope if payload.scope is not None else _parse_scope(session.scope_json)

    existing_messages = _load_session_messages(db, chat_id)
    has_prior_user = any(m.role == "user" for m in existing_messages)
    if (not session.title or session.title == "Nuova chat") and not has_prior_user:
        if text:
            session.title = llm_engine.generate_chat_title(text)
        elif effective_mode == "quiz":
            session.title = f"Quiz su {subject.name}"

    user_msg = ChatMessage(
        session_id=chat_id,
        role="user",
        content=text,
        thinking="",
        mode=effective_mode,
        context_used="",
        citations_json="[]",
        action_json="",
        followups_json="[]",
        token_estimate=llm_engine.estimate_tokens(text or "Quiz"),
    )
    db.add(user_msg)
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(session)

    all_messages = [*existing_messages, user_msg]
    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = _clean_history_for_llm(all_messages, compacted_up_to)
    if not history_messages:
        history_messages = [{"role": user_msg.role, "content": user_msg.content}]

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=None,
        enable_thinking=bool(payload.enable_thinking),
        rollback_notice=None,
    )


@router.post("/{chat_id}/messages/{message_id}/regenerate-stream")
def regenerate_chat_message_stream(
    chat_id: int,
    message_id: int,
    payload: Optional[ChatRegenerateRequest] = Body(default=None),
    db: Session = Depends(get_db),
):
    """Rigenera in-place la risposta assistente indicata in streaming NDJSON dal turno utente precedente."""
    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    target_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not target_msg:
        raise HTTPException(status_code=404, detail="Messaggio da rigenerare non trovato.")
    if target_msg.role != "assistant":
        raise HTTPException(status_code=400, detail="Puoi rigenerare solo un messaggio dell'assistente.")

    all_messages = _load_session_messages(db, chat_id)
    before_target = [m for m in all_messages if m.id < target_msg.id]
    user_msg = next((m for m in reversed(before_target) if m.role == "user"), None)
    if user_msg is None:
        raise HTTPException(status_code=400, detail="Nessun messaggio utente precedente trovato per rigenerare.")

    if target_msg.action_json:
        _execute_action_rollback(db, target_msg.action_json)

    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = _clean_history_for_llm(before_target, compacted_up_to)
    if not history_messages:
        history_messages = [{"role": user_msg.role, "content": user_msg.content}]

    scope = _parse_scope(session.scope_json)
    effective_mode = (user_msg.mode or target_msg.mode or "auto").strip().lower() or "auto"

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=target_msg.id,
        enable_thinking=bool(payload.enable_thinking) if payload else False,
    )


@router.post("/{chat_id}/messages/{message_id}/edit-stream")
def edit_chat_message_stream(
    chat_id: int,
    message_id: int,
    payload: ChatMessageSend,
    db: Session = Depends(get_db),
):
    """Modifica un messaggio utente, elimina i messaggi successivi e avvia lo streaming della nuova risposta."""
    text = (payload.message or "").strip()
    effective_mode = (payload.mode or "auto").strip().lower() or "auto"
    if not text and effective_mode == "auto":
        raise HTTPException(status_code=400, detail="Il messaggio non può essere vuoto.")

    session = _get_session_or_404(db, chat_id)
    subject = _get_subject_or_404(db, session.subject_id)

    user_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.id == message_id, ChatMessage.session_id == chat_id)
        .first()
    )
    if not user_msg:
        raise HTTPException(status_code=404, detail="Messaggio non trovato nella sessione.")
    if user_msg.role != "user":
        raise HTTPException(status_code=400, detail="Puoi modificare solo un messaggio utente.")

    if payload.scope is not None:
        session.scope_json = json.dumps(payload.scope.model_dump(), ensure_ascii=False)
    scope = payload.scope if payload.scope is not None else _parse_scope(session.scope_json)

    effective_mode = (payload.mode or user_msg.mode or "auto").strip().lower() or "auto"

    user_msg.content = text
    user_msg.mode = effective_mode
    user_msg.token_estimate = llm_engine.estimate_tokens(text)

    # 1. Annulla e ripristina le modifiche a quiz/materiali apportate dai messaggi successivi (se non disattivato dall'utente)
    should_rollback_quiz = getattr(payload, "rollback_quiz", True)
    if should_rollback_quiz:
        subsequent_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == chat_id, ChatMessage.id > message_id)
            .order_by(ChatMessage.id.desc())
            .all()
        )
        for s_msg in subsequent_messages:
            if s_msg.action_json:
                _execute_action_rollback(db, s_msg.action_json)

    # 2. Elimina tutti i messaggi successivi in quella sessione (ChatMessage.id > message_id)
    db.query(ChatMessage).filter(
        ChatMessage.session_id == chat_id,
        ChatMessage.id > message_id,
    ).delete(synchronize_session=False)

    compacted_val = session.compacted_up_to_message_id or 0
    if compacted_val >= message_id:
        prev_msg = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == chat_id, ChatMessage.id < message_id)
            .order_by(ChatMessage.id.desc())
            .first()
        )
        if prev_msg:
            session.compacted_up_to_message_id = prev_msg.id
        else:
            session.compacted_up_to_message_id = 0
            session.compacted_summary = None

    has_prior_user = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == chat_id,
            ChatMessage.role == "user",
            ChatMessage.id < message_id,
        )
        .first()
        is not None
    )
    if (not session.title or session.title == "Nuova chat") and not has_prior_user:
        session.title = llm_engine.generate_chat_title(text)

    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)
    db.refresh(session)

    all_messages = _load_session_messages(db, chat_id)
    compacted_up_to = session.compacted_up_to_message_id or 0
    history_messages = _clean_history_for_llm(all_messages, compacted_up_to)
    if not history_messages:
        history_messages = [{"role": user_msg.role, "content": user_msg.content}]

    return _stream_chat_turn(
        session_id=session.id,
        subject_id=subject.id,
        subject_name=subject.name,
        compacted_summary=session.compacted_summary or "",
        user_msg_dict=_serialize_message(user_msg).model_dump(mode="json"),
        user_msg_content=user_msg.content,
        history_messages=history_messages,
        stats_before_dict=_build_context_stats(session, all_messages, scope).model_dump(mode="json"),
        mode=effective_mode,
        scope_lecture_ids=list(scope.lecture_ids or []),
        scope_material_ids=list(scope.material_ids or []),
        replace_message_id=None,
        enable_thinking=bool(payload.enable_thinking),
        rollback_notice=None,
    )
