from __future__ import annotations

import json
import pytest

from database import QuizSet, SessionLocal, StudentConcept
from routers.chats import (
    _QUIZ_COMPLETED_INTENT_RE,
    _QUIZ_EXPAND_INTENT_RE,
    _QUIZ_DELETE_INTENT_RE,
    _QUIZ_REMOVE_QUESTION_INTENT_RE,
    _QUIZ_MODIFY_QUESTION_INTENT_RE,
    _QUADERNO_SAVE_INTENT_RE,
    _QUIZ_INTENT_RE,
    _QUIZ_REVIEW_INTENT_RE,
    _UNDERSTOOD_INTENT_RE,
    _parse_quiz_target_question_num,
    _format_quiz_context_for_llm,
    _sanitize_quiz_feedback_leaks,
)
from routers.study import (
    _extract_quiz_json_items,
    create_and_save_quiz_set_threadsafe,
    expand_and_save_quiz_set_threadsafe,
)
import llm_engine
import rag


def test_ai_generated_quiz_custom_titles():
    # 1. Output strutturato standard generato dall'IA con titolo custom (approccio pulito e snello)
    raw_ai_1 = json.dumps({
        "title": "Pattern Facade",
        "questions": [
            {"id": "q1", "question": "Cos'è il pattern Facade?", "options": ["A", "B", "C", "D"], "correct_index": 0}
        ]
    })
    title1, qs1 = _extract_quiz_json_items(raw_ai_1)
    assert title1 == "Pattern Facade"
    assert len(qs1) == 1

    # 2. Output dell'IA con blocco markdown ```json e titolo custom accademico
    raw_ai_2 = '```json\n{\n  "title": "Crittografia RSA",\n  "questions": [\n    {"id": "q1", "question": "Cos\'è RSA?", "options": ["A", "B", "C", "D"], "correct_index": 0}\n  ]\n}\n```'
    title2, qs2 = _extract_quiz_json_items(raw_ai_2)
    assert title2 == "Crittografia RSA"
    assert len(qs2) == 1

    # 3. Fallback retrocompatibile se l'IA o un mock restituisce un array senza campo title
    raw_legacy = json.dumps([
        {"id": "q1", "question": "Cos'è la memoria virtuale?", "options": ["A", "B", "C", "D"], "correct_index": 0}
    ])
    title3, qs3 = _extract_quiz_json_items(raw_legacy)
    assert title3 is None
    assert len(qs3) == 1


def test_parse_quiz_target_question_num():
    # Ordinali italiani e numeri
    assert _parse_quiz_target_question_num("la prima domanda non mi piace, la puoi cambiare e scrivere più semplice?") == 1
    assert _parse_quiz_target_question_num("no, puoi modificare proprio il quiz che mi hai creato e modificare la prima domanda? non mi piace dovresti farla più semplice, solo la prima") == 1
    assert _parse_quiz_target_question_num("elimina la seconda domanda") == 2
    assert _parse_quiz_target_question_num("modifica la terza domanda") == 3
    assert _parse_quiz_target_question_num("riscrivi la 4ª domanda") == 4
    assert _parse_quiz_target_question_num("cancella la 5a domanda") == 5
    assert _parse_quiz_target_question_num("domanda 1 falla più facile") == 1
    assert _parse_quiz_target_question_num("elimina l'ultima domanda") == -1
    assert _parse_quiz_target_question_num("togli la penultima domanda") == -2
    assert _parse_quiz_target_question_num("fammi un quiz sul Facade") is None


def test_intent_detection_regexes():
    # Quiz creation
    assert bool(_QUIZ_INTENT_RE.search("mi fai capire con un quiz?"))
    assert bool(_QUIZ_INTENT_RE.search("fammi un quiz sul Facade"))
    assert bool(_QUIZ_INTENT_RE.search("crea un test di verifica"))
    assert bool(_QUIZ_INTENT_RE.search("mettimi alla prova"))
    assert bool(_QUIZ_INTENT_RE.search("facciamo un quiz di ripasso"))
    assert bool(_QUIZ_INTENT_RE.search("mi crei un quiz sui pattern?"))
    assert bool(_QUIZ_INTENT_RE.search("mi faresti un quiz?"))
    assert bool(_QUIZ_INTENT_RE.search("fai un quiz di 10 domande"))
    assert bool(_QUIZ_INTENT_RE.search("vorrei un test a crocette"))

    # Quiz expansion
    assert bool(_QUIZ_EXPAND_INTENT_RE.search("aggiungi altre 5 domande a questo quiz"))
    assert bool(_QUIZ_EXPAND_INTENT_RE.search("mi aggiungi 5 domande al quiz?"))
    assert bool(_QUIZ_EXPAND_INTENT_RE.search("espandi il quiz con altre 3 domande"))
    assert bool(_QUIZ_EXPAND_INTENT_RE.search("altre 5 domande"))

    # Quiz review
    assert bool(_QUIZ_REVIEW_INTENT_RE.search("ho completato il quiz, aiutami a ripassare dove ho sbagliato"))
    assert bool(_QUIZ_REVIEW_INTENT_RE.search("rivediamo gli errori del quiz"))
    assert bool(_QUIZ_REVIEW_INTENT_RE.search("perché ho sbagliato la domanda 2?"))

    # Quiz completed
    assert bool(_QUIZ_COMPLETED_INTENT_RE.search("l'ho fatto"))
    assert bool(_QUIZ_COMPLETED_INTENT_RE.search("ho finito"))
    assert bool(_QUIZ_COMPLETED_INTENT_RE.search("fatto"))
    assert bool(_QUIZ_COMPLETED_INTENT_RE.search("come sono andato?"))

    # Understood (Feynman)
    assert bool(_UNDERSTOOD_INTENT_RE.search("ho capito!"))
    assert bool(_UNDERSTOOD_INTENT_RE.search("tutto chiaro grazie"))
    assert bool(_UNDERSTOOD_INTENT_RE.search("chiarissimo"))

    # Quiz deletion
    assert bool(_QUIZ_DELETE_INTENT_RE.search("elimina questo quiz"))
    assert bool(_QUIZ_DELETE_INTENT_RE.search("cancella il quiz appena creato"))
    assert bool(_QUIZ_DELETE_INTENT_RE.search("rimuovi il test"))
    assert bool(_QUIZ_DELETE_INTENT_RE.search("cancella l'ultimo quiz"))

    # Quiz remove question
    assert bool(_QUIZ_REMOVE_QUESTION_INTENT_RE.search("elimina la domanda 2"))
    assert bool(_QUIZ_REMOVE_QUESTION_INTENT_RE.search("cancella la 3ª domanda"))
    assert bool(_QUIZ_REMOVE_QUESTION_INTENT_RE.search("togli la domanda 4"))
    assert bool(_QUIZ_REMOVE_QUESTION_INTENT_RE.search("rimuovi domanda 1"))

    # Quiz modify question
    assert bool(_QUIZ_MODIFY_QUESTION_INTENT_RE.search("modifica la domanda 2"))
    assert bool(_QUIZ_MODIFY_QUESTION_INTENT_RE.search("cambia la domanda 3"))
    assert bool(_QUIZ_MODIFY_QUESTION_INTENT_RE.search("riscrivi la domanda 1"))

    # Quaderno save
    assert bool(_QUADERNO_SAVE_INTENT_RE.search("spiegami questo concetto e poi mettila nel quaderno"))
    assert bool(_QUADERNO_SAVE_INTENT_RE.search("salva nel quaderno"))
    assert bool(_QUADERNO_SAVE_INTENT_RE.search("salvalo negli appunti"))
    assert bool(_QUADERNO_SAVE_INTENT_RE.search("mettilo nel quaderno per favore"))


def test_expand_quiz_set_and_submit_results(client, subject, monkeypatch):
    subject_id = subject["id"]

    # Mock generazione LLM con formato moderno (titolo deciso direttamente dall'IA)
    def fake_generate(*args, **kwargs):
        return json.dumps({
            "title": "Facade",
            "questions": [
                {
                    "id": "q1",
                    "question": "Qual è lo scopo del Facade?",
                    "options": ["Semplificare interfaccia", "Creare copie", "Gestire thread", "Tradurre chiamate"],
                    "correct_index": 0,
                    "explanation": "Fornisce una facciata unificata.",
                    "source_lecture": "Design Pattern",
                }
            ]
        })

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate)
    monkeypatch.setattr("routers.study.is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.clear_metal_cache", lambda *a, **kw: None)

    # 1. Creazione quiz
    created_info = create_and_save_quiz_set_threadsafe(
        subject_id=subject_id,
        chat_session_id=None,
        num_questions=1,
        context_text="Il pattern Facade semplifica l'interfaccia.",
        topic_hint="mi fai un quiz specifico sul Facade",
        subject_name=subject["name"],
    )
    quiz_id = created_info["id"]
    assert created_info["title"] == "Quiz: Facade"
    assert created_info["num_questions"] == 1

    # 2. Espansione quiz
    def fake_generate_extra(*args, **kwargs):
        return json.dumps([
            {
                "id": "q2",
                "question": "Come si relaziona Facade con Adapter?",
                "options": ["Adatta interfaccia", "Uguale", "Nessuna relazione", "Opposto"],
                "correct_index": 0,
                "explanation": "L'Adapter converte un'interfaccia esistente.",
                "source_lecture": "Design Pattern",
            }
        ])

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate_extra)

    expanded_info = expand_and_save_quiz_set_threadsafe(
        subject_id=subject_id,
        quiz_id=quiz_id,
        num_new_questions=1,
        context_text="Il pattern Adapter adatta interfacce.",
        topic_hint="aggiungi altre domande",
        subject_name=subject["name"],
    )
    assert expanded_info["num_questions"] == 2
    assert expanded_info["new_added"] == 1

    # 3. Sottomissione risultati quiz via API
    results_payload = {
        "score_percent": 50,
        "results": [
            {
                "question": "Qual è lo scopo del Facade?",
                "options": ["Semplificare interfaccia", "Creare copie", "Gestire thread", "Tradurre chiamate"],
                "selected_index": 0,
                "correct_index": 0,
                "is_correct": True,
                "explanation": "Fornisce una facciata unificata.",
            },
            {
                "question": "Come si relaziona Facade con Adapter?",
                "options": ["Adatta interfaccia", "Uguale", "Nessuna relazione", "Opposto"],
                "selected_index": 1,
                "correct_index": 0,
                "is_correct": False,
                "explanation": "L'Adapter converte un'interfaccia esistente.",
            },
        ],
    }

    resp = client.patch(f"/api/v1/study/quiz-sets/{quiz_id}/results", json=results_payload)
    assert resp.status_code == 200, resp.text
    saved = resp.json()
    assert saved["completed"] is True
    assert saved["score_percent"] == 50
    assert len(saved["questions"]) == 2

    # 4. Formattazione contesto quiz per il tutor assistente
    with SessionLocal() as db:
        quiz_record = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        ctx_prompt = _format_quiz_context_for_llm(quiz_record)
        assert "50%" in ctx_prompt
        assert "CONCETTI SBAGLIATI DALLO STUDENTE" in ctx_prompt
        assert "Come si relaziona Facade con Adapter?" in ctx_prompt
        assert "Risposta scelta dallo studente (ERRATA): \"Uguale\"" in ctx_prompt
        assert "Risposta CORRETTA: \"Adatta interfaccia\"" in ctx_prompt
        assert "ISTRUZIONE TUTOR ACCADEMICO" not in ctx_prompt

    # 5. Verifica Mastery Summary via API
    resp_mastery = client.get(f"/api/v1/study/{subject_id}/mastery")
    assert resp_mastery.status_code == 200
    mastery_data = resp_mastery.json()
    assert mastery_data["total_tracked"] >= 2
    assert mastery_data["consolidated_count"] >= 1
    assert mastery_data["to_consolidate_count"] >= 1

    # 6. Verifica marcatura manuale concetto
    weak_candidates = [c for c in mastery_data["concepts"] if c["status"] == "da_consolidare"]
    assert len(weak_candidates) >= 1
    weak_topic = weak_candidates[0]["topic"]
    resp_mark = client.post(
        f"/api/v1/study/{subject_id}/concepts/mark",
        json={"topic": weak_topic, "status": "consolidato"},
    )
    assert resp_mark.status_code == 200
    assert resp_mark.json()["status"] == "consolidato"


def test_conversational_understood_feynman_promotion(client, subject, monkeypatch):
    subject_id = subject["id"]

    # Inserisci concetto da consolidare nel DB
    with SessionLocal() as db:
        c = StudentConcept(
            subject_id=subject_id,
            topic="Pattern Decorator",
            status="da_consolidare",
            error_count=2,
            correct_count=0,
            last_context="Test",
        )
        db.add(c)
        db.commit()

    # Mock chat engine
    def fake_stream(*a, **kw):
        yield "Ottimo, concetto chiarito!"

    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "stream_multi_turn_chat", fake_stream)
    monkeypatch.setattr(rag, "query_context_structured", lambda *a, **kw: {"context_text": "", "citations": []})
    monkeypatch.setattr(rag, "unload_embedding_model", lambda *a, **kw: None)

    # Crea sessione chat
    resp_chat = client.post(f"/api/v1/chats/subject/{subject_id}", json={"title": "Sessione Test"})
    assert resp_chat.status_code == 200
    chat_id = resp_chat.json()["id"]

    # Invio messaggio "ho capito!" via stream
    resp_msg = client.post(
        f"/api/v1/chats/{chat_id}/stream",
        json={"message": "Grazie professore, adesso ho capito chiarissimo!"},
    )
    assert resp_msg.status_code == 200

    # Verifica che il concetto sia stato promosso a "consolidato"
    with SessionLocal() as db:
        promoted = (
            db.query(StudentConcept)
            .filter(StudentConcept.subject_id == subject_id, StudentConcept.topic == "Pattern Decorator")
            .first()
        )
        assert promoted is not None
        assert promoted.status == "consolidato"
        assert promoted.correct_count >= 1


def test_sanitize_quiz_feedback_leaks():
    raw_leaky_response = (
        "# 📄 Quiz: ciao (ID 5) - Verifica Finale\n\n"
        "### Istruzione Tutor Accademico\n"
        "Lo studente ha risposto male a tutto.\n\n"
        "Nota: Ho notato che la risposta corretta per tutte le domande è la b.\n"
        "Ecco la spiegazione vera: la Facade serve per semplificare l'accesso."
    )
    cleaned = _sanitize_quiz_feedback_leaks(raw_leaky_response)
    assert "Verifica Finale" not in cleaned
    assert "Istruzione Tutor Accademico" not in cleaned
    assert "la risposta corretta per tutte le domande è la b" not in cleaned
    assert "Ecco la spiegazione vera: la Facade serve per semplificare l'accesso." in cleaned


def test_delete_quiz_set_and_chat_session_decoupling(client, subject, monkeypatch):
    subject_id = subject["id"]

    # 1. Crea chat e crea un quiz associato alla chat
    resp_chat = client.post(f"/api/v1/chats/subject/{subject_id}", json={"title": "Chat Con Quiz"})
    assert resp_chat.status_code == 200
    chat_id = resp_chat.json()["id"]

    def fake_generate(*args, **kwargs):
        return json.dumps([
            {
                "id": "q1",
                "question": "Cos'è Singleton?",
                "options": ["Unica istanza", "Molte istanze", "Un file", "Un server"],
                "correct_index": 0,
                "explanation": "Garantisce una sola istanza della classe.",
                "source_lecture": "Design Pattern",
            }
        ])

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate)
    monkeypatch.setattr("routers.study.is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.clear_metal_cache", lambda *a, **kw: None)

    q_info = create_and_save_quiz_set_threadsafe(
        subject_id=subject_id,
        chat_session_id=chat_id,
        num_questions=1,
        context_text="Pattern Singleton.",
        topic_hint="quiz su Singleton",
        subject_name=subject["name"],
    )
    quiz_id = q_info["id"]

    # 2. Elimina la chat: il quiz DEVE sopravvivere nella materia (chat_session_id diventa None)
    del_chat_resp = client.delete(f"/api/v1/chats/{chat_id}")
    assert del_chat_resp.status_code == 200

    with SessionLocal() as db:
        survived_quiz = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        assert survived_quiz is not None
        assert survived_quiz.chat_session_id is None
        assert survived_quiz.subject_id == subject_id

    # 3. Elimina il quiz tramite endpoint DELETE /study/quiz-sets/{quiz_id}
    del_quiz_resp = client.delete(f"/api/v1/study/quiz-sets/{quiz_id}")
    assert del_quiz_resp.status_code == 200
    assert del_quiz_resp.json()["ok"] is True

    with SessionLocal() as db:
        deleted_quiz = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        assert deleted_quiz is None


def test_concept_deduplication_and_deletion(client, subject):
    subject_id = subject["id"]

    # 1. Simula l'inserimento di concetti duplicati storici
    with SessionLocal() as db:
        db.query(StudentConcept).filter(StudentConcept.subject_id == subject_id).delete()
        c1 = StudentConcept(subject_id=subject_id, topic="al Riguardo", status="da_consolidare", error_count=2, correct_count=0)
        c2 = StudentConcept(subject_id=subject_id, topic="Al Riguardo", status="consolidato", error_count=0, correct_count=1)
        c3 = StudentConcept(subject_id=subject_id, topic="Pattern Facade", status="da_consolidare", error_count=1, correct_count=0)
        db.add_all([c1, c2, c3])
        db.commit()

    # 2. Richiama GET mastery: deve scattare la deduplicazione automatica
    resp_mastery = client.get(f"/api/v1/study/{subject_id}/mastery")
    assert resp_mastery.status_code == 200
    data = resp_mastery.json()
    assert data["total_tracked"] == 2  # "Al Riguardo" unificato + "Pattern Facade"
    riguardo = next(c for c in data["concepts"] if c["topic"].lower() == "al riguardo")
    assert riguardo["error_count"] == 2
    assert riguardo["correct_count"] == 1
    riguardo_id = riguardo["id"]

    # 3. Elimina il singolo concetto "Al Riguardo"
    del_resp = client.delete(f"/api/v1/study/{subject_id}/concepts/{riguardo_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True

    with SessionLocal() as db:
        assert db.query(StudentConcept).filter(StudentConcept.id == riguardo_id).first() is None
        remaining = db.query(StudentConcept).filter(StudentConcept.subject_id == subject_id).all()
        assert len(remaining) == 1
        assert remaining[0].topic == "Pattern Facade"

    # 4. Azzera tutti i concetti della materia
    clear_resp = client.delete(f"/api/v1/study/{subject_id}/concepts")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["ok"] is True
    assert clear_resp.json()["deleted_count"] == 1

    with SessionLocal() as db:
        assert db.query(StudentConcept).filter(StudentConcept.subject_id == subject_id).count() == 0


def test_quiz_action_rollback_on_message_edit(client, subject, monkeypatch):
    subject_id = subject["id"]

    # Mock chat stream
    def fake_stream(*a, **kw):
        yield "Certo, ecco la spiegazione del nuovo argomento!"

    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "stream_multi_turn_chat", fake_stream)
    monkeypatch.setattr(rag, "query_context_structured", lambda *a, **kw: {"context_text": "", "citations": []})
    monkeypatch.setattr(rag, "unload_embedding_model", lambda *a, **kw: None)

    # 1. Crea una sessione di chat
    chat_resp = client.post(f"/api/v1/chats/subject/{subject_id}", json={"title": "Test Chat Rollback"})
    assert chat_resp.status_code == 200
    chat_id = chat_resp.json()["id"]

    # 2. Crea un QuizSet con 2 domande
    with SessionLocal() as db:
        q_original = [
            {"id": "q1", "question": "Domanda 1 ORIGINALE", "options": ["A", "B"], "correct_index": 0},
            {"id": "q2", "question": "Domanda 2", "options": ["C", "D"], "correct_index": 1},
        ]
        quiz = QuizSet(
            subject_id=subject_id,
            chat_session_id=chat_id,
            title="Quiz Rollback Test",
            source_label="Test",
            questions_json=json.dumps(q_original, ensure_ascii=False),
        )
        db.add(quiz)
        db.commit()
        db.refresh(quiz)
        quiz_id = quiz.id

    # 3. Simula la modifica della domanda 1 in un turno di chat
    with SessionLocal() as db:
        from database import ChatMessage
        u_msg = ChatMessage(
            session_id=chat_id,
            role="user",
            content="la prima domanda è troppo difficile, la puoi modificare?",
        )
        db.add(u_msg)
        db.commit()
        db.refresh(u_msg)
        u_msg_id = u_msg.id

        # Simula il quiz aggiornato nel DB
        q_mod = [
            {"id": "q1", "question": "Domanda 1 MODIFICATA", "options": ["A", "B"], "correct_index": 0},
            {"id": "q2", "question": "Domanda 2", "options": ["C", "D"], "correct_index": 1},
        ]
        quiz_db = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        quiz_db.questions_json = json.dumps(q_mod, ensure_ascii=False)

        # Salva lo snapshot di rollback nel messaggio dell'assistente
        rollback_data = json.dumps({
            "rollback_type": "restore_question",
            "question_index": 0,
            "old_question": {"id": "q1", "question": "Domanda 1 ORIGINALE", "options": ["A", "B"], "correct_index": 0},
        })
        act = {
            "type": "quiz_created",
            "quiz_id": quiz_id,
            "title": "Quiz Rollback Test",
            "num_questions": 2,
            "question": rollback_data,
        }
        a_msg = ChatMessage(
            session_id=chat_id,
            role="assistant",
            content="Ho modificato la domanda 1 in Quiz Rollback Test",
            action_json=json.dumps(act),
        )
        db.add(a_msg)
        db.commit()

    # 4. Verifica che prima del rollback la domanda nel DB sia "Domanda 1 MODIFICATA"
    with SessionLocal() as db:
        q_check = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        qs = json.loads(q_check.questions_json)
        assert qs[0]["question"] == "Domanda 1 MODIFICATA"

    # 5. Ora modifica il messaggio utente u_msg_id
    edit_resp = client.post(
        f"/api/v1/chats/{chat_id}/messages/{u_msg_id}/edit-stream",
        json={"message": "spiegami un altro argomento"},
    )
    assert edit_resp.status_code == 200

    # 6. Verifica che nel DB la domanda 1 sia stata AUTOMATICAMENTE RIPRISTINATA a "Domanda 1 ORIGINALE"!
    with SessionLocal() as db:
        q_restored = db.query(QuizSet).filter(QuizSet.id == quiz_id).first()
        qs_restored = json.loads(q_restored.questions_json)
        assert qs_restored[0]["question"] == "Domanda 1 ORIGINALE"

    # 7. Verifica che il banner intrusivo non sia presente nello stream (è stato rimosso a favore del modal)
    raw_stream = edit_resp.text
    assert "Ripristinata la domanda 1" not in raw_stream


def test_rewrite_single_quiz_question_explicit_and_flexible(monkeypatch):
    from routers.chats import _extract_explicit_new_question_text, _rewrite_single_quiz_question

    # 1. Verifica estrazione testo esplicito
    assert _extract_explicit_new_question_text("cambia la prima domanda in: ti piace samuel?") == "ti piace samuel?"
    assert _extract_explicit_new_question_text("sostituisci la seconda domanda con: cos'è un thread?") == "cos'è un thread?"
    assert _extract_explicit_new_question_text('metti la prima domanda in "Chi ha creato Python?"') == "Chi ha creato Python?"
    assert _extract_explicit_new_question_text("la prima domanda è troppo difficile, falla più semplice") is None

    # 2. Mock generate_text che restituisce opzioni plausibili
    def fake_generate(sys_p, user_p, *a, **kw):
        return json.dumps({
            "question": "ti piace samuel?",
            "concept": "Domanda test",
            "options": ["Sì", "No", "Molto", "Per niente"],
            "correct_index": 0,
            "explanation": "Spiegazione",
        })

    monkeypatch.setattr(llm_engine, "generate_text", fake_generate)

    old_q = {
        "id": "q-1",
        "question": "Qual è l'obiettivo principale di TLS Inspection?",
        "concept": "Sicurezza",
        "options": ["A", "B", "C", "D"],
        "correct_index": 1,
    }

    res = _rewrite_single_quiz_question(
        old_item=old_q,
        user_instruction="cambia la prima domanda in: ti piace samuel?",
        subject_name="Sicurezza delle Reti",
        context_text="TLS Inspection",
        llm_model="test-model",
        display_num=1,
    )
    assert res is not None
    assert res["question"] == "ti piace samuel?"
    assert len(res["options"]) == 4
    assert res["correct_index"] == 0


def test_ordinal_not_matching_lecture():
    """Verifica che 'dell'ultima lezione' o 'la prima lezione' non vengano scambiati per numeri di domanda quiz."""
    assert _parse_quiz_target_question_num("mi fai un quiz dell'ultima lezione?") is None
    assert _parse_quiz_target_question_num("parlami dell'ultima lezione") is None
    assert _parse_quiz_target_question_num("la prima lezione mi è piaciuta") is None
    assert _parse_quiz_target_question_num("la prima domanda è troppo difficile") == 1
    assert _parse_quiz_target_question_num("modifica l'ultima domanda") == -1
    assert _parse_quiz_target_question_num("cambia la seconda") == 2


