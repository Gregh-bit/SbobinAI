from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pymupdf
import pytest

import llm_engine
import rag
import routers.study as study_router
import routers.subjects as subjects_router


def _parse_ndjson(response_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in response_text.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _make_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_pptx_bytes(slides_text: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>',
        )
        for idx, text in enumerate(slides_text, start=1):
            slide_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                "<p:cSld><p:spTree><p:sp><p:txBody>"
                f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>"
                "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
            )
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml)
    return buf.getvalue()


@pytest.fixture()
def mock_chat_ai(monkeypatch):
    captured_calls: list[dict[str, Any]] = []

    def fake_query_structured(
        subject_id: int,
        query: str,
        top_k: int = 5,
        priority_previous_lecture: bool = False,
        scope_lecture_ids: list[int] | None = None,
        scope_material_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "context_text": "[FONTE: Slide1.pdf | REF: Slide1.pdf#2 | ETICHETTA: Pag. 2]\nTeorema di Stokes.",
            "citations": [
                {
                    "source_id": "Slide1.pdf",
                    "source_type": "slide",
                    "title": "Slide1.pdf",
                    "page": 2,
                    "label": "Pag. 2",
                }
            ],
        }

    turn_counter = {"n": 0}

    def fake_stream_multi_turn(
        subject_name: str,
        messages: list[dict[str, str]],
        context_text: str = "",
        compacted_summary: str = "",
        mode: str = "auto",
        model_id: str | None = None,
        stop_event=None,
    ):
        turn_counter["n"] += 1
        idx = turn_counter["n"]
        captured_calls.append(
            {
                "subject_name": subject_name,
                "messages": list(messages),
                "context_text": context_text,
                "compacted_summary": compacted_summary,
                "mode": mode,
            }
        )
        yield f"<think>Analizzo la domanda {idx}</think>"
        yield f"Risposta turno {idx} con [[ref:Slide1.pdf#2|Pag. 2]] e formula $x^{idx}$."

    monkeypatch.setattr(rag, "query_context_structured", fake_query_structured)
    monkeypatch.setattr(rag, "unload_embedding_model", lambda: None)
    monkeypatch.setattr(rag, "index_lecture", lambda **kw: 1)
    monkeypatch.setattr(llm_engine, "stream_multi_turn_chat", fake_stream_multi_turn)
    monkeypatch.setattr(llm_engine, "unload_llm", lambda: None)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda: None)
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.is_model_downloaded", lambda *_: True)
    monkeypatch.setattr("routers.study.clear_metal_cache", lambda: None)
    return captured_calls


def test_multi_chat_crud_and_legacy_import(client, subject):
    r1 = client.post(f"/api/v1/chats/subject/{subject['id']}", json={"title": "Sessione A"})
    assert r1.status_code == 200, r1.text
    c1 = r1.json()
    assert c1["title"] == "Sessione A"

    r_imp = client.post(
        f"/api/v1/chats/subject/{subject['id']}/import-legacy",
        json={
            "title": "Chat precedente",
            "messages": [
                {"role": "user", "content": "Cos'è un integrale doppio?"},
                {"role": "assistant", "content": "Misura il volume sotto una superficie."},
            ],
        },
    )
    assert r_imp.status_code == 200, r_imp.text
    imported = r_imp.json()
    assert len(imported["messages"]) == 2
    assert imported["context_stats"]["active_messages_count"] == 2

    listing = client.get(f"/api/v1/chats/subject/{subject['id']}").json()
    assert len(listing) == 2

    upd = client.patch(
        f"/api/v1/chats/{c1['id']}",
        json={"title": "Analisi Ripasso", "scope": {"lecture_ids": [10], "material_ids": ["Slide1.pdf"]}},
    )
    assert upd.status_code == 200
    assert upd.json()["title"] == "Analisi Ripasso"
    assert upd.json()["scope"]["material_ids"] == ["Slide1.pdf"]

    deleted = client.delete(f"/api/v1/chats/{c1['id']}")
    assert deleted.status_code == 200
    assert len(client.get(f"/api/v1/chats/subject/{subject['id']}").json()) == 1


def test_stream_delete_message_from_memory_and_regenerate_inplace(client, subject, mock_chat_ai, monkeypatch):
    chat = client.post(f"/api/v1/chats/subject/{subject['id']}", json={"title": "Nuova chat"}).json()
    chat_id = chat["id"]

    # Turn 1
    r1 = client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Spiega il teorema di Stokes", "mode": "auto"})
    assert r1.status_code == 200, r1.text
    evs1 = _parse_ndjson(r1.text)
    done1 = [e for e in evs1 if e["type"] == "done"][0]
    a1 = done1["assistant_message"]
    assert a1["thinking"] == "Analizzo la domanda 1"
    assert "[[ref:Slide1.pdf#2|Pag. 2]]" in a1["content"]
    assert len(a1["citations"]) == 1

    # Turn 2 sees Turn 1 in LLM memory
    r2 = client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Fai un esempio pratico"})
    assert r2.status_code == 200
    evs2 = _parse_ndjson(r2.text)
    done2 = [e for e in evs2 if e["type"] == "done"][0]
    a2_id = done2["assistant_message"]["id"]
    assert len(mock_chat_ai[1]["messages"]) == 3  # u1, a1, u2
    assert any("Risposta turno 1" in m["content"] for m in mock_chat_ai[1]["messages"])
    tokens_before_delete = done2["context_stats"]["used_tokens"]

    # Failed regenerate preserves existing assistant message intact in DB
    detail_before_fail = client.get(f"/api/v1/chats/{chat_id}").json()
    last_msg_before = detail_before_fail["messages"][-1]

    def failing_stream(*a, **kw):
        raise RuntimeError("GPU OOM simulated")
        yield ""  # pragma: no cover

    monkeypatch.setattr(llm_engine, "stream_multi_turn_chat", failing_stream)
    r_fail = client.post(f"/api/v1/chats/{chat_id}/messages/{last_msg_before['id']}/regenerate-stream")
    fail_evs = _parse_ndjson(r_fail.text)
    assert any(e["type"] == "error" for e in fail_evs)
    detail_after_fail = client.get(f"/api/v1/chats/{chat_id}").json()
    assert detail_after_fail["messages"][-1]["id"] == last_msg_before["id"]
    assert detail_after_fail["messages"][-1]["content"] == last_msg_before["content"]

    # Successful regenerate replaces message IN-PLACE (same ID!)
    def regen_stream(subject_name, messages, **kw):
        mock_chat_ai.append({"subject_name": subject_name, "messages": list(messages)})
        yield "Risposta rigenerata aggiornata."

    monkeypatch.setattr(llm_engine, "stream_multi_turn_chat", regen_stream)
    r_regen = client.post(f"/api/v1/chats/{chat_id}/messages/{a2_id}/regenerate-stream")
    regen_evs = _parse_ndjson(r_regen.text)
    regen_done = [e for e in regen_evs if e["type"] == "done"][0]
    assert regen_done["assistant_message"]["id"] == a2_id
    assert regen_done["assistant_message"]["content"] == "Risposta rigenerata aggiornata."

    # Delete Turn 1 assistant message from UI + memory (CASCADES to a1, u2, a2)
    del_resp = client.delete(f"/api/v1/chats/{chat_id}/messages/{a1['id']}")
    assert del_resp.status_code == 200
    detail_after_del = del_resp.json()
    assert detail_after_del["context_stats"]["used_tokens"] < tokens_before_delete
    assert all(m["id"] not in (a1["id"], a2_id) for m in detail_after_del["messages"])
    assert len(detail_after_del["messages"]) == 1  # Only u1 remains

    # Turn 3 MUST NOT see deleted Turn 1 assistant message nor subsequent messages in LLM memory
    r3 = client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Ricapitola"})
    assert r3.status_code == 200
    msgs_sent_t3 = mock_chat_ai[-1]["messages"]
    assert not any(m["content"] == a1["content"] for m in msgs_sent_t3)
    assert not any(m["content"] == "Risposta rigenerata aggiornata." for m in msgs_sent_t3)
    assert len(msgs_sent_t3) == 2
    assert msgs_sent_t3[0]["content"] == "Spiega il teorema di Stokes"
    assert msgs_sent_t3[1]["content"] == "Ricapitola"


def test_compaction_archive_and_save_note(client, subject, mock_chat_ai, monkeypatch):
    monkeypatch.setattr(
        llm_engine,
        "compact_chat_history",
        lambda *a, **kw: "Riassunto compatto: Stokes e forme differenziali.",
    )
    imported = client.post(
        f"/api/v1/chats/subject/{subject['id']}/import-legacy",
        json={
            "title": "Lunga sessione",
            "messages": [
                {"role": "user", "content": f"Domanda {i} " + ("x" * 180)}
                if i % 2 == 0
                else {"role": "assistant", "content": f"Risposta {i} " + ("y" * 240)}
                for i in range(6)
            ],
        },
    ).json()
    chat_id = imported["id"]
    before_tokens = imported["context_stats"]["used_tokens"]

    compacted = client.post(f"/api/v1/chats/{chat_id}/compact").json()
    assert compacted["compacted_up_to_message_id"] > 0
    assert compacted["compacted_summary"] == "Riassunto compatto: Stokes e forme differenziali."
    assert compacted["context_stats"]["compacted_messages_count"] == 6
    assert compacted["context_stats"]["active_messages_count"] == 0
    assert compacted["context_stats"]["used_tokens"] < before_tokens

    # Next stream receives compacted_summary
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Continua da qui"})
    assert mock_chat_ai[-1]["compacted_summary"] == "Riassunto compatto: Stokes e forme differenziali."

    # Save single assistant note into Quaderno
    target_aid = next(m["id"] for m in compacted["messages"] if m["role"] == "assistant")
    note_res = client.post(f"/api/v1/chats/{chat_id}/save-note", json={"message_id": target_aid, "title": "Nota Stokes"})
    assert note_res.status_code == 200
    assert note_res.json()["lecture_id"] > 0

    # Archive whole chat to Quaderno & clear
    arch = client.post(
        f"/api/v1/chats/{chat_id}/archive",
        json={"title": "Appunti Chat Analisi", "mode": "quaderno_and_clear"},
    )
    assert arch.status_code == 200, arch.text
    arch_data = arch.json()
    assert arch_data["lecture_id"] > 0
    assert arch_data["session"]["messages"] == []
    assert arch_data["session"]["compacted_summary"] == ""

    archived_lecture = client.get(f"/api/v1/lectures/{arch_data['lecture_id']}")
    assert archived_lecture.status_code == 200
    lec_data = archived_lecture.json()
    assert lec_data["title"] == "Appunti Chat Analisi" and lec_data["in_quaderno"] is True


def test_chat_quiz_action_and_oral_action_persistence(client, subject, mock_chat_ai, monkeypatch):
    monkeypatch.setattr(
        llm_engine,
        "generate_text",
        lambda *a, **kw: json.dumps(
            [
                {
                    "question": "Qual è la definizione del rotore?",
                    "options": ["Opzione A", "Opzione B", "Opzione C", "Opzione D"],
                    "correct_index": 1,
                    "explanation": "Deriva dal prodotto vettoriale nabla.",
                }
            ]
        ),
    )

    chat = client.post(f"/api/v1/chats/subject/{subject['id']}", json={}).json()
    r_quiz = client.post(
        f"/api/v1/chats/{chat['id']}/stream",
        json={"message": "Crea un quiz sul teorema di Stokes", "mode": "quiz"},
    )
    assert r_quiz.status_code == 200, r_quiz.text
    evs = _parse_ndjson(r_quiz.text)
    action_evs = [e for e in evs if e["type"] == "action"]
    assert len(action_evs) == 1
    action = action_evs[0]["action"]
    assert action["type"] == "quiz_created"
    quiz_id = action["quiz_id"]

    # Retrieve persisted QuizSet via study endpoints
    q_list = client.get(f"/api/v1/study/{subject['id']}/quiz-sets").json()
    assert any(q["id"] == quiz_id for q in q_list)
    q_detail = client.get(f"/api/v1/study/quiz-sets/{quiz_id}").json()
    assert q_detail["questions"][0]["correct_index"] == 1

    # Oral action trigger
    r_oral = client.post(
        f"/api/v1/chats/{chat['id']}/stream",
        json={"message": "Fammi una domanda orale", "mode": "oral"},
    )
    oral_evs = _parse_ndjson(r_oral.text)
    assert any(e["type"] == "action" and e["action"]["type"] == "oral_question" for e in oral_evs)


def test_pdf_and_pptx_page_extraction_rendering_and_rag_citations(client, subject, monkeypatch, tmp_root):
    # Monkeypatch heavy embedding calls so real LanceDB schema/page indexing executes fast in tests
    monkeypatch.setattr(rag, "_embed_or_explain", lambda texts, *a, **kw: [[0.01] * rag.EMBEDDING_DIM for _ in texts])
    monkeypatch.setattr(rag, "get_cached_query_vector", lambda q, *a, **kw: [0.01] * rag.EMBEDDING_DIM)
    monkeypatch.setattr(rag, "unload_embedding_model", lambda: None)

    pdf_bytes = _make_pdf_bytes(["Pagina uno introduzione limiti", "Pagina due Teorema di Stokes completo"])
    pptx_bytes = _make_pptx_bytes(["Slide 1 Titolo Corso", "Slide 2 Dimostrazione Lagrange"])

    up_pdf = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": ("Appunti.pdf", pdf_bytes, "application/pdf")},
    )
    assert up_pdf.status_code in (200, 201), up_pdf.text

    up_pptx = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": ("Lezione.pptx", pptx_bytes, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
    )
    assert up_pptx.status_code in (200, 201), up_pptx.text

    # Visual PDF page 2 PNG endpoint
    pdf_page = client.get(f"/api/v1/subjects/{subject['id']}/materials/Appunti.pdf/page/2")
    assert pdf_page.status_code == 200, pdf_page.text
    p_data = pdf_page.json()
    assert p_data["kind"] == "image"
    assert p_data["page"] == 2
    assert p_data["page_count"] == 2
    assert p_data["image_data_url"].startswith("data:image/png;base64,")
    assert "Stokes" in p_data["text"]

    # PPTX slide 2 endpoint
    pptx_page = client.get(f"/api/v1/subjects/{subject['id']}/materials/Lezione.pptx/page/2")
    assert pptx_page.status_code == 200, pptx_page.text
    s_data = pptx_page.json()
    assert s_data["kind"] == "text"
    assert s_data["page"] == 2
    assert "Lagrange" in s_data["text"]

    # Raw file download endpoint
    raw_res = client.get(f"/api/v1/subjects/{subject['id']}/materials/Appunti.pdf/file")
    assert raw_res.status_code == 200
    assert raw_res.content.startswith(b"%PDF")

    # Binary rejection (HTTP 415) for non-previewable binaries
    up_png = client.post(
        f"/api/v1/subjects/{subject['id']}/materials/upload",
        files={"file": ("img.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert up_png.status_code in (200, 201)
    png_page = client.get(f"/api/v1/subjects/{subject['id']}/materials/img.png/page/1")
    assert png_page.status_code == 415

    # Path traversal blocked
    trav = client.get(f"/api/v1/subjects/{subject['id']}/materials/..%2Fsettings.json/file")
    assert trav.status_code in (400, 403, 404)

    # Scoped LanceDB per-page retrieval returns page_start + clickable ref tag
    res = rag.query_context_structured(
        subject_id=subject["id"],
        query="Lagrange",
        top_k=3,
        scope_material_ids=["Lezione.pptx"],
    )
    assert "Lezione.pptx#" in res["context_text"]
    assert any(c["source_id"] == "Lezione.pptx" and c["page"] in (1, 2) for c in res["citations"])


def test_is_llm_ready_and_context_stats_fallback(client, subject, monkeypatch):
    from routers.chats import _is_llm_ready

    # 1. Direct _is_llm_ready helper tests
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda m: True)
    assert _is_llm_ready() is True

    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda m: False)
    assert _is_llm_ready() is False

    def explode(*a, **kw):
        raise RuntimeError("Disk error")

    monkeypatch.setattr(llm_engine, "resolve_llm_model", explode)
    assert _is_llm_ready() is False

    # Restore normal resolve
    monkeypatch.setattr(llm_engine, "resolve_llm_model", lambda *_: "qwen-3.5-4b")
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: False)

    # 2. When LLM is NOT downloaded, context stats must be 0 tokens and needs_compaction=False
    imported = client.post(
        f"/api/v1/chats/subject/{subject['id']}/import-legacy",
        json={
            "title": "Chat senza LLM",
            "messages": [
                {"role": "user", "content": "Spiega gli integrali di superficie."},
                {"role": "assistant", "content": "Il modello LLM non è ancora stato scaricato in locale."},
            ],
        },
    ).json()

    stats = imported["context_stats"]
    assert stats["used_tokens"] == 0
    assert stats["percentage"] == 0.0
    assert stats["needs_compaction"] is False
    assert stats["active_messages_count"] == 2

    # 3. When LLM IS downloaded, but all active messages are fallback messages
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    fb_chat = client.post(
        f"/api/v1/chats/subject/{subject['id']}/import-legacy",
        json={
            "title": "Chat con solo fallback",
            "messages": [
                {"role": "assistant", "content": "Il modello LLM non è ancora stato scaricato in locale. Scarica dalle impostazioni."},
            ],
        },
    ).json()

    fb_stats = fb_chat["context_stats"]
    assert fb_stats["used_tokens"] == 0
    assert fb_stats["percentage"] == 0.0
    assert fb_stats["needs_compaction"] is False


def test_compaction_up_to_last_message_and_subsequent_stream(client, subject, mock_chat_ai, monkeypatch):
    monkeypatch.setattr(
        llm_engine,
        "compact_chat_history",
        lambda *a, **kw: "Sintesi compattata fino all'ultimo messaggio.",
    )
    chat = client.post(f"/api/v1/chats/subject/{subject['id']}", json={"title": "Chat compattabile"}).json()
    chat_id = chat["id"]

    # Turn 1
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Messaggio uno"})
    # Turn 2
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Messaggio due"})

    detail = client.get(f"/api/v1/chats/{chat_id}").json()
    assert len(detail["messages"]) == 4  # u1, a1, u2, a2
    last_id = detail["messages"][-1]["id"]

    # Compact session
    compacted = client.post(f"/api/v1/chats/{chat_id}/compact").json()
    assert compacted["compacted_up_to_message_id"] == last_id
    assert compacted["compacted_summary"] == "Sintesi compattata fino all'ultimo messaggio."
    assert compacted["context_stats"]["compacted_messages_count"] == 4
    assert compacted["context_stats"]["active_messages_count"] == 0

    # Compacting again when no new messages exist preserves last_id
    compacted_again = client.post(f"/api/v1/chats/{chat_id}/compact").json()
    assert compacted_again["compacted_up_to_message_id"] == last_id
    assert compacted_again["context_stats"]["compacted_messages_count"] == 4
    assert compacted_again["context_stats"]["active_messages_count"] == 0

    # Turn 3 starts from compacted summary + new turn
    r3 = client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Messaggio tre"})
    assert r3.status_code == 200
    detail3 = client.get(f"/api/v1/chats/{chat_id}").json()
    assert detail3["compacted_up_to_message_id"] == last_id
    assert detail3["context_stats"]["compacted_messages_count"] == 4
    assert detail3["context_stats"]["active_messages_count"] == 2  # u3, a3


def test_edit_chat_message_stream_cascade_and_compaction(client, subject, mock_chat_ai, monkeypatch):
    monkeypatch.setattr(
        llm_engine,
        "compact_chat_history",
        lambda *a, **kw: "Riassunto compatto per edit",
    )
    chat = client.post(f"/api/v1/chats/subject/{subject['id']}", json={"title": "Chat edit test"}).json()
    chat_id = chat["id"]

    # Send 3 turns (6 messages)
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Domanda 1"})
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Domanda 2"})
    client.post(f"/api/v1/chats/{chat_id}/stream", json={"message": "Domanda 3"})

    detail = client.get(f"/api/v1/chats/{chat_id}").json()
    messages = detail["messages"]
    assert len(messages) == 6
    u1, a1, u2, a2, u3, a3 = messages

    # Compact up to message 6 (a3)
    compacted = client.post(f"/api/v1/chats/{chat_id}/compact").json()
    assert compacted["compacted_up_to_message_id"] == a3["id"]

    # Edit user message 2 (u2):
    # This must delete all messages with id > u2['id'] (i.e. a2, u3, a3)
    # and update compacted_up_to_message_id to prev_msg.id (a1['id'])
    # and stream a new assistant response.
    r_edit = client.post(
        f"/api/v1/chats/{chat_id}/messages/{u2['id']}/edit-stream",
        json={"message": "Domanda 2 modificata", "mode": "auto"},
    )
    assert r_edit.status_code == 200
    evs = _parse_ndjson(r_edit.text)
    done_ev = [e for e in evs if e["type"] == "done"][0]
    new_a2 = done_ev["assistant_message"]

    detail_after_edit = client.get(f"/api/v1/chats/{chat_id}").json()
    current_msgs = detail_after_edit["messages"]
    # There should now be u1, a1, u2 (modified), new_a2 -> 4 messages total
    assert len(current_msgs) == 4
    assert current_msgs[0]["id"] == u1["id"]
    assert current_msgs[1]["id"] == a1["id"]
    assert current_msgs[2]["id"] == u2["id"]
    assert current_msgs[2]["content"] == "Domanda 2 modificata"
    assert current_msgs[3]["id"] == new_a2["id"]
    # Verify old downstream turns (u3, a3) and old response a2 are gone
    assert not any(m["id"] in {u3["id"], a3["id"]} for m in current_msgs)
    assert not any(m["content"] == a2["content"] for m in current_msgs)
    assert not any(m["content"] == u3["content"] for m in current_msgs)
    assert not any(m["content"] == a3["content"] for m in current_msgs)
    assert new_a2["content"] != a2["content"]
    # Compaction was adjusted to a1['id']
    assert detail_after_edit["compacted_up_to_message_id"] == a1["id"]

    # Delete message u2 with cascade delete:
    # This deletes ChatMessage.id >= u2['id'], so u2 and new_a2 are deleted.
    # u1 and a1 remain.
    del_res = client.delete(f"/api/v1/chats/{chat_id}/messages/{u2['id']}")
    assert del_res.status_code == 200
    detail_after_del = del_res.json()
    assert len(detail_after_del["messages"]) == 2
    assert detail_after_del["messages"][0]["id"] == u1["id"]
    assert detail_after_del["messages"][1]["id"] == a1["id"]
    assert detail_after_del["compacted_up_to_message_id"] == a1["id"]

    # Delete u1 (the first message):
    # This deletes all remaining messages, compacted_up_to_message_id becomes 0, compacted_summary becomes None.
    del_all = client.delete(f"/api/v1/chats/{chat_id}/messages/{u1['id']}")
    assert del_all.status_code == 200
    detail_empty = del_all.json()
    assert len(detail_empty["messages"]) == 0
    assert detail_empty["compacted_up_to_message_id"] == 0
    assert detail_empty["compacted_summary"] == ""
