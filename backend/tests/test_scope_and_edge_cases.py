from __future__ import annotations

import io
import json
from typing import Any
import pytest
from database import ChatSession, Lecture, SessionLocal


def test_delete_material_scrubs_chat_scope(client, subject):
    # 1. Upload a material file
    content = b"%PDF-1.4 test dummy"
    files = {"file": ("dispensa.pdf", io.BytesIO(content), "application/pdf")}
    up_res = client.post(f"/api/v1/subjects/{subject['id']}/materials/upload", files=files)
    assert up_res.status_code == 201, up_res.text

    # 2. Create a chat session scoped to this material
    c_res = client.post(
        f"/api/v1/chats/subject/{subject['id']}",
        json={"title": "Chat con dispensa", "scope": {"lecture_ids": [], "material_ids": ["dispensa.pdf"]}},
    )
    assert c_res.status_code == 200
    chat_id = c_res.json()["id"]
    assert c_res.json()["scope"]["material_ids"] == ["dispensa.pdf"]

    # 3. Delete the material file
    del_res = client.delete(f"/api/v1/subjects/{subject['id']}/materials/dispensa.pdf")
    assert del_res.status_code == 204

    # 4. Verify that ChatSession.scope_json in DB was automatically scrubbed
    detail = client.get(f"/api/v1/chats/{chat_id}").json()
    assert "dispensa.pdf" not in detail["scope"]["material_ids"]
    assert detail["scope"]["material_ids"] == []


def test_delete_lecture_scrubs_chat_scope(client, subject):
    # 1. Create a lecture
    lec_res = client.post(
        "/api/v1/lectures/",
        json={"subject_id": subject["id"], "title": "Lezione Test Scope", "raw_transcript": "Trascrizione test."},
    )
    assert lec_res.status_code == 201
    lec_id = lec_res.json()["id"]

    # 2. Create a chat session scoped to this lecture
    c_res = client.post(
        f"/api/v1/chats/subject/{subject['id']}",
        json={"title": "Chat con lezione", "scope": {"lecture_ids": [lec_id], "material_ids": []}},
    )
    assert c_res.status_code == 200
    chat_id = c_res.json()["id"]
    assert c_res.json()["scope"]["lecture_ids"] == [lec_id]

    # 3. Delete the lecture
    del_res = client.delete(f"/api/v1/lectures/{lec_id}")
    assert del_res.status_code == 204

    # 4. Verify that ChatSession.scope_json in DB was automatically scrubbed
    detail = client.get(f"/api/v1/chats/{chat_id}").json()
    assert lec_id not in detail["scope"]["lecture_ids"]
    assert detail["scope"]["lecture_ids"] == []


def test_stream_fallback_when_scoped_files_deleted(client, subject, monkeypatch):
    import llm_engine

    monkeypatch.setattr(
        llm_engine,
        "stream_multi_turn_chat",
        lambda *a, **kw: iter(["Risposta", " di", " fallback"]),
    )
    # Create chat with scoped material that does not exist in LanceDB or disk
    c_res = client.post(
        f"/api/v1/chats/subject/{subject['id']}",
        json={"title": "Chat fantasma", "scope": {"lecture_ids": [99999], "material_ids": ["non_esiste.pdf"]}},
    )
    chat_id = c_res.json()["id"]

    # Stream a message: it should not crash, it should fallback cleanly
    s_res = client.post(
        f"/api/v1/chats/{chat_id}/stream",
        json={"message": "Spiegami questo argomento"},
    )
    assert s_res.status_code == 200
    assert "Risposta" in s_res.text

