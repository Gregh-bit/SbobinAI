"""
Test per Step 2 (Map-Reduce adattivo), quantizzazione KV Cache q8_0 su llama.cpp,
fallback sicuro e cancellazione/threading in llm_engine.py.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types

import pytest

import ai_models
import llm_engine


def test_supports_llama_kv_quant_signature_checks():
    class ExplicitKV:
        def __init__(self, model_path, type_k=None, type_v=None):
            pass

    class KwargsOnly:
        def __init__(self, model_path, **kwargs):
            pass

    class LegacyFixed:
        def __init__(self, model_path, n_ctx=512):
            pass

    mod_with_q8 = types.SimpleNamespace(GGML_TYPE_Q8_0=8)
    mod_without_q8 = types.SimpleNamespace()

    assert llm_engine._supports_llama_kv_quant(ExplicitKV, mod_with_q8) is True
    assert llm_engine._supports_llama_kv_quant(KwargsOnly, mod_with_q8) is True
    assert llm_engine._supports_llama_kv_quant(KwargsOnly, mod_without_q8) is False
    assert llm_engine._supports_llama_kv_quant(KwargsOnly, None) is False
    assert llm_engine._supports_llama_kv_quant(LegacyFixed, mod_with_q8) is False


def test_load_llama_kv_q8_and_gc_fallback_order(monkeypatch):
    attempts = []
    gc_calls = {"count": 0}

    class FakeLlama:
        def __init__(self, model_path, n_gpu_layers=0, flash_attn=False, type_k=None, type_v=None, **kw):
            attempts.append((n_gpu_layers, flash_attn, type_k, type_v))
            # Simula fallimento delle configurazioni GPU/flash_attn finché non arriva a CPU base (0, False)
            if (n_gpu_layers, flash_attn, type_k) != (0, False, None):
                raise RuntimeError("unsupported flash/kv")

        def create_chat_completion(self, **kw):
            return {"choices": [{"message": {"content": ""}}]}

        def token_bos(self):
            return 1

        def token_eos(self):
            return 2

        def detokenize(self, toks):
            return b""

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.GGML_TYPE_Q8_0 = 8
    fake_mod.Llama = FakeLlama

    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)
    monkeypatch.setattr(llm_engine, "llama_gpu_offload_supported", lambda: True)
    monkeypatch.setattr(llm_engine.gc, "collect", lambda: gc_calls.__setitem__("count", gc_calls["count"] + 1) or 0)

    model, _ = llm_engine._load_llama("/fake/model.gguf")
    assert isinstance(model, FakeLlama)
    assert attempts[0] == (-1, True, 8, 8)
    assert (-1, True, None, None) in attempts
    assert (-1, False, None, None) in attempts
    assert (0, True, 8, 8) in attempts
    assert attempts[-1] == (0, False, None, None)
    assert gc_calls["count"] == len(attempts) - 1
    assert llm_engine._loaded_model_cfg == (0, False, False)


def test_stream_tokens_runtime_fallback_to_safe_llama(monkeypatch):
    seen_loads = []

    class UnsafeLlama:
        def tokenize(self, b, **kw):
            return [1, 2, 3]

        def n_ctx(self):
            return 2048

        def create_completion(self, **kw):
            raise RuntimeError("CUDA flash_attn kernel failed")

        def close(self):
            seen_loads.append("closed_unsafe")

        def token_bos(self):
            return 1

        def token_eos(self):
            return 2

        def detokenize(self, toks):
            return b""

    class SafeLlama:
        def tokenize(self, b, **kw):
            return [1, 2, 3]

        def n_ctx(self):
            return 2048

        def create_completion(self, **kw):
            return iter([{"choices": [{"text": "funziona"}]}])

        def close(self):
            seen_loads.append("closed_safe")

        def token_bos(self):
            return 1

        def token_eos(self):
            return 2

        def detokenize(self, toks):
            return b""

    unsafe = UnsafeLlama()
    safe = SafeLlama()

    def fake_load_llama(path, *, safe_only=False):
        seen_loads.append(("load", path, safe_only))
        llm_engine._loaded_model_cfg = (-1, False, False) if safe_only else (-1, True, True)
        return (safe if safe_only else unsafe), llm_engine._GGUFChatTokenizer(safe if safe_only else unsafe)

    monkeypatch.setattr(llm_engine, "USE_MLX", False)
    monkeypatch.setattr(llm_engine, "_load_llama", fake_load_llama)
    monkeypatch.setattr(llm_engine, "_loaded_model", unsafe)
    monkeypatch.setattr(llm_engine, "_loaded_tokenizer", llm_engine._GGUFChatTokenizer(unsafe))
    monkeypatch.setattr(llm_engine, "_loaded_model_id", "fake-gguf")
    monkeypatch.setattr(llm_engine, "_loaded_model_path", "/fake/model.gguf")
    monkeypatch.setattr(llm_engine, "_loaded_model_cfg", (-1, True, True))

    try:
        out = "".join(llm_engine.stream_tokens(unsafe, llm_engine._loaded_tokenizer, "ciao", max_tokens=64))
        assert out == "funziona"
        assert "closed_unsafe" in seen_loads
        assert ("load", "/fake/model.gguf", True) in seen_loads
        assert llm_engine._loaded_model is safe
    finally:
        llm_engine._release_loaded_model()


def test_unload_llm_waits_for_generation_lock(monkeypatch):
    events = []

    class DummyModel:
        def close(self):
            events.append("closed")

    monkeypatch.setattr(llm_engine, "USE_MLX", False)
    monkeypatch.setattr(llm_engine, "_loaded_model", DummyModel())
    monkeypatch.setattr(llm_engine, "_loaded_model_id", "dummy")

    entered_gen = threading.Event()
    t_unload_done = threading.Event()

    def run_gen():
        with llm_engine.generation_session("test"):
            entered_gen.set()
            events.append("gen_start")
            time.sleep(0.08)
            events.append("gen_end")

    def run_unload():
        entered_gen.wait(timeout=1.0)
        llm_engine.unload_llm()
        t_unload_done.set()

    t1 = threading.Thread(target=run_gen)
    t2 = threading.Thread(target=run_unload)
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert t_unload_done.is_set()
    # close() deve avvenire SOLO DOPO che la generazione attiva è uscita da _generation_lock
    assert events == ["gen_start", "gen_end", "closed"]


def test_split_transcript_chunks_boundaries_overlap_and_progress():
    # Verifica preferenza per separatore più a destra (\n vs " ") e nessun taglio a metà parola
    s = "Prima frase completa. " + ("parola " * 30) + "\nAltra riga lunga " + ("fine " * 10)
    chunks = llm_engine._split_transcript_chunks(s, target_chars=120, overlap_chars=20)
    assert len(chunks) >= 3
    assert all(len(c) <= 125 for c in chunks)

    # Verifica avanzamento rigoroso su stringa continua priva di spazi (anti loop infinito)
    solid = "A" * 500
    solid_chunks = llm_engine._split_transcript_chunks(solid, target_chars=100, overlap_chars=25)
    assert len(solid_chunks) >= 6


def test_clean_thinking_artifacts_handles_unclosed_and_embedded():
    raw = "Prefazione inutile <think>ragionamento lungo non chiuso..."
    assert llm_engine._clean_thinking_artifacts(raw) == "Prefazione inutile"
    raw2 = "<think>segreto</think>\n# Capitolo\nCorpo"
    assert llm_engine._clean_thinking_artifacts(raw2) == "# Capitolo\nCorpo"


def test_map_reduce_condensation_bookmarks_and_cancellation(monkeypatch):
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "unload_whisper", lambda *a, **kw: None)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda *a, **kw: None)
    monkeypatch.setattr(llm_engine, "SINGLE_PASS_TRANSCRIPT_CHARS", 200)
    monkeypatch.setattr(llm_engine, "MAP_CHUNK_TARGET_CHARS", 140)
    monkeypatch.setattr(llm_engine, "MAP_CHUNK_OVERLAP_CHARS", 20)
    monkeypatch.setattr(llm_engine, "REDUCE_MAX_CHARS", 220)

    # 1. Verifica Map-Reduce + condensazione gerarchica + cap bookmarks_summary a 1200 char
    long_text = " ".join(
        f"Argomento {i}: teorema spettrale sugli operatori lineari con proprietà dimostrata numero {i}."
        for i in range(12)
    )
    seen_calls = []

    def fake_blocking(messages, **kw):
        seen_calls.append(messages[1]["content"])
        if kw.get("max_tokens") == 700:
            # Restituisce note lunghe per forzare _condense_block_notes (> REDUCE_MAX_CHARS)
            return "- " + ("concetto chiave di algebra lineare " * 4)
        if kw.get("max_tokens") == 900:
            return "- sintesi condensata fedele"
        return "# Analisi — Capitolo di Studio\n\n## 1. Mappa\nok"

    monkeypatch.setattr(llm_engine, "_generate_blocking", fake_blocking)
    bm = "B" * 2000
    chap = asyncio.run(
        llm_engine.generate_lecture_chapter(
            title="Algebra",
            transcript=long_text,
            past_references="Riferimento precedente",
            bookmarks_summary=bm,
            model_id=ai_models.LLM_DEFAULT_ID,
        )
    )
    assert "Capitolo di Studio" in chap
    reduce_prompt = seen_calls[-1]
    assert "ESTRATTI CRONOLOGICI STRUTTURATI" in reduce_prompt
    assert "sintesi condensata fedele" in reduce_prompt
    # Segnalibri limitati a 1200 caratteri nel prompt finale
    assert ("B" * 1200) in reduce_prompt and ("B" * 1201) not in reduce_prompt

    # 2. Tutti i blocchi Map falliti -> LLMGenerationError
    monkeypatch.setattr(
        llm_engine,
        "_generate_blocking",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(llm_engine.LLMGenerationError):
        asyncio.run(llm_engine.generate_lecture_chapter(title="Algebra", transcript=long_text))

    # 3. Cancellazione a metà blocco -> InterruptedError immediata
    call_counter = {"n": 0}

    def cancel_on_second(messages, **kw):
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            with llm_engine._state_lock:
                llm_engine._cancel_epoch += 1
        if kw.get("raise_on_cancel") and llm_engine.is_cancelled(kw["epoch"]):
            raise InterruptedError("cancelled")
        return "blocco ok"

    monkeypatch.setattr(llm_engine, "_generate_blocking", cancel_on_second)
    with pytest.raises(InterruptedError):
        asyncio.run(llm_engine.generate_lecture_chapter(title="Algebra", transcript=long_text))


def test_generate_text_and_flash_respect_post_load_cancellation(monkeypatch):
    monkeypatch.setattr(llm_engine, "is_model_downloaded", lambda *_: True)
    monkeypatch.setattr(llm_engine, "clear_metal_cache", lambda *a, **kw: None)
    monkeypatch.setattr(llm_engine, "_touch_model_usage", lambda: None)

    def load_and_cancel(model_id=None):
        with llm_engine._state_lock:
            llm_engine._cancel_epoch += 1
        return object(), object()

    monkeypatch.setattr(llm_engine, "get_or_load_model", load_and_cancel)
    monkeypatch.setattr(
        llm_engine,
        "stream_tokens",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("stream_tokens should not run after cancel")),
    )

    with pytest.raises(InterruptedError):
        llm_engine.generate_text("sys", "usr", model_id=ai_models.LLM_DEFAULT_ID)

    flash_res = asyncio.run(llm_engine.generate_flash_answer("Domanda?", "Contesto", model_id=ai_models.LLM_DEFAULT_ID))
    assert "interrotta" in flash_res.lower()
