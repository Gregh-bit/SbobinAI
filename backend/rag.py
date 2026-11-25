"""
rag.py — SbobinAI RAG Module (LanceDB + Qwen3-Embedding 1024-dim, platform-aware)

  - LanceDB per il vector storage persistente e indicizzazione embedded
  - Qwen3-Embedding-0.6B come UNICO modello di embedding di sistema (1024 dimensioni):
      * macOS Apple Silicon → mlx-embeddings (mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ)
      * Windows / Linux     → llama-cpp-python (Qwen/Qwen3-Embedding-0.6B-GGUF Q8_0, pooling LAST)
    Lo schema LanceDB (vector float32[1024]) è identico su tutte le piattaforme.
  - Prompt con istruzione standard per le query di ricerca:
    "Instruct: Dato un quesito universitario, recupera i passaggi rilevanti dalle lezioni.\nQuery: {domanda}"
  - Ricerca Ibrida: ricerca vettoriale + ricerca lessicale (parole chiave/acronimi), fuse con RRF
  - Embedding calcolati a batch; rilascio della cache Metal dopo ogni indicizzazione e ricerca (Mac)
"""
from __future__ import annotations

from functools import lru_cache
import gc
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ai_models
from config import get_lancedb_dir
from platform_utils import USE_MLX, cpu_threads, llama_gpu_offload_supported

logger = logging.getLogger("sbobinai.rag")

# ── Modello Embedding Unico di Sistema (Qwen3-Embedding 1024-dim) ─────────────
DEFAULT_EMBEDDING_MODEL = ai_models.EMBEDDING_DEFAULT_ID
EMBEDDING_DIM = 1024
EMBED_BATCH_SIZE = 16
EMBED_MAX_TOKENS = 512

QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Dato un quesito universitario, recupera i passaggi rilevanti dalle lezioni.\nQuery: "
)


class EmbeddingModelUnavailable(RuntimeError):
    """Il modello di embedding non è scaricato o non può essere caricato."""


# ── Singleton Model & Table ────────────────────────────────────────────────────
_embed_lock = threading.RLock()      # protegge caricamento E inferenza (contesti non thread-safe)
_loaded_embed_model: Optional[Any] = None
_loaded_embed_processor: Optional[Any] = None
_loaded_embed_model_id: Optional[str] = None

_db_conn: Optional[Any] = None
_db_conn_path: Optional[str] = None
_table_lock = threading.Lock()
TABLE_NAME = "course_chunks"

_SCHEMA = None
_RESULT_COLUMNS = [
    "id",
    "subject_id",
    "lecture_id",
    "source_type",
    "source_id",
    "title",
    "text",
    "page_start",
    "page_end",
    "chunk_label",
    "created_at",
]


def _get_schema():
    """Schema LanceDB (pyarrow importato in modo lazy)."""
    global _SCHEMA
    if _SCHEMA is None:
        import pyarrow as pa
        _SCHEMA = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("subject_id", pa.int32()),
            pa.field("lecture_id", pa.int32()),  # 0 se materiale generico
            pa.field("source_type", pa.string()),  # 'lecture', 'slide', 'material', 'notes', 'infocorso'
            pa.field("source_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("chunk_label", pa.string()),
            pa.field("created_at", pa.float64()),
        ])
    return _SCHEMA


def __getattr__(name: str):  # retrocompatibilità: rag.SCHEMA
    if name == "SCHEMA":
        return _get_schema()
    raise AttributeError(name)


def _sql_str(value: str) -> str:
    """Letterale stringa SQL sicuro per i filtri LanceDB (apostrofi raddoppiati)."""
    return "'" + str(value).replace("'", "''") + "'"


def clear_metal_cache() -> None:
    """Rilascia la memoria di calcolo Metal (solo Mac) ed esegue la GC."""
    if USE_MLX:
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass
    gc.collect()


release_metal_cache = clear_metal_cache


def is_embedding_downloaded(model_id: Optional[str] = None) -> bool:
    """Verifica se il modello di embedding è scaricato in locale (snapshot completa)."""
    spec = ai_models.resolve(model_id, kind="embedding") or ai_models.CATALOG[DEFAULT_EMBEDDING_MODEL]
    return ai_models.is_downloaded(spec)[0]


def is_embedding_loaded() -> bool:
    """True se il modello di embedding è già caldo in RAM (zero latenza di caricamento)."""
    return _loaded_embed_model is not None


def reset_connection() -> None:
    """Chiude la connessione LanceDB in cache (da chiamare dopo un cambio di storage_path)."""
    global _db_conn, _db_conn_path
    with _table_lock:
        _db_conn = None
        _db_conn_path = None
    with _query_cache_lock:
        _query_vector_cache.clear()


def _migrate_table_schema_if_needed(db_conn: Any, tbl: Any) -> Any:
    """Aggiunge le colonne di citazione pagina/slide alle tabelle LanceDB esistenti."""
    col_defaults = {
        "page_start": ("cast(0 as int)", 0),
        "page_end": ("cast(0 as int)", 0),
        "chunk_label": ("cast('' as string)", ""),
    }
    try:
        existing_names = set(tbl.schema.names)
    except Exception:
        return tbl

    missing = [col for col in col_defaults if col not in existing_names]
    if not missing:
        return tbl

    transforms = {col: col_defaults[col][0] for col in missing}
    try:
        tbl.add_columns(transforms)
        logger.info(f"[RAG] Migrate colonne LanceDB mancanti: {missing}")
        return tbl
    except Exception as exc:
        logger.warning(f"[RAG] add_columns fallito ({exc}); ricreazione tabella con righe preservate.")
        try:
            rows = tbl.to_arrow().to_pylist()
            for row in rows:
                for col, (_, default_val) in col_defaults.items():
                    row.setdefault(col, default_val)
            db_conn.drop_table(TABLE_NAME)
            new_tbl = db_conn.create_table(TABLE_NAME, schema=_get_schema(), mode="create")
            if rows:
                new_tbl.add(rows)
            return new_tbl
        except Exception as rec_exc:
            logger.warning(f"[RAG] Fallback migrazione schema tabella fallito: {rec_exc}")
            return tbl


def get_lancedb_table():
    """
    Ottiene o crea la tabella LanceDB per i chunk del corso.
    La connessione è legata al percorso: se storage_path cambia, si riconnette automaticamente.
    Se la tabella esistente ha una dimensione vettore diversa da 1024, la ricrea.
    """
    global _db_conn, _db_conn_path
    import lancedb

    with _table_lock:
        db_dir = get_lancedb_dir()
        if _db_conn is None or _db_conn_path != db_dir:
            os.makedirs(db_dir, exist_ok=True)
            _db_conn = lancedb.connect(db_dir)
            _db_conn_path = db_dir

        try:
            tbl = _db_conn.open_table(TABLE_NAME)
        except Exception:
            return _db_conn.create_table(TABLE_NAME, schema=_get_schema(), mode="create", exist_ok=True)

        try:
            vec_field = tbl.schema.field("vector")
            cur_dim = getattr(vec_field.type, "list_size", None)
            if cur_dim is not None and cur_dim != EMBEDDING_DIM:
                logger.warning(
                    f"[RAG] Tabella {TABLE_NAME} ha dimensione {cur_dim} != {EMBEDDING_DIM}. "
                    "Ricreazione automatica per Qwen3-Embedding (1024 dim) e avvio reindicizzazione automatica."
                )
                _db_conn.drop_table(TABLE_NAME)
                new_tbl = _db_conn.create_table(TABLE_NAME, schema=_get_schema(), mode="create")
                _schedule_background_reindex()
                return new_tbl
        except Exception as exc:
            logger.debug(f"[RAG] Controllo schema tabella: {exc}")
        return _migrate_table_schema_if_needed(_db_conn, tbl)


def _schedule_background_reindex() -> None:
    """Ricostruisce in background i vettori per tutte le lezioni esistenti dopo un cambio di schema/dimensione."""
    def _worker():
        try:
            from database import SessionLocal, Lecture
            with SessionLocal() as db:
                lectures = db.query(Lecture).all()
                for lec in lectures:
                    if lec.raw_transcript or lec.processed_notes:
                        try:
                            index_lecture(
                                subject_id=lec.subject_id,
                                lecture_id=lec.id,
                                title=lec.title,
                                transcript=lec.raw_transcript or "",
                                notes=lec.processed_notes,
                            )
                        except Exception as e:
                            logger.warning(f"[RAG Reindex] Errore reindicizzazione lezione {lec.id}: {e}")
            logger.info("[RAG Reindex] Reindicizzazione automatica completata.")
        except Exception as exc:
            logger.warning(f"[RAG Reindex] Errore worker: {exc}")
        finally:
            unload_embedding_model()
            clear_metal_cache()

    threading.Thread(target=_worker, daemon=True, name="RAG-Reindex-Worker").start()


def _load_portable_embedder(model_path: str) -> Any:
    from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama

    gpu = llama_gpu_offload_supported()
    n_ctx = 4096
    return Llama(
        model_path=model_path,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_LAST,  # Qwen3-Embedding: pooling sull'ultimo token (EOS)
        n_ctx=n_ctx,
        n_batch=n_ctx,
        n_ubatch=n_ctx,
        n_gpu_layers=-1 if gpu else 0,
        n_threads=cpu_threads(),
        n_threads_batch=os.cpu_count() or cpu_threads(),
        verbose=False,
    )


def _get_embedding_model(model_id: Optional[str] = None) -> Optional[Tuple[Any, Any]]:
    """Carica il modello di embedding del backend corrente se scaricato su disco (singleton)."""
    global _loaded_embed_model, _loaded_embed_processor, _loaded_embed_model_id
    spec = ai_models.resolve(model_id, kind="embedding") or ai_models.CATALOG[DEFAULT_EMBEDDING_MODEL]
    target_id = spec.id

    with _embed_lock:
        if _loaded_embed_model is not None and _loaded_embed_model_id == target_id:
            return _loaded_embed_model, _loaded_embed_processor

        local_path = ai_models.local_model_path(spec) if not ai_models.is_downloading(spec.id) else None
        if not local_path:
            return None

        try:
            logger.info(f"[RAG] Caricamento modello embedding in RAM: {target_id} ({local_path})")
            if USE_MLX:
                import mlx_embeddings
                model, processor = mlx_embeddings.load(local_path)
            else:
                model, processor = _load_portable_embedder(local_path), None
                if model.n_embd() != EMBEDDING_DIM:
                    raise EmbeddingModelUnavailable(
                        f"Il modello di embedding produce {model.n_embd()} dimensioni invece di {EMBEDDING_DIM}."
                    )
            _loaded_embed_model = model
            _loaded_embed_processor = processor
            _loaded_embed_model_id = target_id
            return model, processor
        except Exception as exc:
            logger.error(f"[RAG] Errore caricamento embedding {target_id}: {exc}")
            return None


def unload_embedding_model() -> None:
    """Scarica il modello di embedding dalla RAM e azzera la cache (la connessione LanceDB viene riaperta al bisogno)."""
    global _loaded_embed_model, _loaded_embed_processor, _loaded_embed_model_id
    with _embed_lock:
        model = _loaded_embed_model
        _loaded_embed_model = None
        _loaded_embed_processor = None
        _loaded_embed_model_id = None
        if model is not None and not USE_MLX:
            try:
                model.close()
            except Exception:
                pass
        del model
    reset_connection()
    clear_metal_cache()
    logger.info("[RAG] Modello embedding scaricato e cache LanceDB azzerata.")


def _pool_and_normalize_mlx(embed_arr: Any) -> np.ndarray:
    import mlx.core as mx

    # Se mlx_embeddings restituisce un BaseModelOutput (come per Qwen3-Embedding), estrai il tensore
    if hasattr(embed_arr, "text_embeds") and embed_arr.text_embeds is not None:
        embed_arr = embed_arr.text_embeds
    elif hasattr(embed_arr, "embeddings") and getattr(embed_arr, "embeddings", None) is not None:
        embed_arr = embed_arr.embeddings
    elif hasattr(embed_arr, "last_hidden_state") and embed_arr.last_hidden_state is not None:
        embed_arr = embed_arr.last_hidden_state
    elif isinstance(embed_arr, (list, tuple)) and len(embed_arr) > 0:
        embed_arr = embed_arr[0]

    embed_arr_f32 = embed_arr.astype(mx.float32)
    if embed_arr_f32.ndim == 3:
        # Hidden states per token: last-token pooling (Qwen3-Embedding usa padding a sinistra)
        embed_arr_f32 = embed_arr_f32[:, -1, :]
    norm = mx.linalg.norm(embed_arr_f32, axis=-1, keepdims=True)
    embed_norm = embed_arr_f32 / (norm + 1e-9)
    return np.array(embed_norm, dtype=np.float32)


def _embed_batch(model: Any, processor: Any, texts: List[str]) -> np.ndarray:
    if USE_MLX:
        import mlx_embeddings
        out = mlx_embeddings.generate(
            model=model,
            processor=processor,
            texts=texts,
            max_length=EMBED_MAX_TOKENS,
            padding=True,
            truncation=True,
        )
        return _pool_and_normalize_mlx(out)

    # llama.cpp: tronca ogni testo a EMBED_MAX_TOKENS (come max_length su MLX) e normalizza L2
    clipped: List[str] = []
    for t in texts:
        toks = model.tokenize(t.encode("utf-8"), add_bos=False, special=False)
        if len(toks) > EMBED_MAX_TOKENS - 2:
            t = model.detokenize(toks[: EMBED_MAX_TOKENS - 2]).decode("utf-8", "ignore")
        clipped.append(t)
    vecs = model.embed(clipped, normalize=True, truncate=True)
    return np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1)


def compute_embeddings(
    texts: List[str],
    is_query: bool = False,
    model_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Calcola i vettori normalizzati L2 a 1024 dimensioni (shape [N, 1024]) per una lista di testi.
    - Se is_query=True, aggiunge l'istruzione formattata per Qwen3-Embedding.
    - Elaborazione a batch di EMBED_BATCH_SIZE testi per limitare il picco di memoria.
    Restituisce None se il modello non è disponibile o in caso di errore.
    """
    if not texts:
        return None

    model_proc = _get_embedding_model(model_id)
    if model_proc is None:
        return None
    model, processor = model_proc

    if is_query:
        formatted = [f"{QUERY_INSTRUCTION_PREFIX}{t.strip()}" for t in texts]
    else:
        formatted = [t.strip() for t in texts]

    try:
        out = np.empty((len(formatted), EMBEDDING_DIM), dtype=np.float32)
        with _embed_lock:
            for start in range(0, len(formatted), EMBED_BATCH_SIZE):
                batch = formatted[start : start + EMBED_BATCH_SIZE]
                vecs = _embed_batch(model, processor, batch)
                if vecs.shape != (len(batch), EMBEDDING_DIM):
                    raise ValueError(f"shape embedding inattesa {vecs.shape}")
                out[start : start + len(batch)] = vecs
        return out
    except Exception as exc:
        logger.error(f"[RAG] Errore calcolo embeddings: {exc}")
        return None
    finally:
        clear_metal_cache()


# ── Query Embedding Cache ─────────────────────────────────────────────────────

_query_vector_cache: dict[Tuple[str, Optional[str]], Tuple[float, ...]] = {}
_query_cache_lock = threading.Lock()
_QUERY_CACHE_MAXSIZE = 256


def get_cached_query_vector(query: str, model_id: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Calcola (o recupera da cache memoria thread-safe maxsize=256) il vettore di embedding normalizzato a 1024-dim
    per una query di ricerca. Evita ricalcoli GPU/CPU identici durante sessioni di studio e chat.
    Non memorizza in cache risultati None per evitare stale-state prima del download del modello.
    """
    clean_q = query.strip()
    if not clean_q:
        return None
    cache_key = (clean_q, model_id)
    with _query_cache_lock:
        if cache_key in _query_vector_cache:
            return np.array(_query_vector_cache[cache_key], dtype=np.float32)

    vecs = compute_embeddings([clean_q], is_query=True, model_id=model_id)
    if vecs is None or len(vecs) == 0:
        return None

    tup = tuple(vecs[0].tolist())
    with _query_cache_lock:
        if len(_query_vector_cache) >= _QUERY_CACHE_MAXSIZE:
            try:
                first_key = next(iter(_query_vector_cache))
                del _query_vector_cache[first_key]
            except (StopIteration, KeyError):
                pass
        _query_vector_cache[cache_key] = tup
    return np.array(tup, dtype=np.float32)


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 240, overlap: int = 40) -> List[str]:
    """Splitta un testo accademico in chunk coerenti con overlap (240 parole ~320-350 token, limite EMBED_MAX_TOKENS = 512)."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk) > 20:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


# ── Indicizzazione Lezioni & Materiali ─────────────────────────────────────────

def _embed_or_explain(chunks: List[str], what: str) -> Optional[np.ndarray]:
    vectors = compute_embeddings(chunks, is_query=False)
    if vectors is None or len(vectors) != len(chunks):
        if not is_embedding_downloaded():
            logger.error(
                f"[RAG] Indicizzazione di {what} saltata: modello di embedding non scaricato. "
                "Scaricalo da Impostazioni -> Modelli e reindicizza."
            )
        else:
            logger.error(f"[RAG] Indicizzazione di {what} saltata: calcolo embedding non riuscito.")
        return None
    return vectors


def _build_chunk_record(
    *,
    record_id: str,
    vector: Any,
    subject_id: int,
    lecture_id: int,
    source_type: str,
    source_id: str,
    title: str,
    text: str,
    created_at: float,
    page_start: int = 0,
    page_end: int = 0,
    chunk_label: str = "",
) -> Dict[str, Any]:
    """Costruisce un record LanceDB conforme allo schema corrente."""
    return {
        "id": record_id,
        "vector": vector,
        "subject_id": int(subject_id),
        "lecture_id": int(lecture_id),
        "source_type": source_type,
        "source_id": source_id,
        "title": title,
        "text": text,
        "page_start": int(page_start),
        "page_end": int(page_end),
        "chunk_label": chunk_label,
        "created_at": created_at,
    }


def index_lecture(
    subject_id: int,
    lecture_id: int,
    title: str,
    transcript: str,
    notes: Optional[str] = None,
) -> int:
    """
    Indicizza la trascrizione e le note elaborate di una lezione in LanceDB (1024 dim).
    Gli embedding vengono calcolati PRIMA di rimuovere i chunk obsoleti: se il calcolo fallisce
    l'indice precedente resta intatto. Restituisce il numero di chunk indicizzati (0 se saltato).
    """
    if not transcript and not notes:
        return 0

    full_body = ""
    if notes:
        full_body += f"Appunti e schema del Quaderno:\n{notes}\n\n"
    if transcript:
        from transcript_cleaner import clean_transcript_for_llm
        cleaned = clean_transcript_for_llm(transcript)
        if cleaned:
            full_body += f"Trascrizione spiegazione docente:\n{cleaned}"

    chunks = chunk_text(full_body, chunk_size=240, overlap=40)
    if not chunks:
        return 0

    try:
        vectors = _embed_or_explain(chunks, f"lezione {lecture_id}")
        if vectors is None:
            return 0

        now = time.time()
        records = [
            _build_chunk_record(
                record_id=f"lec-{lecture_id}-{idx}-{uuid.uuid4().hex[:6]}",
                vector=vectors[idx],
                subject_id=subject_id,
                lecture_id=lecture_id,
                source_type="lecture",
                source_id=f"lecture_{lecture_id}",
                title=title or f"Lezione {lecture_id}",
                text=ch,
                created_at=now,
            )
            for idx, ch in enumerate(chunks)
        ]

        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {int(subject_id)} AND lecture_id = {int(lecture_id)}")
        tbl.add(records)
        logger.info(f"[RAG] Indicizzati {len(records)} chunk (1024 dim) per la lezione {lecture_id} (Materia {subject_id}).")
        return len(records)
    except Exception as exc:
        logger.error(f"[RAG] Errore indicizzazione lezione {lecture_id}: {exc}")
        return 0
    finally:
        clear_metal_cache()


def index_material_document(
    subject_id: int,
    material_id: str,
    title: str,
    content: str,
    source_type: str = "material",
) -> int:
    """Indicizza un documento di materiale del corso (PDF, slide, InfoCorso.md) in LanceDB."""
    if not content or len(content.strip()) < 10:
        return 0

    chunks = chunk_text(content, chunk_size=240, overlap=40)
    if not chunks:
        return 0

    try:
        vectors = _embed_or_explain(chunks, f"materiale '{title or material_id}'")
        if vectors is None:
            return 0

        now = time.time()
        records = [
            _build_chunk_record(
                record_id=f"mat-{subject_id}-{idx}-{uuid.uuid4().hex[:6]}",
                vector=vectors[idx],
                subject_id=subject_id,
                lecture_id=0,
                source_type=source_type,
                source_id=material_id,
                title=title or material_id,
                text=ch,
                created_at=now,
            )
            for idx, ch in enumerate(chunks)
        ]

        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {int(subject_id)} AND source_id = {_sql_str(material_id)}")
        tbl.add(records)
        logger.info(f"[RAG] Indicizzati {len(records)} chunk per il materiale '{title}'.")
        return len(records)
    except Exception as exc:
        logger.error(f"[RAG] Errore indicizzazione materiale {material_id}: {exc}")
        return 0
    finally:
        clear_metal_cache()


def index_material_pages(
    subject_id: int,
    source_id: str,
    title: str,
    pages: List[Dict[str, Any]],
    source_type: str = "slide",
) -> int:
    """Indicizza un materiale pagina/slide per pagina conservando i riferimenti per le citazioni."""
    if not pages:
        return 0

    chunk_items: List[Tuple[str, int, str]] = []
    for p in pages:
        page_num = int(p.get("page") or 0)
        page_text = str(p.get("text") or "").strip()
        if not page_text:
            continue
        label = str(p.get("label") or "").strip() or (f"Pag. {page_num}" if page_num > 0 else "")
        for ch in chunk_text(page_text, chunk_size=240, overlap=40):
            chunk_items.append((ch, page_num, label))

    if not chunk_items:
        return 0

    try:
        vectors = _embed_or_explain([item[0] for item in chunk_items], f"materiale '{title or source_id}'")
        if vectors is None:
            return 0

        now = time.time()
        records = [
            _build_chunk_record(
                record_id=f"mat-{subject_id}-p{page_num}-{idx}-{uuid.uuid4().hex[:6]}",
                vector=vectors[idx],
                subject_id=subject_id,
                lecture_id=0,
                source_type=source_type,
                source_id=source_id,
                title=title or source_id,
                text=ch,
                created_at=now,
                page_start=page_num,
                page_end=page_num,
                chunk_label=label,
            )
            for idx, (ch, page_num, label) in enumerate(chunk_items)
        ]

        tbl = get_lancedb_table()
        tbl.delete(
            f"subject_id = {int(subject_id)} AND source_id = {_sql_str(source_id)} "
            f"AND source_type IN ({_sql_str(source_type)}, 'material', 'slide', 'pdf')"
        )
        tbl.add(records)
        logger.info(f"[RAG] Indicizzati {len(records)} chunk per pagina/slide del materiale '{title or source_id}'.")
        return len(records)
    except Exception as exc:
        logger.error(f"[RAG] Errore indicizzazione pagine materiale {source_id}: {exc}")
        return 0
    finally:
        clear_metal_cache()


def delete_material_vectors(subject_id: int, material_id: str) -> None:
    """Elimina i vettori associati a un materiale da LanceDB."""
    try:
        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {int(subject_id)} AND source_id = {_sql_str(material_id)}")
        logger.info(f"[RAG] Eliminati vettori LanceDB per materiale '{material_id}' (Materia {subject_id}).")
    except Exception as exc:
        logger.warning(f"[RAG] Errore eliminazione vettori materiale {material_id}: {exc}")


def delete_lecture_vectors(subject_id: int, lecture_id: int) -> None:
    """Elimina i vettori associati a una specifica lezione da LanceDB."""
    try:
        tbl = get_lancedb_table()
        cond = f"subject_id = {int(subject_id)} AND (lecture_id = {int(lecture_id)} OR source_id = 'lecture_{int(lecture_id)}')"
        tbl.delete(cond)
        logger.info(f"[RAG] Eliminati vettori LanceDB per lezione {lecture_id} (Materia {subject_id}).")
    except Exception as exc:
        logger.warning(f"[RAG] Errore eliminazione vettori lezione {lecture_id}: {exc}")


def delete_subject_vectors(subject_id: int) -> None:
    """Elimina TUTTI i vettori di una materia (lezioni e materiali) da LanceDB."""
    try:
        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {int(subject_id)}")
        logger.info(f"[RAG] Eliminati tutti i vettori LanceDB per Materia {subject_id}.")
    except Exception as exc:
        logger.warning(f"[RAG] Errore eliminazione vettori materia {subject_id}: {exc}")


def compact_lancedb() -> None:
    """Compatta i file e le tabelle LanceDB liberando blocchi non usati."""
    try:
        tbl = get_lancedb_table()
        if hasattr(tbl, "optimize"):
            tbl.optimize()
            logger.info("[RAG] LanceDB optimize eseguito con successo.")
    except Exception as exc:
        logger.debug(f"[RAG] LanceDB optimize: {exc}")


# ── Query & Ricerca Ibrida (Vector + Keyword/Acronimi, fusione RRF) ───────────

RRF_K = 60.0

_IT_STOPWORDS = frozenset("""
alla alle allo agli anche ancora avere aveva avevano come con cosa cose dalla dalle dallo dagli degli della delle dello
dentro dove dopo essere fare fatto fino hanno hai però perché poi più quale quali quando quanto questa queste questi
questo quella quelle quelli quello sarà sempre senza siamo sono sopra sotto stata stato sugli sulla sulle sullo tanto
tutta tutte tutti tutto una uno vuol vuoi viene vengono nella nelle nello negli ogni oppure molto mentre cioè ecco
spiegami spiega spiegare dimmi dammi parlami cosè significa vuole intende professore docente lezione lezioni
what which with that this from have about there their would could should
""".split())


def _query_terms(query: str) -> Tuple[List["re.Pattern[str]"], List[str]]:
    """Termini lessicali significativi (>=4 lettere, no stopword) + acronimi esatti (>=2 maiuscole/cifre)."""
    words = {w.lower() for w in re.findall(r"\b\w{4,}\b", query, flags=re.UNICODE)}
    words = {w for w in words if w not in _IT_STOPWORDS and not w.isdigit()}
    acronyms = sorted(set(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", query)))
    patterns = [re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE | re.UNICODE) for w in sorted(words)]
    return patterns, acronyms


def _build_scope_where(
    subject_id: int,
    scope_lecture_ids: Optional[List[int]],
    scope_material_ids: Optional[List[str]],
) -> Tuple[str, set[int], set[str]]:
    """Costruisce il filtro SQL LanceDB e i set di verifica per lo scope opzionale."""
    base_where = f"subject_id = {int(subject_id)}"
    lec_id_set = {int(lid) for lid in (scope_lecture_ids or []) if int(lid) > 0}
    mat_id_set = {str(mid) for mid in (scope_material_ids or []) if str(mid).strip()}
    if not lec_id_set and not mat_id_set:
        return base_where, lec_id_set, mat_id_set

    clauses: List[str] = []
    if lec_id_set:
        ids_csv = ", ".join(str(lid) for lid in sorted(lec_id_set))
        clauses.append(f"lecture_id IN ({ids_csv})")
    if mat_id_set:
        mids_csv = ", ".join(_sql_str(mid) for mid in sorted(mat_id_set))
        clauses.append(f"source_id IN ({mids_csv})")
    return f"{base_where} AND ({' OR '.join(clauses)})", lec_id_set, mat_id_set


def _record_in_scope(r: Dict[str, Any], lec_id_set: set[int], mat_id_set: set[str]) -> bool:
    """Verifica difensiva che un chunk appartenga allo scope selezionato."""
    if not lec_id_set and not mat_id_set:
        return True
    lec_id = int(r.get("lecture_id") or 0)
    src_id = str(r.get("source_id") or "")
    return (lec_id in lec_id_set) or (src_id in mat_id_set)


def _format_snippet_and_citation(r: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Formatta uno snippet RAG con metadati di citazione e il relativo dizionario CitationRef."""
    source_id = str(r.get("source_id") or "")
    source_type = str(r.get("source_type") or "material")
    title = str(r.get("title") or "Appunti del corso")
    text = str(r.get("text") or "")
    page_start = int(r.get("page_start") or 0)
    raw_label = str(r.get("chunk_label") or "").strip()
    lecture_id = int(r.get("lecture_id") or 0)

    if source_type == "lecture" or lecture_id > 0:
        actual_lec_id = lecture_id if lecture_id > 0 else (
            int(source_id.replace("lecture_", ""))
            if source_id.startswith("lecture_") and source_id.replace("lecture_", "").isdigit()
            else 0
        )
        time_sec = page_start
        time_label = f"{time_sec // 60}:{time_sec % 60:02d}" if time_sec > 0 else ""
        label_text = raw_label or (f"{title} ({time_label})" if time_label else title)
        snippet = (
            f"[FONTE AUDIO DOCENTE: {title} | LEZIONE_ID: {actual_lec_id} | SECONDI: {time_sec} | "
            f"LINK: [[audio:{actual_lec_id}|{time_sec}|{label_text}]]]\n{text}"
        )
        citation = {
            "source_id": str(actual_lec_id) if actual_lec_id > 0 else source_id,
            "source_type": "lecture",
            "title": title,
            "page": time_sec if time_sec > 0 else None,
            "label": label_text,
        }
    elif page_start > 0:
        label_or_page = raw_label or f"Pag. {page_start}"
        snippet = (
            f"[FONTE: {title} | REF: {source_id}#{page_start} | ETICHETTA: {label_or_page} | "
            f"LINK: [[slide:{source_id}#{page_start}|{label_or_page}]]]\n{text}"
        )
        citation = {
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "page": page_start,
            "label": label_or_page,
        }
    else:
        link_tag = "doc" if source_type in ("notes", "doc") else "slide"
        page_suffix = "" if link_tag == "doc" else "#1"
        snippet = f"[FONTE: {title} ({source_type}) | REF: {source_id} | LINK: [[{link_tag}:{source_id}{page_suffix}|{title}]]]\n{text}"
        citation = {
            "source_id": source_id,
            "source_type": source_type,
            "title": title,
            "page": None,
            "label": raw_label or title,
        }
    return snippet, citation


def query_context_structured(
    subject_id: int,
    query: str,
    top_k: int = 5,
    priority_previous_lecture: bool = False,
    scope_lecture_ids: Optional[List[int]] = None,
    scope_material_ids: Optional[List[str]] = None,
    use_vector: bool = True,
) -> Dict[str, Any]:
    """
    Recupera i chunk più pertinenti tramite Hybrid Search (vettoriale + lessicale RRF),
    con supporto per scope opzionale su lezioni/materiali e citazioni strutturate.
    Con `use_vector=False` esegue solo ricerca lessicale istantanea (<5ms) senza caricare l'embedder.
    Restituisce {"context_text": str, "citations": list[dict]}.
    """
    clean_q = query.strip()
    if not clean_q:
        return {"context_text": "", "citations": []}

    try:
        tbl = get_lancedb_table()
        where, lec_id_set, mat_id_set = _build_scope_where(subject_id, scope_lecture_ids, scope_material_ids)
        has_scope = bool(lec_id_set or mat_id_set)
        pool_limit = top_k * 6 if has_scope else top_k * 4
        records: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}

        # 1. Ricerca vettoriale con cache LRU (256 query) — saltata in modalità ultra-rapida se richiesto
        q_vec = get_cached_query_vector(clean_q) if use_vector else None
        if q_vec is not None:
            try:
                vec_matches = (
                    tbl.search(q_vec)
                    .where(where, prefilter=True)
                    .select(_RESULT_COLUMNS + ["_distance"])
                    .limit(pool_limit)
                    .to_list()
                )
                for rank, r in enumerate(vec_matches):
                    if not _record_in_scope(r, lec_id_set, mat_id_set):
                        continue
                    rid = str(r.get("id") or uuid.uuid4())
                    records[rid] = r
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception as e:
                logger.warning(f"[RAG] Errore vector search LanceDB: {e}")

        # 2. Ricerca lessicale (colonne di risultato complete, filtrata per materia e scope)
        patterns, acronyms = _query_terms(clean_q)
        if patterns or acronyms:
            try:
                rows = (
                    tbl.search()
                    .where(where)
                    .select(_RESULT_COLUMNS)
                    .limit(500)
                    .to_list()
                )
                lexical: List[Tuple[float, str, Dict[str, Any]]] = []
                for row in rows:
                    if not _record_in_scope(row, lec_id_set, mat_id_set):
                        continue
                    txt = str(row.get("text") or "")
                    title = str(row.get("title") or "")
                    score = 0.0
                    for p in patterns:
                        hits = len(p.findall(txt))
                        if hits:
                            score += 1.0 + min(hits, 5) * 0.2
                        if p.search(title):
                            score += 1.5
                    for acr in acronyms:
                        if re.search(r"\b" + re.escape(acr) + r"\b", txt) or acr in title:
                            score += 2.0
                    if score > 0:
                        lexical.append((score, str(row.get("id")), row))
                lexical.sort(key=lambda x: x[0], reverse=True)
                for rank, (_, rid, row) in enumerate(lexical[:pool_limit]):
                    if rid not in records:
                        records[rid] = row
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception as e:
                logger.debug(f"[RAG] Keyword scan error: {e}")

        if not scores:
            return {"context_text": "", "citations": []}

        # Piccolo boost (scala RRF) per le lezioni se richiesto
        if priority_previous_lecture:
            for rid, r in records.items():
                if r.get("source_type") == "lecture":
                    scores[rid] = scores.get(rid, 0.0) + 0.3 / (RRF_K + 1)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        formatted_chunks: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen_citations: set[Tuple[str, str, Optional[int]]] = set()

        for rid, _ in ranked:
            snippet, cit = _format_snippet_and_citation(records[rid])
            formatted_chunks.append(snippet)
            cit_key = (cit["source_id"], cit["source_type"], cit["page"])
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append(cit)

        return {
            "context_text": "\n\n".join(formatted_chunks),
            "citations": citations,
        }

    except Exception as exc:
        logger.warning(f"[RAG] Ricerca contesto non riuscita: {exc}")
        return {"context_text": "", "citations": []}
    finally:
        clear_metal_cache()


def query_context(
    subject_id: int,
    query: str,
    top_k: int = 4,
    priority_previous_lecture: bool = False,
) -> str:
    """
    Recupera i chunk più pertinenti filtrando RIGIDAMENTE per subject_id tramite Hybrid Search:
      1. Ricerca Vettoriale semantica (Qwen3-Embedding con query instruction prefix standard).
      2. Ricerca lessicale con confini di parola su termini significativi e acronimi esatti.
      3. Fusione Reciprocal Rank Fusion (RRF) dei due ranking.
      4. Fallback solo lessicale se l'embedding non è ancora scaricato.
    """
    return query_context_structured(
        subject_id=subject_id,
        query=query,
        top_k=top_k,
        priority_previous_lecture=priority_previous_lecture,
    )["context_text"]
