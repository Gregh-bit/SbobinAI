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
_RESULT_COLUMNS = ["id", "subject_id", "lecture_id", "source_type", "source_id", "title", "text", "created_at"]


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
            pa.field("source_type", pa.string()),  # 'lecture', 'material', 'notes', 'infocorso'
            pa.field("source_id", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
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


def is_embedding_downloaded(model_id: Optional[str] = None) -> bool:
    """Verifica se il modello di embedding è scaricato in locale (snapshot completa)."""
    spec = ai_models.resolve(model_id, kind="embedding") or ai_models.CATALOG[DEFAULT_EMBEDDING_MODEL]
    return ai_models.is_downloaded(spec)[0]


def reset_connection() -> None:
    """Chiude la connessione LanceDB in cache (da chiamare dopo un cambio di storage_path)."""
    global _db_conn, _db_conn_path
    with _table_lock:
        _db_conn = None
        _db_conn_path = None


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
                    "Ricreazione automatica per Qwen3-Embedding (1024 dim)."
                )
                _db_conn.drop_table(TABLE_NAME)
                return _db_conn.create_table(TABLE_NAME, schema=_get_schema(), mode="create")
        except Exception as exc:
            logger.debug(f"[RAG] Controllo schema tabella: {exc}")
        return tbl


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


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splitta un testo accademico in chunk coerenti con overlap."""
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
        full_body += f"Trascrizione spiegazione docente:\n{transcript}"

    chunks = chunk_text(full_body, chunk_size=350, overlap=50)
    if not chunks:
        return 0

    try:
        vectors = _embed_or_explain(chunks, f"lezione {lecture_id}")
        if vectors is None:
            return 0

        now = time.time()
        records = [
            {
                "id": f"lec-{lecture_id}-{idx}-{uuid.uuid4().hex[:6]}",
                "vector": vectors[idx],
                "subject_id": int(subject_id),
                "lecture_id": int(lecture_id),
                "source_type": "lecture",
                "source_id": f"lecture_{lecture_id}",
                "title": title or f"Lezione {lecture_id}",
                "text": ch,
                "created_at": now,
            }
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

    chunks = chunk_text(content, chunk_size=350, overlap=50)
    if not chunks:
        return 0

    try:
        vectors = _embed_or_explain(chunks, f"materiale '{title or material_id}'")
        if vectors is None:
            return 0

        now = time.time()
        records = [
            {
                "id": f"mat-{subject_id}-{idx}-{uuid.uuid4().hex[:6]}",
                "vector": vectors[idx],
                "subject_id": int(subject_id),
                "lecture_id": 0,
                "source_type": source_type,
                "source_id": material_id,
                "title": title or material_id,
                "text": ch,
                "created_at": now,
            }
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
        tbl.delete(f"subject_id = {int(subject_id)} AND lecture_id = {int(lecture_id)}")
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
    clean_q = query.strip()
    if not clean_q:
        return ""

    try:
        tbl = get_lancedb_table()
        where = f"subject_id = {int(subject_id)}"
        records: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}

        # 1. Ricerca vettoriale
        query_vector = compute_embeddings([clean_q], is_query=True)
        if query_vector is not None and len(query_vector) > 0:
            try:
                vec_matches = (
                    tbl.search(query_vector[0])
                    .where(where, prefilter=True)
                    .select(_RESULT_COLUMNS + ["_distance"])
                    .limit(top_k * 4)
                    .to_list()
                )
                for rank, r in enumerate(vec_matches):
                    rid = str(r.get("id") or uuid.uuid4())
                    records[rid] = r
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception as e:
                logger.warning(f"[RAG] Errore vector search LanceDB: {e}")

        # 2. Ricerca lessicale (solo colonne necessarie, solo la materia richiesta)
        patterns, acronyms = _query_terms(clean_q)
        if patterns or acronyms:
            try:
                rows = (
                    tbl.search()
                    .where(where)
                    .select(["id", "title", "text", "source_type"])
                    .limit(None)
                    .to_list()
                )
                lexical: List[Tuple[float, str, Dict[str, Any]]] = []
                for row in rows:
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
                for rank, (_, rid, row) in enumerate(lexical[: top_k * 4]):
                    if rid not in records:
                        records[rid] = row
                    scores[rid] = scores.get(rid, 0.0) + 1.0 / (RRF_K + rank + 1)
            except Exception as e:
                logger.debug(f"[RAG] Keyword scan error: {e}")

        if not scores:
            return ""

        # Piccolo boost (scala RRF) per le lezioni se richiesto
        if priority_previous_lecture:
            for rid, r in records.items():
                if r.get("source_type") == "lecture":
                    scores[rid] = scores.get(rid, 0.0) + 0.3 / (RRF_K + 1)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        formatted_chunks = []
        for rid, _ in ranked:
            r = records[rid]
            title = r.get("title") or "Appunti del corso"
            text = r.get("text") or ""
            formatted_chunks.append(f"[{title}]:\n{text}")
        return "\n\n".join(formatted_chunks)

    except Exception as exc:
        logger.warning(f"[RAG] Ricerca contesto non riuscita: {exc}")
        return ""
    finally:
        clear_metal_cache()
