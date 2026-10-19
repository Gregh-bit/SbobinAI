"""
rag.py — SbobinAI Pure MLX & LanceDB RAG Module
Motore di Retrieval-Augmented Generation 100% nativo Apple Silicon basato su:
  - LanceDB per il vector storage persistente e indicizzazione embedded
  - Qwen3-Embedding-0.6B-4bit-DWQ come UNICO modello di embedding di sistema (1024 dimensioni, ~300MB RAM)
  - Prompt con istruzione standard per le query di ricerca:
    "Instruct: Dato un quesito universitario, recupera i passaggi rilevanti dalle lezioni.\nQuery: {domanda}"
  - Ricerca Ibrida (Hybrid Search): Ricerca Vettoriale semantica + Keyword/Acronimi per sigle e termini tecnici
  - Zero dipendenze PyTorch / sentence-transformers
  - Rilascio sistematico della cache Metal (mx.metal.clear_cache()) dopo ogni indicizzazione e ricerca
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

import lancedb
import numpy as np
import pyarrow as pa

from config import get_lancedb_dir

logger = logging.getLogger("sbobinai.rag")

# ── Modello Embedding Unico di Sistema (Qwen3-Embedding 1024-dim MLX) ───────────
DEFAULT_EMBEDDING_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
EMBEDDING_DIM = 1024

QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Dato un quesito universitario, recupera i passaggi rilevanti dalle lezioni.\nQuery: "
)

# ── Singleton Model & Table ────────────────────────────────────────────────────
_embed_lock = threading.Lock()
_loaded_embed_model: Optional[Any] = None
_loaded_embed_processor: Optional[Any] = None
_loaded_embed_model_id: Optional[str] = None

_db_conn: Optional[lancedb.DBConnection] = None
_table_lock = threading.Lock()
TABLE_NAME = "course_chunks"

SCHEMA = pa.schema([
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


def clear_metal_cache() -> None:
    """Rilascia immediatamente la memoria grafica allocata da MLX a macOS."""
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
    """Verifica se il modello di embedding è scaricato in locale."""
    target = model_id or DEFAULT_EMBEDDING_MODEL
    try:
        from routers.models import check_repo_downloaded
        downloaded, _ = check_repo_downloaded(target)
        return downloaded
    except Exception:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        folder = "models--" + target.replace("/", "--")
        p = os.path.join(cache_dir, folder, "snapshots")
        return os.path.exists(p) and bool(os.listdir(p))


def get_lancedb_table():
    """
    Ottiene o crea la tabella LanceDB per i chunk del corso.
    Verifica che la tabella esistente abbia la corretta dimensione vettore (1024).
    In caso contrario, la ricrea per il nuovo motore Qwen3-Embedding.
    """
    global _db_conn
    with _table_lock:
        if _db_conn is None:
            db_dir = get_lancedb_dir()
            os.makedirs(db_dir, exist_ok=True)
            _db_conn = lancedb.connect(db_dir)

        table_names = _db_conn.table_names()
        if TABLE_NAME in table_names:
            tbl = _db_conn.open_table(TABLE_NAME)
            try:
                vec_field = tbl.schema.field("vector")
                cur_dim = getattr(vec_field.type, "list_size", None)
                if cur_dim is not None and cur_dim != EMBEDDING_DIM:
                    logger.warning(
                        f"[RAG] Tabella {TABLE_NAME} ha dimensione {cur_dim} != {EMBEDDING_DIM}. "
                        "Ricreazione automatica per Qwen3-Embedding (1024 dim)."
                    )
                    _db_conn.drop_table(TABLE_NAME)
                    return _db_conn.create_table(TABLE_NAME, schema=SCHEMA, mode="create")
            except Exception as exc:
                logger.debug(f"[RAG] Controllo schema tabella: {exc}")
            return tbl
        else:
            return _db_conn.create_table(TABLE_NAME, schema=SCHEMA, mode="create")


def _get_embedding_model(model_id: Optional[str] = None) -> Optional[Tuple[Any, Any]]:
    """Carica il modello di embedding MLX se scaricato su disco."""
    global _loaded_embed_model, _loaded_embed_processor, _loaded_embed_model_id
    target_id = model_id or DEFAULT_EMBEDDING_MODEL

    with _embed_lock:
        if _loaded_embed_model is not None and _loaded_embed_model_id == target_id:
            return _loaded_embed_model, _loaded_embed_processor

        if not is_embedding_downloaded(target_id):
            return None

        try:
            import mlx_embeddings
            logger.info(f"[RAG] Caricamento modello embedding MLX in RAM: {target_id}")
            model, processor = mlx_embeddings.load(target_id)
            _loaded_embed_model = model
            _loaded_embed_processor = processor
            _loaded_embed_model_id = target_id
            return model, processor
        except Exception as exc:
            logger.error(f"[RAG] Errore caricamento embedding {target_id}: {exc}")
            return None


def unload_embedding_model() -> None:
    """Scarica il modello di embedding dalla RAM e azzera la cache Metal e LanceDB per liberare Unified Memory."""
    global _loaded_embed_model, _loaded_embed_processor, _loaded_embed_model_id, _db_conn
    with _embed_lock:
        if _loaded_embed_model is not None:
            del _loaded_embed_model
            del _loaded_embed_processor
            _loaded_embed_model = None
            _loaded_embed_processor = None
            _loaded_embed_model_id = None
    with _table_lock:
        _db_conn = None
    clear_metal_cache()
    logger.info("[RAG] Modello embedding scaricato e cache LanceDB azzerata.")



def compute_embeddings(
    texts: List[str],
    is_query: bool = False,
    model_id: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Calcola i vettori normalizzati L2 a 1024 dimensioni per una lista di testi usando mlx-embeddings.
    - Se is_query=True, aggiunge l'istruzione formattata per Qwen3-Embedding.
    - Se is_query=False, codifica i passaggi direttamente come testo pulito.
    """
    if not texts:
        return None

    model_proc = _get_embedding_model(model_id)
    if model_proc is None:
        return None

    model, processor = model_proc

    # Formattazione istruzione per Qwen3-Embedding
    if is_query:
        formatted_texts = [f"{QUERY_INSTRUCTION_PREFIX}{t.strip()}" for t in texts]
    else:
        formatted_texts = [t.strip() for t in texts]

    try:
        import mlx.core as mx
        import mlx_embeddings

        embed_arr = mlx_embeddings.generate(
            model=model,
            processor=processor,
            texts=formatted_texts,
            max_length=512,
            padding=True,
            truncation=True,
        )
        # Se mlx_embeddings restituisce un BaseModelOutput (come per Qwen3-Embedding), estrai il tensore
        if hasattr(embed_arr, "text_embeds") and embed_arr.text_embeds is not None:
            embed_arr = embed_arr.text_embeds
        elif hasattr(embed_arr, "last_hidden_state") and embed_arr.last_hidden_state is not None:
            embed_arr = embed_arr.last_hidden_state
        elif hasattr(embed_arr, "embeddings") and embed_arr.embeddings is not None:
            embed_arr = embed_arr.embeddings
        elif isinstance(embed_arr, (list, tuple)) and len(embed_arr) > 0:
            embed_arr = embed_arr[0]

        # Normalizzazione L2 e conversione esplicita float32 per compatibilità buffer NumPy (PEP 3118)
        embed_arr_f32 = embed_arr.astype(mx.float32)
        norm = mx.linalg.norm(embed_arr_f32, axis=-1, keepdims=True)
        embed_norm = embed_arr_f32 / (norm + 1e-9)
        arr_np = np.array(embed_norm, dtype=np.float32)

        # Rilascio cache Metal
        clear_metal_cache()
        return arr_np

    except Exception as exc:
        logger.error(f"[RAG] Errore calcolo embeddings: {exc}")
        clear_metal_cache()
        return None


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splitta un testo accademico in chunk coerenti con overlap."""
    clean = re.sub(r"\s+", " ", text.strip())
    words = clean.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── Indicizzazione Lezioni & Materiali ─────────────────────────────────────────

def index_lecture(
    subject_id: int,
    lecture_id: int,
    title: str,
    transcript: str,
    notes: Optional[str] = None,
) -> int:
    """
    Indicizza la trascrizione e le note elaborate di una lezione in LanceDB con Qwen3-Embedding (1024 dim).
    Elimina preventivamente chunk obsoleti della stessa lezione per evitare duplicati.
    Rilascia sistematicamente la cache Metal al termine.
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
        tbl = get_lancedb_table()
        # Rimuovi chunk precedenti di questa specifica lezione
        try:
            tbl.delete(f"subject_id = {subject_id} AND lecture_id = {lecture_id}")
        except Exception:
            pass

        vectors = compute_embeddings(chunks, is_query=False)
        has_vectors = vectors is not None and len(vectors) == len(chunks)

        records = []
        now = time.time()
        for idx, ch in enumerate(chunks):
            v = vectors[idx].tolist() if has_vectors else [0.0] * EMBEDDING_DIM
            records.append({
                "id": f"lec-{lecture_id}-{idx}-{uuid.uuid4().hex[:6]}",
                "vector": v,
                "subject_id": int(subject_id),
                "lecture_id": int(lecture_id),
                "source_type": "lecture",
                "source_id": f"lecture_{lecture_id}",
                "title": title or f"Lezione {lecture_id}",
                "text": ch,
                "created_at": now,
            })

        tbl.add(records)
        clear_metal_cache()
        logger.info(f"[RAG] Indicizzati {len(records)} chunk (1024 dim) per la lezione {lecture_id} (Materia {subject_id}).")
        return len(records)
    except Exception as exc:
        logger.error(f"[RAG] Errore indicizzazione lezione {lecture_id}: {exc}")
        clear_metal_cache()
        return 0


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
        tbl = get_lancedb_table()
        safe_src = material_id.replace("'", "")
        try:
            tbl.delete(f"subject_id = {subject_id} AND source_id = '{safe_src}'")
        except Exception:
            pass

        vectors = compute_embeddings(chunks, is_query=False)
        has_vectors = vectors is not None and len(vectors) == len(chunks)

        records = []
        now = time.time()
        for idx, ch in enumerate(chunks):
            v = vectors[idx].tolist() if has_vectors else [0.0] * EMBEDDING_DIM
            records.append({
                "id": f"mat-{subject_id}-{idx}-{uuid.uuid4().hex[:6]}",
                "vector": v,
                "subject_id": int(subject_id),
                "lecture_id": 0,
                "source_type": source_type,
                "source_id": material_id,
                "title": title or material_id,
                "text": ch,
                "created_at": now,
            })

        tbl.add(records)
        clear_metal_cache()
        logger.info(f"[RAG] Indicizzati {len(records)} chunk per il materiale '{title}'.")
        return len(records)
    except Exception as exc:
        logger.error(f"[RAG] Errore indicizzazione materiale {material_id}: {exc}")
        clear_metal_cache()
        return 0


def delete_material_vectors(subject_id: int, material_id: str) -> None:
    """Elimina i vettori associati a un materiale da LanceDB."""
    try:
        tbl = get_lancedb_table()
        safe_src = material_id.replace("'", "")
        tbl.delete(f"subject_id = {subject_id} AND source_id = '{safe_src}'")
        clear_metal_cache()
        logger.info(f"[RAG] Eliminati vettori LanceDB per materiale '{material_id}' (Materia {subject_id}).")
    except Exception as exc:
        logger.warning(f"[RAG] Errore eliminazione vettori materiale {material_id}: {exc}")


def delete_lecture_vectors(subject_id: int, lecture_id: int) -> None:
    """Elimina i vettori associati a una specifica lezione da LanceDB."""
    try:
        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {subject_id} AND lecture_id = {lecture_id}")
        clear_metal_cache()
        logger.info(f"[RAG] Eliminati vettori LanceDB per lezione {lecture_id} (Materia {subject_id}).")
    except Exception as exc:
        logger.warning(f"[RAG] Errore eliminazione vettori lezione {lecture_id}: {exc}")


def delete_subject_vectors(subject_id: int) -> None:
    """Elimina TUTTI i vettori di una materia (lezioni e materiali) da LanceDB."""
    try:
        tbl = get_lancedb_table()
        tbl.delete(f"subject_id = {subject_id}")
        clear_metal_cache()
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


# ── Query & Ricerca Ibrida (Vector + Keyword/Acronimi) ─────────────────────────


def query_context(
    subject_id: int,
    query: str,
    top_k: int = 4,
    priority_previous_lecture: bool = False,
) -> str:
    """
    Recupera i chunk più pertinenti filtrando RIGIDAMENTE per subject_id tramite Hybrid Search:
      1. Ricerca Vettoriale semantica (Qwen3-Embedding con query instruction prefix standard).
      2. Ricerca Testuale / Keyword per sigle, acronimi e formule esatte.
      3. RRF / Fusione pesata per bilanciare semantica ed esattezza lessicale.
      4. Fallback pulito se l'embedding non è ancora scaricato.
    """
    clean_q = query.strip()
    if not clean_q:
        return ""

    try:
        tbl = get_lancedb_table()

        # Estrazione termini e acronimi per il boost lessicale
        terms = set(w.lower() for w in re.findall(r"\b[A-Za-z0-9_]{2,}\b", clean_q) if len(w) >= 2)
        acronyms = set(re.findall(r"\b[A-Z0-9]{2,}\b", query))

        # Calcolo embedding con query instruction
        query_vector = compute_embeddings([clean_q], is_query=True)

        scored_records: Dict[str, Dict[str, Any]] = {}

        if query_vector is not None and len(query_vector) > 0:
            # 1. Ricerca Vettoriale LanceDB (Top candidates)
            v = query_vector[0].tolist()
            try:
                vec_matches = (
                    tbl.search(v)
                    .where(f"subject_id = {subject_id}")
                    .limit(top_k * 4)
                    .to_list()
                )
                for rank, r in enumerate(vec_matches):
                    rid = r.get("id") or str(uuid.uuid4())
                    # RRF score da ranking vettoriale
                    vec_score = 1.0 / (60.0 + rank + 1)
                    r["hybrid_score"] = vec_score
                    scored_records[rid] = r
            except Exception as e:
                logger.warning(f"[RAG] Errore vector search LanceDB: {e}")

        # 2. Ricerca Lessicale / Keyword matching per acronimi e termini esatti
        try:
            df = tbl.to_pandas()
            if not df.empty and "subject_id" in df.columns:
                sub_df = df[df["subject_id"] == subject_id]
                for _, row in sub_df.iterrows():
                    rid = str(row["id"])
                    txt = str(row["text"]).lower()
                    title = str(row.get("title", "")).lower()

                    match_count = 0
                    for t in terms:
                        if t in txt:
                            match_count += 1
                        if t in title:
                            match_count += 2

                    # Extra boost per acronimi esatti
                    for acr in acronyms:
                        if acr in str(row["text"]) or acr in str(row.get("title", "")):
                            match_count += 3

                    if match_count > 0:
                        lex_score = match_count * 0.05
                        if rid in scored_records:
                            scored_records[rid]["hybrid_score"] += lex_score
                        else:
                            item = row.to_dict()
                            item["hybrid_score"] = lex_score
                            scored_records[rid] = item
        except Exception as e:
            logger.debug(f"[RAG] Keyword scan error: {e}")

        if not scored_records:
            clear_metal_cache()
            return ""

        all_results = list(scored_records.values())

        # Prioritizza lezioni precedenti o note se richiesto
        if priority_previous_lecture:
            for r in all_results:
                if r.get("source_type") == "lecture":
                    r["hybrid_score"] = r.get("hybrid_score", 0) + 0.1

        all_results.sort(key=lambda r: r.get("hybrid_score", 0), reverse=True)
        selected = all_results[:top_k]

        formatted_chunks = []
        for r in selected:
            title = r.get("title", "Appunti del corso")
            text = r.get("text", "")
            formatted_chunks.append(f"[{title}]:\n{text}")

        clear_metal_cache()
        return "\n\n".join(formatted_chunks)

    except Exception as exc:
        logger.warning(f"[RAG] Ricerca contesto non riuscita: {exc}")
        clear_metal_cache()
        return ""
