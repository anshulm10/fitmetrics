"""
ChromaDB retrieval helpers for the fitness agent.

Both search functions accept optional structured metadata filters that are
pushed down to ChromaDB's `where` clause before vector ranking, so only
documents that satisfy the filter are candidates for the top-k result.
Applied filters are logged to stdout for traceability.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, List, Optional

import chromadb

from src.config import cfg
from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


# ── ChromaDB client cache ─────────────────────────────────────────────────────
# `chromadb.PersistentClient` has a global `SharedSystemClient._identifier_to_system`
# registry that is NOT thread-safe.  When multiple LangGraph nodes
# (text_retrieval + progression_analysis + injury_lookup) run in parallel and
# all try to construct a `PersistentClient(path=X)` for the same path, the
# registry races and the second caller hits either
#   `Could not connect to tenant default_tenant`  (init phase)
# or
#   `KeyError: '<path>'`                          (cleanup of failed init).
# We solve this by maintaining our own thread-safe path → client cache and
# serialising the actual `PersistentClient(...)` call with a lock.
_client_lock = threading.Lock()
_client_cache: dict[str, chromadb.PersistentClient] = {}


def _client(chroma_path: Path) -> chromadb.PersistentClient:
    """Return a process-wide cached ``chromadb.PersistentClient`` for *chroma_path*."""
    key = str(chroma_path)
    cached = _client_cache.get(key)
    if cached is not None:
        return cached
    with _client_lock:
        cached = _client_cache.get(key)
        if cached is None:
            cached = chromadb.PersistentClient(path=key)
            _client_cache[key] = cached
        return cached


def _format_rows(raw: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """Flatten a raw ChromaDB query response into a list of record dicts."""
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]
    rows: list[dict[str, Any]] = []
    for i, (rid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        if i >= k:
            break
        metadata = dict(meta or {})
        if "exercise_name" not in metadata and metadata.get("exercise_label"):
            metadata["exercise_name"] = metadata["exercise_label"]
        rows.append(
            {
                "id": rid,
                "score": 1.0 - float(dist),
                "document": doc,
                "metadata": metadata,
            }
        )
    return rows


def _build_where_clause(
    muscle_group: Optional[str],
    equipment: Optional[str],
    exclude_injury_flags: Optional[List[str]],
) -> Optional[dict]:
    """Compose a ChromaDB `where` dict from the supplied filter arguments.

    Returns None when no filters are requested (avoids passing an empty
    `where` dict, which ChromaDB rejects).
    """
    clauses: list[dict] = []

    if muscle_group:
        clauses.append({"muscle_groups": {"$contains": muscle_group}})
    if equipment:
        clauses.append({"equipment": {"$eq": equipment}})
    if exclude_injury_flags:
        for flag in exclude_injury_flags:
            clauses.append({"injury_tags": {"$not_contains": flag}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def search_exercise_by_text(
    query: str,
    top_k: int = cfg.retrieval.top_k,
    chroma_path: str | Path = cfg.chroma.persist_directory,
    embedder: TextEmbedder | None = None,
    muscle_group: Optional[str] = None,
    equipment: Optional[str] = None,
    exclude_injury_flags: Optional[List[str]] = None,
) -> list[dict[str, Any]]:
    """Retrieve exercise documents from the *fitness_text* ChromaDB collection.

    Parameters
    ----------
    query : str
        Natural-language search query.
    top_k : int
        Maximum number of results to return.
    chroma_path : str | Path
        Filesystem path to the persistent ChromaDB directory.
    embedder : TextEmbedder | None
        Pre-initialised embedder; a new instance is created if None.
    muscle_group : str | None
        If set, restricts results to documents whose ``muscle_groups``
        metadata field contains this value (case-sensitive substring match).
    equipment : str | None
        If set, restricts results to an exact ``equipment`` metadata match.
    exclude_injury_flags : List[str] | None
        If set, excludes documents whose ``injury_tags`` field contains any
        of the listed flag strings.

    Returns
    -------
    list[dict]
        Each dict has keys: id, score, document, metadata.
    """
    applied: list[str] = []
    if muscle_group:
        applied.append(f"muscle_group={muscle_group!r}")
    if equipment:
        applied.append(f"equipment={equipment!r}")
    if exclude_injury_flags:
        applied.append(f"exclude_injury_flags={exclude_injury_flags!r}")
    if applied:
        logger.info("[search_exercise_by_text] filters applied: %s", ", ".join(applied))

    chroma_dir = Path(chroma_path)
    client = _client(chroma_dir)
    col = client.get_or_create_collection(cfg.chroma.text_collection)
    emb = embedder or TextEmbedder()
    q_vec = emb.embed_query(query)

    where = _build_where_clause(muscle_group, equipment, exclude_injury_flags)
    query_kwargs: dict[str, Any] = {"query_embeddings": [q_vec], "n_results": top_k}
    if where is not None:
        query_kwargs["where"] = where

    raw = col.query(**query_kwargs)
    return _format_rows(raw, top_k)


def search_lift_records_by_text(
    query: str,
    top_k: int = cfg.retrieval.top_k,
    chroma_path: str | Path = cfg.chroma.persist_directory,
    embedder: TextEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve only the user's personal lift-history records.

    Pushes ``where={"record_type": "lift_record"}`` down to ChromaDB so the
    semantic search runs over the personal strength corpus *only*, never the
    generic exercise library.  This is the canonical entry point for the
    ``progression_analysis`` graph node.
    """
    logger.info("[search_lift_records_by_text] filter: record_type=lift_record")
    chroma_dir = Path(chroma_path)
    client = _client(chroma_dir)
    col = client.get_or_create_collection(cfg.chroma.text_collection)
    emb = embedder or TextEmbedder()
    q_vec = emb.embed_query(query)
    raw = col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        where={"record_type": "lift_record"},
    )
    return _format_rows(raw, top_k)


def get_images_by_exercise_label(
    exercise_label: str,
    chroma_path: str | Path = cfg.chroma.persist_directory,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return image records whose ``exercise_label`` metadata matches exactly.

    Uses ``collection.get(where=...)`` (no embedding needed) so it's fast and
    deterministic — perfect for "give me the demo frames for the exercise we
    just recommended" lookups from the graph's text_retrieval node.
    """
    chroma_dir = Path(chroma_path)
    client = _client(chroma_dir)
    col = client.get_or_create_collection(cfg.chroma.image_collection)
    raw = col.get(
        where={"exercise_label": {"$eq": exercise_label}},
        limit=limit,
        include=["metadatas", "documents"],
    )
    ids = raw.get("ids", []) or []
    docs = raw.get("documents", []) or []
    metas = raw.get("metadatas", []) or []
    rows: list[dict[str, Any]] = []
    for rid, doc, meta in zip(ids, docs, metas):
        rows.append({"id": rid, "score": 1.0, "document": doc, "metadata": meta})
    return rows


def search_similar_exercise_image(
    image_path: str | Path,
    top_k: int = cfg.retrieval.top_k,
    chroma_path: str | Path = cfg.chroma.persist_directory,
    embedder: ImageEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the best matching exercise image from the *fitness_images* collection.

    Parameters
    ----------
    image_path : str | Path
        Path to the query image file.
    top_k : int
        Ignored for image identification; this function returns only the best match.
    chroma_path : str | Path
        Filesystem path to the persistent ChromaDB directory.
    embedder : ImageEmbedder | None
        Pre-initialised CLIP embedder; a new instance is created if None.

    Returns
    -------
    list[dict]
        Each dict has keys: id, score, document, metadata.
    """
    path = Path(image_path)
    client = _client(Path(chroma_path))
    col = client.get_or_create_collection(cfg.chroma.image_collection)
    emb = embedder or ImageEmbedder()
    q_vec = emb.embed_single_image(path)
    raw = col.query(query_embeddings=[q_vec], n_results=1)
    return _format_rows(raw, 1)
