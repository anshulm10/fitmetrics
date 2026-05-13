"""
ChromaDB retrieval helpers for the fitness agent.

Both search functions accept optional structured metadata filters that are
pushed down to ChromaDB's `where` clause before vector ranking, so only
documents that satisfy the filter are candidates for the top-k result.
Applied filters are logged to stdout for traceability.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import chromadb

from config import cfg
from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder

logger = logging.getLogger(__name__)


def _client(chroma_path: Path) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client for *chroma_path*."""
    return chromadb.PersistentClient(path=str(chroma_path))


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
        rows.append(
            {
                "id": rid,
                "score": 1.0 - float(dist),
                "document": doc,
                "metadata": meta,
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


def search_similar_exercise_image(
    image_path: str | Path,
    top_k: int = cfg.retrieval.top_k,
    chroma_path: str | Path = cfg.chroma.persist_directory,
    embedder: ImageEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve similar exercise images from the *fitness_images* ChromaDB collection.

    Parameters
    ----------
    image_path : str | Path
        Path to the query image file.
    top_k : int
        Maximum number of results to return.
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
    raw = col.query(query_embeddings=[q_vec], n_results=top_k)
    return _format_rows(raw, top_k)
