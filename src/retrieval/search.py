from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from embeddings.image_embedder import ImageEmbedder
from embeddings.text_embedder import TextEmbedder


def _client(chroma_path: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(chroma_path))


def _format_rows(raw: dict[str, Any], k: int) -> list[dict[str, Any]]:
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


def search_exercise_by_text(
    query: str,
    top_k: int = 5,
    chroma_path: str | Path = "data/chroma",
    embedder: TextEmbedder | None = None,
) -> list[dict[str, Any]]:
    chroma_dir = Path(chroma_path)
    client = _client(chroma_dir)
    col = client.get_or_create_collection("fitness_text")
    emb = embedder or TextEmbedder()
    q_vec = emb.embed_query(query)
    raw = col.query(query_embeddings=[q_vec], n_results=top_k)
    return _format_rows(raw, top_k)


def search_similar_exercise_image(
    image_path: str | Path,
    top_k: int = 5,
    chroma_path: str | Path = "data/chroma",
    embedder: ImageEmbedder | None = None,
) -> list[dict[str, Any]]:
    path = Path(image_path)
    client = _client(Path(chroma_path))
    col = client.get_or_create_collection("fitness_images")
    emb = embedder or ImageEmbedder()
    q_vec = emb.embed_single_image(path)
    raw = col.query(query_embeddings=[q_vec], n_results=top_k)
    return _format_rows(raw, top_k)

