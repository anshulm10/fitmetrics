from __future__ import annotations

from typing import Any

from fit_support.config import AppSettings
from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.embeddings.embedder import EmbeddingService
from fit_support.retrieval.rerank import rerank_with_injury_awareness
from fit_support.retrieval.vector_store import VectorStore


class RetrievalService:
    def __init__(self, settings: AppSettings, vector_store: VectorStore, embedder: EmbeddingService) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self._settings.top_k
        query_embedding = self._embedder.embed_text(query)

        candidates: list[tuple[ContextChunk, float]] = []
        for modality in [ModalityType.WORKOUT, ModalityType.LIFT, ModalityType.INJURY, ModalityType.IMAGE]:
            raw = self._vector_store.query_text(query_embedding, modality=modality, k=k)
            ids = raw.get("ids", [[]])[0]
            docs = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
                chunk = ContextChunk(
                    chunk_id=chunk_id,
                    source_id=str(meta.get("source_id", "unknown")),
                    modality=modality,
                    content=doc,
                    metadata=meta,
                )
                candidates.append((chunk, 1.0 - float(dist)))

        ranked = rerank_with_injury_awareness(candidates)[:k]
        return [
            {
                "chunk_id": chunk.chunk_id,
                "modality": chunk.modality.value,
                "score": score,
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk, score in ranked
        ]

