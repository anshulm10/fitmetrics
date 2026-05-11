from __future__ import annotations

from collections import defaultdict
from typing import Any

import chromadb

from fit_support.config.settings import AppSettings
from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.embeddings.embedder import EmbeddingService


class VectorStore:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.resolved(settings.chroma_dir)))

    def _collection_name(self, modality: ModalityType) -> str:
        return f"{self._settings.chroma_collection_prefix}_{modality.value}"

    def _collection(self, modality: ModalityType):
        return self._client.get_or_create_collection(self._collection_name(modality))

    def upsert_chunks(self, chunks: list[ContextChunk], embedder: EmbeddingService) -> None:
        grouped: dict[ModalityType, list[ContextChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.modality].append(chunk)

        for modality, modality_chunks in grouped.items():
            collection = self._collection(modality)
            ids = [chunk.chunk_id for chunk in modality_chunks]
            documents = [chunk.content for chunk in modality_chunks]
            metadatas = [{**chunk.metadata, "source_id": chunk.source_id} for chunk in modality_chunks]
            embeddings = []
            for chunk in modality_chunks:
                if chunk.modality == ModalityType.IMAGE:
                    embeddings.append(embedder.embed_image(path=chunk.metadata["path"]))
                else:
                    embeddings.append(embedder.embed_text(chunk.content))
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query_text(self, query_embedding: list[float], modality: ModalityType, k: int) -> dict[str, Any]:
        collection = self._collection(modality)
        return collection.query(query_embeddings=[query_embedding], n_results=k)

