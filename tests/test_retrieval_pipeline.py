from __future__ import annotations

from fit_support.config.settings import AppSettings
from fit_support.retrieval.retrieve import RetrievalService


class FakeEmbedder:
    @staticmethod
    def embed_text(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeStore:
    def query_text(self, _query_embedding, modality, k):
        return {
            "ids": [[f"{modality.value}::1"]],
            "documents": [[f"{modality.value} doc"]],
            "metadatas": [[{"source_id": "src_1", "pain_flag": modality.value == "injury"}]],
            "distances": [[0.2]],
        }


def test_retrieval_returns_ranked_results() -> None:
    settings = AppSettings(project_root=".")
    service = RetrievalService(settings=settings, vector_store=FakeStore(), embedder=FakeEmbedder())
    results = service.retrieve("Need safer lower-body workout", top_k=3)

    assert len(results) == 3
    assert all("chunk_id" in item for item in results)
    assert all("score" in item for item in results)

