from __future__ import annotations

from fit_support.domain.personalization import injury_aware_penalty
from fit_support.domain.schemas import ContextChunk


def rerank_with_injury_awareness(results: list[tuple[ContextChunk, float]]) -> list[tuple[ContextChunk, float]]:
    rescored: list[tuple[ContextChunk, float]] = []
    for chunk, score in results:
        rescored.append((chunk, score * injury_aware_penalty(chunk)))
    return sorted(rescored, key=lambda item: item[1], reverse=True)

