from __future__ import annotations

from fit_support.domain.schemas import ContextChunk


def injury_aware_penalty(chunk: ContextChunk) -> float:
    """Return a multiplicative penalty for risky chunks."""
    pain_flag = bool(chunk.metadata.get("pain_flag", False))
    return 0.6 if pain_flag else 1.0

