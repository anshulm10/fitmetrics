from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class QueryRoute(StrEnum):
    FACTUAL_RETRIEVAL = "factual_retrieval"
    CROSS_MODAL = "cross_modal"
    ANALYTICAL = "analytical"
    PERSONALIZED_FOLLOWUP = "personalized_followup"


@dataclass(frozen=True)
class RoutedQuery:
    query: str
    route: QueryRoute
    rationale: str


class QueryRouter:
    """Lightweight rule-based router for reproducible assignment evaluation."""

    CROSS_MODAL_TERMS = {
        "image",
        "photo",
        "picture",
        "visual",
        "form",
        "show",
        "looks like",
        "what is this",
        "how does this look",
        "similar",
        "snapshot",
    }
    ANALYTICAL_TERMS = {
        "analyze",
        "compare",
        "trend",
        "progress",
        "progression",
        "strongest",
        "weakness",
        "volume",
        "pattern",
    }
    PERSONAL_TERMS = {
        "my",
        "me",
        "personal",
        "baseline",
        "pr",
        "injury",
        "pain",
        "knee",
        "recovery",
        "limited",
    }

    @staticmethod
    def _has_phrase_or_token(query: str, terms: set[str]) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        for term in terms:
            if " " in term:
                if term in query.lower():
                    return True
            elif term in tokens:
                return True
        return False

    def route(self, query: str, image_path: str | None = None) -> RoutedQuery:
        q = query.lower()
        if image_path or self._has_phrase_or_token(q, self.CROSS_MODAL_TERMS):
            return RoutedQuery(query, QueryRoute.CROSS_MODAL, "Image or visual-comparison intent detected.")
        if self._has_phrase_or_token(q, self.ANALYTICAL_TERMS):
            return RoutedQuery(query, QueryRoute.ANALYTICAL, "Progression/comparison/synthesis intent detected.")
        if self._has_phrase_or_token(q, self.PERSONAL_TERMS):
            return RoutedQuery(query, QueryRoute.PERSONALIZED_FOLLOWUP, "Personal strength or injury context requested.")
        return RoutedQuery(query, QueryRoute.FACTUAL_RETRIEVAL, "Default factual exercise lookup.")

