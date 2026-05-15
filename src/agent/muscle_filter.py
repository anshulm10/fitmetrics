"""Query-driven filtering of exercise library records by ``muscle_groups`` metadata.

When the user names a coarse muscle bucket (hamstrings, quads, chest, back),
``exercise_metadata`` rows must match at least one of those buckets. Other
record types (lift history, injury notes, image rows) pass through unchanged.
If filtering would remove every exercise row, the original list is restored.
"""
from __future__ import annotations

import re
from typing import Any

# Canonical keyword -> substrings that must appear in muscle_groups (lowercased).
_MUSCLE_SUBSTRINGS: dict[str, tuple[str, ...]] = {
    "hamstrings": ("hamstring",),
    "quads": ("quad",),
    "chest": ("chest",),
    "back": (
        "lat",
        "rhomboid",
        "trap",
        "rear delt",
        "lower back",
        "mid trap",
        "upper back",
    ),
}

# If "back" appears in these injury/spine phrases, do not treat it as "back training".
_BACK_SPINE_PHRASES: tuple[str, ...] = (
    "back pain",
    "back hurts",
    "hurt your back",
    "hurt my back",
    "sore back",
    "back issue",
    "back injury",
    "my back",
    "lower back",
    "spinal",
    "spine",
)


def _spine_or_injury_back_context(query_lower: str) -> bool:
    return any(p in query_lower for p in _BACK_SPINE_PHRASES)


def detect_muscle_keywords(query: str) -> list[str]:
    """Return coarse muscle buckets named in *query* (possibly empty)."""
    q = query.lower()
    found: list[str] = []
    if re.search(r"\bhamstrings?\b", q):
        found.append("hamstrings")
    if re.search(r"\bquads?\b", q):
        found.append("quads")
    if re.search(r"\bchest\b", q):
        found.append("chest")
    if "back squat" in q or "backsquat" in q.replace(" ", ""):
        pass
    elif (
        re.search(r"\bback\b", q) or re.search(r"\bbacks\b", q)
    ) and not _spine_or_injury_back_context(q):
        found.append("back")
    return found


def _metadata_matches_keywords(mg_lower: str, keywords: list[str]) -> bool:
    """True if muscle text matches any requested bucket (OR semantics)."""
    for key in keywords:
        for sub in _MUSCLE_SUBSTRINGS[key]:
            if sub in mg_lower:
                return True
    return False


def filter_exercise_context_records(
    query: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter *records* in place for generation / evaluation."""
    keywords = detect_muscle_keywords(query)
    if not keywords or not records:
        return records

    kept: list[dict[str, Any]] = []
    meta_hits = 0
    for rec in records:
        meta = rec.get("metadata") or {}
        rt = str(meta.get("record_type", ""))
        if rt != "exercise_metadata":
            kept.append(rec)
            continue
        meta_hits += 1
        mg = str(meta.get("muscle_groups", "")).lower()
        if _metadata_matches_keywords(mg, keywords):
            kept.append(rec)

    if meta_hits == 0:
        return records
    meta_kept = sum(
        1 for r in kept if str((r.get("metadata") or {}).get("record_type", "")) == "exercise_metadata"
    )
    if meta_kept == 0:
        return records
    return kept
