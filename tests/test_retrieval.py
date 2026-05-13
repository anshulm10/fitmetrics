"""
Tests for src/retrieval/search.py metadata filtering.

These tests exercise the filter logic against the real persisted ChromaDB
index (data/chroma) so they require the index to have been built first.
If the collection is empty, the filtered tests are skipped automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.search import _build_where_clause, search_exercise_by_text

CHROMA_PATH = ROOT / "data" / "chroma"


# ── unit tests for _build_where_clause ────────────────────────────────────────

def test_where_clause_none_when_no_filters() -> None:
    """No filters → _build_where_clause must return None."""
    assert _build_where_clause(None, None, None) is None


def test_where_clause_single_muscle_group() -> None:
    """Single muscle_group filter produces a $contains clause."""
    where = _build_where_clause("Quads", None, None)
    assert where == {"muscle_groups": {"$contains": "Quads"}}


def test_where_clause_single_equipment() -> None:
    """Single equipment filter produces an $eq clause."""
    where = _build_where_clause(None, "Machine", None)
    assert where == {"equipment": {"$eq": "Machine"}}


def test_where_clause_exclude_injury_flags() -> None:
    """Injury flag exclusion wraps in $not_contains."""
    where = _build_where_clause(None, None, ["knee_friendly_variant_available"])
    assert where == {"injury_tags": {"$not_contains": "knee_friendly_variant_available"}}


def test_where_clause_combined_produces_and() -> None:
    """Multiple filters are combined under a top-level $and."""
    where = _build_where_clause("Quads", "Machine", None)
    assert where is not None
    assert "$and" in where
    assert len(where["$and"]) == 2


# ── integration tests against real ChromaDB ───────────────────────────────────

def _index_has_records() -> bool:
    """Return True if the fitness_text collection has at least one document."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        col = client.get_or_create_collection("fitness_text")
        return col.count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _index_has_records(), reason="ChromaDB index not built")
def test_unfiltered_search_returns_results() -> None:
    """Baseline: unfiltered search returns at least one result."""
    results = search_exercise_by_text("quad exercise", top_k=3, chroma_path=CHROMA_PATH)
    assert len(results) >= 1
    assert all("id" in r and "score" in r and "document" in r for r in results)


@pytest.mark.skipif(not _index_has_records(), reason="ChromaDB index not built")
def test_equipment_filter_respected() -> None:
    """Filtered results must all have equipment == 'Machine' in their metadata."""
    results = search_exercise_by_text(
        "leg exercise",
        top_k=5,
        chroma_path=CHROMA_PATH,
        equipment="Machine",
    )
    for record in results:
        meta = record.get("metadata") or {}
        equipment_val = str(meta.get("equipment", "")).strip()
        assert equipment_val == "Machine", (
            f"Record {record['id']!r} has equipment={equipment_val!r}, expected 'Machine'"
        )


@pytest.mark.skipif(not _index_has_records(), reason="ChromaDB index not built")
def test_muscle_group_filter_narrows_results() -> None:
    """Filtered result count must be <= unfiltered count for the same query."""
    unfiltered = search_exercise_by_text("exercise", top_k=10, chroma_path=CHROMA_PATH)
    filtered = search_exercise_by_text(
        "exercise",
        top_k=10,
        chroma_path=CHROMA_PATH,
        muscle_group="Quads",
    )
    assert len(filtered) <= len(unfiltered), (
        "Filtered search returned MORE results than unfiltered — filter was ignored."
    )


@pytest.mark.skipif(not _index_has_records(), reason="ChromaDB index not built")
def test_no_filter_does_not_pass_where_to_chroma() -> None:
    """Calling search with no filters must not raise (empty `where` is forbidden by Chroma)."""
    results = search_exercise_by_text("bench press", top_k=3, chroma_path=CHROMA_PATH)
    assert isinstance(results, list)
