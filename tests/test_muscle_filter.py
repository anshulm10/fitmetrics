"""Tests for query-driven muscle metadata filtering."""
from __future__ import annotations

from src.agent.muscle_filter import (
    detect_muscle_keywords,
    filter_exercise_context_records,
)


def _meta_row(muscles: str) -> dict:
    return {
        "id": "x",
        "score": 1.0,
        "document": "doc",
        "metadata": {"record_type": "exercise_metadata", "muscle_groups": muscles},
    }


def test_detect_hamstrings_quad_chest_back() -> None:
    assert "hamstrings" in detect_muscle_keywords("best hamstring curl weight")
    assert "quads" in detect_muscle_keywords("quad isolation today")
    assert "chest" in detect_muscle_keywords("chest press alternative")
    assert "back" in detect_muscle_keywords("lat focused back day")


def test_detect_back_squat_not_back_muscle() -> None:
    assert "back" not in detect_muscle_keywords("what is my back squat pr")


def test_detect_back_injury_phrase_ignored() -> None:
    assert "back" not in detect_muscle_keywords("my back hurts when I squat")


def test_filter_keeps_non_metadata() -> None:
    lift = {
        "id": "lift",
        "document": "lift",
        "metadata": {"record_type": "lift_record", "exercise_name": "Hack Squat"},
    }
    meta_chest = _meta_row("Chest / Triceps")
    meta_row_only = _meta_row("Lats / Biceps")
    out = filter_exercise_context_records("chest press ideas", [lift, meta_chest, meta_row_only])
    assert lift in out
    assert meta_chest in out
    assert meta_row_only not in out


def test_filter_no_keywords_returns_input() -> None:
    rows = [_meta_row("Quads / Glutes")]
    assert filter_exercise_context_records("hack squat weight", rows) == rows


def test_filter_empty_meta_returns_input() -> None:
    assert filter_exercise_context_records("chest work", []) == []
