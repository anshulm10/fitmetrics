from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from pydantic import ValidationError

from ingestion.models import ExerciseMetadata, InjuryRecord, LiftRecord, WorkoutRecord


def _validation_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]


def validate_exercise_metadata(row: dict[str, Any]) -> tuple[ExerciseMetadata | None, list[str]]:
    try:
        return ExerciseMetadata.model_validate(row), []
    except ValidationError as e:
        return None, _validation_errors(e)


def validate_lift_record(row: dict[str, Any]) -> tuple[LiftRecord | None, list[str]]:
    try:
        return LiftRecord.model_validate(row), []
    except ValidationError as e:
        return None, _validation_errors(e)


def validate_workout_record(row: dict[str, Any]) -> tuple[WorkoutRecord | None, list[str]]:
    try:
        return WorkoutRecord.model_validate(row), []
    except ValidationError as e:
        return None, _validation_errors(e)


def validate_injury_record(row: dict[str, Any]) -> tuple[InjuryRecord | None, list[str]]:
    try:
        return InjuryRecord.model_validate(row), []
    except ValidationError as e:
        return None, _validation_errors(e)


def find_duplicate_row_indices(rows: list[dict[str, Any]], key_fields: Iterable[str]) -> dict[int, str]:
    """Map row index -> message for duplicates by key tuple."""
    seen: dict[tuple[Any, ...], int] = {}
    dupes: dict[int, str] = {}
    keys = list(key_fields)
    for i, row in enumerate(rows):
        try:
            key = tuple(row.get(k) for k in keys)
        except Exception:
            continue
        if any(v is not None and str(v).strip() != "" for v in key):
            if key in seen:
                dupes[i] = f"duplicate row for key {dict(zip(keys, key))}"
            else:
                seen[key] = i
    return dupes


def validate_non_negative_number(name: str, value: Any) -> list[str]:
    errs: list[str] = []
    if value is None or value == "":
        return errs
    try:
        n = float(value)
        if n < 0:
            errs.append(f"{name} cannot be negative (got {n})")
    except (TypeError, ValueError):
        pass
    return errs


def validate_date_field(field: str, value: Any) -> list[str]:
    if value is None or str(value).strip() == "":
        return [f"{field} is missing"]
    s = str(value).strip().upper()
    if s in {"BASELINE", "PR", "UNKNOWN"}:
        return [f"{field} looks like a marker not a date: {value!r}"]
    try:
        WorkoutRecord.model_validate(
            {
                "date": value,
                "exercise_name": "placeholder",
                "set_number": 1,
            }
        )
    except ValidationError as e:
        return _validation_errors(e)
    return []


def validate_exercise_in_library(exercise_name: str, library: set[str]) -> list[str]:
    if not exercise_name or not str(exercise_name).strip():
        return ["exercise_name is empty"]
    if exercise_name.strip() not in library:
        return [f"exercise_name not in library: {exercise_name!r}"]
    return []
