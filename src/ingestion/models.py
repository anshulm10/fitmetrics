from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _clean_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "UNKNOWN":
        return None
    return s


class ExerciseMetadata(BaseModel):
    """Row from exercise library / metadata sources."""

    model_config = ConfigDict(str_strip_whitespace=True)

    exercise_name: str = Field(..., min_length=1, description="Canonical exercise name")
    movement_pattern: str | None = None
    equipment: str | None = None
    primary_muscles: str | None = None
    difficulty: str | None = None

    @field_validator("exercise_name")
    @classmethod
    def exercise_name_not_blank(cls, v: str) -> str:
        t = v.strip()
        if not t:
            raise ValueError("exercise_name cannot be empty or whitespace-only")
        return t


class LiftRecord(BaseModel):
    """Strength / PR style lift row (e.g. strength.csv)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    exercise_name: str = Field(..., min_length=1)
    best_weight_kg: float | None = None
    best_reps: int | None = None
    notes: str | None = None
    source_file: str | None = Field(default=None, description="Provenance")

    @field_validator("best_weight_kg")
    @classmethod
    def weight_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("best_weight_kg must be non-negative when set")
        return v

    @field_validator("best_reps")
    @classmethod
    def reps_non_negative(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("best_reps must be non-negative when set")
        return v

    @field_validator("exercise_name")
    @classmethod
    def name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("exercise_name cannot be empty")
        return v.strip()


class WorkoutRecord(BaseModel):
    """Dated session set (workout_log.csv)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    date: date
    exercise_name: str = Field(..., min_length=1)
    set_number: int = Field(..., ge=1)
    weight_kg: str | float | None = None
    reps: int | None = None
    notes: str | None = None
    source_file: str | None = None

    @field_validator("exercise_name")
    @classmethod
    def ex_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("exercise_name cannot be empty")
        return v.strip()

    @field_validator("weight_kg", mode="before")
    @classmethod
    def coerce_weight(cls, v: Any) -> str | float | None:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.upper() == "UNKNOWN":
            return None
        if s.lower() == "bodyweight":
            return "Bodyweight"
        try:
            return float(s)
        except ValueError:
            return s

    @field_validator("reps")
    @classmethod
    def reps_ok(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("reps must be non-negative when set")
        return v

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> date:
        if isinstance(v, date) and not isinstance(v, datetime):
            return v
        if isinstance(v, datetime):
            return v.date()
        s = _clean_str(v)
        if s is None:
            raise ValueError("date is required")
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            y, m, d = (int(x) for x in s.split("-"))
            return date(y, m, d)
        raise ValueError(f"unrecognized date format: {s!r}")


class InjuryRecord(BaseModel):
    """Flexible injury note (JSON per file or flattened row)."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    injury_id: str | None = Field(default=None, description="Optional stable id")
    body_region: str | None = None
    description: str | None = None
    severity: str | None = None
    status: str | None = None
    notes: str | None = None
    source_file: str | None = None

    @model_validator(mode="before")
    @classmethod
    def from_flat_dict(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("injury_id") is None and out.get("id") is not None:
            out["injury_id"] = out.get("id")
        if out.get("description") is None and out.get("summary") is not None:
            out["description"] = out.get("summary")
        return out
