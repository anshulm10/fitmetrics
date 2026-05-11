from __future__ import annotations

import difflib
import re
from datetime import date, datetime
from typing import Any

import pandas as pd


def normalize_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def null_unknown(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.upper() == "UNKNOWN":
        return None
    return value


def normalize_exercise_name(name: str | None, library_names: list[str]) -> tuple[str | None, str | None]:
    """Return (canonical_name, note_if_fuzzy)."""
    if name is None:
        return None, None
    raw = normalize_whitespace(str(name))
    if not raw:
        return None, None
    lib_set = set(library_names)
    if raw in lib_set:
        return raw, None
    lower_map = {n.lower(): n for n in library_names}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()], "case-normalized"
    match = difflib.get_close_matches(raw, library_names, n=1, cutoff=0.72)
    if match:
        return match[0], f"fuzzy:{raw}->{match[0]}"
    return raw, "unresolved"


def normalize_date_value(value: Any) -> date | None:
    v = null_unknown(value)
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def coerce_float(val: Any) -> float | None:
    v = null_unknown(val)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def coerce_int(val: Any) -> int | None:
    v = null_unknown(val)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
