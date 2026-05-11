"""
Split mixed lift logs into strength baselines/PRs vs session workout logs.

Reads legacy `data/raw/lifts/lifts_log.csv` (and similar-shaped CSVs in raw lifts),
writes `data/raw/lifts/strength.csv` and `data/raw/workouts/workout_log.csv`,
validates exercise names against `data/raw/metadata/exercise_library.csv`.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

STRENGTH_MARKERS_FIRST_COL = frozenset({"BASELINE", "PR"})
STRENGTH_NOTE_SUBSTRINGS = (
    "personal baseline",
    "bodyweight baseline",
    "1rm baseline",
    "pr baseline",
)
STRENGTH_NOTE_TOKEN_PR = re.compile(r"(?<![A-Za-z])PR(?![A-Za-z])", re.IGNORECASE)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_exercise_library(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "exercise_name" not in df.columns:
        raise ValueError(f"exercise_library missing exercise_name column: {path}")
    return [str(x).strip() for x in df["exercise_name"].tolist() if str(x).strip()]


def _canon_exercise(name: str, library: list[str]) -> tuple[str, str | None]:
    raw = str(name).strip()
    if not raw:
        return raw, "empty exercise_name"
    lib_set = set(library)
    if raw in lib_set:
        return raw, None
    lower_map = {n.lower(): n for n in library}
    if raw.lower() in lower_map:
        return lower_map[raw.lower()], f"case-normalized: {raw!r}"
    match = difflib.get_close_matches(raw, library, n=1, cutoff=0.72)
    if match:
        return match[0], f"fuzzy-matched: {raw!r} -> {match[0]!r}"
    return raw, f"NOT IN LIBRARY: {raw!r}"


def _clean_cell(val) -> str | float | int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s == "" or s.upper() == "UNKNOWN":
        return None
    return val


def _is_strength_row(row: pd.Series, first_col: str) -> bool:
    marker = str(row.get(first_col, "") or "").strip().upper()
    if marker in STRENGTH_MARKERS_FIRST_COL:
        return True
    notes = str(row.get("notes", "") or "").lower()
    for sub in STRENGTH_NOTE_SUBSTRINGS:
        if sub in notes:
            return True
    full_notes = str(row.get("notes", "") or "")
    if STRENGTH_NOTE_TOKEN_PR.search(full_notes):
        return True
    return False


def _looks_like_date(val: str) -> bool:
    v = val.strip()
    if not v or v.upper() == "BASELINE" or v.upper() == "PR":
        return False
    # ISO yyyy-mm-dd
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return True
    # d/m/y or m/d/y short
    if re.match(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$", v):
        return True
    return False


def _normalize_weight_for_strength(weight, reps, notes: str, exercise: str) -> tuple[str | None, str | None, str]:
    """Return (best_weight_kg str or None, best_reps str or None, augmented notes)."""
    notes = str(notes or "").strip()
    w = _clean_cell(weight)
    r = _clean_cell(reps)
    w_s = "" if w is None else str(w).strip()
    r_s = "" if r is None else str(r).strip()

    n_low = notes.lower()
    bodyweight_hint = "bodyweight" in n_low or exercise.lower() in {"pull ups", "pull-ups", "chin ups"}

    if w_s == "" or w_s.upper() == "UNKNOWN":
        w_out: str | None = None
    elif w_s in {"0", "0.0", "0.00"} and bodyweight_hint:
        w_out = None
        if "bodyweight" in n_low and "Bodyweight" not in notes:
            notes = (notes + " — Bodyweight").strip() if notes else "Bodyweight"
    else:
        w_out = w_s

    if r_s == "" or r_s.upper() == "UNKNOWN":
        r_out = None
    else:
        r_out = r_s

    return w_out, r_out, notes


def refactor(
    project_root: Path,
    *,
    dry_run: bool = False,
) -> None:
    raw_lifts = project_root / "data" / "raw" / "lifts"
    raw_workouts = project_root / "data" / "raw" / "workouts"
    raw_metadata = project_root / "data" / "raw" / "metadata"
    archive_dir = project_root / "data" / "processed" / "migrations"
    library_path = raw_metadata / "exercise_library.csv"
    lifts_log = raw_lifts / "lifts_log.csv"
    strength_out = raw_lifts / "strength.csv"
    workout_out = raw_workouts / "workout_log.csv"

    if not library_path.exists():
        raise FileNotFoundError(library_path)
    library = _load_exercise_library(library_path)
    warnings: list[str] = []

    if not lifts_log.exists():
        print(f"No legacy file at {lifts_log}; nothing to split.")
        return

    df = pd.read_csv(lifts_log)
    # Resolve first column (legacy typo `isdate`)
    cols = list(df.columns)
    if not cols:
        print("Empty CSV; abort.")
        return
    first_col = cols[0]

    strength_rows: list[dict] = []
    workout_rows: list[dict] = []

    for idx, row in df.iterrows():
        ex_raw = row.get("exercise_name", "")
        ex_canon, wmsg = _canon_exercise(str(ex_raw), library)
        if wmsg:
            warnings.append(f"row {idx}: {wmsg}")

        if _is_strength_row(row, first_col):
            w_kg, reps, notes = _normalize_weight_for_strength(
                row.get("weight_kg"),
                row.get("reps"),
                str(row.get("notes", "") or ""),
                ex_canon,
            )
            strength_rows.append(
                {
                    "exercise_name": ex_canon,
                    "best_weight_kg": w_kg if w_kg is not None else "",
                    "best_reps": reps if reps is not None else "",
                    "notes": notes,
                }
            )
        else:
            date_val = str(row.get(first_col, "") or "").strip()
            if not _looks_like_date(date_val):
                warnings.append(
                    f"row {idx}: no clear date in first column {first_col!r}={date_val!r}; "
                    f"treating as workout row anyway"
                )
            w = _clean_cell(row.get("weight_kg"))
            r = _clean_cell(row.get("reps"))
            notes = str(row.get("notes", "") or "").strip()
            if isinstance(w, str) and w.upper() == "UNKNOWN":
                w = None
            if isinstance(r, str) and str(r).upper() == "UNKNOWN":
                r = None
            w_str = "" if w is None else str(w).strip()
            r_str = "" if r is None else str(r).strip()
            if w_str == "" and r_str == "" and "bodyweight" in notes.lower():
                w_str = "Bodyweight"
            workout_rows.append(
                {
                    "date": date_val,
                    "exercise_name": ex_canon,
                    "set_number": int(row["set_number"]) if pd.notna(row.get("set_number")) else "",
                    "weight_kg": w_str,
                    "reps": r_str,
                    "notes": notes,
                }
            )

    for wline in warnings:
        print(f"WARN: {wline}")

    if dry_run:
        print(f"Dry run: would write {len(strength_rows)} strength, {len(workout_rows)} workout rows.")
        return

    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = archive_dir / f"lifts_log_backup_{ts}.csv"
    shutil.copy2(lifts_log, backup)
    print(f"Backed up legacy log to {backup}")

    raw_workouts.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(strength_rows).to_csv(strength_out, index=False)
    pd.DataFrame(
        workout_rows,
        columns=["date", "exercise_name", "set_number", "weight_kg", "reps", "notes"],
    ).to_csv(workout_out, index=False)

    lifts_log.unlink()
    print(f"Wrote {strength_out} ({len(strength_rows)} rows)")
    print(f"Wrote {workout_out} ({len(workout_rows)} rows)")
    print(f"Removed legacy {lifts_log}")

    lib_set = set(library)
    for path, label in ((strength_out, "strength"), (workout_out, "workout")):
        check = pd.read_csv(path)
        if "exercise_name" not in check.columns:
            continue
        for i, name in enumerate(check["exercise_name"].astype(str)):
            if not str(name).strip():
                print(f"WARN: {label} row {i}: empty exercise_name")
            elif name.strip() not in lib_set:
                print(f"WARN: {label} row {i}: exercise not in library: {name!r}")
    print("Validation: exercise_name column checked against exercise_library.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    refactor(_project_root(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
