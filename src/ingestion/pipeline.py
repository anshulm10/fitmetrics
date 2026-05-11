from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion import preprocess
from ingestion.loaders import load_csv_dir, load_json_dir, load_metadata_dir
from ingestion.models import ExerciseMetadata, InjuryRecord, LiftRecord, WorkoutRecord
from ingestion.validators import (
    validate_exercise_in_library,
    validate_exercise_metadata,
    validate_injury_record,
    validate_lift_record,
    validate_non_negative_number,
    validate_workout_record,
)


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}")


def _project_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(__file__).resolve().parents[2]


def _reject(
    rejected: list[dict[str, Any]],
    *,
    phase: str,
    source: str,
    row_index: int | None,
    payload: dict[str, Any],
    errors: list[str],
) -> None:
    rejected.append(
        {
            "phase": phase,
            "source": source,
            "row_index": row_index if row_index is not None else "",
            "payload_json": json.dumps(payload, default=str),
            "errors": "; ".join(errors),
        }
    )


def run_data_ingestion_pipeline(project_root: Path | None = None) -> dict[str, int]:
    root = _project_root(project_root)
    raw_lifts = root / "data" / "raw" / "lifts"
    raw_workouts = root / "data" / "raw" / "workouts"
    raw_injuries = root / "data" / "raw" / "injuries"
    raw_metadata = root / "data" / "raw" / "metadata"
    processed = root / "data" / "processed"
    eval_dir = root / "data" / "eval"
    processed.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = eval_dir / "rejected_rows.csv"

    rejected: list[dict[str, Any]] = []

    _log("LOAD", f"project_root={root}")

    # --- Load library first for name validation ---
    library_path = raw_metadata / "exercise_library.csv"
    library_names: list[str] = []
    if library_path.is_file():
        lib_df = pd.read_csv(library_path)
        if "exercise_name" in lib_df.columns:
            library_names = [
                str(x).strip() for x in lib_df["exercise_name"].dropna().tolist() if str(x).strip()
            ]
    library_set = set(library_names)

    _log("LOAD", f"exercise_library: {len(library_names)} names")

    meta_csv, meta_json = load_metadata_dir(raw_metadata)
    lift_tables = load_csv_dir(raw_lifts, "lifts")
    workout_tables = load_csv_dir(raw_workouts, "workouts")
    injury_json = load_json_dir(raw_injuries, "injuries")

    _log(
        "LOAD",
        f"tables: metadata_csv={len(meta_csv)} metadata_json={len(meta_json)} "
        f"lifts={len(lift_tables)} workouts={len(workout_tables)} injuries_json={len(injury_json)}",
    )

    # --- Exercises from library + optional JSON supplements ---
    exercise_rows: list[dict[str, Any]] = []
    if library_path.is_file():
        lib_df = pd.read_csv(library_path)
        for idx, row in lib_df.iterrows():
            exercise_rows.append({**{k: row[k] for k in lib_df.columns}, "_source": "metadata:exercise_library.csv", "_idx": idx})

    for source, records in meta_json:
        for idx, rec in enumerate(records):
            if "exercise_name" not in rec and "name" in rec:
                rec = {**rec, "exercise_name": rec.get("name")}
            exercise_rows.append({**rec, "_source": source, "_idx": idx})

    for source, df in meta_csv:
        if "exercise_library" in source and "exercise_library.csv" in source:
            continue
        for idx, row in df.iterrows():
            exercise_rows.append({**row.to_dict(), "_source": source, "_idx": idx})

    _log("VALIDATE", f"exercise candidate rows: {len(exercise_rows)}")
    exercises_clean: list[dict[str, Any]] = []
    seen_exercise: set[str] = set()
    for row in exercise_rows:
        row = dict(row)
        src = str(row.pop("_source", "unknown"))
        idx = row.pop("_idx", None)
        for k, v in list(row.items()):
            if isinstance(v, float) and pd.isna(v):
                row[k] = None
        name_raw = row.get("exercise_name")
        name, _note = preprocess.normalize_exercise_name(str(name_raw) if name_raw is not None else None, library_names)
        if name is None:
            _reject(rejected, phase="VALIDATE", source=src, row_index=int(idx) if idx is not None else None, payload=row, errors=["missing exercise_name"])
            continue
        row["exercise_name"] = name
        errs = validate_exercise_in_library(name, library_set) if library_set else []
        if errs and library_set:
            _reject(rejected, phase="VALIDATE", source=src, row_index=int(idx) if idx is not None else None, payload=row, errors=errs)
            continue
        obj, verr = validate_exercise_metadata({k: v for k, v in row.items() if not str(k).startswith("_")})
        if obj is None:
            _reject(rejected, phase="VALIDATE", source=src, row_index=int(idx) if idx is not None else None, payload=row, errors=verr)
            continue
        if name in seen_exercise:
            _reject(rejected, phase="VALIDATE", source=src, row_index=int(idx) if idx is not None else None, payload=row, errors=["duplicate exercise_name in ingest batch"])
            continue
        seen_exercise.add(name)
        exercises_clean.append(obj.model_dump())

    _log("CLEAN", f"exercises accepted: {len(exercises_clean)}")

    # --- Lifts (strength-style CSV in raw/lifts) ---
    lifts_clean: list[dict[str, Any]] = []
    seen_lift_keys: set[tuple[Any, ...]] = set()
    for source, df in lift_tables:
        for idx, row in df.iterrows():
            d = row.to_dict()
            for k, v in list(d.items()):
                if isinstance(v, float) and pd.isna(v):
                    d[k] = None
            w = preprocess.null_unknown(d.get("best_weight_kg"))
            d["best_weight_kg"] = preprocess.coerce_float(w)
            d["best_reps"] = preprocess.coerce_int(preprocess.null_unknown(d.get("best_reps")))
            if d.get("exercise_name") is not None:
                en, _ = preprocess.normalize_exercise_name(str(d["exercise_name"]), library_names)
                d["exercise_name"] = en
            nv = d.get("notes")
            if nv is None or (isinstance(nv, float) and pd.isna(nv)):
                d["notes"] = None
            else:
                d["notes"] = preprocess.normalize_whitespace(str(nv))
            d["source_file"] = source
            errs: list[str] = []
            errs.extend(validate_non_negative_number("best_weight_kg", d.get("best_weight_kg")))
            errs.extend(validate_non_negative_number("best_reps", d.get("best_reps")))
            if library_set and d.get("exercise_name"):
                errs.extend(validate_exercise_in_library(str(d["exercise_name"]), library_set))
            obj, verr = validate_lift_record({k: v for k, v in d.items() if k in LiftRecord.model_fields})
            if verr:
                errs.extend(verr)
            if errs:
                _reject(rejected, phase="VALIDATE", source=source, row_index=int(idx), payload=d, errors=errs)
                continue
            dump = obj.model_dump()
            lk = (
                dump.get("exercise_name"),
                dump.get("best_weight_kg"),
                dump.get("best_reps"),
                dump.get("notes"),
            )
            if lk in seen_lift_keys:
                _reject(
                    rejected,
                    phase="VALIDATE",
                    source=source,
                    row_index=int(idx),
                    payload=d,
                    errors=["duplicate lift row (same exercise, weight, reps, notes)"],
                )
                continue
            seen_lift_keys.add(lk)
            lifts_clean.append(dump)

    _log("CLEAN", f"lifts accepted: {len(lifts_clean)}")

    # --- Workouts ---
    workouts_clean: list[dict[str, Any]] = []
    seen_workout_keys: set[tuple[Any, ...]] = set()
    for source, df in workout_tables:
        if df.empty:
            continue
        for idx, row in df.iterrows():
            d = row.to_dict()
            for k, v in list(d.items()):
                if isinstance(v, float) and pd.isna(v):
                    d[k] = None
            if all(preprocess.null_unknown(d.get(c)) is None for c in ("date", "exercise_name")):
                continue
            if preprocess.null_unknown(d.get("date")) is None:
                _reject(rejected, phase="VALIDATE", source=source, row_index=int(idx), payload=d, errors=["missing date"])
                continue
            nd = preprocess.normalize_date_value(d.get("date"))
            if nd is not None:
                d["date"] = nd.isoformat()
            if d.get("exercise_name") is not None:
                en, _ = preprocess.normalize_exercise_name(str(d["exercise_name"]), library_names)
                d["exercise_name"] = en
            d["set_number"] = preprocess.coerce_int(d.get("set_number")) or 1
            wv = preprocess.null_unknown(d.get("weight_kg"))
            if wv is not None and str(wv).strip().lower() != "bodyweight":
                d["weight_kg"] = preprocess.coerce_float(wv)
            else:
                d["weight_kg"] = wv
            d["reps"] = preprocess.coerce_int(preprocess.null_unknown(d.get("reps")))
            nv = d.get("notes")
            if nv is None or (isinstance(nv, float) and pd.isna(nv)):
                d["notes"] = None
            else:
                d["notes"] = preprocess.normalize_whitespace(str(nv))
            d["source_file"] = source
            errs: list[str] = []
            if isinstance(d.get("weight_kg"), (int, float)):
                errs.extend(validate_non_negative_number("weight_kg", d.get("weight_kg")))
            errs.extend(validate_non_negative_number("reps", d.get("reps")))
            if library_set and d.get("exercise_name"):
                errs.extend(validate_exercise_in_library(str(d["exercise_name"]), library_set))
            obj, verr = validate_workout_record({k: v for k, v in d.items() if k in WorkoutRecord.model_fields})
            if verr:
                errs.extend(verr)
            if errs:
                _reject(rejected, phase="VALIDATE", source=source, row_index=int(idx), payload=d, errors=errs)
                continue
            dump = obj.model_dump(mode="json")
            wk = (
                dump.get("date"),
                dump.get("exercise_name"),
                dump.get("set_number"),
                str(dump.get("weight_kg")),
                dump.get("reps"),
            )
            if wk in seen_workout_keys:
                _reject(
                    rejected,
                    phase="VALIDATE",
                    source=source,
                    row_index=int(idx),
                    payload=d,
                    errors=["duplicate workout set row"],
                )
                continue
            seen_workout_keys.add(wk)
            workouts_clean.append(dump)

    _log("CLEAN", f"workouts accepted: {len(workouts_clean)}")

    # --- Injuries ---
    injuries_clean: list[dict[str, Any]] = []
    for source, records in injury_json:
        for idx, rec in enumerate(records):
            rec = dict(rec)
            if not any(v not in (None, "", [], {}) for v in rec.values()):
                continue
            rec["source_file"] = source
            obj, verr = validate_injury_record(rec)
            if obj is None:
                _reject(rejected, phase="VALIDATE", source=source, row_index=idx, payload=rec, errors=verr)
                continue
            flat = obj.model_dump()
            extras = {k: v for k, v in rec.items() if k not in flat and not str(k).startswith("_")}
            if extras:
                flat["extra_json"] = json.dumps(extras, default=str)
            injuries_clean.append(flat)

    _log("CLEAN", f"injuries accepted: {len(injuries_clean)}")

    # --- SAVE ---
    out_ex = processed / "exercises_clean.csv"
    out_lifts = processed / "lifts_clean.csv"
    out_w = processed / "workouts_clean.csv"
    out_inj = processed / "injuries_clean.csv"

    pd.DataFrame(exercises_clean).to_csv(out_ex, index=False)
    pd.DataFrame(lifts_clean).to_csv(out_lifts, index=False)
    w_cols = list(WorkoutRecord.model_fields.keys())
    if workouts_clean:
        pd.DataFrame(workouts_clean).to_csv(out_w, index=False)
    else:
        pd.DataFrame(columns=w_cols).to_csv(out_w, index=False)
    inj_cols = list(InjuryRecord.model_fields.keys()) + ["extra_json"]
    if injuries_clean:
        pd.DataFrame(injuries_clean).to_csv(out_inj, index=False)
    else:
        pd.DataFrame(columns=inj_cols).to_csv(out_inj, index=False)

    rej_df = pd.DataFrame(rejected)
    if rej_df.empty:
        rej_df = pd.DataFrame(columns=["phase", "source", "row_index", "payload_json", "errors"])
    rej_df.to_csv(rejected_path, index=False)

    _log("SAVE", str(out_ex))
    _log("SAVE", str(out_lifts))
    _log("SAVE", str(out_w))
    _log("SAVE", str(out_inj))
    _log("SAVE", f"rejected_rows -> {rejected_path} ({len(rejected)} rows)")

    return {
        "exercises": len(exercises_clean),
        "lifts": len(lifts_clean),
        "workouts": len(workouts_clean),
        "injuries": len(injuries_clean),
        "rejected": len(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 data ingestion: load → validate → clean → save")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: auto-detect)")
    args = parser.parse_args()
    stats = run_data_ingestion_pipeline(project_root=args.root)
    print("Done:", stats)


if __name__ == "__main__":
    # Allow `python path/to/pipeline.py` by putting `src` on sys.path
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    main()
