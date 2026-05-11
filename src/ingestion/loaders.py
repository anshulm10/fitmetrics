from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_csv_dir(directory: Path, label: str) -> list[tuple[str, pd.DataFrame]]:
    if not directory.is_dir():
        return []
    out: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(directory.glob("*.csv")):
        out.append((f"{label}:{path.name}", pd.read_csv(path)))
    return out


def load_json_dir(directory: Path, label: str) -> list[tuple[str, list[dict[str, Any]]]]:
    if not directory.is_dir():
        return []
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append((f"{label}:{path.name}", _read_json_records(path)))
        except json.JSONDecodeError:
            out.append((f"{label}:{path.name}", []))
    return out


def load_metadata_dir(
    directory: Path,
) -> tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, list[dict[str, Any]]]]]:
    csv_parts = load_csv_dir(directory, "metadata")
    json_parts: list[tuple[str, list[dict[str, Any]]]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                json_parts.append((f"metadata:{path.name}", _read_json_records(path)))
            except json.JSONDecodeError:
                json_parts.append((f"metadata:{path.name}", []))
    return csv_parts, json_parts
