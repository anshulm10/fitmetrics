"""Phase 1: raw fitness data ingestion (load → validate → clean → save).

Prefer importing the entrypoint explicitly::

    from ingestion.pipeline import run_data_ingestion_pipeline
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_data_ingestion_pipeline"]


def __getattr__(name: str) -> Any:
    if name == "run_data_ingestion_pipeline":
        from src.ingestion.pipeline import run_data_ingestion_pipeline

        return run_data_ingestion_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
