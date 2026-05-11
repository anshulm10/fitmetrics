from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.ingest.base import BaseIngestor


class WorkoutIngestor(BaseIngestor):
    def load(self, source_dir: Path) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        for path in sorted(source_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            chunks.append(
                ContextChunk(
                    chunk_id=f"workout::{path.stem}",
                    source_id=path.name,
                    modality=ModalityType.WORKOUT,
                    content=text,
                    metadata={"path": str(path)},
                )
            )
        for path in sorted(source_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            for idx, row in frame.iterrows():
                record = {k: str(v) if pd.notna(v) else "" for k, v in row.to_dict().items()}
                content = " | ".join(f"{k}: {v}" for k, v in record.items() if v != "")
                chunks.append(
                    ContextChunk(
                        chunk_id=f"workout_csv::{path.stem}::{idx}",
                        source_id=path.name,
                        modality=ModalityType.WORKOUT,
                        content=content or str(record),
                        metadata={"path": str(path), "row_index": idx, **record},
                    )
                )
        return chunks

