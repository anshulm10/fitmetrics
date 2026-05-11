from __future__ import annotations

from pathlib import Path

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
        return chunks

