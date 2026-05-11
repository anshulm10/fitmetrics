from __future__ import annotations

from pathlib import Path

from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.ingest.base import BaseIngestor


class InjuryIngestor(BaseIngestor):
    def load(self, source_dir: Path) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        for path in sorted(source_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            chunks.append(
                ContextChunk(
                    chunk_id=f"injury::{path.stem}",
                    source_id=path.name,
                    modality=ModalityType.INJURY,
                    content=text,
                    metadata={"path": str(path), "pain_flag": True},
                )
            )
        return chunks

