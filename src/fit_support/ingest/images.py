from __future__ import annotations

from pathlib import Path

from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.ingest.base import BaseIngestor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class ImageIngestor(BaseIngestor):
    def load(self, source_dir: Path) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        for path in sorted(source_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            chunks.append(
                ContextChunk(
                    chunk_id=f"image::{path.stem}",
                    source_id=path.name,
                    modality=ModalityType.IMAGE,
                    content=f"Exercise image: {path.stem}",
                    metadata={"path": str(path)},
                )
            )
        return chunks

