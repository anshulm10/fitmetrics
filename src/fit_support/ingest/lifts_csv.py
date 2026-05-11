from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit_support.domain.schemas import ContextChunk, ModalityType
from fit_support.ingest.base import BaseIngestor


class LiftCsvIngestor(BaseIngestor):
    def load(self, source_dir: Path) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        for path in sorted(source_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            for idx, row in frame.iterrows():
                record = {k: str(v) for k, v in row.to_dict().items()}
                chunks.append(
                    ContextChunk(
                        chunk_id=f"lift::{path.stem}::{idx}",
                        source_id=path.name,
                        modality=ModalityType.LIFT,
                        content=" | ".join(f"{k}: {v}" for k, v in record.items()),
                        metadata={"path": str(path), "row_index": idx, **record},
                    )
                )
        return chunks

