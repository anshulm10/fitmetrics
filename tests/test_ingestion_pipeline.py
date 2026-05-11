from __future__ import annotations

from pathlib import Path

import pandas as pd

from fit_support.config.settings import AppSettings
from fit_support.ingest.pipeline import run_ingestion_pipeline


def _write_seed_data(base_dir: Path) -> None:
    (base_dir / "data/raw/workouts").mkdir(parents=True, exist_ok=True)
    (base_dir / "data/raw/lifts").mkdir(parents=True, exist_ok=True)
    (base_dir / "data/raw/injuries").mkdir(parents=True, exist_ok=True)
    (base_dir / "data/raw/images").mkdir(parents=True, exist_ok=True)

    (base_dir / "data/raw/workouts/session1.txt").write_text(
        "Back squat 3x5, RPE 7", encoding="utf-8"
    )
    pd.DataFrame([{"exercise": "squat", "weight": 100, "reps": 5}]).to_csv(
        base_dir / "data/raw/lifts/lifts.csv", index=False
    )
    (base_dir / "data/raw/injuries/knee.txt").write_text(
        "Mild knee pain after deep flexion", encoding="utf-8"
    )
    # Tiny valid 1x1 PNG header bytes for smoke tests.
    (base_dir / "data/raw/images/squat.png").write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE0000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"
        )
    )


def test_ingestion_pipeline_counts(tmp_path: Path, monkeypatch) -> None:
    _write_seed_data(tmp_path)

    settings = AppSettings(project_root=tmp_path)

    class _FakeEmbedder:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def embed_text(_text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        @staticmethod
        def embed_image(_path: Path) -> list[float]:
            return [0.3, 0.2, 0.1]

    class _FakeStore:
        def __init__(self, *_args, **_kwargs):
            self.upserts = 0

        def upsert_chunks(self, chunks, embedder):
            self.upserts += len(chunks)
            assert embedder is not None

    monkeypatch.setattr("fit_support.ingest.pipeline.EmbeddingService", _FakeEmbedder)
    monkeypatch.setattr("fit_support.ingest.pipeline.VectorStore", _FakeStore)

    result = run_ingestion_pipeline(settings)
    assert result["workouts"] == 1
    assert result["lifts"] == 1
    assert result["injuries"] == 1
    assert result["images"] == 1
    assert result["total"] == 4

