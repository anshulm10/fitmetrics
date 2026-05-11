from __future__ import annotations

from pathlib import Path

from fit_support.config import AppSettings
from fit_support.domain.schemas import ContextChunk
from fit_support.embeddings.embedder import EmbeddingService
from fit_support.ingest.images import ImageIngestor
from fit_support.ingest.injuries import InjuryIngestor
from fit_support.ingest.lifts_csv import LiftCsvIngestor
from fit_support.ingest.workouts import WorkoutIngestor
from fit_support.retrieval.vector_store import VectorStore


def validate_required_directories(settings: AppSettings) -> None:
    required = [
        settings.raw_metadata_dir,
        settings.raw_workouts_dir,
        settings.raw_lifts_dir,
        settings.raw_images_dir,
        settings.raw_injuries_dir,
        settings.processed_dir,
        settings.chroma_db_dir,
        settings.eval_dir,
    ]
    for rel_path in required:
        abs_path = settings.resolved(rel_path)
        abs_path.mkdir(parents=True, exist_ok=True)


def run_ingestion_pipeline(settings: AppSettings) -> dict[str, int]:
    validate_required_directories(settings)

    workout_chunks = WorkoutIngestor().load(settings.resolved(settings.raw_workouts_dir))
    lift_chunks = LiftCsvIngestor().load(settings.resolved(settings.raw_lifts_dir))
    injury_chunks = InjuryIngestor().load(settings.resolved(settings.raw_injuries_dir))
    image_chunks = ImageIngestor().load(settings.resolved(settings.raw_images_dir))
    all_chunks: list[ContextChunk] = [*workout_chunks, *lift_chunks, *injury_chunks, *image_chunks]

    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings)
    vector_store.upsert_chunks(all_chunks, embedding_service)

    return {
        "workouts": len(workout_chunks),
        "lifts": len(lift_chunks),
        "injuries": len(injury_chunks),
        "images": len(image_chunks),
        "total": len(all_chunks),
    }

