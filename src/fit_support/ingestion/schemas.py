from __future__ import annotations

from pydantic import BaseModel, Field


class ExerciseMetadataRecord(BaseModel):
    exercise_id: str
    exercise_name: str
    body_part: str | None = None
    equipment: str | None = None
    movement_pattern: str | None = None
    injury_tags: list[str] = Field(default_factory=list)
    source_path: str | None = None


class IngestedExerciseRecord(BaseModel):
    chunk_id: str
    source_id: str
    text_content: str
    image_path: str | None = None
    metadata: ExerciseMetadataRecord

