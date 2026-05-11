from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    WORKOUT = "workout"
    LIFT = "lift"
    INJURY = "injury"
    IMAGE = "image"


class ContextChunk(BaseModel):
    chunk_id: str
    source_id: str
    modality: ModalityType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

