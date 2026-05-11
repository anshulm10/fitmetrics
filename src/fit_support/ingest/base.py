from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fit_support.domain.schemas import ContextChunk


class BaseIngestor(ABC):
    @abstractmethod
    def load(self, source_dir: Path) -> list[ContextChunk]:
        raise NotImplementedError

