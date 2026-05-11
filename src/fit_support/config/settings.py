from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    raw_workouts_dir: Path = Field(default=Path("data/raw/workouts"))
    raw_lifts_dir: Path = Field(default=Path("data/raw/lifts"))
    raw_images_dir: Path = Field(default=Path("data/raw/images"))
    raw_injuries_dir: Path = Field(default=Path("data/raw/injuries"))
    processed_dir: Path = Field(default=Path("data/processed"))
    chroma_dir: Path = Field(default=Path("data/chroma"))
    eval_dir: Path = Field(default=Path("data/eval"))

    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_embedding_model: str = "clip-ViT-B-32"
    chroma_collection_prefix: str = "fit_support"
    top_k: int = 5

    def resolved(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    return AppSettings()

