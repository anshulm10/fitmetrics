"""
Centralised configuration loader for the fit_support project.

Reads config/config.yaml from the project root and exposes typed
dataclasses so every module can import constants without hardcoding strings
or numbers.

Usage
-----
    from config import cfg, user_profile

    top_k     = cfg.retrieval.top_k
    seed      = cfg.evaluation.random_seed
    text_col  = cfg.chroma.text_collection
    philosophy = user_profile["coaching_philosophy"]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
_USER_PROFILE_PATH = _ROOT / "data" / "raw" / "user_profile.json"


# ── section dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmbeddingsConfig:
    """Model identifiers for text and image embedding."""

    text_model: str
    image_model: str


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval hyperparameters."""

    top_k: int
    recall_k: int


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB collection names and persistence path."""

    text_collection: str
    image_collection: str
    persist_directory: str

    @property
    def persist_path(self) -> Path:
        """Absolute path to the ChromaDB persist directory."""
        return (_ROOT / self.persist_directory).resolve()


@dataclass(frozen=True)
class LLMConfig:
    """LLM model names and Ollama endpoint."""

    primary_model: str
    secondary_model: str
    ollama_base_url: str


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation harness settings."""

    random_seed: int
    results_path: str
    rejected_rows_path: str

    @property
    def results_file(self) -> Path:
        """Absolute path to the evaluation results CSV."""
        return (_ROOT / self.results_path).resolve()

    @property
    def rejected_rows_file(self) -> Path:
        """Absolute path to the rejected rows CSV."""
        return (_ROOT / self.rejected_rows_path).resolve()


@dataclass(frozen=True)
class AppConfig:
    """Root config object — access sub-sections as attributes."""

    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    chroma: ChromaConfig
    llm: LLMConfig
    evaluation: EvaluationConfig


# ── loader ─────────────────────────────────────────────────────────────────────

def _parse(raw: dict[str, Any]) -> AppConfig:
    """Convert a raw YAML dict into a typed AppConfig."""
    return AppConfig(
        embeddings=EmbeddingsConfig(**raw["embeddings"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        chroma=ChromaConfig(**raw["chroma"]),
        llm=LLMConfig(**raw["llm"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )


@lru_cache(maxsize=1)
def load_config(config_path: Path = _CONFIG_PATH) -> AppConfig:
    """Load and cache the YAML config.  Call once; subsequent calls are free.

    Parameters
    ----------
    config_path : Path
        Override the default config path (useful in tests).
    """
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _parse(raw)


# Module-level singleton — import `cfg` directly for convenience.
cfg: AppConfig = load_config()


# ── user profile ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_user_profile(profile_path: Path = _USER_PROFILE_PATH) -> dict[str, Any]:
    """Load and cache data/raw/user_profile.json.  Call once; subsequent calls are free.

    Parameters
    ----------
    profile_path : Path
        Override the default profile path (useful in tests).
    """
    with open(profile_path, encoding="utf-8") as fh:
        return json.load(fh)


# Module-level singleton — import `user_profile` directly for convenience.
user_profile: dict[str, Any] = load_user_profile()
