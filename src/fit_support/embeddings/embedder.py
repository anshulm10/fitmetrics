from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from fit_support.config.settings import AppSettings


class EmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self._text_model = SentenceTransformer(settings.text_embedding_model)
        self._image_model = SentenceTransformer(settings.image_embedding_model)

    def embed_text(self, text: str) -> list[float]:
        return self._text_model.encode(text, normalize_embeddings=True).tolist()

    def embed_image(self, path: Path) -> list[float]:
        image = Image.open(path).convert("RGB")
        image_np = np.array(image)
        return self._image_model.encode(image_np, normalize_embeddings=True).tolist()

