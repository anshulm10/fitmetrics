from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer


DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _safe_name(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).strip()


def _slugify(name: str) -> str:
    return "_".join(name.lower().strip().replace("/", " ").split())


def _label_from_image_path(path: Path) -> str:
    return _safe_name(path.parent.name).title()


def _normalize_to_library(label: str, library_names: list[str]) -> str:
    if not label:
        return label
    exact = {n: n for n in library_names}
    if label in exact:
        return label
    lower_map = {n.lower(): n for n in library_names}
    if label.lower() in lower_map:
        return lower_map[label.lower()]
    slug_map = {_slugify(n): n for n in library_names}
    return slug_map.get(_slugify(label), label)


def build_image_records(images_root: Path, metadata_csv: Path) -> list[dict[str, Any]]:
    library_names: list[str] = []
    if metadata_csv.is_file():
        df = pd.read_csv(metadata_csv)
        if "exercise_name" in df.columns:
            library_names = [str(x).strip() for x in df["exercise_name"].dropna().tolist() if str(x).strip()]

    records: list[dict[str, Any]] = []
    for path in sorted(images_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        raw_label = _label_from_image_path(path)
        label = _normalize_to_library(raw_label, library_names)
        rel = path.relative_to(images_root)
        records.append(
            {
                "id": f"image_{_slugify(label)}_{_slugify(str(rel.with_suffix('')))}",
                "image_path": str(path),
                "metadata": {
                    "record_type": "exercise_image",
                    "exercise_label": label,
                    "source_path": str(path),
                },
            }
        )
    return records


class ImageEmbedder:
    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        if not image_paths:
            return []
        images = [Image.open(p).convert("RGB") for p in image_paths]
        vectors = self.model.encode(images, normalize_embeddings=True)
        return vectors.tolist()

    def embed_single_image(self, image_path: Path) -> list[float]:
        image = Image.open(image_path).convert("RGB")
        vector = self.model.encode(image, normalize_embeddings=True)
        return vector.tolist()

