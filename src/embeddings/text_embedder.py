from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from config import cfg

DEFAULT_TEXT_MODEL = cfg.embeddings.text_model


def _slugify(name: str) -> str:
    return "_".join(name.lower().strip().replace("/", " ").split())


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _coaching_cues(row: pd.Series) -> str:
    movement = _safe_str(row.get("movement_pattern"))
    equipment = _safe_str(row.get("equipment"))
    cues: list[str] = []
    if "squat" in movement.lower():
        cues.append("Control knee tracking and keep torso braced.")
    if "hinge" in movement.lower():
        cues.append("Maintain neutral spine and hinge from hips.")
    if "press" in movement.lower():
        cues.append("Use full range and avoid shoulder shrugging.")
    if "machine" in equipment.lower():
        cues.append("Set seat and handles to align joint path.")
    return " ".join(cues)


def _injury_tags(row: pd.Series) -> str:
    movement = _safe_str(row.get("movement_pattern")).lower()
    tags: list[str] = []
    if "squat" in movement or "lunge" in movement:
        tags.append("knee_friendly_variant_available")
    if "hinge" in movement:
        tags.append("lower_back_load_awareness")
    if "press" in movement:
        tags.append("shoulder_positioning")
    return ",".join(tags)


def build_text_records(metadata_csv: Path, lifts_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if metadata_csv.is_file():
        df = pd.read_csv(metadata_csv)
        for _, row in df.iterrows():
            exercise_name = _safe_str(row.get("exercise_name"))
            if not exercise_name:
                continue
            movement = _safe_str(row.get("movement_pattern"))
            equipment = _safe_str(row.get("equipment"))
            muscles = _safe_str(row.get("primary_muscles"))
            tags = _injury_tags(row)
            cues = _coaching_cues(row)
            text = (
                f"exercise name: {exercise_name}. "
                f"muscle groups: {muscles}. "
                f"equipment: {equipment}. "
                f"movement pattern: {movement}. "
                f"injury tags: {tags}. "
                f"coaching cues: {cues}."
            )
            records.append(
                {
                    "id": f"text_meta_{_slugify(exercise_name)}",
                    "text": text,
                    "metadata": {
                        "record_type": "exercise_metadata",
                        "exercise_name": exercise_name,
                        "muscle_groups": muscles,
                        "equipment": equipment,
                        "movement_pattern": movement,
                        "injury_tags": tags,
                        "coaching_cues": cues,
                        "source_path": str(metadata_csv),
                    },
                }
            )

    strength_csv = lifts_dir / "strength.csv"
    if strength_csv.is_file():
        s_df = pd.read_csv(strength_csv)
        for i, row in s_df.iterrows():
            exercise_name = _safe_str(row.get("exercise_name"))
            if not exercise_name:
                continue
            best_weight = _safe_str(row.get("best_weight_kg"))
            best_reps = _safe_str(row.get("best_reps"))
            notes = _safe_str(row.get("notes"))
            text = (
                f"lift history for {exercise_name}. "
                f"best weight kg: {best_weight or 'unknown'}. "
                f"best reps: {best_reps or 'unknown'}. "
                f"notes: {notes or 'none'}."
            )
            records.append(
                {
                    "id": f"text_lift_{_slugify(exercise_name)}_{i}",
                    "text": text,
                    "metadata": {
                        "record_type": "lift_history",
                        "exercise_name": exercise_name,
                        "best_weight_kg": best_weight,
                        "best_reps": best_reps,
                        "notes": notes,
                        "source_path": str(strength_csv),
                    },
                }
            )
    return records


class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        vector = self.model.encode(query, normalize_embeddings=True)
        return vector.tolist()

