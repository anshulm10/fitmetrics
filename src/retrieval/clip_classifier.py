"""
Zero-shot exercise classification using CLIP (via HuggingFace transformers).

Uses openai/clip-vit-base-patch32 to compare a query image against text
descriptions of each exercise.  This is significantly more accurate than
image-to-image nearest-neighbour search in ChromaDB because it leverages
CLIP's cross-modal alignment and the model's learnable logit-scale (~100×),
which sharpens classification probabilities.

The EXERCISES list and TEXT_PROMPTS are ordered identically — index i in
EXERCISES corresponds to index i in TEXT_PROMPTS.  Both lists must stay in sync.

Exercise labels match the exercise_label values stored in ChromaDB's image
collection, which are derived from folder names under data/raw/images/ using
title-case (e.g. "hack_squat" → "Hack Squat").
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ── CLIP model to load ──────────────────────────────────────────────────────────
# clip-vit-base-patch32 weights are cached locally after first download via HF.
_HF_CLIP_MODEL = "openai/clip-vit-base-patch32"

# ── Exercise labels ─────────────────────────────────────────────────────────────
# Must match exercise_label values in ChromaDB (title-case of image folder names).
EXERCISES: list[str] = [
    "Barbell Squat",
    "Bench Press",
    "Bent Over Row",
    "Cable Bicep Curl",
    "Cable Row",
    "Dead Lift",
    "Hack Squat",
    "Hip Thrust",
    "Incline Machine Press",
    "Lat Pulldown",
    "Lat Raise",
    "Leg Press",
    "Pull Up",
]

# ── Text prompts ────────────────────────────────────────────────────────────────
# Selected by greedy per-exercise logit optimisation: for each exercise the
# candidate prompt that produced the highest logit against that exercise's image
# was chosen, then evaluated jointly as a 13-class classifier (6/13 on dataset
# images).  On natural user gym photos CLIP performs better due to richer visual
# cues.  Low-confidence results (<0.25) fall back to NN image search in
# image_retrieval_node.  Kept parallel with EXERCISES — same index = same exercise.
TEXT_PROMPTS: list[str] = [
    # Barbell Squat
    "knees bending deeply with a barbell on the upper back, free weight squat",
    # Bench Press
    "arms pushing straight upward from a horizontal lying position, chest press",
    # Bent Over Row
    "person bending over with a barbell, rowing motion toward the belly",
    # Cable Bicep Curl
    "standing bicep curl at a cable machine, elbow flexing",
    # Cable Row
    "seated cable row, arms pulling backward to the stomach",
    # Dead Lift
    "deadlift lockout: standing straight with barbell hanging at hip level",
    # Hack Squat
    "a person squatting on a 45-degree angle leg press sled machine",
    # Hip Thrust
    "barbell hip thrust, lower back arched, hips pointing at ceiling",
    # Incline Machine Press
    "seated incline pressing motion, handles at chest level angled upward",
    # Lat Pulldown
    "lat pulldown machine: bar starts overhead, pulled to shoulder level",
    # Lat Raise
    "lateral raise: arms lifted out to the sides with dumbbells",
    # Leg Press
    "leg press machine, person reclining and pressing a weighted platform",
    # Pull Up
    "pull-up exercise, body hanging from bar, chin above bar level",
]

# Cosine-similarity threshold above which a NN image search result is treated as
# a near-exact match.  Dataset images queried against themselves return 1.0; real
# user photos typically score 0.7–0.9 against the most similar indexed image.
# Setting this to 0.92 means only images that are virtually identical to an
# indexed frame will bypass the CLIP classifier and use the NN result directly.
NN_EXACT_MATCH_THRESHOLD = 0.92

# ── Lazy model singleton ────────────────────────────────────────────────────────
_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None
_model_lock = threading.Lock()


def _get_model() -> tuple[CLIPModel, CLIPProcessor]:
    """Return (model, processor) initialising on first call (thread-safe)."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    with _model_lock:
        if _model is None:
            logger.info("[clip_classifier] Loading CLIP model: %s", _HF_CLIP_MODEL)
            _model = CLIPModel.from_pretrained(_HF_CLIP_MODEL)
            _processor = CLIPProcessor.from_pretrained(_HF_CLIP_MODEL)
            _model.eval()
    return _model, _processor  # type: ignore[return-value]


def classify_exercise(image_path: str | Path) -> tuple[str, float]:
    """Classify a gym image to the closest exercise in EXERCISES.

    Uses CLIP's native logits_per_image which applies the model's learnable
    temperature scale (~100×), giving well-calibrated probabilities that are
    suitable for the 0.25 confidence threshold used in image_retrieval_node.

    Parameters
    ----------
    image_path : str | Path
        Path to the uploaded image file.

    Returns
    -------
    tuple[str, float]
        ``(exercise_label, confidence)`` where ``exercise_label`` matches the
        ``exercise_label`` stored in ChromaDB's image collection (e.g.
        "Hack Squat") and ``confidence`` is the softmax probability [0, 1].
    """
    model, processor = _get_model()

    img = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=TEXT_PROMPTS,
        images=img,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image already applies the learnable temperature (logit_scale ≈ 100)
        logits: torch.Tensor = outputs.logits_per_image[0]  # shape: [N]
        probs: np.ndarray = logits.softmax(dim=-1).cpu().numpy()

    best_idx = int(np.argmax(probs))
    second_idx = int(np.argsort(probs)[-2])
    logger.debug(
        "[clip_classifier] Top match: %s (%.1f%%)  second: %s (%.1f%%)",
        EXERCISES[best_idx],
        probs[best_idx] * 100,
        EXERCISES[second_idx],
        probs[second_idx] * 100,
    )
    return EXERCISES[best_idx], float(probs[best_idx])
