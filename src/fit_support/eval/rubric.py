from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecommendationRubricScore:
    personalization: int
    injury_safety: int
    progression_fit: int
    explanation_quality: int

    @property
    def total(self) -> int:
        return (
            self.personalization
            + self.injury_safety
            + self.progression_fit
            + self.explanation_quality
        )


def score_recommendation(
    personalization: int,
    injury_safety: int,
    progression_fit: int,
    explanation_quality: int,
) -> RecommendationRubricScore:
    return RecommendationRubricScore(
        personalization=personalization,
        injury_safety=injury_safety,
        progression_fit=progression_fit,
        explanation_quality=explanation_quality,
    )

