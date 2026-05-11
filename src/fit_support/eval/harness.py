from __future__ import annotations

from dataclasses import dataclass

from fit_support.eval.rubric import RecommendationRubricScore


@dataclass
class EvalComparison:
    baseline_total: int
    rag_total: int

    @property
    def delta(self) -> int:
        return self.rag_total - self.baseline_total


def compare_baseline_vs_rag(
    baseline: RecommendationRubricScore,
    rag: RecommendationRubricScore,
) -> EvalComparison:
    return EvalComparison(baseline_total=baseline.total, rag_total=rag.total)

