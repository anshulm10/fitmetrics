from fit_support.eval.harness import compare_baseline_vs_rag
from fit_support.eval.rubric import score_recommendation


def test_eval_comparison_delta() -> None:
    baseline = score_recommendation(2, 2, 2, 2)
    rag = score_recommendation(4, 4, 3, 3)

    comparison = compare_baseline_vs_rag(baseline, rag)
    assert comparison.baseline_total == 8
    assert comparison.rag_total == 14
    assert comparison.delta == 6

