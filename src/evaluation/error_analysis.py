"""
Error analysis module for the fit_support evaluation harness.

Identifies the 3 lowest-scoring queries per system condition from the
evaluation results CSV and saves them to data/eval/failure_cases.json
for manual review and labelling.

Usage
-----
    uv run python src/evaluation/error_analysis.py

Or import and call programmatically after run_evaluation():

    from evaluation.error_analysis import generate_failure_cases
    generate_failure_cases(df)

Manual labelling step
---------------------
After this function writes failure_cases.json, open the file and fill in
the ``failure_reason`` field for each entry.  Valid values:

  - wrong_retrieval        The wrong exercise(s) were retrieved.
  - missing_context        No relevant documents were returned at all.
  - hallucination          The response contained information not in the context.
  - unsafe_recommendation  The recommendation may worsen an existing injury.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.config import cfg

_FAILURE_CASES_PATH = ROOT / "data" / "eval" / "failure_cases.json"
_VALID_FAILURE_REASONS = frozenset(
    ["wrong_retrieval", "missing_context", "hallucination", "unsafe_recommendation"]
)
_CASES_PER_CONDITION = 3


def _build_case(row: pd.Series) -> dict[str, Any]:
    """Convert a single result row into a failure-case dict.

    Parameters
    ----------
    row : pd.Series
        A row from the evaluation results DataFrame.
    """
    retrieved_context: list[str] = [
        s.strip() for s in str(row.get("retrieved_top3", "")).split("|") if s.strip()
    ]
    return {
        "query": str(row.get("query", "")),
        "system": str(row.get("system_name", "")),
        "query_id": str(row.get("query_id", "")),
        "category": str(row.get("category", "")),
        "retrieved_context": retrieved_context,
        "response": f"[auto] retrieved: {'; '.join(retrieved_context) or 'none'}",
        "scores": {
            "recall_at_k": float(row.get("recall_at_k", 0.0)),
            "mrr": float(row.get("mrr", 0.0)),
            "relevance_score": int(row.get("relevance_score", 1)),
            "personalization_score": int(row.get("personalization_score", 1)),
            "latency_ms": float(row.get("latency_ms", 0.0)),
        },
        "failure_reason": "UNLABELLED",
    }


def generate_failure_cases(
    df: pd.DataFrame | None = None,
    results_path: Path | None = None,
    output_path: Path | None = None,
    *,
    n_per_condition: int = _CASES_PER_CONDITION,
) -> list[dict[str, Any]]:
    """Identify the lowest-scoring queries per condition and save to JSON.

    Selects the *n_per_condition* rows with the lowest ``relevance_score``
    for each system condition.  Ties are broken by ``recall_at_k`` (ascending)
    then ``mrr`` (ascending).

    Parameters
    ----------
    df : pd.DataFrame | None
        Pre-loaded results DataFrame.  If None, reads from *results_path*.
    results_path : Path | None
        Path to the evaluation results CSV.  Defaults to cfg.evaluation.results_file.
    output_path : Path | None
        Destination for failure_cases.json.  Defaults to data/eval/failure_cases.json.
    n_per_condition : int
        How many failure cases to collect per system condition.

    Returns
    -------
    list[dict]
        The collected failure cases (also written to *output_path*).
    """
    results_path = results_path or cfg.evaluation.results_file
    output_path = output_path or _FAILURE_CASES_PATH

    if df is None:
        if not results_path.is_file():
            raise FileNotFoundError(
                f"Results file not found: {results_path}\n"
                "Run `uv run python src/evaluation/run_evaluation.py` first."
            )
        df = pd.read_csv(results_path)

    required = {"system_name", "relevance_score", "recall_at_k", "mrr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results DataFrame missing columns: {missing}")

    cases: list[dict[str, Any]] = []
    for system_name, group in df.groupby("system_name"):
        bottom_n = (
            group.sort_values(
                by=["relevance_score", "recall_at_k", "mrr"],
                ascending=[True, True, True],
            )
            .head(n_per_condition)
        )
        for _, row in bottom_n.iterrows():
            cases.append(_build_case(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2, ensure_ascii=False)

    print(f"[ERROR ANALYSIS] wrote {len(cases)} failure cases -> {output_path}")
    print(
        "[ERROR ANALYSIS] MANUAL STEP REQUIRED: open failure_cases.json and "
        "label each 'failure_reason' field with one of: "
        + ", ".join(sorted(_VALID_FAILURE_REASONS))
    )
    return cases


def validate_labels(output_path: Path | None = None) -> bool:
    """Check that all failure cases have been manually labelled.

    Returns True if every ``failure_reason`` is a valid label (not 'UNLABELLED').
    Prints a report of any unlabelled or invalid entries.

    Parameters
    ----------
    output_path : Path | None
        Path to failure_cases.json.  Defaults to data/eval/failure_cases.json.
    """
    output_path = output_path or _FAILURE_CASES_PATH
    if not output_path.is_file():
        print(f"[VALIDATE] file not found: {output_path}")
        return False

    with open(output_path, encoding="utf-8") as fh:
        cases = json.load(fh)

    unlabelled: list[str] = []
    invalid: list[str] = []
    for case in cases:
        reason = case.get("failure_reason", "")
        qid = case.get("query_id", case.get("query", "?"))
        sys_name = case.get("system", "?")
        label = f"{sys_name}::{qid}"
        if reason == "UNLABELLED":
            unlabelled.append(label)
        elif reason not in _VALID_FAILURE_REASONS:
            invalid.append(f"{label} ({reason!r})")

    if unlabelled:
        print(f"[VALIDATE] {len(unlabelled)} unlabelled cases: {unlabelled}")
    if invalid:
        print(f"[VALIDATE] {len(invalid)} invalid labels: {invalid}")
    if not unlabelled and not invalid:
        print(f"[VALIDATE] all {len(cases)} cases are correctly labelled.")
        return True
    return False


def main() -> None:
    """CLI entry point: generate failure cases from the latest results CSV."""
    generate_failure_cases()


if __name__ == "__main__":
    main()
