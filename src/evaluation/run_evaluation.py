"""
Evaluation harness for the fit_support multimodal fitness agent.

Runs four system conditions against benchmark_queries.json and writes
results to data/eval/results.csv:

  1. plain_llm_baseline         — no retrieval, generic response
  2. text_only_retrieval        — semantic text search only
  3. full_multimodal_agent      — full LangGraph agent (all tools active)
  4. ablation_no_injury         — full agent with injury_lookup node disabled

Metrics per query × condition:
  - recall_at_k          Recall@3 against expected exercises
  - mrr                  Mean Reciprocal Rank
  - personalization_score  1–5 rubric based on record types used
  - relevance_score       1–5 rubric based on recall
  - latency_ms           Wall-clock retrieval time
  - tool_calls_count     Number of tools fired (from AgentState.tool_calls_log)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.graph import run_graph, run_graph_with_model
from agent.router import QueryRouter
from agent.tools import FitnessToolRouter
from config import cfg
from embeddings.image_embedder import ImageEmbedder
from embeddings.index_builder import build_indexes
from embeddings.text_embedder import TextEmbedder
from retrieval.search import search_exercise_by_text


# ── data helpers ───────────────────────────────────────────────────────────────

def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load benchmark queries from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(name: str) -> str:
    """Normalise an exercise name for fuzzy comparison."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def _exercise_from_record(record: dict[str, Any]) -> str:
    """Extract the exercise name from a retrieval record."""
    meta = record.get("metadata") or {}
    for key in ("exercise_name", "exercise_label"):
        if meta.get(key):
            return str(meta[key])
    doc = str(record.get("document", ""))
    if doc.startswith("exercise name:"):
        return doc.split(".", 1)[0].replace("exercise name:", "").strip()
    if ":" in doc:
        return doc.split(":", 1)[0].strip()
    return ""


# ── metrics ────────────────────────────────────────────────────────────────────

def recall_at_k(records: list[dict[str, Any]], expected: list[str], k: int = cfg.retrieval.recall_k) -> float:
    """Compute Recall@k: fraction of expected exercises found in top-k retrieved.

    Parameters
    ----------
    records : list[dict]
        Retrieved records ordered by relevance score.
    expected : list[str]
        Ground-truth exercise names.
    k : int
        Cutoff rank.
    """
    expected_norm = {_norm(e) for e in expected}
    if not expected_norm:
        return 0.0
    retrieved = [_norm(_exercise_from_record(r)) for r in records[:k]]
    hits = sum(1 for item in retrieved if item in expected_norm)
    return hits / min(k, len(expected_norm))


def mean_reciprocal_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/(rank of first relevant hit), or 0.

    Parameters
    ----------
    relevant_ids : list[str]
        Normalised names or IDs considered relevant.
    retrieved_ids : list[str]
        Retrieved IDs/names in rank order.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _response_relevance(records: list[dict[str, Any]], expected: list[str]) -> int:
    """Map recall score to a 1–5 relevance integer for the results table."""
    if not records:
        return 1
    r = recall_at_k(records, expected)
    if r >= 0.67:
        return 5
    if r >= 0.34:
        return 4
    if r > 0:
        return 3
    return 2


def _personalization_score(
    variant: str,
    category: str,
    records: list[dict[str, Any]],
    route: str | None,
) -> int:
    """Assign a 1–5 personalization score based on record types used."""
    if variant == "plain_llm_baseline":
        return 1
    record_types = {str((r.get("metadata") or {}).get("record_type", "")) for r in records}
    has_strength = "strength_progression" in record_types or "lift_history" in record_types
    has_injury = "injury_memory" in record_types
    has_image = "exercise_image" in record_types

    score = 2
    if variant == "text_only_retrieval" and records:
        score = 3
    if variant in ("full_multimodal_agent", "ablation_no_injury"):
        score = 3
        if category == "personalized_followup" and (has_strength or has_injury):
            score += 1
        if category == "analytical" and has_strength:
            score += 1
        if category == "cross_modal" and has_image:
            score += 1
        if route in {"personalized_followup", "analytical", "cross_modal"}:
            score += 1
    return max(1, min(score, 5))


# ── baseline variants ──────────────────────────────────────────────────────────

def _plain_baseline(query: str) -> list[dict[str, Any]]:
    """Return a generic no-retrieval placeholder record."""
    return [
        {
            "id": "plain_llm_baseline",
            "score": 0.0,
            "document": f"Generic coaching advice for: {query}",
            "metadata": {"record_type": "plain_baseline"},
        }
    ]


def _ablation_no_injury(
    query: str,
    image_path: str | None,
    tool_router: FitnessToolRouter,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], str, int, str]:
    """Run the full multimodal agent but strip injury_lookup records from results.

    Returns (records, route, tool_calls_count, final_response).  The
    tool_calls_count still reflects how many tools the graph tried to fire;
    injury records are simply excluded from the returned context so the
    downstream scorer sees a world without injury awareness.
    """
    if model is not None:
        state = run_graph_with_model(query, model=model, image_path=image_path)
    else:
        state = run_graph(query, image_path=image_path)
    route = state["query_type"]
    tool_calls = len(state["tool_calls_log"])

    result = tool_router.run(query, image_path=image_path, top_k=cfg.retrieval.top_k)
    records = [
        r for r in result["records"]
        if str((r.get("metadata") or {}).get("record_type", "")) != "injury_memory"
    ]
    return records, route, tool_calls, state.get("final_response", "")


# ── main evaluator ─────────────────────────────────────────────────────────────

def _evaluate_variant(
    *,
    variant: str,
    item: dict[str, Any],
    tool_router: FitnessToolRouter,
    text_embedder: TextEmbedder,
    model: str | None = None,
) -> dict[str, Any]:
    """Evaluate one (benchmark item, variant) pair and return a result row dict."""
    query = str(item["query"])
    expected = list(item.get("expected_exercises", []))
    category = str(item.get("category", ""))
    image_path = item.get("image_path")
    route = None
    tool_calls_count = 0
    final_response = ""

    start = time.perf_counter()

    if variant == "plain_llm_baseline":
        records = _plain_baseline(query)

    elif variant == "text_only_retrieval":
        records = search_exercise_by_text(
            query,
            top_k=cfg.retrieval.top_k,
            chroma_path=cfg.chroma.persist_path,
            embedder=text_embedder,
        )
        tool_calls_count = 1

    elif variant == "full_multimodal_agent":
        if model is not None:
            state = run_graph_with_model(query, model=model, image_path=image_path)
        else:
            state = run_graph(query, image_path=image_path)
        route = state["query_type"]
        tool_calls_count = len(state["tool_calls_log"])
        final_response = state.get("final_response", "")
        result = tool_router.run(query, image_path=image_path, top_k=cfg.retrieval.top_k)
        records = list(result["records"])

    elif variant == "ablation_no_injury":
        records, route, tool_calls_count, final_response = _ablation_no_injury(
            query, image_path, tool_router, model=model
        )

    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_names = [_exercise_from_record(r) for r in records[:cfg.retrieval.top_k]]
    expected_norm = [_norm(e) for e in expected]
    retrieved_norm = [_norm(n) for n in retrieved_names]

    return {
        "query_id": item["id"],
        "category": category,
        "system_name": variant,
        "query_type": route or QueryRouter().route(query, image_path=image_path).route.value,
        "query": query,
        "expected_exercises": "|".join(expected),
        "retrieved_top3": "|".join(retrieved_names),
        "recall_at_k": round(recall_at_k(records, expected), 4),
        "mrr": round(mean_reciprocal_rank(expected_norm, retrieved_norm), 4),
        "personalization_score": _personalization_score(variant, category, records, route),
        "relevance_score": _response_relevance(records, expected),
        "latency_ms": round(latency_ms, 2),
        "tool_calls_count": tool_calls_count,
        "final_response": final_response,
        "model": model or cfg.llm.primary_model,
    }


def run_evaluation(
    benchmark_path: Path | None = None,
    results_path: Path | None = None,
    *,
    rebuild_index: bool = False,
    model: str | None = None,
) -> pd.DataFrame:
    """Run all four evaluation conditions and write results to CSV.

    Parameters
    ----------
    benchmark_path : Path | None
        Path to benchmark_queries.json; defaults to tests/benchmark_queries.json.
    results_path : Path | None
        Path for the output CSV; defaults to cfg.evaluation.results_file.
    rebuild_index : bool
        When True, rebuilds the Chroma index before evaluation.
    model : str | None
        Ollama model tag for generation nodes (e.g. "llama3.1:8b" or
        "qwen2.5:7b"). None falls back to cfg.llm.primary_model.
    """
    benchmark_path = benchmark_path or ROOT / "tests/benchmark_queries.json"
    results_path = results_path or cfg.evaluation.results_file
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild_index:
        print("[EVAL] rebuilding Chroma indexes before evaluation")
        build_indexes()

    _model_label = model or cfg.llm.primary_model
    print(f"[EVAL] running with model={_model_label}")

    benchmark = _load_benchmark(benchmark_path)
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    router = FitnessToolRouter(text_embedder=text_embedder, image_embedder=image_embedder)

    variants = [
        "plain_llm_baseline",
        "text_only_retrieval",
        "full_multimodal_agent",
        "ablation_no_injury",
    ]
    rows: list[dict[str, Any]] = []

    for item in benchmark:
        for variant in variants:
            rows.append(
                _evaluate_variant(
                    variant=variant,
                    item=item,
                    tool_router=router,
                    text_embedder=text_embedder,
                    model=model,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[EVAL] wrote {results_path} rows={len(df)}")

    summary_cols = ["recall_at_k", "mrr", "personalization_score", "relevance_score", "latency_ms", "tool_calls_count"]
    summary = df.groupby("system_name")[summary_cols].mean().round(3)
    print("[EVAL] summary")
    print(summary)
    return df


def main() -> None:
    """CLI entry point for the evaluation harness."""
    parser = argparse.ArgumentParser(description="Evaluate fit_support agent conditions.")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model tag for generation (e.g. llama3.1:8b or qwen2.5:7b).",
    )
    args = parser.parse_args()
    run_evaluation(args.benchmark, args.results, rebuild_index=args.rebuild_index, model=args.model)


if __name__ == "__main__":
    main()
