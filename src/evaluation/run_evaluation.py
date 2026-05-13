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

from agent.router import QueryRouter
from agent.tools import FitnessToolRouter
from embeddings.image_embedder import ImageEmbedder
from embeddings.index_builder import build_indexes
from embeddings.text_embedder import TextEmbedder
from retrieval.search import search_exercise_by_text


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def _exercise_from_record(record: dict[str, Any]) -> str:
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


def _recall_at_3(records: list[dict[str, Any]], expected: list[str]) -> float:
    expected_norm = {_norm(e) for e in expected}
    if not expected_norm:
        return 0.0
    retrieved = [_norm(_exercise_from_record(r)) for r in records[:3]]
    hits = sum(1 for item in retrieved if item in expected_norm)
    return hits / min(3, len(expected_norm))


def _response_relevance(records: list[dict[str, Any]], expected: list[str]) -> int:
    if not records:
        return 1
    recall = _recall_at_3(records, expected)
    if recall >= 0.67:
        return 5
    if recall >= 0.34:
        return 4
    if recall > 0:
        return 3
    return 2


def _personalization_score(
    variant: str,
    category: str,
    records: list[dict[str, Any]],
    route: str | None,
) -> int:
    if variant == "plain_llm_baseline":
        return 1
    record_types = {str((r.get("metadata") or {}).get("record_type", "")) for r in records}
    has_strength = "strength_progression" in record_types or "lift_history" in record_types
    has_injury = "injury_memory" in record_types
    has_image = "exercise_image" in record_types

    score = 2
    if variant == "text_only_retrieval" and records:
        score = 3
    if variant == "full_multimodal_agent":
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


def _plain_baseline(query: str) -> list[dict[str, Any]]:
    """Deterministic no-retrieval baseline: generic response with no exercise evidence."""
    return [
        {
            "id": "plain_llm_baseline",
            "score": 0.0,
            "document": f"Generic coaching advice for: {query}",
            "metadata": {"record_type": "plain_baseline"},
        }
    ]


def _evaluate_variant(
    *,
    variant: str,
    item: dict[str, Any],
    tool_router: FitnessToolRouter,
    text_embedder: TextEmbedder,
) -> dict[str, Any]:
    query = str(item["query"])
    expected = list(item.get("expected_exercises", []))
    category = str(item.get("category", ""))
    image_path = item.get("image_path")
    route = None

    start = time.perf_counter()
    if variant == "plain_llm_baseline":
        records = _plain_baseline(query)
    elif variant == "text_only_retrieval":
        records = search_exercise_by_text(
            query,
            top_k=3,
            chroma_path=ROOT / "data/chroma",
            embedder=text_embedder,
        )
    elif variant == "full_multimodal_agent":
        result = tool_router.run(query, image_path=image_path, top_k=3)
        route = str(result["route"])
        records = list(result["records"])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved = [_exercise_from_record(r) for r in records[:3]]
    return {
        "query_id": item["id"],
        "category": category,
        "variant": variant,
        "route": route or QueryRouter().route(query, image_path=image_path).route.value,
        "query": query,
        "expected_exercises": "|".join(expected),
        "retrieved_top3": "|".join(retrieved),
        "recall_at_3": round(_recall_at_3(records, expected), 4),
        "response_relevance": _response_relevance(records, expected),
        "personalization_score": _personalization_score(variant, category, records, route),
        "latency_ms": round(latency_ms, 2),
    }


def run_evaluation(
    benchmark_path: Path | None = None,
    results_path: Path | None = None,
    *,
    rebuild_index: bool = False,
) -> pd.DataFrame:
    benchmark_path = benchmark_path or ROOT / "tests/benchmark_queries.json"
    results_path = results_path or ROOT / "data/eval/results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild_index:
        print("[EVAL] rebuilding Chroma indexes before evaluation")
        build_indexes()

    benchmark = _load_benchmark(benchmark_path)
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    router = FitnessToolRouter(text_embedder=text_embedder, image_embedder=image_embedder)
    variants = ["plain_llm_baseline", "text_only_retrieval", "full_multimodal_agent"]
    rows: list[dict[str, Any]] = []

    for item in benchmark:
        for variant in variants:
            rows.append(
                _evaluate_variant(
                    variant=variant,
                    item=item,
                    tool_router=router,
                    text_embedder=text_embedder,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[EVAL] wrote {results_path} rows={len(df)}")
    summary = (
        df.groupby("variant")[["recall_at_3", "response_relevance", "personalization_score", "latency_ms"]]
        .mean()
        .round(3)
    )
    print("[EVAL] summary")
    print(summary)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs text-only vs full multimodal fitness agent.")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.benchmark, args.results, rebuild_index=args.rebuild_index)


if __name__ == "__main__":
    main()

