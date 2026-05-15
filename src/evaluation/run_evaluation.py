"""
Evaluation harness for the fit_support multimodal fitness agent.

Runs four system conditions against benchmark_queries.json and writes
results to data/eval/results.csv:

  1. plain_llm_baseline   — same LLM as production, **no retrieval**: one direct
     generate call with a no-database system prompt (no LangGraph).
  2. text_only_retrieval  — Chroma text (+ image NN for cross-modal items)
     then a **direct LLM** call conditioned only on those snippets (no full graph).
  3. full_multimodal_agent — compiled LangGraph + production generation.
  4. ablation_no_injury    — same graph **without** ``injury_lookup`` in routing,
     and ``FitnessToolRouter`` skips injury tools; metrics use the same stripped
     record list as before.

Metrics per query × condition:
  - recall_at_k / mrr      From retrieved records vs benchmark ``relevant_ids``.
  - relevance_score       Integer 1–5 **derived from retrieval recall** (not an LLM judge).
  - personalization_score Heuristic from ``record_type`` mix + category.
  - groundedness_score    Heuristic: share of retrieved exercise names cited in
    ``final_response`` (baseline variant is always 1 by definition).
  - latency_ms             End-to-end for that variant (includes eval LLM calls).
  - tool_calls_count       Retrieval tools only; baseline eval uses 0.

The README Appendix A prompt describes an optional **human/LLM judge** rubric;
this script does not call a separate judge model for the CSV columns above.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import call_gemini, call_ollama, run_graph, run_graph_with_model
from src.agent.muscle_filter import filter_exercise_context_records
from src.agent.tools import FitnessToolRouter
from src.config import cfg
from src.embeddings.image_embedder import ImageEmbedder
from src.embeddings.index_builder import build_indexes
from src.embeddings.text_embedder import TextEmbedder
from src.retrieval.search import search_exercise_by_text, search_similar_exercise_image


# ── constants ──────────────────────────────────────────────────────────────────

DISPLAY_ORDER = [
    "plain_llm_baseline",
    "text_only_retrieval",
    "full_multimodal_agent",
    "ablation_no_injury",
]

FAMILY_ORDER = [
    "factual_retrieval",
    "cross_modal",
    "analytical",
    "personalized_followup",
]

_FAMILY_PRIMARY_METRIC = {
    "factual_retrieval": "recall_at_k",
    "cross_modal": "recall_at_k",
    "analytical": "relevance_score",
    "personalized_followup": "relevance_score",
}

_FAMILY_FAIL_REASON = {
    "factual_retrieval": "no retrieval means no personal data",
    "cross_modal": "text-only misses visual patterns",
    "analytical": "multi-hop needs full tool stack",
    "personalized_followup": "injury context critical for safety advice",
}

_PLAIN_BASELINE_SYSTEM = (
    "You are a fitness coach. You have NO access to the user's workout database, "
    "injury files, retrieval results, or uploaded images for this question. "
    "Answer from general knowledge only. If they ask for personal numbers or "
    "what they personally lifted, say you do not have their logs and give safe "
    "general guidance. At most 5 short sentences. Do not use bullet lists."
)

_TEXT_ONLY_EVAL_SYSTEM = (
    "You are FitSupport. Use ONLY the provided retrieved text snippets; they may "
    "be incomplete. Do not invent the user's personal bests unless they appear in "
    "the snippets. At most 5 short sentences. Do not use bullet lists."
)


def _eval_model_tag(model: str | None) -> str:
    return (model or cfg.llm.primary_model or "").strip()


def _eval_provider_and_api_model(model_tag: str) -> tuple[str, str]:
    sel = model_tag.lower()
    if sel in {"gemini", "google"} or sel.startswith("gemini"):
        return "gemini", cfg.llm.gemini_model
    return "ollama", model_tag


def _eval_llm(user_prompt: str, system_prompt: str, model: str | None) -> str:
    tag = _eval_model_tag(model)
    provider, api_model = _eval_provider_and_api_model(tag)
    if provider == "gemini":
        return call_gemini(user_prompt, system_prompt, api_model, timeout=300)
    return call_ollama(user_prompt, system_prompt, api_model, timeout=300)


def _format_records_for_eval_prompt(records: list[dict[str, Any]], limit: int) -> str:
    parts: list[str] = []
    for r in records[:limit]:
        doc = str(r.get("document", "")).strip()
        if doc:
            parts.append(doc)
    return "\n---\n".join(parts) if parts else "(no retrieved text)"


def _generate_plain_baseline_answer(query: str, model: str | None) -> str:
    return _eval_llm(f"User question:\n{query}", _PLAIN_BASELINE_SYSTEM, model)


def _generate_text_only_answer(query: str, records: list[dict[str, Any]], model: str | None) -> str:
    cap = max(cfg.retrieval.top_k, 3) * 3
    ctx = _format_records_for_eval_prompt(records, cap)
    return _eval_llm(
        f"User question:\n{query}\n\nRetrieved snippets:\n{ctx}",
        _TEXT_ONLY_EVAL_SYSTEM,
        model,
    )


# ── data helpers ───────────────────────────────────────────────────────────────

def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load benchmark queries from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(name: str) -> str:
    """Normalise an exercise name or ID for fuzzy comparison."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def _resolve_benchmark_image_path(image_path: str | Path | None) -> Path | None:
    """Return an absolute path to the image if it exists under ROOT."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.is_file() else None


def _merge_recall_records(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Concatenate *primary* then *secondary*, dedupe by record ``id``, cap at *limit*."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in primary + secondary:
        key = str(r.get("id", ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(r)
        if len(merged) >= limit:
            break
    return merged


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
    """Compute Recall@k: fraction of expected IDs found in top-k retrieved.

    Parameters
    ----------
    records : list[dict]
        Retrieved records ordered by relevance score.
    expected : list[str]
        Ground-truth IDs or exercise names (e.g. "hack_squat", "INJ_001").
    k : int
        Cutoff rank.
    """
    expected_norm = {_norm(e) for e in expected}
    if not expected_norm:
        return 0.0
    retrieved = [_norm(_exercise_from_record(r)) for r in records[:k]]
    hits = sum(1 for item in retrieved if item in expected_norm)
    raw = hits / min(k, len(expected_norm))
    return min(1.0, raw)


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
    has_strength = (
        "strength_progression" in record_types
        or "lift_history" in record_types
        or "lift_record" in record_types
    )
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


def _groundedness_score(
    variant: str,
    records: list[dict[str, Any]],
    final_response: str,
) -> int:
    """Assign a 1–5 groundedness score.

    Groundedness measures whether the generated response actually cites the
    retrieved context.  Implemented as a deterministic heuristic matching
    retrieved exercise names against the final_response text.

    Scoring rubric
    --------------
    1 — plain_llm_baseline (no retrieval, no context to be grounded in)
    2 — retrieval ran but no LLM response exists (text_only_retrieval),
        OR response is empty / no exercise names were retrieved
    3 — response mentions at least one retrieved exercise name
    4 — response mentions ≥34 % of retrieved exercise names
    5 — response mentions ≥67 % of retrieved exercise names
    """
    if variant == "plain_llm_baseline":
        return 1
    if not final_response or not records:
        return 2

    retrieved_names = [
        _norm(_exercise_from_record(r))
        for r in records
        if _exercise_from_record(r)
    ]
    if not retrieved_names:
        return 2

    response_lower = final_response.lower()
    hits = sum(1 for name in retrieved_names if name and name in response_lower)
    ratio = hits / len(retrieved_names)

    if ratio >= 0.67:
        return 5
    if ratio >= 0.34:
        return 4
    if ratio > 0:
        return 3
    return 2


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
    """Run the full multimodal agent without injury retrieval in graph or tool merge.

    ``injury_lookup`` is omitted from routing when ``skip_injury_lookup`` is set on
    the graph state, and ``FitnessToolRouter`` skips injury tools so evaluation
    records align with what the model could condition on without injury memory.
    """
    if model is not None:
        state = run_graph_with_model(
            query, model=model, image_path=image_path, skip_injury_lookup=True
        )
    else:
        state = run_graph(query, image_path=image_path, skip_injury_lookup=True)
    route = state["query_type"]
    tool_calls = len(state["tool_calls_log"])

    result = tool_router.run(
        query, image_path=image_path, top_k=cfg.retrieval.top_k, skip_injury=True
    )
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
    image_embedder: ImageEmbedder | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Evaluate one (benchmark item, variant) pair and return a result row dict."""
    query = str(item["query"])
    # Support both new format (query_id / relevant_ids) and legacy (id / expected_exercises)
    query_id = item.get("query_id", item.get("id", "unknown"))
    expected = list(item.get("relevant_ids", item.get("expected_exercises", [])))
    category = str(item.get("category", ""))
    image_path = item.get("image_path")
    route = None
    tool_calls_count = 0
    final_response = ""

    start = time.perf_counter()

    if variant == "plain_llm_baseline":
        records = _plain_baseline(query)
        final_response = _generate_plain_baseline_answer(query, model)

    elif variant == "text_only_retrieval":
        records = search_exercise_by_text(
            query,
            top_k=cfg.retrieval.top_k,
            chroma_path=cfg.chroma.persist_path,
            embedder=text_embedder,
        )
        tool_calls_count = 1
        if category == "cross_modal" and image_embedder is not None:
            img_path = _resolve_benchmark_image_path(image_path)
            if img_path is not None:
                img_recs = search_similar_exercise_image(
                    img_path,
                    top_k=cfg.retrieval.top_k,
                    chroma_path=cfg.chroma.persist_path,
                    embedder=image_embedder,
                )
                cap = max(cfg.retrieval.top_k, 3) * 3
                records = _merge_recall_records(img_recs, records, limit=cap)
                tool_calls_count = 2
        records = filter_exercise_context_records(query, list(records))
        final_response = _generate_text_only_answer(query, records, model)

    elif variant == "full_multimodal_agent":
        if model is not None:
            state = run_graph_with_model(query, model=model, image_path=image_path)
        else:
            state = run_graph(query, image_path=image_path)
        route = state["query_type"]
        tool_calls_count = len(state["tool_calls_log"])
        final_response = state.get("final_response", "")
        result = tool_router.run(
            query, image_path=image_path, top_k=cfg.retrieval.top_k, skip_injury=False
        )
        records = list(result["records"])

    elif variant == "ablation_no_injury":
        records, route, tool_calls_count, final_response = _ablation_no_injury(
            query, image_path, tool_router, model=model
        )

    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    if variant not in ("plain_llm_baseline", "text_only_retrieval"):
        records = filter_exercise_context_records(query, list(records))

    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_names = [_exercise_from_record(r) for r in records[:cfg.retrieval.top_k]]
    expected_norm = [_norm(e) for e in expected]
    retrieved_norm = [_norm(n) for n in retrieved_names]

    reported_route = route if route is not None else str(category)

    return {
        "query_id": query_id,
        "category": category,
        "system_name": variant,
        "query_type": reported_route,
        "query": query,
        "relevant_ids": "|".join(expected),
        "retrieved_top3": "|".join(retrieved_names),
        "recall_at_k": round(recall_at_k(records, expected), 4),
        "mrr": round(mean_reciprocal_rank(expected_norm, retrieved_norm), 4),
        "personalization_score": _personalization_score(variant, category, records, reported_route),
        "relevance_score": _response_relevance(records, expected),
        "groundedness_score": _groundedness_score(variant, records, final_response),
        "latency_ms": round(latency_ms, 2),
        "tool_calls_count": tool_calls_count,
        "final_response": final_response,
        "model": model or cfg.llm.active_model_name,
    }


# ── per-family reporting ───────────────────────────────────────────────────────

def compute_family_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-family average metrics grouped by (category, system_name)."""
    metric_cols = [
        "recall_at_k", "mrr", "relevance_score", "personalization_score",
        "groundedness_score", "latency_ms",
    ]
    available = [c for c in metric_cols if c in df.columns]
    family_df = (
        df.groupby(["category", "system_name"])[available]
        .mean()
        .round(3)
        .reset_index()
    )
    return family_df


def print_family_table(df: pd.DataFrame) -> None:
    """Print per-family breakdown table and per-family analysis notes."""
    fam_w, sys_w, num_w = 24, 24, 16

    header = (
        f"{'Query Family':<{fam_w}}"
        f"{'System':<{sys_w}}"
        f"{'Recall@3':>{num_w}}"
        f"{'MRR':>{num_w}}"
        f"{'Relevance':>{num_w}}"
        f"{'Personalization':>{num_w}}"
    )
    sep = "-" * (fam_w + sys_w + num_w * 4)

    print()
    print(header)
    print(sep)

    family_df = compute_family_breakdown(df)

    all_families = list(FAMILY_ORDER) + [
        f for f in family_df["category"].unique() if f not in FAMILY_ORDER
    ]
    present_families = [f for f in all_families if f in family_df["category"].values]

    for fam in present_families:
        fam_rows = family_df[family_df["category"] == fam]
        printed_any = False
        for sys_name in DISPLAY_ORDER:
            row = fam_rows[fam_rows["system_name"] == sys_name]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"{fam:<{fam_w}}"
                f"{sys_name:<{sys_w}}"
                f"{r.get('recall_at_k', 0.0):>{num_w}.3f}"
                f"{r.get('mrr', 0.0):>{num_w}.3f}"
                f"{r.get('relevance_score', 0.0):>{num_w}.3f}"
                f"{r.get('personalization_score', 0.0):>{num_w}.3f}"
            )
            printed_any = True
        if printed_any:
            print()

    # Analysis notes: best vs worst system per family
    print("[EVAL] Per-family analysis:")
    for fam in present_families:
        fam_rows = family_df[family_df["category"] == fam]
        if fam_rows.empty:
            continue
        primary = _FAMILY_PRIMARY_METRIC.get(fam, "recall_at_k")
        if primary not in fam_rows.columns:
            continue
        best_row = fam_rows.loc[fam_rows[primary].idxmax()]
        worst_row = fam_rows.loc[fam_rows[primary].idxmin()]
        best_sys = best_row["system_name"]
        best_score = best_row[primary]
        worst_sys = worst_row["system_name"]
        worst_score = worst_row[primary]
        fail_reason = _FAMILY_FAIL_REASON.get(fam, "limited context")
        print(
            f"  {fam}: {best_sys} best ({best_score:.3f}), "
            f"{worst_sys} fails ({worst_score:.3f}) — {fail_reason}"
        )


# ── main run ───────────────────────────────────────────────────────────────────

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
        Provider model tag for generation nodes (e.g. "llama3.1:8b",
        "qwen2.5:7b", or "gemini"). None falls back to the configured LLM.
    """
    benchmark_path = benchmark_path or ROOT / "tests/benchmark_queries.json"
    results_path = results_path or cfg.evaluation.results_file
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild_index:
        print("[EVAL] rebuilding Chroma indexes before evaluation")
        build_indexes()

    _model_label = model or cfg.llm.active_model_name
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
                    image_embedder=image_embedder,
                    model=model,
                )
            )

    df = pd.DataFrame(rows)
    df.to_csv(results_path, index=False)
    print(f"[EVAL] wrote {results_path} rows={len(df)}")

    # Per-family breakdown CSV
    family_df = compute_family_breakdown(df)
    family_results_path = results_path.parent / "family_results.csv"
    family_df.to_csv(family_results_path, index=False)
    print(f"[EVAL] wrote {family_results_path}")

    # Overall system-level summary
    summary_cols = [
        "recall_at_k", "mrr", "relevance_score",
        "personalization_score", "groundedness_score",
        "latency_ms", "tool_calls_count",
    ]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df.groupby("system_name")[available_cols].mean().round(3)
    print("\n[EVAL] overall system summary (all families combined)")
    print(summary.to_string())

    # Per-family table
    print("\n[EVAL] per-family breakdown")
    print_family_table(df)

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
