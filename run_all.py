"""
Single-command full pipeline runner for fit_support.

Usage
-----
    uv run python run_all.py

Steps
-----
1. Load and validate config (config/config.yaml).
2. Run the data-ingestion pipeline (src/ingestion/pipeline.py).
3. Rebuild the ChromaDB vector index (src/embeddings/index_builder.py).
4. Run the full evaluation suite including ablation (src/evaluation/run_evaluation.py).
5. Print a formatted summary table of results to stdout.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _section(title: str) -> None:
    """Print a clearly visible section header."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _elapsed(start: float) -> str:
    """Return a human-readable elapsed time string."""
    secs = time.perf_counter() - start
    return f"{secs:.1f}s"


def main() -> None:
    """Execute the full fit_support pipeline end-to-end."""
    pipeline_start = time.perf_counter()

    # ── Step 1: Config ─────────────────────────────────────────────────────────
    _section("Step 1 / 5 — Load config")
    from config import cfg
    print(f"  text_model   : {cfg.embeddings.text_model}")
    print(f"  image_model  : {cfg.embeddings.image_model}")
    print(f"  top_k        : {cfg.retrieval.top_k}")
    print(f"  chroma_path  : {cfg.chroma.persist_path}")
    print(f"  random_seed  : {cfg.evaluation.random_seed}")
    print(f"  results_file : {cfg.evaluation.results_file}")

    # ── Step 2: Ingestion ──────────────────────────────────────────────────────
    _section("Step 2 / 5 — Data ingestion pipeline")
    t = time.perf_counter()
    from ingestion.pipeline import run_data_ingestion_pipeline
    stats = run_data_ingestion_pipeline(project_root=ROOT)
    print(f"  ingestion complete: {stats}  [{_elapsed(t)}]")

    # ── Step 3: Index rebuild ──────────────────────────────────────────────────
    _section("Step 3 / 5 — Rebuild ChromaDB vector index")
    t = time.perf_counter()
    from embeddings.index_builder import build_indexes
    index_stats = build_indexes()
    print(f"  index built: {index_stats}  [{_elapsed(t)}]")

    # ── Step 4: Evaluation ─────────────────────────────────────────────────────
    _section("Step 4 / 5 — Full evaluation suite (4 conditions)")
    t = time.perf_counter()
    from evaluation.run_evaluation import run_evaluation
    df = run_evaluation()
    print(f"  evaluation complete: {len(df)} rows  [{_elapsed(t)}]")

    # ── Step 5: Summary table ──────────────────────────────────────────────────
    _section("Step 5 / 5 — Results summary")
    summary_cols = [
        "recall_at_k",
        "mrr",
        "personalization_score",
        "relevance_score",
        "latency_ms",
        "tool_calls_count",
    ]
    summary = df.groupby("system_name")[summary_cols].mean().round(3)
    print(summary.to_string())
    print(f"\n  Results CSV : {cfg.evaluation.results_file}")
    print(f"\n  Total wall time: {_elapsed(pipeline_start)}")


if __name__ == "__main__":
    main()
