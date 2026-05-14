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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    _section("Step 1 / 7 — Load config")
    from src.config import cfg
    print(f"  text_model   : {cfg.embeddings.text_model}")
    print(f"  image_model  : {cfg.embeddings.image_model}")
    print(f"  top_k        : {cfg.retrieval.top_k}")
    print(f"  chroma_path  : {cfg.chroma.persist_path}")
    print(f"  llm          : {cfg.llm.provider}:{cfg.llm.active_model_name}")
    print(f"  random_seed  : {cfg.evaluation.random_seed}")
    print(f"  results_file : {cfg.evaluation.results_file}")

    # ── Step 2: Ingestion ──────────────────────────────────────────────────────
    _section("Step 2 / 7 — Data ingestion pipeline")
    t = time.perf_counter()
    from src.ingestion.pipeline import run_data_ingestion_pipeline
    stats = run_data_ingestion_pipeline(project_root=ROOT)
    print(f"  ingestion complete: {stats}  [{_elapsed(t)}]")

    # ── Step 3: Index rebuild ──────────────────────────────────────────────────
    _section("Step 3 / 7 — Rebuild ChromaDB vector index")
    t = time.perf_counter()
    from src.embeddings.index_builder import build_indexes
    index_stats = build_indexes()
    print(f"  index built: {index_stats}  [{_elapsed(t)}]")

    # ── Step 4: LLM warmup ─────────────────────────────────────────────────────
    _section("Step 4 / 7 — Warm up configured LLM")
    t = time.perf_counter()
    from src.agent.graph import call_gemini, call_ollama
    if cfg.llm.provider == "gemini":
        _warmup = call_gemini(
            user_prompt="Say OK.",
            system_prompt="You are a fitness coach.",
            model=cfg.llm.gemini_model,
            timeout=300,
        )
    else:
        _warmup = call_ollama(
            user_prompt="Say OK.",
            system_prompt="You are a fitness coach.",
            model=cfg.llm.primary_model,
            timeout=300,
        )
    print(f"  warmup response: {_warmup[:80]!r}  [{_elapsed(t)}]")

    # ── Step 5: Evaluation (primary model) ────────────────────────────────────
    _section("Step 5 / 7 — Full evaluation suite (4 conditions, primary model)")
    t = time.perf_counter()
    from src.evaluation.run_evaluation import run_evaluation
    df = run_evaluation()
    print(f"  evaluation complete: {len(df)} rows  [{_elapsed(t)}]")

    # ── Step 6: Model comparison ───────────────────────────────────────────────
    _section("Step 6 / 7 — Model comparison (llama3.1:8b vs qwen2.5:7b)")

    _eval_dir = cfg.evaluation.results_file.parent
    _eval_dir.mkdir(parents=True, exist_ok=True)

    model_runs = [
        (cfg.llm.primary_model,   _eval_dir / "results_llama.csv"),
        (cfg.llm.secondary_model, _eval_dir / "results_qwen.csv"),
    ]

    comparison_frames = []
    for _model, _path in model_runs:
        print(f"\n  [warmup] model={_model} ...")
        _wu = call_ollama("Say OK.", "You are a fitness coach.", _model, timeout=300)
        print(f"    warmup: {_wu[:60]!r}")
        print(f"  > Running evaluation with model={_model} ...")
        t = time.perf_counter()
        try:
            _df = run_evaluation(results_path=_path, model=_model)
            print(f"    done: {len(_df)} rows  [{_elapsed(t)}]  saved -> {_path.name}")
            _df["model_tag"] = _model
            comparison_frames.append(_df)
        except Exception as exc:
            print(f"    [WARN] model={_model} failed: {exc}")

    if comparison_frames:
        import pandas as _pd
        comp_df = _pd.concat(comparison_frames, ignore_index=True)
        _section("Model comparison table")
        _summary_cols = [
            "recall_at_k",
            "mrr",
            "relevance_score",
            "personalization_score",
            "groundedness_score",
            "latency_ms",
            "tool_calls_count",
        ]
        comp_summary = (
            comp_df[comp_df["system_name"] == "full_multimodal_agent"]
            .groupby("model_tag")[_summary_cols]
            .mean()
            .round(3)
        )
        print(comp_summary.to_string())

    # ── Step 7: Per-family summary table ──────────────────────────────────────
    _section("Step 7 / 7 — Per-family results summary (primary model)")

    import pandas as _pd
    from src.evaluation.run_evaluation import compute_family_breakdown, print_family_table

    # Overall system-level summary (aggregated across all families)
    summary_cols = [
        "recall_at_k",
        "mrr",
        "relevance_score",
        "personalization_score",
        "groundedness_score",
        "latency_ms",
        "tool_calls_count",
    ]
    available_cols = [c for c in summary_cols if c in df.columns]
    summary = df.groupby("system_name")[available_cols].mean().round(3)

    col_labels = {
        "recall_at_k":           "Recall@3",
        "mrr":                   "MRR",
        "relevance_score":       "Relevance",
        "personalization_score": "Personalization",
        "groundedness_score":    "Groundedness",
        "latency_ms":            "Latency(ms)",
        "tool_calls_count":      "ToolCalls",
    }
    display_order = [
        "plain_llm_baseline",
        "text_only_retrieval",
        "full_multimodal_agent",
        "ablation_no_injury",
    ]
    col_w = 16
    sys_w = 24

    header = f"{'System':<{sys_w}}" + "".join(f"{col_labels[c]:>{col_w}}" for c in available_cols)
    sep    = "-" * (sys_w + col_w * len(available_cols))
    rows_str = []
    for sysname in display_order:
        if sysname not in summary.index:
            continue
        row = summary.loc[sysname]
        line = f"{sysname:<{sys_w}}" + "".join(f"{row[c]:>{col_w}.3f}" for c in available_cols)
        rows_str.append(line)

    table_lines = [header, sep] + rows_str
    table_str = "\n".join(table_lines)

    print("\n--- Overall system summary (all families combined) ---")
    print(table_str)

    # Per-family breakdown table
    print("\n--- Per-family breakdown ---")
    print_family_table(df)

    # Save summary to file
    summary_path = cfg.evaluation.results_file.parent / "summary_table.txt"
    rubric_map = (
        "\n\nRubric metric mapping\n"
        + "-" * 40 + "\n"
        "Retrieval metric   -> Recall@3 + MRR\n"
        "Answer quality     -> Relevance + Personalization + Groundedness\n"
        "Efficiency metric  -> Latency(ms) + ToolCalls\n"
    )
    family_df = compute_family_breakdown(df)
    family_table_str = family_df.to_string(index=False)
    summary_path.write_text(
        table_str + rubric_map + "\n\nPer-family breakdown\n" + "-" * 40 + "\n" + family_table_str,
        encoding="utf-8",
    )
    print(f"\n  Summary table saved -> {summary_path}")

    # Rubric mapping
    _section("Rubric metric mapping")
    print("  Retrieval metric   ->  Recall@3 + MRR                          [OK]")
    print("  Answer quality     ->  Relevance + Personalization + Groundedness [OK]")
    print("  Efficiency metric  ->  Latency(ms) + ToolCalls                 [OK]")

    family_results_path = cfg.evaluation.results_file.parent / "family_results.csv"
    print(f"\n  Results CSV        : {cfg.evaluation.results_file}")
    print(f"  Family results CSV : {family_results_path}")
    print(f"  Summary table      : {summary_path}")
    print(f"  results_llama.csv  : {_eval_dir / 'results_llama.csv'}")
    print(f"  results_qwen.csv   : {_eval_dir / 'results_qwen.csv'}")
    print(f"\n  Total wall time: {_elapsed(pipeline_start)}")


if __name__ == "__main__":
    main()
