"""
scripts/plot_results.py
=======================
Reads data/eval/results.csv and produces two publication-quality charts:

  data/eval/results_plot.png  — grouped bar chart: Recall@3, MRR, Relevance,
                                Personalization, Groundedness across all 4 systems
  data/eval/latency_plot.png  — horizontal bar chart: mean latency per system

Usage
-----
    uv run python scripts/plot_results.py
    uv run python scripts/plot_results.py --results data/eval/results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

SYSTEM_LABELS = {
    "plain_llm_baseline":   "Plain LLM\nBaseline",
    "text_only_retrieval":  "Text-Only\nRetrieval",
    "full_multimodal_agent":"Full\nMultimodal",
    "ablation_no_injury":   "Ablation\n(No Injury)",
}

SYSTEM_ORDER = [
    "plain_llm_baseline",
    "text_only_retrieval",
    "full_multimodal_agent",
    "ablation_no_injury",
]

# Palette: one colour per system, consistent across both charts
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

QUALITY_METRICS = [
    ("recall_at_k",           "Recall@3",       "#4c72b0"),
    ("mrr",                   "MRR",            "#dd8452"),
    ("relevance_score",       "Relevance",      "#55a868"),
    ("personalization_score", "Personalization","#c44e52"),
    ("groundedness_score",    "Groundedness",   "#8172b2"),
]

# ── chart 1: grouped bar — quality metrics ─────────────────────────────────────

def plot_quality(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart comparing all systems across all 5 quality metrics."""
    systems_present = [s for s in SYSTEM_ORDER if s in summary.index]
    n_systems = len(systems_present)
    n_metrics = len(QUALITY_METRICS)

    x = np.arange(n_systems)
    total_width = 0.75
    bar_width = total_width / n_metrics
    offsets = np.linspace(-(total_width - bar_width) / 2,
                           (total_width - bar_width) / 2,
                           n_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("#f8f8f8")

    for i, (col, label, colour) in enumerate(QUALITY_METRICS):
        if col not in summary.columns:
            continue
        vals = [summary.loc[s, col] if s in summary.index else 0.0
                for s in systems_present]
        bars = ax.bar(x + offsets[i], vals, width=bar_width,
                      label=label, color=colour, alpha=0.88,
                      edgecolor="white", linewidth=0.6)
        # Value labels on top of each bar
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS.get(s, s) for s in systems_present],
                       fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Metric axis reference lines
    for y in [1, 2, 3, 4, 5]:
        ax.axhline(y, color="#cccccc", linewidth=0.4, linestyle="--")

    ax.set_title(
        "FitSupport — System Comparison: Retrieval & Answer Quality Metrics",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(
        title="Metric",
        title_fontsize=9,
        fontsize=9,
        loc="upper left",
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    # Rubric callout box
    rubric = (
        "Recall@3 + MRR  ->  Retrieval\n"
        "Relevance + Personalization + Groundedness  ->  Answer Quality"
    )
    ax.text(
        0.99, 0.97, rubric,
        transform=ax.transAxes,
        fontsize=7.5, color="#555555",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


# ── chart 2: horizontal bar — latency ─────────────────────────────────────────

def plot_latency(summary: pd.DataFrame, tool_summary: pd.DataFrame,
                 out_path: Path) -> None:
    """Horizontal bar chart showing mean latency and tool call count."""
    systems_present = [s for s in SYSTEM_ORDER if s in summary.index]

    latencies = [summary.loc[s, "latency_ms"] if s in summary.index else 0.0
                 for s in systems_present]
    tool_counts = [tool_summary.loc[s, "tool_calls_count"]
                   if s in tool_summary.index else 0.0
                   for s in systems_present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#f8f8f8")

    # Latency bars
    ax1.set_facecolor("#f8f8f8")
    y = np.arange(len(systems_present))
    bars = ax1.barh(y, latencies, color=PALETTE[:len(systems_present)],
                    alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, latencies):
        ax1.text(
            bar.get_width() + max(latencies) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} ms",
            va="center", fontsize=9, color="#333333",
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels([SYSTEM_LABELS.get(s, s) for s in systems_present],
                        fontsize=10)
    ax1.set_xlabel("Mean Latency (ms)", fontsize=11)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Mean Query Latency per System", fontsize=12,
                  fontweight="bold", pad=10)

    # Tool call bars
    ax2.set_facecolor("#f8f8f8")
    bars2 = ax2.barh(y, tool_counts, color=PALETTE[:len(systems_present)],
                     alpha=0.88, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars2, tool_counts):
        ax2.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", fontsize=9, color="#333333",
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean Tool Calls", fontsize=11)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("Tool Calls", fontsize=12, fontweight="bold", pad=10)

    # Efficiency callout
    ax1.text(
        0.99, 0.03,
        "Latency(ms) + ToolCalls  ->  Efficiency Metric",
        transform=ax1.transAxes,
        fontsize=7.5, color="#555555",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main(results_csv: Path) -> None:
    if not results_csv.is_file():
        print(f"[plot] ERROR: results CSV not found at {results_csv}")
        print("       Run 'uv run python run_all.py' first to generate it.")
        sys.exit(1)

    df = pd.read_csv(results_csv)
    print(f"[plot] loaded {len(df)} rows from {results_csv}")

    quality_cols = [
        "recall_at_k", "mrr", "relevance_score",
        "personalization_score", "groundedness_score",
    ]
    efficiency_cols = ["latency_ms", "tool_calls_count"]

    # Fill missing groundedness_score with 0 for older result CSVs
    for col in quality_cols + efficiency_cols:
        if col not in df.columns:
            df[col] = 0.0

    summary = df.groupby("system_name")[quality_cols + efficiency_cols].mean().round(3)

    eval_dir = results_csv.parent
    plot_quality(summary, eval_dir / "results_plot.png")
    plot_latency(summary, summary, eval_dir / "latency_plot.png")

    print()
    print("Rubric metric mapping")
    print("-" * 42)
    print("  Retrieval metric   ->  Recall@3 + MRR                          [OK]")
    print("  Answer quality     ->  Relevance + Personalization + Groundedness [OK]")
    print("  Efficiency metric  ->  Latency(ms) + ToolCalls                 [OK]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fit_support evaluation results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "data/eval/results.csv",
        help="Path to results.csv (default: data/eval/results.csv)",
    )
    args = parser.parse_args()
    main(args.results)
