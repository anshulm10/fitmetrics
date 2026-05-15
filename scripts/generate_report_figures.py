"""
Generate publication-style figures from evaluation CSVs for the final report.

Reads:
  - data/eval/results.csv       (per-query × system)
  - data/eval/family_results.csv  (aggregated by query family)

Writes high-resolution PNGs under reports/figures/ (Figures 1–3 by default).

Usage:
    uv run python scripts/generate_report_figures.py
    uv run python scripts/generate_report_figures.py --with-failure-table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ───────────────────────────────────────────────────────────────

SYSTEM_ORDER = [
    "plain_llm_baseline",
    "text_only_retrieval",
    "full_multimodal_agent",
    "ablation_no_injury",
]

SYSTEM_LABELS_SHORT = {
    "plain_llm_baseline": "Plain LLM",
    "text_only_retrieval": "Text retrieval",
    "full_multimodal_agent": "Full multimodal",
    "ablation_no_injury": "Ablation\n(no injury)",
}

SYSTEM_LABELS_LEGEND = {
    "plain_llm_baseline": "Plain LLM baseline",
    "text_only_retrieval": "Text-only retrieval",
    "full_multimodal_agent": "Full multimodal agent",
    "ablation_no_injury": "Ablation (no injury memory)",
}

FAMILY_ORDER = [
    "factual_retrieval",
    "cross_modal",
    "analytical",
    "personalized_followup",
]

FAMILY_LABELS = {
    "factual_retrieval": "Factual\nretrieval",
    "cross_modal": "Cross-\nmodal",
    "analytical": "Analytical",
    "personalized_followup": "Personalized\nfollow-up",
}

PALETTE = ["#3d5a80", "#ee6c4d", "#2a9d8f", "#e9c46a"]

DPI = 300
FIG_W = 7.2  # inches (single column ~3.5in; use ~7 for two-column width)


def _setup_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def _systems_present(df: pd.DataFrame) -> list[str]:
    return [s for s in SYSTEM_ORDER if s in set(df["system_name"])]


def _trunc(text: str, max_len: int = 140) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def figure1_overall_comparison(df: pd.DataFrame, out: Path) -> None:
    """Stacked grouped bar panels: retrieval, rubric, latency."""
    systems = _systems_present(df)
    n_sys = len(systems)
    cols_mean = [
        "recall_at_k",
        "mrr",
        "personalization_score",
        "groundedness_score",
        "latency_ms",
    ]
    summary = df.groupby("system_name")[cols_mean].mean(numeric_only=True)

    x = np.arange(n_sys)
    width = 0.32
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 6.8), sharex=True, gridspec_kw={"hspace": 0.38})

    # Panel A: Recall@3, MRR
    ax = axes[0]
    r1 = [summary.loc[s, "recall_at_k"] for s in systems]
    r2 = [summary.loc[s, "mrr"] for s in systems]
    ax.bar(x - width / 2, r1, width, label="Recall@3", color="#264653", edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, r2, width, label="MRR", color="#2a9d8f", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Mean retrieval score (0–1)")
    ax.set_ylim(0, 1.12)
    ax.set_title("(A) Retrieval: Recall@3 and mean reciprocal rank (MRR)")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#cccccc")

    # Panel B: Personalization, Groundedness
    ax = axes[1]
    p1 = [summary.loc[s, "personalization_score"] for s in systems]
    p2 = [summary.loc[s, "groundedness_score"] for s in systems]
    ax.bar(x - width / 2, p1, width, label="Personalization", color="#e76f51", edgecolor="white", linewidth=0.4)
    ax.bar(x + width / 2, p2, width, label="Groundedness", color="#7209b7", edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Mean rubric score (1–5)")
    ax.set_ylim(0, 5.6)
    ax.set_title("(B) Personalization and groundedness (heuristic rubric)")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#cccccc")

    # Panel C: Latency
    ax = axes[2]
    lat = [summary.loc[s, "latency_ms"] for s in systems]
    bars = ax.bar(x, lat, width=0.55, color=PALETTE[:n_sys], edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS_SHORT[s] for s in systems])
    ax.set_title("(C) End-to-end latency")
    ymax = max(lat) * 1.15 if lat else 1.0
    ax.set_ylim(0, ymax)
    for bar, v in zip(bars, lat):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

    fig.suptitle(
        "Figure 1. System comparison on the evaluation benchmark\n"
        "(macro mean over queries; latency is end-to-end per condition).",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # tight_layout conflicts with sharex on some backends; manual margins are stable
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.07, hspace=0.42)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def figure2_family_performance(family_csv: Path, out: Path) -> None:
    """Recall@3 and MRR by query family × system."""
    fam = pd.read_csv(family_csv)
    systems = [s for s in SYSTEM_ORDER if s in set(fam["system_name"])]
    families = [f for f in FAMILY_ORDER if f in set(fam["category"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W * 1.35, 3.8), sharey=False)

    n_f = len(families)
    n_s = len(systems)
    x = np.arange(n_f)
    total_w = 0.78
    bar_w = total_w / n_s
    offsets = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_s)

    for si, sys in enumerate(systems):
        sub = fam[fam["system_name"] == sys].set_index("category")
        recall_vals = [float(sub.loc[c, "recall_at_k"]) if c in sub.index else 0.0 for c in families]
        mrr_vals = [float(sub.loc[c, "mrr"]) if c in sub.index else 0.0 for c in families]
        ax1.bar(x + offsets[si], recall_vals, bar_w, label=SYSTEM_LABELS_LEGEND[sys], color=PALETTE[si], edgecolor="white", linewidth=0.35)
        ax2.bar(x + offsets[si], mrr_vals, bar_w, label=SYSTEM_LABELS_LEGEND[sys], color=PALETTE[si], edgecolor="white", linewidth=0.35)

    for ax, title, ylab in (
        (ax1, "(A) Recall@3 by query family", "Mean Recall@3"),
        (ax2, "(B) MRR by query family", "Mean MRR"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in families])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.12)
        ax.set_title(title)

    h, l = ax1.get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )
    fig.suptitle(
        "Figure 2. Retrieval performance by query family and system condition.",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.24, wspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def figure3_ablation(df: pd.DataFrame, out: Path) -> None:
    """Full multimodal vs ablation: personalization, groundedness (1–5), latency (ms)."""
    systems = ["full_multimodal_agent", "ablation_no_injury"]
    rubric_cols = ["personalization_score", "groundedness_score", "latency_ms"]
    summary = df[df["system_name"].isin(systems)].groupby("system_name")[rubric_cols].mean(numeric_only=True)

    w = 0.36
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 5.4), sharex=False, gridspec_kw={"hspace": 0.45})

    x_r = np.arange(2)
    full_r = [summary.loc["full_multimodal_agent", "personalization_score"], summary.loc["full_multimodal_agent", "groundedness_score"]]
    abl_r = [summary.loc["ablation_no_injury", "personalization_score"], summary.loc["ablation_no_injury", "groundedness_score"]]
    ax1.bar(x_r - w / 2, full_r, w, label="Full multimodal agent", color="#2a9d8f", edgecolor="white", linewidth=0.4)
    ax1.bar(x_r + w / 2, abl_r, w, label="Ablation (no injury tools)", color="#e9c46a", edgecolor="white", linewidth=0.4)
    for i, (vf, va) in enumerate(zip(full_r, abl_r)):
        ax1.text(i - w / 2, vf + 0.12, f"{vf:.2f}", ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")
        ax1.text(i + w / 2, va + 0.12, f"{va:.2f}", ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")
    ax1.set_xticks(x_r)
    ax1.set_xticklabels(["Personalization", "Groundedness"])
    ax1.set_ylabel("Mean rubric (1–5)")
    ax1.set_ylim(0, 6.0)
    ax1.set_title("(A) Answer rubric (heuristic)")
    ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")

    lat_f = summary.loc["full_multimodal_agent", "latency_ms"]
    lat_a = summary.loc["ablation_no_injury", "latency_ms"]
    x0 = 0.0
    ax2.bar(x0 - w / 2, lat_f, w, label="Full multimodal agent", color="#2a9d8f", edgecolor="white", linewidth=0.4)
    ax2.bar(x0 + w / 2, lat_a, w, label="Ablation (no injury tools)", color="#e9c46a", edgecolor="white", linewidth=0.4)
    ymax = max(lat_f, lat_a) * 1.18
    ax2.set_ylim(0, ymax)
    ax2.text(x0 - w / 2, lat_f + ymax * 0.02, f"{lat_f:.0f}", ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")
    ax2.text(x0 + w / 2, lat_a + ymax * 0.02, f"{lat_a:.0f}", ha="center", va="bottom", fontsize=7.5, color="#1a1a1a")
    ax2.set_xticks([x0])
    ax2.set_xticklabels(["End-to-end latency"])
    ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="#cccccc")

    fig.suptitle(
        "Figure 3. Ablation study: full multimodal agent vs. no-injury variant\n"
        "(same backbone; injury routing and tools removed in ablation).",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.08, hspace=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def figure4_failure_table(df: pd.DataFrame, out: Path) -> None:
    """Curated failure cases with short excerpts (from benchmark CSV)."""
    # Rows hand-picked for clear failure modes; IDs match results.csv
    cases: list[tuple[str, str, str, str, str, str]] = []

    def row(qid: str, system: str) -> pd.Series:
        r = df[(df["query_id"] == qid) & (df["system_name"] == system)]
        if r.empty:
            raise ValueError(f"missing {qid} {system}")
        return r.iloc[0]

    # Wrong retrieval → wrong coaching narrative (pulling context for a pushing question)
    r = row("analytical_002", "full_multimodal_agent")
    cases.append(
        (
            "analytical_002",
            _trunc(r["query"], 90),
            _trunc(str(r["retrieved_top3"]), 80),
            _trunc(str(r.get("final_response", "")), 200),
            "Wrong exercise retrieval",
            "Top-3 hits are pull-dominant rows; answer rationalizes with Military Press / face pulls instead of benchmark pushing IDs.",
        )
    )

    # Irrelevant “similar movement” suggestion despite recall@3=1
    r = row("cross_modal_002", "full_multimodal_agent")
    cases.append(
        (
            "cross_modal_002",
            _trunc(r["query"], 90),
            _trunc(str(r["retrieved_top3"]), 80),
            _trunc(str(r.get("final_response", "")), 200),
            "Distractor recommendation",
            "Retrieval is cable-row–centric; answer adds a non-analogous modality (e.g., treadmill walking) as a “similar” movement.",
        )
    )

    # Ablation contradicts knee-safe narrative
    r = row("personal_001", "ablation_no_injury")
    cases.append(
        (
            "personal_001",
            _trunc(r["query"], 100),
            _trunc(str(r["retrieved_top3"]), 80),
            _trunc(str(r.get("final_response", "")), 220),
            "Safety / contraindication",
            "States legs should be avoided for knee history, then prescribes heavy back squats—injury-aware path in the full agent avoids this pattern.",
        )
    )

    col_labels = ["Query ID", "Query (excerpt)", "Retrieved top-3 (excerpt)", "Answer (excerpt)", "Failure category", "Notes"]
    cell_text = [[cases[i][j] for j in range(6)] for i in range(len(cases))]

    fig, ax = plt.subplots(figsize=(FIG_W * 1.55, 4.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colColours=["#e8eef2"] * 6,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.05, 2.1)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_height(0.08)
        else:
            cell.set_height(0.16)
    ax.set_title(
        "Figure 4. Qualitative failure analysis (full multimodal agent and ablation).\n"
        "Examples drawn from logged queries in results.csv.",
        fontsize=10,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "data/eval/results.csv")
    parser.add_argument("--family", type=Path, default=ROOT / "data/eval/family_results.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    parser.add_argument(
        "--with-failure-table",
        action="store_true",
        help="Also write fig4_failure_analysis_table.png (qualitative examples).",
    )
    args = parser.parse_args()

    if not args.results.is_file():
        print(f"[figures] ERROR: missing {args.results}", file=sys.stderr)
        sys.exit(1)
    if not args.family.is_file():
        print(f"[figures] ERROR: missing {args.family}", file=sys.stderr)
        sys.exit(1)

    _setup_pub_style()
    df = pd.read_csv(args.results)
    for col in ("recall_at_k", "mrr", "personalization_score", "relevance_score", "groundedness_score", "latency_ms"):
        if col not in df.columns:
            df[col] = 0.0

    out_dir = args.out_dir
    figure1_overall_comparison(df, out_dir / "fig1_overall_system_comparison.png")
    figure2_family_performance(args.family, out_dir / "fig2_query_family_performance.png")
    figure3_ablation(df, out_dir / "fig3_ablation_injury_memory.png")
    if args.with_failure_table:
        figure4_failure_table(df, out_dir / "fig4_failure_analysis_table.png")
    print("[figures] done.")


if __name__ == "__main__":
    main()
