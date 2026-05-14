"""
End-to-end smoke test for the three required test queries.

Runs each query through the compiled graph and prints:
- routed query_type
- tool chain that fired
- retrieval counts
- final LLM response

Use this to verify Issue 1/2/3/4 fixes before the user opens the UI.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.graph import run_graph
from src.config import cfg

TEST_QUERIES = [
    "hey what can you do?",
    "what weight should I use for hack squat today?",
    "my knee really hurts today, what can I train?",
]


def _counts(state: dict) -> str:
    parts: list[str] = []
    for key, label in (
        ("retrieved_text_context", "text"),
        ("retrieved_image_context", "images"),
        ("progression_context", "progression"),
        ("injury_context", "injury"),
    ):
        n = len(state.get(key, []) or [])
        if n > 0:
            parts.append(f"{label}:{n}")
    return "  ".join(parts) if parts else "no retrieval"


def main() -> None:
    print(f"Active LLM: {cfg.llm.provider}:{cfg.llm.active_model_name}")
    print(f"Top-k: {cfg.retrieval.top_k}")
    print("=" * 80)

    for idx, q in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{idx}/{len(TEST_QUERIES)}] QUERY: {q!r}")
        print("-" * 80)
        t0 = time.perf_counter()
        state = run_graph(q)
        elapsed_s = time.perf_counter() - t0

        tools = " -> ".join(state.get("tool_calls_log", []))
        print(f"query_type : {state.get('query_type')}")
        print(f"tool_chain : {tools}")
        print(f"counts     : {_counts(state)}")
        print(f"latency    : {elapsed_s:.1f}s")
        print("response   :")
        print(state.get("final_response", "<no response>"))
        print("=" * 80)


if __name__ == "__main__":
    main()
