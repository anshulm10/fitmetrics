"""
FitSupport — Personal Fitness Agent  (Streamlit UI)

Run with:
    uv run streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Resolve project root so imports work regardless of cwd.
_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from agent.graph import check_ollama, run_graph_with_model
from config import cfg

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FitSupport — Personal Fitness Agent",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    model = st.selectbox(
        "Model",
        options=[cfg.llm.primary_model, cfg.llm.secondary_model],
        index=0,
        help="Ollama model used for generation.",
    )

    query_type_override = st.selectbox(
        "Query type",
        options=[
            "auto-detect",
            "factual_retrieval",
            "cross_modal",
            "analytical",
            "personalized_followup",
        ],
        index=0,
        help="Force a routing decision or let the router decide automatically.",
    )

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=5,
        value=cfg.retrieval.top_k,
        help="Number of documents retrieved per modality.",
    )

    show_context = st.toggle("Show retrieved context", value=True)

    st.divider()
    st.caption(f"Ollama endpoint: `{cfg.llm.ollama_base_url}`")
    if check_ollama():
        st.success("Ollama reachable", icon="✅")
    else:
        st.error("Ollama unreachable", icon="🔴")

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("FitSupport — Personal Fitness Agent")
st.caption("Powered by LangGraph + Ollama — retrieval-augmented, injury-aware coaching.")

query = st.text_input(
    "Ask your fitness coach…",
    placeholder="e.g. What exercises can I do with a knee injury?",
    label_visibility="collapsed",
)

submitted = st.button("Submit", type="primary", use_container_width=False)

if submitted and query.strip():
    with st.spinner(f"Thinking with **{model}**…"):
        start_ts = time.perf_counter()
        state = run_graph_with_model(query.strip(), model=model, top_k=top_k)
        latency_ms = (time.perf_counter() - start_ts) * 1000

    # ── Response box ───────────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🏋️"):
        st.markdown(state["final_response"])

    # ── Metadata row ──────────────────────────────────────────────────────────
    col_lat, col_model, col_tools = st.columns(3)
    col_lat.metric("Latency", f"{latency_ms:.0f} ms")
    col_model.metric("Model", model)
    col_tools.metric("Tools fired", len(state["tool_calls_log"]))

    # ── Tool call timeline ─────────────────────────────────────────────────────
    if state["tool_calls_log"]:
        st.caption("**Execution path:** " + " → ".join(state["tool_calls_log"]))

    # ── Retrieved context expander ─────────────────────────────────────────────
    if show_context:
        context_sections: list[tuple[str, list[str]]] = [
            ("Exercise documents", state.get("retrieved_text_context", [])),
            ("Injury context", state.get("injury_context", [])),
            ("Progression data", state.get("progression_context", [])),
            ("Image context", state.get("retrieved_image_context", [])),
        ]
        any_context = any(items for _, items in context_sections)

        with st.expander("Retrieved context", expanded=False):
            if not any_context:
                st.info("No context was retrieved for this query.")
            else:
                for label, items in context_sections:
                    if items:
                        st.markdown(f"**{label}**")
                        for i, doc in enumerate(items, 1):
                            # Truncate very long docs so the UI stays readable.
                            preview = doc[:400] + ("…" if len(doc) > 400 else "")
                            st.markdown(f"{i}. {preview}")
                        st.divider()

elif submitted and not query.strip():
    st.warning("Please enter a query before submitting.")
