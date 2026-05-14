"""
FitSupport — Personal Training Assistant  (Streamlit UI)

Run with:
    uv run streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import re
from pathlib import Path

_UI_DIR = Path(__file__).resolve().parent
_ROOT = _UI_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.agent.graph import run_graph_with_model
from src.agent.router import QueryRoute, QueryRouter
from src.config import cfg

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FitSupport",
    page_icon="🏋️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS overrides ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* ── Layout ── */
    .block-container {
        max-width: 720px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #141414;
        border: 1px dashed #2e2e2e;
        border-radius: 10px;
        padding: 6px 10px;
    }
    [data-testid="stFileUploaderDropzone"] { background: transparent !important; }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background: #141414 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        font-size: 1rem !important;
        padding: 0.6rem 0.85rem !important;
    }
    .stTextInput > div > div > input::placeholder { color: #555 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #3b5bdb !important;
        box-shadow: 0 0 0 2px rgba(59,91,219,0.18) !important;
    }

    /* ── Submit button ── */
    div[data-testid="column"]:not(:last-child) .stButton > button {
        background: #2563eb !important;
        border: none !important;
        border-radius: 8px !important;
        color: #fff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1.5rem !important;
        width: 100% !important;
        transition: background 0.15s ease;
    }
    div[data-testid="column"]:not(:last-child) .stButton > button:hover { background: #1d4ed8 !important; }
    div[data-testid="column"]:not(:last-child) .stButton > button:active { background: #1e40af !important; }

    /* ── Clear button ── */
    div[data-testid="column"]:last-child .stButton > button {
        background: transparent !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 8px !important;
        color: #555 !important;
        font-size: 0.8rem !important;
        padding: 0.25rem 0.75rem !important;
        transition: all 0.15s ease;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        border-color: #555 !important;
        color: #888 !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.25rem 0 !important;
    }

    /* ── Response card (border container) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #141414 !important;
        border: 1px solid #242424 !important;
        border-radius: 12px !important;
        padding: 0.25rem 0.5rem !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #111 !important;
        border: 1px solid #1e1e1e !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #555 !important; font-size: 0.8rem !important; }

    /* ── Meta strip ── */
    .meta-strip {
        color: #484848;
        font-size: 0.72rem;
        margin-top: 0.45rem;
        letter-spacing: 0.015em;
        font-family: monospace;
    }

    /* ── Dividers ── */
    hr { border-color: #1e1e1e !important; margin: 0.6rem 0 !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: #555 !important; font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0
# Persist the last uploaded image across turns so follow-up questions still
# have image context even after the file uploader is cleared on rerun.
if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None
if "last_image_suffix" not in st.session_state:
    st.session_state.last_image_suffix = ".jpg"


def _format_ms(ms: float | int | None) -> str:
    if ms is None:
        return "N/A"
    return f"{float(ms):,.0f}ms"


def _format_seconds(ms: float | int | None) -> str:
    """Render a millisecond duration as compact seconds, e.g. 18234ms → '18.2s'."""
    if ms is None:
        return "N/A"
    return f"{float(ms) / 1000:.1f}s"


def _context_preview(items: list[str], limit: int = 3) -> str:
    if not items:
        return "None"
    preview: list[str] = []
    for item in items[:limit]:
        text = str(item).strip()
        first_line = text.splitlines()[0] if text else ""
        preview.append(first_line[:120])
    return ", ".join(preview) if preview else "None"


def _tool_chain(tools: list[str]) -> str:
    labels = {
        "text_retrieval": "text",
        "image_retrieval": "image",
        "injury_lookup": "injury",
        "progression_analysis": "progression",
        "context_fusion": "fusion",
    }
    return " → ".join(labels.get(tool, tool) for tool in tools) if tools else "none"


def _retrieval_counts(state: dict) -> str:
    """Return a compact 'text:N progression:N' string, hiding any count == 0.

    A tool that didn't fire (or fired but returned no results) should never
    appear in the metrics bar — e.g. don't show 'injury:0' when injury_lookup
    didn't run on this query.
    """
    counts = {
        "text": len(state.get("retrieved_text_context", []) or []),
        "images": len(state.get("retrieved_image_context", []) or []),
        "progression": len(state.get("progression_context", []) or []),
        "injury": len(state.get("injury_context", []) or []),
    }
    parts = [f"{name}:{n}" for name, n in counts.items() if n > 0]
    return "  ".join(parts) if parts else "no retrieval"


def _exercise_names(items: list[str], limit: int = 3) -> str:
    names: list[str] = []
    for item in items:
        text = str(item)
        match = re.search(r"exercise name:\s*([^\.]+)", text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
        else:
            match = re.search(r"lift history for\s*([^\.]+)", text, flags=re.IGNORECASE)
            name = match.group(1).strip() if match else text.split(":", 1)[0].strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return ", ".join(names) if names else "None"


def _short_progression(items: list[str], limit: int = 2) -> str:
    cleaned: list[str] = []
    for item in items[:limit]:
        text = str(item).strip()
        text = text.replace("best_weight_kg=", "").replace("best_reps=", "x ")
        text = text.replace(", notes=", ", ")
        cleaned.append(text[:140])
    return ", ".join(cleaned) if cleaned else "None"


def _wants_movement_frames(query: str) -> bool:
    q = query.lower()
    return "show me the movement" in q or "show me the frames" in q


def _render_metrics_bar(msg: dict) -> None:
    metrics = msg.get("metrics") or {}
    state = msg.get("state") or {}
    session_queries = len([m for m in st.session_state.messages if m.get("role") == "assistant"])
    bar = (
        f"⏱ {_format_seconds(metrics.get('total_latency_ms'))}"
        f" · q:{session_queries}"
        f" · {_tool_chain(msg.get('tool_calls_log', []))}"
        f" · {_retrieval_counts(state)}"
    )
    st.markdown(f'<p class="meta-strip">{bar}</p>', unsafe_allow_html=True)


def _render_details(msg: dict) -> None:
    metrics = msg.get("metrics") or {}
    state = msg.get("state") or {}
    node_timings = state.get("node_timings", {})
    if not state and not node_timings:
        return
    ordered_nodes = [
        "router",
        "injury_lookup",
        "text_retrieval",
        "image_retrieval",
        "progression_analysis",
        "context_fusion",
        "generation",
    ]
    labels = {
        "progression_analysis": "progression",
    }
    with st.expander("Details", expanded=False):
        lines: list[str] = []
        lines.append("Node timings:")
        seen: set[str] = set()
        for name in ordered_nodes:
            if name in node_timings:
                label = labels.get(name, name)
                lines.append(f"{label:<16} {_format_ms(node_timings[name])}")
                seen.add(name)
        for name, ms in node_timings.items():
            if name not in seen and name != "total":
                lines.append(f"{name:<16} {_format_ms(ms)}")
        if "total" in node_timings or metrics.get("total_latency_ms") is not None:
            lines.append(f"{'total':<16} {_format_ms(node_timings.get('total', metrics.get('total_latency_ms')))}")
        lines.extend([
            "",
            "Retrieved:",
            f"Exercises: {_exercise_names(state.get('retrieved_text_context', []))}",
            f"Injury flags: {_context_preview(state.get('injury_context', []), limit=2)}",
            f"Progression: {_short_progression(state.get('progression_context', []), limit=2)}",
        ])
        st.code("\n".join(lines), language="text")


def _render_assistant_debug(msg: dict) -> None:
    _render_metrics_bar(msg)
    _render_details(msg)


def _render_session_stats() -> None:
    assistant_messages = [m for m in st.session_state.messages if m.get("role") == "assistant"]
    if not assistant_messages:
        return
    latencies = [
        float(m["metrics"]["total_latency_ms"])
        for m in assistant_messages
        if m.get("metrics", {}).get("total_latency_ms") is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    tool_counts: dict[str, int] = {}
    for msg in assistant_messages:
        for tool in msg.get("tool_calls_log", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    tools_text = " ".join(f"{name}({count})" for name, count in tool_counts.items()) or "none"
    st.caption(
        f"This session: {len(assistant_messages)} queries · "
        f"avg latency {_format_seconds(avg_latency)} · "
        f"tools fired: {tools_text}"
    )

# ── Header row (title + clear button) ─────────────────────────────────────────

header_col, clear_col = st.columns([6, 1])
with header_col:
    st.markdown("## FitSupport")
    st.caption(
        f"Jeff Nippard methodology · injury-aware · "
        f"`{cfg.llm.provider}:{cfg.llm.active_model_name}` · top-k {cfg.retrieval.top_k}"
    )
with clear_col:
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    if st.button("Clear", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.last_image_bytes = None
        st.session_state.last_image_suffix = ".jpg"
        st.session_state.input_counter += 1
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Conversation history display ───────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_debug(msg)
        if msg.get("image_bytes"):
            with st.expander("Uploaded image", expanded=False):
                st.image(msg["image_bytes"], use_container_width=True)
        if msg.get("show_images") and msg.get("retrieved_images"):
            state = msg.get("state") or {}
            can_show_images = state.get("query_type") == QueryRoute.CROSS_MODAL.value or msg.get("image_uploaded")
            if not can_show_images:
                continue
            valid = [p for p in msg["retrieved_images"] if p and Path(p).is_file()]
            if valid:
                if msg.get("show_frames"):
                    cols = st.columns(min(len(valid), 3))
                    for col, p in zip(cols, valid[:3]):
                        with col:
                            st.image(p, use_container_width=True)
                            st.caption(Path(p).parent.name.replace("_", " ").title())
                else:
                    p = valid[0]
                    st.image(p, use_container_width=True)
                    st.caption(Path(p).parent.name.replace("_", " ").title())

# ── Input area ─────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Exercise image — optional (triggers image similarity search)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
    key=f"uploader_{st.session_state.input_counter}",
)

if uploaded_file is not None:
    st.image(uploaded_file, use_container_width=True)

query = st.text_input(
    "query",
    placeholder="What should I load on leg press today?",
    label_visibility="collapsed",
    key=f"query_{st.session_state.input_counter}",
)

ask_col, _ = st.columns([5, 1])
with ask_col:
    submitted = st.button("Ask", type="primary")

# ── Guard ──────────────────────────────────────────────────────────────────────

if submitted and not query.strip():
    st.warning("Enter a question before submitting.")

# ── Run ────────────────────────────────────────────────────────────────────────

if submitted and query.strip():
    # Resolve image: prefer freshly uploaded file, fall back to persisted bytes.
    image_path: str | None = None
    image_bytes_for_history: bytes | None = None
    is_visual_query = QueryRouter().route(query.strip()).route == QueryRoute.CROSS_MODAL

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        img_bytes = uploaded_file.getvalue()
        st.session_state.last_image_bytes = img_bytes
        st.session_state.last_image_suffix = suffix
        image_bytes_for_history = img_bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(img_bytes)
            image_path = tmp.name
    elif is_visual_query and st.session_state.last_image_bytes is not None:
        # Recreate temp file from stored bytes so image context persists.
        suffix = st.session_state.last_image_suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(st.session_state.last_image_bytes)
            image_path = tmp.name

    # Build conversation history from the last 3 exchanges (6 messages).
    recent = st.session_state.messages[-6:]
    conv_history = [{"role": m["role"], "content": m["content"]} for m in recent]
    print(f"[FitSupport] conversation_history={conv_history}", flush=True)

    # Append user message to history immediately so it shows on rerun.
    st.session_state.messages.append({
        "role": "user",
        "content": query.strip(),
        "image_bytes": image_bytes_for_history,
    })

    with st.spinner("Thinking…"):
        t0 = time.perf_counter()
        state = run_graph_with_model(
            query.strip(),
            model=cfg.llm.model,
            top_k=cfg.retrieval.top_k,
            image_path=image_path,
            conversation_history=conv_history,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

    # Collect exercise demo images returned by the graph.
    image_paths = state.get("retrieved_image_context", []) or []
    show_images = bool(state.get("show_images", False))
    query_type = state.get("query_type", "")
    image_uploaded = uploaded_file is not None
    show_frames = _wants_movement_frames(query)
    can_show_images = show_images and (query_type == QueryRoute.CROSS_MODAL.value or image_uploaded)
    valid_paths_all = [p for p in image_paths if p and Path(p).is_file()] if can_show_images else []
    valid_paths = valid_paths_all[:3] if show_frames else valid_paths_all[:1]
    node_timings = state.get("node_timings", {}) or {}
    tools_log = state.get("tool_calls_log", [])
    response_metrics = {
        "total_latency_ms": node_timings.get("total", latency_ms),
        "tools_count": len(tools_log),
    }

    # Build meta string for assistant message.
    tools_str = " → ".join(tools_log) if tools_log else "—"
    meta = f'<p class="meta-strip">{latency_ms:.0f} ms &nbsp;·&nbsp; {tools_str}</p>'

    # Append assistant message.
    st.session_state.messages.append({
        "role": "assistant",
        "content": state["final_response"],
        "retrieved_images": valid_paths,
        "show_images": show_images,
        "image_uploaded": image_uploaded,
        "show_frames": show_frames,
        "metrics": response_metrics,
        "tool_calls_log": tools_log,
        "meta": meta,
        "state": {
            "retrieved_text_context": state.get("retrieved_text_context", []),
            "injury_context": state.get("injury_context", []),
            "progression_context": state.get("progression_context", []),
            "retrieved_image_context": state.get("retrieved_image_context", []),
            "show_images": show_images,
            "query_type": query_type,
            "matched_exercise_name": state.get("matched_exercise_name"),
            "node_timings": node_timings,
        },
    })

    # Increment counter to clear the text input and file uploader on rerun.
    st.session_state.input_counter += 1
    st.rerun()

_render_session_stats()
