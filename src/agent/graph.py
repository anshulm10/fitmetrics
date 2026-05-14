"""
LangGraph StateGraph for the multimodal fitness agent.

Graph topology
--------------
START → router
router  →  (conditional fan-out based on query_type)
    factual_retrieval      → text_retrieval
    cross_modal            → image_retrieval
    analytical             → text_retrieval + injury_lookup + progression_analysis
    personalized_followup  → text_retrieval + injury_lookup
All retrieval nodes → context_fusion → generation → END

Every node appends its own name to tool_calls_log so callers can audit the
full execution path from AgentState.tool_calls_log.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.router import QueryRoute, QueryRouter
from src.agent.state import AgentState
from src.agent.tools import InjuryMemoryTool, StrengthProgressionTool
from src.config import cfg, user_profile
from src.retrieval.search import (
    _client as _chroma_client,
    get_images_by_exercise_label,
    search_exercise_by_text,
    search_lift_records_by_text,
    search_similar_exercise_image,
)

_CHROMA_PATH = cfg.chroma.persist_path

# Pre-warm the ChromaDB client at import time so the SharedSystemClient
# global registry is populated *before* LangGraph forks parallel nodes that
# all want to read from the same persistent DB.  Without this, the parallel
# fan-out (text_retrieval + progression_analysis + injury_lookup) races on
# `chromadb._identifier_to_system` and one of the threads dies with
# `Could not connect to tenant default_tenant`.
_chroma_client(_CHROMA_PATH)

# ── LLM prompts ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are the user's personal training partner who knows their full history. "
    "Be direct and conversational — like a coach texting a friend, not writing a report.\n\n"
    "Rules:\n"
    "- Maximum 3-4 sentences per response\n"
    "- Never use bullet points or numbered lists\n"
    "- Lead with the actual recommendation immediately\n"
    "- Reference their real numbers casually e.g. 'you hit 140kg last time so try 142.5kg today'\n"
    "- If injury is relevant, weave it in naturally e.g. 'given your knee, maybe drop to 130kg "
    "and focus on the stretch'\n"
    "- Never say 'No injury conflicts were found'\n"
    "- Never say 'based on retrieved information'\n"
    "- Never end with generic disclaimers\n"
    "- Sound like you actually know them\n\n"
    "Example of BAD response: 'Weight: 140-145kg. Sets: 4. Reps: 6-7. No injury conflicts found.'\n"
    "Example of GOOD response: 'You hit 140kg x 6 last session so try 142.5kg today for 4 sets. "
    "Keep it to parallel depth given the knee — full depth hack squats with that load will "
    "aggravate it. Control the negative hard.'"
)

# ── Runtime overrides ──────────────────────────────────────────────────────────
# Use set_active_model() / set_top_k_override() to change at runtime (UI, eval).

_active_model: str | None = None
_top_k_override: int | None = None

# Once Ollama is confirmed unreachable (first failed call), we skip further
# network attempts in the same process to avoid long timeouts in batch runs.
_ollama_confirmed_down: bool = False


def set_active_model(model: str | None) -> None:
    """Override the active LLM model. Pass None to revert to cfg.llm.primary_model."""
    global _active_model
    _active_model = model


def set_top_k_override(top_k: int | None) -> None:
    """Override the retrieval top_k. Pass None to revert to cfg.retrieval.top_k."""
    global _top_k_override
    _top_k_override = top_k


def _get_model() -> str:
    return _active_model if _active_model is not None else cfg.llm.primary_model


def _get_top_k() -> int:
    return _top_k_override if _top_k_override is not None else cfg.retrieval.top_k


# ── Ollama integration ─────────────────────────────────────────────────────────

def call_ollama(
    user_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str | None = None,
    timeout: int = 60,
) -> str:
    """Call the Ollama /api/chat endpoint and return the assistant response text.

    Pass ``model`` explicitly to switch between cfg.llm.primary_model and
    cfg.llm.secondary_model — this is the designated model-comparison helper.

    Parameters
    ----------
    user_prompt : str
        Fully-formatted user turn including all retrieved context.
    system_prompt : str
        System instructions placed before the user turn.
    model : str
        Ollama model tag, e.g. "llama3.1:8b" or "qwen2.5:7b".
    base_url : str | None
        Ollama base URL; defaults to cfg.llm.ollama_base_url.
    timeout : int
        Socket timeout in seconds (default 60).  Lower values let the eval
        harness fail fast when Ollama is unreachable.
    """
    _base = base_url or cfg.llm.ollama_base_url
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    global _ollama_confirmed_down
    if _ollama_confirmed_down:
        return f"[Ollama unavailable — start Ollama and retry. Endpoint: {_base}]"

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return str(data["message"]["content"])
    except urllib.error.URLError as exc:
        # Connection refused / DNS failure — Ollama is not running at all.
        _ollama_confirmed_down = True
        return (
            f"[Ollama connection error — is Ollama running at {_base}?]\n"
            f"Error details: {exc}"
        )
    except (TimeoutError, OSError) as exc:
        # Model load timeout or transient OS error — don't permanently blacklist.
        return (
            f"[Ollama response timeout after {timeout}s — model may still be loading.]\n"
            f"Error details: {exc}"
        )
    except (KeyError, json.JSONDecodeError) as exc:
        return f"[Ollama response parse error: {exc}]"


def check_ollama(base_url: str | None = None) -> bool:
    """Return True if the Ollama endpoint is reachable (3-second probe)."""
    global _ollama_confirmed_down
    _base = base_url or cfg.llm.ollama_base_url
    req = urllib.request.Request(f"{_base}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3):
            _ollama_confirmed_down = False
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _build_user_prompt(state: AgentState) -> str:
    """Assemble the user-turn prompt from all accumulated context fields in state."""

    def _fmt(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "None"

    parts: list[str] = []

    history = state.get("conversation_history") or []
    if history:
        history_lines = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )
        parts.append(
            f"Previous messages:\n{history_lines}\n"
            "Use this context for follow-up questions."
        )

    parts.append(
        f"Query: {state['query']}\n\n"
        f"Retrieved exercises:\n{_fmt(state.get('retrieved_text_context', []))}\n\n"
        f"User's personal bests for relevant exercises:\n"
        f"{_fmt(state.get('progression_context', []))}\n\n"
        f"Use these exact numbers when recommending weight. "
        f"If bench press best is 35 kg × 6, recommend 35–37.5 kg. "
        f"Always reference actual numbers, never say 'increase gradually'.\n\n"
        f"Injury context:\n{_fmt(state.get('injury_context', []))}\n\n"
        f"Image context:\n{_fmt(state.get('retrieved_image_context', []))}\n\n"
        "Provide a specific, personalized recommendation."
    )

    return "\n\n".join(parts)


# ── private helpers ────────────────────────────────────────────────────────────

def _extract_docs(records: list[dict[str, Any]]) -> list[str]:
    """Return the document string from each retrieval record."""
    return [str(r.get("document", "")) for r in records if r.get("document")]


# ── node functions ─────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Classify the query and write query_type into state.

    Uses the rule-based QueryRouter so routing is deterministic and
    reproducible across evaluation runs.
    """
    routed = QueryRouter().route(state["query"], image_path=state.get("image_path"))
    return {
        "query_type": routed.route.value,
        "tool_calls_log": ["router"],
    }


def text_retrieval_node(state: AgentState) -> dict:
    """Retrieve top-k exercise documents and the demo image(s) for the top hit.

    After ranking text documents, we look up the top exercise's demo images
    (start/mid/finish frames) by exact ``exercise_label`` match.  This lets
    the UI render the matching exercise visually without requiring the user
    to upload an image first.
    """
    records = search_exercise_by_text(
        state["query"],
        top_k=_get_top_k(),
        chroma_path=_CHROMA_PATH,
    )

    image_docs: list[str] = []
    for rec in records:
        exercise_name = str((rec.get("metadata") or {}).get("exercise_name", "")).strip()
        if not exercise_name:
            continue
        image_records = get_images_by_exercise_label(
            exercise_name, chroma_path=_CHROMA_PATH, limit=3
        )
        if image_records:
            image_docs.extend(_extract_docs(image_records))
            break

    return {
        "retrieved_text_context": _extract_docs(records),
        "retrieved_image_context": image_docs,
        "tool_calls_log": ["text_retrieval"],
    }


def image_retrieval_node(state: AgentState) -> dict:
    """Retrieve similar exercise images via CLIP embedding search against fitness_images."""
    image_path = state.get("image_path")
    if not image_path:
        return {"retrieved_image_context": [], "tool_calls_log": ["image_retrieval"]}
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {"retrieved_image_context": [], "tool_calls_log": ["image_retrieval"]}
    records = search_similar_exercise_image(
        path, top_k=_get_top_k(), chroma_path=_CHROMA_PATH
    )
    return {
        "retrieved_image_context": _extract_docs(records),
        "tool_calls_log": ["image_retrieval"],
    }


def injury_lookup_node(state: AgentState) -> dict:
    """Load injury-memory records that match body-part terms in the query."""
    result = InjuryMemoryTool().run(state["query"], top_k=_get_top_k())
    return {
        "injury_context": _extract_docs(result.records),
        "tool_calls_log": ["injury_lookup"],
    }


def progression_analysis_node(state: AgentState) -> dict:
    """Retrieve personal strength-progression records via a ChromaDB metadata filter.

    Filters ``fitness_text`` by ``record_type == "lift_record"`` so the search
    runs over the user's personal strength corpus only — the generic exercise
    library can never crowd out their actual numbers.

    Falls back to the CSV-based ``StrengthProgressionTool`` if ChromaDB returns
    nothing (e.g. before the index has been built).
    """
    records = search_lift_records_by_text(
        state["query"],
        top_k=_get_top_k(),
        chroma_path=_CHROMA_PATH,
    )
    if not records:
        result = StrengthProgressionTool().run(state["query"], top_k=_get_top_k())
        records = result.records

    return {
        "progression_context": _extract_docs(records),
        "tool_calls_log": ["progression_analysis"],
    }


def context_fusion_node(state: AgentState) -> dict:
    """Signal that all parallel retrieval branches have completed.

    The ADD reducer on each list field means contexts are already merged by
    the time this node runs.  This node exists as an explicit join point so
    the graph topology is readable and the tool_calls_log is complete.
    """
    return {"tool_calls_log": ["context_fusion"]}


def generation_node(state: AgentState) -> dict:
    """Call Ollama to generate a personalised fitness coaching response.

    Uses _get_model() which respects any active model override, defaulting to
    cfg.llm.primary_model.  All four context buckets from AgentState are
    injected into the user prompt so the LLM has full retrieval context.
    """
    model = _get_model()
    user_prompt = _build_user_prompt(state)
    # 300 s covers cold-start model loading (4-5 GB models can take 2-3 min).
    response = call_ollama(user_prompt, _SYSTEM_PROMPT, model, timeout=300)
    return {
        "final_response": response,
        "tool_calls_log": ["generation"],
    }


# ── conditional routing ────────────────────────────────────────────────────────

def route_by_query_type(state: AgentState) -> list[str]:
    """Map query_type to the set of retrieval nodes to activate (fan-out).

    Returns a list so LangGraph runs all listed nodes in parallel within the
    same superstep.  Nodes share state via the ADD reducer.
    """
    qt = state["query_type"]
    if qt == QueryRoute.FACTUAL_RETRIEVAL:
        return ["text_retrieval"]
    if qt == QueryRoute.CROSS_MODAL:
        return ["image_retrieval"]
    if qt == QueryRoute.ANALYTICAL:
        return ["text_retrieval", "injury_lookup", "progression_analysis"]
    if qt == QueryRoute.PERSONALIZED_FOLLOWUP:
        return ["text_retrieval", "injury_lookup", "progression_analysis"]
    return ["text_retrieval"]


# ── graph construction ─────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Construct, wire, and compile the fitness LangGraph StateGraph.

    Returns the compiled graph object, ready to call .invoke() on.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("text_retrieval", text_retrieval_node)
    graph.add_node("image_retrieval", image_retrieval_node)
    graph.add_node("injury_lookup", injury_lookup_node)
    graph.add_node("progression_analysis", progression_analysis_node)
    graph.add_node("context_fusion", context_fusion_node)
    graph.add_node("generation", generation_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_by_query_type)

    for retrieval_node in (
        "text_retrieval",
        "image_retrieval",
        "injury_lookup",
        "progression_analysis",
    ):
        graph.add_edge(retrieval_node, "context_fusion")

    graph.add_edge("context_fusion", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


# Module-level compiled graph — import and call .invoke() directly.
compiled_graph = build_graph()
graph = compiled_graph  # alias for ergonomic imports: from src.agent.graph import graph


def run_graph(
    query: str,
    image_path: str | None = None,
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """Run the compiled graph for a single query and return the final AgentState.

    Parameters
    ----------
    query : str
        User question.
    image_path : str | None
        Optional path to a query image for cross_modal queries.
    conversation_history : list[dict] | None
        Recent chat exchanges for follow-up context, e.g. the last 3 pairs of
        {"role": "user"|"assistant", "content": "..."} dicts.
    """
    initial: AgentState = {
        "query": query,
        "query_type": "",
        "image_path": image_path,
        "retrieved_text_context": [],
        "retrieved_image_context": [],
        "injury_context": [],
        "progression_context": [],
        "tool_calls_log": [],
        "final_response": "",
        "conversation_history": conversation_history or [],
    }
    return compiled_graph.invoke(initial)


def run_graph_with_model(
    query: str,
    model: str,
    image_path: str | None = None,
    top_k: int | None = None,
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """Run the graph with a specific LLM model (and optional top_k override).

    This is the recommended entry point for comparing cfg.llm.primary_model
    against cfg.llm.secondary_model.  Both overrides are always reset after
    the call, even if an exception is raised.

    Parameters
    ----------
    query : str
        User question.
    model : str
        Ollama model tag to use, e.g. cfg.llm.primary_model or
        cfg.llm.secondary_model.
    image_path : str | None
        Optional path to a query image.
    top_k : int | None
        Override retrieval depth; None keeps cfg.retrieval.top_k.
    conversation_history : list[dict] | None
        Recent chat exchanges for follow-up context.
    """
    set_active_model(model)
    set_top_k_override(top_k)
    try:
        return run_graph(query, image_path=image_path, conversation_history=conversation_history)
    finally:
        set_active_model(None)
        set_top_k_override(None)
