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
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.router import QueryRoute, QueryRouter
from agent.state import AgentState
from agent.tools import InjuryMemoryTool, StrengthProgressionTool
from config import cfg, user_profile
from retrieval.search import search_exercise_by_text, search_similar_exercise_image

_CHROMA_PATH = cfg.chroma.persist_path

# ── LLM prompts ────────────────────────────────────────────────────────────────

_COACHING_PHILOSOPHY = (
    "This user follows Jeff Nippard training methodology. "
    "For every exercise recommendation, apply these principles:\n"
    "- Prioritise the stretch position at the bottom of the movement\n"
    "- Always cue controlled negatives (eccentric phase)\n"
    "- Full ROM where injury permits\n"
    "- Never give generic beginner cues — this is an experienced lifter\n\n"
    "Injury context must always override exercise recommendations."
)

_SYSTEM_PROMPT = (
    "You are my personal fitness coach with access to the user's workout "
    "history, injury context, and exercise library. Always:\n"
    "- Reference specific exercises from the retrieved context\n"
    "- Respect injury limitations explicitly\n"
    "- Reference the user's actual progression data when available\n"
    "- Never recommend exercises that conflict with injury flags\n"
    "- Be specific, not generic\n\n"
    + _COACHING_PHILOSOPHY
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
        return "\n".join(items) if items else "None"

    return (
        f"Query: {state['query']}\n\n"
        f"Retrieved exercises: {_fmt(state.get('retrieved_text_context', []))}\n"
        f"Injury context: {_fmt(state.get('injury_context', []))}\n"
        f"Progression data: {_fmt(state.get('progression_context', []))}\n"
        f"Image context: {_fmt(state.get('retrieved_image_context', []))}\n\n"
        "Provide a specific, personalized recommendation."
    )


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
    """Retrieve top-k exercise documents via semantic text search against fitness_text."""
    records = search_exercise_by_text(
        state["query"],
        top_k=_get_top_k(),
        chroma_path=_CHROMA_PATH,
    )
    return {
        "retrieved_text_context": _extract_docs(records),
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
    """Retrieve personal strength-progression records relevant to the query."""
    result = StrengthProgressionTool().run(state["query"], top_k=_get_top_k())
    return {
        "progression_context": _extract_docs(result.records),
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
        return ["text_retrieval", "injury_lookup"]
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


def run_graph(query: str, image_path: str | None = None) -> AgentState:
    """Run the compiled graph for a single query and return the final AgentState.

    Parameters
    ----------
    query : str
        User question.
    image_path : str | None
        Optional path to a query image for cross_modal queries.
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
    }
    return compiled_graph.invoke(initial)


def run_graph_with_model(
    query: str,
    model: str,
    image_path: str | None = None,
    top_k: int | None = None,
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
    """
    set_active_model(model)
    set_top_k_override(top_k)
    try:
        return run_graph(query, image_path=image_path)
    finally:
        set_active_model(None)
        set_top_k_override(None)
