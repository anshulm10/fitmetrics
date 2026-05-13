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

import sys
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
from config import cfg
from retrieval.search import search_exercise_by_text, search_similar_exercise_image

_CHROMA_PATH = cfg.chroma.persist_path


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
        top_k=cfg.retrieval.top_k,
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
    records = search_similar_exercise_image(path, top_k=cfg.retrieval.top_k, chroma_path=_CHROMA_PATH)
    return {
        "retrieved_image_context": _extract_docs(records),
        "tool_calls_log": ["image_retrieval"],
    }


def injury_lookup_node(state: AgentState) -> dict:
    """Load injury-memory records that match body-part terms in the query."""
    result = InjuryMemoryTool().run(state["query"], top_k=cfg.retrieval.top_k)
    return {
        "injury_context": _extract_docs(result.records),
        "tool_calls_log": ["injury_lookup"],
    }


def progression_analysis_node(state: AgentState) -> dict:
    """Retrieve personal strength-progression records relevant to the query."""
    result = StrengthProgressionTool().run(state["query"], top_k=cfg.retrieval.top_k)
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
    """Compose a final response string from all accumulated context fields.

    In production this node calls an LLM; here it produces a deterministic
    template that is used by the evaluation harness.
    """
    parts: list[str] = []
    for bucket in (
        state.get("retrieved_text_context", []),
        state.get("retrieved_image_context", []),
        state.get("injury_context", []),
        state.get("progression_context", []),
    ):
        parts.extend(bucket)

    if parts:
        context_block = "\n".join(f"- {p[:200]}" for p in parts[:9])
        response = f"Based on retrieved context:\n{context_block}\n\nQuery: {state['query']}"
    else:
        response = f"No relevant context found for: {state['query']}"

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

    for retrieval_node in ("text_retrieval", "image_retrieval", "injury_lookup", "progression_analysis"):
        graph.add_edge(retrieval_node, "context_fusion")

    graph.add_edge("context_fusion", "generation")
    graph.add_edge("generation", END)

    return graph.compile()


# Module-level compiled graph — import and call .invoke() directly.
compiled_graph = build_graph()


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
