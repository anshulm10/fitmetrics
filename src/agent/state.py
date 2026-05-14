"""
AgentState definition for the LangGraph fitness agent.

All list fields use Annotated with operator.add so LangGraph applies the ADD
reducer — each node appends its own contribution rather than overwriting the
accumulated list.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Shared mutable state threaded through every node in the fitness graph.

    Fields
    ------
    query : str
        The raw user question.
    query_type : str
        Routing label: factual_retrieval | cross_modal | analytical |
        personalized_followup.
    image_path : Optional[str]
        Optional filesystem path to a query image (cross_modal queries).
    retrieved_text_context : List[str]
        Accumulated text documents from semantic search.
    retrieved_image_context : List[str]
        Accumulated image documents from CLIP search.
    show_images : bool
        Whether the UI should render image results for this turn.
    injury_context : List[str]
        Injury-memory records relevant to the query.
    progression_context : List[str]
        Strength-progression records for the user.
    tool_calls_log : List[str]
        Name of every node/tool that fired, in execution order.
    final_response : str
        The generated answer produced by the generation node.
    conversation_history : Optional[List[Dict[str, Any]]]
        Last N chat exchanges passed in from the UI for follow-up context.
        Each item is {"role": "user"|"assistant", "content": "..."}.
    """

    query: str
    query_type: str
    image_path: Optional[str]
    retrieved_text_context: Annotated[List[str], operator.add]
    retrieved_image_context: Annotated[List[str], operator.add]
    show_images: bool
    injury_context: Annotated[List[str], operator.add]
    progression_context: Annotated[List[str], operator.add]
    tool_calls_log: Annotated[List[str], operator.add]
    final_response: str
    conversation_history: Optional[List[Dict[str, Any]]]
