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
        Accumulated text documents from semantic search (all retrieval branches).
    exercise_context_records : List[Dict[str, Any]]
        Raw Chroma-style exercise rows (metadata + document) merged from text and
        image retrieval; used in context_fusion for muscle-aware filtering.
    generation_retrieved_text : Optional[List[str]]
        Exercise document strings after context_fusion filtering; generation prefers
        this over retrieved_text_context when set (non-greeting paths).
    retrieved_image_context : List[str]
        Accumulated image documents from CLIP search.
    show_images : bool
        Whether the UI should render image results for this turn.
    matched_exercise_name : Optional[str]
        Exercise name used in prompts when an image match is trusted (generation node).
    identified_exercise : Optional[str]
        Best exercise label from the image pipeline (CLIP or NN); None if unknown.
    exercise_confidence : Optional[float]
        CLIP zero-shot classification confidence [0, 1] for the matched exercise.
        None when no image was uploaded or the fallback image-embedding path was used.
    image_identification_note : Optional[str]
        Set by image_retrieval_node when CLIP confidence is too low to trust the match.
        Contains the "could not confidently identify" message passed to generation.
    node_timings : Dict[str, float]
        Per-node runtime in milliseconds for the current graph invocation.
    recall_at_3 : Optional[float]
        Live Recall@3 when ground truth is available; None during normal UI chat.
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
    skip_injury_lookup : bool
        When True, ``route_by_query_type`` omits ``injury_lookup`` even for
        injury-keyword personalized queries (evaluation ablation only).
    """

    query: str
    query_type: str
    image_path: Optional[str]
    retrieved_text_context: Annotated[List[str], operator.add]
    exercise_context_records: Annotated[List[Dict[str, Any]], operator.add]
    generation_retrieved_text: Optional[List[str]]
    retrieved_image_context: Annotated[List[str], operator.add]
    show_images: bool
    matched_exercise_name: Optional[str]
    identified_exercise: Optional[str]
    exercise_confidence: Optional[float]
    image_identification_note: Optional[str]
    node_timings: Annotated[Dict[str, float], operator.or_]
    recall_at_3: Optional[float]
    injury_context: Annotated[List[str], operator.add]
    progression_context: Annotated[List[str], operator.add]
    tool_calls_log: Annotated[List[str], operator.add]
    final_response: str
    conversation_history: Optional[List[Dict[str, Any]]]
    skip_injury_lookup: bool
