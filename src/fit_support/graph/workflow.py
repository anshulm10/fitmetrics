from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from fit_support.config.settings import AppSettings
from fit_support.embeddings.embedder import EmbeddingService
from fit_support.retrieval.retrieve import RetrievalService
from fit_support.retrieval.vector_store import VectorStore


class RetrievalState(TypedDict):
    query: str
    results: list[dict]


def run_retrieval_workflow(query: str, settings: AppSettings) -> list[dict]:
    vector_store = VectorStore(settings)
    embedder = EmbeddingService(settings)
    retrieval = RetrievalService(settings=settings, vector_store=vector_store, embedder=embedder)

    def retrieve_node(state: RetrievalState) -> RetrievalState:
        return {"query": state["query"], "results": retrieval.retrieve(state["query"])}

    graph = StateGraph(RetrievalState)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)
    app = graph.compile()

    final_state = app.invoke({"query": query, "results": []})
    return final_state["results"]

