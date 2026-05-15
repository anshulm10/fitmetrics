# FitSupport — Personalised Multimodal Fitness Agent

## Overview

FitSupport is a local-first assistant that answers strength-training questions using retrieval over personal workout logs, an exercise library, injury notes, and exercise imagery. The research question is whether **multimodal, personalised retrieval** (text + images + injury + progression context) improves grounded, user-specific answers compared with a plain LLM or text-only RAG. The working hypothesis is that **routing retrieval by query type** and fusing modality-specific evidence before generation improves relevance and personalization without sacrificing safety constraints encoded in the user profile.

## Architecture

The agent is implemented as a **LangGraph** `StateGraph`. A router classifies each user turn (e.g. greeting, factual lookup, cross-modal image query, analytical progression question, personalised follow-up). Retrieval nodes run in parallel where the graph fans out; their outputs are merged in **context_fusion**, then a single **generation** node calls the configured LLM (e.g. Ollama or Gemini) with a fixed system prompt plus user profile and coaching philosophy.

```text
User Query → LangGraph Router → [ text_retrieval | image_retrieval | injury_lookup | progression_analysis ] → context_fusion → generation → Response
```

**Image path:** `image_retrieval` first attempts a near-duplicate match in Chroma image embeddings, then runs CLIP zero-shot classification (`clip_classifier.py`); low-confidence cases fall back to similarity search and may set `image_identification_note` instead of guessing an exercise.

## Data

| Source | Format | Records |
|--------|--------|--------|
| Exercise library | CSV | 63 |
| Workout logs | CSV | 13 lifts |
| Exercise images | JPG/PNG | 21+ |
| Injury context | JSON | 3 conditions |

Paths under `data/raw/` and cleaned snapshots under `data/processed/`; vectors live in local Chroma under `data/chroma/`.

## Embedding Models

- **Text:** `all-MiniLM-L6-v2` (sentence-transformers) — chosen for speed and memory use on a typical 16GB RAM laptop.
- **Image:** CLIP **ViT-B/32** (`openai/clip-vit-base-patch32` in the classifier; Chroma image index uses CLIP-compatible embeddings) — zero-shot exercise labels plus similarity search over indexed frames.

## Evaluation Results

Aggregated metrics by query family and system condition (`data/eval/family_results.csv`):

| category | system_name | recall_at_k | mrr | relevance_score | personalization_score | groundedness_score | latency_ms |
|----------|---------------|------------:|----:|----------------:|----------------------:|-------------------:|-----------:|
| analytical | ablation_no_injury | 0.333 | 0.5 | 3.0 | 5.0 | 4.5 | 17314.135 |
| analytical | full_multimodal_agent | 0.333 | 0.5 | 3.0 | 5.0 | 4.5 | 21261.605 |
| analytical | plain_llm_baseline | 0.0 | 0.0 | 2.0 | 1.0 | 1.0 | 0.0 |
| analytical | text_only_retrieval | 0.333 | 0.5 | 3.0 | 3.0 | 2.0 | 19.86 |
| cross_modal | ablation_no_injury | 1.0 | 1.0 | 5.0 | 5.0 | 4.5 | 27263.38 |
| cross_modal | full_multimodal_agent | 1.0 | 1.0 | 5.0 | 5.0 | 4.5 | 28230.865 |
| cross_modal | plain_llm_baseline | 0.0 | 0.0 | 2.0 | 1.0 | 1.0 | 0.0 |
| cross_modal | text_only_retrieval | 1.0 | 1.0 | 5.0 | 3.0 | 2.0 | 221.0 |
| factual_retrieval | ablation_no_injury | 1.0 | 1.0 | 5.0 | 3.5 | 3.0 | 13439.54 |
| factual_retrieval | full_multimodal_agent | 1.0 | 1.0 | 5.0 | 3.5 | 3.5 | 16542.6 |
| factual_retrieval | plain_llm_baseline | 0.0 | 0.0 | 2.0 | 1.0 | 1.0 | 0.005 |
| factual_retrieval | text_only_retrieval | 1.0 | 1.0 | 5.0 | 3.0 | 2.0 | 47.945 |
| personalized_followup | ablation_no_injury | 1.0 | 1.0 | 5.0 | 5.0 | 4.5 | 16518.81 |
| personalized_followup | full_multimodal_agent | 1.0 | 1.0 | 5.0 | 5.0 | 4.5 | 21942.415 |
| personalized_followup | plain_llm_baseline | 0.0 | 0.0 | 2.0 | 1.0 | 1.0 | 0.0 |
| personalized_followup | text_only_retrieval | 0.833 | 1.0 | 4.5 | 3.0 | 2.0 | 19.95 |

Benchmark queries live in `tests/benchmark_queries.json`; the evaluation driver is `src/evaluation/run_evaluation.py`. The **plain** and **text-only** conditions each perform a **real LLM completion** (no LangGraph for plain; no full graph for text-only) so `final_response` and groundedness heuristics are meaningful; the **ablation** condition disables **injury** in both the compiled graph routing and the tool merge used for metrics.

## Reproducibility

```bash
uv sync
uv run python run_all.py
uv run streamlit run ui/app.py --server.maxUploadSize 10
```

Python version is pinned in `.python-version`. Commit `uv.lock` is authoritative; use `uv sync` (not ad-hoc `pip install`) for a reproducible environment. A quick graph smoke test: `uv run python test_agent.py`.

## Appendix A: Optional LLM-as-judge prompt

The harness in `src/evaluation/run_evaluation.py` writes **relevance_score** from retrieval recall and **personalization_score** / **groundedness_score** from deterministic heuristics (see the module docstring). The prompt below is suitable for a **separate** human or LLM judge study; it is **not** what the CSV columns are computed from unless you add a second scoring pass.

The following prompt was designed to evaluate `relevance_score` and `personalization_score` for each system response **if** you run an external judge:

```
You are an expert fitness coach and evaluation assistant.
You will be given a user query, a set of retrieved exercise context,
and a system response. Score the response on two dimensions.

Query: {query}
Retrieved Context: {retrieved_context}
Response: {response}

Score 1 — Relevance (1-5):
Does the response directly address what the user asked for?
1 = completely off-topic
3 = partially relevant, some useful content
5 = directly and fully addresses the query

Score 2 — Personalization (1-5):
Does the response reference the user's specific data
(workout history, injury context, progression)?
1 = generic advice, no personal data referenced
3 = some personal data referenced
5 = deeply personalized, references specific user history

Reply in this exact format only:
relevance_score: <int>
personalization_score: <int>
```

**Methodology notes:** The CSV **relevance_score**, **personalization_score**, and **groundedness_score** columns come from the deterministic rules in `run_evaluation.py` (see that file’s module docstring). Appendix A is for an **optional** separate judge run (human or LLM), not the automated table. If you add a judge pass, use the same prompt across conditions; watch for self-preference if the judge model matches the generator.

## Appendix B: Ground Truth Construction

Ground truth relevance labels for Recall@k were defined manually by the author prior to running evaluation. For each benchmark query, the set of relevant document IDs was determined from domain knowledge of the personal dataset — for example, knowing that a query about machine quad isolation should return leg extension or hack squat variants, not unrelated machines.

**Limitations:** The same person who built the system defined relevance, which introduces **annotator bias**. A stricter setup would use independent annotators or blind labelling and report inter-annotator agreement (e.g. Cohen's Kappa). Recall@k should be read as indicative, not definitive.

## Limitations

1. **Small corpus** — Dozens of text records and on the order of twenty images limit retrieval diversity; scaling the library would stress-test embeddings and routing more fairly.

2. **Author-defined ground truth** — As in Appendix B, retrieval labels are not independently verified.

3. **Single-domain, single-user style data** — Results may not generalize to other sports or populations.

4. **Optional LLM judge** — The default CSV scores are heuristics; a separate judge study (Appendix A) would introduce its own bias trade-offs.

5. **No cross-encoder reranking** — First-stage vector retrieval can surface semantically adjacent but wrong exercises; a reranker would be a natural upgrade.

6. **Hardware** — Models were chosen for CPU / 16GB-class laptops; larger encoders might improve margins on bigger data.

Future work: human evaluation, reranking, richer image metadata, and multi-annotator benchmarks.
