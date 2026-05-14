# FitSupport — Technical Documentation

Local-first multimodal RAG fitness assistant. Ingests personal workout history, lift
progression, injury context, and exercise images, then retrieves personalised evidence
to power a coaching LLM response via a LangGraph agent.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [Data Layer](#5-data-layer)
6. [Ingestion Pipeline](#6-ingestion-pipeline)
7. [Embedding and Indexing](#7-embedding-and-indexing)
8. [Retrieval](#8-retrieval)
9. [LangGraph Agent](#9-langgraph-agent)
10. [Streamlit UI — Chat Interface](#10-streamlit-ui--chat-interface)
11. [LLM Integration](#11-llm-integration)
12. [Evaluation](#12-evaluation)
13. [Reproducibility](#13-reproducibility)
14. [Known Limitations and Future Work](#14-known-limitations-and-future-work)

---

## 1. Quick Start

### Prerequisites

- Python **3.13** (see `.python-version`)
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com/) running locally with `qwen2.5:7b` pulled

```bash
pip install uv                        # install uv if not present
uv sync                               # create .venv from uv.lock
ollama pull qwen2.5:7b               # pull the default VLM
```

### Full pipeline (ingest → index → eval)

```bash
uv run python run_all.py
```

### UI only

```bash
uv run streamlit run ui/app.py
```

### Evaluation only

```bash
uv run python src/evaluation/run_evaluation.py
```

---

## 2. Architecture Overview

```
User Query + Optional Image
         │
         ▼
   ┌─────────────┐
   │  QueryRouter │  (rule-based, deterministic)
   └──────┬──────┘
          │ fan-out based on query_type
    ┌─────┼─────────────────────┐
    │     │                     │
    ▼     ▼                     ▼
TextRetrieval  ImageRetrieval  InjuryLookup + ProgressionAnalysis
    │     │                     │
    └─────┴─────────┬───────────┘
                    ▼
             ContextFusion
                    │
                    ▼
              Generation  ──► Ollama (qwen2.5:7b)
                    │
                    ▼
            Final Response
```

**Retrieval stores:**

| Collection | Model | Content |
|---|---|---|
| `fitness_text` | `all-MiniLM-L6-v2` | Exercise metadata + lift history |
| `fitness_images` | `clip-ViT-B/32` | Exercise demo images |

**Query routing:**

| Route | Nodes activated |
|---|---|
| `factual_retrieval` | `text_retrieval` |
| `cross_modal` | `image_retrieval` |
| `analytical` | `text_retrieval` + `injury_lookup` + `progression_analysis` |
| `personalized_followup` | `text_retrieval` + `injury_lookup` + `progression_analysis` |

---

## 3. Project Structure

```
fit_support/
├── config/
│   └── config.yaml              # Central config (models, top_k, chroma paths)
├── data/
│   ├── raw/
│   │   ├── images/              # Exercise demo photos (nested by exercise name)
│   │   ├── injuries/            # Injury JSON files
│   │   ├── lifts/
│   │   │   └── strength.csv     # Personal PRs and baselines
│   │   ├── metadata/
│   │   │   └── exercise_library.csv
│   │   ├── workouts/
│   │   │   └── workout_log.csv  # Dated session logs
│   │   └── user_profile.json    # Coaching philosophy + injury context
│   ├── processed/               # Validated clean CSVs (output of ingestion)
│   ├── chroma/                  # Persistent ChromaDB vector store (gitignored)
│   └── eval/
│       ├── results.csv          # Evaluation output
│       └── rejected_rows.csv    # Rows that failed ingestion validation
├── src/
│   ├── agent/
│   │   ├── graph.py             # LangGraph StateGraph + Ollama integration
│   │   ├── router.py            # Rule-based query classifier
│   │   ├── state.py             # AgentState TypedDict
│   │   └── tools.py             # InjuryMemoryTool, StrengthProgressionTool
│   ├── embeddings/
│   │   ├── text_embedder.py
│   │   ├── image_embedder.py
│   │   └── index_builder.py
│   ├── evaluation/
│   │   └── run_evaluation.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── models.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   └── validators.py
│   ├── retrieval/
│   │   └── search.py
│   └── config.py                # Typed config loader (lru_cache)
├── tests/
│   └── benchmark_queries.json
├── ui/
│   └── app.py                   # Streamlit chat UI
├── run_all.py
└── pyproject.toml
```

---

## 4. Configuration

All runtime parameters live in `config/config.yaml`:

```yaml
embeddings:
  text_model: sentence-transformers/all-MiniLM-L6-v2
  image_model: clip-ViT-B-32

retrieval:
  top_k: 3
  recall_k: 3

chroma:
  text_collection: fitness_text
  image_collection: fitness_images
  persist_directory: data/chroma

llm:
  primary_model: qwen2.5:7b       # VLM — handles image + text queries
  secondary_model: llama3.1:8b    # Text-only fallback / eval comparison
  ollama_base_url: http://localhost:11434

evaluation:
  random_seed: 42
  results_path: data/eval/results.csv
  rejected_rows_path: data/eval/rejected_rows.csv
```

`src/config.py` loads this once at import time via `@lru_cache(maxsize=1)` and exposes
the `cfg` and `user_profile` constants imported by every module.

**Why `qwen2.5:7b` as primary?**
Llama 3.1 is a text-only model. `qwen2.5:7b` is a vision-language model (VLM), meaning
it can handle image inputs properly for cross-modal queries. It's also competitive on
instruction-following at the 7B scale.

---

## 5. Data Layer

### Raw inputs

| Path | Purpose |
|---|---|
| `data/raw/lifts/strength.csv` | Personal bests / PR rows (`exercise_name`, `best_weight_kg`, `best_reps`, `notes`) |
| `data/raw/workouts/workout_log.csv` | Dated session logs (`date`, `exercise_name`, `set_number`, `weight_kg`, `reps`, `notes`) |
| `data/raw/metadata/exercise_library.csv` | Canonical exercise names and attributes |
| `data/raw/injuries/*.json` | Injury records with body part, severity, and mandatory protocols |
| `data/raw/images/<exercise_name>/` | Demo photos (`start.jpeg`, `mid.jpeg`, `finish.jpeg`) |
| `data/raw/user_profile.json` | Jeff Nippard methodology, coaching principles, and current injury state |

### Processed outputs

The ingestion pipeline writes validated, normalised snapshots to `data/processed/`:

- `exercises_clean.csv`
- `lifts_clean.csv`
- `workouts_clean.csv`
- `injuries_clean.csv`

Rows failing validation land in `data/eval/rejected_rows.csv` with structured error context.

---

## 6. Ingestion Pipeline

Entry point: `uv run python main.py --data-pipeline`

**Stages:**

```
Raw CSVs / JSONs
      │
      ▼ [LOAD]     loaders.py     — scan raw dirs, parse files
      ▼ [PREPROCESS] preprocess.py — normalise whitespace, coerce types,
      │                              fuzzy-map exercise names to library
      ▼ [VALIDATE] validators.py  — Pydantic + extra checks
      │                              (non-negative weight/reps, library
      │                               membership, duplicate key detection)
      ├──► accepted rows → data/processed/*.csv
      └──► rejected rows → data/eval/rejected_rows.csv
```

The pipeline is lazy-imported so `--data-pipeline` does not pull in the full
LangGraph/embedding stack, keeping the CLI lightweight.

---

## 7. Embedding and Indexing

Entry point: `uv run python src/embeddings/index_builder.py`

### Text index (`fitness_text`)

Model: `sentence-transformers/all-MiniLM-L6-v2`

Records embedded:
- Exercise metadata from `exercise_library.csv` (name, movement pattern, equipment, muscle groups)
- Lift history from `strength.csv` (formatted as natural-language text per record)

**Why `all-MiniLM-L6-v2`?** Lightweight (80 MB), fast CPU inference, strong
semantic performance at sentence level. Dataset size (63 text records) makes a
larger model unnecessary.

### Image index (`fitness_images`)

Model: `clip-ViT-B/32` via `sentence-transformers`

Records embedded: all `*.jpg`/`*.jpeg`/`*.png` files found recursively under
`data/raw/images/`, keyed by `<exercise_name>/<frame>`.

**Why `clip-ViT-B/32`?** CLIP aligns images and text in the same embedding space,
enabling image→image and image→text similarity without a separate text query.
`ViT-L/14` is marginally stronger but at the current 21-image scale the delta is
negligible; B/32 fits in 16 GB RAM without a dedicated GPU.

Each rebuild:
1. Resets `fitness_text` and `fitness_images` collections
2. Re-embeds and upserts all records
3. Verifies expected counts and absence of duplicate IDs

---

## 8. Retrieval

Module: `src/retrieval/search.py`

### Key functions

```python
search_exercise_by_text(query: str, top_k: int, chroma_path: str)
    → list[dict]   # exercise metadata + document string

search_lift_records_by_text(query: str, top_k: int, chroma_path: str)
    → list[dict]   # lift history records filtered by record_type == "lift_record"

search_similar_exercise_image(image_path: Path, top_k: int, chroma_path: str)
    → list[dict]   # image records ordered by CLIP embedding similarity

get_images_by_exercise_label(exercise_name: str, chroma_path: str, limit: int)
    → list[dict]   # exact-match image lookup for demo frame display
```

Each result contains: `id`, similarity score, `document` (text), `metadata`
(exercise name, movement pattern, equipment, image label, source path).

### ChromaDB initialisation

`graph.py` pre-warms the ChromaDB `SharedSystemClient` at module import time.
This prevents the race condition that occurs when LangGraph's parallel fan-out
(text + injury + progression nodes) all try to open the same persistent DB
simultaneously.

---

## 9. LangGraph Agent

Module: `src/agent/graph.py`

### State

`AgentState` (TypedDict) fields:

| Field | Type | Reducer |
|---|---|---|
| `query` | `str` | — |
| `query_type` | `str` | — |
| `image_path` | `Optional[str]` | — |
| `retrieved_text_context` | `List[str]` | ADD (append) |
| `retrieved_image_context` | `List[str]` | ADD (append) |
| `injury_context` | `List[str]` | ADD (append) |
| `progression_context` | `List[str]` | ADD (append) |
| `tool_calls_log` | `List[str]` | ADD (append) |
| `final_response` | `str` | — |
| `conversation_history` | `Optional[List[Dict]]` | — (read-only, set at init) |

`conversation_history` carries the last N chat turns from the UI. It is set
once in `run_graph()` and only read by `_build_user_prompt` — no node writes to
it, so it deliberately has no ADD reducer.

### Node execution path

```
START
  │
  router_node          — classifies query → query_type
  │
  (conditional fan-out)
  ├── text_retrieval_node      — Chroma fitness_text + demo images
  ├── image_retrieval_node     — Chroma fitness_images via CLIP
  ├── injury_lookup_node       — InjuryMemoryTool
  └── progression_analysis_node— StrengthProgressionTool / Chroma lift filter
         │ (all parallel branches join here)
  context_fusion_node  — no-op join point
  │
  generation_node      — builds prompt, calls Ollama
  │
  END
```

### Generation system prompt

The generation node uses a coaching-tone system prompt:

> "You are the user's personal training partner who knows their full history.
> Be direct and conversational — like a coach texting a friend, not writing a report.
> Maximum 3-4 sentences. No bullet points. Lead with the recommendation. Reference
> their real numbers casually. Never say 'No injury conflicts were found'. Sound like
> you actually know them."

This replaced an earlier clinical bullet-point style. The new prompt includes
good/bad response examples to anchor the model's output format.

### User prompt assembly

`_build_user_prompt(state)` produces the user turn in this order:

1. **Previous messages** (if `conversation_history` is non-empty):
   ```
   Previous messages:
   User: <msg>
   Assistant: <msg>
   ...
   Use this context for follow-up questions.
   ```
2. **Query**
3. **Retrieved exercises** (text context)
4. **User's personal bests** (progression context) with explicit directive to
   cite exact numbers
5. **Injury context**
6. **Image context**

### Entry points

```python
# Default — uses cfg.llm.primary_model
run_graph(query, image_path=None, conversation_history=None) → AgentState

# Explicit model override — used by UI and evaluation harness
run_graph_with_model(query, model, image_path=None, top_k=None,
                     conversation_history=None) → AgentState
```

Both entry points reset model/top_k overrides in a `finally` block so eval runs
can't leak state.

---

## 10. Streamlit UI — Chat Interface

Entry point: `uv run streamlit run ui/app.py`

### Features

#### Multi-turn conversation memory

- All messages are stored in `st.session_state.messages` as
  `{"role": "user"|"assistant", "content": str, ...}` dicts.
- Previous turns are rendered above the input using `st.chat_message()`.
- The last 6 messages (3 user + 3 assistant = 3 full exchanges) are passed to
  `run_graph_with_model` as `conversation_history`, enabling follow-up questions
  like *"ok what about today?"* without re-stating the exercise.

#### Image persistence

- When a user uploads an image, its bytes are stored in
  `st.session_state.last_image_bytes`.
- On subsequent turns where no new image is uploaded, the stored bytes are
  written to a fresh temp file and passed as `image_path` to the graph.
- This means *"what weight should I use for that?"* two messages after an
  image upload still resolves the image correctly.

#### Input auto-clear

- Both the text input and file uploader use a key suffixed with
  `st.session_state.input_counter`.
- After each submission `input_counter` is incremented and `st.rerun()` is
  called — Streamlit treats the new key as a fresh widget, clearing both fields.

#### Clear chat button

- A "Clear" button in the top-right resets `st.session_state.messages`,
  `last_image_bytes`, and increments `input_counter`.

#### Retrieved image display

- After each assistant response, exercise demo images returned by the graph are
  rendered in a column layout.
- Demo images are also stored in the message dict so they re-render when the
  conversation history is replayed on rerun.

### State flow per submission

```
User submits query + (optional) image
        │
        ├── resolve image_path (new upload OR stored bytes)
        ├── slice last 6 messages → conv_history
        ├── append user message to session_state.messages
        │
        ├── run_graph_with_model(query, model, image_path, conv_history)
        │         └── LangGraph agent executes → returns AgentState
        │
        ├── append assistant message (response + retrieved_images) to messages
        ├── increment input_counter
        └── st.rerun()
```

---

## 11. LLM Integration

Module: `src/agent/graph.py` — `call_ollama()`

All LLM calls go through `call_ollama()`, a thin `urllib` wrapper over
Ollama's `/api/chat` endpoint. No third-party HTTP client is used to keep
the dependency surface minimal.

- **Timeout:** 300 s (covers cold-start model loading for 4–7 GB models).
- **Failure modes handled:**
  - `URLError` (Ollama not running) — sets `_ollama_confirmed_down = True`
    to skip further attempts in the same process.
  - `TimeoutError` / `OSError` — returns error string without blacklisting.
  - `KeyError` / `JSONDecodeError` — parse error returned as string.
- **Model selection:** `_get_model()` returns the active override if set,
  otherwise `cfg.llm.primary_model`.

### Model choices

| Role | Model | Reason |
|---|---|---|
| Primary (default) | `qwen2.5:7b` | VLM — handles image + text queries; strong instruction following |
| Secondary (eval / fallback) | `llama3.1:8b` | Text-only; used for evaluation comparison runs |

---

## 12. Evaluation

Module: `src/evaluation/run_evaluation.py`

### Benchmark

- File: `tests/benchmark_queries.json`
- 11 queries covering all four route types.

### Conditions (ablation)

| Condition | Description |
|---|---|
| `plain_llm_baseline` | No retrieval; LLM answers from parametric knowledge only |
| `text_only_retrieval` | Chroma `fitness_text` search only |
| `full_multimodal_agent` | Full graph: routing + text + image + injury + progression |
| `ablation_no_injury` | Full agent minus injury context (isolates injury contribution) |

### Metrics

| Metric | Description |
|---|---|
| `Recall@3` | Proportion of expected exercise names in top-3 retrieved records |
| `Response relevance (1–5)` | Heuristic derived from Recall@3 |
| `Personalization score (1–5)` | Rewards use of strength/injury/image evidence |
| `Latency (ms)` | Wall-clock time per query per condition |

### Latest results

| Condition | Recall@3 | Relevance | Personalization | Latency ms |
|---|---:|---:|---:|---:|
| Plain LLM baseline | 0.000 | 2.000 | 1.000 | 0 |
| Text-only retrieval | 0.606 | 3.818 | 3.000 | 46 |
| Full multimodal agent | 0.636 | 3.909 | 4.455 | 118 |

The full agent gains +5% Recall@3 and +1.5 personalization points over text-only
at the cost of ~72 ms additional latency from extra parallel tool calls.

### LLM-as-judge

Scores were produced by the same Ollama model used for generation.
**Known limitation:** the judge and generation model share an architecture,
which may inflate scores through self-preference bias. Future work should use a
separate judge model or human raters.

### Ground truth

Relevance labels for Recall@k were defined manually by the author based on
domain knowledge of the personal dataset. This introduces annotator bias;
scores should be treated as indicative rather than definitive.

---

## 13. Reproducibility

### Environment

```bash
pip install uv
uv sync          # installs exact versions from uv.lock — do not modify uv.lock manually
```

To add a new dependency:

```bash
uv add <package>    # updates pyproject.toml and uv.lock atomically
```

### Full pipeline

```bash
uv run python run_all.py
```

Steps executed in order:

1. Load and validate `config/config.yaml`
2. Run data ingestion pipeline → `data/processed/*.csv`
3. Rebuild ChromaDB index → `data/chroma/`
4. Run 4-condition evaluation → `data/eval/results.csv`
5. Print results summary table

### Expected outputs

| Path | Description |
|---|---|
| `data/processed/exercises_clean.csv` | Validated exercise metadata |
| `data/processed/lifts_clean.csv` | Validated lift records |
| `data/processed/workouts_clean.csv` | Validated session logs |
| `data/processed/injuries_clean.csv` | Validated injury records |
| `data/chroma/` | Persistent ChromaDB (gitignored; rebuilt from raw) |
| `data/eval/results.csv` | 44 rows (11 queries × 4 conditions) |

---

## 14. Known Limitations and Future Work

### Current limitations

| # | Limitation |
|---|---|
| 1 | **Small dataset** — 63 text records, 21 images. Retrieval diversity is constrained. |
| 2 | **Self-defined ground truth** — Recall@k labels have annotator bias. |
| 3 | **LLM judge bias** — generation and judge share an architecture. |
| 4 | **No reranker** — pure vector search occasionally surfaces adjacent-but-irrelevant results. |
| 5 | **Eager model loading** — `EmbeddingService` loads models at import time; lazy/cached loading would reduce CLI startup latency. |
| 6 | **No persistent cross-session memory** — `conversation_history` is held in Streamlit session state only; cleared on browser refresh. |

### Planned improvements

- Cross-encoder reranker after initial vector retrieval
- Larger image dataset for meaningful CLIP retrieval depth
- Human evaluation alongside LLM-as-judge
- Persistent conversation memory (database-backed, cross-session)
- Fine-tuned embeddings on fitness-specific vocabulary
- Reciprocal-rank fusion across text and image modalities
- Lazy/cached embedding model initialisation
