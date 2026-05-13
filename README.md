# fit_support

Local-first multimodal RAG fitness assistant for personalized exercise recommendations.

## Current milestone
- M1 through M5 complete.

## Task checklist
- [x] Define architecture-first folder structure and module placeholders.
- [x] Implement ingestion contracts and modality parsers.
- [x] Persist normalized records with embeddings into Chroma modality collections.
- [x] Add ingestion tests and data quality checks.
- [x] Implement multimodal retrieval merge/rerank with injury-aware filtering.
- [x] Add retrieval tests and validate retrieval relevance on sample queries.
- [x] Add baseline-vs-RAG evaluation harness.

## Blockers
- No labeled retrieval dataset yet for robust metric-driven tuning.
- Image metadata remains sparse unless naming conventions are enforced in raw data files.

## Bugs
- No known runtime bugs currently.

## Technical debt
- `EmbeddingService` instantiates models eagerly; should be cached/lazy for faster CLI startup.
- Retrieval currently performs simple modality merge; can add reciprocal-rank fusion later.
- Legacy placeholder modules under `src/ingestion`, `src/embeddings`, and `src/retrieval` should be removed once migration is confirmed.

## Decision log
- Use separate Chroma collections per modality: workouts, lifts, injuries, images.
- Use sentence-transformers for text and CLIP-compatible model path for images.
- Keep implementation local-first using filesystem + persistent Chroma database.

---

## Project Goal
Build a local-first multimodal RAG fitness assistant that ingests workout history, lift progression, injury context, and exercise images, then retrieves personalized evidence for recommendation generation.

## Architecture Decisions
- Python + uv for a reproducible local development workflow.
- ChromaDB persistent local storage for modality-specific vector collections.
- sentence-transformers for text embeddings due to strong local inference support.
- CLIP-style image embeddings for image/text retrieval compatibility.
- LangGraph retrieval orchestration to keep pipeline state and graph logic explicit.
- Shared `ContextChunk` schema to keep ingestion and retrieval interfaces consistent.

## Implementation Log
### Day 1
- What was built:
  - Architecture scaffold under `src/fit_support` (`config`, `domain`, `ingest`, `embeddings`, `retrieval`, `graph`, `eval`).
  - Data directory setup for raw, processed, chroma, and eval datasets.
  - Ingestion pipeline for workouts (`.txt`), lifts (`.csv`), injuries (`.txt`), and exercise images.
  - Embedding service and Chroma upsert/query integration with per-modality collections.
  - Retrieval service with injury-aware reranking and LangGraph workflow wrapper.
  - Evaluation rubric and baseline-vs-RAG comparison harness.
  - Test suite for ingestion, retrieval, and evaluation.
- Issues faced:
  - Initial pytest run had incomplete output due to first-time dependency install timing.
  - Deprecation warning from `datetime.utcnow()`.
  - Sparse image metadata limits retrieval quality.
- Fixes:
  - Re-ran tests with quiet mode after environment setup completed.
  - Switched timestamp default to timezone-aware `datetime.now(UTC)`.
  - Added risk tracking and decision notes to enforce metadata conventions.
- Next milestone:
  - Build labeled retrieval/evaluation dataset and offline relevance metrics (`Recall@k`, `nDCG`) before UI work.

## Current Folder Structure
- `main.py`
- `TASKS.md`
- `src/fit_support/config`
- `src/fit_support/domain`
- `src/fit_support/ingest`
- `src/fit_support/ingestion`
- `src/fit_support/embeddings`
- `src/fit_support/retrieval`
- `src/fit_support/graph`
- `src/fit_support/eval`
- `tests`
- `data/raw/workouts`
- `data/raw/lifts`
- `data/raw/images`
- `data/raw/injuries`
- `data/processed`
- `data/chroma`
- `data/eval`

## Dependencies
- `chromadb`
- `langgraph`
- `numpy`
- `opencv-python`
- `pandas`
- `pillow`
- `pydantic`
- `pydantic-settings`
- `pytest`
- `python-dotenv`
- `sentence-transformers`
- `streamlit`

## Future Improvements
- Add lazy/cached model loading to reduce startup time.
- Introduce stronger retrieval fusion/reranking and modality weighting.
- Enforce a richer image metadata schema (exercise, body part, equipment, constraints).
- Add held-out query benchmark set and automated regression checks.
- Add CLI scoring reports comparing baseline and RAG outputs across scenarios.

### Day 2
- What was built:
  - Plan-aligned `src/fit_support/config.py` module with environment-backed typed settings.
  - Required folder validation updated to include `data/raw/metadata` and `chroma_db`.
  - Added `src/fit_support/ingestion/schemas.py` with Pydantic metadata/ingested record models.
  - Expanded ingestion interfaces with `MetadataRepository` in `src/fit_support/ingestion/interfaces.py`.
  - Updated `TASKS.md` lifecycle sections (`Current phase`, `Checklist by phase`, `Blockers`, `Bugs`, `Technical debt`, `Decision log`).
- Issues faced:
  - `main.py` smoke run initially failed with `ModuleNotFoundError` for local package import.
  - LangGraph emitted a serializer deprecation warning during startup.
- Fixes:
  - Added deterministic `src` path bootstrap in `main.py` so local execution works via `uv run python main.py`.
  - Kept warning documented and scoped as non-blocking; no functional impact on ingestion/retrieval path.
- Next milestone:
  - Add metadata file loader implementation that maps structured exercise metadata directly into retrieval filters.

## Current Folder Structure (Update)
- Added `src/fit_support/config.py` (plan-compliant config module).
- Added `src/fit_support/ingestion/schemas.py`.
- Added `data/raw/metadata`.
- Added `chroma_db`.

## Data architecture

Raw lift and workout data are split by intent so the RAG pipeline can treat baselines differently from session logs.

| Path | Role |
|------|------|
| `data/raw/metadata/exercise_library.csv` | Canonical exercise names and attributes (source of truth for naming). |
| `data/raw/lifts/strength.csv` | Personal baselines, PRs, and max-effort reference rows (`exercise_name`, `best_weight_kg`, `best_reps`, `notes`). |
| `data/raw/workouts/workout_log.csv` | Dated session sets (`date`, `exercise_name`, `set_number`, `weight_kg`, `reps`, `notes`). |
| `data/processed/migrations/` | Timestamped backups of merged legacy files (e.g. `lifts_log_backup_*.csv`) before destructive splits. |
| `data/processed/*_clean.csv` | Validated, normalized snapshots produced by `src/ingestion/pipeline.py` (`--data-pipeline`). |
| `data/eval/rejected_rows.csv` | Rows that failed validation, with JSON payload and error text for debugging. |

Legacy `data/raw/lifts/lifts_log.csv` mixed baselines into a session-shaped schema; `scripts/refactor_lift_workout_data.py` moves baseline/PR rows into `strength.csv`, writes real sessions to `workout_log.csv`, maps `UNKNOWN` to empty fields, normalizes names to the library, and warns on library mismatches. Re-run with `--dry-run` to preview.

### Day 3
- What was built:
  - Automated refactor script `scripts/refactor_lift_workout_data.py` (split, clean, validate against `exercise_library.csv`).
  - `data/raw/lifts/strength.csv` and `data/raw/workouts/workout_log.csv`; legacy `lifts_log.csv` backed up then removed.
  - `WorkoutIngestor` now ingests `*.csv` session logs under `data/raw/workouts/` in addition to `*.txt`.
- Issues faced:
  - Legacy first column was mislabeled (`isdate` vs `date`); script uses the first column as session marker for classification.
- Fixes:
  - Classify rows with `BASELINE` / `PR` / baseline-related notes into strength; session-shaped rows go to `workout_log.csv`.
  - Post-write validation warns if any `exercise_name` is absent from the library.
- Next milestone:
  - Wire explicit `record_type` metadata in Chroma for strength vs session chunks if retrieval should rank them differently.

### Day 4 — Phase 1 data ingestion pipeline
- What was built:
  - New package `src/ingestion/` with `models.py` (Pydantic: `ExerciseMetadata`, `LiftRecord`, `WorkoutRecord`, `InjuryRecord`), `validators.py`, `loaders.py`, `preprocess.py`, and `pipeline.py`.
  - End-to-end flow **load → validate → preprocess → save** with tagged logs: `[LOAD]`, `[VALIDATE]`, `[CLEAN]`, `[SAVE]`.
  - Clean outputs under `data/processed/`: `exercises_clean.csv`, `lifts_clean.csv`, `workouts_clean.csv`, `injuries_clean.csv`.
  - Rejected rows collected in `data/eval/rejected_rows.csv` (schema: `phase`, `source`, `row_index`, `payload_json`, `errors`) without aborting the whole run.
  - CLI: `uv run python main.py --data-pipeline` (lazy-imports RAG stack so this path stays lightweight).
- Architecture:
  - **Loaders** scan `data/raw/lifts/*.csv`, `data/raw/workouts/*.csv`, `data/raw/injuries/*.json`, and `data/raw/metadata/*.{csv,json}`.
  - **Preprocess** normalizes whitespace, maps `UNKNOWN` to null, coerces numbers/dates, and fuzzy-maps exercise names to `exercise_library.csv`.
  - **Validators** combine Pydantic validation with extra checks (non-negative weight/reps, library membership, duplicate keys per stream).
  - **Pipeline** orchestrates row-by-row handling: bad rows go to `rejected_rows.csv`; good rows accumulate then flush to processed CSVs.
- Data flow (high level):

```mermaid
flowchart LR
    rawDirs[RawCSVandJSON] --> loadStep[Loaders]
    loadStep --> preStep[Preprocess]
    preStep --> valStep[Validators]
    valStep --> goodRows[AcceptedRows]
    valStep --> badRows[RejectedRowsCSV]
    goodRows --> procCSV[ProcessedCleanCSVs]
```

- Issues encountered:
  - Initial `main.py` always imported LangGraph-backed RAG modules, which added noise when only running the data pipeline.
  - Empty `workout_log.csv` / injury JSON files needed explicit empty CSV headers on write.
- Fixes:
  - Branch `main.py` so `--data-pipeline` only imports `ingestion.pipeline`.
  - Dedupe strength and workout streams while ingesting; write empty DataFrames with fixed column headers when no rows pass validation.
- Next milestone:
  - Point the existing Chroma / embedding ingest path at `data/processed/*.csv` as an optional second stage, or add a small adapter that reads clean files into `ContextChunk` records.

### Day 5 — Phase 2 multimodal embedding + retrieval
- What was built:
  - New multimodal modules under `src/embeddings/`: `text_embedder.py`, `image_embedder.py`, `index_builder.py`, `__init__.py`.
  - New retrieval query surface under `src/retrieval/`: `search.py`, `__init__.py`.
  - Text embedding index includes exercise metadata (`exercise_name`, `movement_pattern`, `equipment`, `muscle groups`) and lift history (`strength.csv`) in Chroma collection `fitness_text`.
  - Image embedding index ingests nested image folders under `data/raw/images/` into Chroma collection `fitness_images`.
  - Index persistence path standardized to `data/chroma/`.
- Architecture:
  - **Text model**: `sentence-transformers/all-MiniLM-L6-v2`.
  - **Image model**: `clip-ViT-B-32` via sentence-transformers (CLIP-compatible).
  - **Collections**: `fitness_text`, `fitness_images`.
  - **Stored fields**: `id`, embedding vector, metadata, and `source_path` for provenance.
- Data flow:
  - Build records from `exercise_library.csv` + `strength.csv` for text.
  - Build records from image files recursively for image embeddings.
  - Upsert both modalities to Chroma, then run retrieval smoke queries.
- Retrieval examples executed:
  - `"knee friendly quad exercise"` → top included `Hack Squat`, `Leg Extension`, `Back Squat`.
  - `"upper chest press"` → top included `Chest Press Machine`, `Incline Chest Press Machine`.
  - `"lat focused back movement"` → top included `Lat Pulldown` and row-type lift context.
  - Image query on `bent_over_row/finish.jpeg` returned same-exercise frames (`finish`, `start`, `mid`) as top matches.
- Issues encountered:
  - Initial index run was slow because search smoke tests reloaded models repeatedly.
- Fixes:
  - Added optional embedder reuse in `retrieval/search.py` and reused existing model instances in `index_builder.py`.
  - Rebuild now resets `fitness_text` and `fitness_images` before upsert, then verifies expected counts and duplicate-free IDs.
  - Generated Chroma files under `data/chroma/` are ignored by git; the local DB is rebuilt from raw sources.
- Next milestone:
  - Add reranking and modality fusion strategy (text + image + lift priors) for recommendation-time context assembly.

## Report and Interview Notes

### Embedding Models Currently Used

The project currently has two embedding code paths, but both instantiate the same model choices.

#### Active Phase 2 Embedding Pipeline

This is the main multimodal indexing path used by:

```bash
uv run python src/embeddings/index_builder.py
```

Text embedding model:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- File: `src/embeddings/text_embedder.py`
- Class: `TextEmbedder`
- Initialization:

```python
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class TextEmbedder:
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
```

Image embedding model:

- Model: `clip-ViT-B-32`
- File: `src/embeddings/image_embedder.py`
- Class: `ImageEmbedder`
- Initialization:

```python
DEFAULT_IMAGE_MODEL = "clip-ViT-B-32"

class ImageEmbedder:
    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
```

#### Older `fit_support` Embedding Service

There is also an older internal embedding service:

- File: `src/fit_support/embeddings/embedder.py`
- Class: `EmbeddingService`

It initializes models from config:

```python
class EmbeddingService:
    def __init__(self, settings: AppSettings) -> None:
        self._text_model = SentenceTransformer(settings.text_embedding_model)
        self._image_model = SentenceTransformer(settings.image_embedding_model)
```

The config defaults are:

```python
text_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
image_embedding_model = "clip-ViT-B-32"
```

So both the new Phase 2 pipeline and the older service currently use:

- Text: `sentence-transformers/all-MiniLM-L6-v2`
- Image: `clip-ViT-B-32`

### Why These Models Were Chosen

- `all-MiniLM-L6-v2` is lightweight, fast locally, and strong enough for semantic search over exercise names, movement patterns, equipment, muscle groups, coaching cues, and lift-history notes.
- `clip-ViT-B-32` is CLIP-compatible, so it can encode exercise images into vectors suitable for image similarity search.
- Both models run locally through `sentence-transformers`, which keeps the system simple and avoids relying on paid hosted embedding APIs.

### ChromaDB Collections

The Phase 2 index builder creates and persists two Chroma collections in `data/chroma/`:

- `fitness_text`: exercise metadata + lift history text embeddings.
- `fitness_images`: exercise photo embeddings from nested image folders.

Before each rebuild, `index_builder.py` resets the managed collections (`fitness_text`, `fitness_images`) and then verifies:

- expected collection count
- no duplicate IDs

This prevents stale or incorrect vectors from previous runs.

### Retrieval Functions

The main retrieval functions are in `src/retrieval/search.py`:

```python
search_exercise_by_text(query: str)
search_similar_exercise_image(image_path: str)
```

They return:

- record ID
- similarity score
- document/source path
- metadata such as exercise name, movement pattern, equipment, or image label

### Interview Explanation

This project separates ingestion, embedding, indexing, and retrieval into clear stages:

1. Raw data is cleaned into structured CSVs.
2. Exercise metadata and lift history are converted into natural-language text records.
3. Exercise images are loaded recursively from nested folders.
4. Text records are embedded with `all-MiniLM-L6-v2`.
5. Image records are embedded with `clip-ViT-B-32`.
6. Both modalities are stored in ChromaDB collections.
7. Retrieval queries search the relevant collection and return explainable metadata.

The key design decision is that Chroma stores vectors separately by modality (`fitness_text`, `fitness_images`) while preserving source metadata. This makes the system easy to debug, explain, and extend later with reranking or multimodal fusion.

## Assessment Rubric Alignment

### Research Question

Can multimodal retrieval with personalized strength and injury memory outperform a text only baseline for personalized gym coaching?

### Hypothesis

Combining exercise images, structured workout history, and injury-aware routing will improve retrieval relevance and recommendation quality compared with text-only retrieval.

### Agent Architecture

The rubric-aligned agent adds a lightweight routing and tool layer on top of the embedding/retrieval system.

```mermaid
flowchart TD
    userQuery[UserQuery] --> router[QueryRouter]
    router --> factual[FactualRetrieval]
    router --> crossModal[CrossModal]
    router --> analytical[Analytical]
    router --> personalized[PersonalizedFollowup]
    factual --> textTool[TextRetrievalTool]
    crossModal --> textTool
    crossModal --> imageTool[ImageRetrievalTool]
    analytical --> strengthTool[StrengthProgressionTool]
    analytical --> textTool
    personalized --> textTool
    personalized --> injuryTool[InjuryMemoryTool]
    personalized --> strengthTool
    textTool --> chromaText[Chroma fitness_text]
    imageTool --> chromaImages[Chroma fitness_images]
    injuryTool --> injuryFiles[RawInjuryMemory]
    strengthTool --> strengthCsv[StrengthCSV]
```

Query classes:

- `factual_retrieval`: direct exercise metadata lookup.
- `cross_modal`: image or visual similarity query.
- `analytical`: comparison/progression/synthesis query.
- `personalized_followup`: query needing injury, recovery, baseline, or user-specific context.

Tools:

- `TextRetrievalTool`: searches `fitness_text` in Chroma.
- `ImageRetrievalTool`: searches `fitness_images` in Chroma.
- `InjuryMemoryTool`: reads injury notes from `data/raw/injuries/`.
- `StrengthProgressionTool`: reads `data/raw/lifts/strength.csv` and enriches matching with exercise-library muscle/movement context.

### Evaluation Methodology

Benchmark suite:

- File: `tests/benchmark_queries.json`
- Size: 11 queries.
- Coverage:
  - factual retrieval
  - cross-modal retrieval
  - analytical synthesis
  - personalized follow-up

Evaluation script:

```bash
uv run python src/evaluation/run_evaluation.py
```

Output:

- `data/eval/results.csv`

Compared systems:

- **A. Plain LLM baseline**: deterministic no-retrieval baseline used as a lightweight proxy for generic answer behavior.
- **B. Text-only retrieval**: searches only Chroma `fitness_text`.
- **C. Full multimodal agent**: uses query routing plus text, image, injury memory, and strength progression tools.

Metrics:

- `Recall@3`: proportion of expected exercise names found in top 3 retrieved records.
- `Response relevance (1–5)`: heuristic score derived from Recall@3.
- `Personalization score (1–5)`: rewards use of strength, injury, and image evidence when appropriate.
- `Latency`: measured per query/variant in milliseconds.

Latest evaluation summary:

| Variant | Recall@3 | Response relevance | Personalization score | Latency ms |
|---|---:|---:|---:|---:|
| Plain LLM baseline | 0.000 | 2.000 | 1.000 | 0.000 |
| Text-only retrieval | 0.606 | 3.818 | 3.000 | 46.290 |
| Full multimodal agent | 0.636 | 3.909 | 4.455 | 118.351 |

### Ablation Experiments

The evaluation is structured as an ablation:

1. **No retrieval**: tests generic baseline behavior.
2. **Text-only retrieval**: tests whether metadata/lift text improves exercise grounding.
3. **Full multimodal agent**: tests whether adding query routing, image retrieval, injury memory, and strength progression improves personalized coaching.

Current result: the full multimodal agent improves average `Recall@3`, response relevance, and personalization score compared with text-only retrieval, with higher latency due to extra tool calls.

A fourth ablation condition (`ablation_no_injury`) runs the full multimodal agent but strips injury-context records from the retrieved set before scoring.  This isolates the contribution of the injury-aware retrieval path.

---

## Reproducibility

### Python version

Python **3.13** is required (see `.python-version`).

### Install dependencies

```bash
pip install uv          # install uv if not present
uv sync                 # create .venv and install all dependencies from uv.lock
```

> **Do not modify `uv.lock`.**  It is committed to the repository to guarantee
> a byte-for-byte reproducible environment.  Running `uv sync` (not `uv install`)
> uses the lock file.

### Single run command

```bash
uv run python run_all.py
```

This single command executes the full pipeline in order:

| Step | Action |
|------|--------|
| 1 | Load and validate `config/config.yaml` |
| 2 | Run data-ingestion pipeline → writes `data/processed/*.csv` |
| 3 | Rebuild ChromaDB vector index → writes `data/chroma/` |
| 4 | Run evaluation suite (4 conditions) → writes `data/eval/results.csv` |
| 5 | Print results summary table to stdout |

### Expected outputs

After a successful run you should see:

- `data/processed/exercises_clean.csv`, `lifts_clean.csv`, `workouts_clean.csv`, `injuries_clean.csv`
- `data/chroma/` — persistent ChromaDB collections (`fitness_text`, `fitness_images`)
- `data/eval/results.csv` — 44 rows (11 queries × 4 conditions) with columns:
  `system_name`, `query_type`, `recall_at_k`, `mrr`, `personalization_score`,
  `relevance_score`, `latency_ms`, `tool_calls_count`
- A summary table printed to stdout showing mean metrics per condition

### Locking policy

`uv.lock` is committed and **must not be modified manually**.  To add a new
dependency, run `uv add <package>` which updates both `pyproject.toml` and
`uv.lock` atomically.
