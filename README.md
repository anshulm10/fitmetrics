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
