# fit_support TASKS

## Current phase
- Phase 1 foundation alignment complete; entering iterative hardening.

## Status legend
- `todo`
- `in_progress`
- `done`
- `blocked`

## Checklist by phase
### Phase 1: Config, env loader, folder validation
- [done] Implement typed settings in `src/fit_support/config.py`.
- [done] Add startup validation for required folders (`data/raw/images`, `data/raw/metadata`, `data/processed`, `chroma_db`).
- [done] Add helper entrypoint in `main.py` to load config and run validation.

### Phase 2: Text + image ingestion + metadata schema
- [done] Define Pydantic models in `src/fit_support/ingestion/schemas.py`.
- [done] Define ABC ingestion contracts in `src/fit_support/ingestion/interfaces.py`.
- [done] Include `BaseIngestionSource`, `TextIngestionSource`, `ImageIngestionSource`, and `MetadataRepository`.
- [done] Keep loader stubs available for mapping text/image data to validated records.

### Phase 3: Embedding + Chroma storage
- [done] Add text embedding wrapper.
- [done] Add image embedding wrapper.
- [done] Create Chroma repository layer for upsert/query.

### Phase 4: Retrieval engine + similarity search
- [done] Implement retrieval service to embed query, search, merge, and rank.
- [done] Ensure retrieval output remains explainable (id/source/score/metadata).

### Phase 5: Demo CLI query interface
- [done] Add CLI command for ingestion and query usage in `main.py`.

## Blockers
- No labeled benchmark dataset yet for robust retrieval metric tuning.

## Bugs
- None currently identified.

## Technical debt
- Retrieval fusion is basic and should be upgraded (e.g., weighted or reciprocal-rank fusion).
- Embedding model load can be made lazy/cached to improve cold start.

## Decision log
- Standardized on `src/fit_support/config.py` for plan compliance.
- Kept local-first storage with persistent Chroma collections.
- Preserved modular split between ingestion, embedding, retrieval, and evaluation components.

## Data hygiene (completed)
- [done] Inspect CSV inputs under `data/raw/lifts/`, `data/raw/workouts/`, `data/raw/metadata/`.
- [done] Split baseline/PR rows into `data/raw/lifts/strength.csv` (schema: `exercise_name`, `best_weight_kg`, `best_reps`, `notes`).
- [done] Create `data/raw/workouts/workout_log.csv` for dated session sets only.
- [done] Normalize `UNKNOWN` to empty, preserve bodyweight context in notes, map names to `exercise_library.csv`.
- [done] Post-split validation warnings for non-library exercise names.
- [done] Document data layout in `README.md` and archive legacy `lifts_log.csv` under `data/processed/migrations/`.

## Phase 1 — Tabular ingestion pipeline (completed)
- [done] Add `src/ingestion/` package: `models.py`, `validators.py`, `loaders.py`, `preprocess.py`, `pipeline.py`, `__init__.py`.
- [done] Pydantic models with validation messages for exercises, lifts, workouts, injuries.
- [done] Row-level validation: missing fields, duplicates, library name checks, negative numbers, date parsing.
- [done] Load raw CSV/JSON from lifts, workouts, injuries, metadata directories.
- [done] Preprocess: name normalization, casing/whitespace, nulls, date coercion.
- [done] Write `data/processed/{exercises,lifts,workouts,injuries}_clean.csv` and `data/eval/rejected_rows.csv`.
- [done] Tagged logging `[LOAD]` / `[VALIDATE]` / `[CLEAN]` / `[SAVE]`; resilient to single bad rows.
- [done] CLI entry: `uv run python main.py --data-pipeline`.

## Phase 2 — Multimodal embedding + retrieval (completed)
- [done] Create `src/embeddings/{__init__.py,text_embedder.py,image_embedder.py,index_builder.py}`.
- [done] Create `src/retrieval/{__init__.py,search.py}`.
- [done] Build text embeddings from metadata + lift history with `all-MiniLM-L6-v2`.
- [done] Build image embeddings from nested folders under `data/raw/images/` using CLIP-compatible model.
- [done] Create and populate Chroma collections `fitness_text` and `fitness_images` under `data/chroma/`.
- [done] Implement retrieval APIs: `search_exercise_by_text(query)` and `search_similar_exercise_image(image_path)`.
- [done] Execute required search smoke tests and image retrieval validation during indexing.
- [done] Keep diagnostic logs `[EMBED TEXT]`, `[EMBED IMAGE]`, `[INDEX]`, `[SEARCH]`.
- [done] Reset managed Chroma collections before rebuild to avoid stale/incorrect records.
- [done] Verify `fitness_text` and `fitness_images` collection counts and duplicate-free IDs after indexing.
