# fit_support Engineering Tasks

## Status legend
- `todo`
- `in_progress`
- `done`
- `blocked`

## Phase 1 - Config setup
- [done] Config setup (`src/fit_support/config/settings.py`)
- [done] Environment loader
- [done] Folder validation

## Phase 2 - Ingestion foundations
- [done] Text ingestion pipeline (`workouts`, `lifts`, `injuries`)
- [done] Image ingestion pipeline (`images`)
- [done] Metadata schema (`ContextChunk`)

## Phase 3 - Embedding + vector storage
- [done] Embedding pipeline (text + image)
- [done] Chroma vector storage and upsert

## Phase 4 - Retrieval engine
- [done] Retrieval engine scaffold
- [done] Similarity search across modality collections

## Phase 5 - Demo interface
- [done] Demo CLI query interface

## Blockers
- Need representative raw data files to benchmark retrieval quality.

## Bugs
- None currently identified.

## Technical debt
- Retrieval ranking is basic score fusion and should be upgraded with tuned reranking.
