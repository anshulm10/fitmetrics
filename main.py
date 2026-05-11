import argparse
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="fit_support local assistant")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline")
    parser.add_argument(
        "--data-pipeline",
        action="store_true",
        help="Phase 1: load/validate/clean raw CSV+JSON to data/processed/*.csv",
    )
    parser.add_argument("--query", type=str, default="", help="Run retrieval query")
    args = parser.parse_args()

    if args.data_pipeline:
        from ingestion.pipeline import run_data_ingestion_pipeline

        stats = run_data_ingestion_pipeline(project_root=ROOT)
        print(f"Data pipeline complete: {stats}")
        return

    from fit_support.config import AppSettings, load_settings
    from fit_support.graph.workflow import run_retrieval_workflow
    from fit_support.ingest.pipeline import run_ingestion_pipeline, validate_required_directories

    settings: AppSettings = load_settings()
    validate_required_directories(settings)
    if args.ingest:
        counts = run_ingestion_pipeline(settings)
        print(f"Ingestion complete: {counts}")
    elif args.query:
        results = run_retrieval_workflow(args.query, settings)
        pprint(results)
    else:
        print("fit_support config and directory validation passed.")


if __name__ == "__main__":
    main()
