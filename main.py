import argparse
from pprint import pprint

from fit_support.config.settings import AppSettings, load_settings
from fit_support.graph.workflow import run_retrieval_workflow
from fit_support.ingest.pipeline import run_ingestion_pipeline, validate_required_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="fit_support local assistant")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion pipeline")
    parser.add_argument("--query", type=str, default="", help="Run retrieval query")
    args = parser.parse_args()

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
