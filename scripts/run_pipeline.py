"""Run the pipeline end to end, or a single stage.

    python scripts/run_pipeline.py              # ingest -> transform -> quality -> visualize
    python scripts/run_pipeline.py transform quality
    python scripts/run_pipeline.py --list
"""
import argparse
import sys
from typing import Callable, NamedTuple

from arxiv_pipeline import ingest, quality, transform, visualize
from arxiv_pipeline.quality import QualityCheckFailed


class Stage(NamedTuple):
    description: str
    run: Callable[[], None]
    default: bool = True


def build_vector_store() -> None:
    """Deferred so the default pipeline never imports the embedding stack."""
    from arxiv_pipeline.rag import build_vector_store as build

    build()


STAGES = {
    "ingest": Stage("Load the raw snapshot into SQLite", ingest.run),
    "transform": Stage("Build the cleaned analysis tables", transform.run),
    "quality": Stage("Validate the cleaned tables", quality.run),
    "visualize": Stage("Render the analysis charts", visualize.run),
    "embed": Stage("Build the RAG vector store", build_vector_store, default=False),
}

DEFAULT_STAGES = [name for name, stage in STAGES.items() if stage.default]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # No `default=` here: with nargs="*", argparse validates a list default
    # against `choices` as if it were a single value, which always fails.
    parser.add_argument(
        "stages", nargs="*", choices=list(STAGES),
        help=f"stages to run (default: {' '.join(DEFAULT_STAGES)})",
    )
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    if args.list:
        for name, stage in STAGES.items():
            default = "" if stage.default else "  (not run by default)"
            print(f"  {name:<12} {stage.description}{default}")
        return 0

    for name in args.stages or DEFAULT_STAGES:
        stage = STAGES[name]
        print(f"\n{'=' * 60}\n{name.upper()}: {stage.description}\n{'=' * 60}")
        try:
            stage.run()
        except QualityCheckFailed as exc:
            # Stop before charts and embeddings are built on bad data.
            print(f"\n{exc}\nStage '{name}' failed. Stopping.")
            return 1

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
