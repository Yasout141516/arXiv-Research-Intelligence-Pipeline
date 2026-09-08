"""Central configuration: paths and tunables shared across pipeline stages.

Paths derive from the repository root rather than being hardcoded, so the
pipeline runs unchanged on any machine after a `git clone`.

This module holds what more than one stage needs. Settings owned by a single
stage live with that stage — the chart theme in `visualize`, the check
thresholds in `quality`, the harness timings in `evaluate`.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env here so every entrypoint picks it up by importing config.
load_dotenv()

# --- Locations ---------------------------------------------------------
# config.py lives at <root>/src/arxiv_pipeline/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plots"
EVAL_DIR = PROJECT_ROOT / "evaluation"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

RAW_CSV = DATA_DIR / "kaggle_arxiv.csv"
RAW_JSON = DATA_DIR / "papers_raw.json"
DB_PATH = DATA_DIR / "arxiv.db"

QUESTIONS_PATH = EVAL_DIR / "questions.json"
ANSWERS_PATH = EVAL_DIR / "answers.json"

# --- Ingestion --------------------------------------------------------
RAW_TABLE = "raw_papers"
SAMPLE_SIZE = 5000
CHUNK_ROWS = 10_000
MIN_RECORDS = 500

# Papers are kept if they are cross-listed in any of these. Note that a kept
# paper's *primary* category can still be something else (stat.ML is common),
# which is why the plot palette and this list are deliberately different sets.
TARGET_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.RO", "cs.NE", "cs.CL"]

# Column names in the final schema, in order. Every column is TEXT;
# `primary_category` is derived during ingestion rather than read from the CSV.
PAPER_COLUMNS = [
    "arxiv_id",
    "title",
    "abstract",
    "authors",
    "categories",
    "primary_category",
    "submitted",
    "updated",
    "journal_ref",
    "doi",
    "comment",
]
DERIVED_COLUMNS = ["primary_category"]

# Kaggle snapshot column -> our schema.
COLUMN_RENAMES = {
    "id": "arxiv_id",
    "comments": "comment",
    "journal-ref": "journal_ref",
    "update_date": "updated",
}

# Only parse the columns we keep — the snapshot carries three we never read.
_TO_SNAPSHOT_NAME = {ours: theirs for theirs, ours in COLUMN_RENAMES.items()}
RAW_CSV_USECOLS = [
    _TO_SNAPSHOT_NAME.get(column, column)
    for column in PAPER_COLUMNS
    if column not in DERIVED_COLUMNS
]

# Submission years outside this band are data errors, not history. Shared by
# the transform (which excludes them) and the quality check (which counts them).
MIN_YEAR = 1990
MAX_YEAR = 2026

# --- RAG --------------------------------------------------------------
COLLECTION_NAME = "arxiv_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
MODEL_LABEL = f"{GROQ_MODEL} + {EMBED_MODEL}"

RAG_SAMPLE_SIZE = 1000
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
EMBED_BATCH_SIZE = 500
DEFAULT_N_RESULTS = 5


def groq_api_key() -> str | None:
    """Read the key at call time, so a server picks up a rotated value."""
    return os.getenv("GROQ_API_KEY")
