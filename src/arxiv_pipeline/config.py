"""Central configuration: every path and tunable in one place.

Paths are derived from the repository root rather than hardcoded, so the
pipeline runs unchanged on any machine after a `git clone`.
"""
from pathlib import Path

from dotenv import load_dotenv

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

TARGET_CATEGORIES = ["cs.AI", "cs.LG", "cs.CV", "cs.RO", "cs.NE", "cs.CL"]

# Column names in the final schema, in order.
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

# Kaggle snapshot column -> our schema.
COLUMN_RENAMES = {
    "id": "arxiv_id",
    "comments": "comment",
    "journal-ref": "journal_ref",
    "update_date": "updated",
}

# Tables built by the transform stage, in dependency order.
DERIVED_TABLES = [
    "papers",
    "category_stats",
    "yearly_trends",
    "publication_status",
    "author_stats",
]

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

# --- Presentation -----------------------------------------------------
# One colour per category, shared by every plot so the charts read as a set.
CATEGORY_COLORS = {
    "cs.AI": "#4C72B0",
    "cs.LG": "#DD8452",
    "cs.CV": "#55A868",
    "cs.CL": "#C44E52",
    "stat.ML": "#8172B2",
}
FALLBACK_COLOR = "#999999"
