"""Stage 1 — ingest the raw Kaggle arXiv snapshot into JSON + SQLite.

Reads the snapshot in chunks (it is far larger than memory), keeps only the
target CS categories, dedupes, samples, and lands the result in `raw_papers`.
"""
import pandas as pd

from . import config
from .db import connect


def raw_schema() -> str:
    """DDL for the landing table, generated from the configured schema."""
    columns = ",\n        ".join(
        f"{name} TEXT" + (" PRIMARY KEY" if name == "arxiv_id" else "")
        for name in config.PAPER_COLUMNS
    )
    return f"CREATE TABLE {config.RAW_TABLE} (\n        {columns}\n    );"


def load_filtered_papers() -> pd.DataFrame:
    """Stream the snapshot, keeping rows cross-listed in any target category."""
    pattern = "|".join(config.TARGET_CATEGORIES)
    chunks = [
        chunk[chunk["categories"].str.contains(pattern, na=False)]
        for chunk in pd.read_csv(
            config.RAW_CSV,
            chunksize=config.CHUNK_ROWS,
            usecols=config.RAW_CSV_USECOLS,
            dtype={"id": str},
        )
    ]
    return pd.concat(chunks, ignore_index=True)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map to our schema, derive primary_category, drop dupes and null ids."""
    df = df.rename(columns=config.COLUMN_RENAMES)
    df["primary_category"] = df["categories"].str.split().str[0]
    df = df[config.PAPER_COLUMNS]
    df = df.drop_duplicates(subset=["arxiv_id"])
    return df[df["arxiv_id"].notna()].reset_index(drop=True)


def run() -> None:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw snapshot not found at {config.RAW_CSV}. "
            "See the README for download instructions."
        )

    print("Reading CSV in chunks...")
    df = normalise(load_filtered_papers())
    print(f"  {len(df):,} unique papers in target categories")

    if len(df) < config.MIN_RECORDS:
        raise ValueError(
            f"Only {len(df)} records found — need at least {config.MIN_RECORDS}."
        )

    sample = df.sample(
        n=min(config.SAMPLE_SIZE, len(df)), random_state=42
    ).reset_index(drop=True)
    print(f"  Sampled {len(sample):,} records")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_json(config.RAW_JSON, orient="records", indent=2)
    print(f"  Saved JSON -> {config.RAW_JSON.name}")

    with connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {config.RAW_TABLE};")
        conn.execute(raw_schema())
        sample.to_sql(config.RAW_TABLE, conn, if_exists="append", index=False)
    print(f"  Saved DB   -> {config.DB_PATH.name} (table: {config.RAW_TABLE})")

    print("\nPrimary category breakdown:")
    print(sample["primary_category"].value_counts().to_string())


if __name__ == "__main__":
    run()
