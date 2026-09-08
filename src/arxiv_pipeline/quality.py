"""Stage 3 — validate the cleaned tables before anything downstream reads them.

Each check is a scalar SQL query plus a predicate on the number it returns.
Hard checks raise `QualityCheckFailed`, so the pipeline stops rather than
building charts and embeddings on top of bad data. Soft checks only warn, and
only once more than `WARN_THRESHOLD` of rows are affected — a handful of odd
rows in a 5,000-row sample is noise, not a broken warehouse.
"""
import sys
from dataclasses import dataclass, field
from typing import Callable

from . import config, transform
from .db import connect, scalar

# Abstracts outside this band are almost certainly truncated or malformed.
MIN_ABSTRACT_WORDS = 10
MAX_ABSTRACT_WORDS = 1000
# Fraction of rows a soft check may affect before it warns.
WARN_THRESHOLD = 0.01

EXPECTED_TABLES = tuple(transform.TABLE_QUERIES)
RECONCILES = "aggregate differs from papers by {n:,} rows"


class QualityCheckFailed(Exception):
    """Raised when one or more hard checks fail."""


def is_zero(n: int) -> bool:
    """The default predicate: the query counts offending rows, so 0 is healthy."""
    return n == 0


@dataclass
class Check:
    name: str
    sql: str
    #  Given the metric the query returns, is the table healthy?
    passes: Callable[[int], bool] = is_zero
    params: tuple = ()
    hard: bool = True
    detail: str = "{n:,} offending rows"


CHECKS = [
    Check(
        "tables present",
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN "
        f"({', '.join(['?'] * len(EXPECTED_TABLES))})",
        passes=lambda n: n == len(EXPECTED_TABLES),
        params=EXPECTED_TABLES,
        detail=f"{{n}} of {len(EXPECTED_TABLES)} expected tables found",
    ),
    Check(
        "papers not empty",
        "SELECT COUNT(*) FROM papers",
        passes=lambda n: n >= config.MIN_RECORDS,
        detail=f"{{n:,}} rows (need >= {config.MIN_RECORDS})",
    ),
    Check("no null arxiv_id", "SELECT COUNT(*) FROM papers WHERE arxiv_id IS NULL"),
    Check(
        "arxiv_id unique",
        "SELECT COUNT(*) - COUNT(DISTINCT arxiv_id) FROM papers",
        detail="{n:,} duplicate ids",
    ),
    Check(
        "no blank titles",
        "SELECT COUNT(*) FROM papers WHERE TRIM(COALESCE(title, '')) = ''",
    ),
    Check(
        "no blank abstracts",
        "SELECT COUNT(*) FROM papers WHERE TRIM(COALESCE(abstract, '')) = ''",
    ),
    Check(
        "submitted_year parsed",
        "SELECT COUNT(*) FROM papers WHERE submitted_year IS NULL",
        hard=False,
        detail="{n:,} rows with an unparseable submitted date",
    ),
    Check(
        "submitted_year in range",
        "SELECT COUNT(*) FROM papers WHERE submitted_year IS NOT NULL "
        f"AND submitted_year NOT BETWEEN {config.MIN_YEAR} AND {config.MAX_YEAR}",
        hard=False,
        detail=f"{{n:,}} rows outside {config.MIN_YEAR}-{config.MAX_YEAR}",
    ),
    Check(
        "abstract length plausible",
        "SELECT COUNT(*) FROM papers WHERE abstract_word_count NOT BETWEEN "
        f"{MIN_ABSTRACT_WORDS} AND {MAX_ABSTRACT_WORDS}",
        hard=False,
        detail=f"{{n:,}} abstracts outside {MIN_ABSTRACT_WORDS}-{MAX_ABSTRACT_WORDS} words",
    ),
    Check("author_count positive", "SELECT COUNT(*) FROM papers WHERE author_count < 1"),
    Check(
        "pub_status is a known value",
        "SELECT COUNT(*) FROM papers WHERE pub_status NOT IN ('Published', 'Preprint')",
    ),
    Check(
        "category_stats totals reconcile",
        "SELECT (SELECT SUM(total_papers) FROM category_stats) "
        "- (SELECT COUNT(*) FROM papers)",
        detail=RECONCILES,
    ),
    Check(
        "publication_status totals reconcile",
        "SELECT (SELECT SUM(paper_count) FROM publication_status) "
        "- (SELECT COUNT(*) FROM papers)",
        detail=RECONCILES,
    ),
    Check(
        "no orphan authors in author_stats",
        "SELECT COUNT(*) FROM author_stats a "
        "WHERE NOT EXISTS (SELECT 1 FROM papers p WHERE p.first_author = a.author)",
        detail="{n:,} authors absent from papers",
    ),
]


def run() -> None:
    """Run every check and print a report. Raises if a hard check fails."""
    if not config.DB_PATH.exists():
        raise FileNotFoundError(
            f"No database at {config.DB_PATH}. Run the ingest and transform stages first."
        )

    print(f"Quality report for {config.DB_PATH.name}\n" + "-" * 58)
    failures = []

    with connect() as conn:
        # The denominator for the soft-check tolerance, read once up front so
        # the loop does not depend on the order checks happen to be declared in.
        total = scalar(conn, "SELECT COUNT(*) FROM papers")

        for check in CHECKS:
            n = scalar(conn, check.sql, check.params)

            # A soft check affecting very few rows is tolerated as a pass.
            tolerated = (
                not check.hard and total and n / total <= WARN_THRESHOLD
            )
            if check.passes(n) or tolerated:
                status = "PASS"
            elif check.hard:
                status = "FAIL"
                failures.append(check.name)
            else:
                status = "WARN"

            print(f"  [{status}] {check.name:<36} {check.detail.format(n=n)}")

    print("-" * 58)
    if failures:
        raise QualityCheckFailed(
            f"{len(failures)} check(s) failed: {', '.join(failures)}"
        )
    print("All hard checks passed.")


if __name__ == "__main__":
    try:
        run()
    except QualityCheckFailed as exc:
        print(exc)
        sys.exit(1)
