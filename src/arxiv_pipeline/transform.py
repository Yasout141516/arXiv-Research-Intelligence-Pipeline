"""Stage 2 â€” clean `raw_papers` into a star of analysis-ready tables.

Builds one master table (`papers`) with derived fields, then four aggregate
tables the visualisation and API layers read directly. Every table is
rebuilt from scratch, so the stage is idempotent.
"""
from . import config
from .db import connect, replace_table

# `submitted` is an RFC-822 stamp: "Fri, 15 Nov 2024 00:23:59 GMT".
# The 4-digit year always sits immediately before the " HH:" of the time,
# which makes the first colon a reliable anchor regardless of whether the
# day-of-month is one or two digits.
SUBMITTED_YEAR = """
        CASE WHEN INSTR(submitted, ':') > 7
             THEN CAST(SUBSTR(submitted, INSTR(submitted, ':') - 7, 4) AS INTEGER)
        END
"""

PAPERS_SQL = f"""
    SELECT
        arxiv_id,
        TRIM(title)    AS title,
        TRIM(abstract) AS abstract,
        TRIM(authors)  AS authors,
        primary_category,
        submitted,

        -- word count = spaces + 1
        LENGTH(TRIM(abstract)) - LENGTH(REPLACE(TRIM(abstract), ' ', '')) + 1
            AS abstract_word_count,

        -- author count = commas + 1
        LENGTH(TRIM(authors)) - LENGTH(REPLACE(TRIM(authors), ',', '')) + 1
            AS author_count,

        CASE WHEN INSTR(authors, ',') > 0
             THEN TRIM(SUBSTR(authors, 1, INSTR(authors, ',') - 1))
             ELSE TRIM(authors)
        END AS first_author,

        {SUBMITTED_YEAR} AS submitted_year,

        CASE
            WHEN primary_category LIKE 'cs.AI%'   THEN 'Artificial Intelligence'
            WHEN primary_category LIKE 'cs.LG%'   THEN 'Machine Learning'
            WHEN primary_category LIKE 'cs.CV%'   THEN 'Computer Vision'
            WHEN primary_category LIKE 'cs.CL%'   THEN 'Computation & Language'
            WHEN primary_category LIKE 'cs.RO%'   THEN 'Robotics'
            WHEN primary_category LIKE 'cs.NE%'   THEN 'Neural & Evolutionary Computing'
            WHEN primary_category LIKE 'stat.ML%' THEN 'Statistical Machine Learning'
            ELSE 'Other'
        END AS subject_area,

        -- a DOI or journal reference means the preprint reached a venue
        CASE
            WHEN (doi         IS NOT NULL AND TRIM(doi)         != '')
              OR (journal_ref IS NOT NULL AND TRIM(journal_ref) != '')
            THEN 'Published'
            ELSE 'Preprint'
        END AS pub_status

    FROM raw_papers
    WHERE arxiv_id IS NOT NULL
      AND title    IS NOT NULL
      AND abstract IS NOT NULL
"""

CATEGORY_STATS_SQL = """
    SELECT
        primary_category AS category,
        COUNT(*)         AS total_papers,
        SUM(pub_status = 'Published')                       AS published_count,
        ROUND(AVG(pub_status = 'Published') * 100, 2)       AS publish_rate_pct
    FROM papers
    GROUP BY primary_category
    ORDER BY total_papers DESC
"""

YEARLY_TRENDS_SQL = f"""
    SELECT
        submitted_year   AS year,
        primary_category AS category,
        COUNT(*)         AS paper_count
    FROM papers
    WHERE submitted_year BETWEEN {config.MIN_YEAR} AND {config.MAX_YEAR}
    GROUP BY submitted_year, primary_category
    ORDER BY submitted_year, paper_count DESC
"""

PUBLICATION_STATUS_SQL = """
    SELECT
        pub_status,
        primary_category AS category,
        COUNT(*)         AS paper_count
    FROM papers
    GROUP BY pub_status, primary_category
    ORDER BY primary_category, pub_status
"""

# `top_category` is the category an author publishes in most.
#
# Aggregate per (author, category) first, then window over that small result.
# Joining the ranked categories back to `papers` instead costs ~1,200x more
# (measured 61s vs 52ms at 5k rows): `papers.first_author` is unindexed, so
# the join degrades into a nested scan of ~24M row visits.
#
# Ties are broken by category name so the output is reproducible — without it,
# an author with one paper in each of two categories gets an arbitrary winner
# that can change between runs.
AUTHOR_STATS_SQL = """
    WITH per_category AS (
        SELECT
            first_author,
            primary_category,
            COUNT(*)            AS n,
            MIN(submitted_year) AS min_year,
            MAX(submitted_year) AS max_year
        FROM papers
        WHERE TRIM(COALESCE(first_author, '')) != ''
        GROUP BY first_author, primary_category
    ),
    ranked AS (
        SELECT
            first_author,
            primary_category,
            SUM(n)        OVER (PARTITION BY first_author) AS paper_count,
            MIN(min_year) OVER (PARTITION BY first_author) AS first_year,
            MAX(max_year) OVER (PARTITION BY first_author) AS last_year,
            ROW_NUMBER()  OVER (
                PARTITION BY first_author
                ORDER BY n DESC, primary_category
            ) AS rank
        FROM per_category
    )
    SELECT
        first_author     AS author,
        paper_count,
        first_year,
        last_year,
        primary_category AS top_category
    FROM ranked
    WHERE rank = 1
    ORDER BY paper_count DESC, author
"""

# Insertion order is dependency order: the aggregates all read `papers`.
# This dict is the single source of which tables the warehouse contains;
# `quality` reads its keys rather than keeping a parallel list.
TABLE_QUERIES = {
    "papers": PAPERS_SQL,
    "category_stats": CATEGORY_STATS_SQL,
    "yearly_trends": YEARLY_TRENDS_SQL,
    "publication_status": PUBLICATION_STATUS_SQL,
    "author_stats": AUTHOR_STATS_SQL,
}


def run() -> None:
    print("Transforming raw_papers into analysis tables...")
    with connect() as conn:
        for name, select_sql in TABLE_QUERIES.items():
            rows = replace_table(conn, name, select_sql)
            print(f"  {name:<20} built ({rows:,} rows)")

        print("\nSample of category_stats:")
        for row in conn.execute("SELECT * FROM category_stats LIMIT 5"):
            print(f"  {row}")


if __name__ == "__main__":
    run()
