"""Stage 2 — clean `raw_papers` into a star of analysis-ready tables.

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

# `top_category` is the category an author publishes in most. A window
# function ranks each author's categories in one pass, which avoids the
# per-author correlated subquery a naive MODE() workaround would need.
AUTHOR_STATS_SQL = """
    WITH ranked_categories AS (
        SELECT
            first_author,
            primary_category,
            ROW_NUMBER() OVER (
                PARTITION BY first_author ORDER BY COUNT(*) DESC
            ) AS rank
        FROM papers
        WHERE TRIM(COALESCE(first_author, '')) != ''
        GROUP BY first_author, primary_category
    )
    SELECT
        p.first_author            AS author,
        COUNT(*)                  AS paper_count,
        MIN(p.submitted_year)     AS first_year,
        MAX(p.submitted_year)     AS last_year,
        r.primary_category        AS top_category
    FROM papers p
    JOIN ranked_categories r
      ON r.first_author = p.first_author AND r.rank = 1
    GROUP BY p.first_author
    ORDER BY paper_count DESC
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
