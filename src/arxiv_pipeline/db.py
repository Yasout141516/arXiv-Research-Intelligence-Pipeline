"""SQLite helpers shared across the pipeline stages."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config


@contextmanager
def connect(db_path: Path = config.DB_PATH, row_factory: bool = False):
    """Yield a connection that commits on success and always closes."""
    conn = sqlite3.connect(db_path)
    if row_factory:
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> object:
    """Run a query that returns a single value."""
    return conn.execute(sql, params).fetchone()[0]


def replace_table(conn: sqlite3.Connection, name: str, select_sql: str) -> None:
    """(Re)build `name` from a SELECT, so the stage is safe to re-run."""
    conn.execute(f"DROP TABLE IF EXISTS {name};")
    conn.execute(f"CREATE TABLE {name} AS {select_sql}")
    print(f"  {name:<20} built ({scalar(conn, f'SELECT COUNT(*) FROM {name}'):,} rows)")
