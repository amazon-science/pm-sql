"""
SQL validation tool for Step 4a.

Executes SQL against event_log.sqlite and returns ONLY whether it's valid
and how many rows it produces. Does NOT return the actual data.
"""

import sqlite3
import time
from pathlib import Path

from strands import tool

DB_PATH = Path(__file__).resolve().parents[2] / "event_log.sqlite"


@tool
def validate_sql(sql: str) -> str:
    """Validate a SQL query by executing it against the event_log database.

    Returns whether the query is valid and how many rows it produces.
    Does NOT return the actual data — only validation status.

    Args:
        sql: A SQLite SQL query to validate against the event_log table.

    Returns:
        "VALID: N rows returned." or "ERROR: description of the problem."
    """
    conn = sqlite3.connect(str(DB_PATH))

    start = time.time()

    def check_timeout():
        return 1 if time.time() - start > 5 else 0

    conn.set_progress_handler(check_timeout, 500)

    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return f"VALID: {len(rows)} rows returned."

    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return "ERROR: Query timed out (exceeded 5 seconds). Simplify your query."
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
