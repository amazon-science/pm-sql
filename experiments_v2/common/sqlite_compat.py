"""The sqlite3 module every v2 component must use.

Python's sqlite3 links against whatever SQLite the interpreter was built with,
and that varies a lot. Amazon Linux 2's Python 3.12 links SQLite 3.7.17 (2013),
which predates window functions (3.25, 2018). On it, every gold or predicted
query using LEAD/LAG/OVER fails with a syntax error -- so predictions would be
scored wrong, and validate_sql would tell the agent a correct query is broken,
changing what the model does. Neither failure is loud.

So: prefer pysqlite3 (pip install pysqlite3-binary), which bundles a current
SQLite, and refuse outright to run on one without window functions.
"""

from __future__ import annotations

try:
    import pysqlite3.dbapi2 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3  # noqa: F401

MIN_VERSION = (3, 25, 0)

if tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:3]) < MIN_VERSION:
    raise ImportError(
        f"SQLite {sqlite3.sqlite_version} has no window functions (need >= "
        f"{'.'.join(map(str, MIN_VERSION))}); benchmark queries using LEAD/OVER "
        "would silently fail. Fix: python3 -m pip install --user pysqlite3-binary")

__all__ = ["sqlite3"]
