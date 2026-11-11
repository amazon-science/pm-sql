"""The two tools the agent rungs expose, plus their Bedrock schemas.

Both are admissible without qualification, and it is worth being precise about
why, because the tools are the one place where the agent can reach information
that is not in its prompt:

  validate_sql      executes a candidate query against the event log and reports
                    only validity and row count. It never returns data, so it
                    cannot leak a gold answer -- the agent learns "this runs and
                    returns 12 rows", never what those rows contain.

  get_process_info  serves pre-computed pm4py artifacts (DFG, variants, activity
                    and resource statistics, rework, process tree) derived
                    entirely from the XES log by `precompute_pm_artifacts.py`.
                    Class DB: no question text, gold SQL, or evaluation result
                    took part in generating them, and they are fold-independent,
                    so the same files are correct for all five folds.

The artifacts are reused from v1 rather than regenerated. They are a pure
function of the log, v1's generator is checked in, and the leakage gate audit
found this rung's content clean -- so re-deriving them would produce identical
files and add a pm4py dependency to the v2 run.
"""

from __future__ import annotations

from sqlite_compat import sqlite3
import time
from pathlib import Path

from dataset import DB_PATH, PROJECT_ROOT

# Bounded identically to the scorer, so "the agent saw it validate" and "the
# scorer could run it" never disagree because of a timeout.
QUERY_TIMEOUT_S = 5.0

PM_DATA_DIR = PROJECT_ROOT / "experiments" / "step5_pm_artifacts" / "pm_data"

PM_ASPECTS = {
    "dfg": "dfg.txt",
    "variants": "variants.txt",
    "activities": "activities.txt",
    "case_stats": "case_stats.txt",
    "rework": "rework.txt",
    "resources": "resources.txt",
    "process_tree": "process_tree.txt",
}


def validate_sql(sql: str, db_path: Path | None = None) -> str:
    """Execute `sql` and report validity and row count only.

    Deliberately returns no data. An agent that could see result rows could read
    the answer out of the database and iterate against it, which would make the
    tool rungs unscoreable for the same reason v1's prompts were.
    """
    if not sql or not sql.strip():
        return "ERROR: empty query."

    conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    start = time.time()
    conn.set_progress_handler(
        lambda: 1 if time.time() - start > QUERY_TIMEOUT_S else 0, 500)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return f"VALID: {len(cur.fetchall())} rows returned."
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return (f"ERROR: Query timed out (exceeded {QUERY_TIMEOUT_S:.0f} seconds). "
                    "Simplify your query.")
        return f"ERROR: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()


def get_process_info(aspect: str) -> str:
    """Serve one pre-computed process mining artifact."""
    if aspect not in PM_ASPECTS:
        return (f"ERROR: Invalid aspect '{aspect}'. "
                f"Valid options: {', '.join(sorted(PM_ASPECTS))}")
    path = PM_DATA_DIR / PM_ASPECTS[aspect]
    if not path.exists():
        return (f"ERROR: Data file missing for aspect '{aspect}'. "
                "Run experiments/step5_pm_artifacts/precompute_pm_artifacts.py.")
    return path.read_text()


# --------------------------------------------------------------------------
# Bedrock tool schemas
# --------------------------------------------------------------------------

VALIDATE_SQL_SCHEMA = {
    "name": "validate_sql",
    "description": ("Validate a SQL query by executing it against the event_log "
                    "database. Returns 'VALID: N rows returned.' or "
                    "'ERROR: description'. Does not return the data itself."),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "A SQLite query to validate."},
        },
        "required": ["sql"],
    },
}

GET_PROCESS_INFO_SCHEMA = {
    "name": "get_process_info",
    "description": ("Retrieve pre-computed process mining structure for the event "
                    "log: directly-follows relationships, common process variants, "
                    "activity and resource statistics, rework, or a process tree."),
    "input_schema": {
        "type": "object",
        "properties": {
            "aspect": {
                "type": "string",
                "enum": sorted(PM_ASPECTS),
                "description": ("dfg: which activities follow which, with frequency "
                                "and mean time. variants: the most common activity "
                                "sequences. activities: all 34 activities with event "
                                "counts, case counts and rework rates. case_stats: "
                                "case durations, events per case, first/last "
                                "activities. rework: activities repeated within a "
                                "case. resources: per-resource statistics. "
                                "process_tree: hierarchical process model."),
            },
        },
        "required": ["aspect"],
    },
}

DISPATCH = {"validate_sql": lambda a: validate_sql(a.get("sql", "")),
            "get_process_info": lambda a: get_process_info(a.get("aspect", ""))}


def schemas_for_rung(rung: str) -> list[dict]:
    from prompt_builder import RUNGS_WITH_PM_TOOL, RUNGS_WITH_TOOLS

    if rung not in RUNGS_WITH_TOOLS:
        return []
    out = [VALIDATE_SQL_SCHEMA]
    if rung in RUNGS_WITH_PM_TOOL:
        out.append(GET_PROCESS_INFO_SCHEMA)
    return out


def call(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    return fn(args) if fn else f"ERROR: Unknown tool {name}"


def selftest() -> int:
    """Check both tools behave, including that validate_sql leaks no data."""
    failures = []

    def check(label, cond, detail=""):
        if not cond:
            failures.append(f"{label}: {detail}")

    ok = validate_sql("SELECT COUNT(*) FROM event_log")
    check("valid query", ok.startswith("VALID: 1 rows"), ok)

    # The property that matters: no result value appears in the tool output.
    wide = validate_sql("SELECT activity, resource FROM event_log LIMIT 3")
    check("no data leak", wide.startswith("VALID: 3 rows") and "declaration" not in wide.lower(),
          wide)

    check("bad column", validate_sql("SELECT nope FROM event_log").startswith("ERROR:"))
    check("syntax error", validate_sql("SELECT FROM WHERE").startswith("ERROR:"))
    check("empty", validate_sql("   ").startswith("ERROR:"))

    # Read-only: a write must fail rather than modify the log.
    w = validate_sql("DELETE FROM event_log WHERE id = -1")
    check("read-only", w.startswith("ERROR:"), w)

    for aspect in PM_ASPECTS:
        out = get_process_info(aspect)
        check(f"pm:{aspect}", out and not out.startswith("ERROR:"), out[:60])
    check("pm:bad aspect", get_process_info("nonsense").startswith("ERROR:"))

    check("schemas R2", schemas_for_rung("R2") == [])
    check("schemas R4", [s["name"] for s in schemas_for_rung("R4")] == ["validate_sql"])
    check("schemas R5", [s["name"] for s in schemas_for_rung("R5")]
          == ["validate_sql", "get_process_info"])
    check("dispatch", call("validate_sql", {"sql": "SELECT 1"}).startswith("VALID:"))
    check("dispatch unknown", call("nope", {}).startswith("ERROR: Unknown tool"))

    for f in failures:
        print("FAIL", f)
    print(f"tools selftest: {'OK' if not failures else f'{len(failures)} FAILURES'}")
    return len(failures)


if __name__ == "__main__":
    import sys

    sys.exit(1 if selftest() else 0)
