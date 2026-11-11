"""Prompt assembly for the R0-R6 ladder.

Every block is tagged with its admissibility class (EXPERIMENTS_V2_PLAN.md sec.1):

  DB      derivable from the event log by a SELECT -- always admissible
  DOC     from the BPIC 2020 dataset documentation -- always admissible
  FOLD    derived from the current fold's TRAINING split only -- admissible but
          must be regenerated per fold and its provenance recorded

There is deliberately no fourth class. v1's "Key SQL Patterns" and "Critical
Rules" were written by reading evaluation-set failures, which is what made the
ladder unscoreable; they have no home here and are not reproduced.

The rungs, and what each adds:

  R1  bare schema + question                                     (DB)
  R2  + activity list, resource count, case-ID and date formats  (DB, DOC)
  R3  + k mechanically selected few-shot examples                (FOLD)
  R3c + k uniformly sampled few-shot examples  [control]         (FOLD)
  R4  + validate_sql tool and an agent workflow                  (no new content)
  R5  + get_process_info tool, + mined domain vocabulary         (FOLD)
  R6  + k targeted examples from R5's training-fold errors       (FOLD)

R2's content is fold-independent, which is what makes R3's derivation free: one
R2 run over all 1,655 questions supplies the failure signal for all five folds.
"""

from __future__ import annotations

import hashlib
from sqlite_compat import sqlite3
from dataclasses import dataclass, field

from dataset import DB_PATH

RUNGS = ("R0", "R1", "R2", "R3", "R3c", "R4", "R5", "R6")
RUNGS_WITH_EXAMPLES = ("R3", "R3c", "R4", "R5", "R6")
RUNGS_WITH_TOOLS = ("R4", "R5", "R6")
RUNGS_WITH_PM_TOOL = ("R5", "R6")
RUNGS_WITH_VOCAB = ("R5", "R6")


# --------------------------------------------------------------------------
# admissible blocks
# --------------------------------------------------------------------------

SCHEMA_BLOCK = """## Schema

```sql
CREATE TABLE event_log (
    id        INT PRIMARY KEY,
    activity  TEXT COLLATE NOCASE,
    timestamp DATETIME,
    resource  TEXT COLLATE NOCASE,
    cost      NUMERIC,
    idcase    TEXT COLLATE NOCASE
);
```"""

ROLE_LINE = ("You are an expert SQL analyst working with a process mining event log. "
             "Produce the SQLite query that answers the user's question.")


@dataclass(frozen=True)
class LogFacts:
    """Everything R2 states about the data, read straight out of the database."""
    n_cases: int
    n_events: int
    activities: tuple[str, ...]
    n_resources: int
    example_case_id: str
    min_ts: str
    max_ts: str

    @property
    def n_activities(self) -> int:
        return len(self.activities)


def read_log_facts(db_path=None) -> LogFacts:
    """Query the facts R2 needs. No question text or gold SQL is touched."""
    conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    q = lambda s: cur.execute(s).fetchone()[0]  # noqa: E731
    facts = LogFacts(
        n_cases=q("SELECT COUNT(DISTINCT idcase) FROM event_log"),
        n_events=q("SELECT COUNT(*) FROM event_log"),
        activities=tuple(r[0] for r in cur.execute(
            "SELECT DISTINCT activity FROM event_log ORDER BY activity")),
        n_resources=q("SELECT COUNT(DISTINCT resource) FROM event_log"),
        example_case_id=q("SELECT MIN(idcase) FROM event_log"),
        min_ts=q("SELECT MIN(timestamp) FROM event_log"),
        max_ts=q("SELECT MAX(timestamp) FROM event_log"),
    )
    conn.close()
    return facts


def data_context_block(f: LogFacts) -> str:
    """R2's factual context. DB except the DD/MM/YYYY note, which is DOC.

    Says what the data *is*. Deliberately says nothing about how to query it, how
    to read any particular question, or what a business term means -- those are
    the inadmissible parts of v1's prompt.
    """
    return f"""## Data Context

Travel expense reimbursement process from a Dutch university. The log holds
{f.n_events:,} events across {f.n_cases:,} cases, {f.n_activities} distinct activities and
{f.n_resources} distinct resources, spanning {f.min_ts[:10]} to {f.max_ts[:10]}.

Each case (`idcase`) is one declaration moving through approval steps. Case
identifiers have the format "{f.example_case_id}".

Timestamps render as DD/MM/YYYY: "05/02/2017" is 5 February 2017.

Activity and resource values are case-sensitive; match them exactly as listed.

Activities ({f.n_activities}):
{chr(10).join('- ' + a for a in f.activities)}

Use SQLite syntax."""


# --------------------------------------------------------------------------
# fold-derived blocks
# --------------------------------------------------------------------------

def examples_block(examples: list, heading: str = "Examples") -> str:
    """Few-shot examples. FOLD -- every one comes from the training split.

    `examples` are example_selector.Example records (or anything with .question
    and .gold_sql). The tag layout matches what the leakage gate parses, so a
    prompt that reaches the model has necessarily been checked.
    """
    if not examples:
        return ""
    blocks = "\n".join(
        f"<example>\n<question>{e.question}</question>\n<sql>{e.gold_sql}</sql>\n</example>"
        for e in examples)
    return f"## {heading}\n\n<examples>\n{blocks}\n</examples>"


def vocab_block(rows: list[tuple[str, str, int]]) -> str:
    """Mined term -> column associations. FOLD.

    v1 hand-wrote this section from evaluation-set errors ("interventions = ALL
    events" came verbatim from a regression analysis). Here each row must have
    been mined from training-fold questions, and carries its support count so a
    reader can see why it is there.
    """
    if not rows:
        return ""
    lines = "\n".join(f'- "{term}" refers to {col} (seen in {n} training questions)'
                      for term, col, n in rows)
    return ("## Domain Vocabulary\n\n"
            "Associations observed in the training split:\n" + lines)


# --------------------------------------------------------------------------
# tool blocks
# --------------------------------------------------------------------------

VALIDATE_TOOL_BLOCK = """### validate_sql

Check that your query executes. Returns "VALID: N rows returned." or "ERROR: description".

- "VALID: 0 rows" -- the WHERE clause may be too restrictive, or an activity name may be misspelled.
- "ERROR: no such column" -- check the schema.
- "ERROR: query timed out" -- simplify the query."""

PM_TOOL_BLOCK = """### get_process_info

Retrieve process mining structure that the schema alone does not show.

Aspects: dfg, variants, activities, case_stats, rework, resources, process_tree

Use it when:
- validate_sql returned 0 rows and an activity name may not match
- the question depends on which activities follow which (aspect="dfg")
- you cannot tell which activity names a term refers to (aspect="activities")

Do not use it for straightforward counting, filtering or aggregation."""


def workflow_block(has_pm_tool: bool, max_tool_calls: int) -> str:
    steps = [
        "Read the question and decide which columns it asks for.",
        "Write a query. Include only the columns and filters the question requires.",
        "Call validate_sql to check it executes.",
    ]
    if has_pm_tool:
        steps.append("If it errors, or returns 0 rows unexpectedly, fix it -- calling "
                     "get_process_info first if activity names may be the problem.")
    else:
        steps.append("If it errors, or returns 0 rows unexpectedly, fix it and re-validate.")
    steps.append("Reply with only the final SQL inside <final_sql> tags.")
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    return (f"## Workflow\n\n{numbered}\n\n"
            "Do not explain your reasoning or describe the results.\n\n"
            f"Maximum {max_tool_calls} tool calls.")


ANSWER_FORMAT_BLOCK = """## Answer Format

Reply with only the SQL, inside <final_sql> tags:

<final_sql>SELECT ... FROM event_log ...</final_sql>"""


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

@dataclass
class Prompt:
    rung: str
    fold: int | None
    text: str
    blocks: list[str] = field(default_factory=list)
    example_uids: list[str] = field(default_factory=list)
    max_tool_calls: int = 0

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def uses_tools(self) -> bool:
        return self.rung in RUNGS_WITH_TOOLS


def build(
    rung: str,
    fold: int | None = None,
    examples: list | None = None,
    targeted_examples: list | None = None,
    vocab_rows: list[tuple[str, str, int]] | None = None,
    facts: LogFacts | None = None,
    max_tool_calls: int = 5,
) -> Prompt:
    """Assemble the system prompt for one rung (and fold, where it matters).

    Raises on a request that would violate the ladder -- e.g. examples passed to
    R1, or a fold-dependent rung built without a fold -- so a misconfigured run
    fails at build time rather than producing a quietly wrong prompt.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; expected one of {RUNGS}")
    if rung in ("R0",):
        raise ValueError("R0 re-scores the published GPT-3.5 SQL; it has no prompt")

    examples = examples or []
    targeted_examples = targeted_examples or []
    vocab_rows = vocab_rows or []

    if examples and rung not in RUNGS_WITH_EXAMPLES:
        raise ValueError(f"{rung} takes no few-shot examples")
    if rung in RUNGS_WITH_EXAMPLES and not examples:
        raise ValueError(f"{rung} requires few-shot examples")
    if targeted_examples and rung != "R6":
        raise ValueError("targeted examples belong to R6 only")
    if rung == "R6" and not targeted_examples:
        raise ValueError("R6 requires targeted examples")
    if vocab_rows and rung not in RUNGS_WITH_VOCAB:
        raise ValueError(f"{rung} takes no mined vocabulary")
    if rung in RUNGS_WITH_EXAMPLES and fold is None:
        raise ValueError(f"{rung} is fold-specific; pass fold=")

    parts: list[tuple[str, str]] = [("role", ROLE_LINE), ("schema", SCHEMA_BLOCK)]

    if rung != "R1":
        parts.append(("data_context", data_context_block(facts or read_log_facts())))

    if vocab_rows:
        parts.append(("vocab", vocab_block(vocab_rows)))

    if rung in RUNGS_WITH_TOOLS:
        tools = [VALIDATE_TOOL_BLOCK]
        if rung in RUNGS_WITH_PM_TOOL:
            tools.append(PM_TOOL_BLOCK)
        parts.append(("tools", "## Tools\n\n" + "\n\n".join(tools)))
        parts.append(("workflow", workflow_block(rung in RUNGS_WITH_PM_TOOL,
                                                 max_tool_calls)))
    else:
        parts.append(("answer_format", ANSWER_FORMAT_BLOCK))

    if examples:
        parts.append(("examples", examples_block(examples)))
    if targeted_examples:
        # Kept as a separate block so the R5->R6 delta is attributable to a
        # named, countable set of examples rather than to a reshuffled prompt.
        parts.append(("targeted_examples",
                      examples_block(targeted_examples, "Additional Examples")))

    used = [(name, body) for name, body in parts if body.strip()]
    return Prompt(
        rung=rung,
        fold=fold,
        text="\n\n".join(body for _, body in used),
        blocks=[name for name, _ in used],
        example_uids=[e.uid for e in list(examples) + list(targeted_examples)],
        max_tool_calls=max_tool_calls if rung in RUNGS_WITH_TOOLS else 0,
    )


def user_message(question: str) -> str:
    return f"<question>{question}</question>"


def main() -> None:
    """Print each rung's prompt shape, with placeholder fold-derived content."""
    from dataclasses import dataclass as _dc

    @_dc
    class E:
        uid: str
        question: str
        gold_sql: str

    ex = [E("X1", "<training question>", "SELECT ... FROM event_log")]
    facts = read_log_facts()
    print(f"log facts: {facts.n_events:,} events, {facts.n_cases:,} cases, "
          f"{facts.n_activities} activities, {facts.n_resources} resources\n")
    for rung in ("R1", "R2", "R3", "R4", "R5", "R6"):
        p = build(
            rung, fold=0 if rung in RUNGS_WITH_EXAMPLES else None,
            examples=ex if rung in RUNGS_WITH_EXAMPLES else None,
            targeted_examples=ex if rung == "R6" else None,
            vocab_rows=[("declarations", "idcase", 12)]
            if rung in RUNGS_WITH_VOCAB else None,
            facts=facts,
        )
        print(f"{p.rung:<4} {len(p.text):>6} chars  ~{len(p.text) // 4:>5} tokens  "
              f"{p.sha256[:10]}  {'+'.join(p.blocks)}")


if __name__ == "__main__":
    main()
