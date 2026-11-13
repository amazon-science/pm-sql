"""
Prompt templates for Step 3: Enhanced prompts for Claude Sonnet 4.6.

Step 3a: Zero-shot enhanced prompt (system + user, no examples)
Step 3b: Few-shot enhanced prompt (same + worked examples)
"""

# ============================================================================
# System prompt (shared by 3a and 3b)
# ============================================================================

SYSTEM_PROMPT = """You are an expert SQL analyst specializing in process mining event logs. You write SQLite queries against a single table called `event_log` that records business process events.

## Schema

```sql
CREATE TABLE event_log (
    id        INT PRIMARY KEY,       -- unique event identifier
    activity  TEXT COLLATE NOCASE,   -- activity name (34 distinct values)
    timestamp DATETIME,              -- when the event occurred (format: YYYY-MM-DD HH:MM:SS+00:00)
    resource  TEXT COLLATE NOCASE,   -- person who performed the activity (63 distinct values)
    cost      NUMERIC,               -- cost associated with the event
    idcase    TEXT COLLATE NOCASE    -- case identifier (format: 'declaration NNNNN', 6449 cases)
);
```

## Data Context

This is a travel expense reimbursement process from a Dutch university (2016-2020). Each case (`idcase`) represents one declaration going through approval steps. The 34 activities are:

Declaration approved by administration, Declaration approved by budget owner, Declaration approved by pre-approver, Declaration approved by supervisor, Declaration final approved by director, Declaration final approved by supervisor, Declaration rejected by administration, Declaration rejected by budget owner, Declaration rejected by director, Declaration rejected by employee, Declaration rejected by missing, Declaration rejected by pre-approver, Declaration rejected by supervisor, Declaration saved by employee, Declaration submitted by employee, End trip, Payment handled, Permit approved by administration, Permit approved by budget owner, Permit approved by pre-approver, Permit approved by supervisor, Permit final approved by director, Permit final approved by supervisor, Permit rejected by administration, Permit rejected by budget owner, Permit rejected by director, Permit rejected by employee, Permit rejected by missing, Permit rejected by pre-approver, Permit rejected by supervisor, Permit submitted by employee, Request payment, Send reminder, Start trip

## Key Process Mining SQL Patterns

1. **Case-level queries** require `GROUP BY idcase` — each case has multiple events.
2. **Temporal ordering** (activity A before B in same case): use self-join with `e1.idcase = e2.idcase AND e1.timestamp < e2.timestamp`.
3. **Consecutive activities**: use `LEAD(activity) OVER (PARTITION BY idcase ORDER BY timestamp)`.
4. **Rework/repetition**: same activity appearing multiple times in a case → `GROUP BY idcase, activity HAVING COUNT(*) > 1`.
5. **Process variants**: the sequence of activities in a case → `GROUP_CONCAT(activity, ' -> ') ... ORDER BY timestamp`.
6. **Duration**: use `julianday(timestamp2) - julianday(timestamp1)` for time differences.
7. **Use DISTINCT** when counting cases or when joins may produce duplicates.

## Rules

- Return ONLY the SQL query, no explanation or markdown.
- Use SQLite syntax (e.g., `julianday()` not `DATEDIFF`, `GROUP_CONCAT` not `STRING_AGG`).
- Activity and resource values are case-sensitive in the data. Match them exactly as listed above.
- When the question asks "how many cases", count DISTINCT idcase values.
- When the question is ambiguous, prefer the simpler interpretation."""


# ============================================================================
# User prompt template (3a: zero-shot)
# ============================================================================

USER_PROMPT_TEMPLATE = """{question}"""


# ============================================================================
# Few-shot examples for 3b
# Selected from the benchmark dataset (base questions only).
# These groups must be EXCLUDED from evaluation to avoid data leakage.
# ============================================================================

# Group IDs to exclude from evaluation when using few-shot
FEW_SHOT_GROUP_IDS = {"1", "23", "5", "109", "132"}

FEW_SHOT_EXAMPLES = [
    {
        # Group 1: easy / events / Value — simple filter + count
        "question": "How many 'end trip' activities are there in the event log?",
        "sql": "SELECT count(*) FROM event_log WHERE activity = 'End trip'"
    },
    {
        # Group 23: easy / cases / Generic — GROUP BY idcase HAVING
        "question": "What are the cases that have more than 20 activity occurrences?",
        "sql": "SELECT idcase FROM event_log GROUP BY idcase HAVING count(*) > 20"
    },
    {
        # Group 5: no_hardness / cases / Value — strftime for temporal grouping
        "question": "Show all cases that terminated in March 2018.",
        "sql": "SELECT idcase FROM event_log GROUP BY idcase HAVING strftime('%Y-%m', MAX(timestamp)) = '2018-03'"
    },
    {
        # Group 109: hard / cases / Value — self-join with temporal ordering
        "question": "How many cases went through the 'permit final approved by supervisor' activity before going through the 'start trip' activity?",
        "sql": "SELECT count(DISTINCT e1.idcase) FROM event_log e1 INNER JOIN event_log e2 ON e1.idcase = e2.idcase WHERE e1.activity = 'Permit final approved by supervisor' AND e2.activity = 'Start trip' AND e1.timestamp < e2.timestamp"
    },
    {
        # Group 132: medium / cases / Domain — domain vocabulary ("people" = resource, "declarations" = idcase)
        "question": "Which people worked on more than three declarations?",
        "sql": "SELECT resource FROM event_log GROUP BY resource HAVING COUNT(DISTINCT idcase) > 3"
    },
]


def format_few_shot_messages():
    """Format few-shot examples as conversation turns for the messages API."""
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["sql"]})
    return messages


def get_excluded_uids(dataset_path):
    """Get utterance IDs that belong to few-shot example groups (must be excluded from evaluation)."""
    import csv
    excluded = set()
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row.get('Utterance_id', '').strip()
            gid = row.get('Group_id', '').strip()
            if uid and gid in FEW_SHOT_GROUP_IDS:
                excluded.add(uid)
    return excluded


def build_prompt_3a(question):
    """Build the zero-shot enhanced prompt (system + user)."""
    return SYSTEM_PROMPT, question


def build_prompt_3b(question):
    """Build the few-shot enhanced prompt (system + examples + user)."""
    return SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, question
