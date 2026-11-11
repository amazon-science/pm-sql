"""Leakage gate: refuse to score a run whose prompt contains held-out content.

Motivation (FEEDBACK_REVIEW_v2.md). v1 leaked three ways and its own note caught
only part of it, because the check was exact string matching:

  - group 49 "How long did each case last?"          -> exact match, caught
  - group 15 "What are the most frequent activities?" -> exact match, caught
  - group 57 "What is the mean duration [time] of each type of activity?"
        -> ONE word different. Exact matching said "no match", so 26 questions
           (1.6% of the benchmark) went unnoticed. Its gold SQL was also a
           rewrite of the prompt example's SQL, not a copy.

So the gate matches fuzzily on questions, independently on SQL text, and -- since
group 57 shows that a rewrite defeats every text test -- on what the SQL actually
returns:

  C1  prompt example belongs to a held-out group                  FAIL
  C2  question Jaccard >= 0.70 AND the prompt carries that
      question's answer (same SQL text or same result)            FAIL
        near-duplicate wording but a different answer             WARN
        0.55-0.70                                                 WARN
  C3  prompt SQL is a held-out gold answer, and is distinctive    FAIL
        ... but generic enough to be schema-derivable             WARN
        ... or same template with different literals              WARN
  C3d prompt SQL returns a held-out gold answer's exact result
      set despite different SQL text                              WARN
  C4  quoted prose phrases whose content words all occur in a
      held-out question                                           WARN
  C5  prompt hash recorded alongside predictions                  (recorded)

C2's answer condition is what separates leakage from coincidence of phrasing.
Held-out "How many times was each activity performed?" against the training
example "...by each resource?" scores 0.75 while needing a different GROUP BY:
same words, different answer, no leak. Requiring the answer too also means the
answer-class merge in folds.py and this gate agree on what "same question" means.

C3d is the denotational check, and it is what makes C2's answer condition robust
to rewriting. v1's step5c example for group 57 computed the average in one CTE
where the gold used two, so its SQL text similarity is 0.36 and C3 misses it --
but the two queries return byte-identical result sets, so the model was handed the
answer. C3d needs the SQLite DB; if it is absent the gate says so rather than
silently passing. On its own it only WARNs, for the reason given at the check.

Only FAILs block scoring. WARNs are for the write-up: they are the cases where
"is this leakage?" is a judgement call, and the paper is better off listing them
than having a reviewer find them.

Both thresholds are calibrated against v1 rather than guessed; see the constants.

Running `--audit-v1` reports leaked groups {8, 10, 49} for step5b and
{15, 49, 57} for step5c -- 52 questions, 3.1% of the benchmark, in the union.
Groups 8 and 10 are ones the manual audit missed: they were pasted into the "Key
SQL Patterns" prose rather than into an <example> block, which is why the gate
harvests SQL from prose and code spans too.

Usage as a library:
    report = check_prompt(prompt_text, fold=0, example_uids=[...])
    if not report.passed: raise SystemExit(report.render())

Usage as a CLI:
    python leakage_gate.py --audit-v1    # retrospective audit of the v1 prompts
    python leakage_gate.py --selftest    # verify the known leaks are still caught
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# normalise_sql lives in dataset so folds.py can build answer-equivalence
# classes with exactly the canonicalisation the gate checks against.
from dataset import Question, load_questions, normalise_sql, questions_by_uid
from folds import test_groups, test_uids

# Question-similarity thresholds. The group-57 miss scores exactly 0.80, so a
# 0.80 cut-off would catch it with zero margin. Measured on the v1 prompts, the
# whole 0.55-0.80 band contains nothing but true positives (other group-57
# paraphrases at 0.75), so the fail line drops to 0.70 and the band below it is
# reported for human review rather than discarded.
JACCARD_FAIL = 0.70
JACCARD_WARN = 0.55

# Character similarity on SQL. Only used as a backstop: it rates `count(*) > 20`
# and `count(*) <= 20` at 0.98 despite them being opposite predicates, so
# literal-masked template comparison does the real work below.
SQL_SIM_NEAR = 0.95

# A prompt SQL statement equal to a held-out gold answer is only *evidence of
# leakage* if the statement is distinctive. `SELECT COUNT(*), resource FROM
# event_log GROUP BY resource` is some group's gold SQL, but it is also the only
# sensible way to write that query and is derivable from the schema alone.
# Distinctive = long, or uses a construct the model demonstrably struggles with.
SQL_DISTINCTIVE_CHARS = 80
_SQL_DISTINCTIVE = re.compile(
    r"\bover\s*\(|\bwith\b|\bjoin\b|strftime|julianday|\blead\b|\blag\b"
    r"|row_number|\bcase\s+when\b|\bexists\b|\bunion\b|\(\s*select",
    re.IGNORECASE,
)

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "of", "in",
    "on", "at", "to", "for", "by", "with", "and", "or", "that", "this", "these",
    "those", "there", "it", "its", "we", "you", "what", "which", "how", "who",
    "many", "much", "be", "been", "have", "has", "had", "each", "all", "any",
    "from", "as", "not", "no", "s", "t",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP}


def jaccard(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def sql_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_sql(a), normalise_sql(b)).ratio()


def mask_literals(sql: str) -> str:
    """Normalised SQL with numeric and string literals replaced by `?`.

    Two statements with the same mask are the same query template. Comparing
    masks separates "the prompt hands over the gold answer" (same mask, same
    literals) from "the prompt hands over the gold answer's shape" (same mask,
    different literals) -- weaker, but still worth surfacing.
    """
    s = normalise_sql(sql)
    s = re.sub(r"'[^']*'", "?", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", "?", s)
    return s


def is_distinctive_sql(sql: str) -> bool:
    n = normalise_sql(sql)
    return len(n) >= SQL_DISTINCTIVE_CHARS or bool(_SQL_DISTINCTIVE.search(n))


def classify_sql_overlap(prompt_sql: str, gold_sql: str) -> tuple[str, float] | None:
    """Relationship between a prompt SQL and a held-out gold SQL.

    Returns (kind, score) where kind is:
      "verbatim" -- same template, same literals: the gold answer is in the prompt
      "template" -- same shape, different literals or a near-identical string
    or None when the two are unrelated.
    """
    pn, gn = normalise_sql(prompt_sql), normalise_sql(gold_sql)
    if not pn or not gn:
        return None
    pm, gm = mask_literals(prompt_sql), mask_literals(gold_sql)
    if pm == gm:
        return ("verbatim", 1.0) if pn == gn else ("template", 1.0)
    sim = SequenceMatcher(None, pn, gn).ratio()
    return ("template", sim) if sim >= SQL_SIM_NEAR else None


def prompt_sql_digests(sql_list: list[str]) -> dict[str, str] | None:
    """Column-agnostic result digest for each prompt SQL that executes.

    Returns None when the database is unavailable, so callers can distinguish
    "nothing matched" from "could not check". Non-executing statements (a partial
    snippet quoted in prose, say) are simply absent from the mapping.
    """
    try:
        from eval import _connect, digest, execute, norm_col_agnostic
    except Exception:
        return None
    try:
        conn = _connect()
    except Exception:
        return None
    out: dict[str, str] = {}
    try:
        for s in sql_list:
            rows, err = execute(conn, s)
            if err or rows is None:
                continue
            d = digest(norm_col_agnostic(rows))
            if d:
                out[s] = d
    finally:
        conn.close()
    return out


@dataclass
class Finding:
    check: str
    severity: str  # FAIL | WARN
    detail: str
    score: float | None = None


@dataclass
class GateReport:
    fold: int
    prompt_sha256: str
    n_held_out: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "FAIL"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def render(self) -> str:
        head = (f"leakage gate fold={self.fold} prompt={self.prompt_sha256[:12]} "
                f"held_out={self.n_held_out}: "
                f"{'PASS' if self.passed else 'FAIL'} "
                f"({len(self.failures)} fail, {len(self.warnings)} warn)")
        lines = [head]
        for f in self.findings:
            s = f" [{f.score:.2f}]" if f.score is not None else ""
            lines.append(f"  {f.severity} {f.check}{s} {f.detail}")
        return "\n".join(lines)


def extract_prompt_questions(prompt: str) -> list[str]:
    """Questions embedded in the prompt, from <question> tags or Q: lines."""
    out = re.findall(r"<question>(.*?)</question>", prompt, re.DOTALL)
    return [q.strip() for q in out if q.strip()]


_SQL_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def extract_prompt_sql(prompt: str) -> list[str]:
    """Every SQL statement in the prompt, however it is marked up.

    Tagged examples are the obvious source, but v1 also pasted gold SQL into
    prose as inline `backticked` "Key SQL Patterns" -- pattern 5 is group 49's
    gold answer verbatim. A gate that only reads <sql> tags misses that, so all
    four markup forms are harvested.
    """
    out: list[str] = []
    out += re.findall(r"<sql>(.*?)</sql>", prompt, re.DOTALL)
    out += re.findall(r"<final_sql>(.*?)</final_sql>", prompt, re.DOTALL)
    out += re.findall(r"```(?:sql)?\s*(.*?)```", prompt, re.DOTALL)
    out += re.findall(r"`([^`\n]{15,})`", prompt)  # inline code
    return [s.strip() for s in (x.strip() for x in out) if s and _SQL_START.match(s)]


def extract_quoted_phrases(prompt: str, min_words: int = 2) -> list[str]:
    """Double-quoted prose phrases, e.g. "Interventions in declarations"."""
    out = []
    for m in re.findall(r'"([^"\n]{3,80})"', prompt):
        if len(_words(m)) >= min_words:
            out.append(m.strip())
    return out


def check_prompt(
    prompt: str,
    fold: int,
    example_uids: list[str] | None = None,
    held_out: list[Question] | None = None,
    extra_prompt_questions: list[str] | None = None,
    extra_prompt_sql: list[str] | None = None,
) -> GateReport:
    """Validate one fold's prompt against that fold's held-out questions."""
    if held_out is None:
        ho_uids = test_uids(fold)
        held_out = [q for q in load_questions() if q.uid in ho_uids]

    report = GateReport(
        fold=fold,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        n_held_out=len(held_out),
    )

    # C1 -- provenance: every declared example must come from the training split.
    if example_uids:
        held_groups = test_groups(fold)
        by_uid = questions_by_uid()
        for uid in example_uids:
            q = by_uid.get(uid)
            if q is None:
                report.findings.append(Finding(
                    "C1_unknown_uid", "FAIL", f"example uid {uid} not in dataset"))
            elif q.group_id in held_groups:
                report.findings.append(Finding(
                    "C1_example_in_test", "FAIL",
                    f"example uid {uid} (group {q.group_id}) is held out in fold {fold}"))

    p_questions = extract_prompt_questions(prompt) + (extra_prompt_questions or [])
    p_sql = extract_prompt_sql(prompt) + (extra_prompt_sql or [])

    # Denotation of every prompt SQL, and of every held-out gold, so C2 and C3d
    # can ask "does the prompt return this held-out answer?" once rather than
    # per-comparison. None means the DB was unavailable.
    p_digests = prompt_sql_digests(p_sql)
    gold_digest: dict[str, str] = {}
    if p_digests is not None:
        try:
            from eval import load_gold
            g = load_gold()
            gold_digest = {h.uid: g[h.uid]["col_agnostic"] for h in held_out
                           if h.uid in g and g[h.uid]["col_agnostic"]}
        except Exception:
            p_digests = None

    def answer_reachable(h: Question) -> bool:
        """True if some prompt SQL yields h's gold answer, textually or in result."""
        if h.gold_sql and any(
                (classify_sql_overlap(ps, h.gold_sql) or ("", 0.0))[0] == "verbatim"
                for ps in p_sql):
            return True
        d = gold_digest.get(h.uid)
        return bool(d and p_digests and d in p_digests.values())

    # C2 -- fuzzy question match. This is the check that catches group 57.
    #
    # A near-duplicate question only leaks if the prompt also carries that
    # question's answer. "How many times was each activity performed?" held out
    # against the training example "...by each resource?" scores 0.75 on Jaccard
    # while needing a different GROUP BY, so failing on wording alone would block
    # a fold set that leaks nothing. Escalate to FAIL only when some SQL in the
    # prompt is the held-out gold answer verbatim; otherwise WARN and let a human
    # read it. C3 catches verbatim gold SQL on its own, whatever the wording, so
    # nothing is lost by making C2 the weaker signal.
    for pq in p_questions:
        score, hit = max(((jaccard(pq, h.question), h) for h in held_out),
                         key=lambda t: t[0], default=(0.0, None))
        if hit is None or score < JACCARD_WARN:
            continue
        answer_present = answer_reachable(hit)
        report.findings.append(Finding(
            "C2_question_near_dup",
            "FAIL" if score >= JACCARD_FAIL and answer_present else "WARN",
            f'prompt "{pq[:60]}" ~ held-out uid {hit.uid} '
            f'(group {hit.group_id}) "{hit.question[:60]}"'
            + ("" if answer_present else " [gold answer differs; wording only]"),
            score))

    # C3 -- SQL overlap, independent of question wording. Catches a prompt
    # example whose SQL is a rewrite of a held-out gold answer, and the v1 "Key
    # SQL Patterns" block, which pasted several gold answers into prose.
    #
    # Reported per (prompt SQL, kind), keeping only the worst hit, because one
    # generic pattern matches dozens of held-out golds and listing them all
    # buries the findings that matter.
    for ps in p_sql:
        by_kind: dict[str, tuple[float, Question]] = {}
        for h in held_out:
            if not h.gold_sql:
                continue
            res = classify_sql_overlap(ps, h.gold_sql)
            if res is None:
                continue
            kind, score = res
            if kind not in by_kind or score > by_kind[kind][0]:
                by_kind[kind] = (score, h)

        if "verbatim" in by_kind:
            score, hit = by_kind["verbatim"]
            distinctive = is_distinctive_sql(ps)
            n_same = sum(
                1 for h in held_out if h.gold_sql
                and (classify_sql_overlap(ps, h.gold_sql) or ("", 0))[0] == "verbatim")
            report.findings.append(Finding(
                "C3_sql_verbatim_gold",
                "FAIL" if distinctive else "WARN",
                f"prompt SQL is the gold answer for held-out uid {hit.uid} "
                f"(group {hit.group_id})"
                + (f" and {n_same - 1} other held-out question(s)" if n_same > 1 else "")
                + ("" if distinctive else " [generic: schema-derivable, not leakage]"),
                score))
        elif "template" in by_kind:
            score, hit = by_kind["template"]
            report.findings.append(Finding(
                "C3_sql_template_match", "WARN",
                f"prompt SQL has the same shape as the gold answer for held-out "
                f"uid {hit.uid} (group {hit.group_id}), differing only in literals",
                score))

    # C3d -- denotational: a prompt SQL that returns a held-out gold's exact
    # result set hands over the answer regardless of how it is written. Only
    # reported when C3 did not already flag the same statement as verbatim, so a
    # single leak yields one finding.
    #
    # WARN and never FAIL, because on a single 72k-event database distinct
    # questions collide: `SELECT resource FROM event_log GROUP BY resource HAVING
    # COUNT(DISTINCT idcase) > 3` returns exactly the 63 rows of group 121's
    # `SELECT DISTINCT resource ... ORDER BY resource`, since every resource
    # touches more than three cases. Same rows, unrelated question. This is
    # execution accuracy's own weakness -- equivalent-on-this-instance is not
    # equivalent -- so a digest match is corroboration, not proof. It becomes a
    # FAIL only through C2, where a near-duplicate *question* supplies the
    # missing half of the argument.
    if p_digests is None:
        report.findings.append(Finding(
            "C3d_not_checked", "WARN",
            "result-set check skipped: SQLite DB or gold digests unavailable"))
    else:
        by_digest: dict[str, list[Question]] = {}
        for h in held_out:
            d = gold_digest.get(h.uid)
            if d:
                by_digest.setdefault(d, []).append(h)
        for ps, d in p_digests.items():
            hits = by_digest.get(d)
            if not hits:
                continue
            if any((classify_sql_overlap(ps, h.gold_sql) or ("", 0.0))[0] == "verbatim"
                   for h in hits if h.gold_sql):
                continue  # already reported by C3
            hit = min(hits, key=lambda h: int(h.uid))
            report.findings.append(Finding(
                "C3d_sql_returns_gold_result", "WARN",
                f"prompt SQL returns the gold result for held-out uid {hit.uid} "
                f"(group {hit.group_id})"
                + (f" and {len(hits) - 1} other held-out question(s)"
                   if len(hits) > 1 else "")
                + " despite different SQL text",
                1.0))

    # C4 -- prose that quotes benchmark phrasing. Fuzzy by nature, so WARN.
    # (deduplication of the C2/C3 findings happens at the end)
    for phrase in extract_quoted_phrases(prompt):
        toks = _content_tokens(phrase)
        if len(toks) < 2:
            continue
        hits = [h for h in held_out if toks <= _content_tokens(h.question)]
        if hits:
            report.findings.append(Finding(
                "C4_prose_quotes_benchmark", "WARN",
                f'prose phrase "{phrase}" content-words appear in '
                f"{len(hits)} held-out question(s), e.g. uid {hits[0].uid} "
                f"(group {hits[0].group_id})"))

    # The same SQL often appears several times in a prompt (a CORRECT/WRONG rule
    # pair, an example repeated in the workflow section). Report each distinct
    # finding once so the count reflects distinct leaks, not distinct mentions.
    seen: set[tuple[str, str]] = set()
    deduped = []
    for f in report.findings:
        key = (f.check, f.detail)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    report.findings = deduped

    # FAILs first, then by descending score, so the worst problem is line one.
    report.findings.sort(key=lambda f: (f.severity != "FAIL", -(f.score or 0)))
    return report


# ---------------------------------------------------------------------------
# Retrospective audit of the v1 prompts.
#
# Being able to say "our gate flags three leaks in our own prior version" is a
# far stronger position in review than quietly fixing it, so this is a shipped
# feature and not a scratch script.
# ---------------------------------------------------------------------------

V1_FEWSHOT_GROUPS = {"1", "5", "23", "109", "132"}


def audit_v1() -> int:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    targets = {
        "step5_pm_artifacts": root / "experiments/step5_pm_artifacts/prompts.py",
        "step5b_refined": root / "experiments/step5b_refined/prompts.py",
        "step5c_examples": root / "experiments/step5c_examples/prompts.py",
        "step5d_graph_tool_v2": root / "experiments/step5d_graph_tool_v2/prompts.py",
    }

    # v1 evaluated everything except the 5 declared few-shot groups.
    v1_held_out = [q for q in load_questions() if q.group_id not in V1_FEWSHOT_GROUPS]
    print(f"v1 evaluation set: {len(v1_held_out)} questions "
          f"(all but groups {sorted(V1_FEWSHOT_GROUPS, key=int)})\n")

    worst = 0
    for name, path in targets.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        ns: dict = {}
        exec(compile(path.read_text(), str(path), "exec"), ns)
        prompt = ns.get("SYSTEM_PROMPT", "")
        rep = check_prompt(prompt, fold=-1, held_out=v1_held_out)
        print(f"--- {name}")
        print(rep.render())
        print()
        worst = max(worst, len(rep.failures))
    return worst


# Groups the v1 prompts are known to leak, per FEEDBACK_REVIEW_v2.md plus the
# two the gate itself found (8 and 10, pasted into "Key SQL Patterns" prose).
V1_KNOWN_LEAKS = {
    "step5b_refined": {"8", "10", "49"},
    "step5c_examples": {"15", "49", "57"},
}


def selftest() -> int:
    """Pin the gate's behaviour on cases with known ground truth.

    Without this, a future threshold tweak could silently stop catching the
    group-57 near-match -- the exact failure that motivated the gate.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    fails = 0

    def check(desc: str, ok: bool) -> None:
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {desc}")

    print("question similarity")
    check("group-57 near-match (one word differs) is a FAIL",
          jaccard("What is the mean duration of each type of activity?",
                  "What is the mean duration time of each type of activity?")
          >= JACCARD_FAIL)
    check("unrelated questions score below the warn band",
          jaccard("How many 'end trip' activities are there in the event log?",
                  "What is the mean duration time of each type of activity?")
          < JACCARD_WARN)

    print("sql overlap")
    check("identical SQL modulo whitespace/alias -> verbatim",
          classify_sql_overlap(
              "SELECT COUNT(*) AS n, resource FROM event_log GROUP BY resource",
              "SELECT  COUNT(*), resource FROM event_log GROUP  BY resource")
          == ("verbatim", 1.0))
    check("same shape, different literal -> template, not verbatim",
          (classify_sql_overlap(
              "SELECT resource FROM event_log GROUP BY resource HAVING COUNT(DISTINCT idcase) > 3",
              "SELECT resource FROM event_log GROUP BY resource HAVING COUNT(DISTINCT idcase) > 200")
           or (None,))[0] == "template")
    check("opposite predicate (> vs <=) is not verbatim",
          (classify_sql_overlap(
              "SELECT idcase FROM event_log GROUP BY idcase HAVING count(*) > 20",
              "SELECT idcase FROM event_log GROUP BY idcase HAVING count(*) <= 20")
           or (None,))[0] != "verbatim")
    check("different query at 0.85 char similarity is not flagged",
          classify_sql_overlap(
              "SELECT count(*) FROM event_log WHERE activity = 'End trip'",
              "select * from event_log where activity = 'Start trip'") is None)
    check("generic aggregation is not distinctive",
          not is_distinctive_sql("SELECT COUNT(*), resource FROM event_log GROUP BY resource"))
    check("strftime duration query is distinctive",
          is_distinctive_sql("SELECT idcase, MAX(strftime('%s', timestamp)) - "
                             "MIN(strftime('%s', timestamp)) FROM event_log GROUP BY idcase"))

    print("end-to-end on the v1 prompts")
    v1_held_out = [q for q in load_questions() if q.group_id not in V1_FEWSHOT_GROUPS]
    for name, expected in V1_KNOWN_LEAKS.items():
        path = root / f"experiments/{name}/prompts.py"
        if not path.exists():
            check(f"{name}: prompts.py present", False)
            continue
        ns: dict = {}
        exec(compile(path.read_text(), str(path), "exec"), ns)
        rep = check_prompt(ns.get("SYSTEM_PROMPT", ""), fold=-1, held_out=v1_held_out)
        found = {m for f in rep.failures
                 for m in re.findall(r"group (\d+)", f.detail)}
        check(f"{name}: gate FAILs", not rep.passed)
        check(f"{name}: flags groups {sorted(expected, key=int)} "
              f"(found {sorted(found, key=int)})", expected <= found)

    print(f"\n{'selftest PASSED' if not fails else f'selftest FAILED ({fails})'}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage gate")
    ap.add_argument("--audit-v1", action="store_true",
                    help="run the gate retrospectively against the v1 prompts")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the gate catches the known v1 leaks")
    args = ap.parse_args()
    if args.audit_v1:
        n = audit_v1()
        print(f"v1 audit complete: {n} distinct leak findings in the worst prompt.")
    elif args.selftest:
        raise SystemExit(1 if selftest() else 0)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
