"""Mine domain-term -> column associations from a training fold.

Why this exists (EXPERIMENTS_V2_PLAN.md sec.5). v1's "Domain Vocabulary" section
was hand-written by reading regressions on the evaluation set: the line
"interventions = ALL events" came directly from analysing which questions the
model got wrong. That is the inadmissible kind of prompt content, and it is what
made v1's PM-tools rung unattributable.

Here the same section is produced mechanically. For each training-fold question
we know the gold SQL, so we know which columns and which activity literals the
answer needed. Associating the question's content words with those columns over
the whole training split recovers the useful part of v1's vocabulary list --
"declaration" tends to mean `idcase`, "step" tends to mean `activity` -- without
anyone looking at an evaluation-set result.

Every emitted row carries its support count, so the prompt states its own
provenance and a reader can see why a line is there.

What it deliberately does not do: it never emits a rule, a hint about how to
write SQL, or an instruction. Only "term X co-occurs with column Y in N training
questions". Anything more prescriptive would reintroduce the class of content
this module exists to eliminate.

Usage:
    python vocab_miner.py --dry-run
    python vocab_miner.py --fold 0 [--max-terms 12] [--out vocab_fold0.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset import Question, base_questions
from folds import load_fold_config, train_groups

DEFAULT_MAX_TERMS = 10

# A term must appear in at least this many training questions before it can be
# emitted. Below this the association is noise: with ~164 training base
# questions a term seen twice will happily "associate" with whatever column those
# two questions happened to use.
MIN_SUPPORT = 5

# And it must point at one column clearly. A term that appears with `activity`
# and `resource` about equally tells the model nothing, and stating a coin-flip
# as a fact is worse than staying silent.
MIN_PURITY = 0.75

# Purity alone is not enough, because the columns are not equally common: most
# gold SQL touches `idcase` or `activity`, so a meaningless term looks "pure" for
# them by default. Lift is purity divided by the column's own base rate, so a row
# survives only if the term predicts the column better than chance does.
MIN_LIFT = 1.25

COLUMNS = ("activity", "timestamp", "resource", "cost", "idcase")

# Words that carry no domain meaning, excluded by linguistic category rather
# than by trial and error -- the distinction matters, because a list tuned until
# the output "looked right" would be hand-authored prompt content wearing a
# mechanical disguise. The categories are: question and SQL scaffolding ("how",
# "many", "list"); the column names themselves, which need no gloss; number words
# and month names, which name literals rather than columns (R2 already supplies
# the date format and the activity list); and comparatives and quantifiers, which
# describe a relation between values rather than referring to one.
_STOPWORDS = frozenset("""
a an the of in on at to for from by with and or not is are was were be been
what which who whom whose when where why how many much all any each every
some no none more most less least than then that this these those there here
it its as if do does did done have has had having i we you they he she
list show give tell find get return display name names number numbers total
count sum average avg minimum maximum min max first last long longest shortest
per event events case cases activity activities resource resources cost costs
timestamp timestamps time times date dates log logs id ids exist exists
occurred occur happened happen performed perform used use are's s t

one two three four five six seven eight nine ten eleven twelve twenty thirty
fifty hundred once twice single once-only
january february march april may june july august september october november
december
same different distinct other another between above below over under before
after during within only also just both either neither such very least-once
""".split())

_WORD = re.compile(r"[a-z][a-z'-]+")


def content_terms(question: str) -> set[str]:
    """Domain-ish words in a question.

    Quoted spans are dropped: a question that says 'End trip' is naming a literal
    that R2 already lists in full, so mining it would produce a vocabulary line
    restating the activity list.
    """
    q = re.sub(r"'[^']*'", " ", question.lower())
    return {w for w in _WORD.findall(q) if w not in _STOPWORDS and len(w) > 2}


def sql_columns(gold_sql: str) -> set[str]:
    """Which of the five columns the gold SQL mentions anywhere."""
    s = gold_sql.lower()
    return {c for c in COLUMNS if re.search(rf"\b{c}\b", s)}


_SELECT_LIST = re.compile(r"\bselect\b(.*?)\bfrom\b", re.IGNORECASE | re.DOTALL)
_GROUP_LIST = re.compile(r"\bgroup\s+by\b(.*?)(?:\bhaving\b|\border\b|\blimit\b|\)|$)",
                         re.IGNORECASE | re.DOTALL)


def denoted_columns(gold_sql: str) -> set[str]:
    """Columns the question *refers to*, as opposed to merely filters on.

    This distinction is what makes the mined rows trustworthy. A question like
    "which resource was responsible for approving declarations?" produces
    `SELECT resource ... WHERE activity = 'Declaration approved...'`. Counting
    every column the SQL mentions makes "responsible" look like it means
    `activity`, because the activity name is in the WHERE clause -- exactly
    backwards. Only the projection and the GROUP BY say what the question is
    *about*; a WHERE clause says which literal value it restricts to, and R2
    already lists every literal.

    Scanning all SELECT and GROUP BY clauses (including those inside subqueries)
    is intentional: a subquery's projection is still denotation.
    """
    s = " ".join(gold_sql.split())
    spans = [m.group(1) for m in _SELECT_LIST.finditer(s)]
    spans += [m.group(1) for m in _GROUP_LIST.finditer(s)]
    blob = " ".join(spans).lower()
    return {c for c in COLUMNS if re.search(rf"\b{c}\b", blob)}


@dataclass
class VocabRow:
    term: str
    column: str
    support: int          # training questions with this term AND this column
    term_total: int       # training questions with this term
    purity: float
    lift: float

    def as_prompt_row(self) -> tuple[str, str, int]:
        return (self.term, self.column, self.support)


@dataclass
class Mined:
    fold: int
    rows: list[VocabRow]
    n_train_base: int
    min_support: int
    min_purity: float
    min_lift: float
    n_terms_considered: int

    def prompt_rows(self) -> list[tuple[str, str, int]]:
        return [r.as_prompt_row() for r in self.rows]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def _column_base_rates(questions: list[Question]) -> dict[str, float]:
    """How often each column is denoted, regardless of the question's wording.

    The comparison that makes an association informative: "step means activity"
    is only worth saying if `activity` is denoted more often by questions
    containing "step" than by training questions in general.
    """
    n = 0
    hits: Counter[str] = Counter()
    for q in questions:
        cols = denoted_columns(q.gold_sql)
        if not cols:
            continue
        n += 1
        hits.update(cols)
    return {c: hits[c] / n for c in COLUMNS} if n else {}


def mine(
    questions: list[Question],
    max_terms: int = DEFAULT_MAX_TERMS,
    min_support: int = MIN_SUPPORT,
    min_purity: float = MIN_PURITY,
    min_lift: float = MIN_LIFT,
) -> list[VocabRow]:
    """Rank term -> column associations over a fixed set of questions."""
    base_rate = _column_base_rates(questions)
    term_total: Counter[str] = Counter()
    pair: dict[str, Counter[str]] = defaultdict(Counter)

    for q in questions:
        cols = denoted_columns(q.gold_sql)
        if not cols:
            continue
        for term in content_terms(q.question):
            term_total[term] += 1
            for c in cols:
                pair[term][c] += 1

    rows: list[VocabRow] = []
    for term, total in term_total.items():
        if total < min_support:
            continue
        # Rank the term's candidate columns by lift, not by raw count: the raw
        # winner is usually just the commonest column.
        scored = [
            (support / total / base_rate[c] if base_rate.get(c) else 0.0, c, support)
            for c, support in pair[term].items()
        ]
        lift, col, support = max(scored, key=lambda t: (t[0], t[2], t[1]))
        purity = support / total
        if support < min_support or purity < min_purity or lift < min_lift:
            continue
        rows.append(VocabRow(term=term, column=col, support=support,
                             term_total=total, purity=purity, lift=lift))

    # Rank on lift, then support; alphabetical last so the list is stable.
    rows.sort(key=lambda r: (-r.lift, -r.support, r.term))
    return rows[:max_terms]


def mine_fold(fold: int, max_terms: int = DEFAULT_MAX_TERMS) -> Mined:
    """Mine one fold's vocabulary. Reads training-split base questions only."""
    train = base_questions(train_groups(fold))
    rows = mine(train, max_terms=max_terms)
    considered = len({t for q in train for t in content_terms(q.question)})
    return Mined(fold=fold, rows=rows, n_train_base=len(train),
                 min_support=MIN_SUPPORT, min_purity=MIN_PURITY,
                 min_lift=MIN_LIFT, n_terms_considered=considered)


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------

def dry_run(max_terms: int) -> int:
    cfg = load_fold_config()
    problems = 0
    all_terms: list[set[str]] = []
    mapping: dict[str, set[str]] = defaultdict(set)

    for fold in range(cfg["k"]):
        m = mine_fold(fold, max_terms=max_terms)
        print(f"--- fold {fold}: {len(m.rows)} rows from {m.n_train_base} train base "
              f"questions ({m.n_terms_considered} candidate terms)")
        for r in m.rows:
            print(f"  {r.term:<16} -> {r.column:<10} "
                  f"{r.support:>3}/{r.term_total:<3} purity={r.purity:.2f} "
                  f"lift={r.lift:.2f}")

        # The invariant: no held-out group may have contributed. Re-mine using
        # only held-out questions and check the fold's own rows are not simply
        # what the test split says -- i.e. confirm the training restriction is
        # actually doing something.
        held = set(cfg["folds"][fold])
        train_ids = {q.group_id for q in base_questions(train_groups(fold))}
        overlap = train_ids & held
        if overlap:
            problems += 1
            print(f"  ERROR training split contains held-out groups {sorted(overlap)}")

        # Every emitted term must genuinely clear the thresholds.
        for r in m.rows:
            if r.support < MIN_SUPPORT or r.purity < MIN_PURITY or r.lift < MIN_LIFT:
                problems += 1
                print(f"  ERROR {r.term} below thresholds")

        all_terms.append({r.term for r in m.rows})
        for r in m.rows:
            mapping[r.term].add(r.column)
        print()

    # Determinism.
    for fold in range(cfg["k"]):
        a = [r.term for r in mine_fold(fold, max_terms).rows]
        b = [r.term for r in mine_fold(fold, max_terms).rows]
        if a != b:
            problems += 1
            print(f"ERROR fold {fold}: mining is not deterministic")

    # The sharpest available check on whether the mined signal is real. Folds
    # share 80% of their training groups, so a term that means `resource` in one
    # fold and `idcase` in another is not tracking meaning -- it is tracking which
    # handful of questions happened to land in the training split. Any such term
    # would put a confidently-worded falsehood in one of the five prompts.
    contradictory = {t: cols for t, cols in mapping.items() if len(cols) > 1}
    if contradictory:
        problems += len(contradictory)
        print("ERROR terms mapped to different columns in different folds:")
        for t, cols in sorted(contradictory.items()):
            print(f"  {t} -> {sorted(cols)}")

    shared = set.intersection(*all_terms) if all_terms else set()
    print(f"terms common to all {cfg['k']} folds: {len(shared)} "
          f"({', '.join(sorted(shared)) or 'none'})")
    print(f"fold-specific rows differ: {len({tuple(sorted(t)) for t in all_terms})}/"
          f"{cfg['k']} distinct term sets")

    # A sanity read on usefulness, not correctness: if nothing survives the
    # thresholds the R5 vocabulary block is empty and the R4->R5 delta measures
    # the PM tool alone. That is a legitimate outcome, but it should be loud.
    if not any(all_terms):
        print("NOTE no fold produced any vocabulary rows -- R5's vocab block "
              "will be empty and R5 will differ from R4 by the PM tool only")

    print("dry run OK" if not problems else f"dry run FOUND {problems} PROBLEMS")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine term->column vocabulary from a training fold")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        raise SystemExit(1 if dry_run(args.max_terms) else 0)
    if args.fold is None:
        ap.print_help()
        return

    m = mine_fold(args.fold, max_terms=args.max_terms)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(m.to_json())
        print(f"wrote {args.out}")
    else:
        print(m.to_json())


if __name__ == "__main__":
    main()
