"""Deterministic few-shot example selection from a training fold.

Why this exists (EXPERIMENTS_V2_PLAN.md sec.4). v1 chose its examples by hand
after reading failures on the evaluation set -- the thing the reviewers object
to. Hand-authoring five fold-specific prompt variants instead would just move
that discretion around, and add four more prompts of unlogged human judgement.

So selection becomes a function of the training fold: cluster the training-fold
failures by the structural signature of their gold SQL, take the largest
clusters, and emit one representative each. Same fold and same prior-rung
predictions always yield the same examples, and the cluster sizes are the
provenance record that answers "why these examples?".

The random control (R3c) is the same function with the ranking replaced by a
uniform sample, so R3 vs R3c differs in exactly one thing: the criterion.

Usage:
    python example_selector.py --dry-run          # all folds, v1 step2 as prior rung
    python example_selector.py --fold 0 --k 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from dataset import Question, base_questions, questions_by_uid
from folds import load_fold_config, train_groups

DEFAULT_K = 5
DEFAULT_RANDOM_SEED = 20260903


# --------------------------------------------------------------------------
# structural signature of a gold SQL statement
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Signature:
    """What a query *requires*, coarse enough that clusters are populated.

    Deliberately excludes anything about the question's wording: two questions
    needing the same SQL machinery belong in the same cluster even if they read
    nothing alike, which is the whole point of grouping by failure mode.
    """
    self_join: bool
    window_fn: bool
    cte: bool
    subquery: bool
    having: bool
    duration: bool
    date_literal: bool
    distinct: bool
    agg: str          # none | count | sum | avg | min | max | mixed
    spider: str
    vocab: str

    def key(self) -> tuple:
        return tuple(asdict(self).values())

    def label(self) -> str:
        flags = [n for n in ("self_join", "window_fn", "cte", "subquery", "having",
                             "duration", "date_literal", "distinct")
                 if getattr(self, n)]
        return f"{self.spider}/{self.vocab}/agg={self.agg}" + (
            "/" + "+".join(flags) if flags else "")


_AGGS = ("count", "sum", "avg", "min", "max")


def signature(q: Question) -> Signature:
    s = " ".join(q.gold_sql.lower().split())
    aggs = [a for a in _AGGS if re.search(rf"\b{a}\s*\(", s)]
    return Signature(
        # A self-join shows up as either an explicit JOIN or two aliases of the
        # same table in the FROM list.
        self_join=bool(re.search(r"\bjoin\b", s)
                       or re.search(r"event_log\s+\w+\s*,\s*event_log\s+\w+", s)),
        window_fn=bool(re.search(r"\bover\s*\(|\blead\s*\(|\blag\s*\(|row_number", s)),
        cte=bool(re.search(r"\bwith\b\s+\w+\s+as\s*\(", s)),
        subquery=s.count("select") > 1,
        having=" having " in s,
        duration=bool(re.search(r"strftime\s*\(\s*'%s'|julianday", s)),
        date_literal=bool(re.search(r"'\d{4}-\d{2}|'\d{2}/\d{2}/\d{4}", s)),
        distinct=" distinct" in s,
        agg="none" if not aggs else (aggs[0] if len(aggs) == 1 else "mixed"),
        spider=q.spider,
        vocab=q.vocab,
    )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

@dataclass
class Example:
    uid: str
    group_id: str
    question: str
    gold_sql: str
    cluster: str
    cluster_failures: int


@dataclass
class Selection:
    fold: int
    strategy: str            # failure_signature | random
    k: int
    examples: list[Example]
    n_train_base: int
    n_train_failures: int
    n_clusters: int
    seed: int | None = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2) + "\n"


def _median_representative(qs: list[Question]) -> Question:
    """The cluster member of median gold-SQL length.

    Median rather than shortest: the shortest member of a cluster is often a
    degenerate case that does not demonstrate the pattern. Tie-broken on uid so
    the choice is reproducible.
    """
    ordered = sorted(qs, key=lambda q: (len(q.gold_sql.split()), int(q.uid)))
    return ordered[len(ordered) // 2]


def select_failure_driven(
    fold: int,
    wrong_uids: set[str],
    k: int = DEFAULT_K,
    exclude_uids: set[str] | None = None,
) -> Selection:
    """Cluster training-fold failures by gold-SQL signature; one example per cluster.

    `wrong_uids` are the questions a prior rung answered incorrectly. Only
    training-fold base questions are considered: paraphrases in a group share a
    gold SQL, so they land in the same cluster and add nothing but weight.
    """
    train = train_groups(fold)
    exclude = exclude_uids or set()
    candidates = [q for q in base_questions(train)
                  if q.uid in wrong_uids and q.uid not in exclude]

    clusters: dict[tuple, list[Question]] = {}
    labels: dict[tuple, str] = {}
    for q in candidates:
        sig = signature(q)
        clusters.setdefault(sig.key(), []).append(q)
        labels[sig.key()] = sig.label()

    # Largest cluster first; ties broken on the signature itself for determinism.
    ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))

    examples = [
        Example(uid=(r := _median_representative(qs)).uid, group_id=r.group_id,
                question=r.question, gold_sql=r.gold_sql,
                cluster=labels[key], cluster_failures=len(qs))
        for key, qs in ranked[:k]
    ]
    return Selection(
        fold=fold, strategy="failure_signature", k=k, examples=examples,
        n_train_base=len(base_questions(train)),
        n_train_failures=len(candidates), n_clusters=len(clusters),
    )


def select_random(
    fold: int,
    k: int = DEFAULT_K,
    seed: int = DEFAULT_RANDOM_SEED,
    exclude_uids: set[str] | None = None,
) -> Selection:
    """The R3c control: uniform sample of training-fold base questions.

    Deliberately ignores which questions the prior rung got wrong, so any gap
    between R3 and R3c is attributable to the selection criterion alone.
    """
    train = train_groups(fold)
    exclude = exclude_uids or set()
    pool = sorted((q for q in base_questions(train) if q.uid not in exclude),
                  key=lambda q: int(q.uid))
    rng = random.Random(seed + fold)
    picked = rng.sample(pool, min(k, len(pool)))
    picked.sort(key=lambda q: int(q.uid))
    return Selection(
        fold=fold, strategy="random", k=k, seed=seed + fold,
        examples=[Example(uid=q.uid, group_id=q.group_id, question=q.question,
                          gold_sql=q.gold_sql, cluster=signature(q).label(),
                          cluster_failures=0) for q in picked],
        n_train_base=len(pool), n_train_failures=0, n_clusters=0,
    )


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------

V1_WRONG_CACHE = Path(__file__).resolve().parents[1] / "config" / "v1_step2_wrong.json"


def wrong_uids(predictions: Path) -> set[str]:
    """Uids whose prediction is wrong under the primary metric.

    Shared by the dry-run fallback and by real runs deriving one rung's examples
    from the previous rung's errors, so both use identical correctness criteria.
    """
    from eval import correctness, load_predictions

    return {uid for uid, ok in correctness(load_predictions(predictions)).items()
            if not ok}


@lru_cache(maxsize=1)
def wrong_uids_from_v1(rel: str = "experiments/step2_model_upgrade/results_step2.tsv") -> frozenset:
    """Stand-in for a real R1/R2 prediction set, for offline dry runs.

    Uses v1's step-2 predictions -- the closest v1 analogue of R2 -- scored with
    the v2 scorer. Only for exercising the pipeline before any v2 rung exists.

    Cached to disk as well as in memory: scoring 1,655 queries takes minutes, and
    a dry run of the full ladder would otherwise repeat it once per rung. The
    result is a pure function of two checked-in files, so the cache cannot go
    stale without one of them changing.
    """
    root = Path(__file__).resolve().parents[2]
    if V1_WRONG_CACHE.exists():
        return frozenset(json.loads(V1_WRONG_CACHE.read_text())["wrong_uids"])

    out = wrong_uids(root / rel)
    V1_WRONG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    V1_WRONG_CACHE.write_text(json.dumps(
        {"source": rel, "metric": "col_agnostic", "n_wrong": len(out),
         "wrong_uids": sorted(out, key=int)}, indent=2) + "\n")
    return frozenset(out)


def dry_run(k: int) -> int:
    cfg = load_fold_config()
    print(f"scoring v1 step-2 predictions to stand in for R1/R2 failures ...")
    wrong = wrong_uids_from_v1()
    print(f"{len(wrong)} incorrect questions available as failure signal\n")

    byu = questions_by_uid()
    problems = 0
    for fold in range(cfg["k"]):
        sel = select_failure_driven(fold, wrong, k=k)
        rnd = select_random(fold, k=k)
        print(f"--- fold {fold}: {sel.n_train_failures} base failures in "
              f"{sel.n_train_base} train base questions, {sel.n_clusters} clusters")
        for e in sel.examples:
            print(f"  [{e.cluster_failures:>3} fails] g{e.group_id:<4} {e.cluster}")
            print(f"      {e.question[:72]}")
        print(f"  random control: " + ", ".join(f"g{e.group_id}" for e in rnd.examples))

        # The invariant that matters: nothing selected may be in the test fold.
        held = set(cfg["folds"][fold])
        for name, s in (("failure", sel), ("random", rnd)):
            bad = [e.uid for e in s.examples if byu[e.uid].group_id in held]
            if bad:
                problems += 1
                print(f"  ERROR {name}: selected held-out uids {bad}")
        if len({e.uid for e in sel.examples}) != len(sel.examples):
            problems += 1
            print("  ERROR duplicate examples selected")
        print()

    # Determinism check.
    for fold in range(cfg["k"]):
        a = select_failure_driven(fold, wrong, k=k)
        b = select_failure_driven(fold, wrong, k=k)
        if [e.uid for e in a.examples] != [e.uid for e in b.examples]:
            problems += 1
            print(f"ERROR fold {fold}: selection is not deterministic")

    # Fold-to-fold overlap is expected to be high (folds share 80% of groups);
    # what would be wrong is identical selections regardless of fold.
    sels = [tuple(e.uid for e in select_failure_driven(f, wrong, k=k).examples)
            for f in range(cfg["k"])]
    print(f"distinct selections across folds: {len(set(sels))}/{cfg['k']}")

    print("dry run OK" if not problems else f"dry run FOUND {problems} PROBLEMS")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Select few-shot examples from a training fold")
    ap.add_argument("--fold", type=int)
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--random", action="store_true", help="the R3c control")
    ap.add_argument("--wrong-uids", type=Path,
                    help="file of uids the prior rung got wrong, one per line")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        raise SystemExit(1 if dry_run(args.k) else 0)
    if args.fold is None:
        ap.print_help()
        return

    if args.random:
        sel = select_random(args.fold, k=args.k)
    else:
        if not args.wrong_uids:
            raise SystemExit("--wrong-uids is required unless --random")
        wrong = {l.strip() for l in args.wrong_uids.read_text().splitlines() if l.strip()}
        sel = select_failure_driven(args.fold, wrong, k=args.k)

    out = sel.to_json()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"wrote {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
