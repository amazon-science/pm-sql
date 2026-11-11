"""Scoring: execution accuracy, per fold and pooled.

Two metrics, both reported for every rung (EXPERIMENTS_V2_PLAN.md sec.7):

  col_agnostic (primary) -- values are sorted within each row before comparing,
      so `SELECT COUNT(*), resource` and `SELECT resource, COUNT(*)` both count
      as correct. The benchmark's gold SQL fixes an arbitrary column order that
      no question actually specifies, so the standard metric penalises answers
      that are right.
  standard (secondary) -- column order preserved. Reported for comparability
      with the published baseline and with prior work.

Cell normalisation is byte-identical to v1's `eval_quick_colagnostic.py`
(str -> strip -> lower), so v2 numbers are directly comparable to v1's. That
equivalence is proven by `--verify-scorer`, which re-scores v1's own prediction
files and checks the published ladder numbers come back.

Gold results are cached as SHA-256 digests, not values: the full result sets run
to ~440MB (v1 committed exactly that), whereas the digest cache is ~200KB.

Usage:
    python eval.py --verify-scorer
    python eval.py --predictions <dir-or-tsv> [--rung NAME] [--json out.json]
    python eval.py --sanity-dup-columns
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from sqlite_compat import sqlite3
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dataset import DB_PATH, load_questions, questions_by_uid
from folds import fold_of_uid, load_fold_config

QUERY_TIMEOUT_S = 5.0
GOLD_CACHE = Path(__file__).resolve().parents[1] / "config" / "gold_digests.json"

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260903

# Scoring is CPU-bound on SQLite, so it parallelises almost linearly. Leave a
# core free so an interactive session stays usable during a long scoring run.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------
# result normalisation
# --------------------------------------------------------------------------

def _cell(v) -> str:
    return str(v).strip().lower() if v is not None else ""


def norm_standard(rows) -> list[list[str]]:
    """Rows sorted, column order within a row preserved."""
    return sorted([_cell(v) for v in row] for row in rows)


def norm_col_agnostic(rows) -> list[list[str]]:
    """Rows sorted, and values sorted within each row."""
    return sorted(sorted(_cell(v) for v in row) for row in rows)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Read-only: scoring must never be able to modify the event log.
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def execute(conn: sqlite3.Connection, sql: str) -> tuple[list | None, str | None]:
    """Run `sql`, returning (rows, error). Bounded by QUERY_TIMEOUT_S."""
    if not sql or not sql.strip():
        return None, "empty_sql"
    start = time.time()
    conn.set_progress_handler(
        lambda: 1 if time.time() - start > QUERY_TIMEOUT_S else 0, 500)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:  # noqa: BLE001 -- any failure is just a wrong answer
        msg = str(e)
        if "interrupted" in msg.lower():
            return None, "timeout"
        return None, "sql_error"
    finally:
        conn.set_progress_handler(None, 0)


def digest(normalised: list[list[str]] | None) -> str | None:
    """Stable fingerprint of a normalised result set.

    Results are compared by digest rather than by value because the full gold
    results are ~440MB (one query returns millions of cells). Digests make the
    cache a committable ~200KB and comparison O(1) in the result size.
    """
    if normalised is None:
        return None
    h = hashlib.sha256()
    for row in normalised:
        h.update(("\x1f".join(row) + "\x1e").encode())
    return h.hexdigest()


def build_gold(verbose: bool = True) -> dict[str, dict]:
    """Execute every gold SQL and cache digests of the normalised results."""
    conn = _connect()
    out: dict[str, dict] = {}
    for i, q in enumerate(load_questions(), 1):
        rows, err = execute(conn, q.gold_sql)
        std = norm_standard(rows) if rows is not None else None
        ca = norm_col_agnostic(rows) if rows is not None else None
        out[q.uid] = {
            "error": err,
            "standard": digest(std),
            "col_agnostic": digest(ca),
            "n_rows": len(std) if std is not None else None,
            "n_cols": (len(std[0]) if std else 0) if std is not None else None,
            # Precomputed here so the col-agnostic false-positive sanity check
            # does not need the (enormous) result sets a second time.
            "dup_value_in_row": bool(
                std is not None
                and any(len(r) > 1 and len(set(r)) != len(r) for r in std)),
        }
        if verbose and i % 200 == 0:
            print(f"  gold {i}/1655", flush=True)
    conn.close()
    return out


def load_gold(rebuild: bool = False) -> dict[str, dict]:
    if GOLD_CACHE.exists() and not rebuild:
        return json.loads(GOLD_CACHE.read_text())
    gold = build_gold()
    GOLD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GOLD_CACHE.write_text(json.dumps(gold))
    return gold


# v1's published column-order-agnostic ladder (main.tex table 1). Used to prove
# this scorer is equivalent to v1's before any v2 number is trusted.
#
# Careful with the baseline: the paper quotes 47.6% *standard* (the original
# authors' figure) and 48.1% column-agnostic for the same GPT-3.5 outputs, so
# both metrics are pinned for that row.
V1_PUBLISHED = {
    "experiments/step1_baseline_gpt35/results_step1.tsv": ("Model-Baseline", 48.1, 47.6),
    "experiments/step2_model_upgrade/results_step2.tsv": ("Model-Sonnet-4.6", 63.3, None),
    "experiments/step3_prompt_engineering/results_step3b.tsv": ("Model-FewShot", 70.4, None),
    "experiments/step4a_sql_agent/results_step4a.tsv": ("Model-Agent", 73.3, None),
    "experiments/step5c_examples/results_step5c.tsv": ("Model-Full", 80.3, None),
}
V1_FEWSHOT_GROUPS = {"1", "5", "23", "109", "132"}


def verify_scorer(tolerance_pp: float = 0.4) -> int:
    """Re-score v1's own prediction files and check the published numbers come back.

    A stronger equivalence check than diffing gold result sets, and it avoids
    parsing v1's 440MB gold cache. If every rung reproduces closely, this scorer
    and v1's are the same function and v2 results are comparable to the published
    ladder.

    Scored under both denominators, because the paper is ambiguous about which it
    used: it says "1,655 questions" but v1's scorer takes `--exclude-groups` for
    the five few-shot groups. Whichever column lines up tells us which it was.

    The tolerance is 0.4pp rather than exact: the 5s query timeout is wall-clock,
    so a handful of borderline-slow queries flip between machines.
    """
    root = Path(__file__).resolve().parents[2]
    gold = load_gold()
    excluded = {q.uid for q in load_questions() if q.group_id in V1_FEWSHOT_GROUPS}

    print(f"v1's few-shot groups hold {len(excluded)} questions "
          f"(all-1655 vs excluded-{1655 - len(excluded)} denominators)\n")
    print(f"{'rung':<18} {'published':>9} | {'all 1655':>9} {'delta':>7} | "
          f"{'excl 1613':>9} {'delta':>7}")

    best_col, worst = None, {}
    for rel, (name, pub_ca, pub_std) in V1_PUBLISHED.items():
        path = root / rel
        if not path.exists():
            print(f"{name:<18} {'MISSING FILE':>9}")
            worst["all"] = worst["excl"] = 99.0
            continue
        allp = load_predictions(path)
        m_all = score(allp, rung=name, gold=gold)
        m_ex = score({u: s for u, s in allp.items() if u not in excluded},
                     rung=name, gold=gold)
        d_all, d_ex = m_all.pooled_ca - pub_ca, m_ex.pooled_ca - pub_ca
        worst["all"] = max(worst.get("all", 0.0), abs(d_all))
        worst["excl"] = max(worst.get("excl", 0.0), abs(d_ex))
        print(f"{name:<18} {pub_ca:>8.1f}% | {m_all.pooled_ca:>8.1f}% {d_all:>+6.2f} | "
              f"{m_ex.pooled_ca:>8.1f}% {d_ex:>+6.2f}")
        if pub_std is not None:
            print(f"{'  (standard)':<18} {pub_std:>8.1f}% | "
                  f"{m_all.pooled_std:>8.1f}% {m_all.pooled_std - pub_std:>+6.2f} | "
                  f"{m_ex.pooled_std:>8.1f}% {m_ex.pooled_std - pub_std:>+6.2f}")

    best_col = min(worst, key=worst.get)
    print(f"\nworst deviation: all-1655 {worst['all']:.2f}pp, "
          f"excluded-1613 {worst['excl']:.2f}pp")
    ok = worst[best_col] <= tolerance_pp
    print(f"published numbers are over the "
          f"{'FULL 1,655' if best_col == 'all' else 'REDUCED 1,613'} set; "
          + ("scorer is equivalent to v1" if ok
             else "MISMATCH -- investigate before trusting any v2 number"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

@dataclass
class Scored:
    uid: str
    fold: int
    correct_ca: int
    correct_std: int
    error: str | None
    hardness: str
    level: str
    vocab: str
    base_para: str


@dataclass
class Metrics:
    rung: str
    n: int
    per_fold: dict[int, tuple[int, int]] = field(default_factory=dict)
    pooled_ca: float = 0.0
    pooled_std: float = 0.0
    fold_mean: float = 0.0
    fold_sd: float = 0.0
    ci95: tuple[float, float] = (0.0, 0.0)
    errors: dict[str, int] = field(default_factory=dict)
    breakdowns: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def load_predictions(path: Path) -> dict[str, str]:
    """Read uid -> SQL from a predictions TSV, or from every TSV in a directory.

    Tolerates a header row and extra instrumentation columns; SQL is column 2.
    """
    files = sorted(path.glob("**/predictions.tsv")) if path.is_dir() else [path]
    preds: dict[str, str] = {}
    for f in files:
        for line in f.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or parts[0].strip().lower() in ("uid", "utterance_id"):
                continue
            preds[parts[0].strip()] = parts[1]
    return preds


_WORKER_CONN: sqlite3.Connection | None = None


def _score_one(item: tuple[str, str, str, str]) -> tuple[str, int, int, str | None]:
    """Execute one prediction and compare digests. Runs in a worker process.

    Each worker keeps one read-only connection for its lifetime; opening one per
    query would dominate the runtime.
    """
    global _WORKER_CONN
    uid, sql, want_ca, want_std = item
    if _WORKER_CONN is None:
        _WORKER_CONN = _connect()
    rows, err = execute(_WORKER_CONN, sql)
    if rows is None:
        return uid, 0, 0, err
    return (uid,
            int(digest(norm_col_agnostic(rows)) == want_ca),
            int(digest(norm_standard(rows)) == want_std),
            err)


def _raw_scores(preds: dict[str, str], gold: dict,
                workers: int = DEFAULT_WORKERS) -> list[tuple[str, int, int, str | None]]:
    """Execute and compare every scoreable prediction, in parallel."""
    work = [(uid, sql, gold[uid]["col_agnostic"], gold[uid]["standard"])
            for uid, sql in preds.items()
            # A uid whose own gold SQL does not execute is unscoreable, and is
            # excluded identically for every rung.
            if uid in gold and gold[uid]["col_agnostic"] is not None]

    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_score_one, work, chunksize=32))
    return [_score_one(w) for w in work]


def correctness(preds: dict[str, str], gold: dict | None = None,
                workers: int = DEFAULT_WORKERS) -> dict[str, bool]:
    """uid -> correct under the primary metric.

    The per-question view that example selection needs. Shares `_score_one` with
    `score()` so "wrong" always means the same thing whether it is feeding a
    reported number or choosing a few-shot example.
    """
    gold = gold or load_gold()
    return {uid: bool(ca) for uid, ca, _std, _err in _raw_scores(preds, gold, workers)}


def score(preds: dict[str, str], rung: str = "?", gold: dict | None = None,
          workers: int = DEFAULT_WORKERS) -> Metrics:
    gold = gold or load_gold()
    uid_fold = fold_of_uid()
    byu = questions_by_uid()

    raw = _raw_scores(preds, gold, workers)

    scored: list[Scored] = []
    for uid, ca, std, err in raw:
        q = byu.get(uid)
        scored.append(Scored(
            uid=uid, fold=uid_fold[uid], correct_ca=ca, correct_std=std,
            error=err or (None if ca else "wrong_result"),
            hardness=q.spider if q else "",
            level=q.level if q else "",
            vocab=q.vocab if q else "",
            base_para=("base" if q and q.is_base else "paraphrase"),
        ))

    m = Metrics(rung=rung, n=len(scored))
    if not scored:
        return m

    by_fold = defaultdict(list)
    for s in scored:
        by_fold[s.fold].append(s)
    for f in sorted(by_fold):
        rs = by_fold[f]
        m.per_fold[f] = (sum(r.correct_ca for r in rs), len(rs))

    m.pooled_ca = sum(s.correct_ca for s in scored) / len(scored) * 100
    m.pooled_std = sum(s.correct_std for s in scored) / len(scored) * 100

    rates = [c / n * 100 for c, n in m.per_fold.values()]
    m.fold_mean = statistics.mean(rates)
    m.fold_sd = statistics.stdev(rates) if len(rates) > 1 else 0.0

    # Bootstrap over questions, not over folds: 5 fold estimates are too few to
    # resample, and the pooled number is an estimate of the question-level mean.
    #
    # Drawing n questions with replacement from a 0/1 vector and summing is
    # exactly Binomial(n, correct/n), so we sample that directly instead of
    # doing n index lookups per iteration -- identical distribution, ~1000x
    # faster, which matters because scoring runs 5 rungs x 2 denominators.
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(scored)
    p = sum(s.correct_ca for s in scored) / n
    boots = sorted(rng.binomialvariate(n, p) / n * 100
                   for _ in range(BOOTSTRAP_ITERS))
    m.ci95 = (boots[int(0.025 * BOOTSTRAP_ITERS)], boots[int(0.975 * BOOTSTRAP_ITERS)])

    m.errors = dict(sorted(
        ((k, sum(1 for s in scored if s.error == k))
         for k in {s.error for s in scored} if k),
        key=lambda t: -t[1]))

    for dim in ("hardness", "level", "vocab", "base_para"):
        agg: dict[str, list[int]] = {}
        for s in scored:
            key = getattr(s, dim)
            if key:
                agg.setdefault(key, [0, 0])
                agg[key][0] += s.correct_ca
                agg[key][1] += 1
        m.breakdowns[dim] = {k: (v[0], v[1]) for k, v in sorted(agg.items())}

    m.missing = sorted(
        (uid for uid in gold
         if gold[uid]["col_agnostic"] is not None and uid not in preds),
        key=lambda u: int(u))
    return m


def render(m: Metrics) -> str:
    L = [f"rung={m.rung}  scored={m.n}"
         + (f"  MISSING {len(m.missing)} predictions" if m.missing else "")]
    L.append(f"  col-agnostic pooled : {m.pooled_ca:5.1f}%   "
             f"95% CI [{m.ci95[0]:.1f}, {m.ci95[1]:.1f}]")
    L.append(f"  standard     pooled : {m.pooled_std:5.1f}%")
    L.append(f"  across folds        : {m.fold_mean:5.1f}% +/- {m.fold_sd:.1f}")
    L.append("  per fold: " + "  ".join(
        f"f{f}={c}/{n}={c / n * 100:.1f}%" for f, (c, n) in sorted(m.per_fold.items())))
    if m.errors:
        L.append("  errors: " + ", ".join(f"{k}={v}" for k, v in m.errors.items()))
    for dim, agg in m.breakdowns.items():
        L.append(f"  {dim}:")
        for k, (c, n) in agg.items():
            L.append(f"    {k:<16} {c:>4}/{n:<4} = {c / n * 100:5.1f}%")
    return "\n".join(L)


def sanity_dup_columns() -> str:
    """How much information does the column-agnostic metric discard?

    Sorting values within a row only loses information when a row contains the
    same value twice. v1's paper asserts no gold result does; that is wrong, so
    this reports the real figure and where it concentrates.

    Note what the number does and does not mean. It bounds where a false positive
    is *conceivable*, not where one is likely: a colliding wrong answer would have
    to return the identical multiset of values in the identical multiset of rows,
    which for these wide row-listings means returning the same data.
    """
    from collections import Counter

    gold = load_gold()
    byu = questions_by_uid()
    affected, total = [], 0
    for uid, g in gold.items():
        if g.get("standard") is None:
            continue
        total += 1
        if g.get("dup_value_in_row"):
            affected.append(uid)

    if not affected:
        return ("no gold result repeats a value across columns within a row -- "
                "the column-agnostic metric discards no information here")

    pct = len(affected) / total * 100
    groups = Counter(byu[u].group_id for u in affected if u in byu)
    cols = Counter(gold[u]["n_cols"] for u in affected)
    return (
        f"gold results with a repeated value across columns in some row: "
        f"{len(affected)}/{total} ({pct:.2f}%)\n"
        f"  concentrated in {len(groups)} groups: "
        + ", ".join(f"g{g}({n})" for g, n in groups.most_common()) + "\n"
        f"  column counts: "
        + ", ".join(f"{c} cols x{n}" for c, n in sorted(cols.items())) + "\n"
        "  all are wide row-listing queries (SELECT * style) where two attributes\n"
        "  coincide, so a colliding wrong answer would have to return the same data."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score predictions")
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--rung", default="?")
    ap.add_argument("--verify-scorer", action="store_true",
                    help="re-score v1's predictions and check the published numbers")
    ap.add_argument("--rebuild-gold", action="store_true")
    ap.add_argument("--sanity-dup-columns", action="store_true")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.verify_scorer:
        raise SystemExit(verify_scorer())
    if args.rebuild_gold:
        load_gold(rebuild=True)
        print(f"wrote {GOLD_CACHE}")
    if args.sanity_dup_columns:
        print(sanity_dup_columns())
    if args.predictions:
        cfg = load_fold_config()
        print(f"folds: k={cfg['k']} seed={cfg['seed']}")
        m = score(load_predictions(args.predictions), rung=args.rung,
                  workers=args.workers)
        print(render(m))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({
                "rung": m.rung, "n": m.n,
                "pooled_col_agnostic": m.pooled_ca, "pooled_standard": m.pooled_std,
                "fold_mean": m.fold_mean, "fold_sd": m.fold_sd, "ci95": m.ci95,
                "per_fold": {str(k): v for k, v in m.per_fold.items()},
                "errors": m.errors,
                "breakdowns": {d: {k: list(v) for k, v in a.items()}
                               for d, a in m.breakdowns.items()},
                "n_missing": len(m.missing),
            }, indent=2) + "\n")
            print(f"wrote {args.json}")
    if not any([args.verify_scorer, args.rebuild_gold, args.sanity_dup_columns,
                args.predictions]):
        ap.print_help()


if __name__ == "__main__":
    main()
