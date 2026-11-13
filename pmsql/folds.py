"""Answer-unit-level k-fold construction.

Design:
  - The *answer unit* is the atomic unit, and it is wholly in one fold. See below
    -- this is not the same thing as the paraphrase group, and treating it as such
    silently leaks answers.
  - Stratify on Spider class, which is group-constant.
  - Within each Spider class, assign units to folds by greedy bin-packing on SIZE
    (largest first). Unit sizes run 2..60, so assigning equal unit COUNTS per fold
    would leave folds differing ~20% in question count and make per-fold
    accuracies incomparable.
  - Vocab type varies within a group, so it cannot be stratified on. It is
    reported post-hoc and the fold set is rejected if any fold drifts too far
    from the global vocab mix.

Why units and not groups. The unit was originally the paraphrase group, on the
reasoning that a group shares one gold SQL so a paraphrase can never appear in
the prompt that answers it. Running the leakage gate over all 25 fold-prompts
falsified that twice over.

  1. 27 of the 205 groups share a *gold SQL* with a different group -- there are
     only 190 distinct gold statements. Group 91's "Which activities appear most
     often in the event log?" and group 15's "...most frequently..." are one
     question under two group ids, so splitting them put a held-out answer
     verbatim into a training example. dataset.load_answer_classes merges these.

  2. Identical SQL is still too weak, because the metric is execution accuracy:
     what leaks is the *result*, not the text. Group 8's "What is the total
     number of events related to each resource?" (`SELECT COUNT(*), resource
     ...`) and group 12's "In how many activities has each resource already
     acted?" (`SELECT resource, COUNT(*) ...`) are different questions whose gold
     SQL differs only in column order -- so under the column-agnostic primary
     metric they have the same answer, and one in the prompt scores the other.
     Merging by gold result digest takes 190 text classes down to 178 units.

Merging on the result digest is the correct unit precisely because it is the
relation the metric uses to award credit. It over-merges in one direction only:
two unrelated questions that happen to coincide on this 72k-event database (see
the group-121 note in leakage_gate) land in the same unit, which costs a little
fold-balance freedom and leaks nothing. Literal-differing templates (`WHERE
resource='peter'` vs `='sarah'`) return different results and stay separate.

Every question is held out in exactly one fold, so pooled accuracy covers all
1,655 questions with nothing excluded -- unlike v1, which had to drop 42
questions whose groups supplied the few-shot examples.

The output is written once to config/folds.json and committed. Never regenerate
it for a published run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dataset import (SPIDER_CLASSES, VOCAB_TYPES, load_answer_classes,
                     load_groups, load_questions)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
FOLDS_PATH = CONFIG_DIR / "folds.json"
UNITS_PATH = CONFIG_DIR / "answer_units.json"

DEFAULT_K = 5
DEFAULT_SEED = 20260903
# Reject a fold set if any fold's vocab mix drifts more than this from global.
VOCAB_TOLERANCE_PP = 5.0


# Relative weights for the three balance objectives. Question count and vocab
# matter most: count makes per-fold accuracies comparable, and vocab drift was
# 16pp under count-only packing because vocab correlates with group size.
W_COUNT, W_VOCAB, W_SPIDER = 1.0, 1.0, 0.5


@dataclass(frozen=True)
class Unit:
    """An indivisible split unit: groups that share a gold *answer*."""
    unit_id: str
    group_ids: tuple[str, ...]
    spider: str
    questions: tuple

    @property
    def size(self) -> int:
        return len(self.questions)


def build_units(verbose: bool = False) -> tuple[Unit, ...]:
    """Answer classes further merged wherever two share a gold result digest.

    Requires the SQLite DB, via eval's cached gold digests. Imported lazily
    because eval imports this module. If digests are unavailable this raises
    rather than silently degrading to text-only merging: a fold set built without
    the denotational merge leaks, and that must not happen quietly.
    """
    from eval import load_gold  # lazy: eval imports folds

    classes = load_answer_classes()
    gold = load_gold()

    parent = {c.class_id: c.class_id for c in classes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=int)
            parent[hi] = lo

    seen: dict[str, str] = {}
    n_merges = 0
    for c in classes:
        for q in c.questions:
            d = gold.get(q.uid, {}).get("col_agnostic")
            if not d:
                continue
            if d in seen:
                if find(seen[d]) != find(c.class_id):
                    n_merges += 1
                union(seen[d], c.class_id)
            else:
                seen[d] = c.class_id

    members: dict[str, list[str]] = {}
    for c in classes:
        members.setdefault(find(c.class_id), []).append(c.class_id)

    by_cid = {c.class_id: c for c in classes}
    units = []
    for root, cids in members.items():
        cids.sort(key=int)
        gids = tuple(sorted((gid for cid in cids for gid in by_cid[cid].group_ids),
                            key=int))
        qs = tuple(q for cid in cids for q in by_cid[cid].questions)
        units.append(Unit(unit_id=root, group_ids=gids,
                          spider=by_cid[cids[0]].spider, questions=qs))
    units.sort(key=lambda u: int(u.unit_id))

    if verbose:
        print(f"{len(classes)} answer classes -> {len(units)} split units "
              f"({n_merges} digest merges)")
    return tuple(units)


@lru_cache(maxsize=1)
def load_units() -> tuple[Unit, ...]:
    """Split units, read from the frozen config/answer_units.json when present.

    Reading the frozen file keeps every consumer (the gate, report.py) off the
    database and guarantees they see the same units the published folds were
    built from.
    """
    if UNITS_PATH.exists():
        data = json.loads(UNITS_PATH.read_text())
        by_gid = {g.group_id: g for g in load_groups()}
        return tuple(
            Unit(unit_id=u["unit_id"], group_ids=tuple(u["group_ids"]),
                 spider=by_gid[u["group_ids"][0]].spider,
                 questions=tuple(q for gid in u["group_ids"]
                                 for q in by_gid[gid].questions))
            for u in data["units"])
    return build_units()


def write_units(units: tuple[Unit, ...]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    UNITS_PATH.write_text(json.dumps({
        "n_groups": len(load_groups()),
        "n_answer_classes": len(load_answer_classes()),
        "n_units": len(units),
        "merge_relation": "identical normalised gold SQL, then identical "
                          "column-agnostic gold result digest",
        "units": [{"unit_id": u.unit_id, "group_ids": list(u.group_ids),
                   "n_questions": u.size} for u in units],
    }, indent=2) + "\n")


def build_folds(k: int = DEFAULT_K, seed: int = DEFAULT_SEED) -> list[list[str]]:
    """Assign group ids to k folds, packing answer units. Deterministic in (k, seed).

    Greedy multi-objective bin-packing: units are placed largest-first (so the
    60-question units land while there is still freedom to compensate), each into
    whichever fold minimises the combined squared relative deviation from
    proportional targets for question count, vocab mix and Spider mix. A unit's
    member groups always land together.
    """
    import random

    rng = random.Random(seed)
    classes = list(load_units())
    n_total = sum(c.size for c in classes)

    tgt_n = n_total / k
    tgt_vocab = {
        v: sum(1 for c in classes for q in c.questions if q.vocab == v) / k
        for v in VOCAB_TYPES
    }
    tgt_spider = {
        c: sum(1 for cl in classes if cl.spider == c) / k
        for c in SPIDER_CLASSES
    }

    folds: list[list[str]] = [[] for _ in range(k)]
    n_cur = [0] * k
    vocab_cur: list[dict[str, int]] = [{v: 0 for v in VOCAB_TYPES} for _ in range(k)]
    spider_cur: list[dict[str, int]] = [{c: 0 for c in SPIDER_CLASSES} for _ in range(k)]

    # Shuffle before the size sort so equal-sized classes are not ordered by id.
    rng.shuffle(classes)
    classes.sort(key=lambda c: -c.size)

    for g in classes:
        g_vocab = {v: sum(1 for q in g.questions if q.vocab == v) for v in VOCAB_TYPES}

        def cost(i: int) -> float:
            # Sum of squared normalised loads, NOT squared distance to the final
            # target: the latter makes greedy fill one fold to target before
            # touching the next, which can leave folds empty. Sum-of-squares is
            # convex, so placing into the least-loaded fold minimises it and the
            # loads spread evenly.
            c = W_COUNT * ((n_cur[i] + g.size) / tgt_n) ** 2
            for v in VOCAB_TYPES:
                t = max(tgt_vocab[v], 1.0)
                c += W_VOCAB * ((vocab_cur[i][v] + g_vocab[v]) / t) ** 2
            t = max(tgt_spider[g.spider], 1.0)
            c += W_SPIDER * ((spider_cur[i][g.spider] + 1) / t) ** 2
            return c

        target = min(range(k), key=lambda i: (cost(i), i))
        folds[target].extend(g.group_ids)
        n_cur[target] += g.size
        for v in VOCAB_TYPES:
            vocab_cur[target][v] += g_vocab[v]
        spider_cur[target][g.spider] += 1

    return [sorted(f, key=int) for f in folds]


_spider_cache: dict[str, str] | None = None
_class_cache: dict[str, str] | None = None


def _spider_of(group_id: str) -> str:
    global _spider_cache
    if _spider_cache is None:
        _spider_cache = {g.group_id: g.spider for g in load_groups()}
    return _spider_cache[group_id]


def _class_of() -> dict[str, str]:
    """group id -> split unit id."""
    global _class_cache
    if _class_cache is None:
        _class_cache = {gid: u.unit_id for u in load_units() for gid in u.group_ids}
    return _class_cache


def split_classes(folds: list[list[str]]) -> list[tuple[str, list[int]]]:
    """Split units whose member groups landed in more than one fold.

    Must always be empty. A non-empty result means a held-out question's gold
    answer is reachable from a training example, which is the defect that
    group-level splitting had and the leakage gate caught.
    """
    cls = _class_of()
    where: dict[str, set[int]] = {}
    for i, gids in enumerate(folds):
        for gid in gids:
            where.setdefault(cls[gid], set()).add(i)
    return [(c, sorted(fs)) for c, fs in sorted(where.items(), key=lambda kv: int(kv[0]))
            if len(fs) > 1]


def fold_report(folds: list[list[str]]) -> dict:
    """Per-fold composition, plus the checks that decide whether to accept."""
    questions = load_questions()
    by_group = defaultdict(list)
    for q in questions:
        by_group[q.group_id].append(q)

    global_vocab = {v: sum(1 for q in questions if q.vocab == v) / len(questions) * 100
                    for v in VOCAB_TYPES}

    report = {"k": len(folds), "n_questions": len(questions),
              "global_vocab_pct": global_vocab, "folds": []}
    worst_drift = 0.0

    for i, gids in enumerate(folds):
        qs = [q for gid in gids for q in by_group[gid]]
        vocab = {v: (sum(1 for q in qs if q.vocab == v) / len(qs) * 100) for v in VOCAB_TYPES}
        drift = max(abs(vocab[v] - global_vocab[v]) for v in VOCAB_TYPES)
        worst_drift = max(worst_drift, drift)
        report["folds"].append({
            "fold": i,
            "n_groups": len(gids),
            "n_units": len({_class_of()[gid] for gid in gids}),
            "n_questions": len(qs),
            "spider": {c: sum(1 for gid in gids if _spider_of(gid) == c)
                       for c in SPIDER_CLASSES},
            "vocab_pct": {v: round(vocab[v], 1) for v in VOCAB_TYPES},
            "vocab_drift_pp": round(drift, 1),
        })

    counts = [f["n_questions"] for f in report["folds"]]
    report["question_count_spread"] = max(counts) - min(counts)
    report["worst_vocab_drift_pp"] = round(worst_drift, 1)
    report["n_answer_classes"] = len(load_answer_classes())
    report["n_units"] = len(load_units())
    report["split_units"] = split_classes(folds)
    # Unit integrity is a correctness invariant, not a quality target: a fold
    # set that splits one is unusable at any drift.
    report["accepted"] = (worst_drift <= VOCAB_TOLERANCE_PP
                          and not report["split_units"])
    return report


def load_fold_config(path: Path | None = None) -> dict:
    p = path or FOLDS_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run: python pmsql/folds.py --write"
        )
    return json.loads(p.read_text())


def test_groups(fold: int, path: Path | None = None) -> set[str]:
    """Group ids held out in this fold (the evaluation split)."""
    return set(load_fold_config(path)["folds"][fold])


def train_groups(fold: int, path: Path | None = None) -> set[str]:
    """Group ids available for prompt derivation in this fold."""
    cfg = load_fold_config(path)
    held_out = set(cfg["folds"][fold])
    return {gid for f in cfg["folds"] for gid in f} - held_out


def test_uids(fold: int, path: Path | None = None) -> set[str]:
    gids = test_groups(fold, path)
    return {q.uid for q in load_questions() if q.group_id in gids}


def train_uids(fold: int, path: Path | None = None) -> set[str]:
    gids = train_groups(fold, path)
    return {q.uid for q in load_questions() if q.group_id in gids}


def fold_of_uid(path: Path | None = None) -> dict[str, int]:
    """uid -> the fold in which that question is held out. Used to pool results."""
    cfg = load_fold_config(path)
    gid_to_fold = {gid: i for i, f in enumerate(cfg["folds"]) for gid in f}
    return {q.uid: gid_to_fold[q.group_id] for q in load_questions()}


def search_folds(k: int, base_seed: int, tries: int) -> tuple[list[list[str]], int, dict]:
    """Try `tries` consecutive seeds, return the fold set with least drift.

    Deterministic given (k, base_seed, tries). Ranking prefers low vocab drift
    first, then a tight question-count spread.
    """
    best = None
    for offset in range(tries):
        seed = base_seed + offset
        folds = build_folds(k, seed)
        rep = fold_report(folds)
        key = (rep["worst_vocab_drift_pp"], rep["question_count_spread"])
        if best is None or key < best[0]:
            best = (key, folds, seed, rep)
    return best[1], best[2], best[3]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and freeze the fold partition")
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--search", type=int, default=1,
                    help="try this many consecutive seeds and keep the best")
    ap.add_argument("--write", action="store_true", help="write config/folds.json")
    ap.add_argument("--rebuild-units", action="store_true",
                    help="recompute answer_units.json from the DB before packing")
    args = ap.parse_args()

    # Units must be current before folds are packed from them. Rebuilding hits
    # the DB; the frozen file is used otherwise so runs are reproducible offline.
    if args.rebuild_units or not UNITS_PATH.exists():
        units = build_units(verbose=True)
        if args.write or not UNITS_PATH.exists():
            write_units(units)
            print(f"wrote {UNITS_PATH}")
        load_units.cache_clear()

    base_seed = args.seed
    if args.search > 1:
        folds, chosen_seed, report = search_folds(args.k, args.seed, args.search)
        print(f"searched {args.search} seeds, best = {chosen_seed}")
        args.seed = chosen_seed
    else:
        folds = build_folds(args.k, args.seed)
        report = fold_report(folds)

    print(f"k={report['k']}  questions={report['n_questions']}  "
          f"classes={report['n_answer_classes']} units={report['n_units']}  "
          f"count spread={report['question_count_spread']}  "
          f"worst vocab drift={report['worst_vocab_drift_pp']}pp  "
          f"accepted={report['accepted']}")
    print(f"{'fold':>4} {'unit':>4} {'groups':>7} {'quest':>6}  "
          f"spider (e/m/h/x/nh)      vocab V/G/D")
    for f in report["folds"]:
        s = f["spider"]
        v = f["vocab_pct"]
        print(f"{f['fold']:>4} {f['n_units']:>4} {f['n_groups']:>7} {f['n_questions']:>6}  "
              f"{s['easy']:>2}/{s['medium']:>2}/{s['hard']:>2}/{s['extra']:>2}/"
              f"{s['no_hardness']:>2}          "
              f"{v['Value']:.0f}/{v['Generic']:.0f}/{v['Domain']:.0f}")

    if report["split_units"]:
        for cid, fs in report["split_units"]:
            print(f"  SPLIT unit {cid} spans folds {fs}")
        raise SystemExit("\nREJECTED: a split unit straddles folds. This is a "
                         "bug in build_folds, not a bad seed.")
    if not report["accepted"]:
        raise SystemExit(
            f"\nREJECTED: vocab drift {report['worst_vocab_drift_pp']}pp exceeds "
            f"{VOCAB_TOLERANCE_PP}pp. Try another --seed."
        )

    if args.write:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        FOLDS_PATH.write_text(json.dumps(
            {"k": args.k, "seed": args.seed, "base_seed": base_seed,
             "search": args.search, "weights": {"count": W_COUNT, "vocab": W_VOCAB,
                                                "spider": W_SPIDER},
             "folds": folds, "report": report},
            indent=2) + "\n")
        print(f"\nWrote {FOLDS_PATH}")
        print("Commit this file. Do not regenerate it for a published run.")
    else:
        print("\n(dry run -- pass --write to freeze)")


if __name__ == "__main__":
    main()
