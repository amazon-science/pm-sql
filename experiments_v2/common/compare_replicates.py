"""Compare two full ladder replicates, scored together on one machine.

Scoring both in the same process matters: the 5 s query timeout is wall-clock,
so a handful of slow queries flip between machines, and comparing a number
scored on a laptop against one scored on a desktop would mix that into the
replicate variance.

Reports, per rung: accuracy in each replicate, run-to-run difference with an
exact McNemar p (same questions, different run), per-question agreement, and
for each ladder step the delta and McNemar p within each replicate -- the
question being whether every step that was significant once is significant
again.

With three or more replicates it also reports, per rung, the mean and range
across runs, the fraction of questions every run agrees on, and, per step,
whether the significance verdict is the same in every run.

Usage:
    python compare_replicates.py runs runs_rep2 [runs_rep3 ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from eval import correctness, load_predictions

RUNGS = ["R0", "R1", "R2", "R3", "R3c", "R4", "R5", "R6"]
STEPS = [("R0", "R1"), ("R1", "R2"), ("R2", "R3"), ("R3c", "R3"), ("R2", "R3c"),
         ("R3", "R4"), ("R4", "R5"), ("R5", "R6"), ("R4", "R6")]


def pooled(root: Path, rung: str) -> dict[str, bool]:
    preds: dict[str, str] = {}
    paths = sorted(root.glob(f"{rung}/predictions.tsv")) + \
        sorted(root.glob(f"{rung}_f*/predictions.tsv"))
    for p in paths:
        for uid, sql in load_predictions(p).items():
            if uid in preds:
                raise SystemExit(f"{root}: uid {uid} predicted twice for {rung}")
            preds[uid] = sql
    if len(preds) != 1655:
        raise SystemExit(f"{root}: {rung} has {len(preds)} predictions, want 1655")
    return correctness(preds)


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """(a wrong -> b right, a right -> b wrong, exact two-sided p)."""
    gained = sum(1 for u in a if not a[u] and b[u])
    lost = sum(1 for u in a if a[u] and not b[u])
    n, k = gained + lost, min(gained, lost)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return gained, lost, p


def pct(d: dict[str, bool]) -> float:
    return sum(d.values()) / len(d) * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", type=Path, nargs="+")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if len(args.reps) < 2:
        raise SystemExit("need at least two replicate directories")

    reps = [{} for _ in args.reps]
    for r in RUNGS:
        for i, root in enumerate(args.reps):
            reps[i][r] = pooled(root, r)
        print(f"scored {r}", flush=True)

    if len(reps) > 2:
        multi(reps, args)
        return
    A, B = reps

    out = {"rungs": {}, "steps": {}}
    print(f"\n{'rung':<5} {'rep A':>6} {'rep B':>6} {'B-A':>6}  "
          f"{'agree':>6}  {'A-only':>6} {'B-only':>6}  p(run-to-run)")
    for r in RUNGS:
        a, b = A[r], B[r]
        agree = sum(1 for u in a if a[u] == b[u]) / len(a) * 100
        gained, lost, p = mcnemar(a, b)
        out["rungs"][r] = {"a": pct(a), "b": pct(b), "agree_pct": agree,
                           "b_only_correct": gained, "a_only_correct": lost, "p": p}
        print(f"{r:<5} {pct(a):6.1f} {pct(b):6.1f} {pct(b) - pct(a):+6.1f}  "
              f"{agree:5.1f}%  {lost:6d} {gained:6d}  {p:.2g}")

    print(f"\n{'step':<11} {'delta A':>8} {'p A':>8}   {'delta B':>8} {'p B':>8}   replicates?")
    for x, y in STEPS:
        res = []
        for R in (A, B):
            _g, _l, p = mcnemar(R[x], R[y])
            res.append((pct(R[y]) - pct(R[x]), p))
        (da, pa), (db, pb) = res
        sig_a, sig_b = pa < 0.05, pb < 0.05
        verdict = ("yes, significant both" if sig_a and sig_b and da * db > 0
                   else "yes, null both" if not sig_a and not sig_b
                   else "NO")
        out["steps"][f"{x}->{y}"] = {"delta_a": da, "p_a": pa, "delta_b": db,
                                     "p_b": pb, "replicates": verdict}
        print(f"{x + '->' + y:<11} {da:+8.1f} {pa:8.2g}   {db:+8.1f} {pb:8.2g}   {verdict}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


def multi(reps: list[dict], args) -> None:
    k = len(reps)
    out = {"replicates": [str(r) for r in args.reps], "rungs": {}, "steps": {}}
    heads = " ".join(f"{'run' + str(i + 1):>6}" for i in range(k))
    print(f"\n{'rung':<5} {heads}  {'mean':>6} {'range':>6}  all-agree")
    for r in RUNGS:
        accs = [pct(R[r]) for R in reps]
        uids = reps[0][r].keys()
        agree = sum(1 for u in uids if len({R[r][u] for R in reps}) == 1) / len(uids) * 100
        out["rungs"][r] = {"acc": accs, "mean": sum(accs) / k,
                           "range": max(accs) - min(accs), "all_agree_pct": agree}
        print(f"{r:<5} " + " ".join(f"{a:6.1f}" for a in accs)
              + f"  {sum(accs) / k:6.1f} {max(accs) - min(accs):6.1f}  {agree:6.1f}%")

    print(f"\n{'step':<11} " + " ".join(f"{'d' + str(i + 1):>6} {'p' + str(i + 1):>7}"
                                       for i in range(k)) + "   verdict")
    for x, y in STEPS:
        cells, sig = [], []
        for R in reps:
            _g, _l, p = mcnemar(R[x], R[y])
            d = pct(R[y]) - pct(R[x])
            cells.append((d, p))
            sig.append(("+" if d > 0 else "-") if p < 0.05 else "0")
        verdict = (f"consistent ({'significant ' + sig[0] if sig[0] != '0' else 'null'})"
                   if len(set(sig)) == 1 else "INCONSISTENT " + "/".join(sig))
        out["steps"][f"{x}->{y}"] = {"deltas": [c[0] for c in cells],
                                     "p": [c[1] for c in cells], "verdict": verdict}
        print(f"{x + '->' + y:<11} " + " ".join(f"{d:+6.1f} {p:7.2g}" for d, p in cells)
              + f"   {verdict}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
