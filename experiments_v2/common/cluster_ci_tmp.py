"""Naive vs cluster-robust confidence intervals on the v1 published ladder."""
import random
from pathlib import Path
from dataset import load_questions
from folds import load_units
from eval import correctness, load_predictions

CFG = [("step2","step2"),("step3b","step3b"),("step4a","step4a"),("step5c","step5c_nodata")]
REP = Path("../../experiments/v1_replicates/runs/rep1")
EXCL = {"1","5","23","109","132"}

def main():
    units = load_units()
    qs = load_questions()
    keep = {q.uid for q in qs if q.group_id not in EXCL}
    clusters = [[q.uid for q in u.questions if q.uid in keep] for u in units]
    clusters = [c for c in clusters if c]

    C = {}
    for name, fn in CFG:
        C[name] = {u: v for u, v in correctness(load_predictions(REP / f"{fn}.tsv"), workers=1).items() if u in keep}
        print(f"scored {name}", flush=True)

    n = sum(len(c) for c in clusters)
    rng = random.Random(11)
    boots = []
    for _ in range(4000):
        flat = [u for _ in range(len(clusters)) for u in clusters[rng.randrange(len(clusters))]]
        boots.append({k: sum(C[k][u] for u in flat) / len(flat) * 100 for k, _ in CFG})

    def ci(vals):
        v = sorted(vals); return v[int(.025*len(v))], v[int(.975*len(v))]

    print(f"\nscored questions: {n}, answer units: {len(clusters)}")
    print(f"\n{'config':<8} {'acc':>6}  {'naive 95% CI':>16} {'cluster 95% CI':>18}  widen")
    for k, _ in CFG:
        acc = sum(C[k].values())/n*100
        half = 1.96*(acc/100*(1-acc/100)/n)**.5*100
        lo, hi = ci([b[k] for b in boots])
        print(f"{k:<8} {acc:6.1f}  [{acc-half:5.1f},{acc+half:5.1f}]  [{lo:5.1f},{hi:5.1f}]  {(hi-lo)/(2*half):.1f}x")

    print(f"\n{'step':<16} {'delta':>6}  {'cluster 95% CI':>18}  significant?")
    for (a,_),(b,_) in zip(CFG, CFG[1:]):
        d = (sum(C[b].values())-sum(C[a].values()))/n*100
        lo, hi = ci([x[b]-x[a] for x in boots])
        print(f"{a+'->'+b:<16} {d:+6.1f}  [{lo:+5.1f},{hi:+5.1f}]  {'yes' if lo>0 or hi<0 else 'NO'}")

if __name__ == "__main__":
    main()
