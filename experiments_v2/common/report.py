"""Assemble the ladder table from whatever runs exist.

Two kinds of rung need different aggregation, and conflating them would be a
real error rather than a presentational one:

  fold-independent (R0, R1, R2)   one run over all 1,655 questions. Folds still
      partition the questions, so per-fold rates are reported to show variance,
      but every fold saw the same prompt.

  fold-specific (R3..R6)          five runs, each answering only its own
      held-out 20%. Pooling all five gives exactly one prediction per question
      across the whole benchmark, each produced by a prompt that never saw that
      question's group. That pooled number is the one comparable to R2's.

The comparison a reader wants is therefore always "pooled vs pooled", and the
per-fold spread is the honest error bar on it.

Usage:
    python report.py                     # every rung under experiments_v2/runs/
    python report.py --runs DIR --json out.json --markdown out.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from eval import DEFAULT_WORKERS, load_predictions, score
from folds import load_fold_config

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# Presentation order, and what each rung is claimed to add.
LADDER = [
    ("R0", "published GPT-3.5 SQL, re-scored"),
    ("R1", "model upgrade, bare schema"),
    ("R2", "+ factual data context (activities, formats)"),
    ("R3", "+ 5 few-shot examples, failure-clustered"),
    ("R3c", "+ 5 few-shot examples, random [control]"),
    ("R4", "+ validate_sql tool and agent loop"),
    ("R5", "+ get_process_info tool, + mined vocabulary"),
    ("R6", "+ targeted examples from R5 training errors"),
]

_RUN_DIR = re.compile(r"^(?P<rung>R\d+[a-z]?)(?:_f(?P<fold>\d+))?$")


def discover(runs_dir: Path) -> dict[str, list[Path]]:
    """Group run directories by rung, so R3_f0..R3_f4 pool into one R3."""
    out: dict[str, list[Path]] = defaultdict(list)
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        m = _RUN_DIR.match(d.name)
        if m and (d / "predictions.tsv").exists():
            out[m.group("rung")].append(d)
    return out


def _manifests(dirs: list[Path]) -> list[dict]:
    return [json.loads((d / "manifest.json").read_text())
            for d in dirs if (d / "manifest.json").exists()]


def collect(runs_dir: Path, workers: int = DEFAULT_WORKERS) -> list[dict]:
    found = discover(runs_dir)
    cfg = load_fold_config()
    rows = []

    for rung, note in LADDER:
        dirs = found.get(rung)
        if not dirs:
            continue

        preds: dict[str, str] = {}
        overlaps = 0
        for d in dirs:
            p = load_predictions(d / "predictions.tsv")
            # Pooling folds must not double-answer a question. If it does, the
            # fold assignment or the run invocation was wrong, and the pooled
            # accuracy would silently depend on dict ordering.
            overlaps += len(set(p) & set(preds))
            preds.update(p)

        m = score(preds, rung=rung, workers=workers)
        mans = _manifests(dirs)
        fold_specific = len(dirs) > 1 or any(x.get("fold") is not None for x in mans)

        rows.append({
            "rung": rung,
            "note": note,
            "fold_specific": fold_specific,
            "n_runs": len(dirs),
            "n_scored": m.n,
            "n_missing": len(m.missing),
            "duplicate_predictions": overlaps,
            "pooled_col_agnostic": round(m.pooled_ca, 2),
            "pooled_standard": round(m.pooled_std, 2),
            "fold_mean": round(m.fold_mean, 2),
            "fold_sd": round(m.fold_sd, 2),
            "ci95": [round(m.ci95[0], 2), round(m.ci95[1], 2)],
            "per_fold": {str(f): round(c / n * 100, 2)
                         for f, (c, n) in sorted(m.per_fold.items())},
            "errors": m.errors,
            "breakdowns": {d: {k: list(v) for k, v in a.items()}
                           for d, a in m.breakdowns.items()},
            "prompt_sha256": sorted({x.get("prompt_sha256", "") for x in mans}),
            "model_ids": sorted({x.get("model_id", "") for x in mans}),
            "example_uids": sorted({u for x in mans for u in x.get("example_uids", [])}),
            "tool_calls": sum(x.get("tool_calls", 0) for x in mans),
            "tokens_in": sum(x.get("tokens", {}).get("input", 0) for x in mans),
            "tokens_out": sum(x.get("tokens", {}).get("output", 0) for x in mans),
            "elapsed_s": round(sum(x.get("elapsed_s", 0) for x in mans), 1),
        })

    _ = cfg
    return rows


def render_markdown(rows: list[dict]) -> str:
    L = ["| rung | adds | col-agnostic | 95% CI | per-fold mean ± sd | standard | n |",
         "|---|---|---|---|---|---|---|"]
    prev = None
    for r in rows:
        delta = "" if prev is None else f" ({r['pooled_col_agnostic'] - prev:+.1f})"
        L.append(f"| {r['rung']} | {r['note']} | "
                 f"{r['pooled_col_agnostic']:.1f}%{delta} | "
                 f"[{r['ci95'][0]:.1f}, {r['ci95'][1]:.1f}] | "
                 f"{r['fold_mean']:.1f} ± {r['fold_sd']:.1f} | "
                 f"{r['pooled_standard']:.1f}% | {r['n_scored']} |")
        # R3c is a control, not a step in the ladder, so the running delta must
        # not chain through it.
        if r["rung"] != "R3c":
            prev = r["pooled_col_agnostic"]

    warnings = []
    for r in rows:
        if r["duplicate_predictions"]:
            warnings.append(f"{r['rung']}: {r['duplicate_predictions']} questions "
                            "answered by more than one fold run -- fold assignment is wrong")
        if r["n_missing"]:
            warnings.append(f"{r['rung']}: {r['n_missing']} questions have no prediction")
        if r["fold_specific"] and r["n_runs"] != 5:
            warnings.append(f"{r['rung']}: only {r['n_runs']}/5 fold runs present")
        if len(r["model_ids"]) > 1:
            warnings.append(f"{r['rung']}: mixed model ids {r['model_ids']} -- "
                            "the rung's delta is not attributable")
    if warnings:
        L += ["", "**Warnings**", ""] + [f"- {w}" for w in warnings]

    r3 = next((r for r in rows if r["rung"] == "R3"), None)
    r3c = next((r for r in rows if r["rung"] == "R3c"), None)
    if r3 and r3c:
        gap = r3["pooled_col_agnostic"] - r3c["pooled_col_agnostic"]
        L += ["", f"Example-selection criterion is worth {gap:+.1f}pp "
                  f"(R3 failure-clustered {r3['pooled_col_agnostic']:.1f}% vs "
                  f"R3c random {r3c['pooled_col_agnostic']:.1f}%). A gap near zero "
                  "means few-shot examples help by demonstrating the output format, "
                  "not by targeting weaknesses."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the v2 ladder table")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    rows = collect(args.runs, workers=args.workers)
    if not rows:
        print(f"no runs found under {args.runs}")
        raise SystemExit(1)

    md = render_markdown(rows)
    print(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md + "\n")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
