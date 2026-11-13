#!/usr/bin/env python3
"""Run the PM-SQL ladder end to end and report execution accuracy.

For each configuration it generates SQL for all 1,655 benchmark questions with
the Claude models on Amazon Bedrock, scores the predictions with the benchmark
authors' evaluator, and prints the accuracy ladder.

    ./setup_data.sh                      # once: fetch the benchmark + event log
    python3 run.py                       # full run on Bedrock (costs money, hours)
    python3 run.py --limit 50            # quick smoke test on the first 50 questions
    python3 run.py --configs agent full  # only some rungs
    python3 run.py --workers 8           # parallel Bedrock workers

Requires AWS credentials with Bedrock access (default boto3 chain, or set
BEDROCK_ROLE_ARNS to a comma-separated list of cross-account role ARNs).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE / "architecture"
DATASET = HERE / "dataset.tsv"
DB = HERE / "event_log.sqlite"

# name -> (config dir, run script, extra args, needs PM artifacts)
CONFIGS = {
    "model_upgrade": ("model_upgrade", "run_step2.py", [], False),
    "few_shot":      ("few_shot", "run_step3.py", ["--mode", "few-shot"], False),
    "agent":         ("agent", "run_step4a.py", [], False),
    "full":          ("full", "run_step5c.py", [], True),
}
LABELS = {
    "model_upgrade": "Model-Sonnet   (model upgrade)",
    "few_shot":      "Model-FewShot  (+ few-shot)",
    "agent":         "Model-Agent    (+ validation loop)",
    "full":          "Model-Full     (+ domain, PM tool)",
}
PUBLISHED = {"baseline": 47.6, "model_upgrade": 62.7, "few_shot": 70.9,
             "agent": 73.7, "full": 79.4}

# Multi-model comparison: the few-shot configuration across Bedrock models.
# label -> (Bedrock model id, paper accuracy on the 1,613-question subset)
MODELS = {
    # Claude Sonnet 4.5 with extended thinking: its own runner (thinking budget
    # 4000 tokens, temperature 1 as the API requires), so the model id is None.
    "claude-thinking": (None, 75.8),
    "deepseek-v3.2": ("deepseek.v3.2", 71.4),
    "deepseek-r1":   ("us.deepseek.r1-v1:0", 70.1),
    "claude-sonnet": ("us.anthropic.claude-sonnet-4-6", 70.1),
    "gpt-oss-20b":   ("openai.gpt-oss-20b-1:0", 69.7),
    "gpt-oss-120b":  ("openai.gpt-oss-120b-1:0", 66.9),
    "qwen3-32b":     ("qwen.qwen3-32b-v1:0", 63.4),
}
# The paper's model comparison excludes the 5 few-shot example groups (1,613 questions).
FEWSHOT_GROUPS = {"1", "5", "23", "109", "132"}


def ensure_pm_data() -> bool:
    """The full config's get_process_info reads architecture/full/pm_data/."""
    dst = ARCH / "full" / "pm_data"
    if dst.exists():
        return True
    src = ARCH / "process_mining" / "pm_data"
    if src.exists():
        shutil.copytree(src, dst)
        return True
    return False


def remove_pm_data() -> None:
    """Withhold the artifacts: get_process_info stays available to the agent but
    returns an error for every aspect. This is the paper's ablation of the
    process-mining artifacts, isolating them from the rest of the domain content."""
    dst = ARCH / "full" / "pm_data"
    if dst.exists():
        shutil.rmtree(dst)


# A question the runner could not answer -- an API failure (throttling, timeout,
# expired token) or a model that produced no parseable SQL -- is written as this
# sentinel, which always scores 0. Transient API failures should not count as
# wrong answers, so they are retried; whatever still fails afterwards is left in
# place and scored 0 rather than dropped, so a model that cannot produce SQL is
# not flattered by having those questions quietly excluded.
ERROR_SENTINEL = "SELECT 1"
MAX_RETRY_ROUNDS = 3


def count_failed(preds: Path) -> int:
    if not preds.exists():
        return 0
    return sum(1 for line in preds.read_text().splitlines()
               for parts in [line.split("\t")]
               if len(parts) >= 2 and parts[1].strip().rstrip(";") == ERROR_SENTINEL)


def drop_failed_rows(preds: Path) -> int:
    """Remove error-sentinel rows so a re-run regenerates exactly those questions."""
    if not preds.exists():
        return 0
    kept, dropped = [], 0
    for line in preds.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip().rstrip(";") == ERROR_SENTINEL:
            dropped += 1
        elif line.strip():
            kept.append(line)
    if dropped:
        preds.write_text("\n".join(kept) + "\n")
    return dropped


def retry_failures(preds: Path, cmd: list[str], cwd: Path) -> None:
    """Regenerate unanswered questions; leave any that still fail scored 0."""
    for _ in range(MAX_RETRY_ROUNDS):
        if not count_failed(preds):
            return
        n = drop_failed_rows(preds)
        print(f"  retrying {n} unanswered question(s) ...", flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)
    remaining = count_failed(preds)
    if remaining:
        print(f"  ! {remaining} question(s) still unanswered after {MAX_RETRY_ROUNDS} "
              f"retries; scored as incorrect.")


def generate(name: str, out_dir: Path, limit: int | None, workers: int | None,
             no_pm_data: bool = False) -> Path:
    cfg_dir, script, extra, needs_pm = CONFIGS[name]
    suffix = ""
    if needs_pm:
        if no_pm_data:
            remove_pm_data()
            suffix = ".nodata"
            print(f"  {name}: process-mining artifacts withheld (ablation)")
        elif not ensure_pm_data():
            print(f"  ! {name}: no process-mining artifacts; generate them first with\n"
                  f"    ./setup_data.sh --with-xes && python3 architecture/process_mining/"
                  f"precompute_pm_artifacts.py\n    running {name} without them (ablation).")
    preds = out_dir / f"{name}{suffix}.tsv"
    cmd = [sys.executable, script, "--output", str(preds), "--dataset", str(DATASET), *extra]
    if limit is not None:
        cmd += ["--end", str(limit)]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    print(f"  generating {name} -> {preds.name} ...", flush=True)
    subprocess.run(cmd, cwd=ARCH / cfg_dir, check=True)
    # The runners resume by skipping question ids already in the output file, so
    # dropping the unanswered ones and re-running regenerates exactly those.
    retry_failures(preds, cmd, ARCH / cfg_dir)
    return preds


def score(preds: Path, out_dir: Path) -> float:
    # evaluator/score.py writes out_dir/<tag>.json for tag=<name>
    subprocess.run([sys.executable, str(HERE / "evaluator" / "score.py"),
                    str(out_dir), f"{preds.stem}={preds}"], check=True)
    d = json.loads((out_dir / f"{preds.stem}.json").read_text())
    return 100.0 * sum(d.values()) / len(d)


def excluded_uids() -> set[str]:
    """Question ids in the few-shot example groups (excluded from the model comparison)."""
    import csv
    out = set()
    for r in csv.DictReader(open(DATASET, newline=""), delimiter="\t"):
        if (r.get("Group_id") or "").strip() in FEWSHOT_GROUPS:
            uid = (r.get("Utterance_id") or "").strip()
            if uid:
                out.add(uid)
    return out


def run_model_comparison(models: list[str], out_dir: Path, limit, workers) -> None:
    exclude = excluded_uids()
    rows = []
    for label in models:
        model_id, paper = MODELS[label]
        preds = out_dir / f"model_{label}.tsv"
        # claude-thinking has a dedicated runner; the rest share one, parameterised
        # by Bedrock model id.
        if model_id is None:
            cmd = [sys.executable, str(ARCH / "few_shot" / "run_thinking.py"),
                   "--output", str(preds), "--dataset", str(DATASET)]
        else:
            cmd = [sys.executable, str(ARCH / "few_shot" / "run_models.py"),
                   "--model", model_id, "--output", str(preds), "--dataset", str(DATASET)]
        if limit is not None:
            cmd += ["--end", str(limit)]
        if workers is not None:
            cmd += ["--workers", str(workers)]
        print(f"  generating with {label}"
              f"{f' ({model_id})' if model_id else ''} ...", flush=True)
        subprocess.run(cmd, cwd=ARCH / "few_shot", check=True)
        retry_failures(preds, cmd, ARCH / "few_shot")
        subprocess.run([sys.executable, str(HERE / "evaluator" / "score.py"),
                        str(out_dir), f"model_{label}={preds}"], check=True)
        d = json.loads((out_dir / f"model_{label}.json").read_text())
        kept = {u: v for u, v in d.items() if u not in exclude} or d  # tiny --limit runs may exclude all
        rows.append((label, 100.0 * sum(kept.values()) / len(kept), paper))
    print("\nMulti-model comparison, few-shot configuration"
          + (f" (first {limit} questions)" if limit else " (1,613 questions)") + "\n")
    print(f"  {'model':16} {'this run':>9}   {'paper':>6}")
    for label, acc, paper in rows:
        print(f"  {label:16} {acc:8.1f}%   {paper:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--models", nargs="*", choices=list(MODELS),
                    help="run the multi-model comparison instead of the ladder "
                         "(no values = all models)")
    ap.add_argument("--no-pm-data", action="store_true",
                    help="run the Model-Full ablation: the get_process_info tool is\npresent but its artifacts are withheld (paper: 78.9%% against 79.4%%)")
    ap.add_argument("--limit", type=int, help="run on the first N questions only")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    args = ap.parse_args()

    if not DATASET.exists() or not DB.exists():
        sys.exit("dataset.tsv / event_log.sqlite missing -- run ./setup_data.sh first")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.models is not None:
        run_model_comparison(args.models or list(MODELS), args.out, args.limit, args.workers)
        return

    results: dict[str, float] = {}
    for name in args.configs:
        preds = generate(name, args.out, args.limit, args.workers, args.no_pm_data)
        results[name] = score(preds, args.out)

    print("\nExecution accuracy" + (f" (first {args.limit} questions)" if args.limit else "")
          + "\n")
    print(f"  {'configuration':30} {'this run':>9}   {'paper':>6}")
    print(f"  {'Model-Baseline (GPT-3.5)':30} {'--':>9}   {PUBLISHED['baseline']:5.1f}%")
    for name in CONFIGS:
        if name in results:
            label, paper = LABELS[name], PUBLISHED[name]
            if name == "full" and args.no_pm_data:
                label, paper = "Model-Full     (PM artifacts withheld)", 78.9
            print(f"  {label:30} {results[name]:8.1f}%   {paper:5.1f}%")
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
