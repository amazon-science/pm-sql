# experiments_v2 — the leakage-free ladder

Rebuild of the paper's experiments to answer the reviewer objections recorded in
`../FEEDBACK_REVIEW_v2.md`, per the plan in `../EXPERIMENTS_V2_PLAN.md`.

What changed relative to v1, in one line each:

- **Nothing hand-authored from the evaluation set.** Every fold-specific piece of
  prompt content — few-shot examples, mined vocabulary — is a deterministic
  function of that fold's *training* split.
- **5-fold cross-validation** so every one of the 1,655 questions is answered by a
  prompt that never saw its answer. v1 had to discard 42 questions.
- **A leakage gate that can veto a run.** It refuses to score a prompt containing
  held-out content, and it flags three leaks in v1's own prompts. It runs twice:
  as a pre-run sweep over all 25 fold-prompts, and again inside every real run at
  the moment each prompt is built. The sweep cannot see the exact examples R3–R6
  will use (those depend on R2's and R5's results, which do not exist yet), so it
  validates the mechanism and the build-time gate validates the literal prompt.
- **Both metrics reported.** Column-agnostic (matching the benchmark's published
  numbers) and column-order-preserving standard EX.

## Run it

```bash
./run_ladder.sh --preflight-only   # ~6 min, no cost. Do this first.
./run_ladder.sh --mock             # ~25 min, no cost, exercises every path
./run_ladder.sh                    # the real run
```

`--mock` writes to `runs_mock/`, never `runs/`: sharing the directory would let a
later real run resume from mock predictions and report them as results.

The real run is roughly **24,000 Bedrock calls, 2–3 days, $500–600**. It is built
for an unattended cloud desktop:

- Every rung resumes. Re-running the same command skips questions already in
  `predictions.tsv`, so a crash, a throttle storm, or a closed laptop costs only
  the in-flight batch.
- Order is a real dependency, not a preference. R3 picks its examples from what R2
  got wrong and R6 from what R5 got wrong, so rungs cannot be reordered or
  parallelised across rungs. Folds within a rung could be, but the script keeps
  them serial so the throughput line stays readable.
- Expect it to survive a disconnect: `nohup ./run_ladder.sh > ladder.log 2>&1 &`.

### Credentials over a multi-day run

`ada` conduit tokens last ~1h (not 12h -- measured). Set `CRED_REFRESH_CMD` and the script refreshes every 25 min
in the background; `runner.py` reloads the credentials file on an auth error, so
running workers pick up the new token without a restart.

```bash
export PATH=$HOME/.toolbox/bin:$PATH
export CRED_REFRESH_CMD='ada credentials update --account <acct> --provider conduit --role <role> --once'
tmux new -s ladder './run_ladder.sh 2>&1 | tee -a ladder.log'
```

The refresh needs a live **Midway session (~20h)**, so `mwinit` on the desktop
roughly once a day. If it lapses nothing is lost: the runner pauses up to an hour
waiting for valid credentials, then exits incomplete; the script retries the same
step every 10 minutes (for up to ~33h) and continues by itself once `mwinit` has
been run. `grep -n "REFRESH FAILED\|INCOMPLETE" ladder.log` shows whether it is
waiting on you.

Failures of the plumbing -- expired credentials, network errors, exhausted
throttling retries -- are never written to `predictions.tsv`. They are retried on
resume, so an outage cannot show up as lower accuracy. Only a model's actual
answer (including `parse_failed` and deterministic request errors) is recorded.

Set `AWS_REGION` (default `us-west-2`) and `PMSQL_MODEL` to override the model.
Needs Python ≥ 3.12; the script finds `python3.12` or uses `$PYTHON`.

## The ladder

| rung | what it adds | folds | calls |
|---|---|---|---|
| R0 | published GPT-3.5 SQL, re-scored by our scorer | – | 0 |
| R1 | model upgrade, bare schema only | – | 1,655 |
| R2 | + factual data context: activity list, date and case-id formats | – | 1,655 |
| R3 | + 5 few-shot examples chosen by failure-signature clustering | 5 | 1,655 |
| R3c | + 5 few-shot examples chosen at random — **the control for R3** | 5 | 1,655 |
| R4 | + `validate_sql` tool and an agent loop | 5 | ~6,600 |
| R5 | + `get_process_info` tool, + mined domain vocabulary | 5 | ~6,600 |
| R6 | + targeted examples from R5's *training-split* errors | 5 | ~6,600 |

R1 and R2 contain no benchmark-derived content, so there is nothing for a fold to
hold out and they run once over all questions. R3–R6 run five times, each
answering only its own held-out fold; pooling the five gives exactly one
prediction per question, which is what is comparable to R2.

R3c exists to answer "do the examples help because they were *chosen well*, or
just because they show the output format?". It is the same code path as R3 with
the ranking replaced by a uniform sample, so R3 − R3c isolates the criterion.

## What splits the folds, and why it is not the paraphrase group

The obvious unit is the 205-member paraphrase group, and it is wrong. Running the
gate over all 25 fold-prompts failed 7 of them, for two compounding reasons:

1. The 205 groups contain only **190 distinct gold SQL statements**. Group 15
   ("Which activities appear most frequently…") and group 91 ("…most often…") are
   one question under two ids.
2. Execution accuracy scores *results*, not SQL text, so 190 statements are only
   **178 distinct answers**. Group 8's "total number of events per resource" and
   group 12's "in how many activities has each resource acted" have gold SQL
   differing only in column order — under the column-agnostic primary metric they
   are the same answer.

So the split unit is the **answer unit**: groups merged transitively wherever they
share a normalised gold SQL or a gold result digest. 205 groups → 190 classes →
178 units, frozen in `config/answer_units.json`. `folds.py` refuses to emit a fold
set that splits one.

This preserves the intent of group-level CV; it just uses the relation the metric
actually rewards. It over-merges only in the harmless direction — two unrelated
questions that coincide on this one 72k-event database share a unit, costing a
little balance freedom and leaking nothing.

## Layout

```
common/dataset.py           loading, paraphrase groups, gold-SQL answer classes
common/folds.py             answer units, 5-fold packing, config/folds.json
common/eval.py              execution accuracy, both metrics, bootstrap CI
common/prompt_builder.py    assembles each rung's prompt from admissible parts
common/example_selector.py  failure-signature clustering (R3/R6) and the control
common/vocab_miner.py       lift-based domain-term mining from the training split
common/tools.py             validate_sql, get_process_info
common/runner.py            Bedrock tool loop, resume, manifests, --gate-sweep
common/leakage_gate.py      C1–C5 checks; --audit-v1 for the retrospective audit
common/report.py            pools folds, builds the ladder table
config/                     frozen: folds, answer units, gold digests, vocab cache
runs/                       output; one directory per (rung, fold)
```

`config/*.json` is frozen and committed. **Do not regenerate it for a published
run** — the fold partition is part of the experimental design, and re-rolling it
after seeing results is exactly the discretion this rebuild exists to remove.

## Invariants worth knowing before changing anything

- `validate_sql` returns only a row count or an error string, never data, so the
  agent cannot read gold answers out of the database.
- `get_process_info` serves pm4py artifacts precomputed from the XES log. They are
  pure functions of the log, not of the question set.
- Prompt content is admissible only as DB (schema/data facts), DOC (published
  BPIC documentation) or FOLD (derived from the training split). There is
  deliberately no fourth class; see `config/admissibility.md`.
- Every run writes `manifest.json` with the prompt SHA-256 and the gate report, so
  any number in the paper can be traced to the exact prompt that produced it.
- Never commit `runs/`. Materialised result sets are ~440 MB; the scorer stores
  SHA-256 digests instead (396 KB).
