# PM-SQL: An Agentic Architecture for Querying Event Logs

PM-SQL translates natural-language questions into SQL over process-mining event
logs. Built up one component at a time — a model upgrade, few-shot examples, an
agent loop that validates SQL against the database, and domain prompt content
with a process-mining tool — it raises execution accuracy on the
Text-2-SQL-4-PM benchmark from a published **47.6%** to **79.4%**.

`run.py` reproduces this end to end: it generates SQL for the benchmark on
Amazon Bedrock, scores the output with the benchmark authors' evaluator, and
prints the accuracy table.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python >= 3.12
pip install -r requirements.txt
./setup_data.sh                                      # fetch benchmark + event log

python3 run.py --limit 4 --configs model_upgrade     # smoke test, a few cents
python3 run.py                                       # full ladder (or: ./run.sh)
```

A full run generates 1,655 questions per configuration: hours of wall time and
real Bedrock cost. Start with `--limit`.

```
  configuration                   this run    paper
  Model-Baseline (GPT-3.5)              --    47.6%
  Model-Sonnet   (model upgrade)        --    62.7%
  Model-FewShot  (+ few-shot)           --    70.9%
  Model-Agent    (+ validation loop)    --    73.7%
  Model-Full     (+ domain, PM tool)    --    79.4%
```

## Usage

### `run.py` options

| Option | Meaning |
|---|---|
| `--configs NAME…` | Which rungs of the ladder to run: `model_upgrade`, `few_shot`, `agent`, `full`. Default: all four. |
| `--models [NAME…]` | Run the multi-model comparison instead of the ladder. With no values, all models; otherwise a subset (see below). |
| `--no-pm-data` | Run the `full` configuration as the paper's ablation: the `get_process_info` tool is present but its artifacts are withheld. |
| `--limit N` | Use only the first N questions. For smoke tests; accuracies are not comparable to the paper. |
| `--workers N` | Parallel Bedrock workers. Default: one per role ARN, or 1 on default credentials. Higher is faster but risks throttling. |
| `--out DIR` | Where predictions, per-question scores and `results.json` are written. Default `runs/`. |

Predictions are written incrementally and a re-run resumes, skipping questions
already answered. So an interrupted run can simply be repeated, and a question
that failed with a Bedrock API error is retried automatically (up to three
rounds); anything still unanswered is scored as incorrect rather than dropped.

### Reproducing each part of the paper

```bash
# the ablation ladder (Table 1)
python3 run.py

# the process-mining artifact ablation (Model-Full, artifacts withheld)
python3 run.py --configs full --no-pm-data

# the multi-model comparison (Figure 3)
python3 run.py --models
python3 run.py --models qwen3-32b deepseek-r1          # a subset
```

Available models for `--models`: `claude-thinking` (Claude Sonnet with extended
thinking), `claude-sonnet`, `deepseek-v3.2`, `deepseek-r1`, `gpt-oss-20b`,
`gpt-oss-120b`, `qwen3-32b`. The comparison runs the few-shot configuration and
scores the 1,613 questions outside the few-shot example groups, as the paper
does. Request access to each model in your Bedrock account first.

### The `full` configuration and its artifacts

`full` adds a `get_process_info` tool backed by pm4py artifacts computed from the
raw event log. Generate them once:

```bash
./setup_data.sh --with-xes
python3 architecture/process_mining/precompute_pm_artifacts.py
python3 run.py --configs full
```

Without the artifacts the tool returns an error for every request, which is the
ablation condition; pass `--no-pm-data` to run that deliberately.

### Credentials

Generation runs on **Amazon Bedrock** through the default boto3 credential chain
(environment variables, `~/.aws`, or an attached role). To spread load across
several accounts, set `BEDROCK_ROLE_ARNS` to a comma-separated list of role ARNs
to assume; otherwise a single default client is used. Set `AWS_REGION` to
override the region.

### `setup_data.sh`

| Invocation | Fetches |
|---|---|
| `./setup_data.sh` | `dataset.tsv` and `event_log.sqlite` — enough for every configuration except the `full` artifacts |
| `./setup_data.sh --with-xes` | also the raw BPIC 2020 XES log, needed only to regenerate the process-mining artifacts |

Both are idempotent. Set `BENCH_DIR` to reuse an existing benchmark checkout.

### Scoring on its own

To score a predictions file without regenerating it:

```bash
python3 evaluator/score.py OUT_DIR  mytag=runs/full.tsv   # -> OUT_DIR/mytag.json
```

## Layout

- `run.py` / `run.sh` — generate on Bedrock, score, and report
- `architecture/` — the method, one directory per rung: `model_upgrade`,
  `few_shot` (plus the multi-model and extended-thinking runners), `agent`,
  `full`, and `process_mining` (the artifact generator)
- `evaluator/` — the benchmark authors' test-suite evaluator and a scorer wrapper
- `pmsql/` — shared library (dataset loader, scoring helpers, Bedrock client)
- `llm_utils/` — the Bedrock client used by the generators

## Data

`setup_data.sh` fetches, into this directory, from the original sources (nothing
is redistributed here):

- the Text-2-SQL-4-PM benchmark — https://github.com/pm-usp/text-2-sql
- the BPIC 2020 International Declarations event log — https://data.4tu.nl/ (CC BY-NC 4.0)

## Citation

_BibTeX to be added on publication._

## License

[CC BY-NC 4.0](LICENSE). Copyright Amazon.com, Inc. or its affiliates. Research
artifact, provided as-is; not an officially supported Amazon product.
