# PM-SQL: An Agentic Architecture for Querying Event Logs

PM-SQL is a progressive, agentic architecture for translating natural-language
questions into SQL over process-mining event logs. Starting from a published
baseline, it adds one component at a time (a model upgrade, few-shot examples,
an agent loop that validates SQL against the database, and domain prompt
content with an optional process-mining tool) and measures each component's
contribution under 5-fold cross-validation.

This repository contains the **evaluation code, prompts, and few-shot examples**
used in the paper. It does **not** contain the benchmark questions, gold SQL, or
the event log (see [Data](#data)).

## Benchmark

We evaluate on the **Text-2-SQL-4-PM** benchmark (G. Yamate and S. M. Peres,
University of São Paulo; arXiv:2509.09684), 1,655 natural-language questions
with gold SQL over the **BPIC 2020 International Declarations** event log
(B. F. van Dongen, 4TU.ResearchData, licensed CC BY-NC 4.0). Both are publicly
available from their original sources; neither is redistributed here.

## Repository layout

| Path | Contents |
|---|---|
| `experiments_v2/` | The cross-validated pipeline: ladder runner, fold assignments, leakage gate, scorer, prompt builder, and run manifests/predictions |
| `experiments_v2/common/` | Core modules (runner, folds, eval, prompt_builder, tools, example_selector, vocab_miner) |
| `experiments_v2/config/` | Fold partition, answer units, gold digests, admissibility rules |
| `experiments_v2/runs*/` | Run manifests and per-question predictions (model outputs, no gold data) |
| `experiments/oss_models/` | The multi-model comparison (open- and closed-source models via Amazon Bedrock) |
| `experiments/evaluation/` | Execution-accuracy scorers and gold precomputation (`precompute_gold.py`) |
| `llm_utils/prompt_helpers.py` | Bedrock client helper used by the model runners |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install boto3 pm4py networkx   # core runtime dependencies
```

Dependencies (`boto3`, `pm4py`, `networkx`, and the agent SDK used for the
agent configurations) are installed from PyPI and are not bundled here.

### Credentials

Model calls run on **Amazon Bedrock** and use the default boto3 credential
chain. To use cross-account Bedrock role assumption, set your own role ARNs via
the `BEDROCK_ROLE_ARNS` environment variable (comma-separated) or edit the
documented placeholder in `llm_utils/prompt_helpers.py`. With neither set, the
default single-account credentials are used.

## Data

The benchmark and event log are **not included** and must be obtained from their
original sources:

- Text-2-SQL-4-PM benchmark: https://github.com/pm-usp/text-2-sql
- BPIC 2020 International Declarations: https://data.4tu.nl/ (CC BY-NC 4.0)

`experiments/evaluation/gold_results.json` (precomputed gold query results) is
likewise not distributed; regenerate it locally with
`experiments/evaluation/precompute_gold.py` after obtaining the benchmark and
event log.

## Reproducing the results

```bash
# 5-fold cross-validated ladder (main results, complexity breakdown, failures)
bash experiments_v2/run_ladder.sh

# multi-model comparison (open- and closed-source models)
bash experiments/oss_models/run_all_oss.sh
```

See `experiments_v2/README.md` for the ladder's rungs, folds, and scoring
details.

## Citation

If you use this code, please cite the PM-SQL paper and the Text-2-SQL-4-PM
benchmark it builds on.

## License

Licensed under **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**; see [LICENSE](LICENSE). Copyright Amazon.com, Inc. or its
affiliates.

This is a research artifact released to support a scientific publication; it is
provided as-is, without warranty, and is not an officially supported Amazon
product.
