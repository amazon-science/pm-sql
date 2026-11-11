# What may go in a prompt

The reviewers' central objection to v1 was not that the prompts were long. It was
that nobody could tell, from the paper, which parts of them had been written by
looking at evaluation-set answers. Once that is unclear, every rung above the
baseline measures an unknown mixture of "this component helps" and "we fitted the
prompt to the test set", and no ablation delta is interpretable.

So v2 fixes the rule in advance, in writing, and enforces it with a gate. Every
block of every prompt belongs to exactly one of three classes.

## DB — derivable from the event log by a query

Anything a `SELECT` against `event_log.sqlite` returns. Row counts, the list of
34 activity names, the number of distinct resources, the timestamp range, an
example case identifier.

Also in this class: the pre-computed pm4py artifacts served by
`get_process_info` (directly-follows graph, variants, activity and resource
statistics, rework, process tree). They are a pure function of the log, produced
by `experiments/step5_pm_artifacts/precompute_pm_artifacts.py`, and no question
text or gold SQL participates in generating them.

Admissible in every rung, for every fold. Fold-independent.

## DOC — from the published dataset documentation

Facts about BPIC 2020 that a practitioner would read in the dataset description
rather than compute: that this is a Dutch university's travel expense
reimbursement process, that a case is one declaration moving through approval
steps, that timestamps render day-first.

Admissible in every rung, for every fold. Fold-independent.

## FOLD — derived from the current fold's training split only

Few-shot examples, mined term→column vocabulary, and the targeted examples R6
adds. Admissible, but with two obligations: the content must be regenerated for
each of the five folds, and its provenance must be recorded in the run manifest
(which uids, which cluster, what support count).

This is the class that carries the argument. A fold's prompt is derived from 80%
of the benchmark and scored on the other 20%, which the derivation never saw — so
"we improved the prompt by looking at failures" becomes a legitimate method
instead of a confound, because the failures looked at are training failures.

## There is deliberately no fourth class

v1's prompts contained two sections that fit none of the above:

- **"Key SQL Patterns"** — CORRECT/WRONG query pairs written after inspecting
  which evaluation-set questions the model got wrong. Pattern 4's "CORRECT" form
  is a 144-character subquery reproducing gold uid 84 verbatim.
- **"Critical Rules"** — imperative rules, several of which encode a specific
  evaluation-set answer ("interventions = ALL events" came from analysing a
  regression).

Neither is reproduced in v2. If a rung needs to convey a query pattern, it does
so through a training-fold example, where the provenance is checkable.

## Enforcement

`common/leakage_gate.py` runs against every assembled prompt before that prompt
is used, and checks five things:

| check | condition | verdict |
|---|---|---|
| C1 | an example's uid belongs to a held-out group | FAIL |
| C2 | question-text Jaccard ≥ 0.70 against any held-out question (0.55–0.70) | FAIL (WARN) |
| C3 | prompt SQL matches held-out gold verbatim, or matches its literal-masked template | FAIL |
| C4 | a quoted prose phrase reproduces a held-out question | WARN |
| C5 | prompt SHA-256 recorded in the manifest | always |

C3 masks string and numeric literals before comparing, because the leak that
matters is structural: `count(*) > 3` and `count(*) > 200` are the same template,
while `count(*) > 20` and `count(*) <= 20` are character-similar and semantically
opposite. Generic schema-derivable SQL is downgraded to WARN so that
`SELECT COUNT(*) FROM event_log` does not fail a prompt for resembling a gold
answer it could not plausibly have been copied from.

The thresholds were calibrated against v1's prompts rather than guessed: run
`python common/leakage_gate.py --audit-v1` to reproduce. That audit found two
leaks the manual review missed (groups 8 and 10, both in `step5b_refined`), which
is the reason the gate exists as code and not as a checklist.
