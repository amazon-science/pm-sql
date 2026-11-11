| rung | adds | col-agnostic | 95% CI | per-fold mean ± sd | standard | n |
|---|---|---|---|---|---|---|
| R0 | published GPT-3.5 SQL, re-scored | 48.1% | [45.7, 50.5] | 48.1 ± 1.6 | 34.9% | 1655 |
| R1 | model upgrade, bare schema | 67.0% (+18.9) | [64.8, 69.2] | 67.0 ± 8.5 | 52.6% | 1655 |
| R2 | + factual data context (activities, formats) | 70.7% (+3.7) | [68.5, 72.9] | 70.7 ± 4.6 | 56.6% | 1655 |
| R3 | + 5 few-shot examples, failure-clustered | 74.1% (+3.5) | [72.0, 76.2] | 74.1 ± 3.3 | 60.5% | 1655 |
| R3c | + 5 few-shot examples, random [control] | 72.0% (-2.1) | [69.9, 74.2] | 72.0 ± 6.0 | 57.2% | 1655 |
| R4 | + validate_sql tool and agent loop | 76.1% (+2.0) | [74.1, 78.2] | 76.1 ± 4.8 | 61.8% | 1655 |
| R5 | + get_process_info tool, + mined vocabulary | 75.8% (-0.3) | [73.8, 77.9] | 75.8 ± 5.4 | 61.3% | 1655 |
| R6 | + targeted examples from R5 training errors | 76.1% (+0.3) | [74.1, 78.2] | 76.1 ± 4.8 | 60.9% | 1655 |

Example-selection criterion is worth +2.1pp (R3 failure-clustered 74.1% vs R3c random 72.0%). A gap near zero means few-shot examples help by demonstrating the output format, not by targeting weaknesses.
