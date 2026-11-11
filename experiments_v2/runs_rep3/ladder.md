| rung | adds | col-agnostic | 95% CI | per-fold mean ± sd | standard | n |
|---|---|---|---|---|---|---|
| R0 | published GPT-3.5 SQL, re-scored | 48.1% | [45.7, 50.5] | 48.1 ± 1.6 | 34.9% | 1655 |
| R1 | model upgrade, bare schema | 67.1% (+19.0) | [64.8, 69.3] | 67.1 ± 8.0 | 52.5% | 1655 |
| R2 | + factual data context (activities, formats) | 70.3% (+3.3) | [68.2, 72.5] | 70.3 ± 5.0 | 56.1% | 1655 |
| R3 | + 5 few-shot examples, failure-clustered | 74.1% (+3.8) | [72.0, 76.2] | 74.1 ± 3.1 | 60.7% | 1655 |
| R3c | + 5 few-shot examples, random [control] | 72.2% (-1.9) | [70.1, 74.4] | 72.2 ± 6.3 | 57.3% | 1655 |
| R4 | + validate_sql tool and agent loop | 77.5% (+3.4) | [75.5, 79.5] | 77.5 ± 4.0 | 63.0% | 1655 |
| R5 | + get_process_info tool, + mined vocabulary | 76.1% (-1.4) | [74.1, 78.2] | 76.1 ± 5.1 | 61.3% | 1655 |
| R6 | + targeted examples from R5 training errors | 76.5% (+0.4) | [74.5, 78.5] | 76.5 ± 4.5 | 60.9% | 1655 |

Example-selection criterion is worth +1.9pp (R3 failure-clustered 74.1% vs R3c random 72.2%). A gap near zero means few-shot examples help by demonstrating the output format, not by targeting weaknesses.
