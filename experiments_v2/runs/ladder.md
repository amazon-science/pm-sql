| rung | adds | col-agnostic | 95% CI | per-fold mean ± sd | standard | n |
|---|---|---|---|---|---|---|
| R0 | published GPT-3.5 SQL, re-scored | 48.1% | [45.7, 50.5] | 48.1 ± 1.6 | 34.9% | 1655 |
| R1 | model upgrade, bare schema | 66.9% (+18.8) | [64.7, 69.1] | 66.9 ± 8.6 | 52.5% | 1655 |
| R2 | + factual data context (activities, formats) | 70.6% (+3.7) | [68.4, 72.8] | 70.6 ± 4.8 | 56.4% | 1655 |
| R3 | + 5 few-shot examples, failure-clustered | 74.4% (+3.9) | [72.4, 76.6] | 74.4 ± 3.7 | 60.9% | 1655 |
| R3c | + 5 few-shot examples, random [control] | 72.3% (-2.2) | [70.2, 74.4] | 72.3 ± 6.0 | 57.3% | 1655 |
| R4 | + validate_sql tool and agent loop | 77.4% (+3.0) | [75.4, 79.4] | 77.4 ± 4.1 | 62.5% | 1655 |
| R5 | + get_process_info tool, + mined vocabulary | 76.4% (-1.0) | [74.4, 78.5] | 76.4 ± 4.2 | 61.5% | 1655 |
| R6 | + targeted examples from R5 training errors | 76.4% (-0.1) | [74.3, 78.4] | 76.4 ± 4.6 | 60.8% | 1655 |

Example-selection criterion is worth +2.2pp (R3 failure-clustered 74.4% vs R3c random 72.3%). A gap near zero means few-shot examples help by demonstrating the output format, not by targeting weaknesses.
