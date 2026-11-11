"""Quick deterministic evaluation for Step 4b."""

import csv
import json
import re
import sys
import time
from pathlib import Path

GOLD_PATH = Path(__file__).parent / 'gold_results.json'
DATASET_PATH = Path(__file__).resolve().parents[2] / 'dataset.tsv'


def normalize_table(text):
    """Parse markdown table into sorted list of row tuples (data only, no header)."""
    if not text or not text.strip():
        return []
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    table_lines = [l for l in lines if '|' in l]
    if not table_lines:
        return []
    data_rows = []
    for line in table_lines:
        cells = [c.strip().lower() for c in line.split('|') if c.strip()]
        if cells and all(c.replace('-', '').replace(':', '').strip() == '' for c in cells):
            continue
        if cells:
            data_rows.append(tuple(cells))
    # Skip header (first row)
    if len(data_rows) > 1:
        data_rows = data_rows[1:]
    return sorted(data_rows)


def main():
    results_path = sys.argv[1]

    # Load exclusions
    exclude = set()
    group_ids = {'1', '23', '5', '109', '132'}
    with open(DATASET_PATH) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            uid = row.get('Utterance_id', '').strip()
            gid = row.get('Group_id', '').strip()
            if uid and gid in group_ids:
                exclude.add(uid)

    # Load gold (pre-sort to avoid repeated sorting)
    print("Loading gold results...")
    with open(GOLD_PATH) as f:
        gold_raw = json.load(f)

    # Pre-compute normalized gold rows (skip huge results > 500 rows for speed)
    gold = {}
    skipped_large = 0
    for uid, info in gold_raw.items():
        r = info['result']
        if r is None:
            gold[uid] = None
        elif len(r) > 500:
            skipped_large += 1
            gold[uid] = 'LARGE'  # Cannot evaluate — agent can't return all rows
        else:
            gold[uid] = sorted([tuple(str(v).lower().strip() for v in row) for row in r])

    print(f"Gold loaded. {skipped_large} questions with >500 rows skipped.")

    # Load agent answers
    entries = {}
    with open(results_path) as f:
        for line in f:
            entry = json.loads(line)
            entries[entry['uid']] = entry['answer']

    print(f"Entries: {len(entries)}")

    # Evaluate
    correct = 0
    total = 0
    no_table = 0
    large_skip = 0

    for uid, answer in entries.items():
        if uid not in gold or uid in exclude:
            continue
        gold_rows = gold[uid]
        if gold_rows is None:
            continue
        if gold_rows == 'LARGE':
            large_skip += 1
            total += 1
            continue

        total += 1
        agent_rows = normalize_table(answer)

        if not agent_rows:
            no_table += 1
            continue

        if agent_rows == gold_rows:
            correct += 1

    print(f"\n{'='*60}")
    print(f"DETERMINISTIC: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  No table parsed: {no_table}")
    print(f"  Large (>500 rows, skipped): {large_skip}")
    print(f"  Evaluated: {total - large_skip}")
    print(f"  Correct (of evaluated): {correct}/{total - large_skip} = {correct/max(total-large_skip,1)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
