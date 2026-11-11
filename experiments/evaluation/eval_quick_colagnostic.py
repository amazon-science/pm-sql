"""
Column-order-agnostic evaluation script.

Same as eval_quick.py but compares results ignoring column order.
Each row's values are sorted before comparison, so SELECT COUNT(*), resource
and SELECT resource, COUNT(*) produce the same normalized result.

Usage:
    python eval_quick_colagnostic.py <results_file> [--exclude-groups 1,23,5,109,132]
"""

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
GOLD_PATH = Path(__file__).parent / 'gold_results.json'
DB_PATH = project_root / 'event_log.sqlite'
DATASET_PATH = project_root / 'dataset.tsv'


def normalize_col_agnostic(rows):
    """Normalize rows: lowercase, strip, sort values within each row, then sort rows."""
    return sorted([
        sorted([str(v).strip().lower() if v is not None else '' for v in row])
        for row in rows
    ])


def main():
    results_path = sys.argv[1]

    exclude_uids = set()
    if '--exclude-groups' in sys.argv:
        idx = sys.argv.index('--exclude-groups')
        group_ids = set(sys.argv[idx + 1].split(','))
        with open(DATASET_PATH) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                uid = row.get('Utterance_id', '').strip()
                gid = row.get('Group_id', '').strip()
                if uid and gid in group_ids:
                    exclude_uids.add(uid)
        print(f"Excluding {len(exclude_uids)} questions from groups: {','.join(sorted(group_ids))}")

    with open(GOLD_PATH) as f:
        gold_data = json.load(f)

    generated = {}
    with open(results_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                generated[parts[0]] = parts[1]

    print(f"Generated: {len(generated)}, Gold: {len(gold_data)}")
    print(f"Mode: Column-order-agnostic")

    conn = sqlite3.connect(str(DB_PATH))

    results = []
    for uid, gen_sql in generated.items():
        if uid not in gold_data or uid in exclude_uids:
            continue

        gold_result = gold_data[uid]['result']
        if gold_result is None:
            continue

        # Re-normalize gold result as column-agnostic
        gold_col_agnostic = sorted([sorted(row) for row in gold_result])

        start = time.time()

        def make_checker(s):
            def check():
                return 1 if time.time() - s > 5 else 0
            return check

        conn.set_progress_handler(make_checker(start), 500)
        try:
            cur = conn.cursor()
            cur.execute(gen_sql)
            rows = cur.fetchall()
            normalized = normalize_col_agnostic(rows)
            score = 1 if normalized == gold_col_agnostic else 0
            error_type = None if score else 'wrong_result'
        except Exception:
            score = 0
            error_type = 'sql_error'
        finally:
            conn.set_progress_handler(None, 0)

        results.append({
            'uid': uid, 'score': score, 'error_type': error_type,
            'hardness': gold_data[uid].get('hardness', ''),
            'level': gold_data[uid].get('level', ''),
            'vocab': gold_data[uid].get('vocab', ''),
            'base_para': gold_data[uid].get('base_para', ''),
        })

        if len(results) % 200 == 0:
            done = sum(r['score'] for r in results)
            print(f"  [{len(results)}] running accuracy: {done}/{len(results)} = {done/len(results)*100:.1f}%")

    conn.close()

    total = len(results)
    correct = sum(r['score'] for r in results)
    errors = [r for r in results if r['score'] == 0]
    sql_errors = sum(1 for r in errors if r['error_type'] == 'sql_error')
    wrong_results = sum(1 for r in errors if r['error_type'] == 'wrong_result')

    print(f"\n{'='*60}")
    print(f"OVERALL: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*60}")
    print(f"SQL errors: {sql_errors}, Wrong results: {wrong_results}")

    for qualifier, label in [('hardness', 'HARDNESS'), ('level', 'LEVEL'),
                              ('vocab', 'VOCAB'), ('base_para', 'BASE/PARA')]:
        groups = {}
        for r in results:
            val = r.get(qualifier, '')
            if val:
                groups.setdefault(val, [0, 0])
                groups[val][0] += 1
                groups[val][1] += r['score']
        print(f"\n{label}:")
        for k in sorted(groups.keys()):
            t, c = groups[k]
            print(f"  {k:30s} {c:4d}/{t:4d} = {c/t*100:.1f}%")


if __name__ == '__main__':
    main()
