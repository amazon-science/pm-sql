"""
Evaluation harness for text-2-SQL-4-PM benchmark.

Compares generated SQL against pre-computed gold results by executing the
generated SQL against the SQLite database and comparing result sets.

Uses gold_results.json (pre-computed by precompute_gold.py) so gold SQL
never needs to be re-executed.

Usage:
    python evaluate.py --results path/to/results.tsv

Results file format (TSV, no header):
    utterance_id \t generated_sql

Output: per-question pass/fail + aggregate accuracy + breakdown by qualifier.
"""

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path


def execute_sql(conn, sql, timeout_seconds=10):
    """Execute SQL and return results as a sorted list of lists, or None on error/timeout."""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        normalized = []
        for row in rows:
            normalized.append([
                str(v).strip().lower() if v is not None else ''
                for v in row
            ])
        return sorted(normalized)
    except Exception:
        return None


def load_gold_results(gold_path):
    """Load pre-computed gold results from JSON."""
    with open(gold_path, 'r') as f:
        return json.load(f)


def load_generated_sql(results_path):
    """Load generated SQL from results file. Returns dict: utterance_id -> generated_sql."""
    generated = {}
    with open(results_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                uid = parts[0].strip()
                sql = parts[1].strip()
                generated[uid] = sql
    return generated


def evaluate(db_path, gold_path, results_path, exclude_uids=None):
    """Run evaluation and return results."""
    conn = sqlite3.connect(str(db_path))

    gold_data = load_gold_results(gold_path)
    generated = load_generated_sql(results_path)

    if exclude_uids is None:
        exclude_uids = set()

    results = []
    timeouts = 0

    for uid, gen_sql in generated.items():
        if uid not in gold_data:
            continue
        if uid in exclude_uids:
            continue

        gold_info = gold_data[uid]
        gold_result = gold_info['result']  # Pre-computed, already normalized+sorted

        # If gold had an error (shouldn't happen), skip
        if gold_result is None:
            continue

        # Execute generated SQL
        gen_result = execute_sql(conn, gen_sql)

        # Compare
        if gen_result is None:
            score = 0
            error_type = 'sql_error'
        elif gen_result == gold_result:
            score = 1
            error_type = None
        else:
            score = 0
            error_type = 'wrong_result'

        results.append({
            'uid': uid,
            'score': score,
            'error_type': error_type,
            'hardness': gold_info.get('hardness', ''),
            'level': gold_info.get('level', ''),
            'vocab': gold_info.get('vocab', ''),
            'base_para': gold_info.get('base_para', ''),
            'question': gold_info.get('question', ''),
            'gold_sql': gold_info.get('sql', ''),
            'generated_sql': gen_sql,
        })

    conn.close()
    return results


def print_results(results):
    """Print evaluation summary."""
    total = len(results)
    correct = sum(r['score'] for r in results)

    print(f"\n{'='*60}")
    print(f"OVERALL: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"{'='*60}")

    # Error breakdown
    errors = [r for r in results if r['score'] == 0]
    sql_errors = sum(1 for r in errors if r['error_type'] == 'sql_error')
    wrong_results = sum(1 for r in errors if r['error_type'] == 'wrong_result')
    print(f"\nErrors: {len(errors)} total")
    print(f"  SQL errors (couldn't execute): {sql_errors}")
    print(f"  Wrong results (executed but wrong): {wrong_results}")

    # Breakdown by qualifier
    for qualifier, label in [('hardness', 'HARDNESS'), ('level', 'EVENT/CASE LEVEL'),
                              ('vocab', 'VOCABULARY TYPE'), ('base_para', 'BASE vs PARAPHRASE')]:
        groups = {}
        for r in results:
            val = r.get(qualifier, '')
            if val:
                groups.setdefault(val, [0, 0])
                groups[val][0] += 1
                groups[val][1] += r['score']

        print(f"\n=== {label} ===")
        for k in sorted(groups.keys()):
            t, c = groups[k]
            print(f"  {k:30s} {c:4d}/{t:4d} = {c/t*100:.1f}%")


def save_results(results, output_path):
    """Save per-question results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate text-2-SQL-4-PM benchmark results')
    parser.add_argument('--results', required=True, help='Path to generated SQL results (TSV: uid \\t sql)')
    parser.add_argument('--db', default='Text2SQL4PM/event_log.sqlite', help='Path to SQLite database')
    parser.add_argument('--gold', default=None, help='Path to pre-computed gold results JSON')
    parser.add_argument('--output', default=None, help='Path to save detailed results JSON')
    parser.add_argument('--exclude-groups', default=None,
                        help='Comma-separated group IDs to exclude (for few-shot evaluation)')
    parser.add_argument('--dataset', default='Text2SQL4PM/dataset.tsv',
                        help='Path to dataset TSV (needed for --exclude-groups)')

    args = parser.parse_args()

    # Default gold path: same directory as this script
    if args.gold is None:
        args.gold = str(Path(__file__).parent / 'gold_results.json')

    # Build exclusion set from group IDs
    exclude_uids = set()
    if args.exclude_groups:
        import csv
        group_ids = set(args.exclude_groups.split(','))
        with open(args.dataset, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                uid = row.get('Utterance_id', '').strip()
                gid = row.get('Group_id', '').strip()
                if uid and gid in group_ids:
                    exclude_uids.add(uid)
        print(f"Excluding {len(exclude_uids)} questions from groups: {args.exclude_groups}")

    results = evaluate(args.db, args.gold, args.results, exclude_uids=exclude_uids)
    print_results(results)

    if args.output:
        save_results(results, args.output)
