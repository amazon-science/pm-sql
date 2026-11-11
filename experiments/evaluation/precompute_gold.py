"""
Pre-compute gold SQL results for all 1,655 questions.

Executes each gold SQL statement against event_log.sqlite once and saves
the normalized result sets to a JSON file. This eliminates the need to
re-execute gold SQL during every evaluation run.

Output: gold_results.json
    {
        "utterance_id": {
            "result": [["val1", "val2"], ["val3", "val4"]],  // sorted rows
            "error": null
        },
        ...
    }

Usage:
    python precompute_gold.py
"""

import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


class QueryTimeout(Exception):
    pass


def execute_sql(conn, sql, timeout_seconds=10):
    """Execute SQL and return normalized results, with timeout via progress handler."""
    start = time.time()

    def progress_check():
        if time.time() - start > timeout_seconds:
            return 1  # Non-zero = interrupt
        return 0

    conn.set_progress_handler(progress_check, 1000)  # Check every 1000 VM instructions
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
        return sorted(normalized), None
    except sqlite3.OperationalError as e:
        if "interrupt" in str(e).lower():
            return None, "timeout"
        return None, str(e)
    except Exception as e:
        return None, str(e)
    finally:
        conn.set_progress_handler(None, 0)


def main():
    db_path = project_root / 'event_log.sqlite'
    dataset_path = project_root / 'dataset.tsv'
    output_path = Path(__file__).parent / 'gold_results.json'

    conn = sqlite3.connect(str(db_path))

    # Load dataset
    gold_data = {}
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row.get('Utterance_id', '').strip()
            sql = row.get('English_SQL_statement', '').strip()
            if uid and sql:
                gold_data[uid] = {
                    'sql': sql,
                    'question': row.get('English_utterance', ''),
                    'hardness': row.get('Spider_classification', ''),
                    'level': row.get('Events_cases_classification', ''),
                    'vocab': row.get('Domain_value_generic', ''),
                    'base_para': row.get('Base_paraphrase', ''),
                }

    print(f"Loaded {len(gold_data)} gold SQL statements")

    # Execute all gold SQL
    results = {}
    errors = 0
    for i, (uid, info) in enumerate(gold_data.items()):
        result, error = execute_sql(conn, info['sql'])
        results[uid] = {
            'result': result,
            'error': error,
            'sql': info['sql'],
            'question': info['question'],
            'hardness': info['hardness'],
            'level': info['level'],
            'vocab': info['vocab'],
            'base_para': info['base_para'],
        }
        if error:
            errors += 1
            print(f"  ERROR on {uid}: {error}")
            print(f"    SQL: {info['sql'][:100]}")

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(gold_data)}] processed")

    conn.close()

    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f)

    print(f"\nDone! {len(results)} gold results computed")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
