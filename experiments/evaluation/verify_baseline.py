"""
Verify our evaluation harness produces results consistent with the published
GPT-3.5 baseline (47.6% execution accuracy).

Reads the published GPT-3.5 evaluation scores and compares against our
own evaluation of the same generated SQL.
"""

import csv
import sqlite3
import sys
from pathlib import Path


def main():
    db_path = Path('Text2SQL4PM/event_log.sqlite')
    scores_path = Path('Text2SQL4PM/scores_gpt35_execution_accuracy.tsv')
    dataset_path = Path('Text2SQL4PM/dataset.tsv')
    
    # Load published scores
    published = {}
    with open(scores_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                uid = parts[0].strip()
                gen_sql = parts[1].strip()
                gold_sql = parts[2].strip()
                published_score = int(parts[4].strip())
                published[uid] = {
                    'gen_sql': gen_sql,
                    'gold_sql': gold_sql,
                    'published_score': published_score,
                }
    
    print(f"Loaded {len(published)} published scores")
    
    # Re-evaluate a sample using our harness
    conn = sqlite3.connect(str(db_path))
    
    agree = 0
    disagree = 0
    our_correct = 0
    sample_size = min(200, len(published))  # Check first 200
    
    for i, (uid, data) in enumerate(published.items()):
        if i >= sample_size:
            break
        
        try:
            cur = conn.cursor()
            cur.execute(data['gold_sql'])
            gold_rows = cur.fetchall()
            gold_result = sorted([tuple(str(v).strip().lower() if v is not None else '' for v in row) for row in gold_rows])
        except:
            gold_result = None
        
        try:
            cur = conn.cursor()
            cur.execute(data['gen_sql'])
            gen_rows = cur.fetchall()
            gen_result = sorted([tuple(str(v).strip().lower() if v is not None else '' for v in row) for row in gen_rows])
        except:
            gen_result = None
        
        our_score = 1 if (gen_result is not None and gold_result is not None and gen_result == gold_result) else 0
        our_correct += our_score
        
        if our_score == data['published_score']:
            agree += 1
        else:
            disagree += 1
            if disagree <= 5:  # Show first 5 disagreements
                print(f"\n  DISAGREE on {uid}: ours={our_score}, published={data['published_score']}")
                print(f"    Gold SQL: {data['gold_sql'][:100]}")
                print(f"    Gen SQL:  {data['gen_sql'][:100]}")
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Verification on {sample_size} questions:")
    print(f"  Agreement with published scores: {agree}/{sample_size} ({agree/sample_size*100:.1f}%)")
    print(f"  Disagreements: {disagree}")
    print(f"  Our accuracy on sample: {our_correct}/{sample_size} ({our_correct/sample_size*100:.1f}%)")
    print(f"  Published overall accuracy: 47.6%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
