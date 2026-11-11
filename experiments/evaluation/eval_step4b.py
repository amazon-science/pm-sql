"""
Evaluation for Step 4b: Answer agent.

Two evaluation methods:
1. Deterministic: Format gold results as markdown table, normalize both, compare.
2. LLM-as-judge: Give both tables to Claude, ask if they represent the same data.

Usage:
    # Deterministic only
    python eval_step4b.py results_step4b.tsv --mode deterministic

    # LLM judge only
    python eval_step4b.py results_step4b.tsv --mode llm-judge

    # Both (default)
    python eval_step4b.py results_step4b.tsv --mode both
"""

import csv
import json
import logging
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "llm_utils"))

from prompt_helpers import call_llm, ROLE_ARNS, REGION, SONNET_4_6

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

GOLD_PATH = Path(__file__).parent / 'gold_results.json'
DATASET_PATH = project_root / 'dataset.tsv'


# ============================================================================
# Deterministic evaluation
# ============================================================================

def gold_to_markdown(gold_result, gold_sql):
    """Convert gold result (list of lists) to a normalized markdown table."""
    if not gold_result:
        return ""

    # Try to extract column names from the gold SQL
    # Simple heuristic: look for SELECT ... FROM
    columns = extract_columns_from_sql(gold_sql, len(gold_result[0]))

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in gold_result:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(lines)


def extract_columns_from_sql(sql, num_cols):
    """Extract column names from SQL SELECT clause. Fallback to col1, col2, etc."""
    # This is a best-effort heuristic
    try:
        # Find the SELECT ... FROM part
        match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if match:
            select_part = match.group(1)
            # Split by comma, handle aliases
            parts = [p.strip() for p in select_part.split(',')]
            columns = []
            for p in parts:
                # Handle "expr AS alias" or just "expr"
                alias_match = re.search(r'\bAS\s+(\w+)', p, re.IGNORECASE)
                if alias_match:
                    columns.append(alias_match.group(1))
                else:
                    # Use the last word or the whole thing
                    columns.append(p.split('.')[-1].split('(')[0].strip())
            if len(columns) == num_cols:
                return columns
    except Exception:
        pass

    return [f"col{i+1}" for i in range(num_cols)]


def normalize_table(text):
    """Normalize a markdown table for comparison: lowercase, strip, sort rows."""
    if not text or not text.strip():
        return []

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]

    # Filter to only table rows (lines with |)
    table_lines = [l for l in lines if '|' in l]
    if not table_lines:
        return []

    # Parse each row into cells
    data_rows = []
    for line in table_lines:
        cells = [c.strip().lower() for c in line.split('|') if c.strip()]
        # Skip separator rows (all dashes)
        if cells and all(c.replace('-', '').replace(':', '').strip() == '' for c in cells):
            continue
        if cells:
            data_rows.append(tuple(cells))

    # Remove header row (first row is column names)
    if len(data_rows) > 1:
        data_rows = data_rows[1:]
    elif len(data_rows) == 1:
        # Only one data row — could be header or data. Keep it as data.
        pass

    return sorted(data_rows)


def deterministic_compare(agent_answer, gold_result):
    """Compare agent's markdown answer against gold result deterministically."""
    if not gold_result:
        # Gold is empty — agent should also be empty
        return "empty" in agent_answer.lower() or not agent_answer.strip()

    # Normalize agent's answer
    agent_rows = normalize_table(agent_answer)

    # Normalize gold result
    gold_rows = sorted([tuple(str(v).lower().strip() for v in row) for row in gold_result])

    if not agent_rows:
        return False

    return agent_rows == gold_rows


# ============================================================================
# LLM-as-judge evaluation
# ============================================================================

JUDGE_PROMPT = """You are evaluating whether two data tables contain the same information.

Compare the EXPECTED result with the ACTUAL result. They represent the same data if:
- They contain the same values (ignore column names, ordering, and formatting)
- Numbers can differ in formatting (e.g., "6449" vs "6,449" are the same)
- Case differences are acceptable (e.g., "Linda" vs "linda")
- Extra whitespace or formatting differences are acceptable

EXPECTED (gold standard):
{expected}

ACTUAL (agent's answer):
{actual}

Do these contain the same data? Respond with ONLY "YES" or "NO"."""


def llm_judge(agent_answer, gold_result, gold_sql, client_fn):
    """Use LLM to judge if agent's answer matches gold result."""
    if not gold_result:
        return "empty" in agent_answer.lower() or not agent_answer.strip()

    # Format gold as markdown for the judge
    gold_md = gold_to_markdown(gold_result, gold_sql)

    prompt = JUDGE_PROMPT.format(expected=gold_md, actual=agent_answer)

    try:
        response = client_fn(prompt)
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


# ============================================================================
# Main evaluation
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Step 4b answer agent')
    parser.add_argument('results', help='Path to results TSV')
    parser.add_argument('--mode', choices=['deterministic', 'llm-judge', 'both'], default='both')
    parser.add_argument('--exclude-groups', default='1,23,5,109,132')
    args = parser.parse_args()

    # Load exclusions
    exclude_uids = set()
    if args.exclude_groups:
        group_ids = set(args.exclude_groups.split(','))
        with open(DATASET_PATH) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                uid = row.get('Utterance_id', '').strip()
                gid = row.get('Group_id', '').strip()
                if uid and gid in group_ids:
                    exclude_uids.add(uid)
        print(f"Excluding {len(exclude_uids)} questions")

    # Load gold
    with open(GOLD_PATH) as f:
        gold_data = json.load(f)

    # Load agent answers (supports both TSV and JSONL)
    generated = {}
    results_path = Path(args.results)
    if results_path.suffix == '.jsonl':
        with open(results_path) as f:
            for line in f:
                entry = json.loads(line)
                generated[entry['uid']] = entry['answer']
    else:
        with open(results_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    generated[parts[0]] = parts[1]

    print(f"Generated: {len(generated)}, Gold: {len(gold_data)}")

    # Evaluate
    det_results = []
    llm_results = []

    items = [(uid, answer) for uid, answer in generated.items()
             if uid in gold_data and uid not in exclude_uids and gold_data[uid]['result'] is not None]

    print(f"Evaluating {len(items)} questions")

    # Deterministic evaluation
    if args.mode in ('deterministic', 'both'):
        print("\n--- Deterministic Evaluation ---")
        correct = 0
        for uid, answer in items:
            gold_result = gold_data[uid]['result']
            if deterministic_compare(answer, gold_result):
                correct += 1
        print(f"DETERMINISTIC: {correct}/{len(items)} = {correct/len(items)*100:.1f}%")
        det_results = correct

    # LLM judge evaluation
    if args.mode in ('llm-judge', 'both'):
        print("\n--- LLM Judge Evaluation ---")
        print("(Running in parallel across Bedrock accounts)")

        import boto3
        from botocore.config import Config as BotoConfig

        # Create clients for parallel LLM judge calls
        def get_judge_client(role_arn):
            sts = boto3.client("sts", region_name=REGION)
            creds = sts.assume_role(
                RoleArn=role_arn, RoleSessionName="Step4b-Judge", DurationSeconds=3600
            )["Credentials"]
            return boto3.client(
                "bedrock-runtime", region_name=REGION,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                config=BotoConfig(connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}),
            )

        def call_judge(client, prompt):
            import json as _json
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            }
            response = client.invoke_model(modelId=SONNET_4_6, body=_json.dumps(body))
            result = _json.loads(response["body"].read())
            return result["content"][0]["text"]

        # Parallel judge
        judge_lock = threading.Lock()
        judge_correct = [0]
        judge_done = [0]

        def judge_worker(role_arn, chunk):
            try:
                client = get_judge_client(role_arn)
            except Exception as e:
                logger.error(f"Judge worker cannot assume role: {e}")
                with judge_lock:
                    judge_done[0] += len(chunk)
                return

            for uid, answer in chunk:
                gold_result = gold_data[uid]['result']
                gold_sql = gold_data[uid].get('sql', '')

                try:
                    result = llm_judge(
                        answer, gold_result, gold_sql,
                        lambda p: call_judge(client, p)
                    )
                    if result:
                        with judge_lock:
                            judge_correct[0] += 1
                except Exception:
                    pass

                with judge_lock:
                    judge_done[0] += 1

        # Distribute items across workers
        num_judge_workers = min(len(ROLE_ARNS), 10)
        judge_chunks = [[] for _ in range(num_judge_workers)]
        for i, item in enumerate(items):
            judge_chunks[i % num_judge_workers].append(item)

        start_judge = time.time()
        with ThreadPoolExecutor(max_workers=num_judge_workers) as executor:
            futures = []
            for i, chunk in enumerate(judge_chunks):
                if chunk:
                    futures.append(executor.submit(judge_worker, ROLE_ARNS[i % len(ROLE_ARNS)], chunk))

            while not all(f.done() for f in futures):
                time.sleep(5)
                with judge_lock:
                    done = judge_done[0]
                    c = judge_correct[0]
                rate = done / (time.time() - start_judge) if done > 0 else 0
                print(f"  [{done}/{len(items)}] {rate:.1f} q/s, "
                      f"running: {c}/{done} = {c/max(done,1)*100:.1f}%", flush=True)

            for f in futures:
                f.result()

        correct = judge_correct[0]
        elapsed_judge = time.time() - start_judge
        print(f"LLM JUDGE: {correct}/{len(items)} = {correct/len(items)*100:.1f}% ({elapsed_judge:.0f}s)")

    print(f"\n{'='*60}")
    if args.mode == 'both':
        print(f"Deterministic: {det_results}/{len(items)} = {det_results/len(items)*100:.1f}%")
        print(f"LLM Judge:     {correct}/{len(items)} = {correct/len(items)*100:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
