"""
Step 3b with Claude Sonnet 4.5 extended thinking enabled.

Uses the same few-shot prompt as Step 3b but with extended thinking
to compare fairly with DeepSeek R1's built-in reasoning.

Usage:
    python run_step3b_claude_thinking.py --output results_claude_thinking.tsv
"""

import csv
import json
import logging
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "llm_utils"))
sys.path.insert(0, str(project_root / "experiments" / "step3_prompt_engineering"))

from prompt_helpers import ROLE_ARNS, REGION
import prompts as prompts_module

AGENT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
THINKING_BUDGET = 4000

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_client(role_arn):
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn, RoleSessionName="Text2SQL4PM-Claude-Thinking", DurationSeconds=3600
    )["Credentials"]
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}),
    )


def generate_sql(client, role_arn, question):
    """Call Claude with extended thinking + few-shot examples."""
    messages = []
    for ex in prompts_module.FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": [{"type": "text", "text": ex["question"]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["sql"]}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "temperature": 1,  # Required when thinking is enabled
        "thinking": {
            "type": "enabled",
            "budget_tokens": THINKING_BUDGET,
        },
        "system": [{"type": "text", "text": prompts_module.SYSTEM_PROMPT}],
        "messages": messages,
    }

    try:
        response = client.invoke_model(modelId=AGENT_MODEL, body=json.dumps(body))
        result = json.loads(response["body"].read())
        return extract_sql_from_thinking_response(result), client
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException"):
            client = get_client(role_arn)
            response = client.invoke_model(modelId=AGENT_MODEL, body=json.dumps(body))
            result = json.loads(response["body"].read())
            return extract_sql_from_thinking_response(result), client
        raise


def extract_sql_from_thinking_response(result):
    """Extract SQL from Claude thinking response (has thinking + text blocks)."""
    text_parts = []
    for block in result.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block["text"])

    text = "\n".join(text_parts) if text_parts else str(result)
    return clean_sql(text)


def clean_sql(response_text):
    sql = response_text.strip()

    if "```" in sql:
        parts = sql.split("```")
        for part in parts:
            cleaned = part.replace("sql", "", 1).strip() if part.startswith("sql") else part.strip()
            if cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH"):
                sql = cleaned
                break

    if not (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
        for keyword in ["SELECT", "WITH"]:
            idx = sql.upper().find(keyword)
            if idx >= 0:
                sql = sql[idx:]
                break
        else:
            sql = "SELECT 1"

    sql = sql.split(';')[0].strip()
    sql = ' '.join(sql.split())
    return sql


def load_questions(dataset_path):
    questions = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row['Utterance_id'].strip()
            question = row['English_utterance'].strip()
            if uid and question:
                questions.append((uid, question))
    return questions


_results_lock = threading.Lock()
_results = []
_progress = {"done": 0, "errors": 0, "total": 0}


def worker(role_arn, questions_chunk, worker_id):
    try:
        client = get_client(role_arn)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Cannot assume role: {e}")
        with _results_lock:
            for uid, _ in questions_chunk:
                _results.append((uid, "SELECT 1"))
                _progress["errors"] += 1
                _progress["done"] += 1
        return

    for uid, question in questions_chunk:
        try:
            sql, client = generate_sql(client, role_arn, question)
        except Exception as e:
            if _progress["errors"] < 5:
                logger.warning(f"Worker {worker_id}: ERROR on {uid}: {type(e).__name__}: {str(e)[:200]}")
            else:
                logger.warning(f"Worker {worker_id}: ERROR on {uid}: {type(e).__name__}")
            sql = "SELECT 1"
            with _results_lock:
                _progress["errors"] += 1
            if "ExpiredToken" in str(e) or "expired" in str(e).lower():
                try:
                    client = get_client(role_arn)
                except Exception:
                    pass

        with _results_lock:
            _results.append((uid, sql))
            _progress["done"] += 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 3b: Claude Sonnet 4.5 with extended thinking')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=str(project_root / 'dataset.tsv'))
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent / 'results_claude_thinking.tsv')

    num_workers = args.workers if args.workers else len(ROLE_ARNS)

    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")
    print(f"Model: {AGENT_MODEL} (thinking budget: {THINKING_BUDGET} tokens)")

    end = args.end if args.end is not None else len(questions)
    questions = questions[args.start:end]

    existing = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    existing.add(parts[0])
        print(f"Found {len(existing)} existing results (will skip)")

    questions = [(uid, q) for uid, q in questions if uid not in existing]
    print(f"Questions to process: {len(questions)}")

    if not questions:
        print("Nothing to do!")
        return

    _progress["total"] = len(questions)

    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(questions):
        chunks[i % num_workers].append(item)

    active_chunks = [(ROLE_ARNS[i % len(ROLE_ARNS)], chunk, i)
                     for i, chunk in enumerate(chunks) if chunk]

    print(f"Starting {len(active_chunks)} workers\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(active_chunks)) as executor:
        futures = [executor.submit(worker, r, c, w) for r, c, w in active_chunks]
        while not all(f.done() for f in futures):
            time.sleep(10)
            elapsed = time.time() - start_time
            done = _progress["done"]
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (_progress["total"] - done) / rate if rate > 0 else 0
            print(f"  [{done}/{_progress['total']}] {rate:.1f} q/s, "
                  f"~{remaining/60:.1f}min remaining, {_progress['errors']} errors", flush=True)
        for f in futures:
            f.result()

    elapsed = time.time() - start_time

    with open(args.output, 'a') as f:
        for uid, sql in _results:
            f.write(f"{uid}\t{sql}\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(_results)} questions in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Rate: {len(_results)/elapsed:.1f} questions/sec")
    print(f"Errors: {_progress['errors']}")
    print(f"Results: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
