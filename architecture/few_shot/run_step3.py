"""
Step 3: Enhanced prompts for Claude Sonnet 4.6.

3a (--mode zero-shot): Enhanced system prompt with schema details, PM patterns, no examples.
3b (--mode few-shot): Same system prompt + few-shot worked examples as conversation turns.

Uses the same parallel infrastructure as Step 2.

Usage:
    python run_step3.py --mode zero-shot   # Step 3a
    python run_step3.py --mode few-shot    # Step 3b
"""

import csv
import json
import logging
import os
import random
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "llm_utils"))
sys.path.insert(0, str(Path(__file__).parent))

from prompt_helpers import SONNET_4_6, ROLE_ARNS, REGION
import prompts as prompts_module

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Bedrock client helpers (same as Step 2)
# ============================================================================

NON_RETRYABLE_ERROR_CODES = {"ValidationException", "AccessDeniedException"}
REAUTH_ERROR_CODES = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException"}
RETRYABLE_EXCEPTIONS = (ConnectionError, ReadTimeoutError, ConnectTimeoutError)


def _assume_role(role_arn: str):
    if role_arn is None:
        return boto3.client("bedrock-runtime", region_name=REGION,
            config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}))
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="Text2SQL4PM-Step3",
        DurationSeconds=3600,
    )["Credentials"]
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}),
    )


def _calculate_backoff(attempt: int, base: float = 1.0, max_delay: float = 30.0) -> float:
    delay = min(max_delay, base * (2 ** attempt))
    return random.uniform(0, delay)


def call_bedrock(client, role_arn, body, model=SONNET_4_6, max_retries=5):
    """Call Bedrock with retry + re-auth. Returns (response_dict, client)."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = client.invoke_model(modelId=model, body=json.dumps(body))
            result = json.loads(response["body"].read())
            return result, client

        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code", "")
            last_exception = error

            if error_code in NON_RETRYABLE_ERROR_CODES:
                raise

            if error_code in REAUTH_ERROR_CODES:
                if attempt < max_retries:
                    logger.warning(f"[{role_arn[-12:]}] Auth expired, re-authenticating...")
                    try:
                        client = _assume_role(role_arn)
                    except Exception:
                        pass
                    time.sleep(_calculate_backoff(attempt, base=0.5, max_delay=5.0))
                    continue
                raise

            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(f"[{role_arn[-12:]}] {error_code}, retry {attempt+1}/{max_retries+1} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise

        except RETRYABLE_EXCEPTIONS as error:
            last_exception = error
            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(f"[{role_arn[-12:]}] Network error, retry {attempt+1}/{max_retries+1} in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise

    raise last_exception


# ============================================================================
# SQL generation
# ============================================================================

def generate_sql_zero_shot(client, role_arn, question):
    """Step 3a: system prompt + question, no examples."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0,
        "system": prompts_module.SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": question}]}
        ],
    }
    result, client = call_bedrock(client, role_arn, body)
    return clean_sql(result["content"][0]["text"]), client


def generate_sql_few_shot(client, role_arn, question):
    """Step 3b: system prompt + few-shot examples + question."""
    # Build messages: alternating user/assistant for examples, then the actual question
    messages = []
    for ex in prompts_module.FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": [{"type": "text", "text": ex["question"]}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["sql"]}]})
    # Actual question
    messages.append({"role": "user", "content": [{"type": "text", "text": question}]})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0,
        "system": prompts_module.SYSTEM_PROMPT,
        "messages": messages,
    }
    result, client = call_bedrock(client, role_arn, body)
    return clean_sql(result["content"][0]["text"]), client


def clean_sql(response_text):
    """Clean model response into a valid SQL statement."""
    sql = response_text.strip()

    # Remove markdown fences
    if "```" in sql:
        parts = sql.split("```")
        for part in parts:
            cleaned = part.replace("sql", "", 1).strip() if part.startswith("sql") else part.strip()
            if cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH"):
                sql = cleaned
                break

    # Handle WITH clauses
    if sql.upper().startswith("WITH") or sql.upper().startswith("SELECT"):
        pass
    elif not sql.upper().startswith("SELECT"):
        sql = "SELECT " + sql

    # Take only the first statement
    sql = sql.split(';')[0].strip()

    # Collapse to single line
    sql = ' '.join(sql.split())

    return sql


# ============================================================================
# Question loading
# ============================================================================

def load_questions(dataset_path):
    """Load questions from dataset.tsv."""
    questions = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row['Utterance_id'].strip()
            question = row['English_utterance'].strip()
            if uid and question:
                questions.append((uid, question))
    return questions


# ============================================================================
# Worker
# ============================================================================

_results_lock = threading.Lock()
_results = []
_progress = {"done": 0, "errors": 0, "total": 0}


def worker(role_arn, questions_chunk, worker_id, generate_fn):
    """Process a chunk of questions using a dedicated Bedrock account."""
    try:
        client = _assume_role(role_arn)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Cannot assume role {role_arn}: {e}")
        with _results_lock:
            for uid, _ in questions_chunk:
                _results.append((uid, "SELECT 1"))
                _progress["errors"] += 1
                _progress["done"] += 1
        return

    for uid, question in questions_chunk:
        try:
            sql, client = generate_fn(client, role_arn, question)
        except Exception as e:
            logger.warning(f"Worker {worker_id}: ERROR on {uid}: {e}")
            sql = "SELECT 1"
            with _results_lock:
                _progress["errors"] += 1

        with _results_lock:
            _results.append((uid, sql))
            _progress["done"] += 1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Step 3: Enhanced prompts (parallel)')
    parser.add_argument('--mode', choices=['zero-shot', 'few-shot'], required=True,
                        help='zero-shot (3a) or few-shot (3b)')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=str(project_root / 'dataset.tsv'))
    args = parser.parse_args()

    # Output path
    if args.output is None:
        suffix = "3a" if args.mode == "zero-shot" else "3b"
        args.output = str(Path(__file__).parent / f'results_step{suffix}.tsv')

    num_workers = args.workers if args.workers else len(ROLE_ARNS)

    # Select generation function
    generate_fn = generate_sql_zero_shot if args.mode == "zero-shot" else generate_sql_few_shot

    # Load questions
    questions = load_questions(args.dataset)
    total = len(questions)
    print(f"Loaded {total} questions from dataset")
    print(f"Mode: {args.mode}")

    # Slice
    end = args.end if args.end is not None else total
    questions = questions[args.start:end]

    # Resume support
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

    # Distribute
    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(questions):
        chunks[i % num_workers].append(item)

    active_chunks = [(ROLE_ARNS[i % len(ROLE_ARNS)], chunk, i)
                     for i, chunk in enumerate(chunks) if chunk]

    print(f"Starting {len(active_chunks)} workers across {len(ROLE_ARNS)} accounts")
    print(f"~{len(questions) // len(active_chunks)} questions per worker")
    print()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(active_chunks)) as executor:
        futures = []
        for role_arn, chunk, worker_id in active_chunks:
            futures.append(executor.submit(worker, role_arn, chunk, worker_id, generate_fn))

        while not all(f.done() for f in futures):
            time.sleep(5)
            elapsed = time.time() - start_time
            done = _progress["done"]
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (_progress["total"] - done) / rate if rate > 0 else 0
            print(f"  [{done}/{_progress['total']}] {rate:.1f} q/s, "
                  f"~{remaining/60:.1f}min remaining, {_progress['errors']} errors",
                  flush=True)

        for f in futures:
            f.result()

    elapsed = time.time() - start_time

    # Write results
    with open(args.output, 'a') as f:
        for uid, sql in _results:
            f.write(f"{uid}\t{sql}\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(_results)} questions in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Rate: {len(_results)/elapsed:.1f} questions/sec")
    print(f"Errors: {_progress['errors']}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
