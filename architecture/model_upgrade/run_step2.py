"""
Step 2: Claude Sonnet 4.6 with the benchmark's minimal prompt.

Uses the exact same prompt template as the text-2-SQL-4-PM paper's GPT-3.5
baseline, but swaps the model to Claude Sonnet 4.6 via Bedrock.

Runs in parallel across multiple Bedrock accounts (one worker per role ARN)
to maximize throughput. Each worker gets its own client and processes a
chunk of questions independently.

Prompt template:
    ### Complete sqlite SQL query only and with no explanation
    ### SQLite SQL tables, with their properties:
    #
    # event_log(id, activity, timestamp, resource, cost, idcase)
    #
    ### {question}
    SELECT

Usage:
    python run_step2.py [--workers N] [--output results.tsv]

Output: TSV file with utterance_id \t generated_sql (ready for evaluate.py)
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

from prompt_helpers import SONNET_4_6, ROLE_ARNS, REGION

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prompt template (exact replica of the benchmark's minimal prompt)
# ============================================================================

PROMPT_TEMPLATE = """### Complete sqlite SQL query only and with no explanation
### SQLite SQL tables, with their properties:
#
# event_log(id, activity, timestamp, resource, cost, idcase)
#
### {question}
SELECT """

# ============================================================================
# Per-worker Bedrock client (each worker gets its own account)
# ============================================================================

NON_RETRYABLE_ERROR_CODES = {"ValidationException", "AccessDeniedException"}
REAUTH_ERROR_CODES = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException"}
RETRYABLE_EXCEPTIONS = (ConnectionError, ReadTimeoutError, ConnectTimeoutError)


def _assume_role(role_arn: str):
    """Assume a cross-account role and return a bedrock-runtime client."""
    if role_arn is None:
        return boto3.client("bedrock-runtime", region_name=REGION,
            config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}))
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="Text2SQL4PM-Step2",
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


def call_llm_with_client(client, role_arn, prompt, model=SONNET_4_6, max_tokens=256,
                          temperature=0, max_retries=5):
    """Call LLM with a specific client. Handles retry + re-auth."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = client.invoke_model(modelId=model, body=json.dumps(body))
            result = json.loads(response["body"].read())
            return result["content"][0]["text"], client

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
                logger.warning(
                    f"[{role_arn[-12:]}] {error_code}, retry {attempt+1}/{max_retries+1} "
                    f"in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise

        except RETRYABLE_EXCEPTIONS as error:
            last_exception = error
            if attempt < max_retries:
                delay = _calculate_backoff(attempt)
                logger.warning(
                    f"[{role_arn[-12:]}] Network error, retry {attempt+1}/{max_retries+1} "
                    f"in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            raise

    raise last_exception


# ============================================================================
# Question processing
# ============================================================================

def load_questions(dataset_path):
    """Load questions from dataset.tsv. Returns list of (utterance_id, question)."""
    questions = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            uid = row['Utterance_id'].strip()
            question = row['English_utterance'].strip()
            if uid and question:
                questions.append((uid, question))
    return questions


def clean_sql(response_text):
    """Clean model response into a valid SQL statement."""
    sql = response_text.strip()

    # Remove markdown fences
    if "```" in sql:
        parts = sql.split("```")
        for part in parts:
            cleaned = part.replace("sql", "", 1).strip() if part.startswith("sql") else part.strip()
            if cleaned.upper().startswith("SELECT"):
                sql = cleaned
                break

    # Prepend SELECT if model only returned the completion
    if not sql.upper().startswith("SELECT"):
        sql = "SELECT " + sql

    # Take only the first statement
    sql = sql.split(';')[0].strip()
    return sql


# ============================================================================
# Worker function (one per account)
# ============================================================================

# Thread-safe results collection
_results_lock = threading.Lock()
_results = []
_progress = {"done": 0, "errors": 0, "total": 0}


def worker(role_arn, questions_chunk, worker_id):
    """Process a chunk of questions using a dedicated Bedrock account."""
    try:
        client = _assume_role(role_arn)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Cannot assume role {role_arn}: {e}")
        # Mark all as errors
        with _results_lock:
            for uid, _ in questions_chunk:
                _results.append((uid, "SELECT 1"))
                _progress["errors"] += 1
                _progress["done"] += 1
        return

    for uid, question in questions_chunk:
        prompt = PROMPT_TEMPLATE.format(question=question)
        try:
            response_text, client = call_llm_with_client(
                client, role_arn, prompt, model=SONNET_4_6, max_tokens=256, temperature=0
            )
            sql = clean_sql(response_text)
        except Exception as e:
            logger.warning(f"Worker {worker_id}: ERROR on {uid}: {e}")
            sql = "SELECT 1"
            with _results_lock:
                _progress["errors"] += 1

        with _results_lock:
            _results.append((uid, sql.replace('\n', ' ').replace('\r', ' ')))
            _progress["done"] += 1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Step 2: Claude Sonnet 4.6 minimal prompt (parallel)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of role ARNs)')
    parser.add_argument('--start', type=int, default=0, help='Start index (0-based)')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--output', type=str, default=None, help='Output TSV path')
    parser.add_argument('--dataset', type=str,
                        default=str(project_root / 'dataset.tsv'),
                        help='Path to dataset.tsv')
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent / 'results_step2.tsv')

    num_workers = args.workers if args.workers else len(ROLE_ARNS)

    # Load questions
    questions = load_questions(args.dataset)
    total = len(questions)
    print(f"Loaded {total} questions from dataset")

    # Slice
    end = args.end if args.end is not None else total
    questions = questions[args.start:end]

    # Load existing results to support resume
    existing = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    existing.add(parts[0])
        print(f"Found {len(existing)} existing results (will skip)")

    # Filter out already-done questions
    questions = [(uid, q) for uid, q in questions if uid not in existing]
    print(f"Questions to process: {len(questions)}")

    if not questions:
        print("Nothing to do!")
        return

    _progress["total"] = len(questions)

    # Distribute questions across workers (round-robin for even distribution)
    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(questions):
        chunks[i % num_workers].append(item)

    # Remove empty chunks
    active_chunks = [(ROLE_ARNS[i % len(ROLE_ARNS)], chunk, i)
                     for i, chunk in enumerate(chunks) if chunk]

    print(f"Starting {len(active_chunks)} workers across {len(ROLE_ARNS)} accounts")
    print(f"~{len(questions) // len(active_chunks)} questions per worker")
    print()

    start_time = time.time()

    # Launch workers
    with ThreadPoolExecutor(max_workers=len(active_chunks)) as executor:
        futures = []
        for role_arn, chunk, worker_id in active_chunks:
            futures.append(executor.submit(worker, role_arn, chunk, worker_id))

        # Progress monitor
        while not all(f.done() for f in futures):
            time.sleep(5)
            elapsed = time.time() - start_time
            done = _progress["done"]
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (_progress["total"] - done) / rate if rate > 0 else 0
            print(f"  [{done}/{_progress['total']}] {rate:.1f} q/s, "
                  f"~{remaining/60:.1f}min remaining, {_progress['errors']} errors",
                  flush=True)

        # Wait for all to complete
        for f in futures:
            f.result()  # Raises if worker had unhandled exception

    elapsed = time.time() - start_time

    # Write results (append to support resume)
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
