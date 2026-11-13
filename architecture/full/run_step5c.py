"""
Step 5: SQL agent with process mining artifacts tool.

Extends Step 4a by adding a get_process_info tool that provides
pre-computed PM knowledge (DFG, variants, activities, rework, etc.).

Uses Strands Agent with Claude Sonnet 4.5 and two tools:
  1. validate_sql — check SQL executes correctly
  2. get_process_info — retrieve process mining artifacts

Runs in parallel across multiple Bedrock accounts.

Usage:
    python run_step5.py [--workers N] [--start N] [--end N]
"""

import csv
import logging
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

from strands import Agent
from strands.models import BedrockModel

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "llm_utils"))
sys.path.insert(0, str(Path(__file__).parent))

from prompt_helpers import ROLE_ARNS, REGION
from prompts import SYSTEM_PROMPT
from sql_tool import validate_sql
from pm_tool import get_process_info

AGENT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ============================================================================
# Agent creation
# ============================================================================

def create_agent(boto_session):
    """Create a Strands agent with SQL validation + PM artifacts tools."""
    model = BedrockModel(
        boto_session=boto_session,
        model_id=AGENT_MODEL,
        max_tokens=16000,
        temperature=0,
    )

    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[validate_sql, get_process_info],
        callback_handler=None,
    )


def get_boto_session(role_arn):
    """Assume role, or use default credentials when role_arn is None."""
    if role_arn is None:
        return boto3.Session()
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn, RoleSessionName="Text2SQL4PM-Step5c", DurationSeconds=3600
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=REGION,
    )


# ============================================================================
# SQL extraction
# ============================================================================

def extract_sql(result):
    """Extract SQL from agent response using <final_sql> tags."""
    text = str(result)

    match = re.search(r'<final_sql>(.*?)</final_sql>', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = ' '.join(sql.split())
        return sql.split(';')[0].strip()

    for keyword in ["SELECT", "WITH"]:
        idx = text.upper().find(keyword)
        if idx >= 0:
            sql = text[idx:]
            if "```" in sql:
                sql = sql.split("```")[0]
            sql = sql.split(';')[0].strip()
            return ' '.join(sql.split())

    return "SELECT 1"


# ============================================================================
# Question loading
# ============================================================================

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


# ============================================================================
# Worker
# ============================================================================

_results_lock = threading.Lock()
_results = []
_progress = {"done": 0, "errors": 0, "total": 0}


def worker(role_arn, questions_chunk, worker_id):
    """Process a chunk of questions using a dedicated Bedrock account."""
    try:
        boto_session = get_boto_session(role_arn)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Cannot assume role: {e}")
        with _results_lock:
            for uid, _ in questions_chunk:
                _results.append((uid, "SELECT 1", "0:0"))
                _progress["errors"] += 1
                _progress["done"] += 1
        return

    for uid, question in questions_chunk:
        try:
            agent = create_agent(boto_session)
            result = agent(question)
            sql = extract_sql(result)

            usage = result.metrics.accumulated_usage
            token_info = f"{usage.get('inputTokens', 0)}:{usage.get('outputTokens', 0)}"

        except Exception as e:
            logger.warning(f"Worker {worker_id}: ERROR on {uid}: {e}")
            sql = "SELECT 1"
            token_info = "0:0"
            with _results_lock:
                _progress["errors"] += 1

            if "ExpiredToken" in str(e) or "expired" in str(e).lower():
                try:
                    boto_session = get_boto_session(role_arn)
                except Exception:
                    pass

        with _results_lock:
            _results.append((uid, sql, token_info))
            _progress["done"] += 1


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 5c: SQL agent + PM + targeted examples (parallel)')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=str(project_root / 'dataset.tsv'))
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(__file__).parent / 'results_step5c.tsv')

    num_workers = args.workers if args.workers else len(ROLE_ARNS)

    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions from dataset")

    end = args.end if args.end is not None else len(questions)
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

    chunks = [[] for _ in range(num_workers)]
    for i, item in enumerate(questions):
        chunks[i % num_workers].append(item)

    active_chunks = [(ROLE_ARNS[i % len(ROLE_ARNS)], chunk, i)
                     for i, chunk in enumerate(chunks) if chunk]

    print(f"Starting {len(active_chunks)} workers")
    print()

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
                  f"~{remaining/60:.1f}min remaining, {_progress['errors']} errors",
                  flush=True)

        for f in futures:
            f.result()

    elapsed = time.time() - start_time

    with open(args.output, 'a') as f:
        for uid, sql, token_info in _results:
            f.write(f"{uid}\t{sql}\t{token_info}\n")

    # Token summary
    total_in = sum(int(t.split(':')[0]) for _, _, t in _results if ':' in t and t.split(':')[0].isdigit())
    total_out = sum(int(t.split(':')[1]) for _, _, t in _results if ':' in t and t.split(':')[1].isdigit())

    print(f"\n{'='*60}")
    print(f"Done! {len(_results)} questions in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Rate: {len(_results)/elapsed:.1f} questions/sec")
    print(f"Errors: {_progress['errors']}")
    print(f"Tokens: {total_in + total_out:,} total ({total_in:,} in, {total_out:,} out)")
    print(f"Avg per question: {(total_in + total_out) // max(len(_results), 1):,}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
