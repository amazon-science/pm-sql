"""
Step 4a (SQL validation agent) for open-source models via Strands + Bedrock.

Uses the same system prompt as Step 4a but with configurable model ID.

Usage:
    python run_step4a_oss.py --model us.deepseek.r1-v1:0 --output results_deepseek.tsv
    python run_step4a_oss.py --model openai.gpt-oss-120b-1:0 --output results_gptoss120b.tsv
    python run_step4a_oss.py --model openai.gpt-oss-20b-1:0 --output results_gptoss20b.tsv
    python run_step4a_oss.py --model qwen.qwen3-32b-v1:0 --output results_qwen3.tsv
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

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "llm_utils"))

# Import prompts and tools from step4a
sys.path.insert(0, str(project_root / "experiments" / "step4a_sql_agent"))
from prompt_helpers import ROLE_ARNS, REGION
from prompts import SYSTEM_PROMPT
from sql_tool import validate_sql

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def create_agent(boto_session, model_id):
    model = BedrockModel(
        boto_session=boto_session,
        model_id=model_id,
        max_tokens=16000,
        temperature=0,
        streaming=False,
    )
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[validate_sql],
        callback_handler=None,
    )


def get_boto_session(role_arn):
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn, RoleSessionName="Text2SQL4PM-OSS-4a", DurationSeconds=3600
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=REGION,
    )


def extract_sql(result):
    text = str(result)

    # Remove thinking tags (DeepSeek R1)
    if '<think>' in text:
        think_end = text.find('</think>')
        if think_end > 0:
            text = text[think_end + len('</think>'):]

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


def worker(role_arn, questions_chunk, worker_id, model_id):
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
            agent = create_agent(boto_session, model_id)
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 4a: SQL agent on OSS models (Strands)')
    parser.add_argument('--model', type=str, required=True, help='Bedrock model ID')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=str(project_root / 'dataset.tsv'))
    args = parser.parse_args()

    num_workers = args.workers if args.workers else len(ROLE_ARNS)

    questions = load_questions(args.dataset)
    print(f"Loaded {len(questions)} questions")
    print(f"Model: {args.model}")

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
        futures = [executor.submit(worker, r, c, w, args.model) for r, c, w in active_chunks]
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
        for uid, sql, token_info in _results:
            f.write(f"{uid}\t{sql}\t{token_info}\n")

    total_in = sum(int(t.split(':')[0]) for _, _, t in _results if ':' in t and t.split(':')[0].isdigit())
    total_out = sum(int(t.split(':')[1]) for _, _, t in _results if ':' in t and t.split(':')[1].isdigit())

    print(f"\n{'='*60}")
    print(f"Done! {len(_results)} questions in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Rate: {len(_results)/elapsed:.1f} questions/sec")
    print(f"Errors: {_progress['errors']}")
    print(f"Tokens: {total_in + total_out:,} total")
    print(f"Results: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
