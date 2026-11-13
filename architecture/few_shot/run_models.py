"""
Step 3b (few-shot) for open-source models via Bedrock Converse API.

Runs the same enhanced prompt + few-shot examples as Step 3b but using
Converse API which is model-agnostic (works with DeepSeek, GPT-OSS, Qwen, etc.)

Usage:
    python run_step3b_oss.py --model us.deepseek.r1-v1:0 --output results_deepseek.tsv
    python run_step3b_oss.py --model openai.gpt-oss-120b-1:0 --output results_gptoss120b.tsv
    python run_step3b_oss.py --model openai.gpt-oss-20b-1:0 --output results_gptoss20b.tsv
    python run_step3b_oss.py --model qwen.qwen3-32b-v1:0 --output results_qwen3.tsv
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

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "llm_utils"))

from prompt_helpers import ROLE_ARNS, REGION

# Import prompts from step3
sys.path.insert(0, str(Path(__file__).parent))
import prompts as prompts_module

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_client(role_arn):
    if role_arn is None:
        return boto3.client("bedrock-runtime", region_name=REGION)
    sts = boto3.client("sts", region_name=REGION)
    creds = sts.assume_role(
        RoleArn=role_arn, RoleSessionName="Text2SQL4PM-OSS-3b", DurationSeconds=3600
    )["Credentials"]
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        config=BotoConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 3}),
    )


def extract_text_from_response(response):
    """Extract text from Converse API response, handling different content block types.

    Models may return:
    - {"text": "..."} — standard text block
    - {"reasoningContent": {"reasoningText": {"text": "..."}}} — DeepSeek R1 thinking
    - Multiple blocks in sequence (reasoning first, then text)
    """
    content_blocks = response["output"]["message"]["content"]

    # Collect all text blocks (skip reasoning blocks)
    text_parts = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])

    if text_parts:
        return "\n".join(text_parts)

    # If no text blocks found, try to extract from reasoning as fallback
    for block in content_blocks:
        if "reasoningContent" in block:
            rc = block["reasoningContent"]
            if isinstance(rc, dict) and "reasoningText" in rc:
                rt = rc["reasoningText"]
                if isinstance(rt, dict) and "text" in rt:
                    return rt["text"]
                elif isinstance(rt, str):
                    return rt

    # Last resort: stringify the whole content
    return str(content_blocks)


def generate_sql(client, role_arn, question, model_id):
    """Call model via Converse API with few-shot examples."""
    messages = []
    for ex in prompts_module.FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": [{"text": ex["question"]}]})
        messages.append({"role": "assistant", "content": [{"text": ex["sql"]}]})
    messages.append({"role": "user", "content": [{"text": question}]})

    try:
        response = client.converse(
            modelId=model_id,
            system=[{"text": prompts_module.SYSTEM_PROMPT}],
            messages=messages,
            inferenceConfig={"maxTokens": 4096, "temperature": 0},
        )
        text = extract_text_from_response(response)
        return clean_sql(text), client
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException"):
            client = get_client(role_arn)
            response = client.converse(
                modelId=model_id,
                system=[{"text": prompts_module.SYSTEM_PROMPT}],
                messages=messages,
                inferenceConfig={"maxTokens": 4096, "temperature": 0},
            )
            text = extract_text_from_response(response)
            return clean_sql(text), client
        raise


def clean_sql(response_text):
    sql = response_text.strip()

    # Remove thinking tags (DeepSeek R1)
    if '<think>' in sql:
        think_end = sql.find('</think>')
        if think_end > 0:
            sql = sql[think_end + len('</think>'):].strip()

    # Remove markdown fences
    if "```" in sql:
        parts = sql.split("```")
        for part in parts:
            cleaned = part.replace("sql", "", 1).strip() if part.startswith("sql") else part.strip()
            if cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH"):
                sql = cleaned
                break

    # Extract from <final_sql> tags if present
    match = re.search(r'<final_sql>(.*?)</final_sql>', sql, re.DOTALL)
    if match:
        sql = match.group(1).strip()

    # Find the SQL statement
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


def worker(role_arn, questions_chunk, worker_id, model_id):
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
            sql, client = generate_sql(client, role_arn, question, model_id)
        except Exception as e:
            err_msg = str(e)
            # Log full error for first few to help debug
            if _progress["errors"] < 5:
                logger.warning(f"Worker {worker_id}: ERROR on {uid}: {type(e).__name__}: {err_msg[:200]}")
            else:
                logger.warning(f"Worker {worker_id}: ERROR on {uid}: {type(e).__name__}")
            sql = "SELECT 1"
            with _results_lock:
                _progress["errors"] += 1
            if "ExpiredToken" in err_msg or "expired" in err_msg.lower():
                try:
                    client = get_client(role_arn)
                except Exception:
                    pass

        with _results_lock:
            _results.append((uid, sql))
            _progress["done"] += 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Step 3b: Few-shot on OSS models (Converse API)')
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
