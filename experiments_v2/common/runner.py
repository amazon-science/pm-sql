"""Execute one rung x fold against Bedrock, resumably.

Design decisions that matter for a multi-day unattended run:

*Direct `invoke_model`, not the Strands agent loop.* v1 hit a prefill bug in
Strands' `converse_stream` and worked around it with a hand-rolled loop; this
module keeps that loop. The upside beyond avoiding the bug is that the exact
request body is visible and logged, so "which model version answered this
question" is a recorded fact rather than an inference about SDK behaviour. That
also means no Strands version needs pinning for the ladder to be reproducible.

*Resume by default.* 24k calls over two to three days will be interrupted --
throttling, a dropped connection, a closed laptop lid. Every answer is appended
to `predictions.tsv` as it arrives and completed uids are skipped on restart, so
re-running the same command continues instead of starting over or duplicating.

*A manifest per run.* Prompt SHA-256, model id, example uids, fold, git commit
and token counts are written next to the predictions. This is what makes a rung
auditable after the fact: the leakage gate can be re-run against the exact
prompt that produced a given number.

*Mock mode.* `--mock` exercises the whole path -- prompt assembly, tool loop,
parsing, resume, scoring -- with no network and no cost, which is how the dry
runs validate the pipeline before the real spend.

Usage:
    python runner.py --rung R2 --mock --limit 20
    python runner.py --rung R4 --fold 0 --out runs/R4_f0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import prompt_builder
import tools
from dataset import load_questions, questions_by_uid
from folds import load_fold_config, test_groups

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_REGION = "us-west-2"
DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_CONCURRENCY = 4

MAX_TOKENS = 4096
TEMPERATURE = 0

# Bedrock throttles hard at sustained concurrency. Retry on throttling and on
# transient 5xx, but never on a validation error -- that is a bug in our request
# and retrying it just burns the rate limit.
MAX_RETRIES = 8
BASE_BACKOFF_S = 2.0
MAX_INFRA_STREAK = 20
RETRYABLE = ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailable",
             "InternalServerException", "ModelTimeoutException", "ServiceQuotaExceeded",
             # Transport failures. botocore raises these as exceptions with no
             # response, so they match on the class name.
             "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError",
             "ConnectionClosedError", "ProtocolError")


# --------------------------------------------------------------------------
# SQL extraction
# --------------------------------------------------------------------------

_FINAL_SQL = re.compile(r"<final_sql>(.*?)</final_sql>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the answer SQL out of a model response.

    The fallbacks matter for fairness across rungs. R1 and R2 have no tool loop
    and occasionally wrap their answer in a fence or emit bare SQL; if only
    <final_sql> were honoured, those rungs would be penalised for formatting
    rather than for correctness, and the ladder's deltas would partly measure
    instruction-following. `parse_failed` is returned rather than a dummy query
    so the scorer counts it as wrong and the error breakdown can show how often
    it happens.
    """
    if not text:
        return "parse_failed"

    for pattern in (_FINAL_SQL, _FENCE):
        for m in pattern.finditer(text):
            sql = _clean(m.group(1))
            if sql:
                return sql

    m = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if m:
        sql = _clean(text[m.start():].split("```")[0])
        if sql:
            return sql
    return "parse_failed"


def _clean(sql: str) -> str:
    sql = " ".join(sql.split())
    # Keep only the first statement; a trailing prose sentence after the
    # semicolon would make the query unparseable.
    sql = sql.split(";")[0].strip()
    return sql if re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE) else ""


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Answer:
    uid: str
    sql: str
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    api_calls: int = 0
    retries: int = 0
    error: str | None = None
    # True when the failure was infrastructure (auth, network, retries spent).
    # Such answers are traced but never written to predictions.tsv, so resume
    # retries them instead of scoring a dead connection as a wrong answer.
    infra: bool = False
    raw: str = ""


# --------------------------------------------------------------------------
# model invocation
# --------------------------------------------------------------------------

class MockClient:
    """Deterministic stand-in for Bedrock.

    Answers a fixed handful of question shapes correctly and everything else
    with a plausible-but-wrong query, and exercises the tool path for questions
    whose id is divisible by 3. The point is not realistic accuracy -- it is that
    every branch of the loop (tool_use, end_turn, retry, parse fallback) runs at
    least once in a dry run.
    """

    def __init__(self, tool_probability: int = 3):
        self.tool_probability = tool_probability
        self.calls = 0
        self._lock = threading.Lock()

    def invoke_model(self, modelId: str, body: str):  # noqa: N803 -- boto3 signature
        with self._lock:
            self.calls += 1
        req = json.loads(body)
        msgs = req["messages"]
        question = _first_text(msgs[0]["content"])
        uid_seed = sum(ord(c) for c in question)
        has_tools = bool(req.get("tools"))
        already_used_tool = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in msgs)

        if has_tools and not already_used_tool and uid_seed % self.tool_probability == 0:
            content = [{"type": "tool_use", "id": f"t{uid_seed}",
                        "name": "validate_sql",
                        "input": {"sql": "SELECT COUNT(*) FROM event_log"}}]
            stop = "tool_use"
        else:
            sql = _mock_sql(question, uid_seed)
            # Sometimes answer without the tags, to exercise the parse fallback.
            body_text = (f"<final_sql>{sql}</final_sql>" if uid_seed % 7
                         else f"```sql\n{sql}\n```")
            content = [{"type": "text", "text": body_text}]
            stop = "end_turn"

        return {"body": _MockBody({
            "content": content, "stop_reason": stop,
            "usage": {"input_tokens": 500, "output_tokens": 40},
        })}


class _MockBody:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()


def _mock_sql(question: str, seed: int) -> str:
    q = question.lower()
    if "how many" in q and "activit" in q:
        return "SELECT COUNT(DISTINCT activity) FROM event_log"
    if "how many" in q and "case" in q:
        return "SELECT COUNT(DISTINCT idcase) FROM event_log"
    if "how many" in q:
        return "SELECT COUNT(*) FROM event_log"
    return f"SELECT activity FROM event_log LIMIT {seed % 5 + 1}"


def _first_text(content) -> str:
    if isinstance(content, str):
        return content
    for b in content:
        if b.get("type") == "text":
            return b["text"]
    return ""


def _extract_text(content) -> str:
    return " ".join(b["text"] for b in content if b.get("type") == "text")


# Credentials from `ada credentials update` last ~1h, and the ladder runs for
# days. boto3 reads ~/.aws/credentials once per client, so a refreshed file is
# invisible to a running process. On an auth error the client is rebuilt from
# disk and the call waits -- up to CRED_WAIT_S -- for the refresh loop to land a
# new token, rather than failing every in-flight question.
CRED_ERRORS = ("ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException",
               "InvalidClientTokenId", "InvalidSignatureException",
               "NoCredentialsError", "CredentialRetrievalError")
CRED_WAIT_S = 3600
CRED_POLL_S = 60


class InfraError(RuntimeError):
    """A failure of the plumbing, not of the model. Never recorded as an answer."""


class ClientHolder:
    """A Bedrock client that can be rebuilt in place when credentials rotate.

    Shared by all worker threads. The generation counter makes rebuilds
    idempotent: when eight workers hit an expired token at once, the first
    rebuilds and the other seven see the generation moved and just retry.
    """

    def __init__(self, factory):
        self._factory = factory
        self._lock = threading.Lock()
        self.client = factory()
        self.generation = 0

    def rebuild(self, seen_generation: int) -> None:
        with self._lock:
            if self.generation == seen_generation:
                self.client = self._factory()
                self.generation += 1


def _error_code(e: Exception) -> str:
    return type(e).__name__ + ":" + (
        getattr(e, "response", {}) or {}).get("Error", {}).get("Code", "")


def invoke_with_retry(holder, model_id: str, body: dict) -> tuple[dict, int]:
    """One Bedrock call with backoff and credential recovery. Returns (result, retries).

    Raises InfraError when the call cannot be completed for reasons unrelated to
    the model, so the caller can leave the question unanswered for the next
    resume instead of scoring it wrong.
    """
    if not isinstance(holder, ClientHolder):  # MockClient, or a bare client in tests
        holder = ClientHolder(lambda c=holder: c)
    delay = BASE_BACKOFF_S
    last = None
    attempt = 0
    cred_deadline = None
    while True:
        gen = holder.generation
        try:
            resp = holder.client.invoke_model(modelId=model_id, body=json.dumps(body))
            return json.loads(resp["body"].read()), attempt
        except Exception as e:  # noqa: BLE001
            last = e
            code = _error_code(e)
            attempt += 1

            if any(c in code for c in CRED_ERRORS):
                cred_deadline = cred_deadline or time.time() + CRED_WAIT_S
                if time.time() > cred_deadline:
                    raise InfraError(f"credentials still invalid after "
                                     f"{CRED_WAIT_S // 60} min: {code}") from e
                if attempt == 1 or attempt % 10 == 0:
                    print(f"  credentials rejected ({code}); reloading from disk "
                          f"and waiting for refresh", flush=True)
                time.sleep(CRED_POLL_S)
                holder.rebuild(gen)
                continue

            if not any(r in code for r in RETRYABLE):
                raise
            if attempt >= MAX_RETRIES:
                raise InfraError(f"exhausted {MAX_RETRIES} retries: {last}") from e
            # Jitter, so a fleet of workers throttled at the same instant does
            # not retry in lockstep and throttle again.
            time.sleep(delay + random.uniform(0, delay * 0.3))
            delay = min(delay * 2, 60.0)


def answer_question(client, model_id: str, prompt: prompt_builder.Prompt,
                    uid: str, question: str,
                    max_tool_calls: int) -> Answer:
    """Run one question to a final SQL string, looping over tool calls."""
    started = time.time()
    a = Answer(uid=uid, sql="parse_failed")
    schemas = tools.schemas_for_rung(prompt.rung)
    messages = [{"role": "user",
                 "content": [{"type": "text",
                              "text": prompt_builder.user_message(question)}]}]

    try:
        # +1 iteration so the model always gets a turn to answer after its last
        # tool call rather than being cut off mid-loop.
        for _ in range(max_tool_calls + 1):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "system": prompt.text,
                "messages": messages,
            }
            # Withhold the tools once the budget is spent: offering them and
            # then refusing to run them wastes a turn on a tool_use reply.
            if schemas and a.tool_calls < max_tool_calls:
                body["tools"] = schemas

            result, retries = invoke_with_retry(client, model_id, body)
            a.api_calls += 1
            a.retries += retries
            usage = result.get("usage", {})
            a.input_tokens += usage.get("input_tokens", 0)
            a.output_tokens += usage.get("output_tokens", 0)

            content = result.get("content", [])
            messages.append({"role": "assistant", "content": content})

            if result.get("stop_reason") != "tool_use":
                a.raw = _extract_text(content)
                a.sql = extract_sql(a.raw)
                break

            results = []
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                a.tool_calls += 1
                a.tool_names.append(block["name"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": [{"type": "text",
                                 "text": tools.call(block["name"], block.get("input", {}))}],
                })
            messages.append({"role": "user", "content": results})
        else:
            a.error = "tool_loop_exhausted"
    except InfraError as e:
        a.error = f"InfraError: {e}"
        a.infra = True
    except Exception as e:  # noqa: BLE001
        # Deterministic request errors (e.g. ValidationException). Recorded, so
        # resume terminates, and counted wrong -- but surfaced in the error count.
        a.error = f"{type(e).__name__}: {e}"

    a.latency_s = time.time() - started
    return a


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _git_commit() -> str:
    """Commit the code came from. Read from REVISION when run from an rsync'd
    copy with no .git (the cloud-desktop setup), else asked of git."""
    rev = Path(__file__).resolve().parents[1] / "REVISION"
    if rev.exists():
        return rev.read_text().strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).parent,
                              timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _done_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {l.split("\t")[0].strip() for l in path.read_text().splitlines()
            if l.strip() and "\t" in l}


def run(
    rung: str,
    fold: int | None,
    out_dir: Path,
    model_id: str = DEFAULT_MODEL,
    region: str = DEFAULT_REGION,
    mock: bool = False,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    k: int = 5,
    resume: bool = True,
) -> int:
    from concurrent.futures import ThreadPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.tsv"
    trace_path = out_dir / "trace.jsonl"

    examples, targeted, vocab_rows = _fold_inputs(rung, fold, k,
                                                  runs_root=out_dir.parent)
    prompt = prompt_builder.build(
        rung, fold=fold, examples=examples, targeted_examples=targeted,
        vocab_rows=vocab_rows, max_tool_calls=max_tool_calls)

    # The gate runs before the client is built, so a contaminated prompt cannot
    # reach the API at all. This is the check the whole redesign rests on; making
    # it a manual step would reproduce v1's failure, where the contamination was
    # only found by auditing afterwards.
    gate = _gate(prompt, fold, skip=fold is None)
    if gate is not None:
        print(gate.render())
        if not gate.passed:
            print("\nABORTED: prompt leaks held-out content. "
                  "Fix the prompt or the selection, do not override.")
            return 2
        (out_dir / "leakage_gate.json").write_text(json.dumps({
            "fold": gate.fold, "prompt_sha256": gate.prompt_sha256,
            "n_held_out": gate.n_held_out, "passed": gate.passed,
            "findings": [asdict(f) for f in gate.findings],
        }, indent=2) + "\n")

    questions = _questions_for(rung, fold)
    done = _done_uids(preds_path) if resume else set()
    todo = [q for q in questions if q.uid not in done]
    if limit:
        todo = todo[:limit]

    print(f"rung={rung} fold={fold} model={'MOCK' if mock else model_id}")
    print(f"prompt sha256={prompt.sha256[:16]} blocks={'+'.join(prompt.blocks)}")
    print(f"examples={prompt.example_uids or 'none'}")
    print(f"questions: {len(questions)} total, {len(done)} done, {len(todo)} to run")
    if not todo:
        print("nothing to do")
        return 0

    if mock:
        client = MockClient()
    else:
        import boto3
        from botocore.config import Config

        def factory():
            # A fresh Session each time, so a rebuild re-reads ~/.aws/credentials
            # rather than reusing the default session's cached (expired) keys.
            return boto3.session.Session().client(
                "bedrock-runtime", region_name=region,
                # boto3's own retries would race ours and hide the retry count,
                # so disable them and keep a single, logged backoff policy.
                config=Config(retries={"max_attempts": 1}, read_timeout=300))
        client = ClientHolder(factory)

    lock = threading.Lock()
    counters = {"done": 0, "errors": 0, "in_tok": 0, "out_tok": 0, "tools": 0,
                "infra": 0, "infra_streak": 0}
    # Circuit breaker. A long streak of infrastructure failures means something
    # is broken that waiting will not fix (credentials gone for > CRED_WAIT_S, a
    # revoked model, a dead network). Stop, so the outage costs one resume
    # rather than a fold's worth of pointless attempts.
    abort = threading.Event()
    t0 = time.time()

    with open(preds_path, "a") as pf, open(trace_path, "a") as tf:
        def work(q):
            if abort.is_set():
                return None
            a = answer_question(client, model_id, prompt, q.uid, q.question,
                                max_tool_calls)
            with lock:
                if a.infra:
                    counters["infra"] += 1
                    counters["infra_streak"] += 1
                    if counters["infra_streak"] >= MAX_INFRA_STREAK and not abort.is_set():
                        abort.set()
                        print(f"  ABORT: {MAX_INFRA_STREAK} consecutive infrastructure "
                              f"failures, last: {a.error}", flush=True)
                else:
                    counters["infra_streak"] = 0
                    pf.write(f"{q.uid}\t{a.sql}\n")
                    pf.flush()
                rec = asdict(a)
                # The full response text is large and not needed downstream; a
                # prefix is enough to diagnose a parse failure after the fact.
                rec["raw"] = a.raw[:400]
                tf.write(json.dumps(rec) + "\n")
                tf.flush()
                counters["done"] += 1
                counters["errors"] += bool(a.error) and not a.infra
                counters["in_tok"] += a.input_tokens
                counters["out_tok"] += a.output_tokens
                counters["tools"] += a.tool_calls
                n = counters["done"]
                if n % 25 == 0 or n == len(todo):
                    rate = n / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - n) / rate if rate else 0
                    print(f"  {n}/{len(todo)}  {rate * 60:.0f}/min  "
                          f"eta {eta / 60:.0f}m  errors={counters['errors']}  "
                          f"infra={counters['infra']}", flush=True)
            return a

        with ThreadPoolExecutor(max_workers=1 if mock else concurrency) as pool:
            list(pool.map(work, todo))

    manifest = {
        "rung": rung, "fold": fold, "model_id": "MOCK" if mock else model_id,
        "region": region, "prompt_sha256": prompt.sha256,
        "prompt_blocks": prompt.blocks, "prompt_text": prompt.text,
        "example_uids": prompt.example_uids,
        "vocab_rows": [list(r) for r in (vocab_rows or [])],
        "max_tool_calls": max_tool_calls, "temperature": TEMPERATURE,
        "n_questions": len(questions), "n_run_this_invocation": len(todo),
        "git_commit": _git_commit(),
        "elapsed_s": round(time.time() - t0, 1),
        "tokens": {"input": counters["in_tok"], "output": counters["out_tok"]},
        "tool_calls": counters["tools"], "errors": counters["errors"],
        "infra_failures": counters["infra"], "aborted": abort.is_set(),
        "python": sys.version.split()[0],
        # Scoring and validate_sql both execute SQL; record the engine version.
        "sqlite": __import__("sqlite_compat").sqlite3.sqlite_version,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {preds_path} ({len(_done_uids(preds_path))} predictions)")
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"tokens in={counters['in_tok']:,} out={counters['out_tok']:,}  "
          f"tool calls={counters['tools']}  errors={counters['errors']}")

    # Incomplete is not done. A later rung selects examples from this one's
    # failures, so it must not start on a partial predictions.tsv. Exit 3 tells
    # run_ladder.sh to wait and resume this step rather than move on.
    # Under --limit only the attempted slice is expected to be complete.
    expected = {q.uid for q in (todo if limit else questions)}
    missing = len(expected - _done_uids(preds_path))
    if missing:
        print(f"INCOMPLETE: {missing} questions unanswered "
              f"({counters['infra']} infrastructure failures"
              + (", aborted" if abort.is_set() else "") + "); re-run to resume")
        return 3
    return 0


def _gate(prompt: prompt_builder.Prompt, fold: int | None, skip: bool):
    """Run the leakage gate on an assembled prompt.

    Skipped only when there is no fold: R1 and R2 contain no benchmark-derived
    content at all, so there is no held-out split for them to leak. Any rung with
    a fold is gated unconditionally.
    """
    if skip or fold is None:
        return None
    import leakage_gate

    return leakage_gate.check_prompt(prompt.text, fold,
                                     example_uids=prompt.example_uids)


def _questions_for(rung: str, fold: int | None):
    """Which questions this run answers.

    Fold-dependent rungs are evaluated on their held-out fold only; rungs whose
    prompt has no fold-derived content are run once over all 1,655 questions,
    since there is nothing for a fold to hold out.
    """
    if fold is None:
        return list(load_questions())
    held = test_groups(fold)
    return [q for q in load_questions() if q.group_id in held]


def _fold_inputs(rung: str, fold: int | None, k: int,
                 runs_root: Path | None = None):
    """Assemble the fold-derived prompt content for a rung."""
    examples = targeted = None
    vocab_rows = None

    if rung in prompt_builder.RUNGS_WITH_EXAMPLES:
        if fold is None:
            raise SystemExit(f"{rung} is fold-specific; pass --fold")
        import example_selector

        if rung == "R3c":
            examples = example_selector.select_random(fold).examples
        else:
            wrong = _prior_rung_failures(fold, runs_root=runs_root)
            examples = example_selector.select_failure_driven(fold, wrong).examples

    if rung in prompt_builder.RUNGS_WITH_VOCAB:
        import vocab_miner
        vocab_rows = vocab_miner.mine_fold(fold).prompt_rows()

    if rung == "R6":
        import example_selector
        wrong = _prior_rung_failures(fold, rung="R5", runs_root=runs_root)
        base = {e.uid for e in (examples or [])}
        targeted = example_selector.select_failure_driven(
            fold, wrong, exclude_uids=base).examples

    return examples, targeted, vocab_rows


def _prior_rung_failures(fold: int, rung: str = "R2",
                         runs_root: Path | None = None) -> set[str]:
    """Uids the prior rung got wrong on this fold's *training* split.

    Handles both shapes of prior rung. R2 is fold-independent and writes one
    `R2/predictions.tsv`; R5 is fold-specific and writes `R5_f0..R5_f4`, so its
    predictions must be pooled across folds. Pooling is the right signal and not a
    convenience: R6 fold f needs failures among fold f's *training* groups, and
    those questions were held out -- hence answered -- in the other four folds.

    runs_root defaults to experiments_v2/runs but follows --out, so a mock ladder
    writing to runs_mock/ chains off its own rungs rather than crossing over into
    the real run's predictions.

    Falls back to v1's step-2 predictions so dry runs work before any v2 rung has
    been executed -- the fallback is announced, because silently substituting a
    different failure signal would change which examples a real run selects.
    """
    import example_selector

    root = runs_root or (Path(__file__).resolve().parents[1] / "runs")
    single = root / rung / "predictions.tsv"
    if single.exists():
        return example_selector.wrong_uids(single)

    per_fold = sorted(p for p in root.glob(f"{rung}_f*/predictions.tsv"))
    if per_fold:
        from eval import correctness, load_predictions

        pooled: dict[str, str] = {}
        for p in per_fold:
            for uid, sql in load_predictions(p).items():
                # A uid appearing in two folds means the fold partition or a run
                # directory is wrong; that would corrupt the failure signal.
                if uid in pooled:
                    raise SystemExit(
                        f"uid {uid} predicted in more than one {rung} fold run; "
                        f"check {root} for stale or mislabelled run directories")
                pooled[uid] = sql
        print(f"  {rung} failure signal pooled from {len(per_fold)} fold runs "
              f"({len(pooled)} predictions)")
        return {uid for uid, ok in correctness(pooled).items() if not ok}

    print(f"  NOTE no {rung} predictions under {root}; "
          "using v1 step-2 predictions as the failure signal (dry-run only)")
    return set(example_selector.wrong_uids_from_v1())


# --------------------------------------------------------------------------
# gate sweep
# --------------------------------------------------------------------------

def gate_sweep(k: int = 5, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS) -> int:
    """Gate every (fold-dependent rung, fold) prompt before spending any money.

    Built through _fold_inputs and prompt_builder.build, the same path a real run
    takes. Running this is what revealed that the 205 paraphrase groups hold only
    190 distinct answers; see folds.py.

    What it does and does not prove. Before the ladder runs, R2's and R5's
    predictions do not exist, so R3-R6 select examples from the announced v1
    fallback signal rather than from the failures a real run will see. The sweep
    therefore validates the *mechanism* -- selection never reaches into the
    held-out split, assembly leaks nothing, no two rungs produce the same prompt
    -- but not the literal example set R3+ will use. That is why the gate is also
    a hard precondition inside run(): every real prompt is gated when it is built,
    and a failing one aborts before any question is answered.
    """
    rungs = [r for r in prompt_builder.RUNGS if r in prompt_builder.RUNGS_WITH_EXAMPLES]
    n_fail = 0
    n_warn = 0
    seen_sha: dict[str, str] = {}
    for rung in rungs:
        for fold in range(k):
            examples, targeted, vocab_rows = _fold_inputs(rung, fold, k)
            prompt = prompt_builder.build(
                rung, fold=fold, examples=examples, targeted_examples=targeted,
                vocab_rows=vocab_rows, max_tool_calls=max_tool_calls)
            rep = _gate(prompt, fold, skip=False)
            tag = f"{rung} f{fold}"
            n_warn += len(rep.warnings)
            if rep.passed:
                print(f"  PASS {tag:<9} {len(rep.warnings)} warn  "
                      f"sha={rep.prompt_sha256[:12]}")
            else:
                n_fail += 1
                print(f"  FAIL {tag:<9} {rep.render()}")
            # Two rungs producing an identical prompt would mean a rung is not
            # actually adding what it claims to add.
            prev = seen_sha.get(rep.prompt_sha256)
            if prev:
                n_fail += 1
                print(f"  FAIL {tag}: prompt is byte-identical to {prev}")
            seen_sha[rep.prompt_sha256] = tag

    print(f"\ngate sweep: {len(rungs) * k} prompts, {n_fail} FAIL, {n_warn} warn")
    print("Note: R3-R6 examples here come from whatever prior-rung signal exists "
          "now.\nThe real run gates its own prompts again at build time.")
    if n_fail:
        print("Do not run the ladder until every prompt passes.")
    return n_fail


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest() -> int:
    """Check SQL extraction, since every rung's score depends on it."""
    cases = [
        ("<final_sql>SELECT COUNT(*) FROM event_log</final_sql>",
         "SELECT COUNT(*) FROM event_log"),
        ("blah <final_sql>\n  SELECT a\n  FROM t\n</final_sql> more", "SELECT a FROM t"),
        ("```sql\nSELECT 1 FROM event_log\n```", "SELECT 1 FROM event_log"),
        ("```\nWITH x AS (SELECT 1) SELECT * FROM x\n```",
         "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("Here you go: SELECT activity FROM event_log", "SELECT activity FROM event_log"),
        ("SELECT a FROM t; -- and then some prose", "SELECT a FROM t"),
        ("I cannot answer that.", "parse_failed"),
        ("", "parse_failed"),
        # A tag holding prose rather than SQL must not be accepted as SQL.
        ("<final_sql>no query needed</final_sql>", "parse_failed"),
        # Tag wins over a fenced block earlier in the text.
        ("```sql\nSELECT wrong\n```\n<final_sql>SELECT right FROM t</final_sql>",
         "SELECT right FROM t"),
    ]
    failures = []
    for text, want in cases:
        got = extract_sql(text)
        if got != want:
            failures.append(f"extract_sql({text[:40]!r}) -> {got!r}, want {want!r}")

    for f in failures:
        print("FAIL", f)
    print(f"runner selftest: {'OK' if not failures else f'{len(failures)} FAILURES'}")
    return len(failures)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one rung x fold")
    ap.add_argument("--rung", required=False)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default=os.environ.get("PMSQL_MODEL", DEFAULT_MODEL))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    ap.add_argument("--mock", action="store_true", help="no network, no cost")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate-sweep", action="store_true",
                    help="gate all fold-specific prompts; run before the ladder")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(1 if selftest() else 0)
    if args.gate_sweep:
        raise SystemExit(1 if gate_sweep(max_tool_calls=args.max_tool_calls) else 0)
    if not args.rung:
        ap.print_help()
        return

    cfg = load_fold_config()
    out = args.out or (Path(__file__).resolve().parents[1] / "runs" /
                       (args.rung if args.fold is None else f"{args.rung}_f{args.fold}"))
    raise SystemExit(run(
        rung=args.rung, fold=args.fold, out_dir=out, model_id=args.model,
        region=args.region, mock=args.mock, limit=args.limit,
        concurrency=args.concurrency, max_tool_calls=args.max_tool_calls,
        k=cfg["k"], resume=not args.no_resume))


if __name__ == "__main__":
    main()
