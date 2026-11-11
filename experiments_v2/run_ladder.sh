#!/usr/bin/env bash
#
# Run the full v2 experiment ladder. Designed for an unattended cloud desktop:
# every step is resumable, so if this dies at rung R5 fold 3 you re-run the same
# command and it picks up from the last written prediction.
#
#   ./run_ladder.sh            # the real thing, ~24k Bedrock calls
#   ./run_ladder.sh --mock     # no network, no cost, exercises every code path
#   ./run_ladder.sh --preflight-only
#
# Order matters and is not cosmetic. R3 selects its examples from the questions
# R2 got wrong, and R6 from the ones R5 got wrong, so a rung cannot start before
# its predecessor's predictions.tsv exists. That is why the folds run inside the
# rung loop rather than the other way round.

set -euo pipefail

cd "$(dirname "$0")"
COMMON=common
# Override to keep replicates apart, e.g. RUNS=runs_rep2 ./run_ladder.sh
RUNS="${RUNS:-runs}"
K=5

MOCK=""
PREFLIGHT_ONLY=0
for arg in "$@"; do
  case "$arg" in
    # Mock output goes to runs_mock/, never runs/. Sharing the directory would
    # let a real run resume from mock predictions and silently report them.
    --mock) MOCK="--mock"; RUNS=runs_mock ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Needs 3.12+ (random.binomialvariate in the bootstrap). Dev desktops ship 3.7
# as python3, so pick the newest available unless PYTHON is set explicitly.
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
  || { echo "need Python >= 3.12, got $("$PYTHON" --version)" >&2; exit 1; }

export PYTHONPATH="$PWD/$COMMON:${PYTHONPATH:-}"
log() { printf '\n=== %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Preflight. Each of these has cost us a run before; none is decoration.
# ---------------------------------------------------------------------------
log "preflight"

for f in ../dataset.tsv ../event_log.sqlite config/folds.json config/answer_units.json; do
  [ -e "$f" ] || { echo "MISSING $f" >&2; exit 1; }
done

# Fails unless SQLite supports window functions; see common/sqlite_compat.py.
"$PYTHON" -c 'from sqlite_compat import sqlite3; print("sqlite", sqlite3.sqlite_version)'
"$PYTHON" "$COMMON/dataset.py"           > /dev/null
"$PYTHON" "$COMMON/tools.py" --selftest
"$PYTHON" "$COMMON/runner.py" --selftest
"$PYTHON" "$COMMON/leakage_gate.py" --selftest

# Reproduces the benchmark's published 47.6% from its own predictions. If this
# drifts, the scorer changed and no rung's number is comparable to the paper's.
"$PYTHON" "$COMMON/eval.py" --verify-scorer

# Gates all 25 fold-specific prompts before spending anything. This is the check
# that caught the answer-leak in group-level splitting.
"$PYTHON" "$COMMON/runner.py" --gate-sweep

if [ "$PREFLIGHT_ONLY" = 1 ]; then
  log "preflight only: stopping before any model calls"
  exit 0
fi

# ---------------------------------------------------------------------------
# Credentials. ada conduit tokens last ~1h and the ladder runs for days, so a
# background loop refreshes them every CRED_REFRESH_EVERY_S. runner.py reloads
# the file on an auth error, so a refresh reaches running workers without a
# restart. The refresh itself needs a live Midway session (~20h); when that
# lapses the loop logs it loudly, the runner pauses, and the step loop below
# resumes on its own once `mwinit` has been run on the desktop.
# ---------------------------------------------------------------------------
CRED_REFRESH_EVERY_S="${CRED_REFRESH_EVERY_S:-1500}"    # 25 min; conduit tokens last ~1h
STEP_RETRY_WAIT_S="${STEP_RETRY_WAIT_S:-600}"
STEP_MAX_ATTEMPTS="${STEP_MAX_ATTEMPTS:-200}"            # ~33h of waiting per step

refresh_creds() {
  [ -n "${CRED_REFRESH_CMD:-}" ] || return 0
  if sh -c "$CRED_REFRESH_CMD" > /dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] credentials refreshed"
  else
    echo "[$(date -u +%FT%TZ)] CREDENTIAL REFRESH FAILED -- run mwinit on this host;" \
         "the ladder will resume by itself afterwards" >&2
  fi
}

if [ -z "$MOCK" ]; then
  if [ -z "${CRED_REFRESH_CMD:-}" ]; then
    echo "WARNING: CRED_REFRESH_CMD unset; credentials will expire mid-run" >&2
  fi
  refresh_creds
  aws sts get-caller-identity > /dev/null \
    || { echo "no usable AWS credentials" >&2; exit 1; }
  if [ -n "${CRED_REFRESH_CMD:-}" ]; then
    ( while sleep "$CRED_REFRESH_EVERY_S"; do refresh_creds; done ) &
    REFRESH_PID=$!
    trap 'kill "$REFRESH_PID" 2>/dev/null' EXIT
  fi
fi

# Run one rung x fold to completion. runner.py exits 3 when questions are left
# unanswered for infrastructure reasons (expired credentials, network); those
# are retried after a wait rather than letting the next rung start on a partial
# predictions.tsv, which would corrupt its example selection.
run_step() {
  local attempt=1 rc
  while :; do
    set +e
    "$PYTHON" "$COMMON/runner.py" "$@"
    rc=$?
    set -e
    [ "$rc" = 0 ] && return 0
    if [ "$rc" != 3 ] || [ "$attempt" -ge "$STEP_MAX_ATTEMPTS" ]; then
      echo "step failed (exit $rc, attempt $attempt): runner.py $*" >&2
      exit "$rc"
    fi
    echo "[$(date -u +%FT%TZ)] step incomplete; retrying in ${STEP_RETRY_WAIT_S}s" \
         "(attempt $attempt/$STEP_MAX_ATTEMPTS)"
    sleep "$STEP_RETRY_WAIT_S"
    refresh_creds
    attempt=$((attempt + 1))
  done
}

# ---------------------------------------------------------------------------
# R0 -- the published GPT-3.5 predictions, scored by our scorer.
# Copied in rather than special-cased in report.py so every rung in the table is
# read from an identical predictions.tsv and scored by identical code.
# ---------------------------------------------------------------------------
log "R0 (published GPT-3.5 SQL, no model calls)"
mkdir -p "$RUNS/R0"
cp ../experiments/step1_baseline_gpt35/results_step1.tsv "$RUNS/R0/predictions.tsv"

# ---------------------------------------------------------------------------
# Fold-independent rungs: one run each over all 1,655 questions.
# ---------------------------------------------------------------------------
for rung in R1 R2; do
  log "$rung (all 1655 questions)"
  run_step --rung "$rung" $MOCK --out "$RUNS/$rung"
done

# ---------------------------------------------------------------------------
# Fold-specific rungs: five runs each, every question held out exactly once.
# R3c is the control for R3 and must run even though it adds no capability.
# ---------------------------------------------------------------------------
for rung in R3 R3c R4 R5 R6; do
  for fold in $(seq 0 $((K - 1))); do
    log "$rung fold $fold"
    run_step --rung "$rung" --fold "$fold" $MOCK --out "$RUNS/${rung}_f${fold}"
  done
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
log "report"
"$PYTHON" "$COMMON/report.py" --runs "$RUNS" \
  --json "$RUNS/ladder.json" --markdown "$RUNS/ladder.md"

log "done -- see $RUNS/ladder.md"
