#!/usr/bin/env python3
"""Score predictions with the benchmark authors' evaluator.

Executes each prediction and its gold query against the event log and compares
results with the authors' unmodified `result_eq` (test-suite evaluator):
`keep_distinct`, column order ignored, row order compared only when the gold
query has `ORDER BY`. A prediction that errors or times out scores 0. This is
the scoring the paper reports.

    python3 score.py OUT_DIR  step5c=../results/predictions/step5c.rep1.tsv  ...

writes OUT_DIR/<tag>.json = {question_id: 0|1}. Needs the event log in place
(run ../setup_data.sh) and the authors' evaluator in ./testsuite.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "testsuite"))
sys.path.insert(0, str(HERE.parent / "pmsql"))

from dataset import DB_PATH, load_questions  # noqa: E402  (pmsql)
from eval import load_predictions            # noqa: E402  (pmsql)

DB = str(DB_PATH)
TIMEOUT_S = 30.0
_gold_cache: dict[str, object] = {}


def load_gold() -> list[tuple[str, str]]:
    """(question_id, gold_sql) for every question."""
    return [(q.uid, q.gold_sql) for q in load_questions()]


def _exec(sql: str):
    import sqlite3
    from exec_eval import replace_cur_year
    con = sqlite3.connect(DB)
    con.text_factory = lambda b: b.decode(errors="ignore")
    start = time.time()
    con.set_progress_handler(lambda: 1 if time.time() - start > TIMEOUT_S else 0, 1000)
    try:
        return con.execute(replace_cur_year(sql)).fetchall()
    except Exception:  # noqa: BLE001 -- any error, including timeout, scores 0
        return None
    finally:
        con.close()


def _one(item):
    from exec_eval import postprocess, result_eq
    key, pred, gold = item
    p, g = postprocess(pred), postprocess(gold)
    if g not in _gold_cache:
        _gold_cache[g] = _exec(g)
    g_rows, p_rows = _gold_cache[g], _exec(p)
    if g_rows is None or p_rows is None:
        return key, 0
    try:
        ok = result_eq(g_rows, p_rows, order_matters="order by" in g.lower())
    except Exception:  # noqa: BLE001
        ok = False
    return key, int(bool(ok))


def pairs_for_file(path: Path):
    preds = load_predictions(path)
    return [(uid, preds[uid], gold) for uid, gold in load_gold() if uid in preds]


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:
        tag, path = spec.split("=", 1)
        pairs = pairs_for_file(Path(path))
        with ProcessPoolExecutor() as pool:
            res = dict(pool.map(_one, pairs, chunksize=8))
        (out_dir / f"{tag}.json").write_text(json.dumps(res) + "\n")
        acc = 100.0 * sum(res.values()) / len(res)
        print(f"{tag}: {sum(res.values())}/{len(res)} = {acc:.1f}%", flush=True)


if __name__ == "__main__":
    main()
