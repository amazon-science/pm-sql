"""Dataset loading for experiments_v2.

The benchmark is a TSV of 1,655 question/gold-SQL pairs organised into 205
paraphrase groups. The atomic unit for splitting is NOT the group but the
*answer-equivalence class* (see load_answer_classes): the 205 groups contain
only 190 distinct gold answers, so group-level CV leaks by construction.

Facts verified against dataset.tsv (see FEEDBACK_REVIEW_v2.md finding A11):
  - 1,655 questions in 205 groups; the file has trailing all-empty rows.
  - Group sizes range 2..30, NOT "up to 10" as the v1 paper states. 18 groups
    have more than 11 members, so folds must be balanced on question count and
    not on group count.
  - Spider_classification is constant within a group.
  - Domain_value_generic is NOT constant within a group (a paraphrase may swap
    domain vocabulary for a column name), so it is a per-question attribute and
    can only be checked post-hoc, never stratified on.
  - 27 groups in 12 clusters share an identical normalised gold SQL with at
    least one other group, leaving 190 answer-equivalence classes.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "dataset.tsv"
DB_PATH = PROJECT_ROOT / "event_log.sqlite"

SPIDER_CLASSES = ("easy", "medium", "hard", "extra", "no_hardness")
VOCAB_TYPES = ("Value", "Generic", "Domain")


@dataclass(frozen=True)
class Question:
    uid: str
    group_id: str
    question: str
    gold_sql: str
    spider: str
    vocab: str
    is_base: bool
    level: str  # events | cases


@dataclass(frozen=True)
class Group:
    group_id: str
    spider: str
    questions: tuple[Question, ...]

    @property
    def size(self) -> int:
        return len(self.questions)

    @property
    def base(self) -> Question:
        """The base question. Every group in this benchmark has exactly one."""
        for q in self.questions:
            if q.is_base:
                return q
        # Defensive: fall back to the lowest uid so callers never crash.
        return min(self.questions, key=lambda q: int(q.uid))


@dataclass(frozen=True)
class AnswerClass:
    """Paraphrase groups that share a gold answer, hence one indivisible unit.

    Most classes are a single group. The 12 multi-group classes exist because the
    benchmark asks the same question twice under different group ids -- group 15's
    "Which activities appear most frequently in the event log?" and group 91's
    "...most often..." have byte-identical gold SQL. Splitting those two across
    folds would let a training example hand over a held-out answer verbatim.
    """
    class_id: str                     # the lowest member group id
    group_ids: tuple[str, ...]
    spider: str
    questions: tuple[Question, ...]

    @property
    def size(self) -> int:
        return len(self.questions)

    @property
    def merged(self) -> bool:
        return len(self.group_ids) > 1


def normalise_sql(sql: str) -> str:
    """Canonical form for comparing two SQL statements' structure.

    Strips the things a paraphrased rewrite changes without changing meaning:
    case, whitespace, aliases, quote style, trailing semicolons. Literals are
    deliberately kept -- `WHERE resource='peter'` and `='sarah'` have different
    answers and must stay distinct here.
    """
    s = sql.lower().strip().rstrip(";")
    s = re.sub(r'"', "'", s)
    s = re.sub(r"\bas\s+[a-z_][a-z0-9_]*", " ", s)  # column/table aliases
    s = re.sub(r"[\s\n\t]+", " ", s)
    s = re.sub(r"\s*([(),])\s*", r"\1", s)
    return s.strip()


@lru_cache(maxsize=1)
def load_answer_classes(dataset_path: str | None = None) -> tuple[AnswerClass, ...]:
    """Groups merged transitively wherever any two share a normalised gold SQL.

    Merging on *any* question rather than only the base question is deliberate:
    it is the conservative direction. A Value-vocabulary paraphrase occasionally
    carries a gold SQL that another group's base question also uses, and one
    shared answer anywhere is enough to leak.
    """
    groups = load_groups(dataset_path)

    # Union-find over group ids, keyed by normalised gold SQL.
    parent: dict[str, str] = {g.group_id: g.group_id for g in groups}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the numerically lower id as the root so class_id is stable.
            lo, hi = sorted((ra, rb), key=int)
            parent[hi] = lo

    seen: dict[str, str] = {}
    for g in groups:
        for q in g.questions:
            key = normalise_sql(q.gold_sql)
            if not key:
                continue
            if key in seen:
                union(seen[key], g.group_id)
            else:
                seen[key] = g.group_id

    members: dict[str, list[str]] = {}
    for g in groups:
        members.setdefault(find(g.group_id), []).append(g.group_id)

    by_id = {g.group_id: g for g in groups}
    out = []
    for root, gids in members.items():
        gids.sort(key=int)
        qs = tuple(q for gid in gids for q in by_id[gid].questions)
        # Spider class is group-constant but a merged class can straddle two
        # labels; take the lowest member's, and let fold_report show the mix.
        out.append(AnswerClass(class_id=root, group_ids=tuple(gids),
                               spider=by_id[gids[0]].spider, questions=qs))
    out.sort(key=lambda c: int(c.class_id))
    return tuple(out)


def answer_class_of_group(dataset_path: str | None = None) -> dict[str, str]:
    return {gid: c.class_id
            for c in load_answer_classes(dataset_path) for gid in c.group_ids}


@lru_cache(maxsize=1)
def load_questions(dataset_path: str | None = None) -> tuple[Question, ...]:
    path = Path(dataset_path) if dataset_path else DATASET_PATH
    out: list[Question] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            uid = (row.get("Utterance_id") or "").strip()
            question = (row.get("English_utterance") or "").strip()
            gold = (row.get("English_SQL_statement") or "").strip()
            # Trailing rows in the source file are entirely blank.
            if not uid or not question:
                continue
            out.append(
                Question(
                    uid=uid,
                    group_id=(row.get("Group_id") or "").strip(),
                    question=question,
                    gold_sql=gold,
                    spider=(row.get("Spider_classification") or "").strip(),
                    vocab=(row.get("Domain_value_generic") or "").strip(),
                    is_base=(row.get("Base_paraphrase") or "").strip() == "base",
                    level=(row.get("Events_cases_classification") or "").strip(),
                )
            )
    return tuple(out)


@lru_cache(maxsize=1)
def load_groups(dataset_path: str | None = None) -> tuple[Group, ...]:
    by_group: dict[str, list[Question]] = {}
    for q in load_questions(dataset_path):
        by_group.setdefault(q.group_id, []).append(q)

    groups = []
    for gid, qs in by_group.items():
        qs.sort(key=lambda q: int(q.uid))
        # Spider class is group-constant; take it from the base question.
        spider = next((q.spider for q in qs if q.is_base), qs[0].spider)
        groups.append(Group(group_id=gid, spider=spider, questions=tuple(qs)))

    groups.sort(key=lambda g: int(g.group_id))
    return tuple(groups)


def questions_by_uid(dataset_path: str | None = None) -> dict[str, Question]:
    return {q.uid: q for q in load_questions(dataset_path)}


def groups_by_id(dataset_path: str | None = None) -> dict[str, Group]:
    return {g.group_id: g for g in load_groups(dataset_path)}


def uids_for_groups(group_ids: Iterable[str], dataset_path: str | None = None) -> set[str]:
    wanted = set(group_ids)
    return {q.uid for q in load_questions(dataset_path) if q.group_id in wanted}


def base_questions(group_ids: Iterable[str] | None = None,
                   dataset_path: str | None = None) -> list[Question]:
    """Base questions only, optionally restricted to a set of groups.

    Prompt derivation reads only base questions: paraphrases of the same group
    share a gold SQL, so they surface no new failure signatures while costing 8x
    the LLM calls.
    """
    wanted = set(group_ids) if group_ids is not None else None
    return [
        q for q in load_questions(dataset_path)
        if q.is_base and (wanted is None or q.group_id in wanted)
    ]


def summary() -> str:
    qs = load_questions()
    gs = load_groups()
    cs = load_answer_classes()
    sizes = sorted(g.size for g in gs)
    spider = {c: sum(1 for g in gs if g.spider == c) for c in SPIDER_CLASSES}
    vocab = {v: sum(1 for q in qs if q.vocab == v) for v in VOCAB_TYPES}
    merged = [c for c in cs if c.merged]
    return (
        f"{len(qs)} questions in {len(gs)} groups\n"
        f"  group size: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}\n"
        f"  spider (groups): {spider}\n"
        f"  vocab (questions): {vocab}\n"
        f"  answer classes: {len(cs)} "
        f"({len(merged)} merge {sum(len(c.group_ids) for c in merged)} groups)\n"
        + "".join(f"    {list(c.group_ids)}\n" for c in merged)
    )


if __name__ == "__main__":
    print(summary())
