"""Retrieval evaluation: does search find the right passage? (spec 6)

Gold set: data/eval/questions.jsonl, one question per line, expected answers
as (document, page), so the set survives changes to the chunk settings. A
result is relevant when it comes from that document and its page range
covers that page.

Metrics (answerable questions unless stated):
  Recall@k   share of questions with a relevant result in the top k
  MRR        mean of 1 / rank of the first relevant result (0 if none in top k)
  role acc.  share whose top result comes from the question's role
  threshold  ("not found"): a question is answered when the top score is at
             least t. t is chosen to maximise balanced accuracy: the mean of
             (answerable questions answered) and (no-answer questions
             refused). Reported with the false-answer rate at t: the share of
             no-answer questions that would still get an answer.
  latency    mean seconds per query on this machine (embed query + search)
"""
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable
import json
import time

from rag_ingest.config import ENCODING

NO_ANSWER = "no_answer"
K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected: tuple[tuple[str, int], ...]  # (document, page)
    role: str | None
    type: str
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != NO_ANSWER


def load_questions(path: Path) -> list[Question]:
    out = []
    for line in path.read_text(encoding=ENCODING).splitlines():
        if line.strip():
            q = json.loads(line)
            expected = tuple((e["document"], int(e["page"])) for e in q.get("expected", []))
            out.append(Question(q["id"], q["question"], expected, q.get("role"), q["type"], q.get("notes", "")))
    return out


def is_relevant(meta: dict, expected: tuple[tuple[str, int], ...]) -> bool:
    return any(meta.get("document") == doc and meta.get("page_start", 0) <= page <= meta.get("page_end", -1)
               for doc, page in expected)


@dataclass
class QuestionResult:
    question: Question
    first_relevant: int | None  # 1-based rank in the top k, None if absent
    top_score: float
    top_role: str | None
    seconds: float


def evaluate(questions: list[Question], embed_query: Callable, search: Callable,
             k: int = max(K_VALUES)) -> list[QuestionResult]:
    results = []
    for q in questions:
        started = time.perf_counter()
        hits = search(embed_query(q.question), k)
        elapsed = time.perf_counter() - started
        rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h.meta, q.expected)), None)
        top = hits[0] if hits else None
        results.append(QuestionResult(q, rank, top.score if top else 0.0,
                                      top.meta.get("role") if top else None, elapsed))
    return results


@dataclass
class Threshold:
    value: float | None
    balanced_accuracy: float | None
    false_answer_rate: float | None   # no-answer questions still answered at t
    answerable_refused: float | None  # answerable questions refused at t


def best_threshold(results: list[QuestionResult]) -> Threshold:
    answerable = [r.top_score for r in results if r.question.answerable]
    refusable = [r.top_score for r in results if not r.question.answerable]
    if not answerable or not refusable:
        return Threshold(None, None, None, None)
    best = None
    for t in sorted(set(answerable + refusable)):
        answered = sum(s >= t for s in answerable) / len(answerable)
        refused = sum(s < t for s in refusable) / len(refusable)
        score = (answered + refused) / 2
        if best is None or score > best.balanced_accuracy:
            best = Threshold(t, score, 1 - refused, 1 - answered)
    return best


@dataclass
class Metrics:
    questions: int
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float | None = None
    role_accuracy: float | None = None
    latency: float = 0.0


def metrics(results: list[QuestionResult]) -> Metrics:
    answerable = [r for r in results if r.question.answerable]
    m = Metrics(len(results), latency=mean(r.seconds for r in results) if results else 0.0)
    if answerable:
        m.recall = {k: mean(r.first_relevant is not None and r.first_relevant <= k for r in answerable)
                    for k in K_VALUES}
        m.mrr = mean(1 / r.first_relevant if r.first_relevant else 0.0 for r in answerable)
    with_role = [r for r in answerable if r.question.role]
    if with_role:
        m.role_accuracy = mean(r.top_role == r.question.role for r in with_role)
    return m


def by_type(results: list[QuestionResult]) -> dict[str, Metrics]:
    types = sorted({r.question.type for r in results})
    return {t: metrics([r for r in results if r.question.type == t]) for t in types}
