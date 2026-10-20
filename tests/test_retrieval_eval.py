"""Retrieval metrics on a toy gold set, the no-answer threshold, and the eval command."""
import json

import pytest

from rag_ingest.retrieval.evaluation import (Question, QuestionResult, best_threshold, by_type, is_relevant,
                                       load_questions, metrics)
from rag_ingest.retrieval.vectorstore import Hit


def q(qid, qtype="text", role="branch_manager", expected=(("สาขา", 3),)):
    return Question(qid, "คำถาม", expected, role, qtype)


def result(question, rank, score, role="branch_manager"):
    return QuestionResult(question, rank, score, role, 0.01)


def test_relevance_by_document_and_page_range():
    meta = {"document": "สาขา", "page_start": 2, "page_end": 4}
    assert is_relevant(meta, (("สาขา", 3),)) and not is_relevant(meta, (("สาขา", 5),))
    assert not is_relevant(meta, (("เขต", 3),))


def test_recall_and_mrr_on_a_toy_gold_set():
    results = [result(q("a"), 1, 0.9), result(q("b"), 2, 0.8), result(q("c"), 5, 0.7),
               result(q("d"), None, 0.6, role="district_manager")]
    m = metrics(results)
    assert m.recall == {1: 0.25, 3: 0.5, 5: 0.75}
    assert m.mrr == pytest.approx((1 + 1 / 2 + 1 / 5 + 0) / 4)
    assert m.role_accuracy == 0.75


def test_no_answer_questions_excluded_from_recall_and_threshold_chosen():
    answerable = [result(q("a"), 1, 0.90), result(q("b"), 1, 0.85), result(q("c"), 2, 0.70)]
    refusable = [result(q("x", "no_answer", expected=()), None, 0.75),
                 result(q("y", "no_answer", expected=()), None, 0.60)]
    m = metrics(answerable + refusable)
    assert m.recall[1] == pytest.approx(2 / 3)
    t = best_threshold(answerable + refusable)
    # at 0.85: a, b answered (2/3), x, y refused (2/2) -> 0.83; at 0.70: 3/3 and 1/2 -> 0.75
    assert t.value == pytest.approx(0.85) and t.balanced_accuracy == pytest.approx((2 / 3 + 1) / 2)
    assert t.false_answer_rate == 0.0 and t.answerable_refused == pytest.approx(1 / 3)
    assert set(by_type(answerable + refusable)) == {"text", "no_answer"}


def test_threshold_needs_both_kinds():
    assert best_threshold([result(q("a"), 1, 0.9)]).value is None


def test_load_questions(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"id": "q01", "question": "แก้ไขอย่างไร", "expected": [{"document": "สาขา", "page": 7}],
                                "role": "branch_manager", "type": "text"}, ensure_ascii=False) + "\n\n",
                    encoding="utf-8")
    [question] = load_questions(path)
    assert question.expected == (("สาขา", 7),) and question.answerable


def test_eval_command_single_and_grid(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from rag_ingest.cli import main

    monkeypatch.setenv("RAG_INGEST_ROOT", str(tmp_path))
    ingest = tmp_path / "outputs" / "ingest"
    ingest.mkdir(parents=True)
    (ingest / "สาขา.md").write_text("<!-- page 1 -->\n1. แก้ไข\n\nกดปุ่มแก้ไขรายการ\n\n<!-- page 2 -->\n"
                                    "2. ลบ\n\nกดปุ่มลบรายการ\n", encoding="utf-8")
    questions = tmp_path / "data" / "eval" / "questions.jsonl"
    questions.parent.mkdir(parents=True)
    rows = [{"id": "q1", "question": "กดปุ่มลบรายการ", "expected": [{"document": "สาขา", "page": 2}],
             "role": None, "type": "text"},
            {"id": "q2", "question": "สภาพอากาศวันนี้", "expected": [], "role": None, "type": "no_answer"}]
    questions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    main(["chunk", "--embedder", "fake"])
    main(["embed", "--embedder", "fake"])
    main(["eval-retrieval"])
    report = (tmp_path / "outputs" / "reports" / "retrieval_report_index.md").read_text(encoding="utf-8")
    assert "Small evaluation set" in report and "| text |" in report
    main(["eval-retrieval", "--grid", "--models", "fake"])
    csv_rows = (tmp_path / "outputs" / "reports" / "retrieval_report.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 1 + 18  # header + 3 sizes x 3 screenshot table modes x header on/off


def test_report_table_cells_and_no_answer_section():
    from rag_ingest.retrieval.report import ConfigResult, render

    results = [result(q("a"), 1, 0.9), result(q("x", "no_answer", expected=()), None, 0.5)]
    report = render([ConfigResult("m · 400 · keep | x", {}, results, 1.0, 3)], ["สาขา"])
    row = next(line for line in report.splitlines() if line.startswith("| m · 400"))
    assert row.count(" | ") == len(row.split(" | ")) - 1 and "keep \\| x" in row
    assert "| no_answer |" not in report.split("## No-answer")[0]  # not ranked by MRR
    assert "## No-answer questions" in report and "answerable refused" in report
