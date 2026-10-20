"""outputs/reports/retrieval_report.md + .csv, in the style of benchmark_report.md."""
from dataclasses import dataclass
from pathlib import Path
import csv

from rag_ingest.config import ENCODING
from rag_ingest.retrieval.evaluation import (K_VALUES, NO_ANSWER, Metrics, QuestionResult, Threshold,
                                       best_threshold, by_type, metrics)


@dataclass
class ConfigResult:
    name: str          # e.g. "e5-small | 400 | keep | header"
    settings: dict
    results: list[QuestionResult]
    embed_seconds: float
    chunks: int

    @property
    def overall(self) -> Metrics:
        return metrics(self.results)

    @property
    def threshold(self) -> Threshold:
        return best_threshold(self.results)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _row(c: ConfigResult) -> list[str]:
    m, t = c.overall, c.threshold
    return ([cell(c.name), str(c.chunks)] + [_pct(m.recall.get(k)) for k in K_VALUES]
            + [_pct(m.mrr), _pct(m.role_accuracy), _pct(t.value), _pct(t.false_answer_rate),
               _pct(t.answerable_refused), f"{m.latency * 1000:.0f}", f"{c.embed_seconds:.1f}"])


def cell(text: str) -> str:
    """Safe inside a Markdown table cell."""
    return text.replace("|", "\\|")


HEADER = (["configuration", "chunks"] + [f"R@{k}" for k in K_VALUES]
          + ["MRR", "role acc", "threshold", "false answer", "answerable refused", "ms/query", "embed s"])


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def _rank_key(c: ConfigResult):
    m = c.overall
    return (m.mrr or 0.0, m.recall.get(1, 0.0), m.role_accuracy or 0.0)


def best_per_type(configs: list[ConfigResult]) -> list[list[str]]:
    """Answerable types: best MRR. No-answer questions are judged by the threshold instead
    (see best_refusal), since they have no relevant passage to rank."""
    types = sorted({r.question.type for c in configs for r in c.results} - {NO_ANSWER})
    rows = []
    for t in types:
        scored = [(c, by_type(c.results)[t]) for c in configs]
        best_c, best_m = max(scored, key=lambda x: (x[1].mrr or 0.0, x[1].recall.get(1, 0.0)))
        rows.append([t, str(best_m.questions), cell(best_c.name)]
                    + [_pct(best_m.recall.get(k)) for k in K_VALUES] + [_pct(best_m.mrr)])
    return rows


def best_refusal(configs: list[ConfigResult]) -> list[str] | None:
    """The configuration whose threshold best separates answerable from no-answer questions."""
    usable = [c for c in configs if c.threshold.balanced_accuracy is not None]
    if not usable:
        return None
    c = max(usable, key=lambda x: (x.threshold.balanced_accuracy, -(x.threshold.false_answer_rate or 0)))
    t = c.threshold
    return [cell(c.name), _pct(t.value), _pct(t.balanced_accuracy), _pct(t.false_answer_rate),
            _pct(t.answerable_refused)]


def per_question(c: ConfigResult) -> list[list[str]]:
    return [[r.question.id, r.question.type, str(r.first_relevant or "-"), f"{r.top_score:.3f}",
             r.top_role or "-"] for r in c.results]


def caveat(n_questions: int, documents: list[str]) -> str:
    one = 1 / n_questions if n_questions else 0
    return (f"> **Small evaluation set:** {n_questions} questions over {len(documents)} document(s) "
            f"({', '.join(documents)}). One question is {one:.2f} of Recall/MRR. Differences of one or two "
            "questions are noise: do not read them as a winner. The set was written by hand for these "
            "manuals, so it measures this corpus, not Thai retrieval in general.")


def render(configs: list[ConfigResult], documents: list[str]) -> str:
    ranked = sorted(configs, key=_rank_key, reverse=True)
    n = len(configs[0].results) if configs else 0
    lines = ["# Retrieval evaluation\n", caveat(n, documents), "",
             "Recall@k and MRR count answerable questions; *threshold* is the top score below which the "
             "answer is \"not found\" (best balanced accuracy on answerable vs no-answer questions); "
             "*false answer* is the share of no-answer questions still answered at that threshold.\n",
             "## All configurations (best first)\n", *_table([_row(c) for c in ranked], HEADER), "",
             "## Best configuration per question type\n",
             *_table(best_per_type(configs), ["type", "questions", "configuration"]
                     + [f"R@{k}" for k in K_VALUES] + ["MRR"]), ""]
    refusal = best_refusal(configs)
    if refusal:
        lines += ["## No-answer questions: best \"not found\" threshold\n",
                  *_table([refusal], ["configuration", "threshold", "balanced accuracy", "false answer",
                                      "answerable refused"]), ""]
    if ranked:
        lines += [f"## Per question: {cell(ranked[0].name)}\n",
                  *_table(per_question(ranked[0]), ["id", "type", "first relevant rank", "top score", "top role"])]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, configs: list[ConfigResult]) -> None:
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER + ["settings"])
        for c in configs:
            writer.writerow(_row(c) + [repr(c.settings)])
