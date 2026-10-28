"""A/B comparison of two vision result folders."""
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff

from rag_ingest.ocr_eval.metrics import cer
from rag_ingest.stores import VisionStore


def hint_label(record: dict) -> str:
    """Which OCR hints actually reached the prompt: E, T, E+T or '-' (P = PaddleOCR, old runs)."""
    settings = record.get("settings", {})
    used = [tag for tag, key in (("P", "paddle_hint"), ("E", "easyocr_hint"), ("T", "tesseract_hint"))
            if settings.get(key)]
    return "+".join(used) or "-"


@dataclass
class Row:
    stem: str
    similarity: float
    text_a: str
    text_b: str
    cer_a: float | None
    cer_b: float | None
    hint_a: str
    hint_b: str


def compare(a: VisionStore, b: VisionStore) -> tuple[list[Row], list[str]]:
    """(rows for images in both folders, images found in only one)."""
    records_a = {s: a.load(s) for s in a.stems()}
    records_b = {s: b.load(s) for s in b.stems()}
    rows = []
    for stem in sorted(records_a.keys() & records_b.keys()):
        ra, rb = records_a[stem], records_b[stem]
        ta, tb = ra["text"], rb["text"]
        ref = rb.get("corrected_text") or ra.get("corrected_text")
        rows.append(Row(
            stem, SequenceMatcher(None, ta, tb).ratio(), ta, tb,
            cer(ta, ref) if ref is not None else None,
            cer(tb, ref) if ref is not None else None,
            hint_label(ra), hint_label(rb),
        ))
    only = sorted({f"{s}.json" for s in records_a.keys() ^ records_b.keys()})
    return rows, only


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _table(rows: list[Row], label_a: str, label_b: str) -> list[str]:
    lines = [f"# Vision OCR comparison\n\nA = `{label_a}`  |  B = `{label_b}`\n",
             "| image | A~B similarity | len A | len B | CER A | CER B | hint A | hint B |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.stem} | {r.similarity:.2f} | {len(r.text_a)} | {len(r.text_b)} "
              f"| {_fmt(r.cer_a)} | {_fmt(r.cer_b)} | {r.hint_a} | {r.hint_b} |" for r in rows]
    lines.append("\nHint columns: E = EasyOCR, T = Tesseract (P = PaddleOCR, old runs) reached the prompt; - = none.")
    return lines


def _summary(rows: list[Row], only: list[str]) -> list[str]:
    identical = sum(r.text_a == r.text_b for r in rows)
    lines = [f"\n{len(rows)} images compared, {identical} identical, "
             f"mean similarity {sum(r.similarity for r in rows) / len(rows):.2f}."]
    scored = [r for r in rows if r.cer_a is not None]
    if scored:
        lines.append(f"Ground truth available for {len(scored)} images: "
                     f"mean CER A {sum(r.cer_a for r in scored) / len(scored):.3f} vs "
                     f"B {sum(r.cer_b for r in scored) / len(scored):.3f} (lower is better).")
    else:
        lines.append("No corrected_text yet, so no accuracy score. Review some "
                     "results with `rag-ingest review`, then rerun this.")
    if only:
        lines.append(f"Only in one folder (skipped): {', '.join(only)}")
    return lines


def _diffs(rows: list[Row]) -> list[str]:
    lines = []
    for r in (r for r in rows if r.text_a != r.text_b):
        lines.append(f"\n## {r.stem}\n```diff")
        lines.extend(unified_diff(r.text_a.splitlines(), r.text_b.splitlines(),
                                  "A", "B", lineterm="", n=0))
        lines.append("```")
    return lines


def render(rows: list[Row], only: list[str], label_a: str, label_b: str, diff: bool) -> str:
    if not rows:
        raise ValueError("No images in common between the two folders.")
    lines = _table(rows, label_a, label_b) + _summary(rows, only)
    return "\n".join(lines + (_diffs(rows) if diff else []))
