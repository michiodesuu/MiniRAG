"""A/B comparison of two vision result folders."""
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff

from rag_ingest.metrics import cer
from rag_ingest.stores import VisionStore


def hint_label(record: dict) -> str:
    """Which OCR hints actually reached the prompt: P, T, P+T or '-'."""
    settings = record.get("settings", {})
    used = [tag for tag, key in (("P", "paddle_hint"), ("T", "tesseract_hint"))
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


def render(rows: list[Row], only: list[str], label_a: str, label_b: str, diff: bool) -> str:
    if not rows:
        raise ValueError("No images in common between the two folders.")
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    lines = [f"# Vision OCR comparison\n\nA = `{label_a}`  |  B = `{label_b}`\n",
            "| image | A~B similarity | len A | len B | CER A | CER B | hint A | hint B |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.stem} | {r.similarity:.2f} | {len(r.text_a)} | {len(r.text_b)} "
                    f"| {fmt(r.cer_a)} | {fmt(r.cer_b)} | {r.hint_a} | {r.hint_b} |")

    lines.append("\nHint columns: P = PaddleOCR, T = Tesseract reached the prompt; - = none.")
    identical = sum(r.text_a == r.text_b for r in rows)
    lines.append(f"\n{len(rows)} images compared, {identical} identical, "
                f"mean similarity {sum(r.similarity for r in rows) / len(rows):.2f}.")
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

    if diff:
        for r in rows:
            if r.text_a != r.text_b:
                lines.append(f"\n## {r.stem}\n```diff")
                lines.extend(unified_diff(r.text_a.splitlines(), r.text_b.splitlines(),
                                        "A", "B", lineterm="", n=0))
                lines.append("```")
    return "\n".join(lines)
