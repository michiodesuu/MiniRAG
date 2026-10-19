"""Score every OCR method against hand-corrected text.

Ground truth: `corrected_text` from any vision result folder, overridden by
outputs/ground_truth/<image>.txt. Texts are normalized (HTML, markdown marks
and whitespace removed) so layout does not count as error. Per-image CER is
capped at 1.0 so one runaway output cannot dominate the mean.
"""
from dataclasses import dataclass
from pathlib import Path
import csv

from rag_ingest.config import ENCODING, METHODS
from rag_ingest.metrics import edit_distance
from rag_ingest.stores import ResultStore, VisionStore
from rag_ingest.textproc import normalize_for_scoring

SHORT_REF_CHARS = 60  # references this short are "tiny crops"; the rest are "long"


@dataclass(frozen=True)
class Score:
    cer: float     # capped at 1.0
    edits: int
    hyp_len: int


@dataclass
class ImageScores:
    stem: str
    ref_len: int
    scores: dict[str, Score]


def load_ground_truth(stores: list[ResultStore], gt_dir: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    for store in stores:
        if isinstance(store, VisionStore):
            for stem, text in store.corrections().items():
                truth.setdefault(stem, text)
    for path in sorted(gt_dir.glob("*.txt")):
        truth[path.stem] = path.read_text(encoding=ENCODING)
    normalized = {k: normalize_for_scoring(v) for k, v in truth.items()}
    return {k: v for k, v in normalized.items() if v}


def score_all(truth: dict[str, str], methods: dict[str, dict[str, str]]) -> list[ImageScores]:
    rows = []
    for stem in sorted(truth):
        ref = truth[stem]
        scores = {}
        for name, outputs in methods.items():
            if stem in outputs:
                hyp = normalize_for_scoring(outputs[stem])
                edits = edit_distance(hyp, ref)
                scores[name] = Score(min(edits / len(ref), 1.0), edits, len(hyp))
        rows.append(ImageScores(stem, len(ref), scores))
    return rows


def summarize(rows: list[ImageScores], names: list[str]) -> dict[str, tuple]:
    """method -> (mean CER, pooled CER, empty outputs, images)."""
    out = {}
    for name in names:
        items = [(r.scores[name], r.ref_len) for r in rows if name in r.scores]
        if not items:
            continue
        mean = sum(s.cer for s, _ in items) / len(items)
        pooled = sum(s.edits for s, _ in items) / sum(n for _, n in items)
        empty = sum(1 for s, n in items if s.hyp_len < 3 and n > 5)
        out[name] = (mean, pooled, empty, len(items))
    return out


def render(rows: list[ImageScores], names: list[str]) -> str:
    short = [r for r in rows if r.ref_len <= SHORT_REF_CHARS]
    long_ = [r for r in rows if r.ref_len > SHORT_REF_CHARS]
    lines = ["# OCR benchmark\n",
             f"Ground truth: {len(rows)} images ({len(short)} short refs <= {SHORT_REF_CHARS} chars, "
             f"{len(long_)} long). CER lower is better; per-image CER capped at 1.0.\n"]
    if len(rows) < 10:
        lines.append(f"> Only {len(rows)} corrected images: treat these numbers as a rough read, "
                     "not a verdict.\n")

    for title, subset in [("All images", rows), ("Short (tiny crops)", short),
                          ("Long (screens/tables)", long_)]:
        if not subset:
            continue
        lines += [f"## {title} (n={len(subset)})\n",
                  "| method | mean CER | pooled CER | empty outputs | images |",
                  "|---|---|---|---|---|"]
        stats = summarize(subset, names)
        for name, (mean, pooled, empty, n) in sorted(stats.items(), key=lambda kv: kv[1][0]):
            lines.append(f"| {name} | {mean:.3f} | {pooled:.3f} | {empty} | {n} |")
        lines.append("")

    lines += ["## Per image (CER)\n",
              "| image | ref chars | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = [f"{r.scores[m].cer:.2f}" if m in r.scores else "-" for m in names]
        lines.append(f"| {r.stem} | {r.ref_len} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[ImageScores], names: list[str]) -> None:
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ref_chars"] + names)
        for r in rows:
            writer.writerow([r.stem, r.ref_len] +
                            [f"{r.scores[m].cer:.4f}" if m in r.scores else "" for m in names])


def method_names(exclude: list[str]) -> list[str]:
    return [m.name for m in METHODS if m.name not in exclude]
