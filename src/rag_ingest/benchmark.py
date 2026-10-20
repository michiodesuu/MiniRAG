"""Score every OCR method against ground truth (CER), and count its failures.

Ground truth: outputs/ground_truth/<image>.txt with provenance.json (see
ground_truth.py); otherwise a reviewer's `corrected_text` in a vision result
folder (provenance: human, edited from that folder's model text). Only
images worth a ground truth are scored: icons (no text) and byte-identical
duplicates are left out, per the ingest manifest.

Texts are normalized (HTML, markdown marks and whitespace removed) so layout
does not count as error. CER = edit distance / reference length, capped at 1.0
per image so one runaway output cannot dominate the mean. Lower is better.

Failures are counted on every benchmarked image, ground truth or not: an
output the ingest stage would reject (quality.py: empty, "no readable text"
on a real screenshot, the model echoing its instructions), or no output at all.

Methods: the known result folders (config.METHODS) plus any other
outputs/vision_* folder, e.g. a run with another prompt, plus "pipeline": what
ingest actually takes (ingest.DEFAULT_SOURCES, the first output that passes the
quality rules), so the engine choice is measured as it is used.
"""
from dataclasses import dataclass, field
from pathlib import Path
import csv

from rag_ingest.config import ENCODING, METHODS, Method, Paths
from rag_ingest.ground_truth import CLAUDE, HUMAN, GroundTruthStore, benchmark_images, edited_from
from rag_ingest.ingest import DEFAULT_SOURCES
from rag_ingest.metrics import edit_distance
from rag_ingest.quality import rejection
from rag_ingest.stores import ResultStore, VisionStore, open_store
from rag_ingest.textproc import normalize_for_scoring

SHORT_REF_CHARS = 60  # references this short are small crops; the rest are screens/tables
UNKNOWN = "unknown"
PIPELINE = "pipeline"  # ingest's source chain (DEFAULT_SOURCES with fallback), not a folder


@dataclass(frozen=True)
class Truth:
    text: str     # normalized
    by: str       # human / claude / unknown
    method: str


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
    by: str = UNKNOWN


@dataclass
class Failures:
    rejected: dict[str, int] = field(default_factory=dict)  # rule -> count
    missing: int = 0
    images: int = 0

    @property
    def total(self) -> int:
        return sum(self.rejected.values()) + self.missing


def load_ground_truth(stores: list[ResultStore], truth_store: GroundTruthStore,
                      images: list[str] | None = None) -> dict[str, Truth]:
    raw: dict[str, tuple[str, str, str]] = {}
    for store in stores:
        if isinstance(store, VisionStore):
            for stem, text in store.corrections().items():
                raw.setdefault(stem, (text, HUMAN, edited_from(store.name)))
    provenance = truth_store.provenance()
    for stem, text in truth_store.texts().items():  # a ground-truth file wins
        record = provenance.get(stem, {})
        raw[stem] = (text, record.get("by", UNKNOWN), record.get("method", UNKNOWN))
    wanted = set(images) if images is not None else None
    truth = {stem: Truth(normalize_for_scoring(text), by, method)
             for stem, (text, by, method) in raw.items() if wanted is None or stem in wanted}
    return {stem: t for stem, t in truth.items() if t.text}


def discover_methods(paths: Paths, exclude: list[str]) -> list[Method]:
    known = [m for m in METHODS if paths.method(m.name).is_dir()]
    names = {m.name for m in METHODS}
    extra = [Method(p.name, "vision") for p in sorted(paths.outputs.glob("vision_*"))
             if p.is_dir() and p.name not in names and any(p.glob("*.json"))]
    return [m for m in known + extra if m.name not in exclude]


def pipeline_outputs(outputs: dict[str, dict[str, str]], sources: tuple[str, ...]) -> dict[str, str]:
    """What ingest would take per image: the first source whose text passes the quality rules."""
    chosen: dict[str, str] = {}
    for name in reversed([s for s in sources if s in outputs]):  # earlier sources overwrite later ones
        chosen.update({stem: text for stem, text in outputs[name].items() if rejection(text) is None})
    return chosen


def score_all(truth: dict[str, Truth], methods: dict[str, dict[str, str]]) -> list[ImageScores]:
    rows = []
    for stem in sorted(truth):
        ref = truth[stem].text
        scores = {}
        for name, outputs in methods.items():
            if stem in outputs:
                hyp = normalize_for_scoring(outputs[stem])
                edits = edit_distance(hyp, ref)
                scores[name] = Score(min(edits / len(ref), 1.0), edits, len(hyp))
        rows.append(ImageScores(stem, len(ref), scores, truth[stem].by))
    return rows


def count_failures(outputs: dict[str, str], images: list[str]) -> Failures:
    failures = Failures(images=len(images))
    for stem in images:
        if stem not in outputs:
            failures.missing += 1
        elif reason := rejection(outputs[stem]):
            failures.rejected[reason] = failures.rejected.get(reason, 0) + 1
    return failures


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


def _cer_table(title: str, rows: list[ImageScores], names: list[str]) -> list[str]:
    if not rows:
        return []
    lines = [f"## {title} (n={len(rows)})\n", "| method | mean CER | pooled CER | empty outputs | images |",
             "|---|---|---|---|---|"]
    for name, (mean, pooled, empty, n) in sorted(summarize(rows, names).items(), key=lambda kv: kv[1][0]):
        lines.append(f"| {name} | {mean:.3f} | {pooled:.3f} | {empty} | {n} |")
    return lines + [""]


def _failure_table(failures: dict[str, Failures]) -> list[str]:
    lines = ["## Failures on every benchmarked image (ingest would reject these)\n",
             "| method | failed | of | rejected (rule: count) | no output |", "|---|---|---|---|---|"]
    for name, f in sorted(failures.items(), key=lambda kv: kv[1].total):
        rules = ", ".join(f"{rule}: {n}" for rule, n in sorted(f.rejected.items())) or "-"
        lines.append(f"| {name} | {f.total} | {f.images} | {rules} | {f.missing} |")
    return lines + [""]


def render(rows: list[ImageScores], names: list[str], failures: dict[str, Failures] | None = None) -> str:
    by = {who: [r for r in rows if r.by == who] for who in (HUMAN, CLAUDE, UNKNOWN)}
    short = [r for r in rows if r.ref_len <= SHORT_REF_CHARS]
    long_ = [r for r in rows if r.ref_len > SHORT_REF_CHARS]
    lines = ["# OCR benchmark\n",
             f"Ground truth: {len(rows)} images ({len(by[HUMAN])} by a person, {len(by[CLAUDE])} by Claude "
             f"from a blank box and not yet checked by a person, {len(by[UNKNOWN])} of unknown origin). "
             "Icons and duplicate images are not scored. CER lower is better; per-image CER capped at 1.0.\n"]
    if len(rows) < 10:
        lines.append(f"> Only {len(rows)} images with ground truth: a rough read, not a verdict.\n")
    if by[CLAUDE]:
        lines.append("> Ground truth made by Claude is an independent reading of the image, not a human "
                     "check; the table of ground truth by a person is the stricter reference.\n")
    lines += _cer_table("All images", rows, names)
    lines += _cer_table("Ground truth by a person", by[HUMAN], names)
    lines += _cer_table("Ground truth by Claude (blank box)", by[CLAUDE], names)
    lines += _cer_table(f"Small crops (reference <= {SHORT_REF_CHARS} chars)", short, names)
    lines += _cer_table(f"Screens and tables (reference > {SHORT_REF_CHARS} chars)", long_, names)
    if failures:
        lines += _failure_table(failures)
    lines += ["## Per image (CER)\n", "| image | by | ref chars | " + " | ".join(names) + " |",
              "|---|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = [f"{r.scores[m].cer:.2f}" if m in r.scores else "-" for m in names]
        lines.append(f"| {r.stem} | {r.by} | {r.ref_len} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[ImageScores], names: list[str]) -> None:
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "by", "ref_chars"] + names)
        for r in rows:
            writer.writerow([r.stem, r.by, r.ref_len] +
                            [f"{r.scores[m].cer:.4f}" if m in r.scores else "" for m in names])


@dataclass
class BenchmarkResult:
    report: str
    rows: list[ImageScores]
    names: list[str]


def run(paths: Paths, truth_dir: Path, exclude: list[str]) -> BenchmarkResult | None:
    """None when there is no ground truth yet."""
    methods = discover_methods(paths, exclude)
    stores = {m.name: open_store(paths, m.name, m.kind) for m in methods}
    images = benchmark_images(paths.ingest / "manifest.json")
    truth = load_ground_truth(list(stores.values()), GroundTruthStore(truth_dir), images)
    if not truth:
        return None
    outputs = {name: texts for name, store in stores.items() if (texts := store.texts())}
    if sum(s in outputs for s in DEFAULT_SOURCES) > 1:
        outputs[PIPELINE] = pipeline_outputs(outputs, DEFAULT_SOURCES)
    names = list(outputs)
    rows = score_all(truth, outputs)
    scope = images if images is not None else sorted(truth)
    failures = {name: count_failures(texts, scope) for name, texts in outputs.items()}
    return BenchmarkResult(render(rows, names, failures), rows, names)
