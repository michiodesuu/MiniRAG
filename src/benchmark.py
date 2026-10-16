"""Benchmark every OCR/transcription method against hand-corrected text.

Ground truth comes from (first found wins, a .txt file overrides JSON):
  - outputs/ground_truth/<image>.txt            (plain corrected text), or
  - `corrected_text` saved by review_vision.py in any outputs/vision*/ JSON.

Methods scored (folders that exist are included automatically):
  vision, vision_hint, vision_both, vision_typhoon  (JSON, "text" field)
  ocr (PaddleOCR), tesseract                        (<text>\\tconfidence=<n> lines)

Texts are normalized before scoring so layout does not count as error:
HTML tags and markdown marks are removed and all whitespace is dropped (Thai
is written without spaces). Score = character error rate (CER): edit distance
divided by reference length, capped at 1.0 per image so one runaway output
cannot dominate the average. Lower is better.
"""
from pathlib import Path
import argparse
import csv
import json
import re
import sys
import unicodedata

from compare_vision import edit_distance

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

JSON_METHODS = ["vision", "vision_hint", "vision_both", "vision_typhoon"]
TEXT_METHODS = {"ocr": "paddleocr", "tesseract": "tesseract"}

SHORT_REF_CHARS = 60  # references this short are "tiny crops"; the rest are "long"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"</(td|th|tr|p|div|li|h\d)>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*|`​﻿]", "", text)
    return re.sub(r"\s+", "", text)


def read_text_lines(path: Path) -> str:
    parts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text, sep, _ = raw.rpartition("\tconfidence=")
        parts.append(text if sep else raw)
    return "\n".join(parts)


def load_ground_truth(gt_dir: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    for folder in JSON_METHODS:
        for path in sorted((OUTPUTS / folder).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("corrected_text") is not None:
                truth.setdefault(path.stem, data["corrected_text"])
    for path in sorted(gt_dir.glob("*.txt")):
        truth[path.stem] = path.read_text(encoding="utf-8")
    return truth


def load_method(name: str) -> dict[str, str]:
    if name in TEXT_METHODS:
        return {p.stem: read_text_lines(p) for p in (OUTPUTS / name).glob("*.txt")}
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))["text"]
        for p in (OUTPUTS / name).glob("*.json")
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth-dir", default=str(OUTPUTS / "ground_truth"))
    parser.add_argument("--report", default=str(OUTPUTS / "benchmark_report.md"))
    parser.add_argument("--csv", default=str(OUTPUTS / "benchmark.csv"))
    parser.add_argument("--exclude", nargs="*", default=[], metavar="METHOD",
                        help="Methods to leave out, e.g. --exclude vision.")
    args = parser.parse_args()

    truth = {k: normalize(v) for k, v in load_ground_truth(Path(args.ground_truth_dir)).items()}
    truth = {k: v for k, v in truth.items() if v}
    if not truth:
        raise SystemExit(
            "No ground truth yet. Correct some images with:\n"
            "  python src/review_vision.py --dir outputs/vision_hint\n"
            "(or put plain corrected text in outputs/ground_truth/<image>.txt), "
            "then rerun this script."
        )

    methods = {}
    for name in JSON_METHODS + list(TEXT_METHODS):
        if name in args.exclude or not (OUTPUTS / name).is_dir():
            continue
        loaded = load_method(name)
        if loaded:
            methods[name] = loaded

    images = sorted(truth)
    rows = []  # (image, ref_len, {method: (cer, normalized_len)})
    for img in images:
        ref = truth[img]
        scores = {}
        for name, outputs in methods.items():
            if img not in outputs:
                continue
            hyp = normalize(outputs[img])
            cer = edit_distance(hyp, ref) / len(ref)
            scores[name] = (min(cer, 1.0), edit_distance(hyp, ref), len(hyp))
        rows.append((img, len(ref), scores))

    def summarize(subset):
        out = {}
        for name in methods:
            items = [(r[2][name], r[1]) for r in subset if name in r[2]]
            if not items:
                continue
            capped = [s[0] for s, _ in items]
            pooled = sum(s[1] for s, _ in items) / sum(n for _, n in items)
            empty = sum(1 for s, n in items if s[2] < 3 and n > 5)
            out[name] = (sum(capped) / len(capped), pooled, empty, len(items))
        return out

    short = [r for r in rows if r[1] <= SHORT_REF_CHARS]
    long_ = [r for r in rows if r[1] > SHORT_REF_CHARS]

    lines = [f"# OCR benchmark\n",
             f"Ground truth: {len(rows)} images ({len(short)} short refs <= {SHORT_REF_CHARS} chars, "
             f"{len(long_)} long). CER lower is better; per-image CER capped at 1.0.\n"]
    if len(rows) < 10:
        lines.append(f"> Only {len(rows)} corrected images: treat these numbers as a rough read, "
                     "not a verdict.\n")

    for title, subset in [("All images", rows), ("Short (tiny crops)", short), ("Long (screens/tables)", long_)]:
        if not subset:
            continue
        lines += [f"## {title} (n={len(subset)})\n",
                  "| method | mean CER | pooled CER | empty outputs | images |",
                  "|---|---|---|---|---|"]
        stats = summarize(subset)
        for name, (mean, pooled, empty, n) in sorted(stats.items(), key=lambda kv: kv[1][0]):
            lines.append(f"| {name} | {mean:.3f} | {pooled:.3f} | {empty} | {n} |")
        lines.append("")

    names = list(methods)
    lines += ["## Per image (CER)\n",
              "| image | ref chars | " + " | ".join(names) + " |",
              "|---|---|" + "---|" * len(names)]
    for img, n, scores in rows:
        cells = [f"{scores[m][0]:.2f}" if m in scores else "-" for m in names]
        lines.append(f"| {img} | {n} | " + " | ".join(cells) + " |")

    report = "\n".join(lines)
    print(report)

    Path(args.report).write_text(report, encoding="utf-8")
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ref_chars"] + names)
        for img, n, scores in rows:
            writer.writerow([img, n] + [f"{scores[m][0]:.4f}" if m in scores else "" for m in names])
    print(f"\nSaved: {args.report}\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
