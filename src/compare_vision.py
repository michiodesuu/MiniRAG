"""Compare two vision OCR result folders.

Reports per-image similarity between runs (A vs B). If an image has a
`corrected_text` (from review_vision.py) in either folder, it is used as
ground truth and each run's character error rate (CER) is reported.
"""
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(folder: Path) -> dict[str, dict]:
    return {
        f.name: json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(folder.glob("*.json"))
    }


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return edit_distance(hyp, ref) / max(len(ref), 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="outputs/vision", help="Baseline folder.")
    parser.add_argument("--b", default="outputs/vision_hint", help="New folder.")
    parser.add_argument("--diff", action="store_true",
                        help="Print a line diff for images that differ.")
    parser.add_argument("--report", default="outputs/compare_report.md",
                        help="Markdown report path ('' to skip).")
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    a, b = load(resolve(args.a)), load(resolve(args.b))
    names = sorted(set(a) & set(b))
    only = sorted(set(a) ^ set(b))
    if not names:
        raise SystemExit("No images in common between the two folders.")

    rows, cers_a, cers_b = [], [], []
    for n in names:
        ta, tb = a[n]["text"], b[n]["text"]
        sim = SequenceMatcher(None, ta, tb).ratio()
        ref = b[n].get("corrected_text") or a[n].get("corrected_text")
        ca = cer(ta, ref) if ref is not None else None
        cb = cer(tb, ref) if ref is not None else None
        if ca is not None:
            cers_a.append(ca)
            cers_b.append(cb)
        rows.append((n, sim, len(ta), len(tb), ca, cb, ta, tb))

    lines = [f"# Vision OCR comparison\n\nA = `{args.a}`  |  B = `{args.b}`\n",
             "| image | A~B similarity | len A | len B | CER A | CER B |",
             "|---|---|---|---|---|---|"]
    fmt = lambda x: "-" if x is None else f"{x:.2f}"
    for n, sim, la, lb, ca, cb, *_ in rows:
        lines.append(f"| {n[:-5]} | {sim:.2f} | {la} | {lb} | {fmt(ca)} | {fmt(cb)} |")

    identical = sum(r[1] == 1.0 for r in rows)
    lines.append(f"\n{len(rows)} images compared, {identical} identical, "
                 f"mean similarity {sum(r[1] for r in rows) / len(rows):.2f}.")
    if cers_a:
        lines.append(f"Ground truth available for {len(cers_a)} images: "
                     f"mean CER A {sum(cers_a)/len(cers_a):.3f} vs "
                     f"B {sum(cers_b)/len(cers_b):.3f} (lower is better).")
    else:
        lines.append("No corrected_text yet, so no accuracy score. Review some "
                     "results with review_vision.py, then rerun this.")
    if only:
        lines.append(f"Only in one folder (skipped): {', '.join(only)}")

    if args.diff:
        for n, sim, *_, ta, tb in rows:
            if sim < 1.0:
                lines.append(f"\n## {n[:-5]}\n```diff")
                lines.extend(unified_diff(ta.splitlines(), tb.splitlines(),
                                          "A", "B", lineterm="", n=0))
                lines.append("```")

    report = "\n".join(lines)
    print(report)
    if args.report:
        out = resolve(args.report)
        out.write_text(report, encoding="utf-8")
        print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
