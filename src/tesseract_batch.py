"""Run Tesseract (Thai + English) on every image.

Writes outputs/tesseract/<image>.txt in the same `text<TAB>confidence=0.xxxx`
line format as the PaddleOCR files in outputs/ocr/.
"""
from pathlib import Path
import argparse
import re
import sys

import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_DIR = ROOT / "outputs" / "tesseract"

TESSERACT_EXE = Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Tesseract puts spaces between Thai words/characters; Thai is written without them.
THAI_GAP = re.compile(r"(?<=[฀-๿]) (?=[฀-๿])")


def ocr_lines(image_path: Path, psm: int) -> list[tuple[str, float]]:
    with Image.open(image_path) as img:
        data = pytesseract.image_to_data(
            img.convert("RGB"),
            lang="tha+eng",
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

    lines: dict[tuple, list[tuple[str, float]]] = {}
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if not word.strip() or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((word.strip(), conf))

    result = []
    for words in lines.values():
        text = THAI_GAP.sub("", " ".join(w for w, _ in words))
        score = sum(c for _, c in words) / len(words) / 100
        result.append((text, score))
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psm", type=int, default=3,
                        help="Tesseract page segmentation mode (default 3; 6 = one block).")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {IMAGE_DIR}")

    done = skipped = failed = 0
    for i, image_path in enumerate(images, 1):
        label = f"[{i}/{len(images)}] {image_path.name}"
        out = OUTPUT_DIR / f"{image_path.stem}.txt"
        if args.skip_existing and out.exists():
            print(f"{label} - skipped (exists)", flush=True)
            skipped += 1
            continue
        try:
            lines = ocr_lines(image_path, args.psm)
            out.write_text(
                "".join(f"{t}\tconfidence={c:.4f}\n" for t, c in lines),
                encoding="utf-8",
            )
            print(f"{label} - ok ({len(lines)} lines)", flush=True)
            done += 1
        except Exception as error:
            print(f"{label} - FAILED: {error}", flush=True)
            failed += 1

    print(f"\nDone: {done} | Skipped: {skipped} | Failed: {failed}"
          f"\nResults in: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
