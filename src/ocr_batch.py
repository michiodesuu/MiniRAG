from pathlib import Path
import json
import re

from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_DIR = ROOT / "outputs" / "ocr"
OUTPUT_FILE = OUTPUT_DIR / "ocr_results.jsonl"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="th",
)


def parse_source(filename: str) -> dict:
    match = re.match(
        r"(?P<document>.+)_page_(?P<page>\d+)"
        r"_image_(?P<image>\d+)\.",
        filename,
    )

    if not match:
        return {
            "source_document": None,
            "page_number": None,
            "image_number": None,
        }

    return {
        "source_document": match.group("document") + ".pdf",
        "page_number": int(match.group("page")),
        "image_number": int(match.group("image")),
    }


def extract_text(image_path: Path) -> list[dict]:
    results = ocr.predict(input=str(image_path))
    lines = []

    for result in results:
        data = result.json
        if callable(data):
            data = data()

        res = data.get("res", data)
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])

        for text, score in zip(texts, scores):
            lines.append({
                "text": text,
                "confidence": float(score),
            })

    return lines


def main() -> None:
    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        raise FileNotFoundError(
            f"No supported images found in {IMAGE_DIR}"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for image_path in image_files:
            print(f"Processing: {image_path.name}")

            try:
                lines = extract_text(image_path)

                record = {
                    **parse_source(image_path.name),
                    "image_file": image_path.name,
                    "ocr_lines": lines,
                    "line_count": len(lines),
                    "status": "success",
                }

            except Exception as error:
                record = {
                    **parse_source(image_path.name),
                    "image_file": image_path.name,
                    "ocr_lines": [],
                    "line_count": 0,
                    "status": "error",
                    "error": str(error),
                }

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            print(
                f"  Status: {record['status']} | "
                f"Lines: {record['line_count']}"
            )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
