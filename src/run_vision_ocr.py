from pathlib import Path
import argparse
import io
import json
import sys

import ollama
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

PROMPT_PATH = ROOT / "prompts" / "thai_document_ocr.txt"
IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_DIR = ROOT / "outputs" / "vision"
MODEL = "qwen2.5vl:3b"

PADDLE_DIR = ROOT / "outputs" / "ocr"
HINT_MIN_CONFIDENCE = 0.8

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def load_hint(image_path: Path) -> str | None:
    """Confident PaddleOCR lines for this image, if a result file exists."""
    hint_path = PADDLE_DIR / f"{image_path.stem}.txt"
    if not hint_path.is_file():
        return None

    lines = []
    for raw in hint_path.read_text(encoding="utf-8").splitlines():
        text, _, conf = raw.rpartition("\tconfidence=")
        try:
            if text and float(conf) >= HINT_MIN_CONFIDENCE:
                lines.append(text)
        except ValueError:
            continue
    return "\n".join(lines) or None


def prepare_image(image_path: Path, min_side: int) -> bytes | str:
    """Upscale small images so the shorter side is at least min_side px."""
    if not min_side:
        return str(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        scale = min_side / min(img.size)
        if scale <= 1:
            return str(image_path)
        scale = min(scale, 4)
        img = img.resize(
            (round(img.width * scale), round(img.height * scale)),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


def transcribe(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    use_hint: bool,
    upscale: int,
) -> Path:
    hint = load_hint(image_path) if use_hint else None
    full_prompt = prompt
    if hint:
        full_prompt += (
            "\n\nREFERENCE (a separate OCR engine read these lines; it may be "
            "wrong or incomplete, so trust the image over it):\n" + hint
        )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": full_prompt,
                "images": [prepare_image(image_path, upscale)],
            }
        ],
        options={"temperature": 0},
    )

    text = response["message"]["content"].strip()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.json"

    result = {
        "image": str(image_path.relative_to(ROOT))
        if image_path.is_relative_to(ROOT)
        else str(image_path),
        "model": MODEL,
        "prompt_file": str(PROMPT_PATH.relative_to(ROOT)),
        "settings": {
            "temperature": 0,
            "paddle_hint": bool(hint),
            "upscale_min_side": upscale,
        },
        "text": text,
        "review_status": "pending",
    }

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main():
    # Thai output garbles on the default Windows console encoding.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run local vision transcription with Ollama."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        help="Path to one image file, absolute or relative to the project root.",
    )
    group.add_argument(
        "--all",
        nargs="?",
        const=str(IMAGE_DIR),
        metavar="DIR",
        help="Process every image in DIR (default: outputs/images).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In batch mode, skip images that already have a result JSON.",
    )
    parser.add_argument(
        "--hint",
        action="store_true",
        help="Add confident PaddleOCR lines from outputs/ocr/ to the prompt.",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=0,
        metavar="PX",
        help="Upscale images whose shorter side is below PX (e.g. 1000).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Where to save result JSON (default: outputs/vision). "
        "Use a different folder to compare settings side by side.",
    )
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    if args.image:
        image_path = resolve(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_path = transcribe(
            image_path, prompt, output_dir, args.hint, args.upscale
        )
        print("\n=== OCR RESULT ===\n")
        print(json.loads(output_path.read_text(encoding="utf-8"))["text"])
        print(f"\nSaved to: {output_path}")
        return

    image_dir = resolve(args.all)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    done = skipped = failed = 0
    for i, image_path in enumerate(image_files, 1):
        label = f"[{i}/{len(image_files)}] {image_path.name}"

        if args.skip_existing and (output_dir / f"{image_path.stem}.json").exists():
            print(f"{label} - skipped (exists)")
            skipped += 1
            continue

        try:
            transcribe(image_path, prompt, output_dir, args.hint, args.upscale)
            print(f"{label} - ok")
            done += 1
        except Exception as error:
            print(f"{label} - FAILED: {error}")
            failed += 1

    print(
        f"\nDone: {done} | Skipped: {skipped} | Failed: {failed}"
        f"\nResults in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
