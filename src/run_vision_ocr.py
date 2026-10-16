from pathlib import Path
import argparse
import base64
import io
import json
import os
import sys

import httpx
import ollama
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

PROMPT_PATH = ROOT / "prompts" / "thai_document_ocr.txt"
IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_DIR = ROOT / "outputs" / "vision"
MODEL = "qwen2.5vl:3b"
# Ollama's default context (4096) is too small once images are upscaled.
NUM_CTX = 8192

PADDLE_DIR = ROOT / "outputs" / "ocr"
TESSERACT_DIR = ROOT / "outputs" / "tesseract"
HINT_MIN_CONFIDENCE = 0.8

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def confident_lines(hint_dir: Path, image_path: Path) -> list[str]:
    """Lines from `<hint_dir>/<image>.txt` at or above HINT_MIN_CONFIDENCE."""
    hint_path = hint_dir / f"{image_path.stem}.txt"
    if not hint_path.is_file():
        return []

    lines = []
    for raw in hint_path.read_text(encoding="utf-8").splitlines():
        text, _, conf = raw.rpartition("\tconfidence=")
        try:
            if text and float(conf) >= HINT_MIN_CONFIDENCE:
                lines.append(text)
        except ValueError:
            continue
    return lines


def load_hint(image_path: Path, source: str) -> tuple[str | None, list[str]]:
    """Build the hint block for the chosen OCR source(s).

    Returns (hint text or None, names of the sources that contributed).
    """
    sources = {"paddle": PADDLE_DIR, "tesseract": TESSERACT_DIR}
    names = list(sources) if source == "both" else [source]

    parts, used = [], []
    for name in names:
        lines = confident_lines(sources[name], image_path)
        if lines:
            used.append(name)
            parts.append((name, "\n".join(lines)))

    if not parts:
        return None, []
    if len(parts) == 1:
        return parts[0][1], used
    return "\n\n".join(f"[{name} OCR]\n{text}" for name, text in parts), used


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


def call_api(api: dict, prompt: str, image: bytes | str) -> str:
    """OpenAI-compatible /v1/chat/completions call with a base64 image."""
    raw = image if isinstance(image, bytes) else Path(image).read_bytes()
    mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    b64 = base64.b64encode(raw).decode()

    headers = {"Authorization": f"Bearer {api['key']}"} if api.get("key") else {}
    response = httpx.post(
        api["url"],
        headers=headers,
        timeout=300,
        json={
            "model": api["model"],
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def transcribe(
    image_path: Path,
    prompt: str,
    output_dir: Path,
    use_hint: bool,
    upscale: int,
    hint_source: str = "paddle",
    api: dict | None = None,
) -> Path:
    hint, hint_used = load_hint(image_path, hint_source) if use_hint else (None, [])
    full_prompt = prompt
    if hint:
        full_prompt += (
            "\n\nREFERENCE (separate OCR engines read these lines; they may be "
            "wrong or incomplete, so trust the image over them):\n" + hint
        )

    image = prepare_image(image_path, upscale)

    if api:
        text = call_api(api, full_prompt, image)
        model_name, settings = api["model"], {"backend": "api"}
    else:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                    "images": [image],
                }
            ],
            options={"temperature": 0, "num_ctx": NUM_CTX},
        )
        text = response["message"]["content"].strip()
        model_name, settings = MODEL, {"backend": "ollama", "num_ctx": NUM_CTX}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.json"

    result = {
        "image": str(image_path.relative_to(ROOT))
        if image_path.is_relative_to(ROOT)
        else str(image_path),
        "model": model_name,
        "prompt_file": str(PROMPT_PATH.relative_to(ROOT)),
        "settings": {
            **settings,
            "temperature": 0,
            "paddle_hint": "paddle" in hint_used,
            "tesseract_hint": "tesseract" in hint_used,
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
        help="Add confident OCR lines (see --hint-source) to the prompt.",
    )
    parser.add_argument(
        "--hint-source",
        choices=["paddle", "tesseract", "both"],
        default="paddle",
        help="Which OCR output feeds --hint: outputs/ocr (paddle), "
        "outputs/tesseract, or both (default: paddle).",
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
    parser.add_argument(
        "--backend",
        choices=["ollama", "api"],
        default="ollama",
        help="ollama = local model; api = OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("OCR_API_URL"),
        help="Full /v1/chat/completions URL (or env OCR_API_URL).",
    )
    parser.add_argument(
        "--api-model",
        default=os.environ.get("OCR_API_MODEL"),
        help="Model name sent to the API (or env OCR_API_MODEL).",
    )
    args = parser.parse_args()
    output_dir = resolve(args.output_dir)

    api = None
    if args.backend == "api":
        if not (args.api_url and args.api_model):
            parser.error("--backend api needs --api-url and --api-model")
        # Key is read from the environment only, never from the command line.
        api = {
            "url": args.api_url,
            "model": args.api_model,
            "key": os.environ.get("OCR_API_KEY"),
        }

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    if args.image:
        image_path = resolve(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        output_path = transcribe(
            image_path, prompt, output_dir, args.hint, args.upscale,
            args.hint_source, api,
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
            transcribe(
                image_path, prompt, output_dir, args.hint, args.upscale,
                args.hint_source, api,
            )
            print(f"{label} - ok")
            done += 1
        except Exception as error:
            print(f"{label} - FAILED: {error}")
            failed += 1

    print(
        f"\nDone: {done} | Skipped: {skipped} | Failed: {failed}"
        f"\nResults in: {output_dir}"
    )


if __name__ == "__main__":
    main()
