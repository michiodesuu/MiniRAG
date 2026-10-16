# from pathlib import Path
# from paddleocr import PaddleOCR

# ROOT = Path(__file__).resolve().parent.parent
# IMAGE_DIR = ROOT / "outputs" / "images"
# OUTPUT_DIR = ROOT / "outputs" / "ocr"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Start with one image so the result is easy to inspect.
# image_files = sorted(
#     p for p in IMAGE_DIR.iterdir()
#     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
# )

# if not image_files:
#     raise FileNotFoundError(
#         f"No supported images found in {IMAGE_DIR}"
#     )

# image_path = image_files[0]

# ocr = PaddleOCR(
#     use_textline_orientation=True,
#     lang="th",
# )

# results = ocr.predict(input=str(image_path))

# output_path = OUTPUT_DIR / f"{image_path.stem}.txt"

# with output_path.open("w", encoding="utf-8") as f:
#     for result in results:
#         data = result.json
#         if callable(data):
#             data = data()

#         res = data.get("res", data)
#         texts = res.get("rec_texts", [])
#         scores = res.get("rec_scores", [])

#         for text, score in zip(texts, scores):
#             f.write(f"{text}\tconfidence={score:.4f}\n")
#             print(f"{text} (confidence={score:.4f})")

# print(f"\nImage: {image_path.name}")
# print(f"OCR output: {output_path}")


from pathlib import Path
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_DIR = ROOT / "outputs" / "ocr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="th",
)

extensions = {".png", ".jpg", ".jpeg", ".webp"}

image_files = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
)

for image_path in image_files:
    print(f"\nProcessing: {image_path.name}")

    results = ocr.predict(input=str(image_path))
    output_path = OUTPUT_DIR / f"{image_path.stem}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            data = result.json
            if callable(data):
                data = data()

            res = data.get("res", data)
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])

            for text, score in zip(texts, scores):
                f.write(f"{text}\tconfidence={score:.4f}\n")
                print(f"  {text} ({score:.4f})")

    print(f"Saved: {output_path}")

