from pathlib import Path
import pymupdf

INPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
    print(f"\n{'=' * 60}")
    print(f"File: {pdf_path.name}")

    with pymupdf.open(pdf_path) as document:
        print(f"Pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            images = page.get_images(full=True)

            print(
                f"Page {page_number}: "
                f"{len(text.strip())} text characters, "
                f"{len(images)} embedded image(s)"
            )
