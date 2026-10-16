from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "raw"
IMAGE_DIR = ROOT / "outputs" / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            for image_index, image_info in enumerate(
                page.get_images(full=True), start=1
            ):
                xref = image_info[0]
                image = document.extract_image(xref)

                image_bytes = image["image"]
                extension = image["ext"]

                filename = (
                    f"{pdf_path.stem}"
                    f"_page_{page_index + 1}"
                    f"_image_{image_index}.{extension}"
                )

                output_path = IMAGE_DIR / filename
                output_path.write_bytes(image_bytes)

                print(
                    f"Saved: {filename} | "
                    f"{image['width']}x{image['height']} px"
                )
