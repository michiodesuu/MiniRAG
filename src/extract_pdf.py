from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path: Path) -> str:
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")
            pages.append(
                f"--- Page {page_number} ---\n{text}"
            )

    return "\n\n".join(pages)


def main() -> None:
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    for pdf_path in pdf_files:
        extracted_text = extract_pdf(pdf_path)

        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(
            extracted_text,
            encoding="utf-8",
        )

        print(f"Input:  {pdf_path.name}")
        print(f"Output: {output_path}")
        print(f"Characters extracted: {len(extracted_text)}")


if __name__ == "__main__":
    main()