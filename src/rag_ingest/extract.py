"""PDF utilities: inspect, dump embedded images, dump the plain text layer."""
from pathlib import Path

from rag_ingest.pdf import list_pdfs, read_pages


def inspect(pdf_dir: Path) -> None:
    for pdf in list_pdfs(pdf_dir):
        print(f"\n{'=' * 60}\nFile: {pdf.name}")
        pages = list(read_pages(pdf))
        print(f"Pages: {len(pages)}")
        for page in pages:
            print(f"Page {page.number}: {len(page.raw_text.strip())} text characters, "
                  f"{len(page.images)} embedded image(s)")


def extract_images(pdf_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in list_pdfs(pdf_dir):
        for page in read_pages(pdf):
            for image in page.images:
                filename = f"{image.ref.stem}.{image.ext}"
                (out_dir / filename).write_bytes(image.data)
                print(f"Saved: {filename} | {image.width}x{image.height} px")


def extract_text(pdf_dir: Path, out_dir: Path) -> None:
    """The raw text layer per PDF, page-marked. `ingest` is the cleaned, merged version."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in list_pdfs(pdf_dir):
        text = "\n\n".join(f"--- Page {p.number} ---\n{p.raw_text}" for p in read_pages(pdf))
        out = out_dir / f"{pdf.stem}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"Input:  {pdf.name}\nOutput: {out}\nCharacters extracted: {len(text)}")
