"""Document utilities: inspect, dump embedded images, dump the plain text."""
from pathlib import Path

from rag_ingest.config import ENCODING
from rag_ingest.readers import list_documents, read_pages


def inspect(docs_dir: Path) -> None:
    for document in list_documents(docs_dir):
        print(f"\n{'=' * 60}\nFile: {document.name}")
        pages = list(read_pages(document))
        print(f"Pages: {len(pages)}")
        for page in pages:
            print(f"Page {page.number}: {len(page.raw_text.strip())} text characters, "
                f"{len(page.images)} embedded image(s)")


def extract_images(docs_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for document in list_documents(docs_dir):
        for page in read_pages(document):
            for image in page.images:
                filename = f"{image.ref.stem}.{image.ext}"
                (out_dir / filename).write_bytes(image.data)
                print(f"Saved: {filename} | {image.width}x{image.height} px")


def extract_text(docs_dir: Path, out_dir: Path) -> None:
    """The plain text per document, page-marked. `ingest` is the cleaned, merged version."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for document in list_documents(docs_dir):
        text = "\n\n".join(f"--- Page {p.number} ---\n{p.raw_text}" for p in read_pages(document))
        out = out_dir / f"{document.stem}.txt"
        out.write_text(text, encoding=ENCODING)
        print(f"Input:  {document.name}\nOutput: {out}\nCharacters extracted: {len(text)}")
