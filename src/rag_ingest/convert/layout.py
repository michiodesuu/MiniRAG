"""Rebuild the PDF's pages as Word sections (pdf2docx)."""
from pathlib import Path


def rebuild_layout(pdf: Path, docx: Path, pages: list[int] | None, password: str | None) -> None:
    from pdf2docx import Converter

    docx.parent.mkdir(parents=True, exist_ok=True)
    converter = Converter(str(pdf), password=password)
    try:
        # One page after another: multiprocessing would multiply memory use.
        converter.convert(str(docx), pages=pages, multi_processing=False)
    finally:
        converter.close()
