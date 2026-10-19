"""What the PDF holds before converting: pages, scans, images, Thai font defects."""
from dataclasses import dataclass, field
from pathlib import Path

from rag_ingest.thai_glyphs import FontKey, fonts_labelling_aa_as_am, line_text

SCANNED_PAGE_MAX_CHARS = 20  # a page with fewer text characters is treated as a scan


@dataclass
class PdfFacts:
    page_count: int
    page_texts: list[str]          # reading order, Thai vowels repaired: the reference text
    images: int
    aa_as_am_fonts: set[FontKey]   # fonts whose "า" glyph is labelled "ำ"
    scanned_pages: list[int] = field(default_factory=list)  # 1-based


def page_text(page, defective: set[FontKey]) -> str:
    lines = [line_text(line, defective)
             for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])]
    return "\n".join(lines)


def inspect_pdf(pdf: Path, password: str | None) -> PdfFacts:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        if doc.needs_pass and not doc.authenticate(password or ""):
            raise PermissionError(f"{pdf.name} is password-protected; pass --password")
        defective = fonts_labelling_aa_as_am(doc)
        texts = [page_text(page, defective) for page in doc]
        images = sum(len(page.get_images()) for page in doc)
        scanned = [i for i, t in enumerate(texts, 1) if len(t.strip()) < SCANNED_PAGE_MAX_CHARS]
        return PdfFacts(doc.page_count, texts, images, defective, scanned)
