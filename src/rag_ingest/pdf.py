"""Read PDFs: text blocks and embedded images, with their positions on the page."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import hashlib

import pymupdf

from rag_ingest.models import ImageRef
from rag_ingest.textproc import fix_sara_am


@dataclass(frozen=True)
class TextBlock:
    y: float
    x: float
    text: str


@dataclass(frozen=True)
class EmbeddedImage:
    ref: ImageRef
    data: bytes
    ext: str
    width: int
    height: int
    y: float | None  # top-left on the page; None if the image is not drawn
    x: float | None

    @property
    def md5(self) -> str:
        return hashlib.md5(self.data).hexdigest()

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


@dataclass
class Page:
    number: int  # 1-based
    height: float
    raw_text: str
    blocks: list[TextBlock]
    images: list[EmbeddedImage]


def _text_blocks(page) -> list[TextBlock]:
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        lines = ["".join(span["text"] for span in line["spans"]).strip()
                 for line in block["lines"]]
        text = fix_sara_am("\n".join(l for l in lines if l))
        if text:
            x0, y0, *_ = block["bbox"]
            blocks.append(TextBlock(y0, x0, text))
    return blocks


def _images(doc, page, document: str, page_number: int) -> list[EmbeddedImage]:
    out = []
    for index, info in enumerate(page.get_images(full=True), start=1):
        xref = info[0]
        image = doc.extract_image(xref)
        rects = page.get_image_rects(xref)
        y, x = (rects[0].y0, rects[0].x0) if rects else (None, None)
        out.append(EmbeddedImage(ImageRef(document, page_number, index), image["image"],
                                 image["ext"], image["width"], image["height"], y, x))
    return out


def read_pages(pdf_path: Path) -> Iterator[Page]:
    """Pages one at a time, so only one page's images are in memory."""
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            yield Page(number, page.rect.y1, page.get_text("text"),
                       _text_blocks(page), _images(doc, page, pdf_path.stem, number))


def list_pdfs(folder: Path) -> list[Path]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    return pdfs
