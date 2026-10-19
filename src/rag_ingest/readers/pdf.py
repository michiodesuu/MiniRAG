"""Read PDFs: text blocks and embedded images, with their positions on the page."""
from pathlib import Path
from typing import Iterator

import pymupdf

from rag_ingest.models import EmbeddedImage, ImageRef, Page, TextBlock
from rag_ingest.thai_glyphs import FontKey, fonts_labelling_aa_as_am, line_text


def _text_blocks(page, defective: set[FontKey]) -> list[TextBlock]:
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        lines = [line_text(line, defective).strip() for line in block["lines"]]
        text = "\n".join(l for l in lines if l)
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
        defective = fonts_labelling_aa_as_am(doc)  # Thai vowels the fonts mislabel
        for number, page in enumerate(doc, start=1):
            yield Page(number, page.rect.y1, page.get_text("text"),
                       _text_blocks(page, defective), _images(doc, page, pdf_path.stem, number))
