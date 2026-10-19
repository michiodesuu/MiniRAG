"""Read Word (.docx) files into Pages: paragraphs, tables and images in reading order.

Word stores a flow of text, not positions, so positions are reading order and
page numbers are approximate (see breaks.py). What becomes text:
  - headings -> "# " / "## " ... (style "Heading N", "Title", or outline level)
  - lists    -> the numbers Word shows ("1.", "2.1", "ก)") or "-" for bullets,
                nested levels indented (numbering.py)
  - tables   -> markdown, merged cells aligned; images in cells follow the table
  - text boxes (callouts) -> separate blocks after their paragraph
  - tracked changes: insertions kept, deletions and field codes dropped
Images (inline, floating, legacy VML) are read once each, in order.

Not read: headers/footers, footnotes, comments, images linked rather than
embedded, and old binary .doc files (save as .docx first).

Modules: ooxml (XML names) · inline (one paragraph's content) · styles
(inherited properties) · numbering (list labels) · breaks (page strategy) ·
tables (grid + markdown). WordReader below directs them and feeds a
PageBuilder.
"""
from pathlib import Path
from typing import Iterator

from rag_ingest.models import Page
from rag_ingest.readers.builder import PageBuilder
from rag_ingest.readers.word import ooxml as x
from rag_ingest.readers.word.breaks import choose_policy
from rag_ingest.readers.word.inline import ImageMark, Part, inline_parts
from rag_ingest.readers.word.numbering import Numbering
from rag_ingest.readers.word.styles import StyleResolver
from rag_ingest.readers.word.tables import cell_paragraphs, escape_cell, markdown_table, read_grid

_WRAPPERS = (x.SDT, x.SDT_CONTENT, x.CUSTOM_XML)  # content controls, e.g. a table of contents


def body_elements(container) -> Iterator:
    """Top-level paragraphs and tables in order, unwrapping content controls."""
    for child in container:
        if child.tag in (x.P, x.TBL):
            yield child
        elif child.tag in _WRAPPERS:
            yield from body_elements(child)


def numbering_element(document):
    """numbering.xml of the document, or None for a file without any lists
    (python-docx raises instead of returning nothing)."""
    try:
        return document.part.numbering_part.element
    except (NotImplementedError, KeyError):
        return None


class WordReader:
    def __init__(self, document, stem: str):
        self.document = document
        self.styles = StyleResolver(document)
        self.numbering = Numbering(numbering_element(document))
        self.breaks = choose_policy(document.element.body, self.styles)
        self.pages = PageBuilder(stem)
        self.handlers = {x.P: self.paragraph, x.TBL: self.table}

    def read(self) -> list[Page]:
        for element in body_elements(self.document.element.body):
            self.handlers[element.tag](element)
        return self.pages.finish()

    # -- pieces shared by body paragraphs and table cells

    def prefix(self, paragraph) -> str:
        """"## " for a heading, "  2.1 " for a list item, else ""."""
        level = self.styles.heading_level(paragraph)
        if level:
            return "#" * min(level, 6) + " "
        position = self.styles.list_position(paragraph)
        label = self.numbering.label(*position) if position else None
        return "  " * position[1] + label + " " if label else ""

    def image(self, rid: str) -> None:
        part = self.document.part.related_parts.get(rid)
        data = getattr(part, "blob", None)
        if data is not None:  # None: linked from outside the file, not embedded
            ext = Path(str(part.partname)).suffix.lstrip(".").lower() or "bin"
            self.pages.add_image(data, ext)

    # -- body paragraphs

    def paragraph(self, element) -> None:
        if self.breaks.breaks_before(element) and self.pages.page_has_content:
            self.pages.new_page()
        pending = [self.prefix(element)]
        for part in inline_parts(element):
            pending = self.consume(part, pending)
        self.pages.add_text("".join(pending))
        if self.breaks.breaks_after(element):
            self.pages.new_page()

    def consume(self, part: Part, pending: list[str]) -> list[str]:
        """Add one inline part; returns the text still waiting to be written."""
        if isinstance(part, str):
            pending.append(part)
            return pending
        if isinstance(part, ImageMark):
            self.pages.add_text("".join(pending))
            self.image(part.rid)
            return []
        if self.breaks.splits_at(part):
            self.pages.add_text("".join(pending))
            self.pages.new_page()
            return []
        return pending  # a break kind the active policy ignores

    # -- tables

    def table(self, element) -> None:
        images: list[str] = []
        grid = read_grid(element, lambda cell: self.cell_text(cell, images))
        self.pages.add_text(markdown_table(grid))
        for rid in images:  # screenshots inside cells follow the table
            self.image(rid)

    def cell_text(self, cell, images: list[str]) -> str:
        lines = []
        for paragraph in cell_paragraphs(cell):
            text, rids = split_parts(inline_parts(paragraph))
            images.extend(rids)
            if line := (self.prefix(paragraph) + text).strip():
                lines.append(line)
        return escape_cell("<br>".join(lines))


def split_parts(parts: list[Part]) -> tuple[str, list[str]]:
    """(all text, image ids) of a paragraph; breaks are ignored (tables stay on one page)."""
    text = "".join(p for p in parts if isinstance(p, str))
    return text, [p.rid for p in parts if isinstance(p, ImageMark)]


def read_pages(path: Path) -> Iterator[Page]:
    from docx import Document

    yield from WordReader(Document(str(path)), path.stem).read()
