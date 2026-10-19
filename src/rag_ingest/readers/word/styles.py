"""Paragraph properties that may be set directly or inherited through the style chain."""
import re
from typing import Iterator

from rag_ingest.readers.word import ooxml as x

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)
BODY_TEXT_OUTLINE = 9  # w:outlineLvl 9 means "not a heading"


class StyleResolver:
    """Looks up paragraph styles once per style id (memoized) and answers
    "what is this paragraph" questions: heading level, list position, flags."""

    def __init__(self, document):
        self.document = document
        self._styles: dict[str | None, object] = {}

    def style(self, paragraph):
        style_id = x.val(paragraph.find(x.PPR), "w:pStyle")
        if style_id not in self._styles:
            from docx.enum.style import WD_STYLE_TYPE
            # None (no pStyle) resolves to the document's default paragraph style.
            self._styles[style_id] = self.document.part.get_style(style_id,
                                                                WD_STYLE_TYPE.PARAGRAPH)
        return self._styles[style_id]

    def _property_sources(self, paragraph) -> Iterator:
        """w:pPr elements, most specific first: the paragraph, then each style up the chain."""
        yield paragraph.find(x.PPR)
        style = self.style(paragraph)
        while style is not None:
            yield style.element.find(x.PPR)
            style = style.base_style

    def property(self, paragraph, child: str) -> str | None:
        return next((v for ppr in self._property_sources(paragraph)
                    if (v := x.val(ppr, child)) is not None), None)

    def flag(self, paragraph, child: str) -> bool:
        return self.property(paragraph, child) not in (None, "0", "false", "off")

    def heading_level(self, paragraph) -> int | None:
        style = self.style(paragraph)
        name = (style.name or "") if style is not None else ""
        if name.lower() == "title":
            return 1
        if match := _HEADING_STYLE.match(name):
            return int(match[1])
        outline = self.property(paragraph, "w:outlineLvl")
        if outline is not None and int(outline) < BODY_TEXT_OUTLINE:
            return int(outline) + 1
        return None

    def list_position(self, paragraph) -> tuple[str, int] | None:
        """(numId, level) if the paragraph is a list item, from itself or its style."""
        num_pr = next((n for ppr in self._property_sources(paragraph)
                    if ppr is not None and (n := ppr.find(x.q("w:numPr"))) is not None), None)
        if num_pr is None:
            return None
        num_id = x.val(num_pr, "w:numId")
        if num_id in (None, "0"):  # numId 0 explicitly removes inherited numbering
            return None
        return num_id, int(x.val(num_pr, "w:ilvl") or 0)
