"""Inline content of one paragraph: text, breaks and images, in reading order.

The walker visits the paragraph's XML with a dispatch table (tag -> handler);
unknown tags are containers (runs, hyperlinks, insertions...) and are walked
into. Text-box content is collected separately and appended after the
paragraph, so callouts do not split the sentence they are anchored in.
"""
from dataclasses import dataclass

from rag_ingest.readers.word import ooxml as x


class PageBreak:
    """A hard page break (w:br type="page")."""

    def __repr__(self) -> str:
        return "PAGE_BREAK"


class RenderedBreak:
    """Where Word's last layout started a new page."""

    def __repr__(self) -> str:
        return "RENDERED_BREAK"


@dataclass(frozen=True)
class ImageMark:
    rid: str  # relationship id of the embedded image part


Part = str | PageBreak | RenderedBreak | ImageMark
PAGE_BREAK, RENDERED_BREAK = PageBreak(), RenderedBreak()

# Never text: properties, deleted runs, field instructions, symbol-font glyphs.
SKIPPED = frozenset(map(x.q, ["w:pPr", "w:rPr", "w:tblPr", "w:tcPr", "w:trPr", "w:sdtPr",
                              "w:del", "w:delText", "w:instrText", "w:sym", "w:moveFrom"]))


class InlineWalker:
    def __init__(self, collect_text_boxes: bool = True):
        self.parts: list[Part] = []
        self.text_boxes: list = []
        self.collect_text_boxes = collect_text_boxes
        self.handlers = {
            x.T: self.text,
            x.TAB: lambda _: self.parts.append("\t"),
            x.CR: lambda _: self.parts.append("\n"),
            x.BR: self.line_break,
            x.RENDERED_BREAK: lambda _: self.parts.append(RENDERED_BREAK),
            x.TXBX: self.text_box,
            x.ALTERNATE: self.alternate_content,
            x.BLIP: lambda el: self.image(el.get(x.R_EMBED)),
            x.IMAGEDATA: lambda el: self.image(el.get(x.R_ID)),
        }

    def walk(self, element) -> None:
        for child in element:
            # Comments and processing instructions have non-string tags.
            if isinstance(child.tag, str) and child.tag not in SKIPPED:
                self.handlers.get(child.tag, self.walk)(child)

    def text(self, element) -> None:
        self.parts.append(element.text or "")

    def line_break(self, element) -> None:
        kind = element.get(x.TYPE)
        if kind == "page":
            self.parts.append(PAGE_BREAK)
        elif kind != "column":
            self.parts.append("\n")

    def image(self, rid: str | None) -> None:
        if rid:
            self.parts.append(ImageMark(rid))

    def text_box(self, element) -> None:
        if self.collect_text_boxes:
            self.text_boxes.append(element)

    def alternate_content(self, element) -> None:
        # Word stores some drawings twice (modern + fallback for old Word); read one.
        chosen = element.find(x.CHOICE)
        if chosen is None:
            chosen = element.find(x.FALLBACK)
        if chosen is not None:
            self.walk(chosen)


def inline_parts(paragraph) -> list[Part]:
    """Parts of one w:p; text-box paragraphs follow, each on a new line, without breaks."""
    walker = InlineWalker()
    walker.walk(paragraph)
    parts = walker.parts
    for box in walker.text_boxes:
        for box_paragraph in box.iter(x.P):  # nested boxes are flattened into this one
            inner = InlineWalker(collect_text_boxes=False)
            inner.walk(box_paragraph)
            parts.append("\n")
            parts.extend(p for p in inner.parts if not isinstance(p, (PageBreak, RenderedBreak)))
    return parts
