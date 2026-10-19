"""Where Word pages start (Strategy). Word has no fixed pages, so page numbers come
from one of two sources, chosen once per document:

RenderedBreaks  the breaks Word saved from its last layout (w:lastRenderedPageBreak);
                closest to what a reader saw, present in files saved by Word
HardBreaks      fallback for generated files: explicit page breaks, "page break
                before" paragraphs and next-page section breaks
"""
from typing import Protocol

from rag_ingest.readers.word import ooxml as x
from rag_ingest.readers.word.inline import PageBreak, Part, RenderedBreak
from rag_ingest.readers.word.styles import StyleResolver

NEW_PAGE_SECTIONS = (None, "nextPage", "oddPage", "evenPage")  # None = Word's default


class BreakPolicy(Protocol):
    def splits_at(self, part: Part) -> bool: ...

    def breaks_before(self, paragraph) -> bool: ...

    def breaks_after(self, paragraph) -> bool: ...


class RenderedBreaks(BreakPolicy):
    def splits_at(self, part: Part) -> bool:
        return isinstance(part, RenderedBreak)

    def breaks_before(self, paragraph) -> bool:
        return False  # already reflected in the rendered breaks

    def breaks_after(self, paragraph) -> bool:
        return False


class HardBreaks(BreakPolicy):
    def __init__(self, styles: StyleResolver):
        self.styles = styles

    def splits_at(self, part: Part) -> bool:
        return isinstance(part, PageBreak)

    def breaks_before(self, paragraph) -> bool:
        return self.styles.flag(paragraph, "w:pageBreakBefore")

    def breaks_after(self, paragraph) -> bool:
        ppr = paragraph.find(x.PPR)
        section = None if ppr is None else ppr.find(x.q("w:sectPr"))
        return section is not None and x.val(section, "w:type") in NEW_PAGE_SECTIONS


def choose_policy(body, styles: StyleResolver) -> BreakPolicy:
    has_rendered = next(body.iter(x.RENDERED_BREAK), None) is not None
    return RenderedBreaks() if has_rendered else HardBreaks(styles)
