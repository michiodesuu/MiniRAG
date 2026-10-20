"""Table-of-contents pages and sections (heading paths).

Contents pages are dropped: their lines repeat every heading next to dot
leaders and match many queries badly. Their titles are still useful: in a
PDF manual they are the authority on which numbered lines are headings, so
"1.กดเลือกที่เมนู ติดตามสถานะ" (a step) is not mistaken for "1. สิทธิในการใช้งาน".

Sections come from one of two strategies, chosen by document type:
  MarkdownHeadings  Word: "#"/"##" lines; the section is the heading path
                    joined with " › " ("บทนำ › ขั้นตอนที่ 1").
  NumberedHeadings  PDF: a text-layer block that is only "<n>. <title>", where
                    the title is listed on the contents page; without a
                    contents page, any such line with a short title (fallback).
Units before the first heading belong to section "" (the document's front matter).
"""
from dataclasses import dataclass
import re
from typing import Protocol

from rag_ingest.chunking.units import Kind, Unit

MIN_TOC_LINES = 3
DOT_LEADER = "....."
_NUMBERED = re.compile(r"^(\d+)\.\s*(\S.*)$")
FALLBACK_TITLE_MAX_CHARS = 40
SECTION_SEPARATOR = " › "


def _squash(text: str) -> str:
    return " ".join(text.split())


def leader_title(line: str) -> str | None:
    """The text before a dot leader ("title ....... 12" -> "title"), or None.
    String operations, not a regex: long dot runs would make a regex backtrack."""
    rest = line.rstrip().rstrip("0123456789").rstrip()
    return rest.rstrip(".").strip() if rest.endswith(DOT_LEADER) else None


@dataclass
class TableOfContents:
    pages: set[int]
    titles: set[str]  # squashed titles without their number


def find_table_of_contents(units: list[Unit]) -> TableOfContents:
    leader_lines: dict[int, int] = {}
    for unit in units:
        if unit.kind == Kind.TEXT:
            leader_lines[unit.page] = leader_lines.get(unit.page, 0) + sum(
                1 for line in unit.text.splitlines() if leader_title(line) is not None)
    pages = {page for page, count in leader_lines.items() if count >= MIN_TOC_LINES}
    titles = set()
    for unit in (u for u in units if u.page in pages and u.kind == Kind.TEXT):
        entry = leader_title(_squash(unit.text))  # an entry may wrap: "1.\nสิทธิ ....."
        match = _NUMBERED.match(entry) if entry else None
        if match:
            titles.add(_squash(match[2]))
    return TableOfContents(pages, titles)


class SectionStrategy(Protocol):
    def heading(self, unit: Unit) -> tuple[int, str] | None:
        """(level, title) if the unit starts a section, else None."""


class MarkdownHeadings:
    def heading(self, unit: Unit) -> tuple[int, str] | None:
        if unit.kind != Kind.HEADING:
            return None
        return unit.level, unit.text.lstrip("#").strip()


class NumberedHeadings:
    def __init__(self, toc_titles: set[str]):
        self.toc_titles = toc_titles

    def heading(self, unit: Unit) -> tuple[int, str] | None:
        if unit.kind != Kind.TEXT or "\n" in unit.text:
            return None
        match = _NUMBERED.match(unit.text)
        if not match:
            return None
        title = _squash(match[2])
        known = title in self.toc_titles if self.toc_titles else len(title) <= FALLBACK_TITLE_MAX_CHARS
        return (1, f"{match[1]}. {title}") if known else None


def strategy_for(doc_type: str, toc: TableOfContents) -> SectionStrategy:
    return MarkdownHeadings() if doc_type == "docx" else NumberedHeadings(toc.titles)


def assign_sections(units: list[Unit], strategy: SectionStrategy) -> list[Unit]:
    """Each unit gets the heading path it falls under; headings become kind "heading"."""
    path: list[tuple[int, str]] = []
    out = []
    for unit in units:
        found = strategy.heading(unit)
        if found:
            level, title = found
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
            unit = unit.with_(kind=Kind.HEADING, level=level)
        out.append(unit.with_(section=SECTION_SEPARATOR.join(t for _, t in path)))
    return out
