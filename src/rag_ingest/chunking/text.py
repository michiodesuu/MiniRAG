"""What a unit contributes to `embed_text` (the model's input) and `display_text`.

embed form of a unit:
  - Unicode NFC (Thai can arrive decomposed, e.g. "ํา" for "ำ")
  - no page comments and no screenshot tags (the units already exclude them)
  - tables rendered as Markdown per TableFormat (tables.py): --tables for the
    document's own tables, --screenshot-tables for tables in screenshot OCR
  - screenshot OCR tidied: Typhoon's Markdown heading marks ("### ") and
    checkbox glyphs (☐ ☑ ☒) removed; they carry no meaning for search
  - placeholders and already-embedded duplicate screenshots contribute nothing
Each unit becomes a list of Segments: plain lines, or table rows that carry
the header to repeat when a split starts inside the table.

A screenshot's display form is the same rendering inside the screenshot tags,
so what is shown (and checked for personal data) matches what was embedded.
"""
from dataclasses import dataclass
from typing import Callable
import re
import unicodedata

from rag_ingest.chunking import tables
from rag_ingest.chunking.units import Kind, Unit

SCREENSHOT_OPEN, SCREENSHOT_CLOSE = "[ภาพหน้าจอ]", "[/ภาพหน้าจอ]"
_HEADING_MARK = re.compile(r"^#{1,6}\s+")
_CHECKBOX = re.compile("[☐☑☒✓✔✗✘]")
Tidy = Callable[[str], str] | None


@dataclass(frozen=True)
class TableFormat:
    """How tables are rendered: in the document's own text, and in screenshot OCR."""
    document: str = tables.KEEP
    screenshot: str = tables.KEEP


@dataclass(frozen=True)
class Segment:
    text: str
    header: tuple[str, ...] = ()  # table header lines to repeat if a part starts here


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tidy_ocr(text: str) -> str:
    """One line or cell of screenshot OCR without heading marks or checkbox glyphs."""
    return " ".join(_CHECKBOX.sub(" ", _HEADING_MARK.sub("", text.strip())).split())


def _table_segments(rows: list[list[str]], mode: str, tidy: Tidy) -> list[Segment]:
    if tidy:
        rows = [[tidy(cell) for cell in row] for row in rows]
    rendered = tables.render(rows, mode)
    out = [Segment(line) for line in rendered.header]
    out += [Segment(line, tuple(rendered.header)) for line in rendered.rows]
    return out


def _line_segments(text: str, mode: str, tidy: Tidy) -> list[Segment]:
    """Plain lines; runs of "|" lines are a Markdown table."""
    out: list[Segment] = []
    run: list[str] = []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            run.append(line)
            continue
        if run:
            out += _table_segments(tables.markdown_rows("\n".join(run)), mode, tidy)
            run = []
        line = tidy(line) if tidy else line
        if line.strip():
            out.append(Segment(line))
    return out


def _mixed_segments(text: str, mode: str, tidy: Tidy) -> list[Segment]:
    """Lines plus any HTML tables among them (Typhoon OCR inside screenshots)."""
    out, cursor = [], 0
    for match in tables.HTML_TABLE_BLOCK.finditer(text):
        out += _line_segments(text[cursor:match.start()], mode, tidy)
        out += _table_segments(tables.html_rows(match[0]), mode, tidy)
        cursor = match.end()
    return out + _line_segments(text[cursor:], mode, tidy)


def unit_segments(unit: Unit, fmt: TableFormat) -> list[Segment]:
    """The unit rendered for the model (duplicates included; see embed_segments)."""
    if unit.kind == Kind.PLACEHOLDER:
        return []
    if unit.kind == Kind.SCREENSHOT:
        return _mixed_segments(nfc(unit.text), fmt.screenshot, tidy_ocr)
    return _mixed_segments(nfc(unit.text), fmt.document, None)


def embed_segments(unit: Unit, fmt: TableFormat) -> list[Segment]:
    return [] if unit.duplicate_of_md5 else unit_segments(unit, fmt)


def display_form(unit: Unit, segments: list[Segment] | None = None) -> str:
    """The unit as shown with a search hit. A screenshot shows its rendered
    segments when given (else its text as is), inside the screenshot tags."""
    if unit.kind == Kind.SCREENSHOT:
        body = unit.text if segments is None else "\n".join(s.text for s in segments)
        return f"{SCREENSHOT_OPEN}\n{body}\n{SCREENSHOT_CLOSE}"
    return unit.text


def context_header(document: str, section: str) -> str:
    return f"{document} › {section}" if section else document


def squash_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
