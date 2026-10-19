"""What a unit contributes to `embed_text` (the model's input) and `display_text`.

embed form of a unit:
  - Unicode NFC (Thai can arrive decomposed, e.g. "ํา" for "ำ")
  - no page comments and no screenshot tags (the units already exclude them)
  - tables rendered per --tables (tables.py)
  - placeholders and already-embedded duplicate screenshots contribute nothing
Each unit becomes a list of Segments: plain lines, or table rows that carry
the header to repeat when a split starts inside the table.
"""
from dataclasses import dataclass
import re
import unicodedata

from rag_ingest.chunking import tables
from rag_ingest.chunking.units import Kind, Unit

SCREENSHOT_OPEN, SCREENSHOT_CLOSE = "[ภาพหน้าจอ]", "[/ภาพหน้าจอ]"


@dataclass(frozen=True)
class Segment:
    text: str
    header: tuple[str, ...] = ()  # table header lines to repeat if a part starts here


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _table_segments(rows: list[list[str]], mode: str) -> list[Segment]:
    rendered = tables.render(rows, mode)
    out = [Segment(line) for line in rendered.header]
    out += [Segment(line, tuple(rendered.header)) for line in rendered.rows]
    return out


def _line_segments(text: str, mode: str) -> list[Segment]:
    """Plain lines; runs of "|" lines are a Markdown table."""
    out: list[Segment] = []
    run: list[str] = []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            run.append(line)
            continue
        if run:
            out += _table_segments(tables.markdown_rows("\n".join(run)), mode)
            run = []
        if line.strip():
            out.append(Segment(line))
    return out


def _mixed_segments(text: str, mode: str) -> list[Segment]:
    """Lines plus any HTML tables among them (Typhoon OCR inside screenshots)."""
    out, cursor = [], 0
    for match in tables.HTML_TABLE_BLOCK.finditer(text):
        out += _line_segments(text[cursor:match.start()], mode)
        out += _table_segments(tables.html_rows(match[0]), mode)
        cursor = match.end()
    return out + _line_segments(text[cursor:], mode)


def embed_segments(unit: Unit, mode: str) -> list[Segment]:
    if unit.kind == Kind.PLACEHOLDER or unit.duplicate_of_md5:
        return []
    return _mixed_segments(nfc(unit.text), mode)


def display_form(unit: Unit) -> str:
    """The unit as it appears in the ingest Markdown."""
    if unit.kind == Kind.SCREENSHOT:
        return f"{SCREENSHOT_OPEN}\n{unit.text}\n{SCREENSHOT_CLOSE}"
    return unit.text


def context_header(document: str, section: str) -> str:
    return f"{document} › {section}" if section else document


def squash_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
