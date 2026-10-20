"""Tables in ingest Markdown: Markdown tables (Word reader) and HTML tables
(Typhoon OCR inside screenshots). Parsing to rows is separate from rendering,
so a table can be rendered for the model in one of these forms:

  keep     a Markdown table: | h1 | h2 | / |---|---| / | v1 | v2 |
  flatten  one line per data row: "h1: v1, h2: v2" (empty cells left out)
  columns  only the header row, as a Markdown table without rows. For tables
           in screenshots (--screenshot-tables): a list screen shows sample
           records (people, branches, dates) that say nothing about how to
           use the system, crowd out the text that does, and are personal data.

Markdown is the one table format after ingest: Typhoon's HTML is converted,
Word tables already are Markdown.

and split by rows when it is larger than a chunk, with the header repeated.
"""
from dataclasses import dataclass
from html.parser import HTMLParser
import re

from rag_ingest.readers.word.tables import markdown_table

KEEP, FLATTEN, COLUMNS = "keep", "flatten", "columns"
TABLE_MODES = (KEEP, FLATTEN)                         # tables in the document's own text
SCREENSHOT_TABLE_MODES = (KEEP, FLATTEN, COLUMNS)     # tables in screenshot OCR
HTML_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.I | re.S)
_MD_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


class _RowsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._cell: list[str] | None = None
        self._span = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell, self._span = [], int(dict(attrs).get("colspan") or 1)
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self.rows[-1] += [text] + [""] * (self._span - 1)  # keep columns aligned
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_rows(html: str) -> list[list[str]]:
    parser = _RowsParser()
    parser.feed(html)
    return [row for row in parser.rows if row]


def markdown_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if _MD_SEPARATOR.match(line.strip()):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        rows.append([c.strip().replace("\\|", "|") for c in cells])
    return rows


def flatten_row(header: list[str], row: list[str]) -> str:
    pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
    return ", ".join(pairs)


@dataclass(frozen=True)
class TableLines:
    """A rendered table as separate lines, so it can be split by rows."""
    header: list[str]  # lines to repeat at the top of every part ([] for flatten)
    rows: list[str]


def render(rows: list[list[str]], mode: str) -> TableLines:
    if not rows:
        return TableLines([], [])
    if mode == FLATTEN:
        header = rows[0]
        body = rows[1:] or rows
        return TableLines([], [line for r in body if (line := flatten_row(header, r))])
    lines = markdown_table(rows[:1] if mode == COLUMNS else rows).splitlines()
    return TableLines(lines[:2], lines[2:])
