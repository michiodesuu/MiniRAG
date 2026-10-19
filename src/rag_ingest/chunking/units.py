"""Parse one ingest Markdown file into ordered units.

The ingest format (see ingest.py):
  <!-- page N -->                         starts page N
  blocks separated by a blank line        text-layer blocks / Word paragraphs,
                                          "#" headings, Markdown tables
  [ภาพหน้าจอ] ... [/ภาพหน้าจอ]            screenshot OCR (may contain blank lines,
                                          Markdown and HTML tables)
  [ภาพหน้าจอ: ยังไม่ได้ OCR (<stem>)]       screenshot without usable OCR

Every unit keeps its character range in the file, so a chunk can cite the
exact span it came from.
"""
from dataclasses import dataclass, field, replace
import re

PAGE = re.compile(r"<!-- page (\d+) -->")
SCREENSHOT = re.compile(r"\[ภาพหน้าจอ\]\n(.*?)\n\[/ภาพหน้าจอ\]", re.S)
PLACEHOLDER = re.compile(r"\[ภาพหน้าจอ: ยังไม่ได้ OCR \(([^)]*)\)\]")
_SPECIAL = re.compile("|".join(f"(?:{p.pattern})" for p in (PAGE, SCREENSHOT, PLACEHOLDER)), re.S)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
HTML_TABLE = re.compile(r"<table\b", re.I)


class Kind:
    """Unit kinds (spec 3.1)."""
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    SCREENSHOT = "screenshot"
    PLACEHOLDER = "placeholder"


@dataclass(frozen=True)
class Unit:
    kind: str
    text: str                 # the unit's Markdown, without screenshot tags
    page: int
    start: int                # character range in the .md file
    end: int
    section: str = ""
    level: int = 0            # heading depth ("#" = 1); 0 for non-headings
    md5: str | None = None    # screenshots: image bytes hash, from the ingest manifest
    ocr_source: str | None = None
    image: str | None = None  # placeholders: the image stem waiting for OCR
    duplicate_of_md5: bool = False  # screenshot already embedded earlier in the corpus
    pii: bool = False               # a rejected personal number overlaps this unit

    def with_(self, **changes) -> "Unit":
        return replace(self, **changes)


@dataclass
class ParsedDocument:
    units: list[Unit] = field(default_factory=list)


def _block_kind(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if all(line.lstrip().startswith("|") for line in lines):
        return Kind.TABLE, 0
    if HTML_TABLE.search(text):
        return Kind.TABLE, 0
    heading = re.match(r"^(#{1,6})\s+\S", text)
    if heading and len(lines) == 1:
        return Kind.HEADING, len(heading[1])
    return Kind.TEXT, 0


def _text_blocks(text: str, offset: int, page: int) -> list[Unit]:
    """Blank-line separated blocks of plain text between special markers."""
    separators = list(_BLANK_LINE.finditer(text))
    starts = [0] + [m.end() for m in separators]
    ends = [m.start() for m in separators] + [len(text)]
    units = []
    for start, end in zip(starts, ends):
        raw = text[start:end]
        block = raw.strip()
        if not block:
            continue
        absolute = offset + start + len(raw) - len(raw.lstrip())
        kind, level = _block_kind(block)
        units.append(Unit(kind, block, page, absolute, absolute + len(block), level=level))
    return units


def _special_unit(match: re.Match, page: int) -> Unit | None:
    whole = match[0]
    if SCREENSHOT.fullmatch(whole):
        body = SCREENSHOT.fullmatch(whole)[1]
        return Unit(Kind.SCREENSHOT, body, page, match.start(), match.end())
    if PLACEHOLDER.fullmatch(whole):
        return Unit(Kind.PLACEHOLDER, whole, page, match.start(), match.end(),
                    image=PLACEHOLDER.fullmatch(whole)[1])
    return None


def parse_markdown(text: str) -> ParsedDocument:
    doc, page, cursor = ParsedDocument(), 0, 0
    for match in _SPECIAL.finditer(text):
        doc.units += _text_blocks(text[cursor:match.start()], cursor, page)
        if page_match := PAGE.fullmatch(match[0]):
            page = int(page_match[1])
        elif unit := _special_unit(match, page):
            doc.units.append(unit)
        cursor = match.end()
    doc.units += _text_blocks(text[cursor:], cursor, page)
    return doc
