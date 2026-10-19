"""Named rules that keep bad chunks out of the index (like quality.py for OCR).

Content rules find personal data and report positions only, never the
matched value, so the reports themselves cannot leak what they caught:
  thai_national_id  reject  13 digits whose check digit is valid
  phone_number      flag    Thai mobile (0[689] + 8 digits) or landline
                            (0[2-7] + 7 digits) numbers; kept, but listed
Numbers are found on a *scan view* of the text, not with one regex, so the
way a number is written cannot hide it:
  - any Unicode digit counts (Arabic, Thai ๐-๙, full-width ０-９)
  - HTML tags count as separators, so "<td>1</td><td>2345</td>" is one run
  - between two digits, up to 3 separator characters are allowed: spaces,
    line breaks, NBSP, zero-width characters, - – — . / | and ,
  - a run with more digits than the rule needs is not a match (long codes
    such as ECPS202000212 are left alone)
The pipeline scans each whole document before chunking, so a number that a
chunk split would cut in two is still caught (see pipeline.mark_pii_units),
and scans every chunk again afterwards.

Chunk rules (applied by the pipeline, named here so reports use one vocabulary):
  empty             drop    nothing left to embed after cleaning
  duplicate_text    drop    embed_text identical to an earlier chunk's
  duplicate_screenshots drop  a screenshot-only chunk whose screenshot set an
                            earlier chunk already has
Exclusion (whole documents, config.EXCLUDED_DOCUMENTS / --exclude) happens
before a file is read at all.
"""
from dataclasses import dataclass
import re
import unicodedata
from typing import Callable

REJECT, FLAG = "reject", "flag"
EMPTY, DUPLICATE_TEXT, DUPLICATE_SCREENSHOTS = "empty", "duplicate_text", "duplicate_screenshots"
THAI_ID, PHONE = "thai_national_id", "phone_number"

SEPARATORS = frozenset(" \t\r\n   ​‌‍⁠﻿-‐‑‒–—−./|,")
MAX_GAP = 3
_TAG = re.compile(r"<[^<>]{0,200}>")
TAG_FILL = "\x00"  # a removed tag: a separator that does not count towards the gap


def scan_view(text: str) -> str:
    """The text with HTML tags blanked out character by character (positions kept)."""
    return _TAG.sub(lambda m: TAG_FILL * len(m[0]), text)


class _RunScanner:
    """Collects digit runs; a run ends at a gap that is too long or not a separator."""

    def __init__(self):
        self.runs: list[tuple[str, int, int]] = []
        self.digits: list[str] = []
        self.start = self.last = -1
        self.gap, self.gap_ok = 0, True

    def digit(self, i: int, ch: str) -> None:
        if self.digits and (self.gap > MAX_GAP or not self.gap_ok):
            self.flush()
        if not self.digits:
            self.start = i
        self.digits.append(str(unicodedata.digit(ch)))
        self.last, self.gap, self.gap_ok = i, 0, True

    def other(self, ch: str) -> None:
        if ch == TAG_FILL:
            return  # a removed tag separates without counting towards the gap
        self.gap += 1
        self.gap_ok = self.gap_ok and ch in SEPARATORS

    def flush(self) -> None:
        if self.digits:
            self.runs.append(("".join(self.digits), self.start, self.last + 1))
        self.digits = []


def digit_runs(text: str) -> list[tuple[str, int, int]]:
    """(digits as ASCII, start, end) of every run of digits joined by short separator gaps."""
    scanner = _RunScanner()
    for i, ch in enumerate(scan_view(text)):
        if ch.isdigit():
            scanner.digit(i, ch)
        else:
            scanner.other(ch)
    scanner.flush()
    return scanner.runs


def valid_thai_id(text: str) -> bool:
    """Thai national ID check digit: (11 - sum(digit_i * (13 - i)) mod 11) mod 10."""
    digits = [int(unicodedata.digit(d)) for d in text if d.isdigit()]
    if len(digits) != 13:
        return False
    total = sum(d * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - total % 11) % 10 == digits[12]


def _is_phone(digits: str) -> bool:
    mobile = len(digits) == 10 and digits[:2] in ("06", "08", "09")
    landline = len(digits) == 9 and digits[0] == "0" and digits[1] in "234567"
    return mobile or landline


@dataclass(frozen=True)
class Finding:
    rule: str
    action: str
    field: str   # which text of the chunk
    start: int   # character offset in that text
    length: int  # the value itself is never stored


@dataclass(frozen=True)
class ContentRule:
    name: str
    action: str
    matches: Callable[[str], bool]  # on the run's ASCII digits


CONTENT_RULES = (
    ContentRule(THAI_ID, REJECT, valid_thai_id),
    ContentRule(PHONE, FLAG, _is_phone),
)


def spans(text: str) -> list[tuple[str, str, int, int]]:
    """(rule, action, start, end) of every finding in one text."""
    return [(rule.name, rule.action, start, end)
            for digits, start, end in digit_runs(text) for rule in CONTENT_RULES
            if rule.matches(digits)]


def scan(texts: dict[str, str]) -> list[Finding]:
    """Findings of every content rule in every named text (e.g. embed_text, display_text)."""
    return [Finding(rule, action, name, start, end - start)
            for name, text in texts.items() for rule, action, start, end in spans(text)]
