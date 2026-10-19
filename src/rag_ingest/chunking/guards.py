"""Named rules that keep bad chunks out of the index (like quality.py for OCR).

Content rules scan a chunk's text and report positions only, never the
matched value, so the reports themselves cannot leak what they caught:
  thai_national_id  reject  13 digits (optionally "x-xxxx-xxxxx-xx-x", Arabic or
                            Thai digits) whose check digit is valid
  phone_number      flag    Thai mobile (06/08/09 + 8 digits) or landline
                            (02-07 + 7 digits) numbers; kept, but listed
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
from typing import Callable

REJECT, FLAG = "reject", "flag"
EMPTY, DUPLICATE_TEXT, DUPLICATE_SCREENSHOTS = "empty", "duplicate_text", "duplicate_screenshots"
THAI_ID, PHONE = "thai_national_id", "phone_number"

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_D = "[0-9๐-๙]"
_NOT_DIGIT_BEFORE, _NOT_DIGIT_AFTER = f"(?<!{_D})", f"(?!{_D})"
_SEP = "[- ]?"
THAI_ID_PATTERN = re.compile(
    f"{_NOT_DIGIT_BEFORE}{_D}{_SEP}{_D}{{4}}{_SEP}{_D}{{5}}{_SEP}{_D}{{2}}{_SEP}{_D}{_NOT_DIGIT_AFTER}")
PHONE_PATTERN = re.compile(
    f"{_NOT_DIGIT_BEFORE}(?:0[689]{_D}{_SEP}{_D}{{3}}{_SEP}{_D}{{4}}"
    f"|0[2-7]{_SEP}{_D}{{3}}{_SEP}{_D}{{4}}){_NOT_DIGIT_AFTER}")


def valid_thai_id(text: str) -> bool:
    """Thai national ID check digit: 11 - (sum of digit_i * (13 - i)) mod 11, last digit of that."""
    digits = [int(d) for d in text.translate(_THAI_DIGITS) if d.isdigit()]
    if len(digits) != 13:
        return False
    total = sum(d * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - total % 11) % 10 == digits[12]


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
    spans: Callable[[str], list[tuple[int, int]]]


def _thai_ids(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in THAI_ID_PATTERN.finditer(text) if valid_thai_id(m[0])]


def _phones(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in PHONE_PATTERN.finditer(text)]


CONTENT_RULES = (
    ContentRule(THAI_ID, REJECT, _thai_ids),
    ContentRule(PHONE, FLAG, _phones),
)


def scan(texts: dict[str, str]) -> list[Finding]:
    """Findings of every content rule in every named text (e.g. embed_text, display_text)."""
    return [Finding(rule.name, rule.action, name, start, end - start)
            for name, text in texts.items() for rule in CONTENT_RULES
            for start, end in rule.spans(text)]
