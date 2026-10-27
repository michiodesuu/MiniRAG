"""Named rules that keep bad chunks out of the index (like quality.py for OCR).

Content rules find personal data and report positions only, never the
matched value, so the reports themselves cannot leak what they caught:
  thai_national_id  reject  13 digits whose check digit is valid
  phone_number      flag    Thai mobile (0[689] + 8 digits) or landline
                            (0[2-7] + 7 digits) numbers; kept, but listed

How numbers are found, so the way a number is written cannot hide it:
  1. scan view: HTML tags are blanked out ("<td>1</td><td>2345</td>" reads as
     "1 2345"); any Unicode digit counts (Arabic, Thai ๐-๙, full-width ０-９)
  2. digit *groups* (unbroken digits) are chained when only separators lie
     between them (spaces, line breaks, NBSP, zero-width characters,
     - – — . / | ,), at most MAX_GAP characters, tags not counted
  3. candidates are runs of whole consecutive groups of a chain that total the
     rule's length, so an ID stuck to another number ("<ID>, 0812345678",
     "<ID> 25") is still found; one unbroken run longer than the rule
     (e.g. ECPS202000212) is a code, not a match
  4. an ID candidate is either
       a) a whole chain of exactly 13 digits, grouped any way
          ("3101 7001 2345 9", one digit per group, split across cells), or
       b) whole consecutive groups inside a longer chain that total 13 and
          are laid out like an ID: up to 3 groups split anywhere, the printed
          1-4-5-2-1 layout (or merged parts of it), or one digit per group,
          so an ID next to another number ("<ID>, 081-234-5678") is found;
     then its check digit must be valid. (a) and (b) together catch
     everything earlier versions of this rule caught. Without the layout
     limit in (b), rows of dates and amounts ("12/3/2565 | 1,234.50")
     produced false IDs in ~40% of rows. Known gap: an ID in an unusual
     grouping of 4+ parts directly next to another number.
The pipeline scans each whole document before chunking, so a number that a
chunk split would cut in two is still caught (pipeline.mark_pii_units), and
scans every chunk again afterwards.

Names are redacted, not rejected: a Thai title (นาย, นาง, นางสาว, น.ส., ...) plus the
word after it, and one more word if a space follows (a surname), becomes
NAME_REDACTED in both embed_text and display_text, so neither the vectors nor the
databases hold the name. Thai has no spaces, so a title is also the start of
ordinary words (นายจ้าง = employer): NOT_NAMES lists the ones to leave alone.
shortcut: a name without a title, or an ordinary word not in NOT_NAMES, is
missed or caught wrongly; upgrade to a Thai NER model if that matters.

Chunk rules (applied by the pipeline, named here so reports use one vocabulary):
  empty             drop    nothing left to embed after cleaning
  duplicate_text    drop    embed_text identical to an earlier chunk's
  duplicate_screenshots drop  a screenshot-only chunk whose screenshot set an
                            earlier chunk already has
Exclusion (whole documents: by file name, and by the image fingerprints of the
excluded files) happens before a document's text is read.
"""
from dataclasses import dataclass
import re
import unicodedata
from typing import Callable

REJECT, FLAG = "reject", "flag"
EMPTY, DUPLICATE_TEXT, DUPLICATE_SCREENSHOTS = "empty", "duplicate_text", "duplicate_screenshots"
THAI_ID, PHONE = "thai_national_id", "phone_number"

SEPARATORS = frozenset(" \t\r\n   ​‌‍⁠﻿-‐‑‒–—−./|,")
MAX_GAP = 5
_TAG = re.compile(r"<[^<>]{0,200}>")
TAG_FILL = "\x00"  # a removed tag: a separator that does not count towards the gap
ID_LENGTH = 13
PHONE_LENGTHS = (9, 10)

Group = tuple[str, int, int]  # (ASCII digits, start, end) of unbroken digits


def scan_view(text: str) -> str:
    """The text with HTML tags blanked out character by character (positions kept)."""
    return _TAG.sub(lambda m: TAG_FILL * len(m[0]), text)


class _ChainScanner:
    """Splits text into chains of digit groups joined by short separator gaps."""

    def __init__(self):
        self.chains: list[list[Group]] = []
        self.chain: list[Group] = []
        self.digits: list[str] = []
        self.start = -1
        self.gap, self.gap_ok = 0, True

    def digit(self, i: int, ch: str) -> None:
        if not self.digits:
            self._open_group(i)
        self.digits.append(str(unicodedata.digit(ch)))

    def other(self, i: int, ch: str) -> None:
        if self.digits:
            self.chain.append(("".join(self.digits), self.start, i))
            self.digits, self.gap, self.gap_ok = [], 0, True
        if ch != TAG_FILL:  # a removed tag separates without counting towards the gap
            self.gap += 1
            self.gap_ok = self.gap_ok and ch in SEPARATORS

    def _open_group(self, i: int) -> None:
        if self.chain and (self.gap > MAX_GAP or not self.gap_ok):
            self.chains.append(self.chain)
            self.chain = []
        self.start = i

    def finish(self, end: int) -> list[list[Group]]:
        if self.digits:
            self.chain.append(("".join(self.digits), self.start, end))
        if self.chain:
            self.chains.append(self.chain)
        return self.chains


def digit_chains(text: str) -> list[list[Group]]:
    view = scan_view(text)
    scanner = _ChainScanner()
    for i, ch in enumerate(view):
        if ch.isdigit():
            scanner.digit(i, ch)
        else:
            scanner.other(i, ch)
    return scanner.finish(len(view))


def candidates(chain: list[Group], lengths: tuple[int, ...]) -> list[tuple[list[Group], int, int]]:
    """Runs of whole consecutive groups whose digits total one of `lengths`."""
    out = []
    for i in range(len(chain)):
        total = 0
        for j in range(i, len(chain)):
            total += len(chain[j][0])
            if total in lengths:
                out.append((chain[i:j + 1], chain[i][1], chain[j][2]))
            if total >= max(lengths):
                break
    return out


def valid_thai_id(text: str) -> bool:
    """Thai national ID check digit: (11 - sum(digit_i * (13 - i)) mod 11) mod 10."""
    digits = [int(unicodedata.digit(d)) for d in text if d.isdigit()]
    if len(digits) != ID_LENGTH:
        return False
    total = sum(d * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - total % 11) % 10 == digits[12]


ID_GROUP_BOUNDARIES = frozenset({1, 5, 10, 12})  # after digit 1, 5, 10, 12 in "1-2345-67890-12-3"
ID_FREE_SPLIT_GROUPS = 3  # up to this many groups may split an ID anywhere


def id_layout(groups: list[Group]) -> bool:
    """Up to 3 groups split anywhere, the printed 1-4-5-2-1 layout (or merged parts
    of it), or one digit per group."""
    sizes = [len(g[0]) for g in groups]
    boundaries = {sum(sizes[:k]) for k in range(1, len(sizes))}
    return (len(sizes) <= ID_FREE_SPLIT_GROUPS or boundaries <= ID_GROUP_BOUNDARIES
            or all(size == 1 for size in sizes))


def _is_id(groups: list[Group], whole_chain: bool) -> bool:
    return (whole_chain or id_layout(groups)) and valid_thai_id("".join(g[0] for g in groups))


def _is_phone_candidate(groups: list[Group], _whole_chain: bool) -> bool:
    return _is_phone(groups)


def _is_phone(groups: list[Group]) -> bool:
    digits = "".join(g[0] for g in groups)
    mobile = len(digits) == 10 and digits[:2] in ("06", "08", "09")
    landline = len(digits) == 9 and digits[0] == "0" and digits[1] in "234567"
    return mobile or landline


NAME_REDACTED = "[ชื่อ]"
NAME_TITLES = ("นางสาว", "นายแพทย์", "นาย", "นาง", "น.ส.", "ด.ช.", "ด.ญ.", "เด็กชาย", "เด็กหญิง")
NOT_NAMES = ("จ้าง", "ทะเบียน", "อำเภอ", "กรัฐมนตรี", "หน้า", "ทุน", "ช่าง", "พยาบาล", "แบบ", "สาว", "แพทย์")
_NAME = re.compile(
    rf"(?:{'|'.join(map(re.escape, NAME_TITLES))})[ \t]?(?!{'|'.join(NOT_NAMES)})[ก-๙]{{2,}}(?:[ \t][ก-๙]{{2,}})?")


def redact_names(text: str) -> tuple[str, int]:
    """(text with every name replaced by NAME_REDACTED, how many were replaced)."""
    return _NAME.subn(NAME_REDACTED, text)


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
    lengths: tuple[int, ...]
    matches: Callable[[list[Group], bool], bool]  # (groups, they are the whole chain)


CONTENT_RULES = (
    ContentRule(THAI_ID, REJECT, (ID_LENGTH,), _is_id),
    ContentRule(PHONE, FLAG, PHONE_LENGTHS, _is_phone_candidate),
)


def spans(text: str) -> list[tuple[str, str, int, int]]:
    """(rule, action, start, end) of every finding in one text."""
    return [(rule.name, rule.action, start, end)
            for chain in digit_chains(text) for rule in CONTENT_RULES
            for groups, start, end in candidates(chain, rule.lengths)
            if rule.matches(groups, len(groups) == len(chain))]


def scan(texts: dict[str, str]) -> list[Finding]:
    """Findings of every content rule in every named text (e.g. embed_text, display_text)."""
    return [Finding(rule, action, name, start, end - start)
            for name, text in texts.items() for rule, action, start, end in spans(text)]
