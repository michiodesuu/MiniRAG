"""List numbers as Word displays them ("1.", "2.1", "ก)"), computed from numbering.xml.

Word stores only the list definition and each paragraph's (list, level); the
visible numbers are counted at render time, so they are re-counted here.
"""
from dataclasses import dataclass
import re

from rag_ingest.readers.word import ooxml as x

BULLET = "-"
LEVELS = 9
_PLACEHOLDER = re.compile(r"%([1-9])")
_ROMAN = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
        (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
_THAI_DIGITS = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
_THAI_LETTERS = "กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
_LATIN = "abcdefghijklmnopqrstuvwxyz"


def _roman(n: int) -> str:
    out = ""
    for value, letters in _ROMAN:
        count, n = divmod(n, value)
        out += letters * count
    return out


def _letters(n: int, alphabet: str) -> str:
    # Word repeats the letter after the alphabet ends: a..z, aa..zz, aaa...
    return alphabet[(n - 1) % len(alphabet)] * ((n - 1) // len(alphabet) + 1)


# w:numFmt -> formatter; anything unknown falls back to decimal.
FORMATS = {
    "decimal": str,
    "decimalZero": lambda n: f"{n:02d}",
    "lowerLetter": lambda n: _letters(n, _LATIN),
    "upperLetter": lambda n: _letters(n, _LATIN.upper()),
    "lowerRoman": _roman,
    "upperRoman": lambda n: _roman(n).upper(),
    "thaiNumbers": lambda n: str(n).translate(_THAI_DIGITS),
    "thaiLetters": lambda n: _letters(n, _THAI_LETTERS),
    "none": lambda _n: "",
}


def format_number(n: int, fmt: str) -> str:
    return FORMATS.get(fmt, str)(n)


@dataclass(frozen=True)
class Level:
    fmt: str
    text: str  # template, e.g. "%1." or "%1.%2"
    start: int


def _levels(abstract) -> dict[int, Level]:
    return {int(lvl.get(x.q("w:ilvl"))): Level(x.val(lvl, "w:numFmt") or "decimal",
                                x.val(lvl, "w:lvlText") or "",
                                int(x.val(lvl, "w:start") or 1))
            for lvl in abstract.iter(x.q("w:lvl"))}


def _start_overrides(num) -> dict[int, int]:
    return {int(o.get(x.q("w:ilvl"))): int(x.val(o, "w:startOverride"))
            for o in num.iter(x.q("w:lvlOverride")) if x.val(o, "w:startOverride")}


class Numbering:
    def __init__(self, numbering_element=None):
        self.abstract: dict[str, dict[int, Level]] = {}
        self.instances: dict[str, tuple[str | None, dict[int, int]]] = {}
        self.counters: dict[str, list[int | None]] = {}
        if numbering_element is not None:
            for abstract in numbering_element.iter(x.q("w:abstractNum")):
                self.abstract[abstract.get(x.q("w:abstractNumId"))] = _levels(abstract)
            for num in numbering_element.iter(x.q("w:num")):
                self.instances[num.get(x.q("w:numId"))] = (x.val(num, "w:abstractNumId"),
                                                        _start_overrides(num))

    def level(self, num_id: str, ilvl: int) -> Level | None:
        abstract_id, overrides = self.instances.get(num_id, (None, {}))
        level = self.abstract.get(abstract_id, {}).get(ilvl)
        if level is not None and ilvl in overrides:
            return Level(level.fmt, level.text, overrides[ilvl])
        return level

    def _advance(self, num_id: str, ilvl: int, start: int) -> list[int | None]:
        counts = self.counters.setdefault(num_id, [None] * LEVELS)
        counts[ilvl] = start if counts[ilvl] is None else counts[ilvl] + 1
        counts[ilvl + 1:] = [None] * (LEVELS - 1 - ilvl)  # a new item restarts its sub-levels
        return counts

    def label(self, num_id: str, ilvl: int) -> str | None:
        """Advance this list's counter and return the paragraph's label, e.g. "2.1"."""
        level = self.level(num_id, ilvl)
        if level is None:
            return None
        counts = self._advance(num_id, ilvl, level.start)
        if level.fmt == "bullet":
            return BULLET

        def fill(match: re.Match) -> str:
            i = int(match[1]) - 1
            source = self.level(num_id, i) or level
            return format_number(counts[i] if counts[i] is not None else source.start, source.fmt)
        return _PLACEHOLDER.sub(fill, level.text)
