"""Restore Thai vowels that a PDF's font tables mislabel.

Word draws "ำ" as two glyphs: a small circle, then the "า" glyph. A PDF font
table gives each glyph a single label, so exported files often label the
circle as a space (or U+0E4D) and, in some fonts, the "า" glyph as "ำ". The
text then reads "ส ำนัก" for "สำนัก" and "เจ้ำหน้ำที่" for "เจ้าหน้าที่",
although the page looks right. No Thai word starts with "า" or "ำ", so:

  - circle + vowel right after a consonant or mark      -> "ำ"
  - any other "ำ" in a font proven to label "า" as "ำ"  -> "า"

A font is proven defective when a space-labelled glyph is drawn directly
before its "ำ"-labelled glyph, which real Thai text never does. The proof
comes from the PDF's glyphs (PyMuPDF texttrace), so correct fonts are never
touched. Used by the PDF reader (ingest) and the PDF -> DOCX converter.
"""
from dataclasses import dataclass
import re

AA, AM, NIKHAHIT = "า", "ำ", "ํ"
_CONSONANT = re.compile("[ก-ฮ]")
_MARK = re.compile("[ัิ-ฺ็-๎]")  # above/below vowels, tone marks
_STYLE_SUFFIX = re.compile(r"[,-]([A-Za-z]+)$")


@dataclass(frozen=True)
class FontKey:
    family: str
    bold: bool


def font_key(name: str) -> FontKey:
    """"ABCDEE+TH SarabunPSK,Bold" -> FontKey("TH SarabunPSK", True)."""
    name = name.split("+", 1)[-1]
    style = _STYLE_SUFFIX.search(name)
    family = name[:style.start()] if style else name
    return FontKey(family.strip(), bool(style) and "bold" in style[1].lower())


def fonts_labelling_aa_as_am(doc) -> set[FontKey]:
    """Fonts of an open PyMuPDF document whose "า" glyph is labelled "ำ"."""
    defective: set[FontKey] = set()
    for page in doc:
        drawn = [(chr(c[0]), span["font"]) for span in page.get_texttrace() for c in span["chars"]]
        for (before, _), (char, font) in zip(drawn, drawn[1:]):
            if char == AM and before == " ":
                defective.add(font_key(font))
    return defective


def _follows_syllable(flat: list[tuple[str, int, bool]], k: int) -> bool:
    """flat[k] comes right after a consonant, possibly with marks on it ("น", "น้")."""
    i = k - 1
    while i >= 0 and _MARK.match(flat[i][0]):
        i -= 1
    return i >= 0 and bool(_CONSONANT.match(flat[i][0]))


def _circle_length(flat: list[tuple[str, int, bool]], k: int) -> int:
    """How many characters at flat[k:] form "circle + vowel" of a split "ำ" (0 if none)."""
    if not _follows_syllable(flat, k):
        return 0
    j = k
    if flat[j][0] == NIKHAHIT:
        j += 1
    if j < len(flat) and flat[j][0] == " ":
        j += 1
    if j == k or j >= len(flat) or flat[j][0] not in (AA, AM):
        return 0
    return j - k + 1


def repair_thai_vowels(pieces: list[tuple[str, bool]]) -> tuple[list[str], int, int]:
    """pieces: (text, font labels "า" as "ำ") for consecutive runs of one line or paragraph.
    Returns (new texts, "ำ" restored, "า" restored). A split "ำ" may span two pieces;
    the repaired vowel stays in the piece that held the vowel."""
    flat = [(ch, i, bad) for i, (text, bad) in enumerate(pieces) for ch in text]
    out: list[list[str]] = [[] for _ in pieces]
    am = aa = k = 0
    while k < len(flat):
        ch, i, bad = flat[k]
        length = _circle_length(flat, k)
        if length:
            out[flat[k + length - 1][1]].append(AM)
            am, k = am + 1, k + length
            continue
        if ch == AM and bad:
            ch, aa = AA, aa + 1
        out[i].append(ch)
        k += 1
    return ["".join(chars) for chars in out], am, aa


def line_text(line: dict, defective: set[FontKey]) -> str:
    """One PyMuPDF text line ({"spans": [{"text", "font"}, ...]}) with vowels repaired."""
    pieces = [(span["text"], font_key(span["font"]) in defective) for span in line["spans"]]
    return "".join(repair_thai_vowels(pieces)[0])
