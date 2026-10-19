"""Thai vowel repair shared by the PDF reader and the PDF -> DOCX converter."""
from pathlib import Path

import pytest

from rag_ingest.thai_glyphs import FontKey, font_key, line_text, repair_thai_vowels

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "data" / "excluded" / "1.คู่มือเจ้าหน้าที่วินิจฉัยเงินทดแทน.pdf"


@pytest.mark.parametrize("pieces, bad_font, expected, restored", [
    # font labels the "า" glyph as "ำ"; the circle of a real "ำ" reads as a space
    (["ส ำนักงำน เจ้ำหน้ำที่ น้ ำ"], True, ["สำนักงาน เจ้าหน้าที่ น้ำ"], (2, 3)),
    # correct font, but the circle still reads as a space
    (["ให้ท าการ"], False, ["ให้ทำการ"], (1, 0)),
    (["ส่งค าขอ"], False, ["ส่งคำขอ"], (1, 0)),
    # the split falls between two runs; the vowel stays in its own run
    (["กำรท", " ำงำน"], True, ["การท", "ำงาน"], (1, 2)),
    # circle labelled as U+0E4D instead of a space
    (["สํา"], False, ["สำ"], (1, 0)),
    # correct text is left alone
    (["ทำการ น้ำ กดปุ่ม ถัดไป ABC 123"], False, ["ทำการ น้ำ กดปุ่ม ถัดไป ABC 123"], (0, 0)),
    # a space before a word is never mistaken for the circle
    (["การ ทำ"], True, ["การ ทา"], (0, 1)),
    # a stray mark not attached to a consonant is not a syllable
    (["น ้ า"], False, ["น ้ า"], (0, 0)),
])
def test_repair_thai_vowels(pieces, bad_font, expected, restored):
    texts, am, aa = repair_thai_vowels([(p, bad_font) for p in pieces])
    assert texts == expected and (am, aa) == restored


def test_font_key():
    assert font_key("ABCDEE+TH SarabunPSK,Bold") == FontKey("TH SarabunPSK", True)
    assert font_key("TH SarabunPSK") == FontKey("TH SarabunPSK", False)
    assert font_key("Arial-BoldMT") == FontKey("Arial", True)


def test_line_text_uses_each_spans_font():
    line = {"spans": [{"text": "เจ้ำหน้ำที่ ", "font": "TH SarabunPSK"},
                      {"text": "ทำการ", "font": "TH SarabunPSK,Bold"}]}
    assert line_text(line, {FontKey("TH SarabunPSK", False)}) == "เจ้าหน้าที่ ทำการ"


@pytest.mark.skipif(not MANUAL.exists(), reason="sample manual not present")
def test_manual_vowels_match_its_glyphs():
    """Glyph analysis of this PDF: 44 drawn "ำ" (circle + aa glyph), 381 drawn "า"."""
    from rag_ingest.readers.pdf import read_pages

    text = "".join(b.text for page in read_pages(MANUAL) for b in page.blocks)
    assert (text.count("ำ"), text.count("า")) == (44, 381)
    assert "เจ้าหน้าที่" in text and "เจ้ำ" not in text and "ส ำ" not in text
