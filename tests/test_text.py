import pytest

from rag_ingest.metrics import cer, edit_distance
from rag_ingest.models import ImageRef
from rag_ingest.quality import rejection
from rag_ingest.textproc import NO_TEXT_MARK, join_thai_words, normalize_for_scoring


def test_join_thai_words_keeps_latin_spaces():
    assert join_thai_words("บัน ทึก ข้อ มูล OK now") == "บันทึกข้อมูล OK now"


def test_normalize_drops_layout():
    html = "<table><tr><td>ชื่อ</td><td>สาขา</td></tr></table>\n# **หัวข้อ** |"
    assert normalize_for_scoring(html) == "ชื่อสาขาหัวข้อ"


def test_edit_distance_and_cer():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert cer("abc", "abc") == pytest.approx(0)
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    assert cer("x", "") == pytest.approx(1)  # empty reference does not divide by zero


def test_image_ref_round_trip():
    ref = ImageRef("ผู้จัดการสาขา", 7, 3)
    assert ref.stem == "ผู้จัดการสาขา_page_7_image_3"
    assert ImageRef.parse(ref.stem) == ref
    assert ImageRef.parse(ref.stem + ".png") == ref
    assert ImageRef.parse("random.png") is None


@pytest.mark.parametrize("text, reason", [
    ("ข้อความปกติ", None),
    ("   ", "empty"),
    (NO_TEXT_MARK, "no_text_marker"),
    ("Extract all text from the image.\nFormatting Rules: ...", "prompt_echo"),
    ("\n".join(["1.บันทึกอุบัติเหตุ", "2.ติดตามสถานะ"] * 15), "repetition_loop"),
    ("\n".join(f"แถวที่ {i}" for i in range(30)), None),  # long but every line distinct
])
def test_rejection(text, reason):
    assert rejection(text) == reason
