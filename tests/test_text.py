import pytest

from rag_ingest.metrics import cer, edit_distance
from rag_ingest.models import ImageRef
from rag_ingest.quality import rejection
from rag_ingest.textproc import NO_TEXT_MARK, fix_sara_am, join_thai_words, normalize_for_scoring


@pytest.mark.parametrize("broken, fixed", [
    ("ท าการ", "ทำการ"),
    ("ส่งค าขอ", "ส่งคำขอ"),
    ("ส าน", "สำน"),
    ("น ้ า", "น ้ า"),         # no consonant directly before the space: left alone
])
def test_fix_sara_am(broken, fixed):
    assert fix_sara_am(broken) == fixed


def test_fix_sara_am_keeps_normal_text():
    text = "กดปุ่ม ถัดไป เพื่อบันทึก ABC 123"
    assert fix_sara_am(text) == text


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
])
def test_rejection(text, reason):
    assert rejection(text) == reason
