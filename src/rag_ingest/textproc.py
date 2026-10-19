"""Thai text clean-up and normalization."""
import re
import unicodedata

NO_TEXT_MARK = "[ไม่พบข้อความที่อ่านได้]"  # the OCR prompt's "nothing readable" answer

# Some Thai PDFs extract the vowel "ำ" as a space plus "า" ("ท าการ" for "ทำการ").
# No Thai word starts with "า", so a consonant (+ tone mark) then " า" is always this case.
_BROKEN_SARA_AM = re.compile(r"([ก-ฮ][่-๋]?)ํ? า")

# Tesseract puts spaces between Thai words/characters; Thai is written without them.
_THAI_GAP = re.compile(r"(?<=[฀-๿]) (?=[฀-๿])")


def fix_sara_am(text: str) -> str:
    return _BROKEN_SARA_AM.sub(r"\1ำ", text)


def join_thai_words(text: str) -> str:
    return _THAI_GAP.sub("", text)


def normalize_for_scoring(text: str) -> str:
    """Drop layout so only characters are compared: HTML, markdown marks, whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"</(td|th|tr|p|div|li|h\d)>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*|`​﻿]", "", text)
    return re.sub(r"\s+", "", text)
