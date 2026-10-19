"""Thai text clean-up and normalization."""
import re
import unicodedata

NO_TEXT_MARK = "[ไม่พบข้อความที่อ่านได้]"  # the OCR prompt's "nothing readable" answer

# Tesseract puts spaces between Thai words/characters; Thai is written without them.
_THAI_GAP = re.compile(r"(?<=[฀-๿]) (?=[฀-๿])")


def join_thai_words(text: str) -> str:
    return _THAI_GAP.sub("", text)


def normalize_for_scoring(text: str) -> str:
    """Drop layout so only characters are compared: HTML, markdown marks, whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"</(td|th|tr|p|div|li|h\d)>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#*|`​﻿]", "", text)
    return re.sub(r"\s+", "", text)
