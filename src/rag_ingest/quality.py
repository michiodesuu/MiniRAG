"""Rules that reject a bad OCR result before it reaches the index.

Each rule is a small named check, so the ingest manifest can say *why* a
result was rejected. Add a rule by appending to RULES.
"""
from dataclasses import dataclass
from typing import Callable

from rag_ingest.textproc import NO_TEXT_MARK

PROMPT_ECHO_MARKS = ("Extract all text from the image", "Formatting Rules:")


@dataclass(frozen=True)
class Rule:
    name: str
    fails: Callable[[str], bool]


RULES = (
    Rule("empty", lambda text: not text.strip()),
    # Images reaching OCR are not icons, so "nothing readable" means the engine failed.
    Rule("no_text_marker", lambda text: NO_TEXT_MARK in text),
    Rule("prompt_echo", lambda text: any(mark in text for mark in PROMPT_ECHO_MARKS)),
)


def rejection(text: str) -> str | None:
    """Name of the first rule the text fails, or None if it is usable."""
    return next((rule.name for rule in RULES if rule.fails(text)), None)
