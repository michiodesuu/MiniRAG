"""Pure-Python token counter, for tests and the `fake` embedder only.

Real chunking counts with the embedding model's own tokenizer (Thai has no
spaces, so whitespace counting is meaningless). This fallback approximates a
multilingual subword tokenizer: a Latin word or digit run is one token, Thai
counts one token per up-to-3 characters, any other symbol one token.
"""
import re

_PIECES = re.compile(r"[A-Za-z]+|[0-9]+|[฀-๿]{1,3}|\S")


def approximate_tokens(text: str) -> int:
    return len(_PIECES.findall(text))
