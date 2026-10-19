"""Deterministic hashing embedder: tests, and a weak baseline in the evaluation.

Each character 3-gram of the NFC, lower-cased text is hashed (blake2b, so the
result is the same in every run, unlike Python's hash()) into one of `dim`
buckets with a +1/-1 sign; the counts are then L2-normalised. Similar strings
share n-grams, so it rewards literal overlap only: no meaning, no synonyms.
"""
import hashlib
import unicodedata

import numpy as np

from rag_ingest.embeddings.vectors import normalise_rows
from rag_ingest.tokens import approximate_tokens


class FakeEmbedder:
    def __init__(self, dim: int = 64, n: int = 3):
        self.dim, self.n = dim, n
        self.model = f"fake-hash-{n}gram-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        text = unicodedata.normalize("NFC", text).lower()
        vector = np.zeros(self.dim, dtype=np.float32)
        for i in range(max(len(text) - self.n + 1, 1)):
            digest = hashlib.blake2b(text[i:i + self.n].encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dim] += 1.0 if value >> 63 else -1.0
        return vector

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalise_rows(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_passages([text])[0]

    def count_tokens(self, text: str) -> int:
        return approximate_tokens(text)

    def settings(self) -> dict:
        return {"model": self.model, "revision": "1", "dim": self.dim, "pooling": "hashed char n-grams",
                "normalise": True, "max_len": None, "query_prefix": "", "passage_prefix": "",
                "token_counter": "approximate"}
