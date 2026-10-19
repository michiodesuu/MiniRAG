"""Vector store: brute-force search over unit-length vectors (spec 5).

  vectors.npy  float32, N x dim, one row per chunk, each of length 1
  meta.jsonl   one JSON object per row: chunk_id + metadata (no embed_text or
               display_text; search reads the text from chunks.jsonl)

Search: score = vectors @ query. Rows and query have length 1, so the dot
product is the cosine of the angle between them: 1 = same direction,
0 = unrelated, negative = opposite (rare for text). Filters (document, role,
doc_type) select rows before scoring; the top k scores are returned.
Brute force is exact and fast at this size (N in the hundreds); the
VectorStore protocol lets FAISS or a database replace it without changing
callers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json

import numpy as np

from rag_ingest.config import ENCODING

FILTER_FIELDS = ("document", "role", "doc_type")
TEXT_FIELDS = ("embed_text", "display_text")
VECTORS_FILE, META_FILE = "vectors.npy", "meta.jsonl"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float
    meta: dict

    @property
    def pages(self) -> str:
        start, end = self.meta.get("page_start"), self.meta.get("page_end")
        return f"{start}" if start == end else f"{start}-{end}"


class VectorStore(Protocol):
    def add(self, vectors: np.ndarray, metas: list[dict]) -> None: ...

    def save(self, folder: Path) -> None: ...

    def search(self, query: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]: ...


class NumpyVectorStore:
    def __init__(self, dim: int = 0):
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.metas: list[dict] = []

    def add(self, vectors: np.ndarray, metas: list[dict]) -> None:
        if len(vectors) != len(metas):
            raise ValueError(f"{len(vectors)} vectors but {len(metas)} metadata rows")
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if not self.metas else np.concatenate([self.vectors, vectors])
        self.metas += [{k: v for k, v in m.items() if k not in TEXT_FIELDS} for m in metas]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / VECTORS_FILE, self.vectors)
        (folder / META_FILE).write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in self.metas), encoding=ENCODING)

    @classmethod
    def load(cls, folder: Path) -> "NumpyVectorStore":
        store = cls()
        store.vectors = np.load(folder / VECTORS_FILE)
        store.metas = [json.loads(line) for line in
                       (folder / META_FILE).read_text(encoding=ENCODING).splitlines() if line]
        return store

    def _mask(self, filters: dict | None) -> np.ndarray:
        mask = np.ones(len(self.metas), dtype=bool)
        for field, value in (filters or {}).items():
            if field not in FILTER_FIELDS:
                raise ValueError(f"Unknown filter {field!r}; use {', '.join(FILTER_FIELDS)}")
            if value is not None:
                mask &= np.array([m.get(field) == value for m in self.metas], dtype=bool)
        return mask

    def search(self, query: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        rows = np.flatnonzero(self._mask(filters))
        if not len(rows) or k <= 0:
            return []
        scores = self.vectors[rows] @ np.asarray(query, dtype=np.float32)
        top = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep chunk order
        return [Hit(self.metas[rows[i]]["chunk_id"], float(scores[i]), self.metas[rows[i]]) for i in top]
