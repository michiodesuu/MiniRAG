"""Embedding models behind one interface (Strategy), created by name (registry).

Mirrors engines/: heavy libraries (sentence-transformers, torch, httpx) are
imported only when an embedder is constructed, and only one model is loaded
at a time.

  fake   deterministic character n-gram hashing; tests and a weak baseline
  local  sentence-transformers on CPU, revision pinned (profiles.py)
  api    an OpenAI-compatible POST /v1/embeddings (internal service only)

Every embedder returns float32 rows of length 1 (L2-normalised), so a dot
product between a query vector and a passage vector is their cosine similarity.
"""
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # numpy is an optional dependency (the `embed` extra)
    import numpy as np


class Embedder(Protocol):
    model: str
    dim: int

    def embed_passages(self, texts: list[str]) -> "np.ndarray": ...  # (n, dim) float32, unit rows

    def embed_query(self, text: str) -> "np.ndarray": ...            # (dim,) float32, unit length

    def count_tokens(self, text: str) -> int: ...                  # as a passage would be fed

    def settings(self) -> dict: ...  # model, revision, dim, pooling, normalise, max_len, prefixes


EMBEDDERS = {
    "fake": "rag_ingest.embeddings.fake:FakeEmbedder",
    "local": "rag_ingest.embeddings.local:LocalEmbedder",
    "api": "rag_ingest.embeddings.api:ApiEmbedder",
}


def create_embedder(name: str, **options) -> Embedder:
    try:
        module, factory = EMBEDDERS[name].split(":")
    except KeyError:
        raise ValueError(f"Unknown embedder {name!r}; choose from {', '.join(EMBEDDERS)}") from None
    return getattr(import_module(module), factory)(**options)

