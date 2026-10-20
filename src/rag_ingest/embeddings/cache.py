"""Vector cache: only new or changed chunks are embedded.

key = sha256(JSON of the embedder's settings() + NUL + embed_text). settings()
holds the model, its revision, dimension, pooling, max length, prefixes and
library versions, so changing any of them, or the text, gives a new key.
Vectors live in outputs/index/cache/<key[:2]>/<key>.npy (not committed).
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np


def cache_key(settings: dict, text: str) -> str:
    material = json.dumps(settings, sort_keys=True, ensure_ascii=False) + "\0" + text
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class EmbeddingCache:
    folder: Path

    def path(self, key: str) -> Path:
        return self.folder / key[:2] / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        path = self.path(key)
        return np.load(path) if path.exists() else None

    def put(self, key: str, vector: np.ndarray) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector.astype(np.float32))


@dataclass
class CachedEmbedding:
    vectors: np.ndarray
    computed: int  # how many were embedded now; the rest came from the cache


def embed_with_cache(embedder, texts: list[str], cache: EmbeddingCache | None) -> CachedEmbedding:
    """Passage vectors for texts, embedding only those the cache does not hold."""
    if cache is None:
        return CachedEmbedding(embedder.embed_passages(texts), len(texts))
    settings = embedder.settings()
    keys = [cache_key(settings, t) for t in texts]
    found = {i: v for i, k in enumerate(keys) if (v := cache.get(k)) is not None}
    missing = [i for i in range(len(texts)) if i not in found]
    if missing:
        for i, vector in zip(missing, embedder.embed_passages([texts[i] for i in missing])):
            cache.put(keys[i], vector)
            found[i] = vector
    vectors = np.stack([found[i] for i in range(len(texts))]) if texts else np.zeros((0, embedder.dim), np.float32)
    return CachedEmbedding(vectors.astype(np.float32), len(missing))
