"""An OpenAI-compatible POST /v1/embeddings endpoint (an internal service only:
document text must not leave the company network).

  request   {"model": <EMBED_API_MODEL>, "input": [text, ...]}
  response  {"data": [{"index": i, "embedding": [float, ...]}, ...]}

URL and model come from EMBED_API_URL / EMBED_API_MODEL (or --api-url /
--api-model); the key only from EMBED_API_KEY, never the command line, and it
is never written to settings, manifests or reports. Vectors are normalised
here, whatever the server does. Prefixes follow profiles.py when the model is
known. The server's model version cannot be seen from here: set
EMBED_API_REVISION when it changes, or cached vectors would be reused.
Token counting needs a tokenizer: --tokenizer names a Hugging Face tokenizer
matching the served model; without one, counting falls back to an
approximation and the chunk manifest says so.
"""
import argparse
import os
import sys
from urllib.parse import urlsplit

import numpy as np

from rag_ingest.embeddings.profiles import profile
from rag_ingest.embeddings.vectors import normalise_rows
from rag_ingest.net import warn_if_clear_text
from rag_ingest.tokens import approximate_tokens

DEFAULT_API_BATCH = 32
DEFAULT_TIMEOUT = 120


def endpoint(url: str) -> str:
    """Host and port only: never user:password@, path or query string, which can hold secrets."""
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port}" if parts.port else (parts.hostname or "")


class ApiEmbedder:
    def __init__(self, url: str, model: str, key: str | None = None, tokenizer: str | None = None,
                 batch_size: int = DEFAULT_API_BATCH, revision: str | None = None):
        import httpx

        warn_if_clear_text(url, "document text")
        known = profile(model)
        self._httpx, self.url, self.model, self._key = httpx, url, model, key
        self.batch_size, self.revision = batch_size, revision or "unknown"
        self.query_prefix = known.query_prefix if known else ""
        self.passage_prefix = known.passage_prefix if known else ""
        self.dim = known.dim if known else 0
        self.tokenizer_name = tokenizer
        self._tokenizer = None

    def _post(self, texts: list[str]) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        response = self._httpx.post(self.url, json={"model": self.model, "input": texts},
                                    headers=headers, timeout=DEFAULT_TIMEOUT)
        if response.status_code >= 400:
            # httpx's own message quotes the full URL, which may carry credentials
            raise RuntimeError(f"Embedding API at {endpoint(self.url)} answered HTTP {response.status_code}")
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return np.asarray([row["embedding"] for row in rows], dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:
        parts = [self._post(texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        vectors = normalise_rows(np.concatenate(parts)) if parts else np.zeros((0, self.dim), np.float32)
        self.dim = self.dim or vectors.shape[1]
        return vectors

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer_name:
            return approximate_tokens(self.passage_prefix + text)
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return len(self._tokenizer(self.passage_prefix + text, add_special_tokens=True)["input_ids"])

    def settings(self) -> dict:
        return {"model": self.model, "revision": self.revision, "dim": self.dim,
                "pooling": "server", "normalise": True, "max_len": None,
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "endpoint": endpoint(self.url),
                "token_counter": self.tokenizer_name or "approximate"}


def api_options(args: argparse.Namespace) -> dict:
    url = args.api_url or os.environ.get("EMBED_API_URL")
    model = args.api_model or os.environ.get("EMBED_API_MODEL")
    if not (url and model):
        sys.exit("--embedder api needs --api-url and --api-model (or EMBED_API_URL / EMBED_API_MODEL)")
    return {"url": url, "model": model, "key": os.environ.get("EMBED_API_KEY"),
            "tokenizer": args.tokenizer, "revision": os.environ.get("EMBED_API_REVISION")}
