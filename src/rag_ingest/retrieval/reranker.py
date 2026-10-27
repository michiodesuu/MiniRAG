"""Cohere-compatible API client for reranking retrieval candidates."""
import os
from urllib.parse import urlsplit

from rag_ingest.retrieval.vectorstore import Hit

DEFAULT_TIMEOUT = 30


def _endpoint(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port}" if parts.port else (parts.hostname or "")


class RerankerClient:
    def __init__(self, url: str, model: str, key: str | None = None, timeout: int = DEFAULT_TIMEOUT):
        import httpx

        if not url:
            raise ValueError("Reranker API URL must not be empty")
        if not model:
            raise ValueError("Reranker model must not be empty")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Reranker API URL must be an absolute HTTP(S) URL")
        self._httpx = httpx
        self.url = url
        self.model = model
        self._key = key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "RerankerClient":
        url = os.environ.get("RERANK_API_URL")
        model = os.environ.get("RERANK_MODEL")
        if not url or not model:
            raise RuntimeError("RERANK_API_URL and RERANK_MODEL must be set")
        return cls(url, model, os.environ.get("RERANK_API_KEY") or None)

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        texts: dict[str, str],
        top_n: int | None = None,
    ) -> list[Hit]:
        if not hits:
            return []
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if any(hit.chunk_id not in texts for hit in hits):
            raise ValueError("Every reranker candidate must have display text")

        count = min(top_n, len(hits)) if top_n is not None else len(hits)
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            response = self._httpx.post(
                self.url,
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [texts[hit.chunk_id] for hit in hits],
                    "top_n": count,
                },
                headers=headers,
                timeout=self.timeout,
            )
        except self._httpx.HTTPError as error:
            raise RuntimeError(
                f"Reranker API at {_endpoint(self.url)} request failed ({type(error).__name__})"
            ) from error
        if response.status_code >= 400:
            raise RuntimeError(
                f"Reranker API at {_endpoint(self.url)} answered HTTP {response.status_code}"
            )

        try:
            results = response.json()["results"]
            ranked: list[tuple[int, float]] = [
                (int(row["index"]), float(row["relevance_score"])) for row in results
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Reranker API returned an invalid Cohere-style response") from error

        seen: set[int] = set()
        for index, _ in ranked:
            if index < 0 or index >= len(hits) or index in seen:
                raise ValueError("Reranker API returned an invalid candidate index")
            seen.add(index)

        ranked.sort(key=lambda result: result[1], reverse=True)
        return [
            Hit(hits[index].chunk_id, score, hits[index].meta)
            for index, score in ranked[:count]
        ]
