"""QdrantVectorStore: the VectorStore protocol (vectorstore.py) on a Qdrant collection.

One collection per index run, named after the run folder (<model>__<hash>),
cosine distance. Qdrant needs a UUID or integer point id, so the id is
uuid5(chunk_id): the same chunk always lands on the same point, and loading
it again overwrites rather than duplicates. The payload holds what search
filters on and what the evaluation reads (document, role, doc_type, kind,
type, pages, section) plus the chunk id; no chunk text (that is in PostgreSQL).

`QdrantClient(":memory:")` runs the same code without a server (tests, CI).
"""
from pathlib import Path
import uuid

import numpy as np

from rag_ingest.retrieval.vectorstore import FILTER_FIELDS, TEXT_FIELDS, Hit

POINT_NAMESPACE = uuid.UUID("3f1c6a52-6f0e-4d4e-9a57-8c1f0b7e2d11")  # fixed: ids must not change
PAYLOAD_FIELDS = ("chunk_id", "document", "role", "doc_type", "kind", "type", "section", "page_start", "page_end")
UPSERT_BATCH = 256


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, client, collection: str, dim: int):
        self.client, self.collection, self.dim = client, collection, dim

    @classmethod
    def connect(cls, collection: str, dim: int, settings=None) -> "QdrantVectorStore":
        from qdrant_client import QdrantClient
        from rag_ingest.databases.settings import QdrantSettings

        settings = settings or QdrantSettings.from_env()
        return cls(QdrantClient(url=settings.url, api_key=settings.api_key), collection, dim)

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE))
            for field in FILTER_FIELDS:
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
            return
        size = self.client.get_collection(self.collection).config.params.vectors.size
        if size != self.dim:
            raise ValueError(f"Collection {self.collection} has {size} dimensions, not {self.dim}")

    def add(self, vectors: np.ndarray, metas: list[dict]) -> None:
        from qdrant_client import models

        if len(vectors) != len(metas):
            raise ValueError(f"{len(vectors)} vectors but {len(metas)} metadata rows")
        self.ensure_collection()
        points = [models.PointStruct(id=point_id(m["chunk_id"]), vector=np.asarray(v, dtype=np.float32).tolist(),
                                     payload={k: m.get(k) for k in PAYLOAD_FIELDS if k not in TEXT_FIELDS})
                  for v, m in zip(vectors, metas)]
        for start in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(self.collection, points[start:start + UPSERT_BATCH], wait=True)

    def save(self, folder: Path) -> None:
        """Nothing to do: Qdrant persists every write."""

    def _filter(self, filters: dict | None):
        from qdrant_client import models

        conditions = []
        for field, value in (filters or {}).items():
            if field not in FILTER_FIELDS:
                raise ValueError(f"Unknown filter {field!r}; use {', '.join(FILTER_FIELDS)}")
            if value is not None:
                conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
        return models.Filter(must=conditions) if conditions else None

    def search(self, query: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        from qdrant_client import models

        query_filter = self._filter(filters)
        if k <= 0 or not self.client.collection_exists(self.collection):
            return []
        result = self.client.query_points(
            self.collection, query=np.asarray(query, dtype=np.float32).tolist(), limit=k,
            query_filter=query_filter, with_payload=True, search_params=models.SearchParams(exact=True))
        return [Hit(p.payload["chunk_id"], float(p.score), dict(p.payload)) for p in result.points]

    def count(self, document: str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, count_filter=self._filter({"document": document}),
                                 exact=True).count

    def points(self, document: str) -> tuple[np.ndarray, list[dict]]:
        """Every vector + payload of one document (the loader's undo copy)."""
        if not self.client.collection_exists(self.collection):
            return np.zeros((0, self.dim), dtype=np.float32), []
        records, offset = [], None
        while True:
            batch, offset = self.client.scroll(self.collection, scroll_filter=self._filter({"document": document}),
                                               limit=UPSERT_BATCH, offset=offset, with_vectors=True, with_payload=True)
            records += batch
            if offset is None:
                break
        vectors = np.array([r.vector for r in records], dtype=np.float32).reshape(len(records), self.dim)
        return vectors, [dict(r.payload) for r in records]

    def chunk_ids(self, document: str | None = None) -> list[str]:
        if not self.client.collection_exists(self.collection):
            return []
        ids, offset = [], None
        while True:
            batch, offset = self.client.scroll(self.collection, scroll_filter=self._filter({"document": document}),
                                               limit=UPSERT_BATCH, offset=offset, with_payload=["chunk_id"])
            ids += [r.payload["chunk_id"] for r in batch]
            if offset is None:
                return sorted(ids)

    def delete_document(self, document: str, keep: set[str] = frozenset()) -> None:
        """Remove the document's points, except those whose chunk id is in keep."""
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            return
        drop = [point_id(c) for c in self.chunk_ids(document) if c not in keep]
        if drop:
            self.client.delete(self.collection, models.PointIdsList(points=drop), wait=True)

    def drop(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
