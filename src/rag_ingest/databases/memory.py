"""MemoryChunkStore: PostgresChunkStore's interface in memory, for tests and CI.

transaction() works on a copy and keeps it only when the block ends without an
error, like the PostgreSQL transaction it stands in for.
"""
from contextlib import contextmanager
import copy


class MemoryChunkStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.chunks: dict[tuple[str, str], dict] = {}  # (index_run, chunk_id) -> row

    def migrate(self) -> list[str]:
        return []

    def close(self) -> None:
        pass

    @contextmanager
    def transaction(self):
        saved = copy.deepcopy((self.docs, self.chunks))
        try:
            yield self
        except BaseException:
            self.docs, self.chunks = saved
            raise

    def document(self, document_id: str) -> dict | None:
        return self.docs.get(document_id)

    def documents(self) -> list[dict]:
        return [self.docs[k] for k in sorted(self.docs)]

    def chunk_ids(self, document_id: str, index_run: str) -> list[str]:
        return sorted(c for (run, c), row in self.chunks.items()
                      if run == index_run and row["document_id"] == document_id)

    def all_chunk_ids(self) -> list[str]:
        return sorted(c for _, c in self.chunks)

    def index_runs(self, document_id: str) -> list[str]:
        return sorted({run for (run, _), row in self.chunks.items() if row["document_id"] == document_id})

    def replace_document(self, document: dict, chunks: list[dict], index_run: str) -> None:
        old = self.docs.get(document["id"])
        every_run = old is not None and old["content_hash"] != document["content_hash"]
        self.docs[document["id"]] = dict(document)
        self.chunks = {key: row for key, row in self.chunks.items()
                       if row["document_id"] != document["id"] or (not every_run and key[0] != index_run)}
        for c in chunks:
            if (index_run, c["chunk_id"]) in self.chunks:
                raise ValueError(f"duplicate key {c['chunk_id']}")  # as the primary key would
            self.chunks[(index_run, c["chunk_id"])] = dict(c)

    def delete_document(self, document_id: str) -> None:
        self.docs.pop(document_id, None)
        self.chunks = {k: r for k, r in self.chunks.items() if r["document_id"] != document_id}
