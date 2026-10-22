"""PostgresChunkStore: the documents and chunks tables.

Migrations are the numbered .sql files in migrations/, applied in order inside
one transaction each and recorded in schema_migrations, so `migrate()` can run
every time.

Writes happen inside `transaction()`: the loader replaces a document's chunks
and its Qdrant points in one step, and rolls this transaction back when the
Qdrant side fails (loader.py).
"""
from contextlib import contextmanager
from importlib import resources

from rag_ingest.databases.settings import PostgresSettings

DOCUMENT_FIELDS = ("id", "name", "source_path", "content_hash", "role", "doc_type", "status")
CHUNK_FIELDS = ("chunk_id", "index_run", "document_id", "section", "page_start", "page_end",
                "type", "text", "token_count", "embedding_model")


def migration_files() -> list[tuple[str, str]]:
    folder = resources.files("rag_ingest.databases") / "migrations"
    files = sorted((f.name, f.read_text(encoding="utf-8")) for f in folder.iterdir() if f.name.endswith(".sql"))
    return [(name.removesuffix(".sql"), sql) for name, sql in files]


class PostgresChunkStore:
    def __init__(self, conn):
        self.conn = conn

    @classmethod
    def connect(cls, settings: PostgresSettings | None = None) -> "PostgresChunkStore":
        import psycopg
        from psycopg.rows import dict_row

        settings = settings or PostgresSettings.from_env()
        # autocommit: transaction() then issues a real BEGIN/COMMIT, not a savepoint
        return cls(psycopg.connect(**settings.connect_kwargs(), row_factory=dict_row, autocommit=True))

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> list[str]:
        """Apply migrations not yet recorded; returns the versions applied now."""
        with self.conn.transaction():
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                              "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
            done = {r["version"] for r in self.conn.execute("SELECT version FROM schema_migrations")}
        applied = []
        for version, sql in migration_files():
            if version not in done:
                with self.conn.transaction():
                    self.conn.execute(sql)
                    self.conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                applied.append(version)
        return applied

    @contextmanager
    def transaction(self):
        with self.conn.transaction():
            yield self

    def document(self, document_id: str) -> dict | None:
        return self.conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()

    def documents(self) -> list[dict]:
        return self.conn.execute("SELECT * FROM documents ORDER BY id").fetchall()

    def chunk_ids(self, document_id: str, index_run: str) -> list[str]:
        rows = self.conn.execute("SELECT chunk_id FROM chunks WHERE document_id = %s AND index_run = %s "
                                 "ORDER BY chunk_id", (document_id, index_run))
        return [r["chunk_id"] for r in rows]

    def all_chunk_ids(self) -> list[str]:
        return [r["chunk_id"] for r in self.conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")]

    def index_runs(self, document_id: str) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT index_run FROM chunks WHERE document_id = %s ORDER BY 1",
                                 (document_id,))
        return [r["index_run"] for r in rows]

    def replace_document(self, document: dict, chunks: list[dict], index_run: str) -> None:
        """Upsert the document; drop its chunks of this run (and of every run when the
        content changed, since they were cut from the old text), insert the new ones."""
        old = self.document(document["id"])
        columns = ", ".join(DOCUMENT_FIELDS)
        updates = ", ".join(f"{f} = EXCLUDED.{f}" for f in DOCUMENT_FIELDS if f != "id")
        self.conn.execute(
            f"INSERT INTO documents ({columns}, ingested_at) VALUES ({', '.join(['%s'] * len(DOCUMENT_FIELDS))}, now()) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}, ingested_at = now()",
            [document[f] for f in DOCUMENT_FIELDS])
        if old and old["content_hash"] != document["content_hash"]:
            self.conn.execute("DELETE FROM chunks WHERE document_id = %s", (document["id"],))
        else:
            self.conn.execute("DELETE FROM chunks WHERE document_id = %s AND index_run = %s",
                              (document["id"], index_run))
        with self.conn.cursor() as cur:
            cur.executemany(f"INSERT INTO chunks ({', '.join(CHUNK_FIELDS)}) "
                            f"VALUES ({', '.join(['%s'] * len(CHUNK_FIELDS))})",
                            [[c[f] for f in CHUNK_FIELDS] for c in chunks])

    def delete_document(self, document_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))  # chunks cascade
