"""`rag-ingest load`: argument wiring for the database stage."""
import argparse
import sys

from rag_ingest.config import Paths


def open_stores(paths: Paths, collection: str, dim: int):
    """PostgresChunkStore + QdrantVectorStore from the environment (and .env)."""
    from rag_ingest.databases.postgres import PostgresChunkStore
    from rag_ingest.databases.qdrant import QdrantVectorStore
    from rag_ingest.databases.settings import read_env_file

    read_env_file(paths.root / ".env")
    return PostgresChunkStore.connect(), QdrantVectorStore.connect(collection, dim)


def run_load_command(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.databases.loader import UnsafeLoadError, plan_load, run_load
    from rag_ingest.databases.qdrant import QdrantVectorStore
    from rag_ingest.retrieval.indexing import StaleIndexError, UntrustedIndexError, latest_index

    folder = paths.resolve(args.index) if args.index else latest_index(paths.index)
    try:
        plan = plan_load(folder, paths.resolve(args.ingest), paths.root, args.status)
    except (StaleIndexError, UntrustedIndexError, UnsafeLoadError) as error:
        sys.exit(str(error))
    chunks = sum(len(d.chunks) for d in plan.documents)
    print(f"Index: {plan.run} ({plan.model}, {plan.dim} dims): {len(plan.documents)} documents, {chunks} chunks")
    if args.dry_run:
        for d in plan.documents:
            print(f"  {d.id}: {len(d.chunks)} chunks")
        return
    try:
        chunk_store, vectors = open_stores(paths, plan.run, plan.dim)
        chunk_store.migrate()
    except Exception as error:  # noqa: BLE001 - missing settings or databases not running
        sys.exit(f"Cannot reach the databases: {error}\nStart them with `docker compose up -d` (settings: .env).")
    try:
        result = run_load(plan, chunk_store, vectors,
                          lambda run: QdrantVectorStore(vectors.client, run, plan.dim))
    finally:
        chunk_store.close()
    for doc in result.loaded:
        print(f"  loaded   {doc}")
    for doc in result.skipped:
        print(f"  skipped  {doc} (unchanged)")
    for doc, error in result.failed.items():
        print(f"  FAILED   {doc}: {error} (nothing written for it)")
    print(f"Qdrant collection {plan.run}: {vectors.count()} points")
    if result.failed:
        sys.exit(1)


def register(add, paths: Paths) -> None:
    p = add("load", run_load_command, "Load an index run into PostgreSQL and Qdrant (docker compose up -d).")
    p.add_argument("--index", help="Index folder (default: the newest under outputs/index).")
    p.add_argument("--ingest", default=paths.relative(paths.ingest), help="Ingest folder, for the staleness check.")
    p.add_argument("--status", choices=["verified", "unverified"],
                   help="Document status (default: keep the stored one; new documents are unverified).")
    p.add_argument("--dry-run", action="store_true", help="Check and list what would be loaded; write nothing.")
