"""PostgreSQL (documents + chunks) and Qdrant (vectors), filled by `rag-ingest load`.

  settings.py   connection settings, from environment variables only
  postgres.py   PostgresChunkStore + migrations
  qdrant.py     QdrantVectorStore (the VectorStore protocol on Qdrant)
  memory.py     MemoryChunkStore: the in-memory stand-in for tests and CI
  loader.py     index run -> both databases, idempotent, all-or-nothing per document
"""
