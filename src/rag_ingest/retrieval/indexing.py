"""Build, find and load embedding indexes (spec 4.3 and 5).

outputs/index/<run>/
  vectors.npy + meta.jsonl   the vector store (vectorstore.py)
  manifest.json              how it was made: embedder kind + options +
                             settings(), chunk settings, chunks.jsonl sha256,
                             sha256 of every source .md, created time, count

Trust: a manifest may come from someone else (a shared or edited index). The
embedder is rebuilt from it only through an allow-list: known options per
kind, local models only from profiles.py at their pinned revision, and for the
API the URL and key only from the environment, never from the manifest, so a
manifest cannot send the key or the queries to another host, or make search
load arbitrary model weights.

Staleness check: `search` and `eval-retrieval` refuse an index whose recorded
source hashes no longer match the current ingest .md files (or whose
chunks.jsonl changed), and name what changed. The query is always embedded by
an embedder rebuilt from the index manifest, so it matches the passages.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re

from rag_ingest.chunking.sources import sha256_file
from rag_ingest.config import ENCODING
from rag_ingest.embeddings import create_embedder
from rag_ingest.embeddings.cache import EmbeddingCache, embed_with_cache
from rag_ingest.embeddings.profiles import profile
from rag_ingest.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from rag_ingest.retrieval.vectorstore import TEXT_FIELDS, Hit, NumpyVectorStore

MANIFEST, CHUNKS_FILE, CACHE_DIR = "manifest.json", "chunks.jsonl", "cache"


class StaleIndexError(RuntimeError):
    """The index was built from files that have since changed."""


class UntrustedIndexError(RuntimeError):
    """The index manifest asks for an embedder this project does not allow."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding=ENCODING).splitlines() if line]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding=ENCODING))


def run_name(embedder_settings: dict, chunk_settings: dict) -> str:
    """<model>__<8 hex of the settings>: different settings never overwrite each other."""
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", embedder_settings["model"]).strip("-")
    material = json.dumps([embedder_settings, chunk_settings], sort_keys=True, ensure_ascii=False)
    return f"{slug}__{hashlib.sha256(material.encode('utf-8')).hexdigest()[:8]}"


SECRET_OPTIONS = ("key", "url")  # never written to a manifest; read from the environment again
API_ENV = {"url": "EMBED_API_URL", "key": "EMBED_API_KEY"}
MANIFEST_OPTIONS = {  # what may be taken from a manifest, per embedder kind
    "fake": set(),
    "local": {"model", "revision", "batch_size", "max_seq_length"},
    "api": {"model", "tokenizer", "revision", "batch_size"},
}


@dataclass
class EmbedderSpec:
    """How to rebuild the embedder that made an index."""
    kind: str
    options: dict

    def persistable(self) -> dict:
        return {k: v for k, v in self.options.items() if k not in SECRET_OPTIONS}

    def create(self):
        options = dict(self.options)
        if self.kind == "api":  # from_manifest never passes these: they come from the environment
            options.update({k: os.environ.get(env) for k, env in API_ENV.items() if not options.get(k)})
        return create_embedder(self.kind, **options)

    @classmethod
    def from_manifest(cls, embedder: dict) -> "EmbedderSpec":
        """The allow-listed embedder of an index manifest (see the module docstring)."""
        kind = embedder.get("kind")
        if kind not in MANIFEST_OPTIONS:
            raise UntrustedIndexError(f"Index asks for an unknown embedder {kind!r}")
        options = {k: v for k, v in embedder.get("options", {}).items() if k in MANIFEST_OPTIONS[kind]}
        if kind == "local":
            known = profile(options.get("model", ""))
            if known is None or options.get("revision") not in (None, known.revision):
                raise UntrustedIndexError(
                    f"Index asks for model {options.get('model')!r} at revision {options.get('revision')!r}, "
                    "which is not a profiled model at its pinned revision (embeddings/profiles.py)")
        return cls(kind, options)


@dataclass
class BuiltIndex:
    folder: Path
    count: int
    computed: int  # embedded now; the rest came from the cache


def build_index(spec: EmbedderSpec, chunks_dir: Path, index_root: Path, run: str | None = None,
                use_cache: bool = True, embedder=None) -> BuiltIndex:
    embedder = embedder or spec.create()
    chunks = read_jsonl(chunks_dir / CHUNKS_FILE)
    chunk_manifest = read_json(chunks_dir / MANIFEST)
    cache = EmbeddingCache(index_root / CACHE_DIR) if use_cache else None
    embedded = embed_with_cache(embedder, [c["embed_text"] for c in chunks], cache)
    store = NumpyVectorStore(embedder.dim)
    store.add(embedded.vectors, chunks)
    settings = embedder.settings()
    folder = index_root / (run or run_name(settings, chunk_manifest["settings"]))
    store.save(folder)
    manifest = {
        "embedder": {"kind": spec.kind, "options": spec.persistable(), "settings": settings},
        "chunk_settings": chunk_manifest["settings"],
        "chunks_dir": str(chunks_dir),
        "chunks_sha256": sha256_file(chunks_dir / CHUNKS_FILE),
        "sources": chunk_manifest["sources"],
        "excluded": chunk_manifest.get("excluded", []),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(chunks), "dim": int(store.vectors.shape[1]) if len(chunks) else embedder.dim,
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding=ENCODING)
    return BuiltIndex(folder, len(chunks), embedded.computed)


def latest_index(index_root: Path) -> Path:
    runs = [p for p in index_root.glob(f"*/{MANIFEST}") if p.parent.name != CACHE_DIR]
    if not runs:
        raise FileNotFoundError(f"No index in {index_root}; run `rag-ingest embed` first")
    return max(runs, key=lambda p: read_json(p)["created"]).parent


def stale_reasons(manifest: dict, ingest_dir: Path) -> list[str]:
    reasons = []
    for document, recorded in manifest["sources"].items():
        path = ingest_dir / f"{document}.md"
        if not path.exists():
            reasons.append(f"{path} no longer exists")
        elif sha256_file(path) != recorded:
            reasons.append(f"{path} changed since the index was built")
    chunks = Path(manifest["chunks_dir"]) / CHUNKS_FILE
    if not chunks.exists() or sha256_file(chunks) != manifest["chunks_sha256"]:
        reasons.append(f"{chunks} changed since the index was built")
    return reasons


@dataclass
class LoadedIndex:
    folder: Path
    manifest: dict
    store: NumpyVectorStore
    texts: dict[str, str]  # chunk_id -> display_text
    bm25: BM25Index | None = None

    def embedder_spec(self) -> EmbedderSpec:
        return EmbedderSpec.from_manifest(self.manifest["embedder"])

    def search(self, query_vector, k: int, filters: dict | None = None) -> list[Hit]:
        return self.store.search(query_vector, k, filters)

    def hybrid_search(
        self,
        query: str,
        query_vector,
        k: int = 10,
        filters: dict | None = None,
        candidate_k: int | None = None,
        k_rrf: int = 60,
    ) -> list[Hit]:
        """Hybrid search combining Dense Vector Search and Sparse BM25 with RRF."""
        pool_k = candidate_k or max(k * 2, 20)
        dense_hits = self.search(query_vector, k=pool_k, filters=filters)
        if self.bm25 is None:
            return dense_hits[:k]
        bm25_hits = self.bm25.search(query, k=pool_k, filters=filters)
        return reciprocal_rank_fusion([dense_hits, bm25_hits], k_rrf=k_rrf, top_k=k)


def load_index(folder: Path, ingest_dir: Path) -> LoadedIndex:
    manifest = read_json(folder / MANIFEST)
    reasons = stale_reasons(manifest, ingest_dir)
    if reasons:
        raise StaleIndexError("Index is stale; run `rag-ingest chunk` and `rag-ingest embed` again:\n  "
                              + "\n  ".join(reasons))
    chunks = read_jsonl(Path(manifest["chunks_dir"]) / CHUNKS_FILE)
    texts = {c["chunk_id"]: c["display_text"] for c in chunks}
    bm25 = BM25Index().fit(
        (
            c["chunk_id"],
            c.get("embed_text") or c.get("display_text", ""),
            {k: v for k, v in c.items() if k not in TEXT_FIELDS},
        )
        for c in chunks
    )
    return LoadedIndex(folder, manifest, NumpyVectorStore.load(folder), texts, bm25=bm25)
