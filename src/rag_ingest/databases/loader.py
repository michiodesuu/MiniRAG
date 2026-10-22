"""`rag-ingest load`: an index run (outputs/index/<run>/) into PostgreSQL and Qdrant.

What may be loaded (checked before anything is written; any failure refuses
the whole load, it never skips quietly):
  - the index passes load_index(): its sources and chunks.jsonl are unchanged
    (stale check) and its embedder passes the manifest allow-list
  - every vector row is a chunk of the current chunks.jsonl
  - no row is in rejected.jsonl (personal data), from a document the chunk
    manifest excluded, or from an EXCLUDED_DOCUMENTS file (by stem + source
    type; an unknown type fails closed, as in chunking)

Per document, in order:
  - skipped when PostgreSQL already holds it at the same content hash and
    status, and both stores hold exactly its chunk ids for this run
  - otherwise replaced: the PostgreSQL rows are rewritten inside a transaction,
    then the Qdrant points are upserted and the document's other points
    removed, then the transaction commits. If any step fails, the transaction
    rolls back and the document's previous Qdrant points are put back, so the
    two stores never disagree about a document.
  - when its content hash changed, its points in other runs' collections are
    removed too (PostgreSQL already dropped those chunks: they came from the
    old text)
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np

from rag_ingest.chunking.sources import nfc_name
from rag_ingest.config import EXCLUDED_DOCUMENTS
from rag_ingest.retrieval.indexing import CHUNKS_FILE, MANIFEST, load_index, read_json, read_jsonl

STATUSES = ("verified", "unverified")
REJECTED_FILE = "rejected.jsonl"
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE)


class UnsafeLoadError(RuntimeError):
    """The index holds something that must never reach the databases."""


def chunk_type(chunk: dict) -> str:
    """screenshot (any OCR'd screenshot in it), table (a Markdown table), else text."""
    if chunk["kind"] in ("ocr", "mixed"):
        return "screenshot"
    return "table" if TABLE_SEPARATOR.search(chunk["display_text"]) else "text"


def excluded_pairs(excluded_files: dict[str, str]) -> set[tuple[str, str]]:
    return {(nfc_name(Path(name).stem), Path(name).suffix.lstrip(".").lower()) for name in excluded_files}


def is_excluded(document: str, doc_type: str, pairs: set[tuple[str, str]]) -> bool:
    stem = nfc_name(document)
    return any(s == stem and (t == doc_type or doc_type == "unknown") for s, t in pairs)


@dataclass
class DocumentLoad:
    document: dict
    chunks: list[dict]
    vectors: np.ndarray
    metas: list[dict]

    @property
    def id(self) -> str:
        return self.document["id"]


@dataclass
class LoadPlan:
    run: str
    model: str
    dim: int
    documents: list[DocumentLoad]


def plan_load(folder: Path, ingest_dir: Path, root: Path, status: str | None = None,
              excluded_files: dict[str, str] = EXCLUDED_DOCUMENTS) -> LoadPlan:
    """Read and check everything to be loaded; writes nothing."""
    index = load_index(folder, ingest_dir)  # StaleIndexError
    index.embedder_spec()                   # UntrustedIndexError
    manifest = index.manifest
    chunks_dir = Path(manifest["chunks_dir"])
    chunks = {c["chunk_id"]: c for c in read_jsonl(chunks_dir / CHUNKS_FILE)}
    rejected_path = chunks_dir / REJECTED_FILE
    rejected = {r["chunk_id"] for r in read_jsonl(rejected_path)} if rejected_path.exists() else set()
    chunk_manifest = read_json(chunks_dir / MANIFEST)
    excluded_docs = {(nfc_name(e["document"]), e.get("doc_type", "unknown"))
                     for e in chunk_manifest.get("excluded", []) + manifest.get("excluded", [])}
    pairs = excluded_pairs(excluded_files) | excluded_docs

    problems = []
    for meta in index.store.metas:
        cid, chunk = meta["chunk_id"], chunks.get(meta["chunk_id"])
        if chunk is None:
            problems.append(f"{cid}: not in {chunks_dir / CHUNKS_FILE}")
        elif cid in rejected:
            problems.append(f"{cid}: listed in {REJECTED_FILE}")
        elif chunk["document"] not in manifest["sources"]:
            problems.append(f"{cid}: document {chunk['document']!r} is not a recorded source")
        elif is_excluded(chunk["document"], chunk["doc_type"], pairs):
            problems.append(f"{cid}: document {chunk['document']}.{chunk['doc_type']} is excluded")
    if problems:
        raise UnsafeLoadError("Refusing to load:\n  " + "\n  ".join(problems))

    model = manifest["embedder"]["settings"]["model"]
    rows: dict[str, list[int]] = {}
    for i, meta in enumerate(index.store.metas):
        rows.setdefault(chunks[meta["chunk_id"]]["document"], []).append(i)
    documents = []
    for doc, idx in rows.items():
        first = chunks[index.store.metas[idx[0]]["chunk_id"]]
        document = {"id": doc, "name": doc, "source_path": _relative(ingest_dir / f"{doc}.md", root),
                    "content_hash": manifest["sources"][doc], "role": first["role"],
                    "doc_type": first["doc_type"], "status": status}
        doc_chunks = [chunks[index.store.metas[i]["chunk_id"]] for i in idx]
        rows_out = [{"chunk_id": c["chunk_id"], "index_run": folder.name, "document_id": doc,
                     "section": c["section"] or "", "page_start": c["page_start"], "page_end": c["page_end"],
                     "type": chunk_type(c), "text": c["display_text"], "token_count": c["tokens"],
                     "embedding_model": model} for c in doc_chunks]
        metas = [{**index.store.metas[i], "type": rows_out[n]["type"]} for n, i in enumerate(idx)]
        documents.append(DocumentLoad(document, rows_out, index.store.vectors[idx], metas))
    return LoadPlan(folder.name, model, int(index.store.vectors.shape[1]), documents)


def _relative(path: Path, root: Path) -> str:
    return (path.relative_to(root) if path.is_relative_to(root) else path).as_posix()


@dataclass
class LoadResult:
    loaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def up_to_date(load: DocumentLoad, existing: dict | None, chunk_store, vectors, run: str) -> bool:
    if existing is None or existing["content_hash"] != load.document["content_hash"] \
            or existing["status"] != load.document["status"]:
        return False
    wanted = sorted(c["chunk_id"] for c in load.chunks)
    return chunk_store.chunk_ids(load.id, run) == wanted and vectors.chunk_ids(load.id) == wanted


def load_document(load: DocumentLoad, chunk_store, vectors, run: str) -> list[str]:
    """Replace one document in both stores, all or nothing. Returns other runs whose
    collections still hold the document's points from an older content hash."""
    existing = chunk_store.document(load.id)
    changed = existing is not None and existing["content_hash"] != load.document["content_hash"]
    other_runs = [r for r in chunk_store.index_runs(load.id) if r != run] if changed else []
    undo_vectors, undo_metas = vectors.points(load.id)
    try:
        with chunk_store.transaction():
            chunk_store.replace_document(load.document, load.chunks, run)
            vectors.add(load.vectors, load.metas)
            vectors.delete_document(load.id, keep={c["chunk_id"] for c in load.chunks})
    except BaseException:
        vectors.delete_document(load.id)
        if undo_metas:
            vectors.add(undo_vectors, undo_metas)
        raise
    return other_runs


def run_load(plan: LoadPlan, chunk_store, vectors, collection_for=None) -> LoadResult:
    """collection_for(run) -> the vector store of another run, for clearing stale points."""
    chunk_store.migrate()
    vectors.ensure_collection()
    result = LoadResult()
    for load in plan.documents:
        existing = chunk_store.document(load.id)
        if load.document["status"] is None:  # keep a status set earlier (e.g. verified by a person)
            load.document["status"] = existing["status"] if existing else "unverified"
        if up_to_date(load, existing, chunk_store, vectors, plan.run):
            result.skipped.append(load.id)
            continue
        try:
            stale_runs = load_document(load, chunk_store, vectors, plan.run)
        except Exception as error:  # noqa: BLE001 - reported per document, the rest still load
            result.failed[load.id] = f"{type(error).__name__}: {error}"
            continue
        for other in stale_runs:
            if collection_for:
                collection_for(other).delete_document(load.id)
        result.loaded.append(load.id)
    return result
