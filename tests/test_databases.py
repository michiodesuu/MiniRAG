"""PostgreSQL + Qdrant stores and `rag-ingest load`.

Every test runs twice: on the in-memory stand-ins (MemoryChunkStore and
QdrantClient(":memory:"), always) and on the real databases from
docker-compose.yml (skipped when they are not reachable). The real run uses a
throwaway PostgreSQL schema and uniquely named Qdrant collections.
"""
from contextlib import contextmanager
import json
import os
import uuid

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qdrant_client")
from qdrant_client import QdrantClient  # noqa: E402

from rag_ingest.chunking import ChunkSettings, CorpusConfig, chunk_corpus, write_run  # noqa: E402
from rag_ingest.cli import main  # noqa: E402
from rag_ingest.databases.loader import UnsafeLoadError, chunk_type, plan_load, run_load  # noqa: E402
from rag_ingest.databases.memory import MemoryChunkStore  # noqa: E402
from rag_ingest.databases.qdrant import QdrantVectorStore, point_id  # noqa: E402
from rag_ingest.retrieval.indexing import EmbedderSpec, StaleIndexError, build_index, load_index  # noqa: E402
from rag_ingest.tokens import approximate_tokens  # noqa: E402

EXCLUDED_STEM = "1.คู่มือเจ้าหน้าที่วินิจฉัยเงินทดแทน"  # the real excluded PDF's stem (config.py)
INGEST = {
    "คู่มือสาขา": "<!-- page 1 -->\n1. แก้ไขข้อมูล\n\nกดปุ่มแก้ไขเมื่อรายการยังไม่อนุมัติ\n\n"
                  "<!-- page 2 -->\n2. ออกรายงาน\n\nเลือกเดือนแล้วกดปุ่มออกรายงานอุบัติเหตุ\n\n"
                  "| เดือน | จำนวน |\n|---|---|\n| มกราคม | 3 |\n",
    "คู่มือเขต": "<!-- page 1 -->\n1. อนุมัติ\n\nผู้จัดการเขตกดปุ่มอนุมัติรายการของสาขา\n",
}
ROLES = {"คู่มือสาขา": "branch_manager", "คู่มือเขต": "district_manager", EXCLUDED_STEM: "officer"}


def thai_id() -> str:
    base = "310170012345"
    return base + str((11 - sum(int(d) * (13 - i) for i, d in enumerate(base)) % 11) % 10)


# --- the two backends -------------------------------------------------------------------------

class Backend:
    def __init__(self, chunk_store, client):
        self.chunk_store, self.client = chunk_store, client

    def vectors(self, run: str, dim: int) -> QdrantVectorStore:
        return QdrantVectorStore(self.client, run, dim)

    def load(self, plan):
        return run_load(plan, self.chunk_store, self.vectors(plan.run, plan.dim),
                        lambda run: self.vectors(run, plan.dim))


def docker_backend():
    psycopg = pytest.importorskip("psycopg")
    from rag_ingest.databases.postgres import PostgresChunkStore
    from rag_ingest.databases.settings import PostgresSettings, QdrantSettings, read_env_file
    from rag_ingest.config import Paths

    read_env_file(Paths.default().root / ".env")
    try:
        store = PostgresChunkStore.connect(PostgresSettings.from_env())
        settings = QdrantSettings.from_env()
        client = QdrantClient(url=settings.url, api_key=settings.api_key, timeout=3,
                              check_compatibility=False)
        client.get_collections()
    except (RuntimeError, psycopg.OperationalError, Exception) as error:  # noqa: BLE001
        pytest.skip(f"databases not reachable ({type(error).__name__}); run `docker compose up -d`")
    return store, client


@pytest.fixture(params=["memory", "docker"])
def backend(request):
    if request.param == "memory":
        yield Backend(MemoryChunkStore(), QdrantClient(":memory:"))
        return
    store, client = docker_backend()
    schema = f"test_{uuid.uuid4().hex[:12]}"
    store.conn.execute(f"CREATE SCHEMA {schema}")
    store.conn.execute(f"SET search_path TO {schema}")
    before = {c.name for c in client.get_collections().collections}
    try:
        yield Backend(store, client)
    finally:
        store.conn.execute(f"DROP SCHEMA {schema} CASCADE")
        store.close()
        for c in client.get_collections().collections:
            if c.name not in before and c.name.startswith("t-"):
                client.delete_collection(c.name)


# --- a small corpus, chunked and indexed with the fake embedder --------------------------------

class Corpus:
    def __init__(self, root, excluded=None, extra=None):
        self.root, self.ingest, self.chunks = root, root / "ingest", root / "chunks"
        self.ingest.mkdir(exist_ok=True)
        self.excluded = excluded or {}
        for name, text in {**INGEST, **(extra or {})}.items():
            self.write(name, text)

    def write(self, name: str, text: str) -> None:
        (self.ingest / f"{name}.md").write_text(text, encoding="utf-8")

    def declare_source(self, name: str) -> None:
        """Record in the ingest manifest which file a document came from (sets its doc_type)."""
        entry = {"image": f"{name}_page_1_image_1", "document": name, "page": 1, "md5": "e" * 32,
                 "status": "text_only"}
        (self.ingest / "manifest.json").write_text(json.dumps([entry], ensure_ascii=False), encoding="utf-8")

    def index(self, run: str | None = None):
        config = CorpusConfig(self.ingest, self.root / "raw", self.excluded, frozenset(), ROLES)
        result = chunk_corpus(config, ChunkSettings(), approximate_tokens)
        write_run(result, self.chunks, {})
        return build_index(EmbedderSpec("fake", {}), self.chunks, self.root / "index",
                           run or f"t-{uuid.uuid4().hex[:10]}").folder

    def plan(self, folder, **kwargs):
        return plan_load(folder, self.ingest, self.root, **kwargs)


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path)


def state(backend, plan):
    """Everything both stores hold for this run: (documents, PG chunk ids, Qdrant chunk ids)."""
    docs = {d["id"]: d["content_hash"] for d in backend.chunk_store.documents()}
    pg = sorted(c for d in docs for c in backend.chunk_store.chunk_ids(d, plan.run))
    return docs, pg, backend.vectors(plan.run, plan.dim).chunk_ids()


# --- schema and stores -------------------------------------------------------------------------

def test_migrations_apply_once(backend):
    first = backend.chunk_store.migrate()
    assert backend.chunk_store.migrate() == []
    if not isinstance(backend.chunk_store, MemoryChunkStore):
        assert first == ["001_documents_chunks"]


def test_load_writes_both_databases_linked_by_document(backend, corpus):
    plan = corpus.plan(corpus.index())
    result = backend.load(plan)
    assert sorted(result.loaded) == sorted(INGEST) and not result.failed
    docs, pg, qdrant = state(backend, plan)
    assert pg == qdrant and len(pg) == sum(len(d.chunks) for d in plan.documents)
    branch = backend.chunk_store.document("คู่มือสาขา")
    assert branch["role"] == "branch_manager" and branch["status"] == "unverified"
    assert branch["source_path"] == "ingest/คู่มือสาขา.md" and len(branch["content_hash"]) == 64
    vectors = backend.vectors(plan.run, plan.dim)
    assert vectors.count("คู่มือเขต") == len(backend.chunk_store.chunk_ids("คู่มือเขต", plan.run))
    payload = vectors.client.retrieve(plan.run, [point_id(pg[0])])[0].payload
    assert set(payload) >= {"chunk_id", "document", "role", "type", "page_start", "page_end"}
    assert "display_text" not in payload and "embed_text" not in payload


def test_chunk_types(corpus):
    plan = corpus.plan(corpus.index())
    types = {c["text"]: c["type"] for d in plan.documents for c in d.chunks}
    assert "table" in types.values() and "text" in types.values()
    assert chunk_type({"kind": "ocr", "display_text": ""}) == "screenshot"
    assert chunk_type({"kind": "mixed", "display_text": ""}) == "screenshot"


def test_qdrant_search_matches_numpy(backend, corpus):
    folder = corpus.index()
    plan = corpus.plan(folder)
    backend.load(plan)
    index = load_index(folder, corpus.ingest)
    embedder = index.embedder_spec().create()
    vectors = backend.vectors(plan.run, plan.dim)
    for query in ("ออกรายงานอุบัติเหตุ", "อนุมัติรายการ", "แก้ไขข้อมูล"):
        q = embedder.embed_query(query)
        for filters in (None, {"role": "district_manager"}, {"document": "คู่มือสาขา", "doc_type": "unknown"}):
            want, got = index.search(q, 5, filters), vectors.search(q, 5, filters)
            assert [h.chunk_id for h in got] == [h.chunk_id for h in want]
            assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-5)
            assert all(h.meta["page_start"] == w.meta["page_start"] for h, w in zip(got, want))
    with pytest.raises(ValueError, match="Unknown filter"):
        vectors.search(q, 1, {"colour": "red"})


def test_qdrant_delete_by_document(backend, corpus):
    plan = corpus.plan(corpus.index())
    backend.load(plan)
    vectors = backend.vectors(plan.run, plan.dim)
    vectors.delete_document("คู่มือเขต")
    assert vectors.count("คู่มือเขต") == 0 and vectors.count("คู่มือสาขา") > 0


# --- idempotency -------------------------------------------------------------------------------

def test_second_load_skips_everything(backend, corpus):
    plan = corpus.plan(corpus.index())
    backend.load(plan)
    before = state(backend, plan)
    result = backend.load(corpus.plan(plan_folder(corpus, plan)))
    assert sorted(result.skipped) == sorted(INGEST) and result.loaded == []
    assert state(backend, plan) == before


def plan_folder(corpus, plan):
    return corpus.root / "index" / plan.run


def test_changed_document_replaces_its_old_chunks(backend, corpus):
    run = f"t-{uuid.uuid4().hex[:10]}"
    first = corpus.plan(corpus.index(run))
    backend.load(first)
    old_ids = set(backend.chunk_store.chunk_ids("คู่มือเขต", run))
    corpus.write("คู่มือเขต", "<!-- page 3 -->\n9. ใหม่\n\nขั้นตอนใหม่ทั้งหมดของผู้จัดการเขต\n")
    second = corpus.plan(corpus.index(run))
    result = backend.load(second)
    assert result.loaded == ["คู่มือเขต"] and result.skipped == ["คู่มือสาขา"]
    docs, pg, qdrant = state(backend, second)
    new_ids = set(backend.chunk_store.chunk_ids("คู่มือเขต", run))
    assert pg == qdrant and not (old_ids & new_ids) and all("#p3#" in c for c in new_ids)


def test_changed_document_is_cleared_from_other_runs(backend, corpus):
    old = corpus.plan(corpus.index())
    backend.load(old)
    corpus.write("คู่มือเขต", INGEST["คู่มือเขต"] + "\nเพิ่มบรรทัด\n")
    new = corpus.plan(corpus.index())
    backend.load(new)
    assert backend.vectors(old.run, old.dim).count("คู่มือเขต") == 0
    assert backend.vectors(old.run, old.dim).count("คู่มือสาขา") > 0  # unchanged document stays
    assert backend.chunk_store.index_runs("คู่มือเขต") == [new.run]


def test_status_is_kept_unless_given(backend, corpus):
    folder = corpus.index()
    backend.load(corpus.plan(folder, status="verified"))
    result = backend.load(corpus.plan(folder))
    assert result.loaded == [] and backend.chunk_store.document("คู่มือเขต")["status"] == "verified"
    backend.load(corpus.plan(folder, status="unverified"))
    assert backend.chunk_store.document("คู่มือเขต")["status"] == "unverified"


# --- all or nothing ----------------------------------------------------------------------------

class FailingVectors:
    """Wraps a QdrantVectorStore; one method raises once, after the real call when `after`."""
    def __init__(self, inner, method, after=False):
        self.inner, self.method, self.after, self.armed = inner, method, after, True

    def __getattr__(self, name):
        real = getattr(self.inner, name)
        if name != self.method:
            return real

        def call(*args, **kwargs):
            if self.armed and not self.after:
                self.armed = False
                raise ConnectionError("qdrant went away")
            out = real(*args, **kwargs)
            if self.armed:
                self.armed = False
                raise ConnectionError("qdrant went away")
            return out
        return call


@pytest.mark.parametrize("method, after", [("add", False), ("add", True), ("delete_document", False)])
def test_qdrant_failure_leaves_both_stores_as_before(backend, corpus, method, after):
    run = f"t-{uuid.uuid4().hex[:10]}"
    backend.load(corpus.plan(corpus.index(run)))
    corpus.write("คู่มือเขต", "<!-- page 3 -->\n9. ใหม่\n\nขั้นตอนใหม่ทั้งหมดของผู้จัดการเขต\n")
    plan = corpus.plan(corpus.index(run))
    before = state(backend, plan)
    vectors = FailingVectors(backend.vectors(run, plan.dim), method, after)
    result = run_load(plan, backend.chunk_store, vectors)
    assert list(result.failed) == ["คู่มือเขต"] and "qdrant went away" in result.failed["คู่มือเขต"]
    assert state(backend, plan) == before


def test_postgres_failure_restores_qdrant(backend, corpus):
    run = f"t-{uuid.uuid4().hex[:10]}"
    backend.load(corpus.plan(corpus.index(run)))
    corpus.write("คู่มือเขต", "<!-- page 3 -->\n9. ใหม่\n\nขั้นตอนใหม่ทั้งหมดของผู้จัดการเขต\n")
    plan = corpus.plan(corpus.index(run))
    before = state(backend, plan)
    store = backend.chunk_store
    real = store.transaction

    @contextmanager
    def failing_commit():
        with real():
            yield store
            raise ConnectionError("commit failed")  # after Qdrant was written
    store.transaction = failing_commit
    try:
        result = run_load(plan, store, backend.vectors(run, plan.dim))
    finally:
        store.transaction = real
    assert "commit failed" in result.failed["คู่มือเขต"]
    assert state(backend, plan) == before


def test_a_failed_document_does_not_stop_the_others(backend, corpus):
    plan = corpus.plan(corpus.index())
    vectors = FailingVectors(backend.vectors(plan.run, plan.dim), "add")
    result = run_load(plan, backend.chunk_store, vectors)
    assert len(result.failed) == 1 and len(result.loaded) == 1
    docs, pg, qdrant = state(backend, plan)
    assert list(docs) == result.loaded and pg == qdrant


# --- what must never be loaded -----------------------------------------------------------------

def test_rejected_thai_id_chunk_never_reaches_either_database(backend, tmp_path):
    secret = thai_id()
    corpus = Corpus(tmp_path, extra={"คู่มือเขต": INGEST["คู่มือเขต"] + f"\n## บัตร\n\nเลขบัตร {secret}\n"})
    plan = corpus.plan(corpus.index())
    rejected = [json.loads(line) for line in (corpus.chunks / "rejected.jsonl").read_text("utf-8").splitlines()]
    assert rejected and rejected[0]["rule"] == "thai_national_id"
    backend.load(plan)
    docs, pg, qdrant = state(backend, plan)
    assert rejected[0]["chunk_id"] not in pg + qdrant
    texts = [c["text"] for d in plan.documents for c in d.chunks]
    assert texts and not any(secret in t for t in texts)
    if isinstance(backend.chunk_store, MemoryChunkStore):
        assert not any(secret in row["text"] for row in backend.chunk_store.chunks.values())
    else:
        hits = backend.chunk_store.conn.execute("SELECT count(*) AS n FROM chunks WHERE text LIKE %s",
                                                (f"%{secret}%",)).fetchone()
        assert hits["n"] == 0


def test_index_with_a_rejected_chunk_is_refused(backend, corpus):
    folder = corpus.index()
    victim = load_index(folder, corpus.ingest).store.metas[0]["chunk_id"]
    (corpus.chunks / "rejected.jsonl").write_text(
        json.dumps({"chunk_id": victim, "rule": "thai_national_id"}) + "\n", encoding="utf-8")
    with pytest.raises(UnsafeLoadError, match="rejected.jsonl"):
        corpus.plan(folder)


def test_excluded_pdf_never_reaches_either_database(backend, tmp_path):
    excluded = {f"{EXCLUDED_STEM}.pdf": "fake blur patches"}
    corpus = Corpus(tmp_path, excluded=excluded, extra={EXCLUDED_STEM: "<!-- page 1 -->\nชื่อ นามสกุล\n"})
    corpus.declare_source(f"{EXCLUDED_STEM}.pdf")
    plan = corpus.plan(corpus.index(), excluded_files=excluded)
    backend.load(plan)
    docs, pg, qdrant = state(backend, plan)
    assert EXCLUDED_STEM not in docs
    assert not any(c.startswith(EXCLUDED_STEM) for c in pg + qdrant)


def test_index_holding_an_excluded_document_is_refused(tmp_path):
    """Chunked without the exclusion (e.g. an old index), loaded with it: refused, fail closed."""
    corpus = Corpus(tmp_path, extra={EXCLUDED_STEM: "<!-- page 1 -->\nชื่อ นามสกุล\n"})
    folder = corpus.index()
    with pytest.raises(UnsafeLoadError, match="is excluded"):
        corpus.plan(folder, excluded_files={f"{EXCLUDED_STEM}.pdf": "fake blur patches"})


def test_real_exclusion_list_blocks_the_pdf(tmp_path):
    """The default excluded_files is config.EXCLUDED_DOCUMENTS."""
    corpus = Corpus(tmp_path, extra={EXCLUDED_STEM: "<!-- page 1 -->\nชื่อ นามสกุล\n"})
    with pytest.raises(UnsafeLoadError, match=EXCLUDED_STEM):
        corpus.plan(corpus.index())


def test_stale_index_is_refused(corpus):
    folder = corpus.index()
    corpus.write("คู่มือเขต", INGEST["คู่มือเขต"] + "\nเพิ่มบรรทัด\n")
    with pytest.raises(StaleIndexError):
        corpus.plan(folder)


def test_untrusted_manifest_is_refused(corpus):
    from rag_ingest.retrieval.indexing import UntrustedIndexError
    folder = corpus.index()
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    manifest["embedder"] = {"kind": "local", "options": {"model": "someone/unknown-model"}, "settings": {}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(UntrustedIndexError):
        corpus.plan(folder)


# --- the command -------------------------------------------------------------------------------

def test_load_command(corpus, monkeypatch, capsys):
    backend = Backend(MemoryChunkStore(), QdrantClient(":memory:"))
    monkeypatch.setenv("RAG_INGEST_ROOT", str(corpus.root))
    monkeypatch.setattr("rag_ingest.databases.commands.open_stores",
                        lambda paths, run, dim: (backend.chunk_store, backend.vectors(run, dim)))
    folder = corpus.index()
    args = ["load", "--index", str(folder), "--ingest", str(corpus.ingest)]
    main(args + ["--dry-run"])
    assert "2 documents" in capsys.readouterr().out and backend.chunk_store.docs == {}
    main(args)
    out = capsys.readouterr().out
    assert "loaded   คู่มือเขต" in out and "points" in out
    main(args)
    assert "skipped  คู่มือเขต (unchanged)" in capsys.readouterr().out


def test_settings_need_a_password(monkeypatch):
    from rag_ingest.databases.settings import PostgresSettings
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        PostgresSettings.from_env()


def test_env_file_does_not_override(tmp_path, monkeypatch):
    from rag_ingest.databases.settings import read_env_file
    (tmp_path / ".env").write_text("# c\nRAG_T_A=from-file\nRAG_T_B='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("RAG_T_A", "from-env")
    monkeypatch.delenv("RAG_T_B", raising=False)
    read_env_file(tmp_path / ".env")
    assert os.environ["RAG_T_A"] == "from-env" and os.environ["RAG_T_B"] == "quoted"
    monkeypatch.delenv("RAG_T_B")
