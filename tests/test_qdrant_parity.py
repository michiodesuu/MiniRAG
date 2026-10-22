"""Parity gate: plain dense search on Qdrant reproduces the numpy index on the
30-question set (BGE-M3, the committed chunks), R@5 0.96 and MRR 0.80.

Needs the BGE-M3 index under outputs/index (not committed) and the model in the
local Hugging Face cache; skipped otherwise. Runs on Qdrant's local mode, and
on the docker Qdrant too when RAG_PARITY_DOCKER=1.
"""
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("qdrant_client")
pytest.importorskip("sentence_transformers")
from qdrant_client import QdrantClient  # noqa: E402

from rag_ingest.config import Paths  # noqa: E402
from rag_ingest.databases.loader import plan_load, run_load  # noqa: E402
from rag_ingest.databases.memory import MemoryChunkStore  # noqa: E402
from rag_ingest.databases.qdrant import QdrantVectorStore  # noqa: E402
from rag_ingest.retrieval.evaluation import evaluate, load_questions, metrics  # noqa: E402
from rag_ingest.retrieval.indexing import StaleIndexError, load_index, read_json  # noqa: E402

PATHS = Paths.default()


def bge_index():
    runs = sorted((p.parent for p in PATHS.index.glob("BAAI-bge-m3__*/manifest.json")),
                  key=lambda f: read_json(f / "manifest.json")["created"])
    for folder in reversed(runs):
        try:
            return load_index(folder, PATHS.ingest)
        except StaleIndexError:
            continue
    pytest.skip("no current BGE-M3 index; run `rag-ingest embed --model BAAI/bge-m3`")


def qdrant_clients():
    yield "local", QdrantClient(":memory:")
    if os.environ.get("RAG_PARITY_DOCKER") == "1":
        from rag_ingest.databases.settings import QdrantSettings, read_env_file
        read_env_file(PATHS.root / ".env")
        settings = QdrantSettings.from_env()
        yield "docker", QdrantClient(url=settings.url, api_key=settings.api_key, check_compatibility=False)


@pytest.fixture(scope="module")
def numpy_side():
    index = bge_index()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the cached model only; never download in a test
    try:
        embedder = index.embedder_spec().create()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"BGE-M3 not available locally ({type(error).__name__})")
    questions = load_questions(PATHS.eval_questions)
    vectors = {q.id: embedder.embed_query(q.question) for q in questions}
    return index, questions, vectors


@pytest.mark.parametrize("name, client", list(qdrant_clients()), ids=lambda v: v if isinstance(v, str) else "")
def test_qdrant_reproduces_numpy(numpy_side, name, client):
    index, questions, vectors = numpy_side
    plan = plan_load(index.folder, PATHS.ingest, PATHS.root)
    collection = f"parity-{plan.run}"
    store = QdrantVectorStore(client, collection, plan.dim)
    store.drop()
    try:
        plan.run = collection
        result = run_load(plan, MemoryChunkStore(), store)
        assert not result.failed and store.count() == index.manifest["count"]
        by_text = {q.question: vectors[q.id] for q in questions}
        embed = by_text.__getitem__
        numpy_results = evaluate(questions, embed, index.search)
        qdrant_results = evaluate(questions, embed, store.search)
        for q in questions:
            want, got = index.search(vectors[q.id], 5), store.search(vectors[q.id], 5)
            assert [h.chunk_id for h in got] == [h.chunk_id for h in want], q.id
            assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-4), q.id
        n, d = metrics(numpy_results), metrics(qdrant_results)
        assert (d.recall, d.mrr) == (n.recall, n.mrr)
        assert round(d.recall[5], 2) == 0.96 and round(d.mrr, 2) == 0.80
    finally:
        store.drop()
