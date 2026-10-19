"""Vector store, index build/load, staleness check and the embed/search commands."""
import json

import pytest

np = pytest.importorskip("numpy")
from rag_ingest.chunking import ChunkSettings, CorpusConfig, chunk_corpus, write_run  # noqa: E402
from rag_ingest.cli import main  # noqa: E402
from rag_ingest.indexing import EmbedderSpec, StaleIndexError, build_index, latest_index, load_index  # noqa: E402
from rag_ingest.tokens import approximate_tokens  # noqa: E402
from rag_ingest.vectorstore import NumpyVectorStore  # noqa: E402


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def store_with_three():
    store = NumpyVectorStore(3)
    metas = [{"chunk_id": f"c{i}", "document": doc, "role": role, "doc_type": "pdf",
              "page_start": i, "page_end": i, "embed_text": "x", "display_text": "y"}
             for i, (doc, role) in enumerate([("สาขา", "branch_manager"), ("เขต", "district_manager"),
                                               ("สาขา", "branch_manager")])]
    store.add(np.stack([unit(1, 0, 0), unit(0, 1, 0), unit(1, 1, 0)]), metas)
    return store


def test_search_ranks_by_cosine_and_filters():
    store = store_with_three()
    hits = store.search(unit(1, 0, 0), k=2)
    assert [h.chunk_id for h in hits] == ["c0", "c2"]
    assert hits[0].score == pytest.approx(1.0) and hits[1].score == pytest.approx(0.7071, abs=1e-4)
    assert [h.chunk_id for h in store.search(unit(1, 0, 0), 3, {"role": "district_manager"})] == ["c1"]
    assert [h.chunk_id for h in store.search(unit(0, 1, 0), 3, {"document": "สาขา"})] == ["c2", "c0"]
    with pytest.raises(ValueError, match="Unknown filter"):
        store.search(unit(1, 0, 0), 1, {"colour": "red"})


def test_store_round_trip_without_text(tmp_path):
    store_with_three().save(tmp_path / "idx")
    loaded = NumpyVectorStore.load(tmp_path / "idx")
    assert loaded.vectors.dtype == np.float32 and loaded.vectors.shape == (3, 3)
    assert "embed_text" not in loaded.metas[0] and "display_text" not in loaded.metas[0]


INGEST = {
    "คู่มือสาขา": "<!-- page 1 -->\n1. แก้ไขข้อมูล\n\nกดปุ่มแก้ไขเมื่อรายการยังไม่อนุมัติ\n\n"
                  "<!-- page 2 -->\n2. ออกรายงาน\n\nเลือกเดือนแล้วกดปุ่มออกรายงานอุบัติเหตุ\n",
    "คู่มือเขต": "<!-- page 1 -->\n1. อนุมัติ\n\nผู้จัดการเขตกดปุ่มอนุมัติรายการของสาขา\n",
}


@pytest.fixture
def chunked(tmp_path):
    ingest, chunks = tmp_path / "ingest", tmp_path / "chunks"
    ingest.mkdir()
    for name, text in INGEST.items():
        (ingest / f"{name}.md").write_text(text, encoding="utf-8")
    config = CorpusConfig(ingest, tmp_path / "raw", {}, frozenset(),
                          {"คู่มือสาขา": "branch_manager", "คู่มือเขต": "district_manager"})
    write_run(chunk_corpus(config, ChunkSettings(), approximate_tokens), chunks, {})
    return ingest, chunks


def test_build_search_and_cache(chunked, tmp_path):
    ingest, chunks = chunked
    spec = EmbedderSpec("fake", {})
    built = build_index(spec, chunks, tmp_path / "index")
    assert (built.count, built.computed) == (3, 3)
    assert build_index(spec, chunks, tmp_path / "index").computed == 0  # all from the cache
    index = load_index(latest_index(tmp_path / "index"), ingest)
    query = index.embedder_spec().create().embed_query("ออกรายงานอุบัติเหตุ")
    top = index.search(query, 1)[0]
    assert top.meta["section"] == "2. ออกรายงาน" and top.meta["page_start"] == 2
    assert "ออกรายงาน" in index.texts[top.chunk_id]


def test_staleness_check_names_the_changed_file(chunked, tmp_path):
    ingest, chunks = chunked
    folder = build_index(EmbedderSpec("fake", {}), chunks, tmp_path / "index").folder
    (ingest / "คู่มือเขต.md").write_text(INGEST["คู่มือเขต"] + "\nเพิ่มบรรทัด\n", encoding="utf-8")
    with pytest.raises(StaleIndexError, match="คู่มือเขต.md changed"):
        load_index(folder, ingest)


def test_staleness_check_catches_new_chunks(chunked, tmp_path):
    ingest, chunks = chunked
    folder = build_index(EmbedderSpec("fake", {}), chunks, tmp_path / "index").folder
    (chunks / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(StaleIndexError, match="chunks.jsonl changed"):
        load_index(folder, ingest)


def test_manifest_never_holds_api_secrets():
    spec = EmbedderSpec("api", {"url": "https://h/v1/embeddings", "key": "k3y", "model": "m"})
    assert spec.persistable() == {"model": "m"}


def test_embed_and_search_commands(chunked, tmp_path, monkeypatch, capsys):
    ingest, chunks = chunked
    monkeypatch.setenv("RAG_INGEST_ROOT", str(tmp_path))
    main(["embed", "--embedder", "fake", "--chunks", str(chunks), "--index-root", str(tmp_path / "index")])
    assert "Indexed 3 chunks" in capsys.readouterr().out
    main(["search", "ผู้จัดการเขตกดปุ่มอนุมัติ", "-k", "1", "--ingest", str(ingest),
          "--index", str(latest_index(tmp_path / "index"))])
    out = capsys.readouterr().out
    assert "คู่มือเขต#p1#c00" in out
    manifest = json.loads((latest_index(tmp_path / "index") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedder"]["kind"] == "fake" and set(manifest["sources"]) == set(INGEST)


def test_manifest_cannot_redirect_the_api_key(monkeypatch):
    monkeypatch.setenv("EMBED_API_URL", "https://embed.internal/v1/embeddings")
    monkeypatch.setenv("EMBED_API_KEY", "k3y")
    tampered = {"kind": "api", "options": {"model": "m", "url": "https://attacker.example/x", "key": "x"}}
    spec = EmbedderSpec.from_manifest(tampered)
    assert spec.options == {"model": "m"}
    captured = {}
    monkeypatch.setattr("rag_ingest.indexing.create_embedder", lambda kind, **o: captured.update(o))
    spec.create()
    assert captured["url"] == "https://embed.internal/v1/embeddings" and captured["key"] == "k3y"


@pytest.mark.parametrize("embedder", [
    {"kind": "local", "options": {"model": "someone/unknown-model"}},
    {"kind": "local", "options": {"model": "BAAI/bge-m3", "revision": "0" * 40}},
    {"kind": "shell", "options": {}},
])
def test_manifest_cannot_load_unknown_models(embedder):
    from rag_ingest.indexing import UntrustedIndexError
    with pytest.raises(UntrustedIndexError):
        EmbedderSpec.from_manifest(embedder)
