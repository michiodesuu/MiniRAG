"""Tests for W08: BM25 sparse search, Reciprocal Rank Fusion (RRF), and hybrid search."""
import pytest

from rag_ingest.retrieval.hybrid import BM25Index, default_thai_tokenizer, reciprocal_rank_fusion
from rag_ingest.retrieval.vectorstore import Hit


def test_thai_tokenizer_handles_thai_and_english():
    tokens = default_thai_tokenizer("การอนุมัติ เอกสาร Form-101")
    assert "form" in tokens
    assert "101" in tokens
    assert any("อนุมัติ" in t or "อนุ" in t for t in tokens)


def test_bm25_index_search_and_filters():
    docs = [
        ("c1", "ผู้จัดการสาขากดปุ่มแก้ไขข้อมูลเมื่อรายการยังไม่อนุมัติ", {"role": "branch_manager", "document": "สาขา"}),
        ("c2", "ผู้จัดการเขตกดปุ่มอนุมัติรายการของสาขาในระบบ", {"role": "district_manager", "document": "เขต"}),
        ("c3", "การออกรายงานอุบัติเหตุประจำเดือน", {"role": "branch_manager", "document": "สาขา"}),
    ]
    bm25 = BM25Index().fit(docs)
    hits = bm25.search("ออกรายงานอุบัติเหตุ", k=2)
    assert len(hits) >= 1
    assert hits[0].chunk_id == "c3"

    # Filtered search
    filtered = bm25.search("อนุมัติ", k=5, filters={"role": "district_manager"})
    assert [h.chunk_id for h in filtered] == ["c2"]

    with pytest.raises(ValueError, match="Unknown filter"):
        bm25.search("อนุมัติ", k=1, filters={"unknown": "val"})


def test_reciprocal_rank_fusion_combines_rankings():
    dense_hits = [
        Hit("c1", 0.92, {"document": "d1"}),
        Hit("c2", 0.85, {"document": "d1"}),
        Hit("c3", 0.70, {"document": "d2"}),
    ]
    bm25_hits = [
        Hit("c2", 12.4, {"document": "d1"}),
        Hit("c4", 9.1, {"document": "d3"}),
        Hit("c1", 5.2, {"document": "d1"}),
    ]
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits], k_rrf=60, top_k=3)
    assert len(fused) == 3
    # c2 is rank 2 in dense + rank 1 in bm25 -> 1/62 + 1/61
    # c1 is rank 1 in dense + rank 3 in bm25 -> 1/61 + 1/63
    # (1/62 + 1/61) > (1/61 + 1/63), so c2 should rank first
    assert fused[0].chunk_id == "c2"
    assert fused[1].chunk_id == "c1"


def test_loaded_index_hybrid_search(tmp_path):
    pytest.importorskip("numpy")
    from rag_ingest.chunking import ChunkSettings, CorpusConfig, chunk_corpus, write_run
    from rag_ingest.retrieval.indexing import EmbedderSpec, build_index, latest_index, load_index
    from rag_ingest.tokens import approximate_tokens

    ingest, chunks = tmp_path / "ingest", tmp_path / "chunks"
    ingest.mkdir()
    (ingest / "คู่มือสาขา.md").write_text(
        "<!-- page 1 -->\n1. แก้ไขข้อมูล\n\nกดปุ่มแก้ไขเมื่อรายการยังไม่อนุมัติ\n\n"
        "<!-- page 2 -->\n2. ออกรายงาน\n\nเลือกเดือนแล้วกดปุ่มออกรายงานอุบัติเหตุ\n",
        encoding="utf-8",
    )
    config = CorpusConfig(ingest, tmp_path / "raw", {}, frozenset(), {"คู่มือสาขา": "branch_manager"})
    write_run(chunk_corpus(config, ChunkSettings(), approximate_tokens), chunks, {})
    build_index(EmbedderSpec("fake", {}), chunks, tmp_path / "index")
    index = load_index(latest_index(tmp_path / "index"), ingest)
    assert index.bm25 is not None

    q_vec = index.embedder_spec().create().embed_query("ออกรายงานอุบัติเหตุ")
    hits = index.hybrid_search("ออกรายงานอุบัติเหตุ", q_vec, k=2)
    assert len(hits) >= 1
    assert hits[0].meta["section"] == "2. ออกรายงาน"
