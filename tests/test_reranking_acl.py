import pytest

from rag_ingest.databases.memory import MemoryChunkStore
from rag_ingest.retrieval.access import filter_authorized_hits
from rag_ingest.retrieval.reranker import RerankerClient
from rag_ingest.retrieval.vectorstore import Hit


def test_acl_allows_only_verified_documents_for_matching_role():
    store = MemoryChunkStore()
    store.docs.update({
        "verified-branch": {"status": "verified", "role": "branch_manager"},
        "unverified-branch": {"status": "unverified", "role": "branch_manager"},
        "verified-district": {"status": "verified", "role": "district_manager"},
    })
    store.chunks.update({
        ("run-1", "allowed"): {"document_id": "verified-branch"},
        ("run-1", "unverified"): {"document_id": "unverified-branch"},
        ("run-1", "other-role"): {"document_id": "verified-district"},
    })
    hits = [Hit(chunk_id, 0.9, {}) for chunk_id in ("allowed", "unverified", "other-role")]

    filtered = filter_authorized_hits(hits, store, "branch_manager", "run-1")

    assert [hit.chunk_id for hit in filtered] == ["allowed"]


def test_acl_requires_caller_role():
    with pytest.raises(ValueError, match="caller role"):
        filter_authorized_hits([Hit("c1", 1.0, {})], MemoryChunkStore(), "", "run-1")


def test_reranker_sends_cohere_payload_and_preserves_hit_metadata(monkeypatch):
    client = RerankerClient("https://internal.example/rerank", "bge-reranker-v2-m3", "secret")
    hits = [Hit("c1", 0.8, {"section": "first"}), Hit("c2", 0.7, {"section": "second"})]
    request = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.2},
            ]}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(client._httpx, "post", post)
    ranked = client.rerank("question", hits, {"c1": "text one", "c2": "text two"}, top_n=2)

    assert request["json"] == {
        "model": "bge-reranker-v2-m3",
        "query": "question",
        "documents": ["text one", "text two"],
        "top_n": 2,
    }
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert [hit.chunk_id for hit in ranked] == ["c2", "c1"]
    assert ranked[0].meta == {"section": "second"}


def test_reranker_rejects_bad_response_indices(monkeypatch):
    client = RerankerClient("https://internal.example/rerank", "model")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"results": [{"index": 3, "relevance_score": 0.9}]}

    monkeypatch.setattr(client._httpx, "post", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="candidate index"):
        client.rerank("question", [Hit("c1", 0.8, {})], {"c1": "text"})
