from types import SimpleNamespace

import pytest

from rag_ingest.cli import build_parser
from rag_ingest.databases.memory import MemoryChunkStore
from rag_ingest.retrieval.answer import NO_EVIDENCE_ANSWER, generate_grounded_answer
from rag_ingest.retrieval.vectorstore import Hit


def test_grounded_answer_includes_only_retrieved_evidence_and_history():
    class FakeChat:
        def __init__(self):
            self.messages = None

        def complete(self, messages, temperature=0.0):
            self.messages = messages
            return "ทำได้เมื่อรายการยังไม่อนุมัติ [คู่มือ#p2#c01]"

    client = FakeChat()
    history = [{"role": "user", "content": "ถามก่อนหน้า"}]
    answer = generate_grounded_answer(
        client,
        "ทำอย่างไร",
        "วิธีดำเนินการ",
        [Hit("คู่มือ#p2#c01", 0.9, {"page_start": 2, "page_end": 2})],
        {"คู่มือ#p2#c01": "กดปุ่มแก้ไขเมื่อรายการยังไม่อนุมัติ"},
        history,
    )

    assert answer.endswith("[คู่มือ#p2#c01]")
    assert history[0] in client.messages
    assert "กดปุ่มแก้ไขเมื่อรายการยังไม่อนุมัติ" in client.messages[-1]["content"]
    assert "untrusted data" in client.messages[0]["content"]


def test_grounded_answer_returns_no_evidence_message_without_api_call():
    class UnusedChat:
        def complete(self, *args, **kwargs):
            raise AssertionError("No evidence should not be sent to the model")

    assert generate_grounded_answer(UnusedChat(), "question", "query", [], {}) == NO_EVIDENCE_ANSWER


def test_ask_cli_requires_positive_retrieval_limits(paths):
    parser = build_parser(paths)
    with pytest.raises(SystemExit):
        parser.parse_args([
            "ask", "question", "--user-id", "user-1", "--role", "branch_manager", "-k", "0",
        ])


def test_ask_cli_pipeline_applies_acl_before_reranking(monkeypatch, paths, capsys):
    from rag_ingest.databases.postgres import PostgresChunkStore
    from rag_ingest.engines.chat import ChatClient
    from rag_ingest.retrieval.reranker import RerankerClient

    store = MemoryChunkStore()
    store.docs.update({
        "authorized": {"role": "branch_manager", "status": "verified"},
        "unverified": {"role": "branch_manager", "status": "unverified"},
    })
    store.chunks.update({
        ("run-1", "allowed"): {"document_id": "authorized"},
        ("run-1", "denied"): {"document_id": "unverified"},
    })

    class FakeEmbedder:
        @staticmethod
        def embed_query(query):
            assert query == "วิธีแก้ไข"
            return [0.1, 0.2]

    class FakeIndex:
        texts = {"allowed": "หลักฐานที่ได้รับอนุญาต", "denied": "ข้อมูลที่ยังไม่ยืนยัน"}

        @staticmethod
        def embedder_spec():
            return SimpleNamespace(create=lambda: FakeEmbedder())

        @staticmethod
        def hybrid_search(query, vector, k, filters):
            assert filters["role"] == "branch_manager"
            return [Hit("allowed", 0.9, {}), Hit("denied", 0.8, {})]

    index = FakeIndex()
    monkeypatch.setattr(PostgresChunkStore, "connect", lambda: store)
    monkeypatch.setattr("rag_ingest.retrieval.indexing.load_index", lambda folder, ingest: index)

    class FakeChat:
        @staticmethod
        def complete(messages, temperature=0.0):
            return "คำตอบจากเอกสาร [allowed]"

    class FakeReranker:
        @staticmethod
        def rerank(query, hits, texts, top_n):
            assert [hit.chunk_id for hit in hits] == ["allowed"]
            return hits[:top_n]

    monkeypatch.setattr(ChatClient, "from_env", lambda: FakeChat())
    monkeypatch.setattr(RerankerClient, "from_env", lambda: FakeReranker())

    args = build_parser(paths).parse_args([
        "ask", "วิธีแก้ไข", "--user-id", "user-1", "--role", "branch_manager",
        "--index", "run-1",
    ])
    args.func(args, paths)

    output = capsys.readouterr().out
    assert "Conversation:" in output
    assert "คำตอบจากเอกสาร [allowed]" in output
    assert store.conversation_history(output.split("Conversation: ", 1)[1].splitlines()[0], "user-1")
