import pytest

from rag_ingest.databases.memory import MemoryChunkStore
from rag_ingest.engines.chat import ChatClient
from rag_ingest.retrieval.conversation import rewrite_query


def test_chat_client_sends_openai_compatible_request(monkeypatch):
    client = ChatClient("https://internal.example/v1/chat/completions", "typhoon-rag", "secret")
    request = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "standalone query"}}]}

    def post(url, **kwargs):
        request.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(client._httpx, "post", post)
    result = client.complete([{"role": "user", "content": "follow-up"}], max_tokens=50)

    assert result == "standalone query"
    assert request["json"] == {
        "model": "typhoon-rag",
        "messages": [{"role": "user", "content": "follow-up"}],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    assert request["headers"] == {"Authorization": "Bearer secret"}


def test_chat_client_rejects_malformed_response(monkeypatch):
    client = ChatClient("https://internal.example/v1/chat/completions", "typhoon-rag")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": []}

    monkeypatch.setattr(client._httpx, "post", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="invalid OpenAI-compatible response"):
        client.complete([{"role": "user", "content": "question"}])


def test_query_rewrite_uses_history_and_returns_only_rewritten_query():
    class FakeChat:
        def __init__(self):
            self.messages = None

        def complete(self, messages, temperature=0.0):
            self.messages = messages
            return "  วิธีแก้ไขรายการที่ยังไม่อนุมัติ  "

    chat = FakeChat()
    history = [
        {"role": "user", "content": "แก้ไขรายการที่อนุมัติแล้วได้ไหม"},
        {"role": "assistant", "content": "ทำได้เฉพาะรายการที่ยังไม่อนุมัติ"},
    ]

    assert rewrite_query(chat, history, "แล้วถ้ายังไม่อนุมัติล่ะ") == "วิธีแก้ไขรายการที่ยังไม่อนุมัติ"
    assert chat.messages == [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user message as one standalone search query. "
                "Use the conversation only to resolve references. Preserve the user's language "
                "and intent, add no facts, and output only the rewritten query."
            ),
        },
        *history,
        {"role": "user", "content": "แล้วถ้ายังไม่อนุมัติล่ะ"},
    ]


def test_query_rewrite_skips_api_when_history_is_empty():
    class UnusedChat:
        def complete(self, *args, **kwargs):
            raise AssertionError("A new conversation does not need query rewriting")

    assert rewrite_query(UnusedChat(), [], "  ค้นหาเอกสาร  ") == "ค้นหาเอกสาร"


def test_memory_store_keeps_history_scoped_to_owner_and_turn_order():
    store = MemoryChunkStore()
    store.create_conversation("conversation-1", "user-1")
    store.save_turn("conversation-1", "user-1", "question", "answer")

    assert store.conversation_history("conversation-1", "user-1") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ValueError, match="does not belong"):
        store.conversation_history("conversation-1", "user-2")
    with pytest.raises(ValueError, match="does not belong"):
        store.create_conversation("conversation-1", "user-2")


def test_memory_store_transaction_rolls_back_conversations():
    store = MemoryChunkStore()

    with pytest.raises(RuntimeError):
        with store.transaction():
            store.create_conversation("conversation-1", "user-1")
            store.save_turn("conversation-1", "user-1", "question", "answer")
            raise RuntimeError("rollback")

    assert store.conversations == {}
    assert store.messages == {}
