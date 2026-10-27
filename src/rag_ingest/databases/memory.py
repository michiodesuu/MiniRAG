"""MemoryChunkStore: PostgresChunkStore's interface in memory, for tests and CI.

transaction() works on a copy and keeps it only when the block ends without an
error, like the PostgreSQL transaction it stands in for.
"""
from contextlib import contextmanager
import copy


class MemoryChunkStore:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.chunks: dict[tuple[str, str], dict] = {}  # (index_run, chunk_id) -> row
        self.conversations: dict[str, dict] = {}
        self.messages: dict[str, list[dict[str, str]]] = {}

    def migrate(self) -> list[str]:
        return []

    def close(self) -> None:
        pass

    @contextmanager
    def transaction(self):
        saved = copy.deepcopy((self.docs, self.chunks, self.conversations, self.messages))
        try:
            yield self
        except BaseException:
            self.docs, self.chunks, self.conversations, self.messages = saved
            raise

    def document(self, document_id: str) -> dict | None:
        return self.docs.get(document_id)

    def documents(self) -> list[dict]:
        return [self.docs[k] for k in sorted(self.docs)]

    def chunk_ids(self, document_id: str, index_run: str) -> list[str]:
        return sorted(c for (run, c), row in self.chunks.items()
                      if run == index_run and row["document_id"] == document_id)

    def all_chunk_ids(self) -> list[str]:
        return sorted(c for _, c in self.chunks)

    def authorized_chunk_ids(self, role: str, index_run: str, chunk_ids: list[str]) -> set[str]:
        if not role:
            raise ValueError("A caller role is required for ACL filtering")
        candidates = set(chunk_ids)
        return {
            chunk_id
            for (run, chunk_id), chunk in self.chunks.items()
            if run == index_run
            and chunk_id in candidates
            and self.docs[chunk["document_id"]]["status"] == "verified"
            and self.docs[chunk["document_id"]]["role"] == role
        }

    def create_conversation(self, conversation_id: str, owner_id: str) -> None:
        if not conversation_id or not owner_id:
            raise ValueError("Conversation ID and owner ID must not be empty")
        existing = self.conversations.get(conversation_id)
        if existing and existing["owner_id"] != owner_id:
            raise ValueError("Conversation does not belong to this user")
        self.conversations.setdefault(conversation_id, {"owner_id": owner_id})
        self.messages.setdefault(conversation_id, [])

    def conversation_history(
        self, conversation_id: str, owner_id: str, limit: int = 20
    ) -> list[dict[str, str]]:
        if limit <= 0:
            raise ValueError("History limit must be greater than zero")
        conversation = self.conversations.get(conversation_id)
        if not conversation or conversation["owner_id"] != owner_id:
            raise ValueError("Conversation does not belong to this user")
        return copy.deepcopy(self.messages[conversation_id][-limit:])

    def save_turn(
        self, conversation_id: str, owner_id: str, user_text: str, assistant_text: str
    ) -> None:
        if not user_text or not assistant_text:
            raise ValueError("Conversation messages must not be empty")
        conversation = self.conversations.get(conversation_id)
        if not conversation or conversation["owner_id"] != owner_id:
            raise ValueError("Conversation does not belong to this user")
        self.messages[conversation_id].extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])

    def index_runs(self, document_id: str) -> list[str]:
        return sorted({run for (run, _), row in self.chunks.items() if row["document_id"] == document_id})

    def replace_document(self, document: dict, chunks: list[dict], index_run: str) -> None:
        old = self.docs.get(document["id"])
        every_run = old is not None and old["content_hash"] != document["content_hash"]
        self.docs[document["id"]] = dict(document)
        self.chunks = {key: row for key, row in self.chunks.items()
                       if row["document_id"] != document["id"] or (not every_run and key[0] != index_run)}
        for c in chunks:
            if (index_run, c["chunk_id"]) in self.chunks:
                raise ValueError(f"duplicate key {c['chunk_id']}")  # as the primary key would
            self.chunks[(index_run, c["chunk_id"])] = dict(c)

    def delete_document(self, document_id: str) -> None:
        self.docs.pop(document_id, None)
        self.chunks = {k: r for k, r in self.chunks.items() if r["document_id"] != document_id}
