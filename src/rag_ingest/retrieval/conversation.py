"""Conversation-aware query rewriting."""
from typing import Protocol


class ChatCompletion(Protocol):
    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str: ...


def rewrite_query(
    client: ChatCompletion,
    history: list[dict[str, str]],
    query: str,
) -> str:
    """Rewrite a follow-up as a standalone search query without answering it."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty")
    if not history:
        return query
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user message as one standalone search query. "
                "Use the conversation only to resolve references. Preserve the user's language "
                "and intent, add no facts, and output only the rewritten query."
            ),
        },
        *history,
        {"role": "user", "content": query},
    ]
    rewritten = client.complete(messages, temperature=0.0).strip()
    if not rewritten:
        raise ValueError("Query rewrite returned empty content")
    return rewritten
