"""OpenAI-compatible chat-completions client for the internal chat service."""
import os
from urllib.parse import urlsplit

DEFAULT_TIMEOUT = 60


def _endpoint(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.hostname}:{parts.port}" if parts.port else (parts.hostname or "")


class ChatClient:
    def __init__(
        self, url: str, model: str, key: str | None = None, timeout: int = DEFAULT_TIMEOUT
    ):
        import httpx

        if not url:
            raise ValueError("Chat API URL must not be empty")
        if not model:
            raise ValueError("Chat model must not be empty")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Chat API URL must be an absolute HTTP(S) URL")
        self._httpx = httpx
        self.url = url
        self.model = model
        self._key = key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "ChatClient":
        url = os.environ.get("CHAT_API_URL")
        model = os.environ.get("CHAT_MODEL")
        if not url or not model:
            raise RuntimeError("CHAT_API_URL and CHAT_MODEL must be set")
        return cls(url, model, os.environ.get("CHAT_API_KEY") or None)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        if not messages:
            raise ValueError("At least one chat message is required")
        payload: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be greater than zero")
            payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            response = self._httpx.post(
                self.url, json=payload, headers=headers, timeout=self.timeout
            )
        except self._httpx.HTTPError as error:
            raise RuntimeError(
                f"Chat API at {_endpoint(self.url)} request failed ({type(error).__name__})"
            ) from error
        if response.status_code >= 400:
            raise RuntimeError(
                f"Chat API at {_endpoint(self.url)} answered HTTP {response.status_code}"
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Chat API returned an invalid OpenAI-compatible response") from error
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat API returned empty or non-text content")
        return content.strip()
