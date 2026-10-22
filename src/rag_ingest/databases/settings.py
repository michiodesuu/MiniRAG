"""Connection settings, read from the environment only (see .env.example).

Nothing here has a password default: a missing POSTGRES_PASSWORD is an error,
not a silent fallback. A .env file in the project root is read when present,
without overriding variables that are already set.
"""
from dataclasses import dataclass
from pathlib import Path
import os

from rag_ingest.config import ENCODING


def read_env_file(path: Path) -> None:
    """KEY=VALUE lines into os.environ, keeping what is already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding=ENCODING).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        password = os.environ.get("POSTGRES_PASSWORD")
        if not password:
            raise RuntimeError("POSTGRES_PASSWORD is not set (copy .env.example to .env)")
        return cls(os.environ.get("POSTGRES_HOST", "localhost"), int(os.environ.get("POSTGRES_PORT", "5432")),
                   os.environ.get("POSTGRES_DB", "rag"), os.environ.get("POSTGRES_USER", "rag"), password)

    def connect_kwargs(self) -> dict:
        return {"host": self.host, "port": self.port, "dbname": self.dbname,
                "user": self.user, "password": self.password}


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    api_key: str | None

    @classmethod
    def from_env(cls) -> "QdrantSettings":
        return cls(os.environ.get("QDRANT_URL", "http://localhost:6333"), os.environ.get("QDRANT_API_KEY") or None)
