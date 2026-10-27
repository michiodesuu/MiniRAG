"""PostgreSQL-backed document access checks for retrieval candidates."""
from typing import Protocol

from rag_ingest.retrieval.vectorstore import Hit


class AclStore(Protocol):
    def authorized_chunk_ids(self, role: str, index_run: str, chunk_ids: list[str]) -> set[str]: ...


def filter_authorized_hits(
    hits: list[Hit],
    store: AclStore,
    role: str,
    index_run: str,
) -> list[Hit]:
    """Keep only chunks belonging to verified documents assigned to the caller's role."""
    if not role:
        raise ValueError("A caller role is required for ACL filtering")
    if not hits:
        return []
    authorized = store.authorized_chunk_ids(role, index_run, [hit.chunk_id for hit in hits])
    return [hit for hit in hits if hit.chunk_id in authorized]
