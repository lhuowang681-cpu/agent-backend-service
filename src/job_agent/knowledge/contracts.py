from __future__ import annotations

from typing import Any, Protocol, Sequence

from pydantic import Field

from job_agent.schemas import StrictModel


class KnowledgeChunk(StrictModel):
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(StrictModel):
    chunk: KnowledgeChunk
    score: float


class EmbeddingBackend(Protocol):
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...


class VectorIndex(Protocol):
    def add(
        self,
        *,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        ...

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        ...