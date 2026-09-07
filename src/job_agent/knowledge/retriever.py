from __future__ import annotations

from job_agent.knowledge.contracts import (
    EmbeddingBackend,
    RetrievedChunk,
    VectorIndex,
)


class KnowledgeRetriever:
    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        vector_index: VectorIndex,
    ) -> None:
        self.embedding_backend = embedding_backend
        self.vector_index = vector_index

    def retrieve(
        self,
        *,
        query: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:

        query_vector = self.embedding_backend.embed_query(query)

        hits = self.vector_index.search(query_vector=query_vector,limit=limit)

        return hits