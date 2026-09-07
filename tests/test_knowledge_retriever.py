from typing import Sequence

from job_agent.knowledge.contracts import KnowledgeChunk
from job_agent.knowledge.retriever import KnowledgeRetriever
from job_agent.knowledge.vector_index import InMemoryVectorIndex


class FakeEmbeddingBackend:
    def _embed(self, text: str) -> list[float]:
        lowered = text.casefold()

        if "agent" in lowered or "mcp" in lowered:
            return [1.0, 0.0]

        if "vision" in lowered or "visual" in lowered:
            return [0.0, 1.0]

        return [0.5, 0.5]

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)


def test_knowledge_retriever_returns_relevant_chunk() -> None:
    embedding_backend = FakeEmbeddingBackend()
    vector_index = InMemoryVectorIndex()

    chunks = [
        KnowledgeChunk(
            chunk_id="agent-runtime",
            text="Implemented an Agent Runtime and MCP integration.",
            source="projects/job_agent.md",
        ),
        KnowledgeChunk(
            chunk_id="visual-grounding",
            text="Built a visual grounding system for object localization.",
            source="projects/scvib.md",
        ),
    ]

    vectors = embedding_backend.embed_documents(
        [chunk.text for chunk in chunks]
    )

    vector_index.add(
        chunks=chunks,
        vectors=vectors,
    )

    retriever = KnowledgeRetriever(
        embedding_backend=embedding_backend,
        vector_index=vector_index,
    )

    hits = retriever.retrieve(
        query="What Agent Runtime work have I done?",
        limit=1,
    )

    assert len(hits) == 1
    assert hits[0].chunk.chunk_id == "agent-runtime"