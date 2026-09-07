import os

import pytest

from job_agent.knowledge.contracts import KnowledgeChunk
from job_agent.knowledge.qwen_embedding import QwenEmbeddingBackend
from job_agent.knowledge.retriever import KnowledgeRetriever
from job_agent.knowledge.vector_index import InMemoryVectorIndex


MODEL_PATH = os.environ.get("QWEN_EMBEDDING_MODEL_PATH")


@pytest.mark.skipif(
    not MODEL_PATH,
    reason="QWEN_EMBEDDING_MODEL_PATH is not set",
)
def test_real_embedding_retrieves_semantic_match() -> None:
    backend = QwenEmbeddingBackend(
        model_path=MODEL_PATH,
    )

    chunks = [
        KnowledgeChunk(
            chunk_id="agent-runtime",
            text=(
                "Implemented an Agent Runtime with tool execution, "
                "policy enforcement, checkpoints, and MCP integration."
            ),
            source="projects/job_agent.md",
        ),
        KnowledgeChunk(
            chunk_id="visual-grounding",
            text=(
                "Built a visual grounding system for object "
                "localization across images."
            ),
            source="projects/scvib.md",
        ),
        KnowledgeChunk(
            chunk_id="materials",
            text=(
                "Studied material properties and metallurgical "
                "engineering."
            ),
            source="education.md",
        ),
    ]

    vectors = backend.embed_documents(
        [chunk.text for chunk in chunks]
    )

    index = InMemoryVectorIndex()
    index.add(
        chunks=chunks,
        vectors=vectors,
    )

    retriever = KnowledgeRetriever(
        embedding_backend=backend,
        vector_index=index,
    )

    hits = retriever.retrieve(
        query="What experience do I have building agent infrastructure?",
        limit=3,
    )

    for hit in hits:
        print(
            hit.chunk.chunk_id,
            round(hit.score, 4),
        )

    assert hits[0].chunk.chunk_id == "agent-runtime"
    assert hits[0].score > hits[1].score