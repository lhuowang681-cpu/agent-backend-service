import pytest

from job_agent.knowledge.vector_index import cosine_similarity


from job_agent.knowledge.contracts import KnowledgeChunk
from job_agent.knowledge.vector_index import (
    InMemoryVectorIndex,
    cosine_similarity,
)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)
def test_cosine_similarity_same_direction_is_one() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert result == pytest.approx(1.0)
    
def test_vector_index_returns_most_similar_chunks_first() -> None:
    index = InMemoryVectorIndex()

    chunks = [
        KnowledgeChunk(
            chunk_id="a",
            text="Agent Runtime",
            source="project.md",
        ),
        KnowledgeChunk(
            chunk_id="b",
            text="Visual Grounding",
            source="project.md",
        ),
        KnowledgeChunk(
            chunk_id="c",
            text="MCP Runtime Integration",
            source="project.md",
        ),
    ]

    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
    ]

    index.add(
        chunks=chunks,
        vectors=vectors,
    )

    hits = index.search(
        query_vector=[1.0, 0.0],
        limit=2,
    )

    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "a"
    assert hits[1].chunk.chunk_id == "c"