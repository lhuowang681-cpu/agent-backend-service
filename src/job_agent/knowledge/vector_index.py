from __future__ import annotations

import math
from typing import Sequence

from job_agent.knowledge.contracts import (
    KnowledgeChunk,
    RetrievedChunk,
)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")

    if not left:
        raise ValueError("vectors must not be empty")

    dot_product = sum(a * b  for a,b in zip(left,right))

    left_norm = math.sqrt(
        sum(a * a for a in left)
    )
    right_norm = math.sqrt(
        sum(a * a for a in right)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._items: list[
            tuple[KnowledgeChunk, list[float]]
        ] = []

    def add(
        self,
        *,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors must have the same length"
            )

        for chunk, vector in zip(chunks, vectors):
            self._items.append(
                (
                    chunk,
                    list(vector),
                )
            )

    def search(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            return []

        hits: list[RetrievedChunk] = []

        for chunk, vector in self._items:
            score = cosine_similarity(vector, query_vector)

            hits.append(RetrievedChunk(chunk=chunk,score=score))

        hits.sort(
            key=lambda hit: hit.score,
            reverse=True,
        )

        return hits[:limit]