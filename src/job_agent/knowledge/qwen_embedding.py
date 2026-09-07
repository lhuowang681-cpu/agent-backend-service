from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sentence_transformers import SentenceTransformer


class QwenEmbeddingBackend:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
    ) -> None:
        self.model = SentenceTransformer(
            str(model_path),
            device=device,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embedding = self.model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
        )

        return embedding.tolist()