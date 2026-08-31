from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Sequence

from job_agent.memory.contracts import (
    MemoryHit,
    MemoryKind,
    MemoryRetriever as MemoryRetrieverProtocol,
    MemorySource,
)
from job_agent.memory.user_state_store import UserStateStore


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class MemoryRetriever(MemoryRetrieverProtocol):
    def __init__(self, store: UserStateStore, *, now=lambda: datetime.now(UTC)) -> None:
        self.store = store
        self._now = now

    def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        kinds: Sequence[MemoryKind] | None = None,
        limit: int = 5,
        include_unconfirmed: bool = False,
    ) -> list[MemoryHit]:
        if limit <= 0:
            return []
        query_tokens = {token.casefold() for token in _TOKEN_RE.findall(query)}
        allowed_kinds = set(kinds) if kinds is not None else None
        hits: list[MemoryHit] = []
        for record in self.store.load(user_id=user_id).memories:
            if allowed_kinds is not None and record.kind not in allowed_kinds:
                continue
            if not include_unconfirmed and record.source == MemorySource.MODEL_SUGGESTED:
                continue
            if record.expires_at and datetime.fromisoformat(record.expires_at) <= self._now():
                continue
            content_tokens = {token.casefold() for token in _TOKEN_RE.findall(record.content)}
            overlap = query_tokens & content_tokens
            if query_tokens and not overlap and query.casefold() not in record.content.casefold():
                continue
            score = 1.0 if query and query.casefold() in record.content.casefold() else (
                len(overlap) / max(len(query_tokens), 1)
            )
            hits.append(
                MemoryHit(
                    record=record,
                    score=min(max(score, 0.0), 1.0),
                    reason=f"matched {len(overlap)} query tokens; source={record.source.value}",
                    artifact_refs=list(record.artifact_refs),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, -hit.record.confidence, hit.record.memory_id))
        return hits[:limit]
