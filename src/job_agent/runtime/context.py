from __future__ import annotations

from enum import IntEnum

from pydantic import Field

from job_agent.schemas import StrictModel


class ContextPriority(IntEnum):
    CURRENT_TASK = 0
    VERIFIED_RESULT = 1
    CONFIRMED_MEMORY = 2
    HISTORY_SUMMARY = 3


class ContextSegment(StrictModel):
    name: str
    content: str
    priority: ContextPriority


class ContextSelection(StrictModel):
    selected: list[ContextSegment] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    truncated: list[str] = Field(default_factory=list)
    total_chars: int = 0


def select_context(segments: list[ContextSegment], *, max_chars: int) -> ContextSelection:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    selected: list[ContextSegment] = []
    dropped: list[str] = []
    truncated: list[str] = []
    remaining = max_chars
    for segment in sorted(segments, key=lambda item: (item.priority, item.name)):
        if remaining <= 0:
            dropped.append(segment.name)
            continue
        if len(segment.content) <= remaining:
            selected.append(segment)
            remaining -= len(segment.content)
            continue
        if segment.priority == ContextPriority.CURRENT_TASK:
            selected.append(segment.model_copy(update={"content": segment.content[:remaining]}))
            truncated.append(segment.name)
            remaining = 0
        else:
            dropped.append(segment.name)
    return ContextSelection(
        selected=selected,
        dropped=dropped,
        truncated=truncated,
        total_chars=sum(len(segment.content) for segment in selected),
    )
