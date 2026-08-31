from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from pydantic import Field, TypeAdapter

from job_agent.schemas import StrictModel


class QuestionAnchor(StrictModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    topics: list[str] = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    suitable_roles: list[str] = Field(min_length=1)
    verification_goal: str = Field(min_length=1)
    question: str = Field(min_length=1)
    follow_up_axes: list[str] = Field(default_factory=list)
    reference_points: list[str] = Field(min_length=1)

    def public_view(self) -> dict:
        return self.model_dump(exclude={"reference_points"}, mode="json")


_ANCHOR_LIST = TypeAdapter(list[QuestionAnchor])


class QuestionAnchorLibrary:
    def __init__(self, anchors: list[QuestionAnchor] | None = None) -> None:
        self._anchors = list(anchors) if anchors is not None else self._load_packaged()

    @staticmethod
    def _load_packaged() -> list[QuestionAnchor]:
        resource_root = files("job_agent.interview").joinpath("resources")
        anchors: list[QuestionAnchor] = []
        for item in sorted(resource_root.iterdir(), key=lambda entry: entry.name):
            if item.name.endswith(".json"):
                anchors.extend(_ANCHOR_LIST.validate_json(item.read_text(encoding="utf-8")))
        return anchors

    def search(
        self,
        *,
        topics: set[str],
        roles: set[str],
        difficulty: str | None,
        seed: str,
        limit: int = 5,
    ) -> list[QuestionAnchor]:
        if limit < 1:
            return []
        normalized_topics = {value.strip().casefold() for value in topics if value.strip()}
        normalized_roles = {value.strip().casefold() for value in roles if value.strip()}
        candidates = []
        for anchor in self._anchors:
            anchor_topics = {value.casefold() for value in anchor.topics}
            anchor_roles = {value.casefold() for value in anchor.suitable_roles}
            if normalized_topics and not normalized_topics.intersection(anchor_topics):
                continue
            if normalized_roles and not normalized_roles.intersection(anchor_roles):
                continue
            if difficulty and anchor.difficulty != difficulty:
                continue
            candidates.append(anchor)
        candidates.sort(
            key=lambda anchor: hashlib.sha256(f"{seed}:{anchor.id}".encode("utf-8")).hexdigest()
        )
        return candidates[:limit]
