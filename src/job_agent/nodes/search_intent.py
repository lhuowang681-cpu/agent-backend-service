from __future__ import annotations

from job_agent.schemas import SearchIntent


POSTTRAINING_KEYWORDS = [
    "后训练",
    "RLHF",
    "RLAIF",
    "SFT",
    "LoRA",
    "DPO",
    "GRPO",
    "reward model",
    "偏好对齐",
    "大模型评测",
]


def build_search_intent(user_request: str, cities: list[str] | None = None) -> SearchIntent:
    selected_cities = cities or ["北京", "上海", "杭州", "成都"]
    return SearchIntent(
        target_role="posttraining_rlhf",
        cities=selected_cities,
        keywords=POSTTRAINING_KEYWORDS,
    )
