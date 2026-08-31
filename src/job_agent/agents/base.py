from __future__ import annotations

from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

from job_agent.llm.provider import AgentResult, ProviderTrace
from job_agent.llm.skill_registry import SkillSpec


class SkillLookup(Protocol):
    def get(self, skill_id: str) -> SkillSpec:
        ...


class AgentGuardError(ValueError):
    def __init__(self, error_code: str, trace: ProviderTrace, *, node_id: str):
        self.error_code = error_code
        self.trace = trace
        self.node_id = node_id
        super().__init__(f"{node_id} guard failed: {error_code}")


GuardedT = TypeVar("GuardedT", bound=BaseModel)
GuardFeedback = dict[str, str]


def invoke_with_guard_repair(
    *,
    invoke: Callable[[GuardFeedback | None], AgentResult[GuardedT]],
    guard: Callable[[AgentResult[GuardedT]], None],
    repair_instruction: str,
    max_repairs: int = 1,
) -> AgentResult[GuardedT]:
    """Retry schema-valid semantic output only after a deterministic guard failure."""
    feedback: GuardFeedback | None = None
    for attempt in range(max_repairs + 1):
        result = invoke(feedback)
        try:
            guard(result)
        except AgentGuardError as exc:
            if attempt >= max_repairs:
                raise
            feedback = {
                "error_code": exc.error_code,
                "instruction": repair_instruction,
            }
            continue
        return result
    raise AssertionError("unreachable semantic guard repair loop")


def reject_forbidden_output(
    rendered_output: str,
    *,
    forbidden_output_strings: tuple[str, ...],
    trace: ProviderTrace,
    node_id: str,
) -> None:
    if any(value and value in rendered_output for value in forbidden_output_strings):
        raise AgentGuardError("untrusted_input_leakage", trace, node_id=node_id)
