from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from job_agent.llm.skill_registry import SkillSpec


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def with_output_schema(base: SkillSpec, output_schema: type[SchemaT]) -> SkillSpec:
    """Create a local schema view without mutating the release registry or files."""
    return SkillSpec(
        skill_id=base.skill_id,
        version=base.version,
        instructions=base.instructions,
        reference_paths=base.reference_paths,
        output_schema=output_schema,
        allowed_tools=base.allowed_tools,
        guardrails=base.guardrails,
        examples=base.examples,
    )
