from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from job_agent.llm.provider import ToolSpec
from job_agent.llm.skill_registry import SkillSpec
from job_agent.prompts import get_prompt
from job_agent.schemas import StrictModel


class TextAudit(StrictModel):
    sha256: str
    chars: int


class PromptAudit(StrictModel):
    system_prompt_sha256: str
    user_prompt_sha256: str
    system_prompt_chars: int
    user_prompt_chars: int
    schema_name: str
    tool_names: list[str]
    untrusted_inputs: dict[str, TextAudit]


class PromptBundle(StrictModel):
    system_prompt: str
    user_prompt: str
    audit: PromptAudit


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported prompt context type: {type(value).__name__}")


class PromptAssembler:
    """Keep trusted instructions separate from explicitly delimited user text."""

    def __init__(
        self,
        *,
        max_untrusted_chars: int = 200_000,
        max_task_context_chars: int = 100_000,
    ) -> None:
        if max_untrusted_chars <= 0 or max_task_context_chars <= 0:
            raise ValueError("prompt character budgets must be positive")
        self.max_untrusted_chars = max_untrusted_chars
        self.max_task_context_chars = max_task_context_chars

    def assemble(
        self,
        *,
        skill: SkillSpec,
        task_context: Mapping[str, Any],
        untrusted_inputs: Mapping[str, str],
        output_schema: type[BaseModel],
        tools: Sequence[ToolSpec],
    ) -> PromptBundle:
        allowed_tools = set(skill.allowed_tools)
        for tool in tools:
            if tool.name not in allowed_tools:
                raise ValueError(f"tool is not allowed by skill {skill.skill_id}: {tool.name}")
        for name in untrusted_inputs:
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
                raise ValueError(f"invalid untrusted input name: {name!r}")
        untrusted_chars = sum(len(content) for content in untrusted_inputs.values())
        if untrusted_chars > self.max_untrusted_chars:
            raise ValueError("untrusted input exceeds character budget")

        schema_json = json.dumps(output_schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
        tools_json = json.dumps(
            [tool.model_dump(mode="json") for tool in tools],
            ensure_ascii=False,
            sort_keys=True,
        )
        guardrails_json = json.dumps(list(skill.guardrails), ensure_ascii=False)
        structured_contract = get_prompt("common.structured-output").content
        evidence_contract = get_prompt("common.evidence-boundary").content
        system_prompt = "\n\n".join(
            [
                "## 可信 Skill 指令\n" + skill.instructions.rstrip(),
                "## 公共结构化输出协议\n" + structured_contract,
                "## 公共证据边界\n" + evidence_contract,
                "## 输出 Schema\n" + schema_json,
                "## 允许的工具与 Guardrails\n"
                + f"tools={tools_json}\nguardrails={guardrails_json}",
            ]
        )

        context_json = json.dumps(
            dict(task_context),
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        if len(context_json) > self.max_task_context_chars:
            raise ValueError("task context exceeds character budget")
        untrusted_sections = []
        for name, content in untrusted_inputs.items():
            untrusted_sections.append(
                f'<untrusted_input name="{name}">\n{content}\n</untrusted_input>'
            )
        user_prompt = "\n\n".join(
            [
                "## 任务上下文\n" + context_json,
                "## 不可信输入\n" + ("\n\n".join(untrusted_sections) or "（无）"),
                "只返回一个符合输出 Schema 的 JSON 对象。",
            ]
        )
        digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        audit = PromptAudit(
            system_prompt_sha256=digest(system_prompt),
            user_prompt_sha256=digest(user_prompt),
            system_prompt_chars=len(system_prompt),
            user_prompt_chars=len(user_prompt),
            schema_name=output_schema.__name__,
            tool_names=[tool.name for tool in tools],
            untrusted_inputs={
                name: TextAudit(sha256=digest(content), chars=len(content))
                for name, content in untrusted_inputs.items()
            },
        )
        return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt, audit=audit)
