from __future__ import annotations

import json
from typing import Sequence

from job_agent.agent_runtime.contracts import (
    AgentDecisionEnvelope,
    AgentModelContext,
    AgentModelResponse,
)
from job_agent.llm.harness import LLMHarness, NodePolicy
from job_agent.llm.provider import ToolSpec, TraceContext
from job_agent.llm.skill_registry import SkillSpec


class LLMDecisionModel:
    """Adapt the existing structured LLM harness to one Agent decision step.

    Domain tools are described to the model, but the provider is only allowed
    to return an ``AgentDecisionEnvelope``. The runtime, not the provider,
    executes the selected tool after policy authorization.
    """

    def __init__(
        self,
        *,
        harness: LLMHarness,
        skill: SkillSpec,
        tools: Sequence[ToolSpec],
        session_id: str,
        run_id: str,
        agent_id: str,
        prompt_version: str,
        policy: NodePolicy | None = None,
    ) -> None:
        if skill.output_schema is not AgentDecisionEnvelope:
            raise ValueError("agent controller skill must use AgentDecisionEnvelope")
        tool_names = tuple(tool.name for tool in tools)
        if set(tool_names) - set(skill.allowed_tools):
            raise ValueError("agent controller tools exceed skill allowlist")
        self.harness = harness
        self.skill = skill
        self.tools = tuple(tools)
        self.session_id = session_id
        self.run_id = run_id
        self.agent_id = agent_id
        self.prompt_version = prompt_version
        self.policy = policy or NodePolicy(max_retries=2, max_output_tokens=2048)
        self.call_index = 0

    @property
    def traces(self):
        return self.harness.traces

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        self.call_index += 1
        result = self.harness.invoke_structured(
            skill=self.skill,
            task_context={
                "agent_id": self.agent_id,
                "step_count": len(context.steps),
                "budget_usage": context.budget_usage,
                "decision_contract": (
                    "Return decision.kind=action to request exactly one allowed tool, "
                    "or decision.kind=finish with completion evidence."
                ),
            },
            # Goal text, job content and tool observations may contain user or
            # external instructions, so the complete dynamic context remains
            # in the PromptAssembler's untrusted section.
            untrusted_inputs={
                "agent_context_json": json.dumps(
                    context.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            },
            output_schema=AgentDecisionEnvelope,
            tools=self.tools,
            policy=self.policy,
            trace=TraceContext(
                session_id=self.session_id,
                run_id=self.run_id,
                node_id=f"{self.agent_id}:decision:{self.call_index}",
                skill_id=self.skill.skill_id,
                skill_version=self.skill.version,
                prompt_version=self.prompt_version,
            ),
        )
        return AgentModelResponse(
            decision=result.value.decision,
            input_tokens=result.trace.input_tokens,
            output_tokens=result.trace.output_tokens,
            provider=result.trace.provider,
            model=result.trace.model,
        )
