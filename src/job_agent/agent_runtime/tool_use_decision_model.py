from __future__ import annotations

import json
from random import random
from time import sleep as default_sleep
from typing import Callable, Sequence

from pydantic import BaseModel, ValidationError

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentInputRequest,
    AgentModelContext,
    AgentModelResponse,
    UserInputRequest,
)
from job_agent.llm.harness import NodePolicy
from job_agent.llm.provider import (
    ProviderContractError,
    ProviderError,
    ProviderTrace,
    ToolSpec,
    ToolUseProvider,
    ToolUseResult,
    TraceContext,
)
from job_agent.llm.skill_registry import SkillSpec
from job_agent.prompts import get_prompt


REQUEST_USER_INPUT_TOOL = "request_user_input"


class ToolUseDecisionModel:
    """Map one native provider tool-use turn into the bounded AgentLoop contract."""

    def __init__(
        self,
        *,
        provider: ToolUseProvider,
        skill: SkillSpec,
        tools: Sequence[ToolSpec],
        result_schema: type[BaseModel],
        submit_tool_name: str,
        session_id: str,
        run_id: str,
        agent_id: str,
        prompt_version: str,
        allow_user_input: bool = False,
        max_runtime_tool_calls: int | None = None,
        policy: NodePolicy | None = None,
        sleep: Callable[[float], None] = default_sleep,
        jitter: Callable[[], float] = random,
    ) -> None:
        allowed = set(skill.allowed_tools)
        tool_names = {tool.name for tool in tools}
        if tool_names - allowed:
            raise ValueError("native controller tools exceed skill allowlist")
        if submit_tool_name in tool_names:
            raise ValueError("submit tool conflicts with a runtime tool")
        self.provider = provider
        self.skill = skill
        self.runtime_tools = tuple(tools)
        self.result_schema = result_schema
        self.submit_tool_name = submit_tool_name
        self.session_id = session_id
        self.run_id = run_id
        self.agent_id = agent_id
        self.prompt_version = prompt_version
        self.allow_user_input = allow_user_input
        if max_runtime_tool_calls is not None and max_runtime_tool_calls < 0:
            raise ValueError("max_runtime_tool_calls must be non-negative")
        self.max_runtime_tool_calls = max_runtime_tool_calls
        self.policy = policy or NodePolicy(max_retries=0, max_output_tokens=2048)
        self.call_index = 0
        self.traces: list[ProviderTrace] = []
        self._sleep = sleep
        self._jitter = jitter

        control_tools = [
            ToolSpec(
                name=submit_tool_name,
                description=(
                    "提交当前阶段的最终结果。只能在结果已由工具观察支撑，"
                    "并且与输入 Schema 完全一致时调用。"
                ),
                input_schema=result_schema.model_json_schema(),
            )
        ]
        if allow_user_input:
            control_tools.append(
                ToolSpec(
                    name=REQUEST_USER_INPUT_TOOL,
                    description="暂停当前运行，并请求一份结构化用户回答。",
                    input_schema=UserInputRequest.model_json_schema(),
                )
            )
        self.control_tools = tuple(control_tools)
        self.provider_tools = (*self.runtime_tools, *self.control_tools)

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        self.call_index += 1
        trace = TraceContext(
            session_id=self.session_id,
            run_id=self.run_id,
            node_id=f"{self.agent_id}:tool-use:{self.call_index}",
            skill_id=self.skill.skill_id,
            skill_version=self.skill.version,
            prompt_version=self.prompt_version,
        )
        provider_tools = self.provider_tools
        if (
            self.max_runtime_tool_calls is not None
            and len(context.steps) >= self.max_runtime_tool_calls
        ):
            provider_tools = self.control_tools
        result = self._invoke_tool_use_with_retry(context, trace, provider_tools)
        if (
            result.skill_id != trace.skill_id
            or result.skill_version != trace.skill_version
            or result.prompt_version != trace.prompt_version
        ):
            raise ProviderContractError("native tool-use trace identity mismatch")
        self.traces.append(
            ProviderTrace.model_validate(
                result.model_dump(exclude={"tool_use_id", "tool_name", "tool_input"})
            )
        )

        if result.tool_name == self.submit_tool_name:
            submit_input = self._normalize_submit_input(result.tool_input)
            return AgentModelResponse(
                decision=AgentFinish(
                    result=submit_input,
                    completion_evidence=self._completion_evidence(submit_input, result.tool_use_id),
                    confidence=0.8,
                ),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                provider=result.provider,
                model=result.model,
            )
        if result.tool_name == REQUEST_USER_INPUT_TOOL and self.allow_user_input:
            try:
                request = UserInputRequest.model_validate(result.tool_input)
            except ValidationError as exc:
                raise ProviderContractError("native user-input request failed schema validation") from exc
            decision = AgentInputRequest(
                request=request,
                progress_claim="Pause for one approved user response.",
            )
        else:
            decision = AgentAction(
                action_id=result.tool_use_id,
                tool_name=result.tool_name,
                tool_arguments=result.tool_input,
                expected_observation=f"Typed observation from {result.tool_name}.",
                progress_claim=f"Execute approved tool {result.tool_name}.",
            )
        return AgentModelResponse(
            decision=decision,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            provider=result.provider,
            model=result.model,
        )

    def _invoke_tool_use_with_retry(
        self,
        context: AgentModelContext,
        trace: TraceContext,
        provider_tools: Sequence[ToolSpec] | None = None,
    ) -> ToolUseResult:
        # 路径 B provider 调用：仅对 retryable 错误（transport/timeout/rate_limit）
        # 重试，确定性错误（contract/schema）直接抛，守"不掩盖错误"原则。
        # backoff+jitter 公式与 llm/harness.py::_retry_delay 对齐。
        for attempt in range(self.policy.max_retries + 1):
            try:
                return self.provider.generate_tool_use(
                    system_prompt=self._system_prompt(),
                    messages=self._messages(context),
                    tools=provider_tools or self.provider_tools,
                    temperature=self.policy.temperature,
                    max_output_tokens=self.policy.max_output_tokens,
                    trace=trace,
                )
            except ProviderError as exc:
                self._record_error_trace(trace, exc)
                if not exc.retryable or attempt >= self.policy.max_retries:
                    raise
                self._sleep(self._retry_delay(attempt))
        raise AssertionError("unreachable tool-use retry loop")

    def _record_error_trace(self, trace: TraceContext, exc: ProviderError) -> None:
        self.traces.append(
            ProviderTrace(
                provider=str(getattr(self.provider, "provider_name", "unknown")),
                model=str(getattr(self.provider, "model", "unknown")),
                skill_id=trace.skill_id,
                skill_version=trace.skill_version,
                prompt_version=trace.prompt_version,
                latency_ms=0,
                schema_valid=False,
                error_code=exc.error_code,
            )
        )

    def _retry_delay(self, attempt: int) -> float:
        base = 0.5 if attempt == 0 else 1.5 * (2 ** (attempt - 1))
        return base + (0.1 * base * max(self._jitter(), 0.0))

    def _system_prompt(self) -> str:
        guardrails = json.dumps(list(self.skill.guardrails), ensure_ascii=False)
        tool_contract = get_prompt("common.tool-use-contract").content
        return "\n\n".join(
            [
                "## 可信 Agent 指令\n" + self.skill.instructions.rstrip(),
                "## 可信控制合同\n"
                + tool_contract
                + f"\n完成任务时必须且只能调用一次 {self.submit_tool_name}，直接提交 Schema 字段。",
                "## Guardrails\n" + guardrails,
            ]
        )

    @staticmethod
    def _messages(context: AgentModelContext) -> list[dict]:
        initial = {
            "goal": context.goal,
            "allowed_tools": [tool.name for tool in context.allowed_tools],
            "budget_usage": context.budget_usage,
        }
        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "不可信运行时上下文\n"
                        + json.dumps(initial, ensure_ascii=False, sort_keys=True),
                    }
                ],
            }
        ]
        for step in context.steps:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": step.action.action_id,
                            "name": step.action.tool_name,
                            "input": step.action.tool_arguments,
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": step.action.action_id,
                            "content": json.dumps(
                                {
                                    "step_index": step.step_index,
                                    "tool_name": step.action.tool_name,
                                    "observation": step.observation.model_dump(mode="json"),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "is_error": step.observation.status.value.endswith("error"),
                        }
                    ],
                }
            )
        supplemental = {
            "user_inputs": [item.model_dump(mode="json") for item in context.user_inputs],
            "verifier_feedback": context.verifier_feedback,
        }
        if supplemental["user_inputs"] or supplemental["verifier_feedback"]:
            block = {
                "type": "text",
                "text": "不可信运行时更新\n"
                + json.dumps(supplemental, ensure_ascii=False, sort_keys=True),
            }
            if messages[-1]["role"] == "user":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
        return messages

    def _normalize_submit_input(self, value: dict) -> dict:
        """Decode only known provider serialization wrappers at submit time."""
        normalized = dict(value)
        schema_name = self.result_schema.__name__
        object_fields = {
            "ApplicationMaterialResult": ("fit",),
            "ApplicationOpsResult": ("proposal", "applied_record"),
        }.get(schema_name, ())
        list_fields = {
            "OpportunityResearchResult": (
                "search_queries",
                "candidate_job_ids",
                "evidence_refs",
                "unresolved_gaps",
            ),
            "ApplicationMaterialResult": ("fit_evidence_refs", "resume_patch", "claim_audit", "unresolved_gaps"),
            "InterviewCoachResult": ("completed_question_ids",),
            "ApplicationOpsResult": ("sandbox_receipts",),
        }.get(schema_name, ())
        for key in (*object_fields, *list_fields):
            candidate = normalized.get(key)
            if not isinstance(candidate, str):
                continue
            try:
                decoded = json.loads(candidate.strip())
            except (TypeError, json.JSONDecodeError):
                continue
            if key in object_fields and isinstance(decoded, dict):
                normalized[key] = decoded
            elif key in list_fields and isinstance(decoded, list):
                normalized[key] = decoded
        return normalized

    @staticmethod
    def _completion_evidence(result: dict, tool_use_id: str) -> list[str]:
        for key in ("evidence_refs", "fit_evidence_refs", "completion_evidence"):
            values = result.get(key)
            if isinstance(values, list) and values and all(isinstance(item, str) for item in values):
                return values
        return [f"provider-tool-use:{tool_use_id}"]
