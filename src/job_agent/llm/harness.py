from __future__ import annotations

import hashlib
from random import random
from time import perf_counter, sleep as default_sleep
from typing import Any, Callable, Mapping, Sequence

from pydantic import Field, ValidationError

from job_agent.llm.prompt_assembler import PromptAssembler, PromptAudit
from job_agent.llm.provider import (
    AgentResult,
    LLMProvider,
    ProviderContractError,
    ProviderError,
    ProviderResult,
    ProviderTrace,
    StructuredT,
    ToolSpec,
    TraceContext,
)
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import StrictModel


class NodePolicy(StrictModel):
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_schema_repairs: int = Field(default=1, ge=0)
    temperature: float = Field(default=0.0, ge=0.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    allow_rule_fallback: bool = False


class LLMInvocationError(RuntimeError):
    """Sanitized invocation failure; prompts and raw output are intentionally omitted."""

    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(f"structured LLM invocation failed: {error_code}")


class InvocationAuditEvent(StrictModel):
    prompt: PromptAudit
    trace: ProviderTrace


class LLMHarness:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        prompt_assembler: PromptAssembler | None = None,
        fallback_provider: LLMProvider | None = None,
        sleep: Callable[[float], None] = default_sleep,
        jitter: Callable[[], float] = random,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.prompt_assembler = prompt_assembler or PromptAssembler()
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock
        self.traces: list[ProviderTrace] = []
        self.audit_events: list[InvocationAuditEvent] = []

    def invoke_structured(
        self,
        *,
        skill: SkillSpec,
        task_context: Mapping[str, Any],
        untrusted_inputs: Mapping[str, str],
        output_schema: type[StructuredT],
        tools: Sequence[ToolSpec],
        policy: NodePolicy,
        trace: TraceContext,
    ) -> AgentResult[StructuredT]:
        self._validate_contract(skill=skill, output_schema=output_schema, trace=trace)
        prompts = self.prompt_assembler.assemble(
            skill=skill,
            task_context=task_context,
            untrusted_inputs=untrusted_inputs,
            output_schema=output_schema,
            tools=tools,
        )

        failure_code = "provider_error"
        schema_repairs = 0
        transport_attempt = 0
        current_user_prompt = prompts.user_prompt
        current_prompt_audit = prompts.audit
        while True:
            result, error = self._invoke_once(
                self.provider,
                prompts.system_prompt,
                current_user_prompt,
                output_schema,
                tools,
                policy,
                trace,
                current_prompt_audit,
                fallback_used=False,
            )
            if result is not None:
                parsed = self._validated_value(result, output_schema)
                if parsed is not None:
                    return AgentResult(
                        value=parsed,
                        trace=self.traces[-1],
                        fallback_used=False,
                    )
                failure_code = result.error_code or "schema_error"
                if schema_repairs < policy.max_schema_repairs:
                    schema_repairs += 1
                    detail = result.error_detail or failure_code
                    current_user_prompt = (
                        prompts.user_prompt
                        + "\n\n## 结构化输出修复\n"
                        + f"上一份响应未通过校验（{detail}）。"
                        "请重新生成完整结果，只提交一个符合给定 Schema 的对象；"
                        "必须包含全部必填字段，只使用允许的枚举值，不得添加未声明字段。"
                    )
                    current_prompt_audit = prompts.audit.model_copy(
                        update={
                            "user_prompt_sha256": hashlib.sha256(
                                current_user_prompt.encode("utf-8")
                            ).hexdigest(),
                            "user_prompt_chars": len(current_user_prompt),
                        }
                    )
                    continue
                break
            assert error is not None
            failure_code = error.error_code
            if not error.retryable or transport_attempt >= policy.max_retries:
                break
            self._sleep(self._retry_delay(transport_attempt))
            transport_attempt += 1

        if not policy.allow_rule_fallback or self.fallback_provider is None:
            raise LLMInvocationError(failure_code)

        fallback_result, fallback_error = self._invoke_once(
            self.fallback_provider,
            prompts.system_prompt,
            prompts.user_prompt,
            output_schema,
            tools,
            policy,
            trace,
            prompts.audit,
            fallback_used=True,
        )
        if fallback_result is None:
            assert fallback_error is not None
            raise LLMInvocationError(fallback_error.error_code)
        parsed = self._validated_value(fallback_result, output_schema)
        if parsed is None:
            raise LLMInvocationError(fallback_result.error_code or "schema_error")
        return AgentResult(
            value=parsed,
            trace=self.traces[-1],
            warnings=[f"Primary provider failed with {failure_code}; explicit fallback used."],
            fallback_used=True,
        )

    def _invoke_once(
        self,
        provider: LLMProvider,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[StructuredT],
        tools: Sequence[ToolSpec],
        policy: NodePolicy,
        trace: TraceContext,
        prompt_audit: PromptAudit,
        *,
        fallback_used: bool,
    ) -> tuple[ProviderResult | None, ProviderError | None]:
        started = self._clock()
        try:
            result = provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                tools=tools,
                temperature=policy.temperature,
                max_output_tokens=policy.max_output_tokens,
                trace=trace,
            )
        except ProviderError as exc:
            self._record_trace(
                ProviderTrace(
                    provider=str(getattr(provider, "provider_name", "unknown")),
                    model=str(getattr(provider, "model", "unknown")),
                    skill_id=trace.skill_id,
                    skill_version=trace.skill_version,
                    prompt_version=trace.prompt_version,
                    latency_ms=max(int((self._clock() - started) * 1000), 0),
                    schema_valid=False,
                    fallback_used=fallback_used,
                    error_code=exc.error_code,
                ),
                prompt_audit,
            )
            return None, exc

        if (
            result.skill_id != trace.skill_id
            or result.skill_version != trace.skill_version
            or result.prompt_version != trace.prompt_version
        ):
            error = ProviderContractError("provider returned mismatched trace identity")
            self._record_trace(
                ProviderTrace(
                    provider=result.provider,
                    model=result.model,
                    skill_id=trace.skill_id,
                    skill_version=trace.skill_version,
                    prompt_version=trace.prompt_version,
                    latency_ms=result.latency_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    schema_valid=False,
                    fallback_used=fallback_used,
                    error_code=error.error_code,
                ),
                prompt_audit,
            )
            return None, error
        provider_trace = ProviderTrace.model_validate(
            result.model_dump(exclude={"raw_output", "parsed_output"})
        ).model_copy(update={"fallback_used": fallback_used})
        self._record_trace(provider_trace, prompt_audit)
        return result, None

    def _record_trace(self, trace: ProviderTrace, prompt_audit: PromptAudit) -> None:
        self.traces.append(trace)
        self.audit_events.append(InvocationAuditEvent(prompt=prompt_audit, trace=trace))

    @staticmethod
    def _validated_value(
        result: ProviderResult,
        output_schema: type[StructuredT],
    ) -> StructuredT | None:
        if not result.schema_valid or result.parsed_output is None:
            return None
        try:
            return output_schema.model_validate(result.parsed_output)
        except ValidationError:
            return None

    @staticmethod
    def _validate_contract(
        *,
        skill: SkillSpec,
        output_schema: type[StructuredT],
        trace: TraceContext,
    ) -> None:
        if output_schema is not skill.output_schema:
            raise ValueError("output schema does not match SkillSpec")
        if trace.skill_id != skill.skill_id or trace.skill_version != skill.version:
            raise ValueError("trace skill identity does not match SkillSpec")

    def _retry_delay(self, attempt: int) -> float:
        base = 0.5 if attempt == 0 else 1.5 * (2 ** (attempt - 1))
        return base + (0.1 * base * max(self._jitter(), 0.0))
