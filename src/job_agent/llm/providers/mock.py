from __future__ import annotations

import json
from collections.abc import Iterable
from time import perf_counter
from typing import Any, Sequence

from pydantic import BaseModel, ValidationError

from job_agent.llm.provider import ProviderResult, ProviderTransportError, ToolSpec, TraceContext


_SAFE_SCHEMA_PATH_PARTS = {
    "decision",
    "kind",
    "action_id",
    "tool_name",
    "tool_arguments",
    "expected_observation",
    "progress_claim",
    "request",
    "request_id",
    "prompt",
    "response_schema",
    "context_summary",
    "allow_cancel",
    "result",
    "completion_evidence",
    "unresolved_items",
    "confidence",
    "value",
}


def _validation_error_detail(error: ValidationError) -> str:
    """Return safe schema paths/types; never include rejected names, values, or raw output."""
    items = []
    for item in error.errors():
        parts = []
        for part in item.get("loc", ()):
            text = str(part)
            parts.append(text if text.isdigit() or text in _SAFE_SCHEMA_PATH_PARTS else "<field>")
        location = ".".join(parts) or "root"
        # Include only the runtime type, never the rejected value itself. This is
        # useful for provider compatibility debugging (e.g. a provider may wrap
        # a structured decision in a JSON string) while keeping diagnostics safe.
        input_value = item.get("input")
        input_type = type(input_value).__name__ if "input" in item else None
        suffix = f":{input_type}" if input_type else ""
        items.append(f"{location}:{item.get('type', 'validation_error')}{suffix}")
    return ";".join(items)[:500] or "schema_validation_failed"


def result_from_raw_output(
    *,
    provider: str,
    model: str,
    raw_output: str,
    output_schema: type[BaseModel],
    trace: TraceContext,
    latency_ms: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    parse_output: str | None = None,
) -> ProviderResult:
    common = {
        "provider": provider,
        "model": model,
        "skill_id": trace.skill_id,
        "skill_version": trace.skill_version,
        "prompt_version": trace.prompt_version,
        "raw_output": raw_output,
        "latency_ms": max(latency_ms, 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "fallback_used": False,
    }
    try:
        payload = json.loads(raw_output if parse_output is None else parse_output)
    except (json.JSONDecodeError, TypeError):
        return ProviderResult(
            **common,
            parsed_output=None,
            schema_valid=False,
            error_code="invalid_json",
            error_detail="response:not_json",
        )
    try:
        validated = output_schema.model_validate(payload)
    except ValidationError as exc:
        return ProviderResult(
            **common,
            parsed_output=None,
            schema_valid=False,
            error_code="schema_error",
            error_detail=_validation_error_detail(exc),
        )
    return ProviderResult(
        **common,
        parsed_output=validated.model_dump(mode="json"),
        schema_valid=True,
        error_code=None,
    )


class MockLLMProvider:
    provider_name = "mock"

    def __init__(
        self,
        responses: Iterable[dict[str, Any] | str | Exception],
        *,
        provider_name: str = "mock",
        model: str = "fixture-model",
        clock=perf_counter,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self._responses = list(responses)
        self._clock = clock
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[BaseModel],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_output_tokens: int,
        trace: TraceContext,
    ) -> ProviderResult:
        started = self._clock()
        self.calls.append(
            {
                "output_schema": output_schema,
                "tools": tuple(tools),
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "trace": trace,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if not self._responses:
            raise ProviderTransportError("mock response queue is empty")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        raw_output = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        elapsed_ms = int((self._clock() - started) * 1000)
        return result_from_raw_output(
            provider=self.provider_name,
            model=self.model,
            raw_output=raw_output,
            output_schema=output_schema,
            trace=trace,
            latency_ms=elapsed_ms,
        )
