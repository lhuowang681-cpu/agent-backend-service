from __future__ import annotations

import json
import socket
from time import perf_counter
from typing import Callable, Sequence
from urllib import error, request

from pydantic import BaseModel

from job_agent.llm.provider import (
    ProviderContractError,
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderTransportError,
    ToolUseResult,
    ToolSpec,
    TraceContext,
)
from job_agent.llm.providers.mock import result_from_raw_output


Transport = Callable[[request.Request, float], bytes]
STRUCTURED_OUTPUT_TOOL_NAME = "submit_structured_output"


def _default_transport(http_request: request.Request, timeout_s: float) -> bytes:
    with request.urlopen(http_request, timeout=timeout_s) as response:
        return response.read()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped
    opening = stripped[:first_newline].casefold()
    if opening not in {"```", "```json"}:
        return stripped
    return stripped[first_newline + 1 : -3].strip()


def _normalize_nested_decision(value: object) -> object:
    """Unwrap providers that serialize the discriminated decision as JSON text.

    Some Anthropic-compatible endpoints accept the forced tool schema but emit
    ``{"decision": "{...}"}`` instead of nesting the decision object.  Only
    this known compatibility shape is normalized; all other payloads are left
    untouched so schema validation remains strict.
    """
    if not isinstance(value, dict):
        return value
    decision = value.get("decision")
    if isinstance(decision, dict):
        nested = dict(decision)
    elif isinstance(decision, str):
        text = _strip_json_fence(decision)
        try:
            nested = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return value
        if not isinstance(nested, dict):
            return value
    else:
        return value
    normalized = dict(value)
    # The same endpoint has also been observed wrapping the finish payload or
    # list-valued finish fields as JSON text. Unwrap only known structured keys.
    if nested.get("kind") == "finish":
        for key in (
            "result",
            "completion_evidence",
            "unresolved_items",
            "search_queries",
            "candidate_job_ids",
            "evidence_refs",
        ):
            candidate = nested.get(key)
            if not isinstance(candidate, str):
                continue
            candidate_text = _strip_json_fence(candidate)
            try:
                decoded = json.loads(candidate_text)
            except (TypeError, json.JSONDecodeError):
                continue
            if key == "result" and isinstance(decoded, dict):
                for list_key in ("search_queries", "candidate_job_ids", "evidence_refs"):
                    list_value = decoded.get(list_key)
                    if not isinstance(list_value, str):
                        continue
                    try:
                        list_decoded = json.loads(_strip_json_fence(list_value))
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(list_decoded, list):
                        decoded[list_key] = list_decoded
                nested[key] = decoded
            elif key != "result" and isinstance(decoded, list):
                nested[key] = decoded
    normalized["decision"] = nested
    return normalized


def _parse_text_payload(text: str) -> object:
    """Parse strict JSON, fenced JSON, or one embedded JSON object."""
    candidate = _strip_json_fence(text)
    try:
        return json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return candidate
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return candidate


class AnthropicCompatibleProvider:
    provider_name = "anthropic_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        anthropic_version: str = "2023-06-01",
        timeout_s: float = 30.0,
        transport: Transport = _default_transport,
        clock=perf_counter,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not api_key:
            raise ValueError("api_key is required")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.anthropic_version = anthropic_version
        self.timeout_s = timeout_s
        self._transport = transport
        self._clock = clock

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
        protocol_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        protocol_tools.append(
            {
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
                "description": "Submit the final response matching the required output schema.",
                "input_schema": output_schema.model_json_schema(),
            }
        )
        payload: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "tools": protocol_tools,
            "tool_choice": {
                "type": "tool",
                "name": STRUCTURED_OUTPUT_TOOL_NAME,
            },
        }
        http_request = request.Request(
            url=f"{self.base_url}/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )
        started = self._clock()
        try:
            response_bytes = self._transport(http_request, self.timeout_s)
        except error.HTTPError as exc:
            if exc.code == 429:
                raise ProviderRateLimitError("provider rate limit") from exc
            if 500 <= exc.code < 600:
                raise ProviderTransportError(f"provider server error: {exc.code}") from exc
            raise ProviderHTTPError(f"provider HTTP error: {exc.code}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderTimeoutError("provider request timed out") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ProviderTimeoutError("provider request timed out") from exc
            raise ProviderTransportError("provider transport failed") from exc
        except OSError as exc:
            raise ProviderTransportError("provider transport failed") from exc
        elapsed_ms = int((self._clock() - started) * 1000)

        try:
            envelope = json.loads(response_bytes.decode("utf-8"))
            content = envelope["content"]
            if not isinstance(content, list):
                raise TypeError("content must be a list")
            output_blocks = [
                block
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == STRUCTURED_OUTPUT_TOOL_NAME
                and isinstance(block.get("input"), dict)
            ]
            text_blocks = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if output_blocks:
                raw_input = output_blocks[0]["input"]
                parsed_output = _normalize_nested_decision(raw_input)
                raw_output = json.dumps(raw_input, ensure_ascii=False)
                parse_output = json.dumps(parsed_output, ensure_ascii=False)
            elif text_blocks:
                raw_output = "".join(text_blocks)
                parsed_text = _parse_text_payload(raw_output)
                parse_output = json.dumps(
                    _normalize_nested_decision(parsed_text),
                    ensure_ascii=False,
                ) if isinstance(parsed_text, (dict, list)) else str(parsed_text)
            else:
                raise ValueError("no structured output or text content")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return ProviderResult(
                provider=self.provider_name,
                model=self.model,
                skill_id=trace.skill_id,
                skill_version=trace.skill_version,
                prompt_version=trace.prompt_version,
                raw_output=None,
                parsed_output=None,
                latency_ms=max(elapsed_ms, 0),
                input_tokens=None,
                output_tokens=None,
                schema_valid=False,
                fallback_used=False,
                error_code="invalid_response",
                error_detail="response:missing_structured_content",
            )
        usage = envelope.get("usage", {})
        response_model = envelope.get("model")
        return result_from_raw_output(
            provider=self.provider_name,
            model=response_model if isinstance(response_model, str) else self.model,
            raw_output=raw_output,
            parse_output=parse_output,
            output_schema=output_schema,
            trace=trace,
            latency_ms=elapsed_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def generate_tool_use(
        self,
        *,
        system_prompt: str,
        messages: Sequence[dict],
        tools: Sequence[ToolSpec],
        temperature: float,
        max_output_tokens: int,
        trace: TraceContext,
    ) -> ToolUseResult:
        """Run one native Anthropic tool-use turn without a structured envelope."""
        if not tools:
            raise ValueError("native tool-use requires at least one tool")
        protocol_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "tools": protocol_tools,
            "tool_choice": {"type": "any"},
        }
        http_request = request.Request(
            url=f"{self.base_url}/v1/messages",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )
        started = self._clock()
        try:
            response_bytes = self._transport(http_request, self.timeout_s)
        except error.HTTPError as exc:
            if exc.code == 429:
                raise ProviderRateLimitError("provider rate limit") from exc
            if 500 <= exc.code < 600:
                raise ProviderTransportError(f"provider server error: {exc.code}") from exc
            raise ProviderHTTPError(f"provider HTTP error: {exc.code}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderTimeoutError("provider request timed out") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise ProviderTimeoutError("provider request timed out") from exc
            raise ProviderTransportError("provider transport failed") from exc
        except OSError as exc:
            raise ProviderTransportError("provider transport failed") from exc
        elapsed_ms = max(int((self._clock() - started) * 1000), 0)

        try:
            envelope = json.loads(response_bytes.decode("utf-8"))
            content = envelope["content"]
            if not isinstance(content, list):
                raise TypeError("content must be a list")
            allowed_names = {tool.name for tool in tools}
            blocks = [
                block
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "tool_use"
                and isinstance(block.get("id"), str)
                and isinstance(block.get("name"), str)
                and block.get("name") in allowed_names
                and isinstance(block.get("input"), dict)
            ]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ProviderContractError("provider returned invalid native tool-use content") from exc
        if len(blocks) != 1:
            raise ProviderContractError("provider must return exactly one allowed tool_use block")
        block = blocks[0]
        usage = envelope.get("usage", {})
        response_model = envelope.get("model")
        return ToolUseResult(
            provider=self.provider_name,
            model=response_model if isinstance(response_model, str) else self.model,
            skill_id=trace.skill_id,
            skill_version=trace.skill_version,
            prompt_version=trace.prompt_version,
            latency_ms=elapsed_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            schema_valid=True,
            fallback_used=False,
            error_code=None,
            tool_use_id=block["id"],
            tool_name=block["name"],
            tool_input=block["input"],
        )
