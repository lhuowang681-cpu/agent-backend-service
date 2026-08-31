from __future__ import annotations

import json
import socket
from time import perf_counter
from typing import Callable, Literal, Sequence
from urllib import error, request

from pydantic import BaseModel

from job_agent.llm.provider import (
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
    ProviderTransportError,
    ToolSpec,
    TraceContext,
)
from job_agent.llm.providers.mock import result_from_raw_output


Transport = Callable[[request.Request, float], bytes]


def _default_transport(http_request: request.Request, timeout_s: float) -> bytes:
    with request.urlopen(http_request, timeout=timeout_s) as response:
        return response.read()


class OpenAICompatibleProvider:
    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        response_format_mode: Literal["json_schema", "json_object"] = "json_schema",
        transport: Transport = _default_transport,
        clock=perf_counter,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.response_format_mode = response_format_mode
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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if self.response_format_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema.__name__,
                    "strict": True,
                    "schema": output_schema.model_json_schema(),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
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
            raw_output = envelope["choices"][0]["message"]["content"]
            if not isinstance(raw_output, str):
                raise TypeError("message content must be text")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
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
            )
        usage = envelope.get("usage", {})
        return result_from_raw_output(
            provider=self.provider_name,
            model=self.model,
            raw_output=raw_output,
            output_schema=output_schema,
            trace=trace,
            latency_ms=elapsed_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
