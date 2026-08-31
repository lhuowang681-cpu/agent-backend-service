from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
from uuid import uuid4

from pydantic import BaseModel
from redis import Redis
from redis.exceptions import RedisError

from job_agent.llm.provider import (
    LLMProvider,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTransportError,
    ToolSpec,
    TraceContext,
)


@dataclass(frozen=True)
class ProviderPermit:
    token: str
    user_id: str
    provider_key: str


class ProviderAdmissionRejected(RuntimeError):
    def __init__(self, error_code: str, *, retry_after_seconds: float) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retry_after_seconds = retry_after_seconds


class RedisProviderAdmissionController:
    """Redis-backed provider concurrency, per-user cooldown and global circuit."""

    _ACQUIRE = """
    local now_parts = redis.call('TIME')
    local now_ms = (tonumber(now_parts[1]) * 1000) + math.floor(tonumber(now_parts[2]) / 1000)
    local expiry = now_ms + tonumber(ARGV[2])
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms)
    redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms)
    local circuit_ttl = redis.call('PTTL', KEYS[3])
    if circuit_ttl > 0 then return {0, 1, circuit_ttl} end
    local cooldown_ttl = redis.call('PTTL', KEYS[4])
    if cooldown_ttl > 0 then return {0, 2, cooldown_ttl} end
    if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then return {0, 3, 250} end
    if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[4]) then return {0, 4, 250} end
    redis.call('ZADD', KEYS[1], expiry, ARGV[1])
    redis.call('ZADD', KEYS[2], expiry, ARGV[1])
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 2)
    redis.call('PEXPIRE', KEYS[2], tonumber(ARGV[2]) * 2)
    return {1, 0, tonumber(ARGV[2])}
    """

    _RELEASE = """
    redis.call('ZREM', KEYS[1], ARGV[1])
    redis.call('ZREM', KEYS[2], ARGV[1])
    return 1
    """

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "job-agent:provider",
        global_limit: int = 4,
        per_user_limit: int = 4,
        permit_ttl_seconds: float = 60.0,
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 10.0,
    ) -> None:
        if global_limit < 1 or per_user_limit < 1:
            raise ValueError("provider_limits_must_be_positive")
        self._redis = Redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.global_limit = global_limit
        self.per_user_limit = per_user_limit
        self.permit_ttl_ms = max(1, int(permit_ttl_seconds * 1000))
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds

    def close(self) -> None:
        self._redis.close()

    def acquire(self, *, user_id: str, provider_key: str) -> ProviderPermit:
        token = str(uuid4())
        keys = self._keys(user_id=user_id, provider_key=provider_key)
        try:
            result = self._redis.eval(
                self._ACQUIRE,
                4,
                keys["global"],
                keys["user"],
                keys["circuit"],
                keys["cooldown"],
                token,
                self.permit_ttl_ms,
                self.global_limit,
                self.per_user_limit,
            )
        except RedisError as exc:
            raise ProviderAdmissionRejected(
                "provider_limiter_unavailable",
                retry_after_seconds=1.0,
            ) from exc
        allowed, reason, retry_ms = (int(result[0]), int(result[1]), int(result[2]))
        if allowed:
            return ProviderPermit(token=token, user_id=user_id, provider_key=provider_key)
        codes = {
            1: "provider_circuit_open",
            2: "provider_user_cooldown",
            3: "provider_global_backpressure",
            4: "provider_user_backpressure",
        }
        raise ProviderAdmissionRejected(
            codes[reason],
            retry_after_seconds=max(retry_ms / 1000.0, 0.05),
        )

    def record_success(self, permit: ProviderPermit) -> None:
        self._release(permit)
        self._redis.delete(self._keys(permit.user_id, permit.provider_key)["failures"])

    def record_rate_limit(
        self,
        permit: ProviderPermit,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        self._release(permit)
        keys = self._keys(permit.user_id, permit.provider_key)
        cooldown = max(retry_after_seconds or 1.0, 0.05)
        self._redis.set(keys["cooldown"], "1", px=max(1, int(cooldown * 1000)))

    def record_failure(self, permit: ProviderPermit) -> None:
        self._release(permit)
        keys = self._keys(permit.user_id, permit.provider_key)
        failures = int(self._redis.incr(keys["failures"]))
        self._redis.expire(keys["failures"], 60)
        if failures >= self.circuit_failure_threshold:
            self._redis.set(
                keys["circuit"],
                "1",
                px=max(1, int(self.circuit_cooldown_seconds * 1000)),
            )

    def release(self, permit: ProviderPermit) -> None:
        self._release(permit)

    def _release(self, permit: ProviderPermit) -> None:
        keys = self._keys(permit.user_id, permit.provider_key)
        self._redis.eval(
            self._RELEASE,
            2,
            keys["global"],
            keys["user"],
            permit.token,
        )

    def _keys(self, user_id: str, provider_key: str) -> dict[str, str]:
        base = f"{self.prefix}:{provider_key}"
        return {
            "global": f"{base}:permits",
            "user": f"{base}:user:{user_id}:permits",
            "circuit": f"{base}:circuit",
            "cooldown": f"{base}:user:{user_id}:cooldown",
            "failures": f"{base}:failures",
        }


class AdmissionControlledProvider:
    """Acquire one permit for exactly one outbound Provider call."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        controller: RedisProviderAdmissionController,
        user_id: str,
        provider_key: str,
    ) -> None:
        self.provider = provider
        self.controller = controller
        self.user_id = user_id
        self.provider_key = provider_key
        self.provider_name = str(getattr(provider, "provider_name", provider_key))
        self.model = str(getattr(provider, "model", "unknown"))

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
        permit = self.controller.acquire(
            user_id=self.user_id,
            provider_key=self.provider_key,
        )
        permit_settled = False
        try:
            result = self.provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                trace=trace,
            )
        except ProviderRateLimitError:
            self._best_effort(lambda: self.controller.record_rate_limit(permit))
            permit_settled = True
            raise
        except ProviderTransportError:
            self._best_effort(lambda: self.controller.record_failure(permit))
            permit_settled = True
            raise
        else:
            if result.error_code == "invalid_response":
                self._best_effort(lambda: self.controller.record_failure(permit))
            else:
                self._best_effort(lambda: self.controller.record_success(permit))
            permit_settled = True
            return result
        finally:
            if not permit_settled:
                self._best_effort(lambda: self.controller.release(permit))

    @staticmethod
    def _best_effort(operation: Callable[[], None]) -> None:
        try:
            operation()
        except RedisError:
            return
