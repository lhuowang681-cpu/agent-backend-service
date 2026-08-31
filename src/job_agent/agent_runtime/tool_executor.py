from __future__ import annotations

from time import monotonic, perf_counter
from typing import Callable

from pydantic import ValidationError

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AuthorizationResult,
    ToolContext,
    ToolObservation,
    ToolObservationStatus,
)
from job_agent.agent_runtime.failures import ToolRetryableError
from job_agent.agent_runtime.policy import compute_action_digest
from job_agent.agent_runtime.tool_registry import ToolRegistry


class ToolExecutor:
    """Execute trusted in-process tools with a cooperative deadline.

    Handlers must propagate ``ToolContext.remaining_seconds()`` to blocking
    clients and call ``ensure_active()`` around long operations. The executor
    rejects late results, but it does not claim to preempt an arbitrary blocked
    Python handler.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        clock: Callable[[], float] = perf_counter,
        deadline_clock: Callable[[], float] = monotonic,
    ) -> None:
        self.registry = registry
        self._clock = clock
        self._deadline_clock = deadline_clock

    def execute(
        self,
        action: AgentAction,
        authorization: AuthorizationResult,
        context: ToolContext,
    ) -> ToolObservation:
        started = self._clock()
        if not authorization.allowed:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "authorization_required",
            )
        if authorization.tool_name != action.tool_name:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "authorization_tool_mismatch",
            )
        if authorization.action_digest != compute_action_digest(action):
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "authorization_digest_mismatch",
            )

        try:
            tool = self.registry.get(action.tool_name)
        except Exception:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "unknown_tool",
            )

        try:
            parsed_input = tool.input_model.model_validate(action.tool_arguments)
        except ValidationError:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "tool_input_error",
            )

        execution_context = context.model_copy(
            update={"deadline_monotonic": self._deadline_clock() + tool.spec.timeout_s}
        )
        try:
            execution_context.ensure_active(clock=self._deadline_clock)
            raw_output = tool.handler(parsed_input, execution_context)
            execution_context.ensure_active(clock=self._deadline_clock)
        except ToolRetryableError as exc:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.RETRYABLE_ERROR,
                exc.error_code,
            )
        except Exception:
            return self._error_observation(
                action,
                started,
                ToolObservationStatus.FATAL_ERROR,
                "tool_execution_error",
            )

        duration_ms = self._duration_ms(started)
        if duration_ms > int(tool.spec.timeout_s * 1000):
            return ToolObservation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                status=ToolObservationStatus.RETRYABLE_ERROR,
                duration_ms=duration_ms,
                error_code="tool_timeout",
            )
        try:
            output = tool.output_model.model_validate(raw_output)
        except ValidationError:
            return ToolObservation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                status=ToolObservationStatus.FATAL_ERROR,
                duration_ms=duration_ms,
                error_code="tool_output_error",
            )
        data = output.model_dump(mode="json")
        is_empty = not any(value not in (None, "", [], {}) for value in data.values())
        return ToolObservation(
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=ToolObservationStatus.EMPTY if is_empty else ToolObservationStatus.OK,
            data=data,
            duration_ms=duration_ms,
        )

    def _error_observation(
        self,
        action: AgentAction,
        started: float,
        status: ToolObservationStatus,
        error_code: str,
    ) -> ToolObservation:
        return ToolObservation(
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=status,
            duration_ms=self._duration_ms(started),
            error_code=error_code,
        )

    def _duration_ms(self, started: float) -> int:
        return max(int((self._clock() - started) * 1000), 0)
