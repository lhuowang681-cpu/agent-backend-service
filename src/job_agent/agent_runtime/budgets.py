from __future__ import annotations

from enum import Enum
from time import monotonic
from typing import Callable

from pydantic import ConfigDict, Field, model_validator

from job_agent.agent_runtime.failures import BudgetExceededError
from job_agent.schemas import StrictModel


class BudgetProfile(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EVAL = "eval"


class AgentBudget(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    profile: BudgetProfile
    max_model_calls: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_active_seconds: float = Field(gt=0)
    max_no_progress: int = Field(default=3, gt=0)
    max_format_repairs: int = Field(default=2, ge=0)
    max_repeat_query: int = Field(default=2, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, gt=0)

    @classmethod
    def for_profile(cls, profile: BudgetProfile | str, **overrides) -> "AgentBudget":
        profile = BudgetProfile(profile)
        defaults = {
            BudgetProfile.QUICK: dict(
                max_model_calls=12,
                max_tool_calls=30,
                max_active_seconds=5 * 60,
            ),
            BudgetProfile.STANDARD: dict(
                max_model_calls=30,
                max_tool_calls=80,
                max_active_seconds=20 * 60,
            ),
            BudgetProfile.DEEP: dict(
                max_model_calls=80,
                max_tool_calls=200,
                max_active_seconds=60 * 60,
                max_no_progress=5,
                max_format_repairs=3,
            ),
        }
        if profile == BudgetProfile.EVAL and not {
            "max_model_calls",
            "max_tool_calls",
            "max_active_seconds",
        }.issubset(overrides):
            raise ValueError("eval budget requires explicit model/tool/time limits")
        values = {**defaults.get(profile, {}), **overrides, "profile": profile}
        return cls(**values)


class BudgetUsage(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    format_repairs: int = Field(default=0, ge=0)
    no_progress_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost: float = Field(default=0.0, ge=0.0)
    active_seconds: float = Field(default=0.0, ge=0.0)
    repeated_query_count: int = Field(default=0, ge=0)
    last_tool_fingerprint: str | None = None
    last_observation_digest: str | None = None


class BudgetCheck(StrictModel):
    allowed: bool
    reason_code: str | None = None
    usage: BudgetUsage

    @model_validator(mode="after")
    def validate_reason(self):
        if self.allowed and self.reason_code is not None:
            raise ValueError("allowed checks cannot have a reason")
        if not self.allowed and self.reason_code is None:
            raise ValueError("denied checks require a reason")
        return self


class BudgetManager:
    def __init__(
        self,
        budget: AgentBudget,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.budget = budget
        self.usage = BudgetUsage()
        self._clock = clock
        self._active_started = clock()
        self._accumulated_active = 0.0
        self._paused = False

    def pause_for_user(self) -> None:
        if self._paused:
            return
        self._accumulated_active += max(self._clock() - self._active_started, 0.0)
        self._paused = True

    def resume_from_user(self) -> None:
        if not self._paused:
            return
        self._active_started = self._clock()
        self._paused = False

    def restore(self, usage: BudgetUsage, *, paused: bool) -> None:
        """Restore persisted counters without counting process downtime as active time."""
        self.usage = usage.model_copy(deep=True)
        self._accumulated_active = usage.active_seconds
        self._active_started = self._clock()
        self._paused = paused

    def snapshot(self) -> BudgetUsage:
        active_seconds = self._accumulated_active
        if not self._paused:
            active_seconds += max(self._clock() - self._active_started, 0.0)
        self.usage = self.usage.model_copy(update={"active_seconds": active_seconds})
        return self.usage.model_copy(deep=True)

    def check(self) -> BudgetCheck:
        usage = self.snapshot()
        reason = self._reason_exceeded(usage)
        return BudgetCheck(allowed=reason is None, reason_code=reason, usage=usage)

    def ensure_available(self) -> None:
        check = self.check()
        if not check.allowed:
            raise BudgetExceededError(check.reason_code or "budget_exceeded")

    def record_model_call(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost: float | None = None,
    ) -> BudgetUsage:
        self.ensure_available()
        tokens = max(input_tokens or 0, 0) + max(output_tokens or 0, 0)
        self.usage = self.usage.model_copy(
            update={
                "model_calls": self.usage.model_calls + 1,
                "total_tokens": self.usage.total_tokens + tokens,
                "total_cost": self.usage.total_cost + max(cost or 0.0, 0.0),
            }
        )
        return self.snapshot()

    def record_tool_call(
        self,
        *,
        fingerprint: str,
        observation_digest: str,
        query_repeated: bool = False,
    ) -> BudgetUsage:
        usage = self.snapshot()
        if usage.tool_calls >= self.budget.max_tool_calls:
            raise BudgetExceededError("tool_call_budget_exceeded")
        if usage.active_seconds >= self.budget.max_active_seconds:
            raise BudgetExceededError("wall_time_budget_exceeded")
        made_progress = not (
            self.usage.last_tool_fingerprint == fingerprint
            and self.usage.last_observation_digest == observation_digest
        )
        self.usage = self.usage.model_copy(
            update={
                "tool_calls": self.usage.tool_calls + 1,
                "no_progress_count": 0 if made_progress else self.usage.no_progress_count + 1,
                "repeated_query_count": self.usage.repeated_query_count + 1 if query_repeated else 0,
                "last_tool_fingerprint": fingerprint,
                "last_observation_digest": observation_digest,
            }
        )
        return self.snapshot()

    def record_format_repair(self) -> BudgetUsage:
        self.ensure_available()
        self.usage = self.usage.model_copy(
            update={"format_repairs": self.usage.format_repairs + 1}
        )
        return self.snapshot()

    def _reason_exceeded(self, usage: BudgetUsage) -> str | None:
        # Limits are checked before the next operation. Equality means the
        # configured number of operations has already been consumed.
        if usage.model_calls >= self.budget.max_model_calls:
            return "model_call_budget_exceeded"
        if usage.tool_calls >= self.budget.max_tool_calls:
            return "tool_call_budget_exceeded"
        if usage.active_seconds >= self.budget.max_active_seconds:
            return "wall_time_budget_exceeded"
        if usage.no_progress_count >= self.budget.max_no_progress:
            return "no_progress_budget_exceeded"
        if usage.format_repairs >= self.budget.max_format_repairs and self.budget.max_format_repairs >= 0:
            if self.budget.max_format_repairs == 0 or usage.format_repairs > 0:
                return "format_repair_budget_exceeded"
        if usage.repeated_query_count >= self.budget.max_repeat_query:
            return "repeat_query_budget_exceeded"
        if self.budget.max_total_tokens is not None and usage.total_tokens >= self.budget.max_total_tokens:
            return "token_budget_exceeded"
        if self.budget.max_cost is not None and usage.total_cost >= self.budget.max_cost:
            return "cost_budget_exceeded"
        return None
