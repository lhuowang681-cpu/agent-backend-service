from __future__ import annotations

import pytest

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.failures import BudgetExceededError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_budget_profiles_match_approved_defaults() -> None:
    quick = AgentBudget.for_profile(BudgetProfile.QUICK)
    standard = AgentBudget.for_profile(BudgetProfile.STANDARD)
    deep = AgentBudget.for_profile(BudgetProfile.DEEP)

    assert (quick.max_model_calls, quick.max_tool_calls, quick.max_active_seconds) == (12, 30, 300)
    assert (standard.max_model_calls, standard.max_tool_calls, standard.max_active_seconds) == (30, 80, 1200)
    assert (deep.max_model_calls, deep.max_tool_calls, deep.max_active_seconds) == (80, 200, 3600)


def test_eval_budget_requires_explicit_hard_limits() -> None:
    with pytest.raises(ValueError, match="explicit"):
        AgentBudget.for_profile(BudgetProfile.EVAL)
    budget = AgentBudget.for_profile(
        BudgetProfile.EVAL,
        max_model_calls=4,
        max_tool_calls=9,
        max_active_seconds=60,
    )
    assert budget.profile == BudgetProfile.EVAL


def test_model_call_limit_is_checked_before_next_call() -> None:
    manager = BudgetManager(
        AgentBudget.for_profile(
            BudgetProfile.EVAL,
            max_model_calls=2,
            max_tool_calls=9,
            max_active_seconds=60,
        )
    )
    manager.record_model_call(input_tokens=3, output_tokens=2)
    manager.record_model_call(input_tokens=5, output_tokens=1)
    with pytest.raises(BudgetExceededError) as exc:
        manager.record_model_call()
    assert exc.value.reason_code == "model_call_budget_exceeded"
    assert manager.snapshot().total_tokens == 11


def test_no_progress_and_repeat_query_caps_remain_small() -> None:
    manager = BudgetManager(
        AgentBudget.for_profile(
            BudgetProfile.DEEP,
            max_no_progress=2,
            max_repeat_query=2,
        )
    )
    manager.record_tool_call(fingerprint="search:q", observation_digest="empty")
    manager.record_tool_call(fingerprint="search:q", observation_digest="empty")
    manager.record_tool_call(fingerprint="search:q", observation_digest="empty")
    assert manager.check().reason_code == "no_progress_budget_exceeded"


def test_user_wait_does_not_consume_active_wall_time() -> None:
    clock = FakeClock()
    manager = BudgetManager(
        AgentBudget.for_profile(
            BudgetProfile.EVAL,
            max_model_calls=2,
            max_tool_calls=2,
            max_active_seconds=10,
        ),
        clock=clock,
    )
    clock.advance(3)
    manager.pause_for_user()
    clock.advance(100)
    assert manager.snapshot().active_seconds == 3
    manager.resume_from_user()
    clock.advance(4)
    assert manager.snapshot().active_seconds == 7


def test_format_repairs_have_independent_limit() -> None:
    manager = BudgetManager(
        AgentBudget.for_profile(
            BudgetProfile.EVAL,
            max_model_calls=9,
            max_tool_calls=9,
            max_active_seconds=60,
            max_format_repairs=2,
        )
    )
    manager.record_format_repair()
    manager.record_format_repair()
    with pytest.raises(BudgetExceededError) as exc:
        manager.record_format_repair()
    assert exc.value.reason_code == "format_repair_budget_exceeded"
