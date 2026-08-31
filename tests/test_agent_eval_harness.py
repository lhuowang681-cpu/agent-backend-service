from __future__ import annotations

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentRunResult,
    AgentRunState,
    AgentRunStatus,
    AuthorizationResult,
    ToolEffect,
    ToolObservation,
    ToolObservationStatus,
    VerificationResult,
)
from job_agent.agent_runtime.evaluation import (
    EvalCase,
    EvalRunSnapshot,
    HarnessEvaluator,
    LLMJudgeResult,
)
from job_agent.agent_runtime.events import (
    ModelDecisionEvent,
    PolicyDecisionEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolObservedEvent,
    VerificationEvent,
    event_identity,
)


def _identity(step):
    return event_identity(
        session_id="session-1",
        run_id="run-1",
        agent_id="opportunity-research",
        step_index=step,
    )


def _good_snapshot(tool_name="jobs.search") -> EvalRunSnapshot:
    action = AgentAction(
        action_id="search-1",
        tool_name=tool_name,
        tool_arguments={"query": "agent intern"},
        expected_observation="jobs",
        progress_claim="search approved jobs",
    )
    finish = AgentFinish(
        result={"candidate_job_ids": ["job-1"]},
        completion_evidence=["step:1:jobs.search"],
        confidence=0.9,
    )
    events = [
        RunStartedEvent(**_identity(0), goal={"role": "agent"}, budget_profile="eval"),
        ModelDecisionEvent(
            **_identity(1),
            decision=action,
            provider="fixture",
            model="fixture",
        ),
        PolicyDecisionEvent(
            **_identity(1),
            authorization=AuthorizationResult(
                action_id="search-1",
                action_digest="digest",
                tool_name=tool_name,
                effect=ToolEffect.READ_ONLY,
                allowed=True,
                requires_approval=False,
                reason_code="allowed",
            ),
        ),
        ToolObservedEvent(
            **_identity(1),
            observation=ToolObservation(
                action_id="search-1",
                tool_name=tool_name,
                status=ToolObservationStatus.OK,
                data={"jobs": ["job-1"]},
                duration_ms=1,
            ),
        ),
        ModelDecisionEvent(
            **_identity(2),
            decision=finish,
            provider="fixture",
            model="fixture",
        ),
        VerificationEvent(
            **_identity(2),
            verification=VerificationResult(
                passed=True,
                reason_code="complete",
                feedback="complete",
            ),
        ),
        RunFinishedEvent(
            **_identity(2),
            status=AgentRunStatus.COMPLETED,
            result=finish.result,
        ),
    ]
    return EvalRunSnapshot(
        run_id="run-1",
        agent_results={
            "opportunity-research": AgentRunResult(
                state=AgentRunState(
                    session_id="session-1",
                    run_id="run-1",
                    agent_id="opportunity-research",
                    status=AgentRunStatus.COMPLETED,
                    goal={"role": "agent"},
                    step_index=2,
                    budget_profile="eval",
                ),
                result=finish.result,
            )
        },
        trajectories={"opportunity-research": events},
        artifacts={"opportunities": {"job_ids": ["job-1"]}},
    )


def _case(**updates) -> EvalCase:
    values = dict(
        case_id="agent-algorithm-1",
        role_family="agent_algorithm",
        required_agents=["opportunity-research"],
        allowed_tools={"opportunity-research": ["jobs.search"]},
        required_artifacts=["opportunities"],
        max_model_calls=4,
        max_tool_calls=2,
    )
    values.update(updates)
    return EvalCase(**values)


def test_evaluator_accepts_grounded_golden_trajectory() -> None:
    report = HarnessEvaluator().evaluate(_case(), _good_snapshot())
    assert report.passed is True
    assert report.hard_checks_passed is True
    assert all(metric.passed for metric in report.metrics if metric.severity == "hard")


def test_llm_judge_cannot_override_hard_failure() -> None:
    snapshot = _good_snapshot(tool_name="jobs.unapproved")
    snapshot = snapshot.model_copy(update={"artifacts": {}})
    judge = LLMJudgeResult(
        passed=True,
        score=1.0,
        summary="Looks good to the optional judge.",
        judge_version="fixture-judge-v1",
    )
    report = HarnessEvaluator().evaluate(_case(), snapshot, llm_judge=judge)
    assert report.passed is False
    assert report.hard_checks_passed is False
    assert report.llm_judge.passed is True
    assert {metric.reason_code for metric in report.metrics if not metric.passed} >= {
        "unexpected_tool_used",
        "required_artifact_missing",
    }


def test_evaluator_reports_missing_agent_input_instead_of_crashing() -> None:
    snapshot = EvalRunSnapshot(
        run_id="missing-1",
        agent_results={},
        trajectories={},
    )
    report = HarnessEvaluator().evaluate(_case(required_artifacts=[]), snapshot)
    assert report.passed is False
    assert any(metric.reason_code == "missing_agent_eval_input" for metric in report.metrics)


def test_reports_with_different_versions_are_not_comparable() -> None:
    first = HarnessEvaluator().evaluate(_case(), _good_snapshot())
    second = HarnessEvaluator().evaluate(
        _case(evaluator_version="harness-evaluator-v2"),
        _good_snapshot(),
    )
    assert first.comparable_with(second) is False


def test_evaluator_allows_only_declared_external_sandbox_tools() -> None:
    snapshot = _good_snapshot()
    events = list(snapshot.trajectories["opportunity-research"])
    policy_index = next(
        index for index, event in enumerate(events) if isinstance(event, PolicyDecisionEvent)
    )
    policy_event = events[policy_index]
    events[policy_index] = policy_event.model_copy(
        update={
            "authorization": policy_event.authorization.model_copy(
                update={"effect": ToolEffect.EXTERNAL_REVERSIBLE_WRITE}
            )
        }
    )
    snapshot = snapshot.model_copy(
        update={"trajectories": {"opportunity-research": events}}
    )
    denied = HarnessEvaluator().evaluate(_case(), snapshot)
    assert any(metric.reason_code == "real_external_bypass" for metric in denied.metrics)

    allowed = HarnessEvaluator().evaluate(
        _case(sandbox_external_tools=["jobs.search"]),
        snapshot,
    )
    assert allowed.passed is True
