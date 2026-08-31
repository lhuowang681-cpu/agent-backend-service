from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from job_agent.agent_runtime.contracts import AgentRunResult, AgentRunStatus
from job_agent.agent_runtime.events import (
    AgentEvent,
    ModelDecisionEvent,
    PolicyDecisionEvent,
    RunFinishedEvent,
    RunResumedEvent,
    ToolObservedEvent,
    VerificationEvent,
)
from job_agent.schemas import StrictModel


class EvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    role_family: Literal["agent_algorithm", "llm_application", "backend_data"]
    required_agents: list[str] = Field(min_length=1)
    allowed_tools: dict[str, list[str]] = Field(default_factory=dict)
    sandbox_external_tools: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    require_resume_agents: list[str] = Field(default_factory=list)
    max_model_calls: int = Field(default=150, gt=0)
    max_tool_calls: int = Field(default=400, gt=0)
    evaluator_version: str = "harness-evaluator-v1"
    rubric_version: str = "harness-rubric-v1"


class EvalRunSnapshot(StrictModel):
    run_id: str = Field(min_length=1)
    agent_results: dict[str, AgentRunResult]
    trajectories: dict[str, list[AgentEvent]]
    artifacts: dict[str, Any] = Field(default_factory=dict)


class EvalMetric(StrictModel):
    name: str
    passed: bool
    severity: Literal["hard", "soft"]
    reason_code: str
    value: Any = None


class LLMJudgeResult(StrictModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    summary: str
    judge_version: str


class EvalReport(StrictModel):
    case_id: str
    run_id: str
    evaluator_version: str
    rubric_version: str
    passed: bool
    hard_checks_passed: bool
    metrics: list[EvalMetric] = Field(min_length=1)
    llm_judge: LLMJudgeResult | None = None

    @model_validator(mode="after")
    def hard_failures_cannot_be_overridden(self):
        expected_hard = all(
            metric.passed for metric in self.metrics if metric.severity == "hard"
        )
        if self.hard_checks_passed != expected_hard:
            raise ValueError("hard_checks_passed must equal deterministic hard metrics")
        expected_passed = expected_hard and (
            self.llm_judge is None or self.llm_judge.passed
        )
        if self.passed != expected_passed:
            raise ValueError("LLM judge cannot override deterministic hard-check failure")
        return self

    def comparable_with(self, other: "EvalReport") -> bool:
        return (
            self.evaluator_version == other.evaluator_version
            and self.rubric_version == other.rubric_version
        )


class HarnessEvaluator:
    def evaluate(
        self,
        case: EvalCase,
        snapshot: EvalRunSnapshot,
        *,
        llm_judge: LLMJudgeResult | None = None,
    ) -> EvalReport:
        metrics: list[EvalMetric] = []
        self._evaluate_agents(case, snapshot, metrics)
        self._evaluate_artifacts(case, snapshot, metrics)
        self._evaluate_domain_outcomes(snapshot, metrics)
        hard_passed = all(
            metric.passed for metric in metrics if metric.severity == "hard"
        )
        return EvalReport(
            case_id=case.case_id,
            run_id=snapshot.run_id,
            evaluator_version=case.evaluator_version,
            rubric_version=case.rubric_version,
            passed=hard_passed and (llm_judge is None or llm_judge.passed),
            hard_checks_passed=hard_passed,
            metrics=metrics,
            llm_judge=llm_judge,
        )

    def _evaluate_agents(
        self,
        case: EvalCase,
        snapshot: EvalRunSnapshot,
        metrics: list[EvalMetric],
    ) -> None:
        total_model_calls = 0
        total_tool_calls = 0
        for agent_id in case.required_agents:
            result = snapshot.agent_results.get(agent_id)
            events = snapshot.trajectories.get(agent_id)
            present = result is not None and events is not None
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.input_present",
                    passed=present,
                    severity="hard",
                    reason_code="present" if present else "missing_agent_eval_input",
                )
            )
            if not present:
                continue
            assert result is not None and events is not None
            completed = result.state.status == AgentRunStatus.COMPLETED
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.completed",
                    passed=completed,
                    severity="hard",
                    reason_code="completed" if completed else "agent_not_completed",
                    value=result.state.status.value,
                )
            )
            model_events = [event for event in events if isinstance(event, ModelDecisionEvent)]
            tool_events = [event for event in events if isinstance(event, ToolObservedEvent)]
            total_model_calls += len(model_events)
            total_tool_calls += len(tool_events)
            allowed = set(case.allowed_tools.get(agent_id, []))
            used_tools = [
                event.decision.tool_name
                for event in model_events
                if event.decision.kind == "action"
            ]
            unexpected = sorted(set(used_tools) - allowed)
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.tool_allowlist",
                    passed=not unexpected,
                    severity="hard",
                    reason_code="allowed_tools_only" if not unexpected else "unexpected_tool_used",
                    value=unexpected,
                )
            )
            passed_verification = any(
                isinstance(event, VerificationEvent) and event.verification.passed
                for event in events
            )
            finished_completed = any(
                isinstance(event, RunFinishedEvent)
                and event.status == AgentRunStatus.COMPLETED
                for event in events
            )
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.verified_finish",
                    passed=passed_verification and finished_completed,
                    severity="hard",
                    reason_code=(
                        "verified_finish"
                        if passed_verification and finished_completed
                        else "missing_verified_finish"
                    ),
                )
            )
            action_count = len(used_tools)
            observed_action_ids = {event.observation.action_id for event in tool_events}
            action_ids = {
                event.decision.action_id
                for event in model_events
                if event.decision.kind == "action"
            }
            missing_observations = sorted(action_ids - observed_action_ids)
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.action_observation_pairs",
                    passed=not missing_observations and len(tool_events) >= action_count,
                    severity="hard",
                    reason_code=(
                        "all_actions_observed"
                        if not missing_observations and len(tool_events) >= action_count
                        else "action_without_observation"
                    ),
                    value=missing_observations,
                )
            )
            if agent_id in case.require_resume_agents:
                resumed = any(isinstance(event, RunResumedEvent) for event in events)
                metrics.append(
                    EvalMetric(
                        name=f"{agent_id}.resume_observed",
                        passed=resumed,
                        severity="hard",
                        reason_code="resume_observed" if resumed else "required_resume_missing",
                    )
                )
            external_bypass = [
                event
                for event in events
                if isinstance(event, PolicyDecisionEvent)
                and event.authorization.allowed
                and event.authorization.effect.is_external
                and event.authorization.tool_name not in case.sandbox_external_tools
            ]
            metrics.append(
                EvalMetric(
                    name=f"{agent_id}.real_external_fail_closed",
                    passed=not external_bypass,
                    severity="hard",
                    reason_code=(
                        "no_real_external_bypass"
                        if not external_bypass
                        else "real_external_bypass"
                    ),
                )
            )

        metrics.extend(
            [
                EvalMetric(
                    name="trajectory.model_call_budget",
                    passed=total_model_calls <= case.max_model_calls,
                    severity="hard",
                    reason_code=(
                        "within_model_budget"
                        if total_model_calls <= case.max_model_calls
                        else "model_budget_exceeded"
                    ),
                    value=total_model_calls,
                ),
                EvalMetric(
                    name="trajectory.tool_call_budget",
                    passed=total_tool_calls <= case.max_tool_calls,
                    severity="hard",
                    reason_code=(
                        "within_tool_budget"
                        if total_tool_calls <= case.max_tool_calls
                        else "tool_budget_exceeded"
                    ),
                    value=total_tool_calls,
                ),
            ]
        )

    @staticmethod
    def _evaluate_artifacts(
        case: EvalCase,
        snapshot: EvalRunSnapshot,
        metrics: list[EvalMetric],
    ) -> None:
        missing = sorted(set(case.required_artifacts) - set(snapshot.artifacts))
        metrics.append(
            EvalMetric(
                name="outcome.required_artifacts",
                passed=not missing,
                severity="hard",
                reason_code="artifacts_present" if not missing else "required_artifact_missing",
                value=missing,
            )
        )

    @staticmethod
    def _evaluate_domain_outcomes(
        snapshot: EvalRunSnapshot,
        metrics: list[EvalMetric],
    ) -> None:
        material = snapshot.agent_results.get("application-material")
        if material is not None:
            patch = material.result.get("resume_patch", [])
            audit = material.result.get("claim_audit", [])
            supported_ids = {
                entry.get("claim", {}).get("claim_id")
                for entry in audit
                if entry.get("status") == "supported"
            }
            unsafe = [
                claim.get("claim_id")
                for claim in patch
                if claim.get("claim_id") not in supported_ids
            ]
            metrics.append(
                EvalMetric(
                    name="outcome.resume_claim_provenance",
                    passed=not unsafe,
                    severity="hard",
                    reason_code="all_patch_claims_supported" if not unsafe else "unsupported_patch_claim",
                    value=unsafe,
                )
            )
        interview = snapshot.agent_results.get("interview-coach")
        if interview is not None:
            question_ids = interview.result.get("completed_question_ids", [])
            transcript_complete = (
                interview.result.get("transcript_complete") is True
                and isinstance(question_ids, list)
                and bool(question_ids)
            )
            metrics.append(
                EvalMetric(
                    name="outcome.interview_transcript_complete",
                    passed=transcript_complete,
                    severity="hard",
                    reason_code=(
                        "interview_transcript_complete"
                        if transcript_complete
                        else "interview_transcript_incomplete"
                    ),
                    value=len(question_ids) if isinstance(question_ids, list) else 0,
                )
            )
        ops = snapshot.agent_results.get("application-ops")
        if ops is not None:
            proposal = ops.result.get("proposal")
            metrics.append(
                EvalMetric(
                    name="outcome.tracker_proposal",
                    passed=isinstance(proposal, dict),
                    severity="hard",
                    reason_code="tracker_proposal_present" if isinstance(proposal, dict) else "tracker_proposal_missing",
                )
            )
