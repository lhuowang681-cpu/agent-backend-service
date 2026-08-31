from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from pydantic import Field

from job_agent.agent_runtime.contracts import AgentRunResult
from job_agent.agent_runtime.eval_cases import HarnessEvalCase
from job_agent.agent_runtime.evaluation import (
    EvalCase,
    EvalReport,
    EvalRunSnapshot,
    HarnessEvaluator,
)
from job_agent.agent_runtime.events import AgentEvent
from job_agent.schemas import StrictModel


E2E_STAGE_ORDER = (
    "opportunity-research",
    "application-material",
    "interview-coach",
    "application-ops",
)


class E2EStageOutput(StrictModel):
    agent_id: str
    result: AgentRunResult
    events: list[AgentEvent]
    artifacts: dict[str, Any] = Field(default_factory=dict)


class E2ERunRecord(StrictModel):
    case_id: str
    run_id: str
    stage_order: list[str]
    started_at: str
    finished_at: str
    snapshot: EvalRunSnapshot
    report: EvalReport


StageExecutor = Callable[[HarnessEvalCase, dict[str, Any]], E2EStageOutput]


class HarnessE2ERunner:
    """Deterministic coordinator for fixture, replay, mock-provider, or live stage adapters."""

    def __init__(
        self,
        *,
        stages: dict[str, StageExecutor],
        allowed_tools: dict[str, list[str]],
        sandbox_external_tools: list[str] | None = None,
        evaluator: HarnessEvaluator | None = None,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        missing = sorted(set(E2E_STAGE_ORDER) - set(stages))
        extra = sorted(set(stages) - set(E2E_STAGE_ORDER))
        if missing or extra:
            raise ValueError(f"invalid E2E stages: missing={missing}, extra={extra}")
        self.stages = stages
        self.allowed_tools = allowed_tools
        self.sandbox_external_tools = sandbox_external_tools or []
        self.evaluator = evaluator or HarnessEvaluator()
        self._clock = clock

    def run(self, case: HarnessEvalCase, *, run_id: str) -> E2ERunRecord:
        started = self._clock()
        artifacts: dict[str, Any] = {}
        results: dict[str, AgentRunResult] = {}
        trajectories: dict[str, list[AgentEvent]] = {}
        completed_order: list[str] = []
        for agent_id in E2E_STAGE_ORDER:
            output = self.stages[agent_id](case, dict(artifacts))
            if output.agent_id != agent_id:
                raise ValueError(f"stage identity mismatch: expected {agent_id}, got {output.agent_id}")
            collisions = sorted(set(artifacts) & set(output.artifacts))
            if collisions:
                raise ValueError(f"artifact overwrite is forbidden: {collisions}")
            artifacts.update(output.artifacts)
            results[agent_id] = output.result
            trajectories[agent_id] = output.events
            completed_order.append(agent_id)
        snapshot = EvalRunSnapshot(
            run_id=run_id,
            agent_results=results,
            trajectories=trajectories,
            artifacts=artifacts,
        )
        eval_case = EvalCase(
            case_id=case.case_id,
            role_family=case.role_family,
            required_agents=list(E2E_STAGE_ORDER),
            allowed_tools=self.allowed_tools,
            sandbox_external_tools=self.sandbox_external_tools,
            required_artifacts=[
                "opportunities",
                "material_patch",
                "interview_debrief",
                "tracker_result",
            ],
            require_resume_agents=["interview-coach"],
        )
        report = self.evaluator.evaluate(eval_case, snapshot)
        finished = self._clock()
        return E2ERunRecord(
            case_id=case.case_id,
            run_id=run_id,
            stage_order=completed_order,
            started_at=started.astimezone(UTC).isoformat(),
            finished_at=finished.astimezone(UTC).isoformat(),
            snapshot=snapshot,
            report=report,
        )
