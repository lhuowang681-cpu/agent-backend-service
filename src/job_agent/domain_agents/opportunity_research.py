from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.contracts import (
    AgentFinish,
    AgentRunResult,
    AgentStepRecord,
    ToolContext,
    ToolEffect,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, AgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.schemas import RawJob, StrictModel
from job_agent.tools.job_search import load_raw_jobs


SEARCH_TOOL = "jobs.search"
READ_TOOL = "jobs.read"
DEDUPE_TOOL = "jobs.deduplicate"
OPPORTUNITY_TOOLS = (SEARCH_TOOL, READ_TOOL, DEDUPE_TOOL)


class OpportunityResearchGoal(StrictModel):
    target_role: str = Field(min_length=1)
    cities: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    min_candidates: int = Field(default=2, gt=0)


class JobSearchInput(StrictModel):
    keywords: list[str] = Field(min_length=1)
    cities: list[str] = Field(default_factory=list)


class JobSearchOutput(StrictModel):
    jobs: list[RawJob] = Field(default_factory=list)


class JobReadInput(StrictModel):
    job_id: str = Field(min_length=1)


class JobReadOutput(StrictModel):
    job: RawJob | None = None


class JobDedupeInput(StrictModel):
    job_ids: list[str] = Field(min_length=1)


class JobDedupeOutput(StrictModel):
    job_ids: list[str] = Field(default_factory=list)


class OpportunityResearchResult(StrictModel):
    search_queries: list[str] = Field(min_length=1)
    candidate_job_ids: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    unresolved_gaps: list[str] = Field(default_factory=list)


def _safe_result_schema_detail(error: ValidationError) -> str:
    """Summarize result-shape failures without exposing model output values."""
    allowed = {
        "search_queries",
        "candidate_job_ids",
        "evidence_refs",
        "unresolved_gaps",
    }
    parts: list[str] = []
    for item in error.errors():
        location = ".".join(
            str(part) if str(part).isdigit() or str(part) in allowed else "<field>"
            for part in item.get("loc", ())
        ) or "root"
        input_type = type(item.get("input")).__name__ if "input" in item else "unknown"
        parts.append(f"{location}:{item.get('type', 'validation_error')}:{input_type}")
    return ";".join(parts)[:300] or "schema_validation_failed"


def build_opportunity_tool_registry(source_path: Path | str) -> ToolRegistry:
    path = Path(source_path).resolve()
    catalog = {job.job_id: job for job in load_raw_jobs(path)}
    registry = ToolRegistry()

    def search_jobs(value: JobSearchInput, context: ToolContext) -> JobSearchOutput:
        keyword_keys = [keyword.casefold().strip() for keyword in value.keywords if keyword.strip()]
        city_keys = {city.casefold().strip() for city in value.cities if city.strip()}
        jobs = []
        for job in catalog.values():
            haystack = f"{job.title} {job.desc} {job.company}".casefold()
            keyword_match = any(keyword in haystack for keyword in keyword_keys)
            city_match = not city_keys or job.location.casefold().strip() in city_keys
            if keyword_match and city_match:
                jobs.append(job)
        return JobSearchOutput(jobs=jobs)

    def read_job(value: JobReadInput, context: ToolContext) -> JobReadOutput:
        return JobReadOutput(job=catalog.get(value.job_id))

    def deduplicate(value: JobDedupeInput, context: ToolContext) -> JobDedupeOutput:
        seen: set[str] = set()
        result = []
        for job_id in value.job_ids:
            if job_id in catalog and job_id not in seen:
                seen.add(job_id)
                result.append(job_id)
        return JobDedupeOutput(job_ids=result)

    registry.register(
        name=SEARCH_TOOL,
        description="Search the approved local job catalog by keywords and cities.",
        input_model=JobSearchInput,
        output_model=JobSearchOutput,
        handler=search_jobs,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=READ_TOOL,
        description="Read one job from the approved local catalog by job_id.",
        input_model=JobReadInput,
        output_model=JobReadOutput,
        handler=read_job,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=DEDUPE_TOOL,
        description="Deduplicate approved job ids while preserving order.",
        input_model=JobDedupeInput,
        output_model=JobDedupeOutput,
        handler=deduplicate,
        effect=ToolEffect.READ_ONLY,
    )
    return registry


class OpportunityResearchVerifier:
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: Sequence[AgentStepRecord],
    ) -> VerificationResult:
        try:
            typed_goal = OpportunityResearchGoal.model_validate(goal)
            result = OpportunityResearchResult.model_validate(finish.result)
        except ValidationError as exc:
            return VerificationResult(
                passed=False,
                reason_code=f"opportunity_result_schema_error:{_safe_result_schema_detail(exc)}",
                feedback=(
                    "Return a valid opportunity research result with queries, candidates, and evidence refs. "
                    f"Shape diagnostic: {_safe_result_schema_detail(exc)}"
                ),
            )
        except Exception:
            return VerificationResult(
                passed=False,
                reason_code="opportunity_result_schema_error",
                feedback="Return a valid opportunity research result with queries, candidates, and evidence refs.",
            )

        observed_job_ids: set[str] = set()
        observed_event_refs: set[str] = set()
        for step in steps:
            if step.observation.status.value not in {"ok", "empty"}:
                continue
            observed_event_refs.add(f"step:{step.step_index}:{step.action.tool_name}")
            if step.action.tool_name == SEARCH_TOOL:
                for job in step.observation.data.get("jobs", []):
                    if isinstance(job, dict) and isinstance(job.get("job_id"), str):
                        observed_job_ids.add(job["job_id"])
            if step.action.tool_name == READ_TOOL:
                job = step.observation.data.get("job")
                if isinstance(job, dict) and isinstance(job.get("job_id"), str):
                    observed_job_ids.add(job["job_id"])
            if step.action.tool_name == DEDUPE_TOOL:
                observed_job_ids.update(
                    job_id
                    for job_id in step.observation.data.get("job_ids", [])
                    if isinstance(job_id, str)
                )

        if len(result.candidate_job_ids) < typed_goal.min_candidates:
            return VerificationResult(
                passed=False,
                reason_code="insufficient_candidates",
                feedback=f"Find at least {typed_goal.min_candidates} grounded candidate jobs.",
            )
        unknown = sorted(set(result.candidate_job_ids) - observed_job_ids)
        if unknown:
            return VerificationResult(
                passed=False,
                reason_code="ungrounded_candidate",
                feedback=f"Candidate job ids were not observed from tools: {', '.join(unknown)}.",
            )
        if not set(result.evidence_refs).issubset(observed_event_refs):
            return VerificationResult(
                passed=False,
                reason_code="invalid_evidence_ref",
                feedback="Use step:<index>:<tool_name> evidence refs from successful observations.",
            )
        return VerificationResult(
            passed=True,
            reason_code="opportunity_research_complete",
            feedback="Opportunity research completion criteria passed.",
            evidence_refs=result.evidence_refs,
        )


class OpportunityResearchAgent:
    def __init__(
        self,
        *,
        model: AgentModel,
        source_path: Path | str,
        budget: AgentBudget | None = None,
        trajectory=None,
        max_same_verifier_reason: int | None = None,
    ) -> None:
        self.source_path = Path(source_path).resolve()
        self.registry = build_opportunity_tool_registry(self.source_path)
        policy_context = PolicyContext(
            skill_allowed_tools=OPPORTUNITY_TOOLS,
            agent_allowed_tools=OPPORTUNITY_TOOLS,
            runtime_allowed_tools=OPPORTUNITY_TOOLS,
        )
        self.loop = AgentLoop(
            agent_id="opportunity-research",
            model=model,
            registry=self.registry,
            policy=PolicyEngine(),
            policy_context=policy_context,
            budget=BudgetManager(budget or AgentBudget.for_profile(BudgetProfile.STANDARD)),
            verifier=OpportunityResearchVerifier(),
            trajectory=trajectory,
            max_same_verifier_reason=max_same_verifier_reason,
        )

    def run(
        self,
        goal: OpportunityResearchGoal,
        *,
        session_id: str,
        run_id: str,
        workspace_root: Path | str,
    ) -> AgentRunResult:
        return self.loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal.model_dump(mode="json"),
            tool_context=ToolContext(
                session_id=session_id,
                run_id=run_id,
                agent_id="opportunity-research",
                workspace_root=str(Path(workspace_root).resolve()),
            ),
        )
