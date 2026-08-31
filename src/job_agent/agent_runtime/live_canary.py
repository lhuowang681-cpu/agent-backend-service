from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Callable, Literal, Protocol
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, model_validator

from job_agent.agent_runtime.contracts import AgentDecisionEnvelope, AgentRunStatus
from job_agent.agent_runtime.decision_model import LLMDecisionModel
from job_agent.agent_runtime.tool_use_decision_model import ToolUseDecisionModel
from job_agent.agent_runtime.eval_cases import HarnessEvalCase, load_harness_eval_cases
from job_agent.agent_runtime.trajectory import InMemoryTrajectory
from job_agent.domain_agents.opportunity_research import (
    OPPORTUNITY_TOOLS,
    OpportunityResearchAgent,
    OpportunityResearchGoal,
    OpportunityResearchResult,
    build_opportunity_tool_registry,
)
from job_agent.llm.harness import LLMHarness
from job_agent.llm.provider import LLMProvider, ProviderError
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import StrictModel


class LiveCanaryError(RuntimeError):
    pass


class LiveCanaryConfig(StrictModel):
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    cases_path: str = "data/eval/job_agent_harness_v1_cases.json"
    source_path: str = "data/fixtures/raw_jobs_llm_intern_skill_scout_sample.json"
    timeout_s: float = Field(default=60.0, gt=0, le=300)
    allow_network: bool = False
    allow_fallback: Literal[False] = False
    scope: Literal["opportunity", "full"] = "opportunity"
    allow_sandbox_side_effects: bool = False
    agent_protocol: Literal["legacy_structured", "tool_use_v2"] = "legacy_structured"
    semantic_artifacts: Literal["fixture", "api"] = "fixture"
    skill_root: str | None = None
    allowed_hosts: tuple[str, ...] = ("open.bigmodel.cn",)

    @model_validator(mode="after")
    def validate_endpoint(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https":
            raise ValueError("live canary requires https")
        if parsed.hostname not in self.allowed_hosts:
            raise ValueError("live canary endpoint host is not allowlisted")
        if self.semantic_artifacts == "api":
            if self.scope != "full":
                raise ValueError("semantic_artifacts=api requires scope=full")
            if not self.skill_root:
                raise ValueError("semantic_artifacts=api requires skill_root")
        return self


class LiveStageAttempt(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    stage: str = Field(min_length=1)
    status: Literal["passed", "failed"]
    model_calls: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    resumes: int = Field(default=0, ge=0)
    approvals: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_detail: str | None = None
    outcome: str | None = None


class LiveCanaryAttempt(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    case_id: str
    stage_coverage: list[str] = Field(min_length=1)
    status: Literal["passed", "failed"]
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    provider_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    fallback_used: Literal[False] = False
    error_code: str | None = None
    stages: list[LiveStageAttempt] = Field(default_factory=list)
    resumes: int = Field(default=0, ge=0)
    approvals: int = Field(default=0, ge=0)
    evaluation_passed: bool | None = None
    error_detail: str | None = None


class LiveCanaryReport(StrictModel):
    report_version: Literal[2] = 2
    started_at: str
    finished_at: str
    provider: str
    model: str
    base_url: str
    credential_source: str
    cases_path: str
    scope: Literal["opportunity", "full"] = "opportunity"
    sandbox_side_effects: bool = False
    agent_protocol: Literal["legacy_structured", "tool_use_v2"] = "legacy_structured"
    semantic_artifacts: Literal["fixture", "api"] = "fixture"
    passed: bool
    attempts: list[LiveCanaryAttempt] = Field(min_length=1)
    disclaimer: str = "Single canary runs do not establish SLA or cross-sample model stability."


class CanaryCaseExecutor(Protocol):
    def __call__(
        self,
        case: HarnessEvalCase,
        provider: LLMProvider,
        config: LiveCanaryConfig,
    ) -> LiveCanaryAttempt:
        ...


def opportunity_live_case_executor(
    case: HarnessEvalCase,
    provider: LLMProvider,
    config: LiveCanaryConfig,
) -> LiveCanaryAttempt:
    registry = build_opportunity_tool_registry(config.source_path)
    skill = SkillSpec(
        skill_id="opportunity-research-live-canary",
        version="v1",
        instructions=(
            "Use only approved local job tools. Search before finishing. After a successful search, finish "
            "with result fields search_queries, candidate_job_ids, evidence_refs, and unresolved_gaps. Copy only "
            "job_id values observed in tool results and exact refs in the form step:<step_index>:<tool_name>. "
            "Once min_candidates are grounded, return decision.kind=finish."
        ),
        reference_paths=(),
        output_schema=AgentDecisionEnvelope,
        allowed_tools=OPPORTUNITY_TOOLS,
        guardrails=("Never invent a job id.", "Do not request external writes."),
    )
    run_id = f"live-{case.case_id}"
    if config.agent_protocol == "tool_use_v2":
        skill = SkillSpec(
            skill_id="opportunity-research-live-canary-v2",
            version="v2",
            instructions=skill.instructions,
            reference_paths=(),
            output_schema=OpportunityResearchResult,
            allowed_tools=OPPORTUNITY_TOOLS,
            guardrails=skill.guardrails,
        )
        model = ToolUseDecisionModel(
            provider=provider,
            skill=skill,
            tools=registry.provider_specs(),
            result_schema=OpportunityResearchResult,
            submit_tool_name="submit_opportunity_result",
            session_id="live-canary",
            run_id=run_id,
            agent_id="opportunity-research",
            prompt_version="opportunity-tool-use-v2",
        )
    else:
        model = LLMDecisionModel(
            harness=LLMHarness(provider),
            skill=skill,
            tools=registry.provider_specs(),
            session_id="live-canary",
            run_id=run_id,
            agent_id="opportunity-research",
            prompt_version="live-canary-controller-v1",
        )
    trajectory = InMemoryTrajectory()
    agent = OpportunityResearchAgent(
        model=model,
        source_path=config.source_path,
        trajectory=trajectory,
        max_same_verifier_reason=2 if config.agent_protocol == "tool_use_v2" else None,
    )
    started = perf_counter()
    result = agent.run(
        OpportunityResearchGoal(
            target_role=case.target_role,
            cities=case.cities,
            keywords=case.keywords,
            min_candidates=case.min_candidates,
        ),
        session_id="live-canary",
        run_id=run_id,
        workspace_root=Path.cwd(),
    )
    latency_ms = max(int((perf_counter() - started) * 1000), 0)
    model_events = [event for event in trajectory.events if event.event_type == "model_decision"]
    tool_events = [event for event in trajectory.events if event.event_type == "tool_observed"]
    verification_events = [event for event in trajectory.events if event.event_type == "verification"]
    input_values = [event.input_tokens for event in model_events if event.input_tokens is not None]
    output_values = [event.output_tokens for event in model_events if event.output_tokens is not None]
    passed = result.state.status == AgentRunStatus.COMPLETED
    error_code = result.error_code
    if not passed and model.traces:
        provider_error = model.traces[-1].error_code
        if provider_error and error_code in {None, "model_error"}:
            error_code = provider_error
    error_detail = model.traces[-1].error_detail if model.traces else None
    if not passed and error_detail is None and verification_events:
        error_detail = f"last_verification:{verification_events[-1].verification.reason_code}"
    stage = LiveStageAttempt(
        stage="opportunity-research",
        status="passed" if passed else "failed",
        model_calls=len(model_events),
        provider_calls=len(model.traces),
        tool_calls=len(tool_events),
        input_tokens=sum(input_values) if input_values else None,
        output_tokens=sum(output_values) if output_values else None,
        error_code=error_code,
        error_detail=error_detail,
    )
    return LiveCanaryAttempt(
        case_id=case.case_id,
        stage_coverage=["opportunity-research"],
        status="passed" if passed else "failed",
        provider=getattr(provider, "provider_name", "unknown"),
        model=getattr(provider, "model", config.model),
        latency_ms=latency_ms,
        model_calls=len(model_events),
        provider_calls=len(model.traces),
        tool_calls=len(tool_events),
        input_tokens=sum(input_values) if input_values else None,
        output_tokens=sum(output_values) if output_values else None,
        error_code=error_code,
        stages=[stage],
        error_detail=stage.error_detail,
    )


def full_live_case_executor(
    case: HarnessEvalCase,
    provider: LLMProvider,
    config: LiveCanaryConfig,
) -> LiveCanaryAttempt:
    """Run all four real domain controllers in a disposable local sandbox."""

    from job_agent.agent_runtime.live_full_e2e import run_full_live_case

    return run_full_live_case(case, provider, config)


class LiveCanaryRunner:
    def __init__(
        self,
        *,
        provider_factory: Callable[[LiveCanaryConfig, str], LLMProvider] | None = None,
        case_executor: CanaryCaseExecutor | None = None,
        clock=lambda: datetime.now(UTC),
    ) -> None:
        self.provider_factory = provider_factory or self._provider
        self.case_executor = case_executor
        self._clock = clock

    def run(self, config: LiveCanaryConfig) -> LiveCanaryReport:
        if not config.allow_network:
            raise LiveCanaryError("network_not_explicitly_enabled")
        if config.scope == "full" and not config.allow_sandbox_side_effects:
            raise LiveCanaryError("sandbox_side_effects_not_explicitly_enabled")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise LiveCanaryError("credential_not_injected")
        provider = self.provider_factory(config, api_key)
        del api_key
        cases = load_harness_eval_cases(config.cases_path)
        started = self._clock()
        attempts: list[LiveCanaryAttempt] = []
        case_executor = self.case_executor or (
            full_live_case_executor if config.scope == "full" else opportunity_live_case_executor
        )
        for case in cases:
            attempt_started = perf_counter()
            try:
                attempt = case_executor(case, provider, config)
            except ProviderError as exc:
                attempt = self._failure_attempt(
                    case,
                    config,
                    provider,
                    max(int((perf_counter() - attempt_started) * 1000), 0),
                    exc.error_code,
                )
            except Exception:
                attempt = self._failure_attempt(
                    case,
                    config,
                    provider,
                    max(int((perf_counter() - attempt_started) * 1000), 0),
                    "canary_execution_error",
                )
            attempts.append(attempt)
        finished = self._clock()
        return LiveCanaryReport(
            started_at=started.astimezone(UTC).isoformat(),
            finished_at=finished.astimezone(UTC).isoformat(),
            provider=getattr(provider, "provider_name", "unknown"),
            model=getattr(provider, "model", config.model),
            base_url=config.base_url,
            credential_source=config.api_key_env,
            cases_path=config.cases_path,
            scope=config.scope,
            sandbox_side_effects=config.allow_sandbox_side_effects,
            agent_protocol=config.agent_protocol,
            semantic_artifacts=config.semantic_artifacts,
            passed=all(attempt.status == "passed" for attempt in attempts),
            attempts=attempts,
        )

    @staticmethod
    def write_report(path: Path | str, report: LiveCanaryReport) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(report.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(target)

    @staticmethod
    def _provider(config: LiveCanaryConfig, api_key: str) -> LLMProvider:
        return AnthropicCompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout_s=config.timeout_s,
        )

    @staticmethod
    def _failure_attempt(
        case: HarnessEvalCase,
        config: LiveCanaryConfig,
        provider: LLMProvider,
        latency_ms: int,
        error_code: str,
    ) -> LiveCanaryAttempt:
        return LiveCanaryAttempt(
            case_id=case.case_id,
            stage_coverage=["opportunity-research"],
            status="failed",
            provider=getattr(provider, "provider_name", "unknown"),
            model=getattr(provider, "model", config.model),
            latency_ms=latency_ms,
            model_calls=0,
            tool_calls=0,
            error_code=error_code,
        )
