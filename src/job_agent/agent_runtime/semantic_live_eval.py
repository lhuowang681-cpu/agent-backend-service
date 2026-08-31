from __future__ import annotations

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Callable, Literal, Mapping, Protocol
from urllib.parse import urlparse

from pydantic import Field, model_validator

from job_agent.agent_runtime.semantic_evaluation import (
    SemanticEvalSuiteReport,
    build_agent_replay_snapshot,
    build_semantic_execution_failure,
    evaluate_semantic_snapshot,
    load_semantic_eval_cases,
    select_stratified_semantic_cases,
    summarize_provider_traces,
)
from job_agent.agents.base import AgentGuardError
from job_agent.graph import run_semantic_job_flow
from job_agent.llm.harness import LLMHarness, LLMInvocationError, NodePolicy
from job_agent.llm.provider import LLMProvider, ProviderError, ProviderTrace
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.skill_registry import SkillRegistry
from job_agent.schemas import StrictModel
from job_agent.tools.job_search import load_raw_jobs


_SAMPLE_STRATEGY = (
    "stratified_v1:per_family=2;priorities=grounding,prompt_injection,negative_control"
)
_SAFE_ERROR_CODE = re.compile(r"^[a-z0-9_]{1,80}$")


class SemanticLiveEvalError(RuntimeError):
    pass


class SemanticLiveEvalConfig(StrictModel):
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = Field(min_length=1)
    api_key_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,127}$")
    cases_path: str = "data/eval/job_agent_semantic_phase_c_cases.json"
    source_path: str = "data/fixtures/raw_jobs_llm_intern_skill_scout_sample.json"
    skill_root: str = Field(min_length=1)
    timeout_s: float = Field(default=90.0, gt=0, le=300)
    allow_network: bool = False
    allow_fallback: Literal[False] = False
    per_family: int = Field(default=2, ge=1, le=4)
    case_ids: tuple[str, ...] = ()
    max_retries: int = Field(default=1, ge=0, le=2)
    max_provider_calls: int = Field(default=80, ge=1, le=160)
    allowed_hosts: tuple[str, ...] = ("open.bigmodel.cn",)

    @model_validator(mode="after")
    def validate_endpoint_and_budget(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https":
            raise ValueError("semantic live eval requires https")
        if parsed.hostname not in self.allowed_hosts:
            raise ValueError("semantic live eval endpoint host is not allowlisted")
        return self


ProviderFactory = Callable[[SemanticLiveEvalConfig, str], LLMProvider]
RegistryFactory = Callable[[str], object]


class SemanticCaseExecutor(Protocol):
    def __call__(
        self,
        *,
        case,
        selected_job,
        resume_path: Path,
        registry: SkillRegistry,
        harness: LLMHarness,
        config: SemanticLiveEvalConfig,
        forbidden_output_strings: tuple[str, ...],
    ) -> Mapping[str, object]: ...


class SemanticLiveEvalRunner:
    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        registry_factory: RegistryFactory | None = None,
        case_executor: SemanticCaseExecutor | None = None,
    ) -> None:
        self.provider_factory = provider_factory or self._provider
        self.registry_factory = registry_factory or SkillRegistry
        self.case_executor = case_executor or self._execute_case

    def run(self, config: SemanticLiveEvalConfig) -> SemanticEvalSuiteReport:
        if not config.allow_network:
            raise SemanticLiveEvalError("network_not_explicitly_enabled")
        loaded_cases = load_semantic_eval_cases(config.cases_path)
        if config.case_ids:
            if len(config.case_ids) != len(set(config.case_ids)):
                raise SemanticLiveEvalError("duplicate_case_id")
            by_id = {case.case_id: case for case in loaded_cases}
            try:
                cases = [by_id[case_id] for case_id in config.case_ids]
            except KeyError as exc:
                raise SemanticLiveEvalError("unknown_case_id") from exc
            sample_strategy = "explicit_case_ids"
        else:
            cases = select_stratified_semantic_cases(
                loaded_cases,
                per_family=config.per_family,
            )
            sample_strategy = _SAMPLE_STRATEGY
        semantic_nodes = sum(
            2 if case.expected_outcome == "stop_and_reselect" else 4 for case in cases
        )
        maximum_attempts = semantic_nodes * 2 * (config.max_retries + 1)
        if maximum_attempts > config.max_provider_calls:
            raise SemanticLiveEvalError("provider_call_budget_too_small")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise SemanticLiveEvalError("credential_not_injected")
        provider = self.provider_factory(config, api_key)
        del api_key

        jobs = {job.job_id: job for job in load_raw_jobs(Path(config.source_path))}
        registry = self.registry_factory(config.skill_root)
        reports = []
        all_traces: list[ProviderTrace] = []
        for case in cases:
            harness = LLMHarness(provider)
            try:
                source_job = jobs[case.selected_job_id]
                selected_job = source_job.model_copy(
                    update={"desc": source_job.desc + case.untrusted_job_suffix}
                )
                resume_text = (
                    Path(case.resume_fixture).read_text(encoding="utf-8")
                    + case.untrusted_resume_suffix
                )
                with TemporaryDirectory(prefix="semantic-live-eval-") as temporary:
                    resume_path = Path(temporary) / "resume.md"
                    resume_path.write_text(resume_text, encoding="utf-8")
                    state = self.case_executor(
                        case=case,
                        selected_job=selected_job,
                        resume_path=resume_path,
                        registry=registry,
                        harness=harness,
                        config=config,
                        forbidden_output_strings=(
                            (case.injection_sentinel,)
                            if case.injection_sentinel
                            else ()
                        ),
                    )
                snapshot = build_agent_replay_snapshot(case, state)
                reports.append(evaluate_semantic_snapshot(case, snapshot))
            except (KeyError, ProviderError, LLMInvocationError, AgentGuardError) as exc:
                reports.append(
                    build_semantic_execution_failure(
                        case,
                        error_code=self._error_code(exc),
                        traces=list(harness.traces),
                    )
                )
            except Exception:
                reports.append(
                    build_semantic_execution_failure(
                        case,
                        error_code="semantic_eval_execution_error",
                        traces=list(harness.traces),
                    )
                )
            all_traces.extend(harness.traces)
            if len(all_traces) > config.max_provider_calls:
                raise SemanticLiveEvalError("provider_call_budget_exceeded")

        passed_cases = sum(report.passed for report in reports)
        return SemanticEvalSuiteReport(
            mode="agent_replay",
            case_count=len(cases),
            passed=passed_cases == len(cases),
            passed_cases=passed_cases,
            reports=reports,
            provider=getattr(provider, "provider_name", "unknown"),
            model=getattr(provider, "model", config.model),
            base_url=config.base_url,
            credential_source=config.api_key_env,
            sample_strategy=sample_strategy,
            provider_summary=summarize_provider_traces(all_traces),
        )

    @staticmethod
    def write_report(path: Path | str, report: SemanticEvalSuiteReport) -> None:
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
    def _execute_case(
        *,
        case,
        selected_job,
        resume_path: Path,
        registry: SkillRegistry,
        harness: LLMHarness,
        config: SemanticLiveEvalConfig,
        forbidden_output_strings: tuple[str, ...],
    ) -> Mapping[str, object]:
        return run_semantic_job_flow(
            user_request=f"semantic evaluation case {case.case_id}",
            selected_job=selected_job,
            resume_path=resume_path,
            registry=registry,
            harness=harness,
            policy=NodePolicy(
                timeout_s=config.timeout_s,
                max_retries=config.max_retries,
                temperature=0.0,
                max_output_tokens=4096,
                allow_rule_fallback=False,
            ),
            session_id=f"semantic-eval:{case.case_id}",
            run_id=f"semantic-eval:{case.case_id}:agent-replay",
            forbidden_output_strings=forbidden_output_strings,
        )

    @staticmethod
    def _provider(config: SemanticLiveEvalConfig, api_key: str) -> LLMProvider:
        return AnthropicCompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout_s=config.timeout_s,
        )

    @staticmethod
    def _error_code(exc: Exception) -> str:
        code = getattr(exc, "error_code", None)
        if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code):
            return code
        if isinstance(exc, KeyError):
            return "unknown_selected_job"
        return "semantic_eval_execution_error"
