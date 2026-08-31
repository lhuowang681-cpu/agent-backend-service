from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Callable, Protocol

from pydantic import ValidationError

from job_agent.agent_runtime.contracts import AgentRunStatus, ToolContext
from job_agent.agent_runtime.failures import CheckpointError
from job_agent.agent_runtime.loop import AgentLoop
from job_agent.backend_service.checkpoint_adapter import PostgresAgentCheckpointStore
from job_agent.backend_service.career_snapshot import CareerSnapshotNotFoundError
from job_agent.backend_service.contracts import (
    AgentApprovalPause,
    AgentExecutionOutcome,
    ClaimedRun,
    ExecutionManifest,
    RunRecord,
)
from job_agent.backend_service.postgres_repository import (
    ApprovalPersistenceError,
    PostgresRunRepository,
)
from job_agent.backend_service.provider_admission import (
    AdmissionControlledProvider,
    RedisProviderAdmissionController,
)
from job_agent.backend_service.tool_operations import (
    ToolOperationOutcomeUncertainError,
)
from job_agent.graph import run_semantic_job_flow
from job_agent.llm.harness import LLMHarness, LLMInvocationError, NodePolicy
from job_agent.llm.provider import LLMProvider
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import (
    AnswerCardDeck,
    EvidenceMappingResult,
    FitInput,
    InterviewPrep,
    RawJob,
    StructuredJD,
    TargetedResume,
)


class AgentExecutionFailedError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class AgentExecutionAdapter(Protocol):
    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome | AgentApprovalPause: ...


class RoutedExecutionAdapter:
    """Select an execution adapter from the persisted, allowlisted Provider profile."""

    def __init__(self, adapters: dict[str, AgentExecutionAdapter]) -> None:
        self.adapters = dict(adapters)

    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome | AgentApprovalPause:
        try:
            adapter = self.adapters[record.request.provider_profile]
        except KeyError as exc:
            raise AgentExecutionFailedError("provider_profile_not_configured") from exc
        return adapter.execute(
            record,
            progress_callback=progress_callback,
            claim=claim,
        )


class TaskRoutedExecutionAdapter:
    """Route only from the persisted task_type, never from queue metadata."""

    def __init__(self, adapters: dict[str, AgentExecutionAdapter]) -> None:
        self.adapters = dict(adapters)

    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome | AgentApprovalPause:
        try:
            adapter = self.adapters[record.task_type]
        except KeyError as exc:
            raise AgentExecutionFailedError("task_type_not_configured") from exc
        return adapter.execute(
            record,
            progress_callback=progress_callback,
            claim=claim,
        )


class ManifestValidatingExecutionAdapter:
    """Fail closed when a recovered Run no longer matches its frozen runtime."""

    def __init__(
        self,
        *,
        adapter: AgentExecutionAdapter,
        expected_manifest_factory: Callable[[RunRecord], ExecutionManifest],
    ) -> None:
        self.adapter = adapter
        self.expected_manifest_factory = expected_manifest_factory

    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome | AgentApprovalPause:
        expected = self.expected_manifest_factory(record)
        if record.execution_manifest is None:
            raise AgentExecutionFailedError("execution_manifest_missing")
        if record.execution_manifest != expected:
            raise AgentExecutionFailedError("execution_manifest_incompatible")
        return self.adapter.execute(
            record,
            progress_callback=progress_callback,
            claim=claim,
        )


class DomainAgentLoopFactory(Protocol):
    def __call__(
        self,
        record: RunRecord,
        checkpoint_store: PostgresAgentCheckpointStore,
        claim: ClaimedRun,
    ) -> AgentLoop: ...


class DomainAgentExecutionAdapter:
    """Service boundary around the existing AgentLoop and its runtime contracts."""

    def __init__(
        self,
        *,
        repository: PostgresRunRepository,
        loop_factory: DomainAgentLoopFactory,
        goal_factory: Callable[[RunRecord], dict[str, object]],
        tool_context_factory: Callable[[RunRecord, str], ToolContext],
    ) -> None:
        self.repository = repository
        self.loop_factory = loop_factory
        self.goal_factory = goal_factory
        self.tool_context_factory = tool_context_factory

    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome | AgentApprovalPause:
        if claim is None:
            raise AgentExecutionFailedError("agent_attempt_fence_missing")
        checkpoint_store = PostgresAgentCheckpointStore(
            repository=self.repository,
            claim=claim,
        )
        try:
            loop = self.loop_factory(record, checkpoint_store, claim)
            goal = self.goal_factory(record)
            context = self.tool_context_factory(record, loop.agent_id)
            if (
                context.session_id != record.session_id
                or context.run_id != record.run_id
                or context.agent_id != loop.agent_id
            ):
                raise ValueError("tool_context_identity_mismatch")
            approvals = self.repository.load_resolved_approval_decisions(
                user_id=record.user_id,
                run_id=record.run_id,
            )
            resume = checkpoint_store.exists()
            progress_callback("agent_resuming" if resume else "agent_running")
            result = loop.run(
                session_id=record.session_id,
                run_id=record.run_id,
                goal=goal,
                tool_context=context,
                approvals=approvals,
                resume=resume,
            )
            provider_call_count = loop.budget.snapshot().model_calls
        except LLMInvocationError as exc:
            raise AgentExecutionFailedError(exc.error_code) from exc
        except ToolOperationOutcomeUncertainError as exc:
            raise AgentExecutionFailedError(
                "non_idempotent_execution_uncertain"
            ) from exc
        except (CheckpointError, ApprovalPersistenceError, ValidationError) as exc:
            raise AgentExecutionFailedError("agent_contract_failed") from exc
        except ValueError as exc:
            raise AgentExecutionFailedError("agent_input_failed") from exc
        except OSError as exc:
            raise AgentExecutionFailedError("agent_io_failed") from exc
        except RuntimeError as exc:
            raise AgentExecutionFailedError("agent_runtime_failed") from exc

        if result.state.status == AgentRunStatus.COMPLETED:
            progress_callback("agent_completed")
            return AgentExecutionOutcome(
                result=result.result,
                provider_call_count=provider_call_count,
            )
        if result.pending_approval is not None and result.pending_action is not None:
            progress_callback("approval_required")
            return AgentApprovalPause(
                approval_request=result.pending_approval,
                pending_action=result.pending_action,
                provider_call_count=provider_call_count,
            )
        if result.pending_user_input is not None:
            raise AgentExecutionFailedError("unsupported_user_input_wait")
        if result.state.status == AgentRunStatus.CANCELLED:
            raise AgentExecutionFailedError("agent_cancelled")
        if result.state.status == AgentRunStatus.BUDGET_EXCEEDED:
            raise AgentExecutionFailedError(result.error_code or "budget_exceeded")
        if result.state.status == AgentRunStatus.FAILED:
            raise AgentExecutionFailedError(result.error_code or "agent_runtime_failed")
        raise AgentExecutionFailedError("agent_result_contract_invalid")


class ResumableSemanticCheckpointStore(Protocol):
    def exists(self) -> bool: ...


class InMemoryCareerSnapshotStore:
    """Private, user-scoped resume snapshots for the tracer bullet."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[tuple[str, str], str] = {}

    def put(self, *, user_id: str, resume_ref: str, resume_text: str) -> None:
        if not user_id or not resume_ref:
            raise ValueError("user_id and resume_ref are required")
        with self._lock:
            self._snapshots[(user_id, resume_ref)] = resume_text

    def resolve(self, *, user_id: str, resume_ref: str) -> str:
        with self._lock:
            try:
                return self._snapshots[(user_id, resume_ref)]
            except KeyError as exc:
                raise CareerSnapshotNotFoundError("career_snapshot_not_found") from exc


class TracerSemanticRegistry:
    """Schema-only registry adapter for the controlled Mock Provider tracer."""

    version = "backend-tracer-v1"

    _SCHEMAS = {
        "jd-analysis": StructuredJD,
        "evidence-contract": EvidenceMappingResult,
        "resume-tailoring": TargetedResume,
        "interview-grilling": InterviewPrep,
        "answer-cards": AnswerCardDeck,
    }

    def get(self, skill_id: str) -> SkillSpec:
        return SkillSpec(
            skill_id=skill_id,
            version=self.version,
            instructions=f"Controlled backend tracer instructions for {skill_id}.",
            reference_paths=(Path(f"skill-references/{skill_id}.md"),),
            output_schema=self._SCHEMAS[skill_id],
            guardrails=("Never fabricate evidence.",),
        )


class SchemaMappedMockLLMProvider:
    """Deterministic fixtures keyed by schema so checkpoint resume cannot misalign them."""

    provider_name = "mock"
    model = "fixture-model"
    _OUTPUT_SCHEMAS = (
        StructuredJD,
        EvidenceMappingResult,
        TargetedResume,
        InterviewPrep,
        AnswerCardDeck,
    )

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = dict(zip(self._OUTPUT_SCHEMAS, responses, strict=False))

    def generate_structured(self, **kwargs):
        output_schema = kwargs["output_schema"]
        try:
            response = self._responses[output_schema]
        except KeyError as exc:
            raise ValueError("mock_response_schema_not_configured") from exc
        return MockLLMProvider([response]).generate_structured(**kwargs)


def build_mock_semantic_responses(*, job: RawJob, resume_path: Path) -> list[dict[str, object]]:
    """Build deterministic provider fixtures by reusing existing rule-domain functions."""
    from job_agent.nodes.answer_cards import build_answer_cards
    from job_agent.nodes.evidence_mapping import map_evidence
    from job_agent.nodes.fit_verdict import evaluate_fit
    from job_agent.nodes.interview_prep import prepare_interview
    from job_agent.nodes.jd_structurer import structure_jd
    from job_agent.nodes.resume_tailoring import tailor_resume

    structured = structure_jd(job)
    evidence = map_evidence(resume_path, structured.must_have)
    fit_result = evaluate_fit(
        FitInput(
            role_type=structured.role_type,
            requirements=structured.must_have,
            evidence=evidence,
            toy_signals=[],
        )
    )
    responses: list[dict[str, object]] = [
        structured.model_dump(mode="json"),
        {"items": [item.model_dump(mode="json") for item in evidence]},
    ]
    if fit_result.verdict.value != "not recommended":
        targeted = tailor_resume(structured, evidence)
        interview = prepare_interview(structured, evidence, fit_result, targeted)
        answer_cards = build_answer_cards(interview, evidence, targeted)
        responses.extend(
            [
                targeted.model_dump(mode="json"),
                interview.model_dump(mode="json"),
                answer_cards.model_dump(mode="json"),
            ]
        )
    return responses


class SemanticJobExecutionAdapter:
    """Maps a service Run onto the existing guarded semantic runtime seam."""

    def __init__(
        self,
        *,
        snapshots: InMemoryCareerSnapshotStore,
        registry: TracerSemanticRegistry | None = None,
        response_factory: Callable[..., list[dict[str, object]]] = build_mock_semantic_responses,
        checkpoint_store_factory: Callable[
            [RunRecord, ClaimedRun | None], ResumableSemanticCheckpointStore
        ]
        | None = None,
        provider_trace_callback: Callable[
            [str, float, bool, int | None, int | None], None
        ]
        | None = None,
        provider_factory: Callable[[RunRecord, Path], LLMProvider] | None = None,
        provider_admission: RedisProviderAdmissionController | None = None,
        policy: NodePolicy | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.registry = registry or TracerSemanticRegistry()
        self.response_factory = response_factory
        self.checkpoint_store_factory = checkpoint_store_factory
        self.provider_trace_callback = provider_trace_callback
        self.provider_factory = provider_factory
        self.provider_admission = provider_admission
        self.policy = policy
        self._lock = RLock()
        self._execution_counts: dict[str, int] = {}

    def execution_count(self, run_id: str) -> int:
        with self._lock:
            return self._execution_counts.get(run_id, 0)

    def execute(
        self,
        record: RunRecord,
        *,
        progress_callback: Callable[[str], None],
        claim: ClaimedRun | None = None,
    ) -> AgentExecutionOutcome:
        with self._lock:
            self._execution_counts[record.run_id] = self._execution_counts.get(record.run_id, 0) + 1
        harness: LLMHarness | None = None
        try:
            job_input = record.request.input
            resume_text = self.snapshots.resolve(
                user_id=record.user_id,
                resume_ref=job_input.resume_ref,
            )
            with TemporaryDirectory(prefix="job-agent-backend-") as temporary_dir:
                resume_path = Path(temporary_dir) / "resume.md"
                resume_path.write_text(resume_text, encoding="utf-8")
                provider = (
                    self.provider_factory(record, resume_path)
                    if self.provider_factory is not None
                    else SchemaMappedMockLLMProvider(
                        self.response_factory(
                            job=job_input.selected_job,
                            resume_path=resume_path,
                        )
                    )
                )
                if self.provider_admission is not None:
                    provider = AdmissionControlledProvider(
                        provider,
                        controller=self.provider_admission,
                        user_id=record.user_id,
                        provider_key=record.request.provider_profile,
                    )
                checkpoint_store = (
                    self.checkpoint_store_factory(record, claim)
                    if self.checkpoint_store_factory is not None
                    else None
                )
                harness = LLMHarness(provider)
                state = run_semantic_job_flow(
                    user_request="backend_semantic_job_flow",
                    selected_job=job_input.selected_job,
                    resume_path=resume_path,
                    registry=self.registry,
                    harness=harness,
                    policy=self.policy,
                    session_id=record.session_id,
                    run_id=record.run_id,
                    checkpoint_store=checkpoint_store,
                    resume=checkpoint_store is not None and checkpoint_store.exists(),
                    progress_callback=progress_callback,
                )
        except CareerSnapshotNotFoundError as exc:
            raise AgentExecutionFailedError("career_snapshot_not_found") from exc
        except LLMInvocationError as exc:
            raise AgentExecutionFailedError(exc.error_code) from exc
        except (CheckpointError, ValidationError) as exc:
            raise AgentExecutionFailedError("agent_contract_failed") from exc
        except OSError as exc:
            raise AgentExecutionFailedError("agent_io_failed") from exc
        except ValueError as exc:
            raise AgentExecutionFailedError("agent_input_failed") from exc
        finally:
            if harness is not None and self.provider_trace_callback is not None:
                for trace in harness.traces:
                    self.provider_trace_callback(
                        trace.provider,
                        trace.latency_ms / 1000.0,
                        trace.error_code is None,
                        trace.input_tokens,
                        trace.output_tokens,
                    )

        fit_result = state["fit_result"]
        verdict_route = state["verdict_route"]
        result: dict[str, object] = {
            "job_id": job_input.selected_job.job_id,
            "company": job_input.selected_job.company,
            "title": job_input.selected_job.title,
            "verdict": fit_result.verdict.value,
            "score": fit_result.score,
            "risk_level": fit_result.risk_level.value,
            "route": verdict_route.route,
            "gate": verdict_route.gate,
            "runtime_mode": str(state["runtime_mode"]),
            "llm_call_count": int(state["llm_call_count"]),
        }
        return AgentExecutionOutcome(
            result=result,
            provider_call_count=len(harness.traces) if harness is not None else 0,
        )
