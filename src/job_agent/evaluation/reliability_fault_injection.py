from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable
from unittest.mock import patch

from job_agent.agent_runtime.budgets import (
    AgentBudget,
    BudgetManager,
    BudgetProfile,
)
from job_agent.agent_runtime.checkpoint import JsonCheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentInputRequest,
    AgentRunStatus,
    UserInputRequest,
    UserInputResponse,
)
from job_agent.agent_runtime.failures import BudgetExceededError
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.agents.base import AgentGuardError
from job_agent.atomic_io import atomic_write_text, atomic_write_text_batch
from job_agent.domain_agents.interview_coach import (
    InterviewCoachAgent,
    InterviewCoachGoal,
    LOAD_INTERVIEW_TOOL,
)
from job_agent.llm.harness import LLMHarness, LLMInvocationError
from job_agent.llm.provider import (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderTrace,
)
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import (
    AnswerCard,
    AnswerCardDeck,
    EvidenceItem,
    EvidenceLevel,
    MockInterviewEvaluationResult,
    MockInterviewPlan,
    MockInterviewQuestion,
)
from job_agent.ui.live_action import execute_live_action
from job_agent.ui.mock_interview_evaluation_loop import (
    evaluation_paths,
    generate_mock_interview_evaluation_live,
)


EVIDENCE_DATE = "2026-07-28"
SYNTHETIC_SECRET = "fixture-sentinel-DO-NOT-LEAK"
SYNTHETIC_PRIVATE_PATH = r"private/candidate_resume.md"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, str | None]:
    return {path.name: _sha256(path) for path in paths}


def _scan_for_sentinels(
    values: Iterable[str | None],
    *,
    sentinels: tuple[str, ...] = (SYNTHETIC_SECRET, SYNTHETIC_PRIVATE_PATH),
) -> bool:
    rendered = "\n".join(value or "" for value in values)
    return any(sentinel in rendered for sentinel in sentinels)


def _diagnostic_text(result) -> str:
    if result.diagnostic_path is None or not result.diagnostic_path.is_file():
        return ""
    return result.diagnostic_path.read_text(encoding="utf-8")


def _case(
    *,
    case_id: str,
    action_id: str,
    runtime_path: str,
    injection_layer: str,
    evidence_scope: str,
    injected_failure: str,
    terminal_status: str | None,
    error_code: str | None,
    before_hashes: dict[str, str | None],
    after_hashes: dict[str, str | None],
    previous_version_preserved: bool | None,
    artifact_freshness_preserved: bool | None,
    checkpoint_preserved: bool | None,
    user_answer_preserved: bool | None,
    waiting_time_excluded_from_active_wall_time: bool | None,
    retry_after_recovery_success: bool | None,
    unhandled_traceback: bool | None,
    diagnostic_secret_leak: bool | None,
    repair_attempt_count: int | None,
    assertions: dict[str, bool],
    notes: list[str],
    result_override: str | None = None,
    budget_usage: dict[str, Any] | None = None,
    not_applicable_fields: list[str] | None = None,
    not_proven_fields: list[str] | None = None,
) -> dict[str, Any]:
    result = result_override or (
        "PASS" if assertions and all(assertions.values()) else "FAIL"
    )
    return {
        "case_id": case_id,
        "action_id": action_id,
        "runtime_path": runtime_path,
        "injection_layer": injection_layer,
        "evidence_scope": evidence_scope,
        "injected_failure": injected_failure,
        "terminal_status": terminal_status,
        "error_code": error_code,
        "before_artifact_sha256": before_hashes,
        "after_artifact_sha256": after_hashes,
        "previous_version_preserved": previous_version_preserved,
        "artifact_freshness_preserved": artifact_freshness_preserved,
        "checkpoint_preserved": checkpoint_preserved,
        "user_answer_preserved": user_answer_preserved,
        "waiting_time_excluded_from_active_wall_time": (
            waiting_time_excluded_from_active_wall_time
        ),
        "retry_after_recovery_success": retry_after_recovery_success,
        "unhandled_traceback": unhandled_traceback,
        "diagnostic_secret_leak": diagnostic_secret_leak,
        "repair_attempt_count": repair_attempt_count,
        "budget_usage": budget_usage,
        "assertions": assertions,
        "not_applicable_fields": sorted(not_applicable_fields or []),
        "not_proven_fields": sorted(not_proven_fields or []),
        "result": result,
        "notes": notes,
    }


def _seed_files(case_dir: Path, contents: dict[str, str]) -> list[Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, content in contents.items():
        path = case_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def _trace(*, error_code: str | None = None) -> ProviderTrace:
    return ProviderTrace(
        provider="fixture-provider",
        model="fixture-model",
        skill_id="resume-tailoring",
        skill_version="fixture-v1",
        prompt_version="fixture-p1",
        latency_ms=7,
        input_tokens=13,
        output_tokens=21,
        schema_valid=error_code is None,
        fallback_used=False,
        error_code=error_code,
        error_detail=f"{SYNTHETIC_SECRET}:{SYNTHETIC_PRIVATE_PATH}",
    )


def _run_live_action_case(
    *,
    case_dir: Path,
    case_id: str,
    action_id: str,
    runtime_path: str,
    injection_layer: str,
    injected_failure: str,
    seed_contents: dict[str, str],
    failure: Callable[[], Any],
    recovery: Callable[[], Any],
    expected_error_code: str,
    current_stage: str,
    expected_repair_count: int | None = None,
    extra_assertions: (
        dict[str, bool] | Callable[[], dict[str, bool]] | None
    ) = None,
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    protected = _seed_files(case_dir, seed_contents)
    before = _artifact_hashes(protected)
    failure_result = execute_live_action(
        action_id=action_id,
        operation=failure,
        session_dir=case_dir,
        current_stage=current_stage,
        previous_artifacts=protected,
    )
    after = _artifact_hashes(protected)
    diagnostic = _diagnostic_text(failure_result)
    no_traceback = "Traceback" not in (
        f"{failure_result.user_message or ''}\n{diagnostic}"
    )
    secret_leak = _scan_for_sentinels(
        [failure_result.user_message, diagnostic]
    )
    failure_diagnostic = failure_result.diagnostic or {}
    retry_result = execute_live_action(
        action_id=action_id,
        operation=recovery,
        session_dir=case_dir,
        current_stage=current_stage,
        previous_artifacts=protected,
    )
    state_name = "session_state.json"
    assertions = {
        "failure_was_contained": not failure_result.ok,
        "error_code_matches": failure_result.error_code == expected_error_code,
        "all_protected_sha256_match": before == after,
        "live_action_reports_previous_version_preserved": (
            failure_result.previous_version_preserved
        ),
        "retry_succeeds": retry_result.ok,
        "no_unhandled_traceback": no_traceback,
        "no_synthetic_secret_or_private_path_in_diagnostic": not secret_leak,
    }
    if expected_repair_count is not None:
        assertions["repair_count_matches"] = (
            failure_diagnostic.get("schema_repair_count")
            == expected_repair_count
        )
    resolved_extra_assertions = (
        extra_assertions()
        if callable(extra_assertions)
        else extra_assertions
    )
    assertions.update(resolved_extra_assertions or {})
    freshness_preserved = (
        state_name not in before or before[state_name] == after[state_name]
    )
    return _case(
        case_id=case_id,
        action_id=action_id,
        runtime_path=runtime_path,
        injection_layer=injection_layer,
        evidence_scope="ui_action_boundary_component",
        injected_failure=injected_failure,
        terminal_status=str(failure_diagnostic.get("status") or "failed"),
        error_code=failure_result.error_code,
        before_hashes=before,
        after_hashes=after,
        previous_version_preserved=(
            failure_result.previous_version_preserved and before == after
        ),
        artifact_freshness_preserved=freshness_preserved,
        checkpoint_preserved=None,
        user_answer_preserved=None,
        waiting_time_excluded_from_active_wall_time=None,
        retry_after_recovery_success=retry_result.ok,
        unhandled_traceback=not no_traceback,
        diagnostic_secret_leak=secret_leak,
        repair_attempt_count=(
            failure_diagnostic.get("schema_repair_count")
            if expected_repair_count is not None
            else None
        ),
        assertions=assertions,
        notes=extra_notes or [],
        not_applicable_fields=[
            "checkpoint_preserved",
            "user_answer_preserved",
            "waiting_time_excluded_from_active_wall_time",
        ],
    )


def _targeted_resume_provider_timeout(case_dir: Path) -> dict[str, Any]:
    resume = case_dir / "06_targeted_resume.md"

    def failure() -> None:
        raise ProviderTimeoutError(
            f"synthetic timeout {SYNTHETIC_SECRET} {SYNTHETIC_PRIVATE_PATH}"
        )

    def recovery() -> str:
        atomic_write_text(resume, "# Recovered targeted resume\n")
        return "recovered"

    return _run_live_action_case(
        case_dir=case_dir,
        case_id="resume_provider_timeout",
        action_id="regenerate_resume",
        runtime_path="Domain Agent -> UI live action",
        injection_layer="provider_error_at_operation_boundary",
        injected_failure="ProviderTimeoutError containing synthetic secret/path",
        seed_contents={
            "06_targeted_resume.md": "# Last known good resume\n",
            "07_interview_grilling.md": "# Existing interview prep\n",
            "session_state.json": '{"revision": 7, "targeted_resume": "fresh"}',
        },
        failure=failure,
        recovery=recovery,
        expected_error_code="timeout",
        current_stage="resume_tailoring",
        extra_notes=[
            "The provider exception is injected at the UI operation boundary; "
            "this is a component boundary test, not a real network call."
        ],
    )


def _targeted_resume_schema_destructive_rollback(
    case_dir: Path,
) -> dict[str, Any]:
    resume = case_dir / "06_targeted_resume.md"
    prep = case_dir / "07_interview_grilling.md"

    def failure() -> None:
        resume.write_text("# HALF-WRITTEN\n", encoding="utf-8")
        prep.unlink()
        raise LLMInvocationError("schema_error")

    def recovery() -> str:
        atomic_write_text(resume, "# Recovered schema-valid resume\n")
        return "recovered"

    return _run_live_action_case(
        case_dir=case_dir,
        case_id="resume_schema_error_destructive_rollback",
        action_id="regenerate_resume",
        runtime_path="Domain Agent -> UI live action",
        injection_layer="live_action.operation_after_partial_write",
        injected_failure=(
            "operation overwrites resume, deletes prep, then raises schema_error"
        ),
        seed_contents={
            "06_targeted_resume.md": "# Previous successful resume\n字节必须恢复。\n",
            "07_interview_grilling.md": "# Previous successful prep\n",
            "session_state.json": '{"revision": 11, "freshness": "fresh"}',
        },
        failure=failure,
        recovery=recovery,
        expected_error_code="schema_error",
        current_stage="resume_tailoring",
        extra_notes=[
            "This case intentionally mutates and deletes protected artifacts "
            "before the sanitized schema error is raised."
        ],
    )


def _targeted_resume_guard_error(case_dir: Path) -> dict[str, Any]:
    resume = case_dir / "06_targeted_resume.md"
    trace = _trace(error_code="ungrounded_high_risk_fact")

    def failure() -> None:
        raise AgentGuardError(
            "ungrounded_high_risk_fact",
            trace,
            node_id="resume-tailoring",
        )

    def recovery() -> str:
        atomic_write_text(resume, "# Guard-approved recovered resume\n")
        return "recovered"

    return _run_live_action_case(
        case_dir=case_dir,
        case_id="resume_semantic_guard_rejection",
        action_id="regenerate_resume",
        runtime_path="Semantic/Domain guard -> UI live action",
        injection_layer="semantic_guard",
        injected_failure="AgentGuardError(ungrounded_high_risk_fact)",
        seed_contents={
            "06_targeted_resume.md": "# Evidence-grounded old resume\n",
            "session_state.json": '{"revision": 5, "freshness": "fresh"}',
        },
        failure=failure,
        recovery=recovery,
        expected_error_code="ungrounded_high_risk_fact",
        current_stage="agent_guard",
        extra_notes=[
            "Guard thresholds and error codes are reused unchanged."
        ],
    )


def _interview_prep_provider_rate_limit(case_dir: Path) -> dict[str, Any]:
    prep = case_dir / "07_interview_grilling.md"
    cards = case_dir / "08_answer_cards.md"
    plan = case_dir / "09_mock_interview_plan.md"

    def failure() -> None:
        raise ProviderRateLimitError(
            f"synthetic rate limit {SYNTHETIC_SECRET}"
        )

    def recovery() -> str:
        atomic_write_text_batch(
            {
                prep: "# Recovered prep\n",
                cards: "# Recovered cards\n",
                plan: "# Recovered mock plan\n",
            }
        )
        return "recovered"

    return _run_live_action_case(
        case_dir=case_dir,
        case_id="interview_prep_provider_rate_limit",
        action_id="regenerate_interview_prep",
        runtime_path="Semantic Agent chain -> UI live action",
        injection_layer="provider_error_at_operation_boundary",
        injected_failure="ProviderRateLimitError containing synthetic secret",
        seed_contents={
            "07_interview_grilling.md": "# Previous prep\n",
            "08_answer_cards.md": "# Previous cards\n",
            "09_mock_interview_plan.md": "# Previous plan\n",
            "session_state.json": '{"revision": 9, "prep": "fresh"}',
        },
        failure=failure,
        recovery=recovery,
        expected_error_code="rate_limit",
        current_stage="interview_prep",
    )


def _interview_prep_atomic_commit_error(case_dir: Path) -> dict[str, Any]:
    import job_agent.atomic_io as atomic_io

    prep = case_dir / "07_interview_grilling.md"
    cards = case_dir / "08_answer_cards.md"
    plan = case_dir / "09_mock_interview_plan.md"
    protected = _seed_files(
        case_dir,
        {
            "07_interview_grilling.md": "# Atomic old prep\n",
            "08_answer_cards.md": "# Atomic old cards\n",
            "09_mock_interview_plan.md": "# Atomic old plan\n",
            "session_state.json": '{"revision": 12, "prep": "fresh"}',
        },
    )
    before = _artifact_hashes(protected)
    atomic_layer_restored = {"value": False}
    real_replace = atomic_io.os.replace
    replace_calls = {"count": 0}

    def fail_second_replace(source, target) -> None:
        replace_calls["count"] += 1
        if replace_calls["count"] == 2:
            raise OSError("synthetic second commit failure")
        real_replace(source, target)

    def failure() -> None:
        try:
            with patch.object(
                atomic_io.os,
                "replace",
                side_effect=fail_second_replace,
            ):
                atomic_write_text_batch(
                    {
                        prep: "# New prep that must roll back\n",
                        cards: "# New cards that must roll back\n",
                        plan: "# New plan that must roll back\n",
                    }
                )
        except OSError:
            atomic_layer_restored["value"] = (
                _artifact_hashes(protected) == before
            )
            raise

    def recovery() -> str:
        atomic_write_text_batch(
            {
                prep: "# Recovered atomic prep\n",
                cards: "# Recovered atomic cards\n",
                plan: "# Recovered atomic plan\n",
            }
        )
        return "recovered"

    return _run_live_action_case(
        case_dir=case_dir,
        case_id="interview_prep_atomic_batch_commit_error",
        action_id="regenerate_interview_prep",
        runtime_path="Semantic Agent artifact commit -> UI live action",
        injection_layer="atomic_io.os.replace",
        injected_failure="OSError on the second file commit",
        seed_contents={
            path.name: path.read_text(encoding="utf-8")
            for path in protected
        },
        failure=failure,
        recovery=recovery,
        expected_error_code="artifact_write_error",
        current_stage="artifact_commit",
        extra_assertions=lambda: {
            "atomic_batch_restored_before_ui_wrapper": (
                atomic_layer_restored["value"]
            )
        },
        extra_notes=[
            "The assertion distinguishes atomic_write_text_batch rollback "
            "from the outer UI last-known-good restoration."
        ],
    )


class _EvaluationRegistry:
    def get(self, skill_id: str) -> SkillSpec:
        return SkillSpec(
            skill_id=skill_id,
            version="fixture-v1",
            instructions="整轮评价；不得预测录用结果。",
            reference_paths=(),
            output_schema=MockInterviewEvaluationResult,
            guardrails=("不得编造经历。",),
        )


def _evaluation_contracts(
    *,
    run_id: str = "2026-07-28T120000",
) -> tuple[
    MockInterviewPlan,
    AnswerCardDeck,
    list[EvidenceItem],
    list[dict[str, Any]],
]:
    plan = MockInterviewPlan(
        company="示例公司",
        title="Agent 开发实习生",
        persona="技术面试官",
        questions=[
            MockInterviewQuestion(
                question_id="q1",
                requirement_id="req-1",
                prompt="如何恢复 Agent checkpoint？",
                focus="checkpoint",
            )
        ],
        scoring_dimensions=["技术深度"],
        live_rules=["逐题提问，整轮结束后评价。"],
    )
    cards = AnswerCardDeck(
        company=plan.company,
        title=plan.title,
        cards=[
            AnswerCard(
                requirement_id="req-1",
                question=plan.questions[0].prompt,
                short_answer="使用 checkpoint 和版本指纹。",
                evidence_level=EvidenceLevel.C2,
                supporting_evidence="代码和恢复测试。",
                boundary="未做生产容灾。",
            )
        ],
    )
    evidence = [
        EvidenceItem(
            evidence_id="ev-1",
            requirement_id="req-1",
            claim="实现 checkpoint 恢复",
            level=EvidenceLevel.C2,
            proof="状态恢复测试。",
            risk="仅在本地验证。",
        )
    ]
    history = [
        {
            "question_id": "q1",
            "requirement_id": "req-1",
            "prompt": plan.questions[0].prompt,
            "answer": "我保存 checkpoint，并验证版本指纹。",
            "elapsed_seconds": 60,
        }
    ]
    del run_id
    return plan, cards, evidence, history


def _evaluation_payload(
    *,
    run_id: str = "2026-07-28T120000",
    evidence_id: str = "ev-1",
    strength: str = "说明了 checkpoint。",
) -> dict[str, Any]:
    return {
        "company": "示例公司",
        "title": "Agent 开发实习生",
        "run_id": run_id,
        "question_evaluations": [
            {
                "question_id": "q1",
                "requirement_id": "req-1",
                "status": "partial",
                "relevance": 5,
                "technical_depth": 4,
                "reasoning_clarity": 4,
                "evidence_grounding": 4,
                "truth_boundary": 5,
                "strengths": [strength],
                "gaps": ["缺少失败路径。"],
                "improved_answer": "我使用 checkpoint 和版本指纹恢复状态。",
                "evidence_ids": [evidence_id],
                "next_focus": "补充失败语义。",
            }
        ],
        "overall_score": 1.0,
        "strengths": ["事实边界明确。"],
        "priority_gaps": ["失败路径。"],
        "next_mock_topics": ["checkpoint 版本漂移"],
        "limitations": ["只依据当前记录。"],
    }


def _evaluation_runtime(responses: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(
        registry=_EvaluationRegistry(),
        harness=LLMHarness(
            MockLLMProvider(responses),
            sleep=lambda _: None,
            jitter=lambda: 0.0,
        ),
    )


def _seed_evaluation_session(
    case_dir: Path,
) -> tuple[
    MockInterviewPlan,
    AnswerCardDeck,
    list[dict[str, Any]],
    Path,
    list[Path],
]:
    run_id = "2026-07-28T120000"
    plan, cards, evidence, history = _evaluation_contracts(run_id=run_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "02_evidence_mapping.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_path = case_dir / f"10_mock_interview_run_{run_id}.json"
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "result": {"transcript_complete": True},
                "history": history,
                "rule_evidence_audit": [],
                "evaluation_status": "ai_complete",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    json_path, markdown_path = evaluation_paths(run_path)
    json_path.write_text(
        json.dumps(
            _evaluation_payload(run_id=run_id),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text("# Previous AI evaluation\n", encoding="utf-8")
    debrief = case_dir / "15_mock_interview_debrief.md"
    debrief.write_text("# Previous debrief\n", encoding="utf-8")
    session_state = case_dir / "session_state.json"
    session_state.write_text(
        '{"revision": 4, "mock_evaluation": "fresh"}',
        encoding="utf-8",
    )
    return (
        plan,
        cards,
        history,
        run_path,
        [run_path, json_path, markdown_path, debrief, session_state],
    )


def _run_mock_evaluation_failure(
    *,
    case_dir: Path,
    case_id: str,
    responses: list[Any],
    expected_error_code: str,
    injected_failure: str,
    injection_layer: str,
    expected_schema_repairs: int,
) -> dict[str, Any]:
    plan, cards, history, run_path, protected = _seed_evaluation_session(
        case_dir
    )
    before = _artifact_hashes(protected)
    failure_runtime = _evaluation_runtime(responses)

    def failure() -> None:
        generate_mock_interview_evaluation_live(
            case_dir,
            run_path,
            plan=plan,
            history=history,
            answer_cards=cards,
            runtime=failure_runtime,
        )

    failure_result = execute_live_action(
        action_id="evaluate_mock_interview",
        operation=failure,
        session_dir=case_dir,
        harness=failure_runtime.harness,
        current_stage="mock_interview_evaluation",
        previous_artifacts=protected,
    )
    after = _artifact_hashes(protected)
    failure_diagnostic = failure_result.diagnostic or {}
    failure_diagnostic_text = _diagnostic_text(failure_result)
    secret_leak = _scan_for_sentinels(
        [failure_result.user_message, failure_diagnostic_text]
    )
    no_traceback = "Traceback" not in (
        f"{failure_result.user_message or ''}\n{failure_diagnostic_text}"
    )
    recovery_runtime = _evaluation_runtime([_evaluation_payload()])

    def recovery() -> Any:
        return generate_mock_interview_evaluation_live(
            case_dir,
            run_path,
            plan=plan,
            history=history,
            answer_cards=cards,
            runtime=recovery_runtime,
        )

    retry_result = execute_live_action(
        action_id="evaluate_mock_interview",
        operation=recovery,
        session_dir=case_dir,
        harness=recovery_runtime.harness,
        current_stage="mock_interview_evaluation",
        previous_artifacts=protected,
    )
    recovered_status = json.loads(run_path.read_text(encoding="utf-8")).get(
        "evaluation_status"
    )
    assertions = {
        "failure_was_contained": not failure_result.ok,
        "error_code_matches": failure_result.error_code == expected_error_code,
        "all_previous_evaluation_sha256_match": before == after,
        "previous_version_preserved_flag": (
            failure_result.previous_version_preserved
        ),
        "attempt_count_is_two": failure_diagnostic.get("attempt_count") == 2,
        "schema_repair_count_matches": (
            failure_diagnostic.get("schema_repair_count")
            == expected_schema_repairs
        ),
        "retry_succeeds": retry_result.ok,
        "recovery_marks_ai_complete": recovered_status == "ai_complete",
        "no_unhandled_traceback": no_traceback,
        "no_synthetic_secret_in_diagnostic": not secret_leak,
    }
    return _case(
        case_id=case_id,
        action_id="evaluate_mock_interview",
        runtime_path="MockInterviewEvaluationAgent -> LLMHarness -> UI live action",
        injection_layer=injection_layer,
        evidence_scope="semantic_agent_component_with_ui_boundary",
        injected_failure=injected_failure,
        terminal_status=str(failure_diagnostic.get("status") or "failed"),
        error_code=failure_result.error_code,
        before_hashes=before,
        after_hashes=after,
        previous_version_preserved=(
            failure_result.previous_version_preserved and before == after
        ),
        artifact_freshness_preserved=(
            before.get("session_state.json")
            == after.get("session_state.json")
        ),
        checkpoint_preserved=None,
        user_answer_preserved=None,
        waiting_time_excluded_from_active_wall_time=None,
        retry_after_recovery_success=(
            retry_result.ok and recovered_status == "ai_complete"
        ),
        unhandled_traceback=not no_traceback,
        diagnostic_secret_leak=secret_leak,
        repair_attempt_count=int(
            failure_diagnostic.get("schema_repair_count") or 0
        ),
        assertions=assertions,
        notes=[
            "The previous evaluation JSON/Markdown, run status, debrief, and "
            "session freshness bytes are protected as one last-known-good set."
        ],
        not_applicable_fields=[
            "checkpoint_preserved",
            "user_answer_preserved",
            "waiting_time_excluded_from_active_wall_time",
        ],
    )


def _mock_evaluation_schema_repair_exhausted(
    case_dir: Path,
) -> dict[str, Any]:
    return _run_mock_evaluation_failure(
        case_dir=case_dir,
        case_id="mock_evaluation_schema_repair_exhausted",
        responses=[SYNTHETIC_SECRET, SYNTHETIC_SECRET],
        expected_error_code="invalid_json",
        injected_failure=(
            "two invalid JSON provider responses exhaust one schema repair"
        ),
        injection_layer="LLMHarness.schema_validation_and_repair",
        expected_schema_repairs=1,
    )


def _mock_evaluation_guard_repair_exhausted(
    case_dir: Path,
) -> dict[str, Any]:
    invalid = _evaluation_payload(
        evidence_id="unknown-evidence",
        strength=SYNTHETIC_SECRET,
    )
    return _run_mock_evaluation_failure(
        case_dir=case_dir,
        case_id="mock_evaluation_guard_repair_exhausted",
        responses=[invalid, invalid],
        expected_error_code="unknown_evidence_id",
        injected_failure=(
            "two schema-valid outputs use an unknown evidence id and exhaust "
            "one deterministic guard repair"
        ),
        injection_layer="MockInterviewEvaluationAgent.semantic_guard",
        expected_schema_repairs=0,
    )


def _interview_contracts() -> tuple[MockInterviewPlan, AnswerCardDeck]:
    plan = MockInterviewPlan(
        company="示例公司",
        title="Agent 开发实习生",
        persona="技术面试官",
        questions=[
            MockInterviewQuestion(
                question_id="q1",
                requirement_id="req-1",
                prompt="如何保证 checkpoint 可恢复？",
                focus="checkpoint",
            )
        ],
        scoring_dimensions=["技术深度"],
        live_rules=["逐题提问。"],
    )
    cards = AnswerCardDeck(
        company=plan.company,
        title=plan.title,
        cards=[
            AnswerCard(
                requirement_id="req-1",
                question=plan.questions[0].prompt,
                short_answer="保存版本指纹。",
                evidence_level=EvidenceLevel.C2,
                supporting_evidence="恢复测试。",
                boundary="未做生产容灾。",
            )
        ],
    )
    return plan, cards


def _eval_budget(
    *,
    max_model_calls: int,
    max_tool_calls: int,
    max_format_repairs: int = 2,
) -> AgentBudget:
    return AgentBudget.for_profile(
        BudgetProfile.EVAL,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        max_active_seconds=30,
        max_format_repairs=max_format_repairs,
    )


def _interview_agent(
    *,
    responses: list[Any],
    plan: MockInterviewPlan,
    cards: AnswerCardDeck,
    store: JsonCheckpointStore,
    budget: AgentBudget,
    clock: FakeClock,
) -> tuple[InterviewCoachAgent, ScriptedAgentModel]:
    model = ScriptedAgentModel(responses)
    agent = InterviewCoachAgent(
        model=model,
        plan=plan,
        answer_cards=cards,
        budget=budget,
        checkpoint_store=store,
    )
    agent.loop.budget = BudgetManager(budget, clock=clock)
    return agent, model


def _mock_interview_budget_resume(
    *,
    case_dir: Path,
    case_id: str,
    budget_kind: str,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    plan, cards = _interview_contracts()
    plan_path = case_dir / "09_mock_interview_plan.md"
    plan_path.write_text("# Stable mock interview plan\n", encoding="utf-8")
    before = _artifact_hashes([plan_path])
    checkpoint_path = case_dir / "interview-checkpoint.json"
    store = JsonCheckpointStore(checkpoint_path)
    clock = FakeClock()
    request = UserInputRequest(
        request_id="answer-q1",
        prompt=plan.questions[0].prompt,
        response_schema={"type": "object", "required": ["answer"]},
        context_summary="checkpoint reliability question",
    )
    goal = InterviewCoachGoal(
        company=plan.company,
        title=plan.title,
        max_questions=1,
    )
    session_id = "workbench-interview:reliability"
    run_id = f"reliability-{budget_kind}-budget"
    low_budget = (
        _eval_budget(max_model_calls=1, max_tool_calls=5)
        if budget_kind == "model"
        else _eval_budget(max_model_calls=8, max_tool_calls=1)
    )
    first_agent, _ = _interview_agent(
        responses=[
            AgentInputRequest(
                request=request,
                progress_claim="ask one checkpoint question",
            )
        ],
        plan=plan,
        cards=cards,
        store=store,
        budget=low_budget,
        clock=clock,
    )
    clock.advance(3)
    waiting = first_agent.run(
        goal,
        session_id=session_id,
        run_id=run_id,
        workspace_root=case_dir,
    )
    waiting_checkpoint_exists = checkpoint_path.is_file()
    clock.advance(100)
    answer = "我保存 checkpoint，并用版本指纹拒绝不兼容恢复。"
    response = UserInputResponse(
        request_id=request.request_id,
        value={
            "question_id": "q1",
            "requirement_id": "req-1",
            "question_prompt": plan.questions[0].prompt,
            "focus": plan.questions[0].focus,
            "answer": answer,
            "elapsed_seconds": 42,
        },
    )
    if budget_kind == "model":
        second_responses: list[Any] = []
        expected_error = "model_call_budget_exceeded"
    else:
        second_responses = [
            AgentAction(
                action_id="load-plan-after-answer",
                tool_name=LOAD_INTERVIEW_TOOL,
                tool_arguments={"include_answer_cards": True},
                expected_observation="approved interview plan",
                progress_claim="load plan before finishing",
            )
        ]
        expected_error = "tool_call_budget_exceeded"
    second_agent, _ = _interview_agent(
        responses=second_responses,
        plan=plan,
        cards=cards,
        store=store,
        budget=low_budget,
        clock=clock,
    )
    budget_result = second_agent.run(
        goal,
        session_id=session_id,
        run_id=run_id,
        workspace_root=case_dir,
        user_inputs={request.request_id: response},
        resume=True,
    )
    checkpoint_after_budget = store.load()
    checkpoint_preserved_after_budget = checkpoint_path.is_file()
    checkpoint_answer = (
        checkpoint_after_budget.user_inputs[0].response.value.get("answer")
        if checkpoint_after_budget.user_inputs
        else None
    )
    active_seconds_after_wait = checkpoint_after_budget.budget_usage.active_seconds
    after = _artifact_hashes([plan_path])
    raised_budget = (
        _eval_budget(max_model_calls=2, max_tool_calls=5)
        if budget_kind == "model"
        else _eval_budget(max_model_calls=8, max_tool_calls=2)
    )
    finish = AgentFinish(
        result={
            "company": plan.company,
            "title": plan.title,
            "completed_question_ids": ["q1"],
            "transcript_complete": True,
            "final_note": "checkpoint answer preserved",
        },
        completion_evidence=["question:q1"],
        confidence=0.9,
    )
    recovery_agent, recovery_model = _interview_agent(
        responses=[finish],
        plan=plan,
        cards=cards,
        store=store,
        budget=raised_budget,
        clock=clock,
    )
    recovered = recovery_agent.run(
        goal,
        session_id=session_id,
        run_id=run_id,
        workspace_root=case_dir,
        resume=True,
    )
    answer_in_recovery_context = bool(
        recovery_model.calls
        and recovery_model.calls[0].user_inputs
        and recovery_model.calls[0].user_inputs[0].response.value.get("answer")
        == answer
    )
    assertions = {
        "initial_run_waits_for_user": (
            waiting.state.status == AgentRunStatus.WAITING_FOR_USER
        ),
        "waiting_checkpoint_exists": waiting_checkpoint_exists,
        "budget_status_is_terminal_and_bounded": (
            budget_result.state.status == AgentRunStatus.BUDGET_EXCEEDED
        ),
        "budget_error_code_matches": budget_result.error_code == expected_error,
        "checkpoint_exists_after_budget_exceeded": (
            checkpoint_preserved_after_budget
        ),
        "user_answer_is_in_budget_checkpoint": checkpoint_answer == answer,
        "one_hundred_second_wait_excluded": active_seconds_after_wait == 3,
        "same_profile_higher_limit_resumes": (
            recovered.state.status == AgentRunStatus.COMPLETED
        ),
        "answer_reaches_recovery_model_context": answer_in_recovery_context,
        "checkpoint_deleted_only_after_completion": not checkpoint_path.exists(),
        "plan_artifact_sha256_unchanged": before == after,
    }
    return _case(
        case_id=case_id,
        action_id="continue_mock_interview",
        runtime_path="InterviewCoachAgent -> AgentLoop checkpoint/resume",
        injection_layer=f"AgentBudget.{budget_kind}_call_limit",
        evidence_scope="domain_agent_component",
        injected_failure=f"{expected_error} after one approved user answer",
        terminal_status=budget_result.state.status.value,
        error_code=budget_result.error_code,
        before_hashes=before,
        after_hashes=after,
        previous_version_preserved=before == after,
        artifact_freshness_preserved=True,
        checkpoint_preserved=(
            waiting_checkpoint_exists and checkpoint_answer == answer
        ),
        user_answer_preserved=(
            checkpoint_answer == answer and answer_in_recovery_context
        ),
        waiting_time_excluded_from_active_wall_time=(
            active_seconds_after_wait == 3
        ),
        retry_after_recovery_success=(
            recovered.state.status == AgentRunStatus.COMPLETED
        ),
        unhandled_traceback=False,
        diagnostic_secret_leak=None,
        repair_attempt_count=None,
        assertions=assertions,
        budget_usage=checkpoint_after_budget.budget_usage.model_dump(
            mode="json"
        ),
        notes=[
            "The recovery keeps BudgetProfile.EVAL unchanged and raises only "
            f"the {budget_kind}-call limit.",
            "The 100-second synthetic user wait leaves active_seconds at 3.",
        ],
        not_applicable_fields=[
            "diagnostic_secret_leak",
            "repair_attempt_count",
        ],
    )


def _mock_interview_model_budget_resume(case_dir: Path) -> dict[str, Any]:
    return _mock_interview_budget_resume(
        case_dir=case_dir,
        case_id="mock_interview_model_budget_resume",
        budget_kind="model",
    )


def _mock_interview_tool_budget_resume(case_dir: Path) -> dict[str, Any]:
    return _mock_interview_budget_resume(
        case_dir=case_dir,
        case_id="mock_interview_tool_budget_resume",
        budget_kind="tool",
    )


def _format_repair_budget_contract(case_dir: Path) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    clock = FakeClock()
    budget = _eval_budget(
        max_model_calls=9,
        max_tool_calls=9,
        max_format_repairs=2,
    )
    manager = BudgetManager(budget, clock=clock)
    clock.advance(3)
    manager.pause_for_user()
    clock.advance(100)
    waiting_time_excluded = manager.snapshot().active_seconds == 3
    manager.resume_from_user()
    manager.record_format_repair()
    manager.record_format_repair()
    error_code = None
    try:
        manager.record_format_repair()
    except BudgetExceededError as exc:
        error_code = exc.reason_code
    usage = manager.snapshot()
    raised = _eval_budget(
        max_model_calls=9,
        max_tool_calls=9,
        max_format_repairs=3,
    )
    recovery_manager = BudgetManager(raised, clock=clock)
    recovery_manager.restore(usage, paused=True)
    recovery_manager.resume_from_user()
    recovery_succeeded = True
    try:
        recovery_manager.record_format_repair()
    except BudgetExceededError:
        recovery_succeeded = False
    assertions = {
        "format_repair_limit_raises_stable_code": (
            error_code == "format_repair_budget_exceeded"
        ),
        "format_repair_counter_is_two_at_failure": (
            usage.format_repairs == 2
        ),
        "waiting_time_excluded": waiting_time_excluded,
        "same_profile_higher_limit_allows_next_repair": recovery_succeeded,
    }
    return _case(
        case_id="format_repair_budget_manager_contract",
        action_id="budget_contract",
        runtime_path="BudgetManager standalone contract",
        injection_layer="BudgetManager.record_format_repair",
        evidence_scope="unit_contract",
        injected_failure="third format repair with max_format_repairs=2",
        terminal_status=AgentRunStatus.BUDGET_EXCEEDED.value,
        error_code=error_code,
        before_hashes={},
        after_hashes={},
        previous_version_preserved=None,
        artifact_freshness_preserved=None,
        checkpoint_preserved=None,
        user_answer_preserved=None,
        waiting_time_excluded_from_active_wall_time=waiting_time_excluded,
        retry_after_recovery_success=recovery_succeeded,
        unhandled_traceback=False,
        diagnostic_secret_leak=None,
        repair_attempt_count=usage.format_repairs,
        assertions=assertions,
        budget_usage=usage.model_dump(mode="json"),
        notes=[
            "This proves the standalone BudgetManager contract only; the next "
            "case records the action-runtime integration gap."
        ],
        not_applicable_fields=[
            "previous_version_preserved",
            "artifact_freshness_preserved",
            "checkpoint_preserved",
            "user_answer_preserved",
            "diagnostic_secret_leak",
        ],
    )


def _format_repair_action_integration_not_proven(
    case_dir: Path,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    return _case(
        case_id="format_repair_budget_action_integration",
        action_id="evaluate_mock_interview",
        runtime_path="Semantic Agent / Domain Agent action runtime",
        injection_layer="NOT_PROVEN",
        evidence_scope="integration_gap",
        injected_failure=(
            "format repair counter propagated from model/harness into "
            "AgentBudget and checkpoint"
        ),
        terminal_status=None,
        error_code=None,
        before_hashes={},
        after_hashes={},
        previous_version_preserved=None,
        artifact_freshness_preserved=None,
        checkpoint_preserved=None,
        user_answer_preserved=None,
        waiting_time_excluded_from_active_wall_time=None,
        retry_after_recovery_success=None,
        unhandled_traceback=None,
        diagnostic_secret_leak=None,
        repair_attempt_count=None,
        assertions={},
        notes=[
            "The repository exposes BudgetManager.record_format_repair, but "
            "the inspected action runtimes do not currently propagate semantic "
            "schema/format repairs into that counter.",
            "No production code or Guard was changed to manufacture a passing "
            "integration result.",
        ],
        result_override="NOT_PROVEN",
        not_proven_fields=[
            "terminal_status",
            "error_code",
            "previous_version_preserved",
            "artifact_freshness_preserved",
            "checkpoint_preserved",
            "user_answer_preserved",
            "waiting_time_excluded_from_active_wall_time",
            "retry_after_recovery_success",
            "unhandled_traceback",
            "diagnostic_secret_leak",
            "repair_attempt_count",
        ],
    )


CASE_RUNNERS: tuple[tuple[str, Callable[[Path], dict[str, Any]]], ...] = (
    ("resume_provider_timeout", _targeted_resume_provider_timeout),
    (
        "resume_schema_error_destructive_rollback",
        _targeted_resume_schema_destructive_rollback,
    ),
    ("resume_semantic_guard_rejection", _targeted_resume_guard_error),
    (
        "interview_prep_provider_rate_limit",
        _interview_prep_provider_rate_limit,
    ),
    (
        "interview_prep_atomic_batch_commit_error",
        _interview_prep_atomic_commit_error,
    ),
    (
        "mock_evaluation_schema_repair_exhausted",
        _mock_evaluation_schema_repair_exhausted,
    ),
    (
        "mock_evaluation_guard_repair_exhausted",
        _mock_evaluation_guard_repair_exhausted,
    ),
    (
        "mock_interview_model_budget_resume",
        _mock_interview_model_budget_resume,
    ),
    (
        "mock_interview_tool_budget_resume",
        _mock_interview_tool_budget_resume,
    ),
    (
        "format_repair_budget_manager_contract",
        _format_repair_budget_contract,
    ),
    (
        "format_repair_budget_action_integration",
        _format_repair_action_integration_not_proven,
    ),
)


def run_reliability_fault_injection(
    workspace: Path | str,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    cases = [
        runner(workspace_path / case_id)
        for case_id, runner in CASE_RUNNERS
    ]
    status_counts = {
        status: sum(case["result"] == status for case in cases)
        for status in ("PASS", "FAIL", "NOT_PROVEN", "NOT_APPLICABLE")
    }
    applicable_assertions = sum(
        len(case["assertions"])
        for case in cases
        if case["result"] != "NOT_PROVEN"
    )
    passed_assertions = sum(
        sum(bool(value) for value in case["assertions"].values())
        for case in cases
        if case["result"] != "NOT_PROVEN"
    )
    applicable_cases = len(cases) - status_counts["NOT_PROVEN"] - status_counts[
        "NOT_APPLICABLE"
    ]
    return {
        "schema_version": 1,
        "evidence_pack": "Reliability Fault Injection Evidence Pack",
        "evidence_date": EVIDENCE_DATE,
        "execution_mode": "offline_fixture_only",
        "real_api_executed": False,
        "credential_files_read": False,
        "guard_relaxed": False,
        "case_count": len(cases),
        "applicable_case_count": applicable_cases,
        "case_status_counts": status_counts,
        "applicable_assertion_count": applicable_assertions,
        "passed_assertion_count": passed_assertions,
        "case_proof_rate": (
            round(status_counts["PASS"] / applicable_cases, 4)
            if applicable_cases
            else None
        ),
        "assertion_proof_rate": (
            round(passed_assertions / applicable_assertions, 4)
            if applicable_assertions
            else None
        ),
        "metric_boundary": (
            "Proof rates describe deterministic contract cases and assertions; "
            "they are not production incident probabilities, SLA, or LLM "
            "semantic-quality metrics."
        ),
        "cases": cases,
    }


def run_in_temporary_directory() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="job-agent-reliability-"
    ) as directory:
        return run_reliability_fault_injection(Path(directory))
