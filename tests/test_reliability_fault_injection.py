from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_agent.evaluation.reliability_fault_injection import (
    CASE_RUNNERS,
    run_reliability_fault_injection,
)


@pytest.fixture(scope="module")
def reliability_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return run_reliability_fault_injection(
        tmp_path_factory.mktemp("reliability-fault-injection")
    )


def test_fault_matrix_has_no_failed_case(reliability_result: dict) -> None:
    assert reliability_result["case_status_counts"]["FAIL"] == 0
    assert reliability_result["case_status_counts"] == {
        "PASS": 10,
        "FAIL": 0,
        "NOT_PROVEN": 1,
        "NOT_APPLICABLE": 0,
    }
    assert reliability_result["case_proof_rate"] == 1.0
    assert reliability_result["assertion_proof_rate"] == 1.0


def test_fault_matrix_covers_required_actions_and_failures(
    reliability_result: dict,
) -> None:
    cases = reliability_result["cases"]
    action_ids = {case["action_id"] for case in cases}
    failures = " ".join(case["injected_failure"] for case in cases)

    assert {
        "regenerate_resume",
        "regenerate_interview_prep",
        "continue_mock_interview",
        "evaluate_mock_interview",
    }.issubset(action_ids)
    for required in (
        "Provider",
        "schema",
        "guard",
        "model_call_budget_exceeded",
        "tool_call_budget_exceeded",
        "format repair",
        "commit",
    ):
        assert required.casefold() in failures.casefold()


def test_every_case_has_auditable_required_fields(
    reliability_result: dict,
) -> None:
    required_fields = {
        "action_id",
        "injected_failure",
        "terminal_status",
        "error_code",
        "before_artifact_sha256",
        "after_artifact_sha256",
        "previous_version_preserved",
        "checkpoint_preserved",
        "retry_after_recovery_success",
        "unhandled_traceback",
        "diagnostic_secret_leak",
        "result",
    }
    for case in reliability_result["cases"]:
        assert required_fields.issubset(case)
        assert case["result"] in {
            "PASS",
            "FAIL",
            "NOT_PROVEN",
            "NOT_APPLICABLE",
        }
        if case["result"] == "NOT_PROVEN":
            assert case["not_proven_fields"]


def test_budget_resume_cases_preserve_answer_checkpoint_and_wait_time(
    reliability_result: dict,
) -> None:
    budget_cases = [
        case
        for case in reliability_result["cases"]
        if case["case_id"]
        in {
            "mock_interview_model_budget_resume",
            "mock_interview_tool_budget_resume",
        }
    ]
    assert len(budget_cases) == 2
    for case in budget_cases:
        assert case["terminal_status"] == "BUDGET_EXCEEDED"
        assert case["checkpoint_preserved"] is True
        assert case["user_answer_preserved"] is True
        assert case["waiting_time_excluded_from_active_wall_time"] is True
        assert case["retry_after_recovery_success"] is True
        assert case["budget_usage"]["active_seconds"] == 3


def test_schema_destructive_case_restores_exact_sha256(
    reliability_result: dict,
) -> None:
    case = next(
        item
        for item in reliability_result["cases"]
        if item["case_id"] == "resume_schema_error_destructive_rollback"
    )
    assert case["before_artifact_sha256"] == case["after_artifact_sha256"]
    assert case["previous_version_preserved"] is True
    assert case["retry_after_recovery_success"] is True


def test_json_result_is_serializable_and_does_not_claim_live_api(
    reliability_result: dict,
) -> None:
    rendered = json.dumps(reliability_result, ensure_ascii=False)
    assert json.loads(rendered)["schema_version"] == 1
    assert reliability_result["real_api_executed"] is False
    assert reliability_result["credential_files_read"] is False
    assert reliability_result["guard_relaxed"] is False


@pytest.mark.parametrize(
    "case_id",
    [case_id for case_id, _ in CASE_RUNNERS],
)
def test_case_ids_are_unique(case_id: str) -> None:
    all_ids = [item[0] for item in CASE_RUNNERS]
    assert all_ids.count(case_id) == 1
