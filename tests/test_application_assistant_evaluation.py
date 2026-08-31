from __future__ import annotations

import json
from pathlib import Path

from job_agent.backend_service.application_evaluation import (
    load_application_assistant_eval_cases,
    run_application_assistant_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "eval" / "application_assistant_phase_c_cases.json"


def test_application_assistant_eval_corpus_has_required_coverage() -> None:
    cases = load_application_assistant_eval_cases(CASES)

    assert len(cases) == 24
    assert sum(case.expected_outcome == "SUCCEEDED" for case in cases) == 20
    assert sum(case.expected_outcome == "UNCERTAIN" for case in cases) == 4
    assert {case.draft_channel for case in cases} == {"email", "form"}
    assert sum(bool(case.forbidden_markers) for case in cases) == 5


def test_application_assistant_eval_runs_real_loop_approval_and_recovery(
    tmp_path,
) -> None:
    cases = load_application_assistant_eval_cases(CASES)
    report = run_application_assistant_evaluation(cases, output_root=tmp_path)

    assert report.case_count == 24
    assert report.passed_case_count == 24
    assert report.task_success_rate == 1.0
    assert report.tool_selection_precision == 1.0
    assert report.tool_selection_recall == 1.0
    assert report.tool_argument_schema_valid_rate == 1.0
    assert report.approval_required_recall == 1.0
    assert report.preapproval_side_effect_count == 0
    assert report.grounding_pass_rate == 1.0
    assert report.checkpoint_resume_success_rate == 1.0
    assert report.completed_node_replay_count == 0
    assert report.duplicate_side_effect_rate == 0.0
    assert report.uncertain_classification_accuracy == 1.0
    assert report.total_model_calls > 0
    assert report.total_tool_calls > 0
    assert report.latency_p50_ms <= report.latency_p95_ms <= report.latency_p99_ms
    assert "resume_text" not in json.dumps(report.model_dump(mode="json"))

