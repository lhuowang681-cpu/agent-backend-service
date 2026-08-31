import json

import pytest

from job_agent.evidence.evaluation.contracts import EvidenceEvalObservation
from job_agent.evidence.evaluation.metrics import evaluate_observations
from job_agent.evidence.evaluation.runner import main


def _rows():
    return [
        EvidenceEvalObservation(
            case_id="p1",
            paraphrase_group="p",
            expected_requirement_ids=["a", "b"],
            predicted_requirement_ids=["a", "b"],
            expected_link_ids=["l1"],
            predicted_link_ids=["l1"],
            expected_partial={"a": True},
            predicted_partial={"a": True},
            expected_span_ids=["s1"],
            predicted_span_ids=["s1"],
            predicted_fit_band="medium",
            resume_claim_supported=[True],
        ),
        EvidenceEvalObservation(
            case_id="p2",
            paraphrase_group="p",
            expected_requirement_ids=["a", "b"],
            predicted_requirement_ids=["a", "extra"],
            expected_link_ids=["l1"],
            predicted_link_ids=["wrong"],
            expected_contradiction_ids=["c1"],
            predicted_contradiction_ids=[],
            expected_partial={"a": False},
            predicted_partial={"a": True},
            expected_span_ids=["s1"],
            predicted_span_ids=["wrong"],
            predicted_fit_band="low",
            resume_claim_supported=[False],
        ),
    ]


def test_metrics_quantify_errors_and_stability():
    metrics = evaluate_observations(_rows())
    assert metrics.requirement_extraction.true_positive == 3
    assert metrics.requirement_extraction.false_positive == 1
    assert metrics.requirement_extraction.false_negative == 1
    assert metrics.partial_accuracy == 0.5
    assert metrics.span_exactness == 0.5
    assert metrics.verdict_flip_rate == 1.0
    assert metrics.paraphrase_stability == pytest.approx(1 / 3, abs=1e-6)
    assert metrics.unsupported_resume_claim_rate == 0.5


def test_metrics_reject_duplicate_case_ids():
    rows = _rows()
    rows[1] = rows[1].model_copy(update={"case_id": "p1"})
    with pytest.raises(ValueError, match="case ids"):
        evaluate_observations(rows)


def test_deterministic_runner_writes_non_quality_report(tmp_path):
    report_path = tmp_path / "report.json"
    assert main(["--suite", "deterministic", "--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["synthetic_only"] is True
    assert payload["quality_claim_allowed"] is False
    assert payload["case_count"] == 2
