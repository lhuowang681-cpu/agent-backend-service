from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from job_agent.evidence.evaluation.contracts import (
    EvidenceEvalManifest,
    EvidenceEvalObservation,
    EvidenceEvalReport,
    LiveEvaluationApproval,
)
from job_agent.evidence.evaluation.metrics import evaluate_observations


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[4] / "data" / "eval" / "evidence_v2" / "manifest_dev_v4.json"


def load_manifest(path: Path) -> EvidenceEvalManifest:
    return EvidenceEvalManifest.model_validate_json(path.read_text(encoding="utf-8"))


def require_live_approval(approval: LiveEvaluationApproval | None) -> None:
    if approval is None:
        raise RuntimeError("live evaluation requires explicit approval and budget")
    if not os.environ.get("JOB_AGENT_LIVE_API_KEY"):
        raise RuntimeError("live evaluation requires JOB_AGENT_LIVE_API_KEY")


def deterministic_observations() -> list[EvidenceEvalObservation]:
    """Synthetic rows exercise evaluator math; they are not model-quality results."""
    return [
        EvidenceEvalObservation(
            case_id="compound-a",
            paraphrase_group="compound",
            expected_requirement_ids=["train", "evaluate"],
            predicted_requirement_ids=["train", "evaluate"],
            expected_link_ids=["evaluate:resume"],
            predicted_link_ids=["evaluate:resume"],
            expected_partial={"evaluate": False},
            predicted_partial={"evaluate": False},
            expected_span_ids=["resume:0:12"],
            predicted_span_ids=["resume:0:12"],
            predicted_fit_band="medium",
            resume_claim_supported=[True],
        ),
        EvidenceEvalObservation(
            case_id="compound-b",
            paraphrase_group="compound",
            expected_requirement_ids=["train", "evaluate"],
            predicted_requirement_ids=["train", "evaluate"],
            expected_link_ids=["evaluate:resume"],
            predicted_link_ids=["evaluate:resume", "train:unsupported"],
            expected_contradiction_ids=["ownership"],
            predicted_contradiction_ids=[],
            expected_partial={"evaluate": True},
            predicted_partial={"evaluate": False},
            expected_span_ids=["resume:0:12"],
            predicted_span_ids=["resume:0:11"],
            predicted_fit_band="low",
            resume_claim_supported=[True, False],
        ),
    ]


def build_deterministic_report(manifest: EvidenceEvalManifest) -> EvidenceEvalReport:
    observations = deterministic_observations()
    return EvidenceEvalReport(
        manifest_id=manifest.manifest_id,
        metric_version=manifest.metric_version,
        case_count=len(observations),
        metrics=evaluate_observations(observations),
    )


def _atomic_report(path: Path, report: EvidenceEvalReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(report.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence v2 deterministic evaluator")
    parser.add_argument("--suite", choices=["deterministic"], required=True)
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = build_deterministic_report(manifest)
    _atomic_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
