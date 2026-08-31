from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from job_agent.backend_service.application_evaluation import (
    load_application_assistant_eval_cases,
    run_application_assistant_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "eval" / "application_assistant_phase_c_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic Application Assistant Agent evaluation."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        ROOT
        / "output"
        / "backend_service"
        / "application_evaluation"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    cases = load_application_assistant_eval_cases(args.cases)
    report = run_application_assistant_evaluation(cases, output_root=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": report.case_count,
                "passed_case_count": report.passed_case_count,
                "task_success_rate": report.task_success_rate,
                "uncertain_classification_accuracy": (
                    report.uncertain_classification_accuracy
                ),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report.passed_case_count == report.case_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
