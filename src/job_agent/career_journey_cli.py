from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from job_agent.agent_runtime.live_canary import LiveCanaryConfig, LiveCanaryError, LiveCanaryRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-agent career-journey-live",
        description=(
            "Run the bounded Job Agent Harness career journey with real provider calls and "
            "disposable sandbox side effects only."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--base-url", default="https://open.bigmodel.cn/api/anthropic")
    parser.add_argument("--cases", default="data/eval/job_agent_harness_v1_cases.json")
    parser.add_argument(
        "--source", default="data/fixtures/raw_jobs_llm_intern_skill_scout_sample.json"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--report", required=True)
    parser.add_argument("--scope", choices=("opportunity", "full"), default="full")
    parser.add_argument(
        "--agent-protocol",
        choices=("legacy_structured", "tool_use_v2"),
        default="tool_use_v2",
    )
    parser.add_argument(
        "--semantic-artifacts", choices=("fixture", "api"), default="fixture"
    )
    parser.add_argument("--skill-root", default=None)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--allow-sandbox-side-effects", action="store_true")
    return parser


def _print_report(report_path: Path, report) -> None:
    print(f"Sanitized report: {report_path.resolve()}")
    print(f"Passed: {report.passed}; attempts: {len(report.attempts)}")
    print(f"Agent protocol: {report.agent_protocol}")
    print(f"Semantic artifacts: {report.semantic_artifacts}")
    for attempt in report.attempts:
        print(
            f"Case {attempt.case_id}: {attempt.status}; "
            f"stages={','.join(attempt.stage_coverage)}; "
            f"error={attempt.error_code or '-'}; detail={attempt.error_detail or '-'}"
        )
        for stage in attempt.stages:
            print(
                f"  Stage {stage.stage}: {stage.status}; "
                f"models={stage.model_calls}; providers={stage.provider_calls}; "
                f"tools={stage.tool_calls}; resumes={stage.resumes}; "
                f"approvals={stage.approvals}; error={stage.error_code or '-'}; "
                f"detail={stage.error_detail or '-'}; outcome={stage.outcome or '-'}"
            )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: LiveCanaryRunner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    config = LiveCanaryConfig(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        cases_path=args.cases,
        source_path=args.source,
        timeout_s=args.timeout,
        allow_network=args.allow_network,
        scope=args.scope,
        allow_sandbox_side_effects=args.allow_sandbox_side_effects,
        agent_protocol=args.agent_protocol,
        semantic_artifacts=args.semantic_artifacts,
        skill_root=args.skill_root,
    )
    active_runner = runner or LiveCanaryRunner()
    try:
        report = active_runner.run(config)
    except LiveCanaryError as exc:
        print(f"Live career journey did not start: {exc}")
        return 2
    report_path = Path(args.report)
    active_runner.write_report(report_path, report)
    _print_report(report_path, report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
