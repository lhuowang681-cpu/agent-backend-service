from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from pydantic import ValidationError

from job_agent.graph import run_fixture_flow, run_funnel_only, run_jobs_flow, run_manual_jd_flow
from job_agent.nodes.fit_verdict import LocalLoraFitVerdictBackend, OpenAICompatibleFitVerdictBackend
from job_agent.nodes.job_scout import scout_jobs
from job_agent.nodes.search_intent import build_search_intent
from job_agent.outputs import (
    write_agent_runtime_failure_audit,
    write_funnel_report,
    write_session_outputs,
)
from job_agent.reports import render_funnel_report
from job_agent.schemas import FitBackendKind, JobSourceKind, RuntimeConfig, RuntimeProfileKind
from job_agent.session_orchestrator import plan_next_action, run_next_action, write_session_state
from job_agent.agents.base import AgentGuardError
from job_agent.llm.harness import LLMHarness, LLMInvocationError, NodePolicy
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.providers.openai_compatible import OpenAICompatibleProvider
from job_agent.llm.providers.transformers import LocalTransformersProvider
from job_agent.llm.skill_registry import SkillRegistry, SkillRegistryError
from job_agent.runtime.graph_runtime import AgentGraphRuntime
from job_agent.runtime.modes import AgentRuntimeConfig, RuntimeMode
from job_agent.runtime.semantic_checkpoint import SemanticCheckpointStore
from job_agent.agent_runtime.failures import CheckpointError


RUNTIME_CONFIG_KEYS = {
    "selected_job_id",
    "job_fixture",
    "raw_jobs",
    "resume_path",
    "request",
    "runtime_profile",
    "fit_backend",
    "server_base_url",
    "server_model",
    "server_api_key_env",
    "local_model_path",
    "local_adapter_path",
    "manual_jd",
    "company",
    "title",
    "location",
    "output_dir",
    "no_write_output",
    "list_jobs",
    "mode",
    "skill_root",
    "agent_provider",
    "agent_model",
    "semantic_checkpoint_path",
    "resume_semantic",
}


def _runtime_config_path_from_argv(argv: list[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-config", default=None)
    args, _ = parser.parse_known_args(argv)
    return args.runtime_config


def _load_runtime_config_defaults(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime config must be a JSON object")
    unknown = sorted(set(data) - RUNTIME_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unsupported runtime config keys: {', '.join(unknown)}")
    return data


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    if argv and argv[0] == "career-journey-live":
        from job_agent.career_journey_cli import main as career_journey_main

        raise SystemExit(career_journey_main(argv[1:]))
    if argv and argv[0] == "private-capture-maintenance":
        from job_agent.private_capture_cli import main as private_capture_main

        raise SystemExit(private_capture_main(argv[1:]))
    runtime_config_path = _runtime_config_path_from_argv(argv)
    config_defaults = {}
    if runtime_config_path:
        try:
            config_defaults = _load_runtime_config_defaults(Path(runtime_config_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser = argparse.ArgumentParser(description="Run the job agent MVP fixture flow.")
            parser.error(str(exc))
    parser = argparse.ArgumentParser(description="Run the job agent MVP fixture flow.")
    parser.add_argument("--runtime-config", default=runtime_config_path)
    parser.add_argument("--selected-job-id", default=config_defaults.get("selected_job_id"))
    parser.add_argument("--job-fixture", default=config_defaults.get("job_fixture", "data/fixtures/raw_jobs_posttraining.json"))
    parser.add_argument("--raw-jobs", default=config_defaults.get("raw_jobs"))
    parser.add_argument("--resume-path", default=config_defaults.get("resume_path", "data/fixtures/resume_seed.md"))
    parser.add_argument("--request", default=config_defaults.get("request", "帮我找后训练 RLHF 实习"))
    parser.add_argument(
        "--runtime-profile",
        choices=[item.value for item in RuntimeProfileKind],
        default=config_defaults.get("runtime_profile", RuntimeProfileKind.LOCAL_DEV.value),
    )
    parser.add_argument("--fit-backend", choices=[item.value for item in FitBackendKind], default=config_defaults.get("fit_backend", FitBackendKind.RULE.value))
    parser.add_argument("--server-base-url", default=config_defaults.get("server_base_url"))
    parser.add_argument("--server-model", default=config_defaults.get("server_model"))
    parser.add_argument("--server-api-key-env", default=config_defaults.get("server_api_key_env"))
    parser.add_argument("--local-model-path", default=config_defaults.get("local_model_path"))
    parser.add_argument("--local-adapter-path", default=config_defaults.get("local_adapter_path"))
    parser.add_argument("--manual-jd", default=config_defaults.get("manual_jd"))
    parser.add_argument("--company", default=config_defaults.get("company", "Manual"))
    parser.add_argument("--title", default=config_defaults.get("title", "Manual JD"))
    parser.add_argument("--location", default=config_defaults.get("location", "unknown"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir", "output"))
    parser.add_argument("--no-write-output", action="store_true", default=bool(config_defaults.get("no_write_output", False)))
    parser.add_argument("--list-jobs", action="store_true", default=bool(config_defaults.get("list_jobs", False)))
    parser.add_argument(
        "--mode",
        choices=[item.value for item in RuntimeMode],
        default=config_defaults.get("mode", RuntimeMode.OFFLINE_RULE.value),
    )
    parser.add_argument(
        "--skill-root",
        default=os.environ.get("LLM_INTERN_SKILL_ROOT") or config_defaults.get("skill_root"),
    )
    parser.add_argument(
        "--agent-provider",
        choices=["mock", "openai_compatible", "anthropic_compatible", "transformers"],
        default=config_defaults.get("agent_provider"),
    )
    parser.add_argument("--agent-model", default=config_defaults.get("agent_model", "fixture-model"))
    parser.add_argument(
        "--semantic-checkpoint-path",
        default=config_defaults.get("semantic_checkpoint_path"),
        help="Repository-external full-private semantic checkpoint with a fixed 30-day TTL.",
    )
    parser.add_argument(
        "--resume-semantic",
        action="store_true",
        default=bool(config_defaults.get("resume_semantic", False)),
    )
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--continue-request", default=None)
    parser.add_argument("--plan-next-action", default=None)
    parser.add_argument("--run-next-action", default=None)
    args = parser.parse_args(argv)
    try:
        config = RuntimeConfig(
            runtime_profile=RuntimeProfileKind(args.runtime_profile),
            fit_backend=FitBackendKind(args.fit_backend),
            server_base_url=args.server_base_url,
            server_model=args.server_model,
            server_api_key_env=args.server_api_key_env,
            local_model_path=args.local_model_path,
            local_adapter_path=args.local_adapter_path,
        )
    except ValidationError as exc:
        message = str(exc)
        if "local_lora requires runtime_profile=server_agent" in message:
            parser.error("local_lora requires --runtime-profile server_agent")
        parser.error(message)
    try:
        agent_config = AgentRuntimeConfig(
            mode=RuntimeMode(args.mode),
            skill_root=args.skill_root,
            provider=args.agent_provider,
            model=args.agent_model,
        )
    except ValidationError as exc:
        parser.error(str(exc))

    if args.session_dir or args.continue_request or args.plan_next_action or args.run_next_action:
        if not args.session_dir:
            parser.error("--session-dir is required for session route planning or execution.")
        session_request_args = [args.continue_request, args.plan_next_action, args.run_next_action]
        if sum(value is not None for value in session_request_args) > 1:
            parser.error("--continue-request, --plan-next-action, and --run-next-action cannot be used together.")
        request_text = args.continue_request or args.plan_next_action or args.run_next_action
        if not request_text:
            parser.error("--session-dir must be used with --continue-request, --plan-next-action, or --run-next-action.")
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            parser.error(f"session_dir not found: {session_dir}")
        if args.plan_next_action:
            decision = plan_next_action(session_dir, request_text)
            print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return
        if args.run_next_action:
            execution = run_next_action(session_dir, request_text)
            print(f"Session Intent: {execution.decision.intent.value}")
            print(f"Runnable: {execution.decision.runnable}")
            print(f"Next Step: {execution.decision.next_step}")
            print(f"Executed: {execution.executed}")
            if execution.written_artifacts:
                print(f"Written Artifacts: {', '.join(execution.written_artifacts)}")
            if execution.decision.missing_artifacts:
                print(f"Missing Artifacts: {', '.join(execution.decision.missing_artifacts)}")
            print(f"Message: {execution.message}")
            return
        write_session_state(session_dir)
        decision = plan_next_action(session_dir, request_text)
        print(f"Session Intent: {decision.intent.value}")
        print(f"Runnable: {decision.runnable}")
        print(f"Next Step: {decision.next_step}")
        if decision.required_artifacts:
            print(f"Required Artifacts: {', '.join(decision.required_artifacts)}")
        if decision.missing_artifacts:
            print(f"Missing Artifacts: {', '.join(decision.missing_artifacts)}")
        print(f"Message: {decision.message}")
        return

    fit_backend = None
    if config.fit_backend == FitBackendKind.API:
        try:
            fit_backend = OpenAICompatibleFitVerdictBackend.from_config(config)
        except ValueError as exc:
            parser.error(str(exc))
    elif config.fit_backend == FitBackendKind.LOCAL_LORA:
        try:
            fit_backend = LocalLoraFitVerdictBackend.from_config(config)
        except ValueError as exc:
            parser.error(str(exc))

    print(f"Runtime Profile: {config.runtime_profile.value}")
    print(f"Runtime Mode: {agent_config.mode.value}")

    if args.manual_jd:
        if args.list_jobs:
            parser.error("--list-jobs is only available for fixture/raw job search flows.")
        result = run_manual_jd_flow(
            jd_text=args.manual_jd,
            resume_path=Path(args.resume_path),
            company=args.company,
            title=args.title,
            location=args.location,
            fit_backend=fit_backend,
        )
    elif args.raw_jobs:
        intent = build_search_intent(args.request)
        job_scout = scout_jobs(
            intent=intent,
            source_kind=JobSourceKind.RAW_JOBS,
            source_path=Path(args.raw_jobs),
        )
        if args.list_jobs:
            result = run_funnel_only(
                user_request=args.request,
                jobs=job_scout.jobs,
                search_intent=job_scout.intent,
                job_scout=job_scout,
            )
        else:
            result = run_jobs_flow(
                user_request=args.request,
                jobs=job_scout.jobs,
                resume_path=Path(args.resume_path),
                selected_job_id=args.selected_job_id,
                fit_backend=fit_backend,
                search_intent=job_scout.intent,
                job_scout=job_scout,
            )
        print(render_funnel_report(result["leads"]))
    else:
        if args.list_jobs:
            intent = build_search_intent(args.request)
            job_scout = scout_jobs(
                intent=intent,
                source_kind=JobSourceKind.FIXTURE,
                source_path=Path(args.job_fixture),
            )
            result = run_funnel_only(
                user_request=args.request,
                jobs=job_scout.jobs,
                search_intent=job_scout.intent,
                job_scout=job_scout,
            )
        else:
            result = run_fixture_flow(
                user_request=args.request,
                job_fixture=Path(args.job_fixture),
                resume_path=Path(args.resume_path),
                selected_job_id=args.selected_job_id or "job_001",
                fit_backend=fit_backend,
            )
        print(render_funnel_report(result["leads"]))

    if args.list_jobs:
        if not args.no_write_output:
            path = write_funnel_report(result, Path(args.output_dir))
            print(f"Funnel Report: {path}")
        print("Select a job with: --selected-job-id <job_id>")
        return

    if agent_config.uses_semantic_agents:
        try:
            registry = SkillRegistry.from_sources(skill_root=agent_config.skill_root)
        except SkillRegistryError as exc:
            parser.error(str(exc))
        if agent_config.provider == "mock":
            provider = MockLLMProvider(
                [
                    result["structured_jd"].model_dump(mode="json"),
                    {
                        "items": [
                            item.model_dump(mode="json")
                            for item in result["fit_input"].evidence
                        ]
                    },
                    result["resume_tailoring"].model_dump(mode="json"),
                    result["interview_prep"].model_dump(mode="json"),
                    result["answer_cards"].model_dump(mode="json"),
                ],
                model=agent_config.model,
            )
        elif agent_config.provider in {"openai_compatible", "anthropic_compatible"}:
            if not args.server_base_url:
                parser.error("--server-base-url is required for API agent providers")
            api_key = None
            if args.server_api_key_env:
                api_key = os.environ.get(args.server_api_key_env)
                if not api_key:
                    parser.error("configured API key environment variable is missing or empty")
            if agent_config.provider == "anthropic_compatible":
                if not api_key:
                    parser.error("--server-api-key-env is required for anthropic_compatible provider")
                provider = AnthropicCompatibleProvider(
                    base_url=args.server_base_url,
                    model=agent_config.model,
                    api_key=api_key,
                    timeout_s=agent_config.timeout_s,
                )
            else:
                provider = OpenAICompatibleProvider(
                    base_url=args.server_base_url,
                    model=agent_config.model,
                    api_key=api_key,
                    timeout_s=agent_config.timeout_s,
                )
        else:
            if not args.local_model_path:
                parser.error("--local-model-path is required for transformers agent provider")
            from job_agent.nodes.fit_verdict import LocalCausalLMVerdictRunner

            provider = LocalTransformersProvider(
                runner=LocalCausalLMVerdictRunner(
                    model_path=args.local_model_path,
                    adapter_path=args.local_adapter_path,
                    max_new_tokens=agent_config.max_output_tokens,
                ),
                model=agent_config.model,
            )
        harness = LLMHarness(provider)
        if args.resume_semantic and not args.semantic_checkpoint_path:
            parser.error("--resume-semantic requires --semantic-checkpoint-path")
        checkpoint_store = None
        if args.semantic_checkpoint_path:
            try:
                checkpoint_store = SemanticCheckpointStore(
                    args.semantic_checkpoint_path,
                    workspace_root=Path.cwd(),
                )
            except (CheckpointError, ValueError) as exc:
                parser.error(str(exc))
        graph_runtime = AgentGraphRuntime(
            registry=registry,
            harness=harness,
            policy=NodePolicy(
                timeout_s=agent_config.timeout_s,
                max_retries=agent_config.max_retries,
                temperature=agent_config.temperature,
                max_output_tokens=agent_config.max_output_tokens,
                allow_rule_fallback=False,
            ),
            runtime_mode=agent_config.mode,
            checkpoint_store=checkpoint_store,
        )
        try:
            agent_result = graph_runtime.run_selected_job(
                selected_job=result["selected_job"],
                resume_text=Path(args.resume_path).read_text(encoding="utf-8"),
                session_id=f"{result['selected_job'].company}:{result['selected_job'].job_id}",
                run_id="cli-run",
                resume=args.resume_semantic,
            )
        except (LLMInvocationError, AgentGuardError, CheckpointError) as exc:
            audit_path = None
            if not args.no_write_output:
                audit_path = write_agent_runtime_failure_audit(
                    Path(args.output_dir),
                    runtime_mode=agent_config.mode.value,
                    harness=harness,
                    error=exc,
                )
            print(f"Agent Runtime Failed: {exc}", file=sys.stderr)
            if audit_path is not None:
                print(f"Failure Audit: {audit_path}", file=sys.stderr)
            raise SystemExit(1) from None
        for key in ("user_request", "search_intent", "job_scout", "jobs", "leads"):
            if key in result:
                agent_result[key] = result[key]
        result = agent_result
        print(f"Graph Engine: {graph_runtime.engine_name}")
        print(f"LLM Call Count: {result['llm_call_count']}")
        print(f"Agent Provider: {agent_config.provider}")
        print(f"Agent Model: {agent_config.model}")
        print(
            "Skill Versions: "
            + ", ".join(
                f"{trace.skill_id}={trace.skill_version}" for trace in result["provider_traces"]
            )
        )

    fit_result = result["fit_result"]
    action = result["action"]
    verdict_route = result["verdict_route"]
    print(f"Selected Job: {result['selected_job'].company} - {result['selected_job'].title}")
    print(f"Fit Verdict: {fit_result.verdict.value} ({fit_result.score:.0%})")
    print(f"Reason Codes: {', '.join(fit_result.reason_codes)}")
    print(f"Verdict Route: {verdict_route.route} (gate={verdict_route.gate})")
    print(f"Route Next Actions: {', '.join(verdict_route.next_actions)}")
    print(f"Next Action: {action.next_action}")
    print(f"Summary: {action.summary}")
    if not args.no_write_output:
        paths = write_session_outputs(result, Path(args.output_dir))
        print(f"Output Session: {paths.session_dir}")
        print(f"Funnel Report: {paths.funnel_report}")


if __name__ == "__main__":
    main()
