from __future__ import annotations

from pathlib import Path

import pytest


def _fixture_contracts(tmp_path: Path, *, supported: bool = True):
    from job_agent.nodes.evidence_mapping import map_evidence
    from job_agent.nodes.answer_cards import build_answer_cards
    from job_agent.nodes.fit_verdict import evaluate_fit
    from job_agent.nodes.interview_prep import prepare_interview
    from job_agent.nodes.jd_structurer import structure_jd
    from job_agent.nodes.resume_tailoring import tailor_resume
    from job_agent.schemas import FitInput, RawJob

    job = RawJob(
        job_id="job-001",
        company="Example",
        title="LLM Post-training Intern",
        desc="SFT LoRA RLHF evaluation internship",
        url="https://example.invalid/job",
        location="Beijing",
    )
    structured = structure_jd(job)
    resume_path = tmp_path / "resume.md"
    resume_text = "LoRA loss checkpoint evaluation exposure" if supported else "generic coursework"
    resume_path.write_text(resume_text, encoding="utf-8")
    evidence = map_evidence(resume_path, structured.must_have)
    fit_input = FitInput(
        role_type=structured.role_type,
        requirements=structured.must_have,
        evidence=evidence,
        toy_signals=[],
    )
    fit_result = evaluate_fit(fit_input)
    targeted = tailor_resume(structured, evidence)
    interview = prepare_interview(structured, evidence, fit_result, targeted)
    answer_cards = build_answer_cards(interview, evidence, targeted)
    responses = [
        structured.model_dump(mode="json"),
        {"items": [item.model_dump(mode="json") for item in evidence]},
    ]
    if fit_result.verdict.value != "not recommended":
        responses.extend(
            [
                targeted.model_dump(mode="json"),
                interview.model_dump(mode="json"),
                answer_cards.model_dump(mode="json"),
            ]
        )
    return job, resume_text, responses, fit_result


class _Registry:
    def __init__(self, version="fixture-v1"):
        self.version = version

    def get(self, skill_id):
        from job_agent.llm.skill_registry import SkillSpec
        from job_agent.schemas import (
            AnswerCardDeck,
            EvidenceMappingResult,
            InterviewPrep,
            StructuredJD,
            TargetedResume,
        )

        schemas = {
            "jd-analysis": StructuredJD,
            "evidence-contract": EvidenceMappingResult,
            "resume-tailoring": TargetedResume,
            "interview-grilling": InterviewPrep,
            "answer-cards": AnswerCardDeck,
        }
        return SkillSpec(
            skill_id=skill_id,
            version=self.version,
            instructions=f"Trusted {skill_id} instructions.",
            reference_paths=(Path(f"skill-references/{skill_id}.md"),),
            output_schema=schemas[skill_id],
            guardrails=("Never fabricate evidence.",),
        )


def test_runtime_modes_default_offline_and_require_agent_configuration():
    from pydantic import ValidationError

    from job_agent.runtime.modes import AgentRuntimeConfig, RuntimeMode

    assert AgentRuntimeConfig().mode == RuntimeMode.OFFLINE_RULE
    with pytest.raises(ValidationError, match="skill_root is required"):
        AgentRuntimeConfig(mode=RuntimeMode.AGENT_API, provider="openai_compatible")
    configured = AgentRuntimeConfig(
        mode=RuntimeMode.AGENT_API,
        model="fixture-model",
        skill_root="fixtures/skill",
    )
    assert configured.uses_semantic_agents is True
    assert configured.provider == "openai_compatible"
    anthropic = AgentRuntimeConfig(
        mode=RuntimeMode.AGENT_API,
        provider="anthropic_compatible",
        model="glm-4.7",
        skill_root="fixtures/skill",
    )
    assert anthropic.provider == "anthropic_compatible"


def test_agent_graph_runtime_executes_five_llm_nodes_and_writes_no_artifacts(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, responses, expected_fit = _fixture_contracts(tmp_path)
    provider = MockLLMProvider(responses)
    runtime = AgentGraphRuntime(registry=_Registry(), harness=LLMHarness(provider))
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    state = runtime.run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-001",
        run_id="run-001",
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert runtime.engine_name in {"langgraph", "equivalent_state_graph"}
    assert runtime.topology[:4] == (
        "jd_structurer",
        "evidence_mapping",
        "fit_verdict",
        "verdict_route",
    )
    assert state["fit_result"].verdict == expected_fit.verdict
    assert state["runtime_mode"] == "agent_api"
    assert state["llm_call_count"] == 5
    assert len(state["provider_traces"]) == 5
    assert [trace.skill_id for trace in state["provider_traces"]] == [
        "jd-analysis",
        "evidence-contract",
        "resume-tailoring",
        "interview-grilling",
        "answer-cards",
    ]
    assert len(provider.calls) == 5
    assert [call["max_output_tokens"] for call in provider.calls] == [
        1024,
        4096,
        4096,
        4096,
        8192,
    ]
    assert after == before


def test_native_langgraph_schema_excludes_application_checkpoint_metadata():
    pytest.importorskip("langgraph")

    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import (
        AgentGraphRuntime,
        AgentGraphState,
        LangGraphExecutionState,
    )

    runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(MockLLMProvider([])),
    )

    assert "checkpoint_id" in AgentGraphState.__annotations__
    assert "checkpoint_id" not in LangGraphExecutionState.__annotations__
    assert runtime.engine_name == "langgraph"
    assert runtime._compiled_graph is not None


def test_agent_graph_runtime_preserves_explicit_nondefault_output_budget(tmp_path):
    from job_agent.llm.harness import LLMHarness, NodePolicy
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, responses, _ = _fixture_contracts(tmp_path)
    provider = MockLLMProvider(responses)
    runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(provider),
        policy=NodePolicy(max_output_tokens=1536),
    )

    runtime.run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-budget",
        run_id="run-budget",
    )

    assert [call["max_output_tokens"] for call in provider.calls] == [1536] * 5


def test_agent_graph_runtime_stops_after_route_gate_for_not_recommended(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime
    from job_agent.schemas import FitVerdictResult, RiskLevel, Verdict

    class StopBackend:
        def evaluate(self, fit_input):
            return FitVerdictResult(
                verdict=Verdict.NOT_RECOMMENDED,
                score=0.0,
                coverage=0.0,
                risk_level=RiskLevel.HIGH,
                need_human_review=True,
                reason_codes=["fixture_stop"],
                explanation="fixture stop",
            )

    job, resume_text, responses, _ = _fixture_contracts(tmp_path, supported=False)
    provider = MockLLMProvider(responses[:2])
    runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(provider),
        fit_backend=StopBackend(),
    )

    state = runtime.run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-stop",
        run_id="run-stop",
    )

    assert state["fit_result"].verdict.value == "not recommended"
    assert state["verdict_route"].gate == "stop_and_reselect"
    assert state["llm_call_count"] == 2
    assert "resume_tailoring" not in state
    assert len(provider.calls) == 2


def test_graph_idempotency_cache_includes_skill_version(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, responses, _ = _fixture_contracts(tmp_path)
    registry = _Registry("fixture-v1")
    provider = MockLLMProvider([*responses, *responses])
    runtime = AgentGraphRuntime(registry=registry, harness=LLMHarness(provider))
    kwargs = dict(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-cache",
        run_id="run-cache",
    )

    first = runtime.run_selected_job(**kwargs)
    replay = runtime.run_selected_job(**kwargs)
    registry.version = "fixture-v2"
    changed_skill = runtime.run_selected_job(**kwargs)

    assert first["fit_result"] == replay["fit_result"]
    assert len(provider.calls) == 10
    assert changed_skill["provider_traces"][0].skill_version == "fixture-v2"


def test_graph_idempotency_key_tracks_current_prompt_versions(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime, SEMANTIC_PROMPT_VERSIONS

    job, resume_text, _, _ = _fixture_contracts(tmp_path)
    runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(MockLLMProvider([])),
    )

    key = runtime._idempotency_key(
        selected_job=job,
        resume_text=resume_text,
        session_id="prompt-version-test",
    )
    from job_agent.agents.interview_prep import PROMPT_VERSION as INTERVIEW_PROMPT
    from job_agent.agents.resume_tailoring import PROMPT_VERSION as RESUME_PROMPT

    assert SEMANTIC_PROMPT_VERSIONS["resume"] == RESUME_PROMPT
    assert SEMANTIC_PROMPT_VERSIONS["interview"] == INTERVIEW_PROMPT
    assert key


def test_graph_idempotency_key_hashes_forbidden_output_policy(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, _, _ = _fixture_contracts(tmp_path)
    common = dict(registry=_Registry(), harness=LLMHarness(MockLLMProvider([])))
    plain = AgentGraphRuntime(**common)
    guarded = AgentGraphRuntime(
        **common,
        forbidden_output_strings=("UNTRUSTED_TEST_SENTINEL",),
    )

    plain_key = plain._idempotency_key(
        selected_job=job,
        resume_text=resume_text,
        session_id="forbidden-policy-test",
    )
    guarded_key = guarded._idempotency_key(
        selected_job=job,
        resume_text=resume_text,
        session_id="forbidden-policy-test",
    )

    assert plain_key != guarded_key
    assert len(plain_key) == len(guarded_key) == 64


def test_run_semantic_job_flow_is_the_public_guarded_entrypoint(tmp_path):
    from job_agent.graph import run_semantic_job_flow
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider

    job, resume_text, responses, expected_fit = _fixture_contracts(tmp_path)
    resume_path = tmp_path / "resume-entrypoint.md"
    resume_path.write_text(resume_text, encoding="utf-8")
    state = run_semantic_job_flow(
        user_request="find an internship",
        selected_job=job,
        resume_path=resume_path,
        registry=_Registry(),
        harness=LLMHarness(MockLLMProvider(responses)),
    )

    assert state["user_request"] == "find an internship"
    assert state["runtime_mode"] == "agent_api"
    assert state["fit_result"].verdict == expected_fit.verdict
    assert state["llm_call_count"] == 5


def test_semantic_graph_process_resume_skips_verified_provider_calls(tmp_path):
    from job_agent.llm.harness import LLMHarness, LLMInvocationError, NodePolicy
    from job_agent.llm.provider import ProviderTransportError
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime
    from job_agent.runtime.semantic_checkpoint import SemanticCheckpointStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SemanticCheckpointStore(
        tmp_path / "private-checkpoints" / "semantic.json",
        workspace_root=workspace,
    )
    job, resume_text, responses, _ = _fixture_contracts(workspace)
    first_provider = MockLLMProvider(
        [responses[0], responses[1], ProviderTransportError("interrupted after evidence")]
    )
    first_runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(first_provider),
        policy=NodePolicy(max_retries=0),
        checkpoint_store=store,
    )

    with pytest.raises(LLMInvocationError, match="transport_error"):
        first_runtime.run_selected_job(
            selected_job=job,
            resume_text=resume_text,
            session_id="process-resume",
            run_id="semantic-run",
        )

    checkpoint = store.load()
    assert checkpoint.completed_node == "verdict_route"
    assert len(checkpoint.provider_traces) == 2
    second_provider = MockLLMProvider(responses[2:])
    resumed = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(second_provider),
        policy=NodePolicy(max_retries=0),
        checkpoint_store=store,
    ).run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="process-resume",
        run_id="semantic-run",
        resume=True,
    )

    assert len(second_provider.calls) == 3
    assert resumed["llm_call_count"] == 5
    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["checkpoint_id"] == checkpoint.checkpoint_id
    assert not store.path.exists()


def test_semantic_graph_resume_rejects_changed_input_and_preserves_checkpoint(tmp_path):
    from job_agent.agent_runtime.failures import CheckpointError
    from job_agent.llm.harness import LLMHarness, LLMInvocationError, NodePolicy
    from job_agent.llm.provider import ProviderTransportError
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime
    from job_agent.runtime.semantic_checkpoint import SemanticCheckpointStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = SemanticCheckpointStore(
        tmp_path / "private-checkpoints" / "semantic.json",
        workspace_root=workspace,
    )
    job, resume_text, responses, _ = _fixture_contracts(workspace)
    runtime = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(
            MockLLMProvider(
                [responses[0], responses[1], ProviderTransportError("interrupt")]
            )
        ),
        policy=NodePolicy(max_retries=0),
        checkpoint_store=store,
    )
    with pytest.raises(LLMInvocationError):
        runtime.run_selected_job(
            selected_job=job,
            resume_text=resume_text,
            session_id="changed-input",
            run_id="semantic-run",
        )

    with pytest.raises(CheckpointError, match="input_mismatch"):
        AgentGraphRuntime(
            registry=_Registry(),
            harness=LLMHarness(MockLLMProvider(responses[2:])),
            checkpoint_store=store,
        ).run_selected_job(
            selected_job=job,
            resume_text=resume_text + " changed",
            session_id="changed-input",
            run_id="semantic-run",
            resume=True,
        )
    assert store.path.exists()


def test_agent_graph_state_replays_through_existing_artifact_writer(tmp_path):
    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.outputs import write_session_outputs
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, responses, _ = _fixture_contracts(tmp_path)
    state = AgentGraphRuntime(
        registry=_Registry(),
        harness=LLMHarness(MockLLMProvider(responses)),
    ).run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-writer",
        run_id="run-writer",
    )

    paths = write_session_outputs(state, tmp_path / "artifacts")

    assert (paths.session_dir / "01_jd_structured.json").is_file()
    assert (paths.session_dir / "02_evidence_mapping.json").is_file()
    assert (paths.session_dir / "06_targeted_resume.md").is_file()
    assert (paths.session_dir / "07_interview_grilling.md").is_file()


def test_local_transformers_provider_uses_same_structured_contract():
    from job_agent.llm.provider import TraceContext
    from job_agent.llm.providers.transformers import LocalTransformersProvider
    from job_agent.schemas import JobRequirement

    class Runner:
        def generate(self, prompt):
            assert "trusted" in prompt
            return '{"id":"req-1","text":"LoRA","required":true,"probe":"Explain"}'

    provider = LocalTransformersProvider(runner=Runner(), model="fixture-local")
    result = provider.generate_structured(
        system_prompt="trusted",
        user_prompt="untrusted",
        output_schema=JobRequirement,
        tools=[],
        temperature=0.0,
        max_output_tokens=128,
        trace=TraceContext(
            session_id="s",
            run_id="r",
            node_id="n",
            skill_id="skill",
            skill_version="v1",
            prompt_version="p1",
        ),
    )

    assert result.schema_valid is True
    assert result.provider == "transformers"
    assert result.parsed_output["id"] == "req-1"


def test_context_selection_prioritizes_current_verified_then_memory_and_history():
    from job_agent.runtime.context import ContextPriority, ContextSegment, select_context

    selection = select_context(
        [
            ContextSegment(name="history", content="h" * 10, priority=ContextPriority.HISTORY_SUMMARY),
            ContextSegment(name="memory", content="m" * 10, priority=ContextPriority.CONFIRMED_MEMORY),
            ContextSegment(name="verified", content="v" * 10, priority=ContextPriority.VERIFIED_RESULT),
            ContextSegment(name="task", content="t" * 10, priority=ContextPriority.CURRENT_TASK),
        ],
        max_chars=25,
    )

    assert [segment.name for segment in selection.selected] == ["task", "verified"]
    assert selection.dropped == ["memory", "history"]
    assert selection.total_chars == 20


def test_artifact_memory_retrieves_only_explicit_safe_refs(tmp_path):
    from job_agent.memory.artifact_memory import ArtifactMemoryRetriever

    session = tmp_path / "session"
    session.mkdir()
    (session / "01_jd_structured.json").write_text('{"company":"Example"}', encoding="utf-8")
    (session / "06_targeted_resume.md").write_text("private resume", encoding="utf-8")
    retriever = ArtifactMemoryRetriever(session)

    hits = retriever.retrieve(["01_jd_structured.json"])

    assert [hit.path for hit in hits] == ["01_jd_structured.json"]
    assert "private resume" not in hits[0].content
    with pytest.raises(ValueError, match="escapes session"):
        retriever.retrieve(["../outside.json"])


def test_user_state_store_rejects_credential_like_memory(tmp_path):
    from job_agent.memory.contracts import MemoryRecord, UserState
    from job_agent.memory.user_state_store import MemoryPolicyError, UserStateStore

    record = MemoryRecord(
        memory_id="m-sensitive",
        user_id="default-user",
        kind="fact",
        content="api_key=fixture",
        source="user_stated",
        confidence=1.0,
        created_at="2026-07-16T00:00:00+00:00",
        updated_at="2026-07-16T00:00:00+00:00",
    )
    store = UserStateStore(tmp_path / "state.json")

    with pytest.raises(MemoryPolicyError, match="credential-like"):
        store.save(
            user_id="default-user",
            state=UserState(user_id="default-user", memories=[record]),
            expected_version=1,
        )
    assert not store.path.exists()


def test_execution_coordinator_serializes_same_session_without_global_state():
    import threading
    import time

    from job_agent.runtime.concurrency import ExecutionCoordinator

    coordinator = ExecutionCoordinator(max_parallel_sessions=2)
    active = 0
    max_active = 0
    guard = threading.Lock()

    def operation(value):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return value

    results = []
    threads = [
        threading.Thread(
            target=lambda value=index: results.append(
                coordinator.execute(
                    session_id="same-session",
                    idempotency_key=f"key-{value}",
                    operation=lambda: operation(value),
                )[0]
            )
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [0, 1]
    assert max_active == 1
    independent = ExecutionCoordinator(max_parallel_sessions=1)
    assert independent is not coordinator


def test_cli_agent_mock_mode_runs_five_calls_and_writes_sanitized_runtime_audit(
    tmp_path,
    monkeypatch,
    capsys,
):
    import json
    import sys

    from job_agent.cli import main

    skill_root = tmp_path / "skill"
    references = skill_root / "skill-references"
    references.mkdir(parents=True)
    (skill_root / "release-manifest.txt").write_text(
        "# version: fixture-v1\nSKILL.md\nrelease-manifest.txt\nskill-references/\n",
        encoding="utf-8",
    )
    (skill_root / "SKILL.md").write_text("Never fabricate evidence.\n", encoding="utf-8")
    for name in (
        "jd-analysis.md",
        "materials-audit.md",
        "truth-boundary.md",
        "evidence-contract.md",
        "resume-tailoring.md",
        "interview-grilling.md",
        "answer-cards.md",
    ):
        (references / name).write_text(f"Instructions for {name}.\n", encoding="utf-8")
    output_dir = tmp_path / "agent-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job-agent",
            "--mode",
            "agent_api",
            "--agent-provider",
            "mock",
            "--skill-root",
            str(skill_root),
            "--selected-job-id",
            "job_001",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()
    stdout = capsys.readouterr().out
    audit_path = next(output_dir.rglob("agent_runtime_audit.json"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert "Runtime Mode: agent_api" in stdout
    assert "LLM Call Count: 5" in stdout
    assert "Agent Provider: mock" in stdout
    assert audit["llm_call_count"] == 5
    assert len(audit["traces"]) == 5
    assert len(audit["calls"]) == 5
    assert all(trace["schema_valid"] is True for trace in audit["traces"])
    assert all(call["prompt"]["system_prompt_sha256"] for call in audit["calls"])
    assert all(call["prompt"]["user_prompt_chars"] > 0 for call in audit["calls"])
    assert "raw_output" not in audit_path.read_text(encoding="utf-8")


def test_cli_agent_failure_writes_sanitized_failure_audit(tmp_path, monkeypatch, capsys):
    import json
    import sys

    import job_agent.cli as cli_module
    from job_agent.llm.providers.mock import MockLLMProvider as RealMockLLMProvider

    skill_root = tmp_path / "skill"
    references = skill_root / "skill-references"
    references.mkdir(parents=True)
    (skill_root / "release-manifest.txt").write_text(
        "# version: fixture-v1\nSKILL.md\nrelease-manifest.txt\nskill-references/\n",
        encoding="utf-8",
    )
    (skill_root / "SKILL.md").write_text("Never fabricate evidence.\n", encoding="utf-8")
    for name in (
        "jd-analysis.md",
        "materials-audit.md",
        "truth-boundary.md",
        "evidence-contract.md",
        "resume-tailoring.md",
        "interview-grilling.md",
        "answer-cards.md",
    ):
        (references / name).write_text(f"Instructions for {name}.\n", encoding="utf-8")
    output_dir = tmp_path / "failed-agent-output"
    monkeypatch.setattr(
        cli_module,
        "MockLLMProvider",
        lambda responses, model: RealMockLLMProvider(
            [{"unexpected": "shape"}, {"unexpected": "shape"}],
            model=model,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job-agent",
            "--mode",
            "agent_api",
            "--agent-provider",
            "mock",
            "--skill-root",
            str(skill_root),
            "--selected-job-id",
            "job_001",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_module.main()

    stderr = capsys.readouterr().err
    audit_path = output_dir / ".internal" / "agent_runtime_failure_audit.json"
    audit_text = audit_path.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert exc_info.value.code == 1
    assert "Agent Runtime Failed: structured LLM invocation failed: schema_error" in stderr
    assert f"Failure Audit: {audit_path}" in stderr
    assert audit["status"] == "failed"
    assert audit["error"]["error_code"] == "schema_error"
    assert audit["llm_call_count"] == 2
    assert audit["traces"][0]["skill_id"] == "jd-analysis"
    assert audit["traces"][0]["schema_valid"] is False
    assert "raw_output" not in audit_text
    assert "unexpected" not in audit_text


def test_agent_graph_runtime_counts_resume_guard_repair_provider_attempt(tmp_path):
    import copy

    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.runtime.graph_runtime import AgentGraphRuntime

    job, resume_text, responses, _ = _fixture_contracts(tmp_path)
    invalid_resume = copy.deepcopy(responses[2])
    bullet = next(
        bullet
        for key in ("conservative_bullets", "standard_bullets", "stronger_after_evidence")
        for bullet in invalid_resume[key]
    )
    source_level = next(
        item["level"]
        for item in responses[1]["items"]
        if item["requirement_id"] == bullet["requirement_id"]
    )
    bullet["evidence_level"] = "C3" if source_level != "C3" else "C2"
    responses.insert(2, invalid_resume)
    provider = MockLLMProvider(responses)
    runtime = AgentGraphRuntime(registry=_Registry(), harness=LLMHarness(provider))

    state = runtime.run_selected_job(
        selected_job=job,
        resume_text=resume_text,
        session_id="session-guard-repair",
        run_id="run-guard-repair",
    )

    assert state["llm_call_count"] == 6
    assert len(state["provider_traces"]) == 6
    assert [trace.skill_id for trace in state["provider_traces"]].count("resume-tailoring") == 2
