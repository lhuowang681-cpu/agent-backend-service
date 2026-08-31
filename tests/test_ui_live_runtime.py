from __future__ import annotations

from job_agent.ui.local_config import load_local_runtime_defaults
from job_agent.ui.modes import check_live_runtime


def test_live_readiness_requires_environment_key_and_existing_skill_root(tmp_path, monkeypatch):
    monkeypatch.delenv("JOB_AGENT_LIVE_API_KEY", raising=False)
    assert not check_live_runtime(skill_root=str(tmp_path), base_url="https://example.test", model="model").ready
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "secret")
    ready = check_live_runtime(skill_root=str(tmp_path), base_url="https://example.test", model="model")
    assert ready.ready and ready.api_key_source == "JOB_AGENT_LIVE_API_KEY"


def test_runtime_defaults_do_not_read_glm_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "glm.txt").write_text("not-a-key", encoding="utf-8")
    monkeypatch.delenv("JOB_AGENT_LIVE_API_KEY", raising=False)
    assert load_local_runtime_defaults().api_key is None


def test_live_runner_failure_does_not_write_replacement_session(tmp_path, monkeypatch):
    from job_agent.ui.jd_pool import parse_jd_pool
    from job_agent.ui.modes import RuntimeContext, WorkbenchMode
    from job_agent.ui.runner import run_initial
    jobs = parse_jd_pool("SFT engineer")
    monkeypatch.setattr("job_agent.ui.runner.run_semantic_job_flow", lambda **_: (_ for _ in ()).throw(RuntimeError("network failed")))
    try:
        run_initial(jobs, "resume", tmp_path, mode=WorkbenchMode.AGENT_API_LIVE, runtime=RuntimeContext(provider=object(), harness=object(), registry=object()))
    except RuntimeError as exc:
        assert "network failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("live failure must propagate")
    assert not list((tmp_path / "sessions").glob("*")) if (tmp_path / "sessions").exists() else True


def test_v2_live_runner_commits_sparse_bundle_without_legacy_artifacts(tmp_path):
    from pathlib import Path

    from job_agent.llm.harness import LLMHarness
    from job_agent.llm.providers.mock import MockLLMProvider
    from job_agent.llm.skill_registry import SkillSpec
    from job_agent.schemas import RawJob, StructuredJD
    from job_agent.ui.modes import RuntimeContext
    from job_agent.ui.runner import run_initial_v2_live

    class Registry:
        def get(self, skill_id: str) -> SkillSpec:
            return SkillSpec(
                skill_id=skill_id,
                version="fixture-v1",
                instructions="Extract explicit requirements only.",
                reference_paths=(Path("skill-references/jd-analysis.md"),),
                output_schema=StructuredJD,
            )

    provider = MockLLMProvider(
            [{"parents": [], "atoms": [], "warnings": []}]
    )
    runtime = RuntimeContext(
        provider=provider,
        harness=LLMHarness(provider),
        registry=Registry(),
    )
    job = RawJob(
        job_id="job-1",
        company="Acme",
        title="Agent Engineer",
        desc="岗位介绍",
        url="",
        location="",
    )
    session = run_initial_v2_live([job], "原始简历", tmp_path, runtime=runtime)
    assert (session / "evidence_current.json").exists()
    assert (session / "06_targeted_resume.md").read_text(encoding="utf-8").endswith(
        "原始简历\n"
    )
    assert not any((session / name).exists() for name in (
        "01_jd_structured.json",
        "02_evidence_mapping.json",
        "03_fit_verdict.json",
    ))
    assert len(provider.calls) == 1


def test_v2_live_runner_removes_staging_session_on_failure(tmp_path, monkeypatch):
    from job_agent.schemas import RawJob
    from job_agent.ui.modes import RuntimeContext
    from job_agent.ui.runner import run_initial_v2_live

    monkeypatch.setattr(
        "job_agent.evidence.pipeline.EvidenceV2Pipeline.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    job = RawJob(
        job_id="job-1",
        company="Acme",
        title="Agent Engineer",
        desc="明确要求",
        url="",
        location="",
    )
    try:
        run_initial_v2_live(
            [job],
            "原始简历",
            tmp_path,
            runtime=RuntimeContext(provider=object(), harness=object(), registry=object()),
        )
    except RuntimeError as exc:
        assert "provider failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("live failure must propagate")
    sessions = tmp_path / "sessions"
    assert sessions.exists()
    assert list(sessions.iterdir()) == []
