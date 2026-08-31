from pathlib import Path

import pytest

from job_agent.agent_runtime.contracts import AgentAction, ToolContext, ToolEffect
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import ProjectDossier, ProjectFact
from job_agent.interview.question_bank import QuestionAnchorLibrary
from job_agent.interview.tools import READ_SOURCE_TOOL, SEARCH_ANCHORS_TOOL, build_interview_registry, resolve_project_source


def _snapshot():
    dossier = ProjectDossier(
        project_id="p1",
        name="Project",
        source_fingerprint="hash",
        revision=1,
        facts=[ProjectFact(fact_id="f1", kind="implementation", statement="checkpoint recovery exists", status="verified")],
        confirmed_at="now",
    )
    return InterviewContextBuilder.for_project_grill(dossier=dossier, run_id="run-1")


def test_project_source_guard_rejects_learning_and_escape(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("print('ok')", encoding="utf-8")
    assert resolve_project_source(tmp_path, "src/ok.py").name == "ok.py"
    with pytest.raises(ValueError):
        resolve_project_source(tmp_path, "docs/learning/README.md")
    with pytest.raises(ValueError):
        resolve_project_source(tmp_path, "../secret.txt")


def test_anchor_tool_returns_public_anchor_without_reference_answer(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    registry = build_interview_registry(
        _snapshot(), question_library=QuestionAnchorLibrary(), repo_root=tmp_path, run_seed="run-1"
    )
    action = AgentAction(
        action_id="a1",
        tool_name=SEARCH_ANCHORS_TOOL,
        tool_arguments={"topics": ["agent-loop"], "roles": ["agent-engineer"], "difficulty": "medium"},
        expected_observation="anchors",
        progress_claim="search",
    )
    authorization = PolicyEngine().authorize(
        action,
        registry.get(SEARCH_ANCHORS_TOOL).spec,
        PolicyContext(
            skill_allowed_tools=(SEARCH_ANCHORS_TOOL,),
            agent_allowed_tools=(SEARCH_ANCHORS_TOOL,),
            runtime_allowed_tools=(SEARCH_ANCHORS_TOOL,),
        ),
        session_id="s",
        run_id="r",
    )
    observation = ToolExecutor(registry).execute(
        action, authorization, ToolContext(session_id="s", run_id="r", agent_id="a", workspace_root=str(tmp_path))
    )
    assert observation.data["items"]
    assert "reference_points" not in observation.data["items"][0]
    assert observation.data["evidence_refs"][0].startswith("question_anchor:")


def test_interview_registry_has_no_evidence_write_capability(tmp_path: Path) -> None:
    registry = build_interview_registry(
        _snapshot(),
        question_library=QuestionAnchorLibrary(),
        repo_root=tmp_path,
        run_seed="run-1",
    )
    assert registry.names()
    assert all(registry.get(name).spec.effect == ToolEffect.READ_ONLY for name in registry.names())
    assert not any("commit" in name or "update_evidence" in name for name in registry.names())
