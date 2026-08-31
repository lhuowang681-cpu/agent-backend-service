from pathlib import Path

import pytest

from job_agent.agent_runtime.contracts import AgentAction, AgentFinish
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import (
    InterviewBlueprint,
    InterviewPhaseBudget,
    ProjectDossier,
    ProjectFact,
    VerificationTarget,
)
from job_agent.interview.controller import AdaptiveInterviewController, AdaptiveInterviewError
from job_agent.interview.run_store import AdaptiveInterviewRunStore


def _setup(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "checkpoint.py").write_text("def save_checkpoint(): pass\n", encoding="utf-8")
    dossier = ProjectDossier(
        project_id="p1",
        name="Project",
        source_fingerprint="hash",
        revision=1,
        facts=[ProjectFact(fact_id="f1", kind="implementation", statement="checkpoint recovery exists", status="verified")],
        confirmed_at="now",
    )
    context = InterviewContextBuilder.for_project_grill(dossier=dossier, run_id="run-1")
    blueprint = InterviewBlueprint(
        duration_minutes=15,
        phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=900)],
        targets=[VerificationTarget(target_id="t1", competency="checkpoint", intent="verify recovery", priority="must", source_refs=["project_fact:f1"])],
    )
    store = AdaptiveInterviewRunStore(tmp_path)
    return store, store.create(context, blueprint)


def _turn_result(turn_id="turn-1", action="switch_target"):
    return {
        "turn_id": turn_id,
        "action": action,
        "phase": "resume_project",
        "target_id": "t1",
        "public_message": "How does checkpoint recovery avoid duplicate work?",
        "question_origin": "project_generated",
        "assessment": {
            "answered": False,
            "correctness": "not_applicable",
            "specificity": "vague",
            "ownership": "not_applicable",
            "conflicts": [],
            "new_leads": [],
            "source_refs": ["project_fact:f1"]
        },
        "target_updates": {"t1": "probing"},
        "source_refs": ["project_fact:f1"]
    }


def test_controller_can_generate_grounded_turn_without_tool(tmp_path: Path) -> None:
    store, run = _setup(tmp_path)
    model = ScriptedAgentModel([
        AgentFinish(result=_turn_result(), completion_evidence=["project_fact:f1"], confidence=0.8)
    ])
    decision = AdaptiveInterviewController(tmp_path, repo_root=tmp_path).next_turn(run, answer=None, model=model)
    assert decision.public_message.startswith("How does checkpoint")
    committed = store.commit_turn(run, decision, answer=None, elapsed_seconds=0)
    assert committed.pending_turn.turn_id == "turn-1"


def test_controller_can_read_source_then_submit(tmp_path: Path) -> None:
    _, run = _setup(tmp_path)
    action = AgentAction(
        action_id="read-1",
        tool_name="interview.read_project_source",
        tool_arguments={"relative_path": "src/checkpoint.py", "start_line": 1, "end_line": 5},
        expected_observation="source",
        progress_claim="verify source",
    )
    result = _turn_result()
    result["source_refs"] = ["code:src/checkpoint.py#L1-L1"]
    result["assessment"]["source_refs"] = ["code:src/checkpoint.py#L1-L1"]
    model = ScriptedAgentModel([
        action,
        AgentFinish(result=result, completion_evidence=["code:src/checkpoint.py#L1-L1"], confidence=0.8),
    ])
    decision = AdaptiveInterviewController(tmp_path, repo_root=tmp_path).next_turn(run, answer=None, model=model)
    assert decision.source_refs[0].startswith("code:")


def test_controller_rejects_live_score_leak(tmp_path: Path) -> None:
    _, run = _setup(tmp_path)
    result = _turn_result()
    result["public_message"] = "评分：3/5。下一题是什么？"
    model = ScriptedAgentModel([
        AgentFinish(result=result, completion_evidence=["project_fact:f1"], confidence=0.8),
        AgentFinish(result=result, completion_evidence=["project_fact:f1"], confidence=0.8),
    ])
    with pytest.raises(AdaptiveInterviewError, match="repeated_verifier_failure"):
        AdaptiveInterviewController(tmp_path, repo_root=tmp_path).next_turn(run, answer=None, model=model)
