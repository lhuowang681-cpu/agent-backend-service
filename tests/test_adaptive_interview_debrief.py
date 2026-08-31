from pathlib import Path

import pytest

from job_agent.agent_runtime.contracts import AgentFinish
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import (
    InterviewBlueprint,
    InterviewPhaseBudget,
    ProjectDossier,
    ProjectFact,
    VerificationTarget,
)
from job_agent.interview.debrief import AdaptiveDebriefError, AdaptiveDebriefService
from job_agent.interview.run_store import AdaptiveInterviewRunStore


def _completed_run(tmp_path: Path):
    dossier = ProjectDossier(project_id="p1", name="Project", source_fingerprint="hash", revision=1, facts=[ProjectFact(fact_id="f1", kind="implementation", statement="checkpoint exists", status="verified")], confirmed_at="now")
    context = InterviewContextBuilder.for_project_grill(dossier=dossier, run_id="run-1")
    blueprint = InterviewBlueprint(duration_minutes=15, phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=780), InterviewPhaseBudget(phase="closing", target_seconds=120)], targets=[VerificationTarget(target_id="t1", competency="checkpoint", intent="verify", priority="must", source_refs=["project_fact:f1"])])
    store = AdaptiveInterviewRunStore(tmp_path)
    run = store.create(context, blueprint)
    from tests.test_adaptive_interview_run_store import _decision
    run = store.commit_turn(run, _decision("turn-1"), answer=None, elapsed_seconds=0)
    run = store.commit_turn(run, _decision("turn-2"), answer="The checkpoint saves state.", elapsed_seconds=10)
    run = store.commit_turn(run, _decision("closing", finish=True), answer="Recovery reads it.", elapsed_seconds=10)
    return run


def _result(run_id="run-1", ref="turn:turn-1"):
    dimensions = []
    for name in ["technical_correctness", "project_depth", "ownership_clarity", "evidence_specificity", "communication_structure", "role_fit"]:
        dimensions.append({"dimension": name, "score": 3, "rationale": "Grounded but incomplete.", "evidence_refs": [ref]})
    return {
        "run_id": run_id,
        "conclusion": "borderline",
        "conclusion_rationale": "The answers were partially grounded.",
        "conclusion_evidence_refs": [ref],
        "dimensions": dimensions,
        "strengths": ["Clear checkpoint concept"],
        "risks": ["Recovery details are thin"],
        "improvement_cards": [{"turn_id": "turn-1", "issue": "Missing failure boundary", "recommended_structure": ["state goal", "explain mechanism", "name boundary"], "usable_fact_refs": ["project_fact:f1"], "honesty_boundary": "Do not claim production exactly-once.", "retry_question": "How does recovery handle a lost response?"}],
        "final_note": "Practice signal only; not a hiring prediction."
    }


def test_grounded_debrief_writes_json_and_markdown(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    model = ScriptedAgentModel([AgentFinish(result=_result(), completion_evidence=["turn:turn-1"], confidence=0.8)])
    debrief = AdaptiveDebriefService(tmp_path).generate(run, model=model)
    assert len(debrief.dimensions) == 6
    markdown = (tmp_path / "interview_runs" / "run-1" / "debrief.md").read_text(encoding="utf-8")
    assert "average" not in markdown.lower()
    assert "turn:turn-1" in markdown


def test_debrief_rejects_active_run(tmp_path: Path) -> None:
    dossier = ProjectDossier(project_id="p1", name="Project", source_fingerprint="hash", revision=1, facts=[ProjectFact(fact_id="f1", kind="implementation", statement="checkpoint exists", status="verified")], confirmed_at="now")
    context = InterviewContextBuilder.for_project_grill(dossier=dossier, run_id="active")
    blueprint = InterviewBlueprint(duration_minutes=15, phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=900)], targets=[VerificationTarget(target_id="t1", competency="checkpoint", intent="verify", priority="must", source_refs=["project_fact:f1"])])
    run = AdaptiveInterviewRunStore(tmp_path).create(context, blueprint)
    with pytest.raises(AdaptiveDebriefError, match="must finish"):
        AdaptiveDebriefService(tmp_path).generate(run, model=ScriptedAgentModel([]))


def test_debrief_rejects_unknown_reference(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    bad = _result(ref="turn:invented")
    model = ScriptedAgentModel([AgentFinish(result=bad, completion_evidence=["turn:invented"], confidence=0.8)])
    with pytest.raises(AdaptiveDebriefError, match="repeated_verifier_failure"):
        AdaptiveDebriefService(tmp_path).generate(run, model=model)


def test_debrief_requires_each_dimension_exactly_once(tmp_path: Path) -> None:
    run = _completed_run(tmp_path)
    malformed = _result()
    malformed["dimensions"] = malformed["dimensions"][:-1]
    model = ScriptedAgentModel(
        [AgentFinish(result=malformed, completion_evidence=["turn:turn-1"], confidence=0.8)]
    )
    with pytest.raises(AdaptiveDebriefError, match="repeated_verifier_failure"):
        AdaptiveDebriefService(tmp_path).generate(run, model=model)


def test_debrief_markdown_replacement_is_atomic(tmp_path: Path, monkeypatch) -> None:
    run = _completed_run(tmp_path)
    markdown_path = tmp_path / "interview_runs" / "run-1" / "debrief.md"
    markdown_path.write_text("previous complete debrief", encoding="utf-8")
    model = ScriptedAgentModel(
        [AgentFinish(result=_result(), completion_evidence=["turn:turn-1"], confidence=0.8)]
    )
    from job_agent.interview import run_store

    real_replace = run_store.os.replace

    def fail_markdown_replace(source, destination):
        if Path(destination).suffix == ".md":
            raise OSError("simulated markdown replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(run_store.os, "replace", fail_markdown_replace)
    with pytest.raises(OSError, match="simulated markdown replace failure"):
        AdaptiveDebriefService(tmp_path).generate(run, model=model)
    assert markdown_path.read_text(encoding="utf-8") == "previous complete debrief"
