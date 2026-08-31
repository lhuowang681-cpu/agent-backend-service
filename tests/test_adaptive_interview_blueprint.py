from pathlib import Path

import pytest

from job_agent.agent_runtime.contracts import AgentFinish
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import ProjectDossier, ProjectFact
from job_agent.interview.controller import AdaptiveInterviewController, AdaptiveInterviewError


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


def test_project_blueprint_has_targets_not_question_queue(tmp_path: Path) -> None:
    result = {
        "duration_minutes": 15,
        "pressure": "normal",
        "focus": "reliability",
        "phases": [
            {"phase": "resume_project", "target_seconds": 780, "required": True},
            {"phase": "closing", "target_seconds": 120, "required": True}
        ],
        "targets": [
            {"target_id": "t1", "competency": "checkpoint", "intent": "verify recovery", "priority": "must", "source_refs": ["project_fact:f1"], "status": "unseen"}
        ]
    }
    model = ScriptedAgentModel([AgentFinish(result=result, completion_evidence=["project_fact:f1"], confidence=0.8)])
    blueprint = AdaptiveInterviewController(tmp_path, repo_root=tmp_path).build_blueprint(
        _snapshot(), duration_minutes=15, pressure="normal", focus="reliability", model=model
    )
    assert blueprint.targets[0].target_id == "t1"
    assert "questions" not in blueprint.model_dump()


def test_blueprint_rejects_unknown_source_ref(tmp_path: Path) -> None:
    result = {
        "duration_minutes": 15,
        "pressure": "normal",
        "phases": [
            {"phase": "resume_project", "target_seconds": 780, "required": True},
            {"phase": "closing", "target_seconds": 120, "required": True}
        ],
        "targets": [
            {"target_id": "t1", "competency": "checkpoint", "intent": "verify", "priority": "must", "source_refs": ["invented"], "status": "unseen"}
        ]
    }
    model = ScriptedAgentModel([AgentFinish(result=result, completion_evidence=["invented"], confidence=0.8)])
    with pytest.raises(AdaptiveInterviewError, match="repeated_verifier_failure"):
        AdaptiveInterviewController(tmp_path, repo_root=tmp_path).build_blueprint(
            _snapshot(), duration_minutes=15, pressure="normal", model=model
        )
