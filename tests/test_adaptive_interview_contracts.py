from pathlib import Path

import pytest
from pydantic import ValidationError

from job_agent.interview.contracts import (
    AdaptiveInterviewRun,
    InterviewBlueprint,
    InterviewContextSnapshot,
    InterviewPhaseBudget,
    InterviewSourceRef,
    InterviewTurnDecision,
    ProjectDossier,
    ProjectFact,
    TurnAssessment,
    VerificationTarget,
)
from job_agent.interview.paths import InterviewArtifactPaths


def _dossier() -> ProjectDossier:
    return ProjectDossier(
        project_id="agent-harness",
        name="Agent Harness",
        source_fingerprint="sha256:repo",
        revision=1,
        facts=[
            ProjectFact(
                fact_id="fact-1",
                kind="implementation",
                statement="The runtime persists checkpoints.",
                status="verified",
                evidence_refs=["code:checkpoint"],
                source_hashes={"src/checkpoint.py": "abc"},
            )
        ],
        confirmed_at="2026-07-31T00:00:00Z",
    )


def _blueprint() -> InterviewBlueprint:
    return InterviewBlueprint(
        duration_minutes=15,
        pressure="normal",
        phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=900, required=True)],
        targets=[
            VerificationTarget(
                target_id="target-1",
                competency="checkpoint recovery",
                intent="verify failure handling",
                priority="must",
                source_refs=["fact:1"],
            )
        ],
    )


def _assessment() -> TurnAssessment:
    return TurnAssessment(
        answered=False,
        correctness="not_applicable",
        specificity="vague",
        ownership="not_applicable",
    )


def test_project_context_and_run_round_trip() -> None:
    context = InterviewContextSnapshot(
        run_id="run-1",
        kind="project_grill",
        created_at="2026-07-31T00:00:00Z",
        source_refs=[
            InterviewSourceRef(
                ref_id="fact:1", kind="project_fact", artifact_path="dossier.json", content_hash="abc"
            )
        ],
        project_dossier=_dossier(),
    )
    decision = InterviewTurnDecision(
        turn_id="turn-1",
        action="switch_target",
        phase="resume_project",
        target_id="target-1",
        public_message="Why did you choose checkpoint persistence?",
        question_origin="project_generated",
        assessment=_assessment(),
        source_refs=["fact:1"],
    )
    run = AdaptiveInterviewRun(
        run_id="run-1",
        kind="project_grill",
        status="waiting",
        context_path="context.json",
        blueprint=_blueprint(),
        pending_turn=decision,
        started_at="2026-07-31T00:00:00Z",
        updated_at="2026-07-31T00:00:00Z",
    )
    assert AdaptiveInterviewRun.model_validate(run.model_dump()).pending_turn.turn_id == "turn-1"
    assert context.project_dossier.ready_for_interview


def test_full_mock_requires_job_linked_sources() -> None:
    with pytest.raises(ValidationError, match="full mock requires"):
        InterviewContextSnapshot(
            run_id="run-1",
            kind="full_mock",
            created_at="2026-07-31T00:00:00Z",
            source_refs=[
                InterviewSourceRef(ref_id="resume:1", kind="resume", artifact_path="resume.md", content_hash="abc")
            ],
            job_id="job-1",
        )


def test_finish_turn_cannot_keep_target() -> None:
    with pytest.raises(ValidationError, match="finish must close"):
        InterviewTurnDecision(
            turn_id="turn-1",
            action="finish",
            phase="closing",
            target_id="target-1",
            public_message="That concludes the interview.",
            question_origin="closing",
            assessment=_assessment(),
        )


def test_interview_paths_reject_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        InterviewArtifactPaths.for_run(tmp_path, "../escape")
    paths = InterviewArtifactPaths.for_run(tmp_path, "run-1")
    assert paths.run == tmp_path.resolve() / "interview_runs" / "run-1" / "run.json"
