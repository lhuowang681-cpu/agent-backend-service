from pathlib import Path

from job_agent.interview.contracts import (
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
from job_agent.interview.run_store import AdaptiveInterviewRunStore


def _context() -> InterviewContextSnapshot:
    dossier = ProjectDossier(
        project_id="p1",
        name="Project",
        source_fingerprint="hash",
        revision=1,
        facts=[ProjectFact(fact_id="f1", kind="implementation", statement="fact", status="verified")],
        confirmed_at="now",
    )
    return InterviewContextSnapshot(
        run_id="run-1",
        kind="project_grill",
        created_at="now",
        source_refs=[InterviewSourceRef(ref_id="project_fact:f1", kind="project_fact", artifact_path="dossier.json", content_hash="hash")],
        project_dossier=dossier,
    )


def _blueprint() -> InterviewBlueprint:
    return InterviewBlueprint(
        duration_minutes=15,
        phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=900)],
        targets=[VerificationTarget(target_id="t1", competency="architecture", intent="verify", priority="must", source_refs=["project_fact:f1"])],
    )


def _decision(turn_id: str, *, finish: bool = False) -> InterviewTurnDecision:
    return InterviewTurnDecision(
        turn_id=turn_id,
        action="finish" if finish else "switch_target",
        phase="closing" if finish else "resume_project",
        target_id=None if finish else "t1",
        public_message="That concludes the interview." if finish else "Explain the architecture.",
        question_origin="closing" if finish else "project_generated",
        assessment=TurnAssessment(answered=finish, correctness="not_applicable", specificity="mixed", ownership="not_applicable"),
        source_refs=[] if finish else ["project_fact:f1"],
    )


def test_run_store_commits_turns_and_exports_only_when_finished(tmp_path: Path) -> None:
    store = AdaptiveInterviewRunStore(tmp_path)
    run = store.create(_context(), _blueprint())
    run = store.commit_turn(run, _decision("turn-1"), answer=None, elapsed_seconds=0)
    assert run.pending_turn.turn_id == "turn-1"
    assert not (tmp_path / "interview_runs" / "run-1" / "transcript.json").exists()
    run = store.commit_turn(run, _decision("closing", finish=True), answer="My answer", elapsed_seconds=12)
    assert run.status == "completed"
    assert run.transcript[0].answer == "My answer"
    assert (tmp_path / "interview_runs" / "run-1" / "transcript.json").exists()


def test_pause_resume_and_stale_revision_protection(tmp_path: Path) -> None:
    store = AdaptiveInterviewRunStore(tmp_path)
    original = store.create(_context(), _blueprint())
    paused = store.set_status("run-1", "paused")
    assert paused.status == "paused"
    resumed = store.set_status("run-1", "waiting")
    assert resumed.status == "waiting"
    try:
        store.commit_turn(original, _decision("turn-1"), answer=None, elapsed_seconds=0)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale run must not commit")


def test_commit_turn_retry_is_idempotent_and_conflicts_are_rejected(tmp_path: Path) -> None:
    store = AdaptiveInterviewRunStore(tmp_path)
    original = store.create(_context(), _blueprint())
    decision = _decision("turn-1")
    committed = store.commit_turn(original, decision, answer=None, elapsed_seconds=0)
    retried = store.commit_turn(original, decision, answer=None, elapsed_seconds=0)
    assert retried.revision == committed.revision
    assert len(retried.decision_log) == 1

    changed = decision.model_copy(update={"public_message": "A different question."})
    try:
        store.commit_turn(original, changed, answer=None, elapsed_seconds=0)
    except ValueError as exc:
        assert "idempotency conflict" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("same turn_id with different payload must fail")
