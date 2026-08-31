import json
from pathlib import Path

import pytest

from job_agent.career import CareerStore, CompanyDraft, JobDraft
from job_agent.interview.context import InterviewContextBuilder
from job_agent.interview.contracts import (
    InterviewBlueprint,
    InterviewPhaseBudget,
    InterviewTurnDecision,
    ProjectDossier,
    ProjectFact,
    TurnAssessment,
    VerificationTarget,
)
from job_agent.ui.adaptive_interview_loop import (
    index_finished_interview,
    start_full_mock,
    start_project_grill,
    submit_adaptive_answer,
)
from tests.evidence_v2_fixtures import commit_zero_atom_evidence_v2


class FakeController:
    def build_blueprint(self, snapshot, *, duration_minutes, pressure, focus=None, runtime=None):
        return InterviewBlueprint(duration_minutes=duration_minutes, pressure=pressure, focus=focus, phases=[InterviewPhaseBudget(phase="resume_project", target_seconds=duration_minutes * 60)], targets=[VerificationTarget(target_id="t1", competency="architecture", intent="verify", priority="must", source_refs=[snapshot.source_refs[0].ref_id])])

    def next_turn(self, run, *, answer, runtime=None):
        if answer is None:
            return InterviewTurnDecision(turn_id="turn-1", action="switch_target", phase="resume_project", target_id="t1", public_message="Explain the project architecture.", question_origin="project_generated", assessment=TurnAssessment(answered=False, correctness="not_applicable", specificity="vague", ownership="not_applicable"), source_refs=["project_fact:f1"])
        return InterviewTurnDecision(turn_id="turn-2", action="follow_up", phase="resume_project", target_id="t1", public_message="What failure changed that design?", question_origin="answer_follow_up", assessment=TurnAssessment(answered=True, correctness="partially_supported", specificity="mixed", ownership="clear", new_leads=["failure handling"]), source_refs=["project_fact:f1"])


def _dossier():
    return ProjectDossier(project_id="p1", name="Project", source_fingerprint="hash", revision=1, facts=[ProjectFact(fact_id="f1", kind="implementation", statement="architecture exists", status="verified")], confirmed_at="now")


def test_project_ui_adapter_starts_and_advances_without_live_scores(tmp_path: Path) -> None:
    controller = FakeController()
    root, run = start_project_grill(output_root=tmp_path, repo_root=tmp_path, dossier=_dossier(), duration_minutes=15, pressure="normal", focus=None, runtime=object(), run_id="run-1", controller=controller)
    assert run.pending_turn.public_message == "Explain the project architecture."
    advanced = submit_adaptive_answer(root=root, repo_root=tmp_path, run_id="run-1", answer="I separated workflow and runtime.", elapsed_seconds=20, runtime=object(), controller=controller)
    assert advanced.transcript[0].answer.startswith("I separated")
    assert advanced.pending_turn.question_origin == "answer_follow_up"
    assert "score" not in advanced.pending_turn.model_dump()


def test_answer_that_reaches_time_limit_closes_without_an_extra_question(tmp_path: Path) -> None:
    class TimeAwareController(FakeController):
        def next_turn(self, run, *, answer, runtime=None):
            if answer is None:
                return super().next_turn(run, answer=answer, runtime=runtime)
            assert run.active_seconds == 15 * 60
            assert run.turn_count == 1
            return InterviewTurnDecision(
                turn_id="closing",
                action="finish",
                phase="closing",
                target_id=None,
                public_message="本场面试结束。",
                question_origin="closing",
                assessment=TurnAssessment(
                    answered=True,
                    correctness="not_applicable",
                    specificity="mixed",
                    ownership="not_applicable",
                ),
            )

    controller = TimeAwareController()
    root, run = start_project_grill(
        output_root=tmp_path,
        repo_root=tmp_path,
        dossier=_dossier(),
        duration_minutes=15,
        pressure="normal",
        focus=None,
        runtime=object(),
        run_id="timed-run",
        controller=controller,
    )
    finished = submit_adaptive_answer(
        root=root,
        repo_root=tmp_path,
        run_id=run.run_id,
        answer="Final answer",
        elapsed_seconds=15 * 60,
        runtime=object(),
        controller=controller,
    )
    assert finished.status == "completed"
    assert finished.pending_turn is None
    assert len(finished.transcript) == 1


def test_failed_next_turn_does_not_advance_run(tmp_path: Path) -> None:
    controller = FakeController()
    root, run = start_project_grill(output_root=tmp_path, repo_root=tmp_path, dossier=_dossier(), duration_minutes=15, pressure="normal", focus=None, runtime=object(), run_id="run-1", controller=controller)
    class Broken(FakeController):
        def next_turn(self, run, *, answer, runtime=None):
            raise RuntimeError("provider failed")
    with pytest.raises(RuntimeError):
        submit_adaptive_answer(root=root, repo_root=tmp_path, run_id="run-1", answer="kept", elapsed_seconds=20, runtime=object(), controller=Broken())
    from job_agent.interview.run_store import AdaptiveInterviewRunStore
    preserved = AdaptiveInterviewRunStore(root).load("run-1")
    assert preserved.revision == run.revision
    assert preserved.transcript == []


def test_full_mock_adapter_uses_exact_job_session_and_indexes_once(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "linked"
    session.mkdir(parents=True)
    commit_zero_atom_evidence_v2(session, raw_jd="Acme Agent Engineer")
    (session / "06_targeted_resume.md").write_text("LINKED RESUME", encoding="utf-8")
    (tmp_path / "06_targeted_resume.md").write_text("UNRELATED RESUME", encoding="utf-8")
    career_store = CareerStore.open(tmp_path)
    company = career_store.save_company(CompanyDraft("Acme"))
    job = career_store.attach_job(
        JobDraft("Agent Engineer", company_id=company.id), session_dir=session
    )

    root, run = start_full_mock(
        career_store=career_store,
        repo_root=tmp_path,
        job_id=job.id,
        dossier=_dossier(),
        duration_minutes=15,
        pressure="normal",
        runtime=object(),
        run_id="full-1",
        controller=FakeController(),
    )
    assert root == session.resolve()
    snapshot = InterviewContextBuilder(career_store).for_full_mock(
        job_id=job.id, run_id="comparison", dossier=_dossier()
    )
    assert snapshot.resume_text == "LINKED RESUME"

    run = submit_adaptive_answer(
        root=root,
        repo_root=tmp_path,
        run_id=run.run_id,
        answer="I designed the checkpoint boundary.",
        elapsed_seconds=15,
        runtime=object(),
        controller=FakeController(),
    )
    from job_agent.interview.run_store import AdaptiveInterviewRunStore

    finished = AdaptiveInterviewRunStore(root).set_status(run.run_id, "ended_by_user")
    index_finished_interview(career_store=career_store, root=root, run=finished)
    index_finished_interview(career_store=career_store, root=root, run=finished)
    rows = career_store.connection().execute("SELECT * FROM interviews").fetchall()
    assert len(rows) == 1
    assert rows[0]["job_id"] == job.id
    assert rows[0]["artifact_path"] == "interview_runs/full-1/transcript.json"


def test_adaptive_page_disables_live_generation_without_environment_key(tmp_path: Path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("JOB_AGENT_OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.delenv("JOB_AGENT_LIVE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_INTERN_SKILL_ROOT", raising=False)
    at = AppTest.from_string(
        "from job_agent.ui.app import _render_adaptive_interview\n"
        "_render_adaptive_interview()\n"
    )
    at.run()
    assert not at.exception
    dossier_buttons = [
        button
        for button in at.button
        if button.label == "读取当前仓库并生成 Dossier 草稿"
    ]
    assert dossier_buttons and dossier_buttons[0].disabled
