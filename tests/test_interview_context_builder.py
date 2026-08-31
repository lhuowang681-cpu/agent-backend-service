import json
from pathlib import Path

import pytest

from job_agent.career import CareerStore, CompanyDraft, JobDraft
from job_agent.interview.context import InterviewContextBuilder, InterviewContextError
from job_agent.interview.contracts import ProjectDossier, ProjectFact
from tests.evidence_v2_fixtures import commit_zero_atom_evidence_v2


def _dossier() -> ProjectDossier:
    return ProjectDossier(
        project_id="agent-harness",
        name="Agent Harness",
        source_fingerprint="repo-1",
        revision=1,
        facts=[
            ProjectFact(
                fact_id="checkpoint",
                kind="implementation",
                statement="The runtime persists checkpoints.",
                status="verified",
                evidence_refs=["src:checkpoint"],
                source_hashes={"src/checkpoint.py": "abc"},
            )
        ],
        confirmed_at="2026-07-31T00:00:00Z",
    )


def test_project_grill_snapshot_does_not_require_job(tmp_path: Path) -> None:
    snapshot = InterviewContextBuilder.for_project_grill(dossier=_dossier(), run_id="run-project")
    assert snapshot.kind == "project_grill"
    assert snapshot.job_id is None
    assert snapshot.source_refs[0].ref_id == "project_fact:checkpoint"


def test_project_grill_rejects_unconfirmed_dossier(tmp_path: Path) -> None:
    draft = _dossier().model_copy(update={"confirmed_at": None})
    with pytest.raises(InterviewContextError, match="not confirmed"):
        InterviewContextBuilder.for_project_grill(dossier=draft, run_id="run-draft")


def test_full_mock_uses_exact_job_linked_artifacts(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "job-session"
    session.mkdir(parents=True)
    commit_zero_atom_evidence_v2(session, raw_jd="Acme Agent Engineer")
    (session / "06_targeted_resume.md").write_text("JOB-LINKED RESUME", encoding="utf-8")
    (tmp_path / "06_targeted_resume.md").write_text("WRONG LATEST RESUME", encoding="utf-8")
    store = CareerStore.open(tmp_path)
    company = store.save_company(CompanyDraft("Acme"))
    job = store.attach_job(JobDraft("Agent Engineer", company_id=company.id), session_dir=session)
    snapshot = InterviewContextBuilder(store).for_full_mock(job_id=job.id, run_id="run-full", dossier=_dossier())
    assert snapshot.resume_text == "JOB-LINKED RESUME"
    assert "WRONG" not in snapshot.resume_text
    assert snapshot.job_id == job.id
    assert {ref.kind for ref in snapshot.source_refs} >= {"jd", "resume", "jd_evidence", "project_fact"}


def test_full_mock_missing_linked_resume_fails_closed(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "job-session"
    session.mkdir(parents=True)
    commit_zero_atom_evidence_v2(session)
    store = CareerStore.open(tmp_path)
    job = store.attach_job(JobDraft("Agent Engineer"), session_dir=session)
    with pytest.raises(InterviewContextError, match="cannot read"):
        InterviewContextBuilder(store).for_full_mock(job_id=job.id, run_id="run-full")


def test_full_mock_does_not_fall_back_to_legacy_evidence(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "legacy-job"
    session.mkdir(parents=True)
    (session / "01_jd_structured.json").write_text("{}", encoding="utf-8")
    (session / "02_evidence_mapping.json").write_text("[]", encoding="utf-8")
    (session / "06_targeted_resume.md").write_text("LEGACY RESUME", encoding="utf-8")
    store = CareerStore.open(tmp_path)
    job = store.attach_job(JobDraft("Agent Engineer"), session_dir=session)
    with pytest.raises(InterviewContextError, match="requires a current Evidence v2"):
        InterviewContextBuilder(store).for_full_mock(job_id=job.id, run_id="run-legacy")
