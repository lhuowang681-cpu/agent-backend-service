from __future__ import annotations

import json
from pathlib import Path

import pytest

from job_agent.career import CareerStore, CompanyDraft, JobDraft, LegacyCareerMigration, StageEventDraft


def test_company_dedup_events_and_persistence(tmp_path: Path):
    store = CareerStore.open(tmp_path)
    first = store.save_company(CompanyDraft("Ａcme", tags={"ai"}, priority="high"))
    second = store.save_company(CompanyDraft(" Acme ", tags={"infra"}, priority="normal"))
    assert first.id == second.id
    job = store.attach_job(JobDraft("Engineer", company_id=first.id))
    store.append_stage_event(job.id, StageEventDraft("applied", occurred_at="2026-01-02T00:00:00+00:00", source_key="manual-1"))
    store.close()
    detail = CareerStore.open(tmp_path).get_company_detail(first.id)
    assert detail.jobs[0].current_state == "applied"
    assert detail.company.tags == ("infra",)


def test_event_write_rolls_back_when_foreign_key_fails(tmp_path: Path):
    store = CareerStore.open(tmp_path)
    with pytest.raises(Exception):
        store.append_stage_event("missing", StageEventDraft("applied", source_key="bad"))
    assert store.connection().execute("SELECT count(*) FROM stage_events").fetchone()[0] == 0


def test_stage_event_rejects_invalid_transition(tmp_path: Path):
    store = CareerStore.open(tmp_path)
    company = store.save_company(CompanyDraft("Acme"))
    job = store.attach_job(JobDraft("Engineer", company_id=company.id))
    with pytest.raises(ValueError, match="invalid transition"):
        store.append_stage_event(job.id, StageEventDraft("offer", source_key="invalid"))
    assert store.get_company_detail(company.id).jobs[0].current_state == "to_apply"


def test_interview_index_is_idempotent_by_artifact(tmp_path: Path):
    store = CareerStore.open(tmp_path)
    company = store.save_company(CompanyDraft("Acme"))
    job = store.attach_job(JobDraft("Engineer", company_id=company.id), session_dir=tmp_path / "session")
    first = store.add_interview(
        job_id=job.id,
        company_id=company.id,
        kind="full_mock",
        round_label="完整模拟面试",
        occurred_at="2026-07-31T00:00:00+00:00",
        session_dir=tmp_path / "session",
        artifact_path="interview_runs/run-1/transcript.json",
    )
    second = store.add_interview(
        job_id=job.id,
        company_id=company.id,
        kind="full_mock",
        round_label="完整模拟面试",
        occurred_at="2026-07-31T00:01:00+00:00",
        session_dir=tmp_path / "session",
        artifact_path="interview_runs/run-1/transcript.json",
        ai_review_path="interview_runs/run-1/debrief.json",
    )
    assert second.id == first.id
    assert second.ai_review_path == "interview_runs/run-1/debrief.json"
    assert store.connection().execute("SELECT count(*) FROM interviews").fetchone()[0] == 1


def test_legacy_migration_is_auditable_idempotent_and_preserves_files(tmp_path: Path):
    tracker = {"applications": [{"id": "app-1", "job_id": "legacy-1", "company": "Acme", "title": "Engineer", "city": "Shanghai", "url": "https://example.test/job", "source": "manual", "current_state": "applied", "state_history": [{"state": "to_apply", "timestamp": "2026-01-01T00:00:00+00:00", "notes": "created"}, {"state": "applied", "timestamp": "2026-01-02T00:00:00+00:00", "notes": "sent"}]}]}
    path = tmp_path / "tracker.json"; path.write_text(json.dumps(tracker), encoding="utf-8"); before = path.read_bytes()
    session = tmp_path / "sessions" / "unknown"; session.mkdir(parents=True)
    (session / "selected_job.json").write_text(json.dumps({"job_id": "legacy-unknown", "company": "unknown", "title": "Analyst", "location": ""}), encoding="utf-8")
    store = CareerStore.open(tmp_path); migration = LegacyCareerMigration(store)
    first = migration.apply(tmp_path); second = migration.apply(tmp_path)
    assert first.imported >= 2
    assert first.pending == 1
    assert second.skipped >= 1
    assert path.read_bytes() == before
    assert store.connection().execute("SELECT count(*) FROM migration_runs").fetchone()[0] == 2
    assert store.connection().execute("SELECT count(*) FROM jobs").fetchone()[0] == 2
