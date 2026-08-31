from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from job_agent.career.store import CareerStore
from job_agent.evidence.projections import EvidenceProjectionService
from job_agent.evidence.repository import EvidenceArtifactRepository
from job_agent.interview.contracts import (
    InterviewContextSnapshot,
    InterviewSourceRef,
    ProjectDossier,
)


class InterviewContextError(ValueError):
    pass


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_stable(path: Path) -> tuple[bytes, str]:
    try:
        first = path.read_bytes()
        digest = _hash_bytes(first)
        if path.read_bytes() != first:
            raise InterviewContextError(f"source changed during snapshot: {path.name}")
        return first, digest
    except OSError as exc:
        raise InterviewContextError(f"cannot read interview source: {path.name}") from exc


def _json_source(path: Path) -> tuple[object, str]:
    raw, digest = _read_stable(path)
    try:
        return json.loads(raw.decode("utf-8")), digest
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InterviewContextError(f"invalid interview JSON source: {path.name}") from exc


class InterviewContextBuilder:
    def __init__(self, store: CareerStore) -> None:
        self.store = store

    @staticmethod
    def for_project_grill(*, dossier: ProjectDossier, run_id: str) -> InterviewContextSnapshot:
        if not dossier.ready_for_interview:
            raise InterviewContextError("project dossier is not confirmed or is stale")
        refs = [
            InterviewSourceRef(
                ref_id=f"project_fact:{fact.fact_id}",
                kind="project_fact",
                artifact_path=f"projects/{dossier.project_id}/revisions/{dossier.revision}.json",
                content_hash=_hash_bytes(fact.model_dump_json().encode("utf-8")),
            )
            for fact in dossier.facts
            if fact.status in {"verified", "confirmed"}
        ]
        return InterviewContextSnapshot(
            run_id=run_id,
            kind="project_grill",
            created_at=datetime.now(UTC).isoformat(),
            source_refs=refs,
            project_dossier=dossier.model_copy(deep=True),
        )

    def for_full_mock(
        self,
        *,
        job_id: str,
        run_id: str,
        dossier: ProjectDossier | None = None,
    ) -> InterviewContextSnapshot:
        try:
            job = self.store.get_job(job_id)
        except KeyError as exc:
            raise InterviewContextError("interview job does not exist") from exc
        if not job.session_dir:
            raise InterviewContextError("job has no linked session")
        session_dir = Path(job.session_dir)
        resume_path = session_dir / "06_targeted_resume.md"
        evidence_bundle = EvidenceArtifactRepository().load_current(session_dir)
        if evidence_bundle is None:
            raise InterviewContextError("job requires a current Evidence v2 run before a new full mock")
        evidence_view = EvidenceProjectionService().for_interview(evidence_bundle)
        jd = evidence_bundle.requirements.model_dump(mode="json")
        evidence = [item.model_dump(mode="json") for item in evidence_view.atoms]
        run_dir = session_dir / "evidence_runs" / evidence_bundle.manifest.run_id
        requirements_path = run_dir / "requirements.json"
        mapping_path = run_dir / "mapping.json"
        _, jd_hash = _read_stable(requirements_path)
        _, evidence_hash = _read_stable(mapping_path)
        resume_raw, resume_hash = _read_stable(resume_path)
        try:
            resume_text = resume_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InterviewContextError("targeted resume is not UTF-8") from exc
        refs = [
            InterviewSourceRef(
                ref_id="jd:structured",
                kind="jd",
                artifact_path=f"evidence_runs/{evidence_bundle.manifest.run_id}/requirements.json",
                content_hash=jd_hash,
            ),
            InterviewSourceRef(
                ref_id="resume:targeted",
                kind="resume",
                artifact_path="06_targeted_resume.md",
                content_hash=resume_hash,
            ),
            InterviewSourceRef(
                ref_id="jd_evidence:mapping",
                kind="jd_evidence",
                artifact_path=f"evidence_runs/{evidence_bundle.manifest.run_id}/mapping.json",
                content_hash=evidence_hash,
            ),
        ]
        if dossier is not None:
            if not dossier.ready_for_interview:
                raise InterviewContextError("project dossier is not confirmed or is stale")
            refs.extend(
                InterviewSourceRef(
                    ref_id=f"project_fact:{fact.fact_id}",
                    kind="project_fact",
                    artifact_path=f"projects/{dossier.project_id}/revisions/{dossier.revision}.json",
                    content_hash=_hash_bytes(fact.model_dump_json().encode("utf-8")),
                )
                for fact in dossier.facts
                if fact.status in {"verified", "confirmed"}
            )
        try:
            return InterviewContextSnapshot(
                run_id=run_id,
                kind="full_mock",
                job_id=job.id,
                company_id=job.company_id,
                created_at=datetime.now(UTC).isoformat(),
                source_refs=refs,
                jd=jd,
                resume_text=resume_text,
                jd_evidence=evidence,
                project_dossier=dossier.model_copy(deep=True) if dossier else None,
            )
        except ValidationError as exc:
            raise InterviewContextError("job-linked interview context is incomplete") from exc
