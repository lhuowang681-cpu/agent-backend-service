from datetime import UTC, datetime
from pathlib import Path

import pytest

from job_agent.evidence.assessment import FitAssessmentPolicy, initial_fit_policy_v2
from job_agent.evidence.contracts import (
    EvidenceBundle,
    EvidenceMappingV2,
    EvidenceRunManifest,
    RequirementAnalysis,
    SourceInput,
    EvidenceV2Error,
)
from job_agent.evidence.repository import EvidenceArtifactRepository
from job_agent.evidence.sources import EvidenceSourceStore, sha256_bytes


def _bundle(tmp_path: Path, run_id: str) -> EvidenceBundle:
    catalog = EvidenceSourceStore(tmp_path).snapshot(
        run_id,
        [SourceInput(kind="raw_jd", display_label="JD", text="general role")],
    )
    analysis = RequirementAnalysis(warnings=["insufficient_jd_detail"])
    assessment = FitAssessmentPolicy().assess(
        analysis, EvidenceMappingV2(), config=initial_fit_policy_v2()
    )
    fingerprint = sha256_bytes("".join(item.content_hash for item in catalog.documents).encode())
    return EvidenceBundle(
        manifest=EvidenceRunManifest(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            jd_parser_version="v2-test",
            evidence_prompt_version="v2-test",
            fit_policy_version=assessment.fit_policy_version,
            provider="scripted",
            model="fixture",
            source_fingerprint=fingerprint,
        ),
        sources=catalog,
        requirements=analysis,
        mapping=EvidenceMappingV2(),
        assessment=assessment,
    )


def test_repository_commits_and_reloads_current_bundle(tmp_path: Path):
    repository = EvidenceArtifactRepository()
    ref = repository.commit(tmp_path, _bundle(tmp_path, "run-1"))
    loaded = repository.load_current(tmp_path)
    assert loaded is not None
    assert loaded.manifest.run_id == ref.run_id
    assert set(loaded.manifest.artifact_hashes) == {
        "sources.json",
        "requirements.json",
        "mapping.json",
        "assessment.json",
    }


def test_repository_keeps_legacy_artifacts_byte_identical(tmp_path: Path):
    legacy = {
        "01_jd_structured.json": b'{"must_have": []}',
        "02_evidence_mapping.json": b"[]",
        "03_fit_verdict.json": b'{"verdict": "weak fit"}',
    }
    for name, value in legacy.items():
        (tmp_path / name).write_bytes(value)
    before = {name: sha256_bytes(value) for name, value in legacy.items()}
    repository = EvidenceArtifactRepository()
    view = repository.load_legacy_view(tmp_path)
    repository.commit(tmp_path, _bundle(tmp_path, "run-1"))
    after = {name: sha256_bytes((tmp_path / name).read_bytes()) for name in legacy}
    assert view is not None and view.provenance_unverified is True
    assert before == after


def test_pointer_failure_preserves_previous_current(tmp_path: Path, monkeypatch):
    import job_agent.evidence.repository as repository_module

    repository = EvidenceArtifactRepository()
    repository.commit(tmp_path, _bundle(tmp_path, "run-1"))
    second = _bundle(tmp_path, "run-2")
    original_atomic = repository_module._atomic_bytes

    def fail_pointer(path: Path, value: bytes) -> None:
        if path.name == "evidence_current.json":
            raise OSError("pointer unavailable")
        original_atomic(path, value)

    monkeypatch.setattr(repository_module, "_atomic_bytes", fail_pointer)
    with pytest.raises(OSError, match="pointer unavailable"):
        repository.commit(tmp_path, second)
    monkeypatch.setattr(repository_module, "_atomic_bytes", original_atomic)
    assert repository.load_current(tmp_path).manifest.run_id == "run-1"


def test_invalid_new_source_preserves_previous_current(tmp_path: Path):
    repository = EvidenceArtifactRepository()
    repository.commit(tmp_path, _bundle(tmp_path, "run-1"))
    second = _bundle(tmp_path, "run-2")
    source = tmp_path / second.sources.documents[0].snapshot_path
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(EvidenceV2Error, match="stale_source_snapshot"):
        repository.commit(tmp_path, second)
    assert repository.load_current(tmp_path).manifest.run_id == "run-1"
