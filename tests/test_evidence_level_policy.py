from job_agent.evidence.contracts import EvidenceLevel, EvidenceV2Error, SourceDocumentRef
from job_agent.evidence.level_policy import EvidenceLevelPolicy


def _source(artifact_type: str | None) -> SourceDocumentRef:
    return SourceDocumentRef(
        source_id="s1",
        kind="project_source",
        content_hash="a" * 64,
        snapshot_path="sources/s1.txt",
        display_label="source",
        artifact_type=artifact_type,
    )


def test_evidence_level_policy_caps_sources_conservatively():
    policy = EvidenceLevelPolicy()
    assert policy.maximum_level(_source(None)) == EvidenceLevel.C1
    assert policy.maximum_level(_source("source_code")) == EvidenceLevel.C1
    assert policy.maximum_level(_source("experiment_record")) == EvidenceLevel.C2
    assert policy.maximum_level(_source("production_log")) == EvidenceLevel.C3


def test_evidence_level_policy_accepts_lower_level_and_rejects_upgrade():
    policy = EvidenceLevelPolicy()
    policy.validate(_source("experiment_record"), EvidenceLevel.NONE)
    policy.validate(_source("experiment_record"), EvidenceLevel.C0)
    policy.validate(_source("experiment_record"), EvidenceLevel.C1)
    policy.validate(_source("experiment_record"), EvidenceLevel.C2)
    try:
        policy.validate(_source("experiment_record"), EvidenceLevel.C3)
    except EvidenceV2Error as exc:
        assert exc.error_code == "evidence_level_exceeds_source_cap"
    else:
        raise AssertionError("over-cap evidence must fail closed")
