from __future__ import annotations

from job_agent.evidence.contracts import EvidenceLevel, EvidenceV2Error, SourceDocumentRef


_LEVEL_RANK = {
    EvidenceLevel.NONE: 0,
    EvidenceLevel.C0: 0,
    EvidenceLevel.C1: 1,
    EvidenceLevel.C2: 2,
    EvidenceLevel.C3: 3,
}

_C2_ARTIFACTS = {
    "experiment",
    "experiment_record",
    "comparison",
    "comparison_record",
    "bad_case",
    "bad_case_record",
    "design_record",
}

_C3_ARTIFACTS = {
    "metric",
    "metric_record",
    "production_log",
    "rollout",
    "rollout_record",
    "incident_report",
}


class EvidenceLevelPolicy:
    """Conservative deterministic maximum evidence level from source provenance."""

    def maximum_level(self, source: SourceDocumentRef) -> EvidenceLevel:
        artifact_type = (source.artifact_type or "").casefold()
        if artifact_type in _C3_ARTIFACTS:
            return EvidenceLevel.C3
        if artifact_type in _C2_ARTIFACTS:
            return EvidenceLevel.C2
        return EvidenceLevel.C1

    def validate(self, source: SourceDocumentRef, proposed: EvidenceLevel) -> None:
        if _LEVEL_RANK[proposed] > _LEVEL_RANK[self.maximum_level(source)]:
            raise EvidenceV2Error("evidence_level_exceeds_source_cap")
