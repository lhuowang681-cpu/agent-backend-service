from pathlib import Path

from job_agent.evidence.contracts import EvidenceLevel
from job_agent.evidence.projections import EvidenceProjectionService
from tests.evidence_v2_fixtures import (
    commit_single_atom_evidence_v2,
    commit_zero_atom_evidence_v2,
)


def test_zero_atom_bundle_has_safe_empty_projections(tmp_path: Path):
    bundle = commit_zero_atom_evidence_v2(tmp_path, run_id="run-1")
    service = EvidenceProjectionService()
    assert service.for_resume(bundle).claims == []
    assert service.for_interview(bundle).atoms == []
    assert service.for_ui(bundle).summary.verdict_label == "信息不足"


def test_c1_is_visible_for_interview_but_excluded_from_resume(tmp_path: Path):
    bundle = commit_single_atom_evidence_v2(
        tmp_path, run_id="run-c1", level=EvidenceLevel.C1
    )
    service = EvidenceProjectionService()
    assert service.for_resume(bundle).claims == []
    assert service.for_interview(bundle).atoms[0].links[0].level == EvidenceLevel.C1
    assert service.for_ui(bundle).summary.needs_confirmation == ["独立设计评测"]


def test_resume_agent_adapter_exposes_only_verified_links(tmp_path: Path):
    bundle = commit_single_atom_evidence_v2(tmp_path, run_id="run-c2")
    inputs = EvidenceProjectionService().for_resume_agent(
        bundle,
        company="Acme",
        title="Agent Engineer",
        raw_jd="需要独立设计评测",
    )
    assert [item.evidence_id for item in inputs.evidence] == ["l1"]
    assert inputs.structured_jd.must_have[0].id == "a1"
    assert not hasattr(inputs, "commit")


def test_partial_claim_keeps_boundary_but_is_not_sent_to_legacy_resume_agent(
    tmp_path: Path,
):
    bundle = commit_single_atom_evidence_v2(
        tmp_path, run_id="run-partial", support_status="partial"
    )
    service = EvidenceProjectionService()
    assert service.for_resume(bundle).claims[0].boundary is not None
    inputs = service.for_resume_agent(
        bundle,
        company="Acme",
        title="Agent Engineer",
        raw_jd="需要独立设计评测",
    )
    assert inputs.evidence == []
