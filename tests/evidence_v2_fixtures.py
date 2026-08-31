from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from job_agent.evidence.assessment import FitAssessmentPolicy, initial_fit_policy_v2
from job_agent.evidence.contracts import (
    AtomEvidenceResult,
    EvidenceBundle,
    EvidenceLevel,
    EvidenceLink,
    EvidenceMappingV2,
    EvidenceRunManifest,
    ParentRequirement,
    RequirementAnalysis,
    RequirementAtom,
    SourceInput,
)
from job_agent.evidence.repository import EvidenceArtifactRepository
from job_agent.evidence.sources import EvidenceSourceStore, sha256_bytes


def commit_zero_atom_evidence_v2(
    session_dir: Path,
    *,
    run_id: str = "evidence-fixture",
    raw_jd: str = "Agent Engineer role",
    original_resume: str = "Original resume facts",
) -> EvidenceBundle:
    """Commit a valid current v2 bundle without inventing JD requirements."""
    source_store = EvidenceSourceStore(session_dir)
    catalog = source_store.snapshot(
        run_id,
        [
            SourceInput(kind="raw_jd", display_label="JD", text=raw_jd),
            SourceInput(
                kind="original_resume",
                display_label="Original resume",
                text=original_resume,
            ),
        ],
    )
    requirements = RequirementAnalysis(warnings=["insufficient_jd_detail"])
    mapping = EvidenceMappingV2()
    assessment = FitAssessmentPolicy().assess(
        requirements,
        mapping,
        config=initial_fit_policy_v2(),
    )
    fingerprint = sha256_bytes(
        "".join(item.content_hash for item in catalog.documents).encode("utf-8")
    )
    bundle = EvidenceBundle(
        manifest=EvidenceRunManifest(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            jd_parser_version="fixture-v2",
            evidence_prompt_version="fixture-v2",
            fit_policy_version=assessment.fit_policy_version,
            provider="scripted",
            model="fixture",
            source_fingerprint=fingerprint,
        ),
        sources=catalog,
        requirements=requirements,
        mapping=mapping,
        assessment=assessment,
    )
    EvidenceArtifactRepository().commit(session_dir, bundle)
    return bundle


def commit_single_atom_evidence_v2(
    session_dir: Path,
    *,
    run_id: str = "evidence-fixture",
    level: EvidenceLevel = EvidenceLevel.C2,
    support_status: str = "supported",
) -> EvidenceBundle:
    jd_text = "需要独立设计评测"
    resume_text = "独立设计并运行过模型评测"
    source_store = EvidenceSourceStore(session_dir)
    catalog = source_store.snapshot(
        run_id,
        [
            SourceInput(kind="raw_jd", display_label="JD", text=jd_text),
            SourceInput(kind="original_resume", display_label="Original resume", text=resume_text),
        ],
    )
    jd_source, resume_source = catalog.documents
    jd_span = source_store.make_span(catalog, jd_source.source_id, 0, len(jd_text))
    resume_span = source_store.make_span(
        catalog, resume_source.source_id, 0, len(resume_text)
    )
    requirements = RequirementAnalysis(
        parents=[
            ParentRequirement(
                requirement_id="r1",
                text=jd_text,
                importance="must",
                source_span=jd_span,
                atom_ids=["a1"],
            )
        ],
        atoms=[
            RequirementAtom(
                atom_id="a1",
                parent_requirement_id="r1",
                text="独立设计评测",
                source_span=jd_span,
            )
        ],
        warnings=["insufficient_jd_detail"],
    )
    match_status = (
        "unverified_lead"
        if level == EvidenceLevel.C1
        else "partial" if support_status == "partial" else "supported"
    )
    link = EvidenceLink(
        link_id="l1",
        atom_id="a1",
        source_span=resume_span,
        claim=resume_text,
        support_status=support_status,
        level=level,
        confidence=0.8,
        independence_group=source_store.independence_group(resume_span),
        needs_confirmation=level == EvidenceLevel.C1,
    )
    mapping = EvidenceMappingV2(
        atom_results=[
            AtomEvidenceResult(
                atom_id="a1",
                match_status=match_status,
                links=[link],
                rationale="fixture",
            )
        ]
    )
    assessment = FitAssessmentPolicy().assess(
        requirements,
        mapping,
        config=initial_fit_policy_v2(),
    )
    fingerprint = sha256_bytes(
        "".join(item.content_hash for item in catalog.documents).encode("utf-8")
    )
    bundle = EvidenceBundle(
        manifest=EvidenceRunManifest(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            jd_parser_version="fixture-v2",
            evidence_prompt_version="fixture-v2",
            fit_policy_version=assessment.fit_policy_version,
            provider="scripted",
            model="fixture",
            source_fingerprint=fingerprint,
        ),
        sources=catalog,
        requirements=requirements,
        mapping=mapping,
        assessment=assessment,
    )
    EvidenceArtifactRepository().commit(session_dir, bundle)
    return bundle
