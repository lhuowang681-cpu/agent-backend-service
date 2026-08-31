from job_agent.evidence.assessment import FitAssessmentPolicy, initial_fit_policy_v2
from job_agent.evidence.contracts import (
    AtomEvidenceResult,
    EvidenceLevel,
    EvidenceLink,
    EvidenceMappingV2,
    ParentRequirement,
    RequirementAnalysis,
    RequirementAtom,
    SourceSpan,
)


def _span(source: str, quote: str = "proof") -> SourceSpan:
    return SourceSpan(
        source_id=source,
        content_hash="a" * 64,
        start_offset=0,
        end_offset=len(quote),
        exact_quote=quote,
    )


def _analysis(atom_count: int, *, hard_gate: bool = False) -> RequirementAnalysis:
    parent_span = _span("src_001_raw_jd", "requirement")
    ids = [f"a{index}" for index in range(atom_count)]
    return RequirementAnalysis(
        parents=[
            ParentRequirement(
                requirement_id="r1",
                text="requirement",
                importance="must",
                hard_gate=hard_gate,
                source_span=parent_span,
                atom_ids=ids,
            )
        ] if ids else [],
        atoms=[
            RequirementAtom(
                atom_id=atom_id,
                parent_requirement_id="r1",
                text=atom_id,
                source_span=parent_span,
            )
            for atom_id in ids
        ],
    )


def _result(atom_id: str, level: EvidenceLevel, status: str = "supported") -> AtomEvidenceResult:
    link = EvidenceLink(
        link_id=f"l-{atom_id}",
        atom_id=atom_id,
        source_span=_span("src_002_original_resume"),
        claim="proof",
        support_status=status,
        level=level,
        confidence=0.8,
    )
    return AtomEvidenceResult(
        atom_id=atom_id,
        match_status="supported" if level in {EvidenceLevel.C2, EvidenceLevel.C3} else "unverified_lead",
        links=[link],
        rationale="mapped",
    )


def test_c1_can_estimate_match_but_never_increases_verified_coverage():
    analysis = _analysis(3)
    mapping = EvidenceMappingV2(atom_results=[_result(f"a{i}", EvidenceLevel.C1) for i in range(3)])
    assessment = FitAssessmentPolicy().assess(analysis, mapping, config=initial_fit_policy_v2())
    assert assessment.estimated_fit_band == "high"
    assert assessment.verified_coverage == 0
    assert assessment.evidence_confidence == "low"


def test_sparse_jd_and_partial_hard_gate_require_review():
    analysis = _analysis(2, hard_gate=True)
    mapping = EvidenceMappingV2(
        atom_results=[_result("a0", EvidenceLevel.C2), _result("a1", EvidenceLevel.C2, "partial")]
    )
    assessment = FitAssessmentPolicy().assess(analysis, mapping, config=initial_fit_policy_v2())
    assert assessment.display_summary.verdict_label == "信息不足"
    assert assessment.hard_blocker_atom_ids == ["a1"]
    assert "insufficient_jd_detail" in assessment.uncertainty_reasons


def test_zero_atoms_returns_information_insufficient():
    assessment = FitAssessmentPolicy().assess(
        _analysis(0), EvidenceMappingV2(), config=initial_fit_policy_v2()
    )
    assert assessment.estimated_fit_band == "insufficient_information"
    assert assessment.needs_review is True


def test_canonicalized_offsets_force_review():
    analysis = _analysis(3).model_copy(
        update={"warnings": ["source_span_offsets_canonicalized"]}
    )
    mapping = EvidenceMappingV2(
        atom_results=[_result(f"a{i}", EvidenceLevel.C2) for i in range(3)]
    )
    assessment = FitAssessmentPolicy().assess(
        analysis, mapping, config=initial_fit_policy_v2()
    )
    assert assessment.needs_review is True
    assert "source_span_offsets_canonicalized" in assessment.uncertainty_reasons
