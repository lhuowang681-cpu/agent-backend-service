import pytest
from pydantic import ValidationError

from job_agent.evidence.contracts import (
    EvidenceMappingCandidate,
    ParentRequirement,
    RequirementAnalysis,
    RequirementAnalysisCandidate,
    RequirementAtom,
    SourceSpan,
)


def _span(start: int = 0, end: int = 4) -> SourceSpan:
    return SourceSpan(
        source_id="src_001_raw_jd",
        content_hash="a" * 64,
        start_offset=start,
        end_offset=end,
        exact_quote="abcd"[start:end],
    )


def test_requirement_analysis_accepts_bounded_parent_atom_graph():
    analysis = RequirementAnalysis(
        parents=[
            ParentRequirement(
                requirement_id="r1",
                text="abcd",
                importance="must",
                hard_gate=True,
                source_span=_span(),
                atom_ids=["a1", "a2"],
            )
        ],
        atoms=[
            RequirementAtom(atom_id="a1", parent_requirement_id="r1", text="ab", source_span=_span(0, 2)),
            RequirementAtom(atom_id="a2", parent_requirement_id="r1", text="cd", source_span=_span(2, 4)),
        ],
    )
    assert len(analysis.atoms) == 2


def test_requirement_analysis_rejects_orphan_atom_and_nice_hard_gate():
    with pytest.raises(ValidationError):
        ParentRequirement(
            requirement_id="r1",
            text="abcd",
            importance="nice",
            hard_gate=True,
            source_span=_span(),
            atom_ids=["a1"],
        )
    with pytest.raises(ValidationError):
        RequirementAnalysis(
            parents=[],
            atoms=[RequirementAtom(atom_id="a1", parent_requirement_id="missing", text="ab", source_span=_span())],
        )


def test_zero_requirement_analysis_is_valid():
    value = RequirementAnalysis(warnings=["insufficient_jd_detail"])
    assert value.parents == [] and value.atoms == []


def test_provider_candidate_schemas_exclude_offsets_hashes_and_derived_fields():
    requirement_schema = str(RequirementAnalysisCandidate.model_json_schema())
    mapping_schema = str(EvidenceMappingCandidate.model_json_schema())
    for forbidden in ("start_offset", "end_offset", "content_hash"):
        assert forbidden not in requirement_schema
        assert forbidden not in mapping_schema
    for forbidden in ("match_status", "has_contradiction", "independence_group"):
        assert forbidden not in mapping_schema
