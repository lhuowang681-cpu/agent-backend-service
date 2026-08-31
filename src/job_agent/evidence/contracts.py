from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import EvidenceLevel, StrictModel


EvidenceImportance = Literal["must", "should", "nice"]
EvidenceSupportStatus = Literal["supported", "partial", "unsupported", "contradictory"]
AtomMatchStatus = Literal["supported", "partial", "unverified_lead", "unsupported", "gap"]
SourceKind = Literal["raw_jd", "original_resume", "user_artifact", "project_source"]


class EvidenceV2Error(ValueError):
    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(f"evidence v2 error: {error_code}")


class SourceInput(StrictModel):
    kind: SourceKind
    display_label: str = Field(min_length=1, max_length=200)
    text: str | None = None
    path: str | None = None
    artifact_type: str | None = None

    @model_validator(mode="after")
    def exactly_one_payload(self) -> "SourceInput":
        if (self.text is None) == (self.path is None):
            raise ValueError("source input requires exactly one of text or path")
        return self


class SourceDocumentRef(StrictModel):
    source_id: str = Field(min_length=1)
    kind: SourceKind
    version: int = Field(default=1, ge=1)
    raw_hash: str | None = None
    content_hash: str = Field(min_length=64, max_length=64)
    snapshot_path: str = Field(min_length=1)
    display_label: str = Field(min_length=1)
    artifact_type: str | None = None


class SourceQuoteLocator(StrictModel):
    """Semantic provider output resolved against an immutable source in Python."""

    source_id: str = Field(min_length=1)
    exact_quote: str = Field(min_length=1)
    prefix_anchor: str | None = Field(default=None, min_length=1)
    suffix_anchor: str | None = Field(default=None, min_length=1)


class SourceSpan(StrictModel):
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    exact_quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_range(self) -> "SourceSpan":
        if self.end_offset <= self.start_offset:
            raise ValueError("source span end must follow start")
        return self


class SourceCatalog(StrictModel):
    documents: list[SourceDocumentRef] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> "SourceCatalog":
        ids = [item.source_id for item in self.documents]
        paths = [item.snapshot_path for item in self.documents]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("source ids and snapshot paths must be unique")
        return self


class ParentRequirement(StrictModel):
    requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    importance: EvidenceImportance
    hard_gate: bool = False
    source_span: SourceSpan
    atom_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_hard_gate(self) -> "ParentRequirement":
        if self.importance == "nice" and self.hard_gate:
            raise ValueError("nice requirements cannot be hard gates")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ValueError("parent atom ids must be unique")
        return self


class RequirementAtom(StrictModel):
    atom_id: str = Field(min_length=1)
    parent_requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_span: SourceSpan
    verification_signals: list[str] = Field(default_factory=list)


class RequirementAnalysis(StrictModel):
    schema_version: Literal[2] = 2
    parents: list[ParentRequirement] = Field(default_factory=list)
    atoms: list[RequirementAtom] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "RequirementAnalysis":
        parent_ids = [item.requirement_id for item in self.parents]
        atom_ids = [item.atom_id for item in self.atoms]
        if len(parent_ids) != len(set(parent_ids)) or len(atom_ids) != len(set(atom_ids)):
            raise ValueError("requirement and atom ids must be unique")
        if not self.parents and self.atoms:
            raise ValueError("atoms require parents")
        atoms_by_parent: dict[str, list[RequirementAtom]] = {key: [] for key in parent_ids}
        for atom in self.atoms:
            if atom.parent_requirement_id not in atoms_by_parent:
                raise ValueError("atom references unknown parent")
            atoms_by_parent[atom.parent_requirement_id].append(atom)
        for parent in self.parents:
            observed = {atom.atom_id for atom in atoms_by_parent[parent.requirement_id]}
            if observed != set(parent.atom_ids):
                raise ValueError("parent atom coverage mismatch")
            for atom in atoms_by_parent[parent.requirement_id]:
                if atom.source_span.source_id != parent.source_span.source_id:
                    raise ValueError("atom source must match parent source")
                if not (
                    parent.source_span.start_offset <= atom.source_span.start_offset
                    and atom.source_span.end_offset <= parent.source_span.end_offset
                ):
                    raise ValueError("atom span must be contained by parent span")
        return self


class ParentRequirementCandidate(StrictModel):
    requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    importance: EvidenceImportance
    hard_gate: bool = False
    source_locator: SourceQuoteLocator
    atom_ids: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_shape(self) -> "ParentRequirementCandidate":
        if self.importance == "nice" and self.hard_gate:
            raise ValueError("nice requirements cannot be hard gates")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ValueError("parent atom ids must be unique")
        return self


class RequirementAtomCandidate(StrictModel):
    atom_id: str = Field(min_length=1)
    parent_requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_locator: SourceQuoteLocator
    verification_signals: list[str] = Field(default_factory=list)


class RequirementAnalysisCandidate(StrictModel):
    parents: list[ParentRequirementCandidate] = Field(default_factory=list)
    atoms: list[RequirementAtomCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "RequirementAnalysisCandidate":
        parent_ids = [item.requirement_id for item in self.parents]
        atom_ids = [item.atom_id for item in self.atoms]
        if len(parent_ids) != len(set(parent_ids)) or len(atom_ids) != len(set(atom_ids)):
            raise ValueError("requirement and atom ids must be unique")
        if not self.parents and self.atoms:
            raise ValueError("atoms require parents")
        observed_by_parent: dict[str, set[str]] = {key: set() for key in parent_ids}
        for atom in self.atoms:
            if atom.parent_requirement_id not in observed_by_parent:
                raise ValueError("atom references unknown parent")
            observed_by_parent[atom.parent_requirement_id].add(atom.atom_id)
        for parent in self.parents:
            if observed_by_parent[parent.requirement_id] != set(parent.atom_ids):
                raise ValueError("parent atom coverage mismatch")
        return self


class EvidenceLink(StrictModel):
    link_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    source_span: SourceSpan
    claim: str = Field(min_length=1)
    support_status: EvidenceSupportStatus
    level: EvidenceLevel
    confidence: float = Field(ge=0.0, le=1.0)
    independence_group: str = ""
    needs_confirmation: bool = False


class AtomEvidenceResult(StrictModel):
    atom_id: str = Field(min_length=1)
    match_status: AtomMatchStatus
    has_contradiction: bool = False
    links: list[EvidenceLink] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> "AtomEvidenceResult":
        link_ids = [item.link_id for item in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("evidence link ids must be unique within an atom")
        if any(item.atom_id != self.atom_id for item in self.links):
            raise ValueError("evidence link atom mismatch")
        if self.match_status == "gap" and self.links:
            raise ValueError("gap atom cannot contain links")
        return self


class EvidenceMappingV2(StrictModel):
    schema_version: Literal[2] = 2
    atom_results: list[AtomEvidenceResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_atom_results(self) -> "EvidenceMappingV2":
        ids = [item.atom_id for item in self.atom_results]
        if len(ids) != len(set(ids)):
            raise ValueError("atom result ids must be unique")
        return self


class EvidenceLinkCandidate(StrictModel):
    link_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    source_locator: SourceQuoteLocator
    claim: str = Field(min_length=1)
    support_status: EvidenceSupportStatus
    level: EvidenceLevel
    confidence: float = Field(ge=0.0, le=1.0)
    needs_confirmation: bool = False


class AtomEvidenceCandidate(StrictModel):
    atom_id: str = Field(min_length=1)
    links: list[EvidenceLinkCandidate] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> "AtomEvidenceCandidate":
        link_ids = [item.link_id for item in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("evidence link ids must be unique within an atom")
        if any(item.atom_id != self.atom_id for item in self.links):
            raise ValueError("evidence link atom mismatch")
        return self


class EvidenceMappingCandidate(StrictModel):
    atom_results: list[AtomEvidenceCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_atom_results(self) -> "EvidenceMappingCandidate":
        ids = [item.atom_id for item in self.atom_results]
        if len(ids) != len(set(ids)):
            raise ValueError("atom result ids must be unique")
        return self


class EvidenceDisplaySummary(StrictModel):
    verdict_label: Literal["较匹配", "部分匹配", "信息不足", "不匹配"]
    confidence_label: Literal["高", "中", "低"]
    strengths: list[str] = Field(default_factory=list)
    needs_confirmation: list[str] = Field(default_factory=list)
    critical_gaps: list[str] = Field(default_factory=list)


class FitPolicyConfig(StrictModel):
    version: str = Field(min_length=1)
    importance_weights: dict[str, float]
    fit_band_thresholds: dict[str, float]
    confidence_parameters: dict[str, float]

    @model_validator(mode="after")
    def validate_policy(self) -> "FitPolicyConfig":
        if set(self.importance_weights) != {"must", "should", "nice"}:
            raise ValueError("importance weights must define must/should/nice")
        if any(value <= 0 for value in self.importance_weights.values()):
            raise ValueError("importance weights must be positive")
        if not (
            self.importance_weights["nice"] < self.importance_weights["should"]
            <= self.importance_weights["must"]
        ):
            raise ValueError("importance weights must preserve must/should/nice order")
        for key in ("high", "medium"):
            if key not in self.fit_band_thresholds:
                raise ValueError("fit band thresholds require high and medium")
        return self


class EvidenceAssessment(StrictModel):
    schema_version: Literal[2] = 2
    fit_policy_version: str = Field(min_length=1)
    estimated_fit_band: Literal["high", "medium", "low", "insufficient_information"]
    evidence_confidence: Literal["high", "medium", "low"]
    verified_coverage: float = Field(ge=0.0, le=1.0)
    hard_blocker_atom_ids: list[str] = Field(default_factory=list)
    unresolved_atom_ids: list[str] = Field(default_factory=list)
    uncertainty_reasons: list[str] = Field(default_factory=list)
    needs_review: bool
    display_summary: EvidenceDisplaySummary


class EvidenceRunManifest(StrictModel):
    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    jd_parser_version: str = Field(min_length=1)
    evidence_prompt_version: str = Field(min_length=1)
    fit_policy_version: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=64, max_length=64)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)


class EvidenceBundle(StrictModel):
    manifest: EvidenceRunManifest
    sources: SourceCatalog
    requirements: RequirementAnalysis
    mapping: EvidenceMappingV2
    assessment: EvidenceAssessment


class EvidenceRunRef(StrictModel):
    run_id: str
    relative_path: str
    manifest_hash: str


class LegacyEvidenceView(StrictModel):
    evidence_schema_version: Literal["v1"] = "v1"
    provenance_unverified: Literal[True] = True
    structured_jd: dict | None = None
    evidence: list[dict] = Field(default_factory=list)
    fit_verdict: dict | None = None


class ResumeEvidenceClaim(StrictModel):
    atom_id: str
    claim: str
    level: EvidenceLevel
    link_ids: list[str]
    boundary: str | None = None


class ResumeEvidenceView(StrictModel):
    run_id: str
    claims: list[ResumeEvidenceClaim] = Field(default_factory=list)


class InterviewEvidenceAtom(StrictModel):
    atom_id: str
    requirement_text: str
    atom_text: str
    importance: EvidenceImportance
    match_status: AtomMatchStatus
    has_contradiction: bool
    links: list[EvidenceLink] = Field(default_factory=list)


class InterviewEvidenceView(StrictModel):
    run_id: str
    atoms: list[InterviewEvidenceAtom] = Field(default_factory=list)


class EvidenceDisplayItem(StrictModel):
    atom_id: str
    label: str
    jd_quote: str
    evidence_quotes: list[str] = Field(default_factory=list)


class EvidenceDisplayView(StrictModel):
    run_id: str
    summary: EvidenceDisplaySummary
    items: list[EvidenceDisplayItem] = Field(default_factory=list)
