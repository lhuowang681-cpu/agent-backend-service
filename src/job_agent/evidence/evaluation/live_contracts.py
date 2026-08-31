from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from job_agent.evidence.evaluation.contracts import (
    EvidenceEvalMetrics,
    EvidenceEvalVariantId,
)
from job_agent.schemas import EvidenceLevel, StrictModel


class LiveEvalSource(StrictModel):
    source_key: str = Field(min_length=1)
    kind: Literal["user_artifact", "project_source"]
    display_label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    artifact_type: str | None = None


class GoldAtom(StrictModel):
    atom_id: str = Field(min_length=1)
    jd_quote: str = Field(min_length=1)
    importance: Literal["must", "should", "nice"]
    hard_gate: bool = False


class GoldLink(StrictModel):
    link_id: str = Field(min_length=1)
    atom_id: str = Field(min_length=1)
    resume_quote: str = Field(min_length=1)
    source_key: str = "resume"
    support_status: Literal["supported", "partial", "unsupported", "contradictory"]
    level: EvidenceLevel


class LiveEvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    family_id: str | None = Field(default=None, min_length=1)
    slice_tags: list[str] = Field(min_length=1)
    paraphrase_group: str | None = None
    jd_text: str = Field(min_length=1)
    resume_text: str = Field(min_length=1)
    evidence_sources: list[LiveEvalSource] = Field(default_factory=list)
    gold_atoms: list[GoldAtom] = Field(default_factory=list)
    gold_links: list[GoldLink] = Field(default_factory=list)
    expected_fit_band: Literal["high", "medium", "low", "insufficient_information"]

    @model_validator(mode="after")
    def validate_gold(self) -> "LiveEvalCase":
        atom_ids = [item.atom_id for item in self.gold_atoms]
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("gold atom ids must be unique")
        for atom in self.gold_atoms:
            if self.jd_text.count(atom.jd_quote) != 1:
                raise ValueError("gold JD quote must occur exactly once")
        source_texts = {"resume": self.resume_text}
        for source in self.evidence_sources:
            if source.source_key in source_texts:
                raise ValueError("live evaluation source keys must be unique")
            source_texts[source.source_key] = source.text
        for link in self.gold_links:
            if link.atom_id not in atom_ids:
                raise ValueError("gold link references unknown atom")
            if link.source_key not in source_texts:
                raise ValueError("gold link references unknown evidence source")
            if source_texts[link.source_key].count(link.resume_quote) != 1:
                raise ValueError("gold evidence quote must occur exactly once in its source")
        return self


class LiveEvalCorpus(StrictModel):
    schema_version: Literal[1] = 1
    corpus_id: str = Field(min_length=1)
    cases: list[LiveEvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> "LiveEvalCorpus":
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("live evaluation case ids must be unique")
        return self


class PredictedSpan(StrictModel):
    source: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    exact_quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_range(self) -> "PredictedSpan":
        if self.end_offset <= self.start_offset:
            raise ValueError("predicted span end must follow start")
        return self


class PredictedRequirement(StrictModel):
    atom_key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    importance: Literal["must", "should", "nice"]
    hard_gate: bool = False
    source_span: PredictedSpan


class PredictedEvidenceLink(StrictModel):
    link_key: str = Field(min_length=1)
    atom_key: str = Field(min_length=1)
    source_span: PredictedSpan
    support_status: Literal["supported", "partial", "unsupported", "contradictory"]
    level: EvidenceLevel


class PredictedResumeClaim(StrictModel):
    atom_key: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_span: PredictedSpan


class LiveVariantPrediction(StrictModel):
    requirements: list[PredictedRequirement] = Field(default_factory=list)
    evidence_links: list[PredictedEvidenceLink] = Field(default_factory=list)
    estimated_fit_band: Literal["high", "medium", "low", "insufficient_information"]
    resume_claims: list[PredictedResumeClaim] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_refs(self) -> "LiveVariantPrediction":
        atom_keys = [item.atom_key for item in self.requirements]
        if len(atom_keys) != len(set(atom_keys)):
            raise ValueError("predicted atom keys must be unique")
        link_keys = [item.link_key for item in self.evidence_links]
        if len(link_keys) != len(set(link_keys)):
            raise ValueError("predicted evidence link keys must be unique")
        # Unknown semantic references remain parseable so the evaluator can count
        # them as linking/claim errors instead of discarding the entire row.
        return self


class LiveEvalRow(StrictModel):
    case_id: str
    variant_id: EvidenceEvalVariantId
    status: Literal["passed", "failed"]
    prediction: LiveVariantPrediction | None = None
    error_code: str | None = None
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    # Budget accounting survives interruption even when the provider returned no
    # usage metadata and the runner had to reserve the worst-case allowance.
    budget_tokens: int = Field(default=0, ge=0)
    normalization_warnings: list[str] = Field(default_factory=list)


class LiveVariantMetrics(StrictModel):
    variant_id: EvidenceEvalVariantId
    passed_rows: int = Field(ge=0)
    failed_rows: int = Field(ge=0)
    fit_band_accuracy: float = Field(ge=0.0, le=1.0)
    metrics: EvidenceEvalMetrics


class LiveEvaluationReport(StrictModel):
    schema_version: Literal[1] = 1
    manifest_id: str
    manifest_hash: str = ""
    corpus_id: str
    provider: str
    model: str
    temperature: float
    max_total_tokens: int
    observed_input_tokens: int = Field(ge=0)
    observed_output_tokens: int = Field(ge=0)
    rows: list[LiveEvalRow]
    variants: list[LiveVariantMetrics]
    split: Literal["development", "held_out"] = "development"
    resume_quality_claim_allowed: Literal[False] = False
