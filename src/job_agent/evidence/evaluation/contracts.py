from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import StrictModel


EvidenceEvalVariantId = Literal[
    "direct_llm_judge",
    "atomized_single_link",
    "full_evidence_v2",
]


class EvidenceEvalVariant(StrictModel):
    variant_id: EvidenceEvalVariantId
    corpus_id: str = Field(min_length=1)
    corpus_hash: str = Field(min_length=64, max_length=64)
    jd_parser_version: str = Field(min_length=1)
    evidence_prompt_version: str = Field(min_length=1)
    fit_policy_version: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_case_tokens: int = Field(gt=0)


class EvidenceEvalCaseInventory(StrictModel):
    case_id: str = Field(min_length=1)
    family_id: str | None = Field(default=None, min_length=1)
    slice_tags: list[str] = Field(min_length=1)
    paraphrase_group: str | None = None


class EvidenceEvalManifest(StrictModel):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    split: Literal["development", "held_out"]
    corpus_id: str = Field(min_length=1)
    corpus_hash: str = Field(min_length=64, max_length=64)
    metric_version: str = Field(min_length=1)
    case_inventory: list[EvidenceEvalCaseInventory] = Field(min_length=1)
    required_slices: list[str] = Field(min_length=1)
    variants: list[EvidenceEvalVariant] = Field(min_length=3, max_length=3)
    held_out_frozen: bool = False
    synthetic_only: bool = True

    @model_validator(mode="after")
    def validate_fair_comparison(self) -> "EvidenceEvalManifest":
        expected = {
            "direct_llm_judge",
            "atomized_single_link",
            "full_evidence_v2",
        }
        if {item.variant_id for item in self.variants} != expected:
            raise ValueError("manifest requires the three registered Evidence v2 variants")
        fairness = {
            (
                item.corpus_id,
                item.corpus_hash,
                item.provider,
                item.model,
                item.temperature,
                item.max_case_tokens,
            )
            for item in self.variants
        }
        if len(fairness) != 1:
            raise ValueError("variants must share corpus, model, temperature, and budget")
        only = self.variants[0]
        if (only.corpus_id, only.corpus_hash) != (self.corpus_id, self.corpus_hash):
            raise ValueError("variant corpus identity must match manifest")
        case_ids = [item.case_id for item in self.case_inventory]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case ids must be unique")
        observed_slices = {
            tag for item in self.case_inventory for tag in item.slice_tags
        }
        if not set(self.required_slices).issubset(observed_slices):
            raise ValueError("required evaluation slices are missing")
        if self.split == "held_out" and not self.held_out_frozen:
            raise ValueError("held-out manifest must be frozen before use")
        return self


class EvidenceEvalObservation(StrictModel):
    case_id: str = Field(min_length=1)
    paraphrase_group: str | None = None
    expected_requirement_ids: list[str] = Field(default_factory=list)
    predicted_requirement_ids: list[str] = Field(default_factory=list)
    expected_link_ids: list[str] = Field(default_factory=list)
    predicted_link_ids: list[str] = Field(default_factory=list)
    expected_contradiction_ids: list[str] = Field(default_factory=list)
    predicted_contradiction_ids: list[str] = Field(default_factory=list)
    expected_partial: dict[str, bool] = Field(default_factory=dict)
    predicted_partial: dict[str, bool] = Field(default_factory=dict)
    expected_span_ids: list[str] = Field(default_factory=list)
    predicted_span_ids: list[str] = Field(default_factory=list)
    predicted_fit_band: Literal[
        "high", "medium", "low", "insufficient_information"
    ]
    resume_claim_supported: list[bool] = Field(default_factory=list)


class PRFMetric(StrictModel):
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class EvidenceEvalMetrics(StrictModel):
    requirement_extraction: PRFMetric
    evidence_linking: PRFMetric
    contradiction_detection: PRFMetric
    partial_accuracy: float = Field(ge=0.0, le=1.0)
    span_exactness: float = Field(ge=0.0, le=1.0)
    verdict_flip_rate: float = Field(ge=0.0, le=1.0)
    paraphrase_stability: float = Field(ge=0.0, le=1.0)
    unsupported_resume_claim_rate: float = Field(ge=0.0, le=1.0)


class EvidenceEvalReport(StrictModel):
    schema_version: Literal[1] = 1
    manifest_id: str
    metric_version: str
    suite: Literal["deterministic"] = "deterministic"
    synthetic_only: Literal[True] = True
    quality_claim_allowed: Literal[False] = False
    case_count: int = Field(gt=0)
    metrics: EvidenceEvalMetrics


class LiveEvaluationApproval(StrictModel):
    approved: Literal[True]
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    case_count: int = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
