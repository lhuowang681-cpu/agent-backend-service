from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import EvidenceLevel, StrictModel


SCHEMA_VERSION = "evidence-grounding-v2.0"
DATASET_VERSION = "pilot-ai-draft-v1"


class ExperimentVariant(str, Enum):
    JD_DIRECT = "JD_DIRECT"
    A_CURRENT_V1 = "A"
    B_EXTRACTIVE_PROMPT = "B"
    C_MEMBERSHIP_GUARD = "C"
    C_BLOCK = "C_BLOCK"
    C_BLOCK_ATTR = "C_BLOCK_ATTR"
    C_ATOMIC = "C_ATOMIC"
    D_ENTAILMENT_GATE = "D"
    E_LEXICAL_BASELINE = "E"


class EntailmentRelation(str, Enum):
    ENTAILED = "entailed"
    PARTIAL = "partial"
    RELATED_ONLY = "related_only"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class EvidenceSpan(StrictModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class GoldSpanDraft(StrictModel):
    """A reviewable quote; offsets are optional until a human freezes the label."""

    quote: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_optional_range(self):
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("gold span offsets must both be set or both be omitted")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char <= self.start_char
        ):
            raise ValueError("gold span end_char must be greater than start_char")
        return self


class GoldRequirementUnit(StrictModel):
    requirement_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    gold_level: EvidenceLevel
    gold_spans: list[GoldSpanDraft] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_spans_for_level(self):
        if self.gold_level == EvidenceLevel.NONE and self.gold_spans:
            raise ValueError("NONE gold units cannot contain supporting spans")
        if self.gold_level != EvidenceLevel.NONE and not self.gold_spans:
            raise ValueError("non-NONE gold units require at least one supporting span")
        return self


class DatasetCase(StrictModel):
    case_id: str = Field(min_length=1)
    split: Literal["dev", "heldout"]
    source_group: str = Field(min_length=1)
    source_kind: Literal["synthetic", "sparse", "counterfactual", "real_sanitized"]
    slices: list[str] = Field(min_length=1)
    resume_text: str = Field(min_length=1)
    requirements: list[GoldRequirementUnit] = Field(min_length=1)
    label_provenance: Literal["ai_draft", "human_gold"]
    review_status: Literal["pending_human_review", "frozen"]

    @model_validator(mode="after")
    def validate_case_contract(self):
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement_id must be unique within a case")
        if self.review_status == "frozen" and self.label_provenance != "human_gold":
            raise ValueError("only human_gold labels may be frozen")
        for unit in self.requirements:
            for span in unit.gold_spans:
                if span.start_char is None:
                    if self.resume_text.count(span.quote) != 1:
                        raise ValueError(
                            f"draft quote must occur exactly once: {unit.requirement_id}"
                        )
                    continue
                assert span.end_char is not None
                if self.resume_text[span.start_char : span.end_char] != span.quote:
                    raise ValueError(f"gold span offset mismatch: {unit.requirement_id}")
        return self

    def materialized_gold_spans(self, requirement_id: str) -> list[EvidenceSpan]:
        unit = next(item for item in self.requirements if item.requirement_id == requirement_id)
        spans: list[EvidenceSpan] = []
        for draft in unit.gold_spans:
            start = (
                draft.start_char
                if draft.start_char is not None
                else self.resume_text.index(draft.quote)
            )
            end = draft.end_char if draft.end_char is not None else start + len(draft.quote)
            spans.append(EvidenceSpan(start_char=start, end_char=end, quote=draft.quote))
        return spans


class EvidenceGroundingDataset(StrictModel):
    schema_version: Literal["evidence-grounding-v2.0"] = SCHEMA_VERSION
    dataset_version: str = DATASET_VERSION
    cases: list[DatasetCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset_contract(self):
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id must be unique")
        groups: dict[str, set[str]] = {}
        for case in self.cases:
            groups.setdefault(case.source_group, set()).add(case.split)
        leaked = [group for group, splits in groups.items() if len(splits) > 1]
        if leaked:
            raise ValueError(f"source_group split leakage: {','.join(sorted(leaked))}")
        return self


class EvidencePrediction(StrictModel):
    evidence_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    proposed_level: EvidenceLevel
    spans: list[EvidenceSpan] = Field(default_factory=list)
    risk: str = ""
    abstain: bool = False


class ValidatedEvidencePrediction(EvidencePrediction):
    source_valid: bool
    entailment: EntailmentRelation | None = None
    final_level: EvidenceLevel
    reason_codes: list[str] = Field(default_factory=list)


class SanitizedTrace(StrictModel):
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    schema_valid: bool
    error_code: str | None = None


class PredictionRecord(StrictModel):
    schema_version: Literal["evidence-grounding-v2.0"] = SCHEMA_VERSION
    protocol_hash: str = Field(min_length=64, max_length=64)
    dataset_hash: str = Field(min_length=64, max_length=64)
    case_id: str
    variant: ExperimentVariant
    schema_valid: bool
    fallback_used: Literal[False] = False
    execution_error: str | None = None
    items: list[ValidatedEvidencePrediction] = Field(default_factory=list)
    traces: list[SanitizedTrace] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_failure_semantics(self):
        if self.schema_valid and self.execution_error is not None:
            raise ValueError("schema-valid records cannot contain execution_error")
        if not self.schema_valid and self.items:
            raise ValueError("failed records cannot contain scored prediction items")
        return self


def load_jsonl_dataset(path: Path | str) -> EvidenceGroundingDataset:
    source = Path(path)
    cases = [
        DatasetCase.model_validate(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return EvidenceGroundingDataset(cases=cases)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_protocol_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
