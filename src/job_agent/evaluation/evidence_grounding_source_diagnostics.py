from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from job_agent.evaluation.evidence_grounding_atomic import (
    AtomicEvidenceRef,
    SourceLocationError,
    ValidatedAtomicEvidenceGraph,
    locate_atomic_evidence_refs,
    sha256_text,
)
from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    ExperimentVariant,
    PredictionRecord,
)
from job_agent.evaluation.evidence_grounding_runner import (
    BlockMappingOutput,
    ExtractiveMappingOutput,
)
from job_agent.evaluation.evidence_grounding_validators import (
    validate_span_membership,
)
from job_agent.schemas import StrictModel


class SourceDiagnosticSummary(StrictModel):
    variant: str
    case_count: int = Field(ge=0)
    source_reference_count: int = Field(ge=0)
    raw_offset_reference_count: int = Field(ge=0)
    raw_offset_mismatch_count: int = Field(ge=0)
    quote_not_in_source_count: int = Field(ge=0)
    ambiguous_quote_count: int = Field(ge=0)
    unknown_block_count: int = Field(ge=0)
    occurrence_out_of_range_count: int = Field(ge=0)
    canonicalization_success_count: int = Field(ge=0)
    post_guard_invalid_reference_count: int = Field(ge=0)
    quote_not_in_source_rate: float | None = Field(default=None, ge=0, le=1)
    raw_offset_mismatch_rate: float | None = Field(default=None, ge=0, le=1)
    ambiguous_quote_rate: float | None = Field(default=None, ge=0, le=1)
    canonicalization_success_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    post_guard_invalid_span_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _occurrence_count(text: str, quote: str) -> int:
    count = 0
    start = 0
    while quote:
        position = text.find(quote, start)
        if position < 0:
            break
        count += 1
        start = position + 1
    return count


def audit_membership_guard_raw_outputs(
    *,
    cases: list[DatasetCase],
    raw_root: Path | str,
    records: list[PredictionRecord],
) -> SourceDiagnosticSummary:
    raw_root = Path(raw_root)
    records_by_case = {
        record.case_id: record
        for record in records
        if record.variant == ExperimentVariant.C_MEMBERSHIP_GUARD
    }
    source_references = 0
    raw_offset_mismatches = 0
    quote_not_in_source = 0
    ambiguous_quotes = 0
    canonicalization_successes = 0
    post_guard_invalid = 0

    for case in cases:
        raw_path = raw_root / f"{case.case_id}_C.json"
        wrapper = json.loads(raw_path.read_text(encoding="utf-8"))
        if not wrapper.get("schema_valid") or not wrapper.get("raw_output"):
            raise ValueError(f"missing schema-valid C raw output: {case.case_id}")
        parsed = ExtractiveMappingOutput.model_validate_json(wrapper["raw_output"])
        for item in parsed.items:
            for span in item.spans:
                source_references += 1
                occurrences = _occurrence_count(case.resume_text, span.quote)
                raw_matches = validate_span_membership(
                    case.resume_text,
                    start=span.start_char,
                    end=span.end_char,
                    quote=span.quote,
                )
                if occurrences == 0:
                    quote_not_in_source += 1
                elif occurrences > 1:
                    ambiguous_quotes += 1
                if not raw_matches:
                    raw_offset_mismatches += 1
                    if occurrences == 1:
                        canonicalization_successes += 1

        record = records_by_case.get(case.case_id)
        if record is None or not record.schema_valid:
            raise ValueError(f"missing schema-valid C prediction: {case.case_id}")
        post_guard_invalid += sum(
            not validate_span_membership(
                case.resume_text,
                start=span.start_char,
                end=span.end_char,
                quote=span.quote,
            )
            for item in record.items
            for span in item.spans
        )

    return SourceDiagnosticSummary(
        variant=ExperimentVariant.C_MEMBERSHIP_GUARD.value,
        case_count=len(cases),
        source_reference_count=source_references,
        raw_offset_reference_count=source_references,
        raw_offset_mismatch_count=raw_offset_mismatches,
        quote_not_in_source_count=quote_not_in_source,
        ambiguous_quote_count=ambiguous_quotes,
        unknown_block_count=0,
        occurrence_out_of_range_count=0,
        canonicalization_success_count=canonicalization_successes,
        post_guard_invalid_reference_count=post_guard_invalid,
        quote_not_in_source_rate=_ratio(
            quote_not_in_source,
            source_references,
        ),
        raw_offset_mismatch_rate=_ratio(
            raw_offset_mismatches,
            source_references,
        ),
        ambiguous_quote_rate=_ratio(ambiguous_quotes, source_references),
        canonicalization_success_rate=_ratio(
            canonicalization_successes,
            raw_offset_mismatches,
        ),
        post_guard_invalid_span_rate=_ratio(
            post_guard_invalid,
            source_references,
        ),
    )


def audit_atomic_locations(
    graphs: list[ValidatedAtomicEvidenceGraph],
) -> SourceDiagnosticSummary:
    located = [
        evidence
        for graph in graphs
        for evidence in graph.located_evidence_refs
    ]
    source_references = len(located)
    quote_not_in_source = sum(
        evidence.source_error == SourceLocationError.QUOTE_NOT_IN_BLOCK
        for evidence in located
    )
    ambiguous_quotes = sum(
        evidence.source_error == SourceLocationError.AMBIGUOUS_QUOTE
        for evidence in located
    )
    unknown_blocks = sum(
        evidence.source_error == SourceLocationError.UNKNOWN_BLOCK
        for evidence in located
    )
    occurrence_out_of_range = sum(
        evidence.source_error == SourceLocationError.OCCURRENCE_OUT_OF_RANGE
        for evidence in located
    )
    valid = sum(evidence.source_valid for evidence in located)
    invalid = source_references - valid
    return SourceDiagnosticSummary(
        variant=ExperimentVariant.C_ATOMIC.value,
        case_count=len(graphs),
        source_reference_count=source_references,
        raw_offset_reference_count=0,
        raw_offset_mismatch_count=0,
        quote_not_in_source_count=quote_not_in_source,
        ambiguous_quote_count=ambiguous_quotes,
        unknown_block_count=unknown_blocks,
        occurrence_out_of_range_count=occurrence_out_of_range,
        canonicalization_success_count=valid,
        post_guard_invalid_reference_count=invalid,
        quote_not_in_source_rate=_ratio(
            quote_not_in_source,
            source_references,
        ),
        raw_offset_mismatch_rate=None,
        ambiguous_quote_rate=_ratio(ambiguous_quotes, source_references),
        canonicalization_success_rate=_ratio(valid, source_references),
        post_guard_invalid_span_rate=_ratio(invalid, source_references),
    )


def audit_block_locations(
    *,
    cases: list[DatasetCase],
    raw_root: Path | str,
) -> SourceDiagnosticSummary:
    raw_root = Path(raw_root)
    located_all = []
    for case in cases:
        initial = raw_root / f"{case.case_id}_C_BLOCK.json"
        repair = raw_root / f"{case.case_id}_C_BLOCK_repair.json"
        selected = repair if repair.exists() else initial
        wrapper = json.loads(selected.read_text(encoding="utf-8"))
        if not wrapper.get("raw_output"):
            raise ValueError(
                f"missing C_BLOCK raw output: {case.case_id}"
            )
        parsed = BlockMappingOutput.model_validate_json(wrapper["raw_output"])
        refs = [
            AtomicEvidenceRef(
                evidence_ref=f"{item.evidence_id}::ref-{index}",
                source_block_id=evidence.source_block_id,
                quote=evidence.quote,
                occurrence_index=evidence.occurrence_index,
            )
            for item in parsed.items
            for index, evidence in enumerate(item.evidence_refs, start=1)
        ]
        _, located = locate_atomic_evidence_refs(
            case.resume_text,
            refs,
            expected_resume_sha256=sha256_text(case.resume_text),
        )
        located_all.extend(located)

    source_references = len(located_all)
    quote_not_in_source = sum(
        evidence.source_error == SourceLocationError.QUOTE_NOT_IN_BLOCK
        for evidence in located_all
    )
    ambiguous_quotes = sum(
        evidence.source_error == SourceLocationError.AMBIGUOUS_QUOTE
        for evidence in located_all
    )
    unknown_blocks = sum(
        evidence.source_error == SourceLocationError.UNKNOWN_BLOCK
        for evidence in located_all
    )
    occurrence_out_of_range = sum(
        evidence.source_error == SourceLocationError.OCCURRENCE_OUT_OF_RANGE
        for evidence in located_all
    )
    valid = sum(evidence.source_valid for evidence in located_all)
    invalid = source_references - valid
    return SourceDiagnosticSummary(
        variant=ExperimentVariant.C_BLOCK.value,
        case_count=len(cases),
        source_reference_count=source_references,
        raw_offset_reference_count=0,
        raw_offset_mismatch_count=0,
        quote_not_in_source_count=quote_not_in_source,
        ambiguous_quote_count=ambiguous_quotes,
        unknown_block_count=unknown_blocks,
        occurrence_out_of_range_count=occurrence_out_of_range,
        canonicalization_success_count=valid,
        post_guard_invalid_reference_count=invalid,
        quote_not_in_source_rate=_ratio(
            quote_not_in_source,
            source_references,
        ),
        raw_offset_mismatch_rate=None,
        ambiguous_quote_rate=_ratio(ambiguous_quotes, source_references),
        canonicalization_success_rate=_ratio(valid, source_references),
        post_guard_invalid_span_rate=_ratio(invalid, source_references),
    )
