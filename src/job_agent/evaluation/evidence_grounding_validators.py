from __future__ import annotations

import re
import unicodedata

from job_agent.evaluation.evidence_grounding_contracts import (
    EntailmentRelation,
    EvidencePrediction,
    EvidenceSpan,
    ValidatedEvidencePrediction,
)
from job_agent.schemas import EvidenceLevel


_LEVEL_ORDER = {
    EvidenceLevel.NONE: -1,
    EvidenceLevel.C0: 0,
    EvidenceLevel.C1: 1,
    EvidenceLevel.C2: 2,
    EvidenceLevel.C3: 3,
}
_ENTAILMENT_CAP = {
    EntailmentRelation.ENTAILED: EvidenceLevel.C3,
    EntailmentRelation.PARTIAL: EvidenceLevel.C1,
    EntailmentRelation.RELATED_ONLY: EvidenceLevel.C0,
    EntailmentRelation.CONTRADICTED: EvidenceLevel.NONE,
    EntailmentRelation.UNCERTAIN: EvidenceLevel.C0,
}


def normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def validate_span_membership(resume_text: str, *, start: int, end: int, quote: str) -> bool:
    if start < 0 or end <= start or end > len(resume_text):
        return False
    exact = resume_text[start:end]
    if exact == quote:
        return True
    return normalized_text(exact) == normalized_text(quote)


def locate_unique_quote(resume_text: str, quote: str) -> tuple[int, int] | None:
    if not quote or resume_text.count(quote) != 1:
        return None
    start = resume_text.index(quote)
    return start, start + len(quote)


def canonicalize_unique_quote_offsets(
    resume_text: str,
    prediction: EvidencePrediction,
) -> tuple[EvidencePrediction, int]:
    """Correct offsets only when the model's verbatim quote has one source location."""
    repaired = []
    canonicalized = 0
    for span in prediction.spans:
        if validate_span_membership(
            resume_text,
            start=span.start_char,
            end=span.end_char,
            quote=span.quote,
        ):
            repaired.append(span)
            continue
        located = locate_unique_quote(resume_text, span.quote)
        if located is None:
            repaired.append(span)
            continue
        repaired.append(
            EvidenceSpan(
                start_char=located[0],
                end_char=located[1],
                quote=span.quote,
            )
        )
        canonicalized += 1
    return prediction.model_copy(update={"spans": repaired}), canonicalized


def validate_prediction_sources(
    resume_text: str,
    prediction: EvidencePrediction,
    *,
    enforce_source_guard: bool = True,
) -> ValidatedEvidencePrediction:
    reason_codes: list[str] = []
    valid_spans = [
        span
        for span in prediction.spans
        if validate_span_membership(
            resume_text,
            start=span.start_char,
            end=span.end_char,
            quote=span.quote,
        )
    ]
    source_valid = len(valid_spans) == len(prediction.spans)
    if prediction.spans and source_valid:
        reason_codes.append("span_membership_valid")
    elif prediction.spans:
        reason_codes.append("fabricated_span")
    else:
        reason_codes.append("no_span")

    final_level = prediction.proposed_level
    if prediction.abstain:
        final_level = EvidenceLevel.NONE
        reason_codes.append("mapper_abstained")
    if (
        enforce_source_guard
        and final_level not in {EvidenceLevel.NONE, EvidenceLevel.C0}
        and not valid_spans
    ):
        final_level = EvidenceLevel.C0
        reason_codes.append("source_guard_downgrade")
    return ValidatedEvidencePrediction(
        **prediction.model_dump(),
        source_valid=source_valid,
        final_level=final_level,
        reason_codes=reason_codes,
    )


def apply_entailment_cap(
    prediction: ValidatedEvidencePrediction,
    relation: EntailmentRelation,
) -> ValidatedEvidencePrediction:
    cap = _ENTAILMENT_CAP[relation]
    current = prediction.final_level
    final = current if _LEVEL_ORDER[current] <= _LEVEL_ORDER[cap] else cap
    reasons = list(prediction.reason_codes)
    reasons.append(f"entailment_{relation.value}")
    if final != current:
        reasons.append("entailment_downgrade")
    return prediction.model_copy(
        update={
            "entailment": relation,
            "final_level": final,
            "reason_codes": reasons,
        }
    )
