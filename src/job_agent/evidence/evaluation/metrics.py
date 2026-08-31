from __future__ import annotations

from itertools import combinations

from job_agent.evidence.evaluation.contracts import (
    EvidenceEvalMetrics,
    EvidenceEvalObservation,
    PRFMetric,
)


def _round(value: float) -> float:
    return round(value, 6)


def _prf(expected: set[str], predicted: set[str]) -> PRFMetric:
    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else 1.0 if not expected else 0.0
    recall = true_positive / (true_positive + false_negative) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRFMetric(
        precision=_round(precision),
        recall=_round(recall),
        f1=_round(f1),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def _scoped(values: list[EvidenceEvalObservation], field: str) -> set[str]:
    return {
        f"{item.case_id}:{value}"
        for item in values
        for value in getattr(item, field)
    }


def _partial_accuracy(values: list[EvidenceEvalObservation]) -> float:
    correct = 0
    total = 0
    for item in values:
        keys = set(item.expected_partial) | set(item.predicted_partial)
        for key in keys:
            total += 1
            correct += item.expected_partial.get(key) == item.predicted_partial.get(key)
    return _round(correct / total) if total else 1.0


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _group_metrics(values: list[EvidenceEvalObservation]) -> tuple[float, float]:
    groups: dict[str, list[EvidenceEvalObservation]] = {}
    for item in values:
        if item.paraphrase_group:
            groups.setdefault(item.paraphrase_group, []).append(item)
    comparable = [items for items in groups.values() if len(items) >= 2]
    if not comparable:
        return 0.0, 1.0
    flipped = sum(len({item.predicted_fit_band for item in items}) > 1 for items in comparable)
    similarities: list[float] = []
    for items in comparable:
        for left, right in combinations(items, 2):
            similarities.append(
                _jaccard(
                    set(left.predicted_requirement_ids),
                    set(right.predicted_requirement_ids),
                )
            )
    return _round(flipped / len(comparable)), _round(sum(similarities) / len(similarities))


def evaluate_observations(
    observations: list[EvidenceEvalObservation],
) -> EvidenceEvalMetrics:
    if not observations:
        raise ValueError("evaluation requires at least one observation")
    case_ids = [item.case_id for item in observations]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation observation case ids must be unique")
    expected_spans = _scoped(observations, "expected_span_ids")
    predicted_spans = _scoped(observations, "predicted_span_ids")
    span_exactness = len(expected_spans & predicted_spans) / len(predicted_spans) if predicted_spans else 1.0 if not expected_spans else 0.0
    claim_flags = [flag for item in observations for flag in item.resume_claim_supported]
    unsupported_rate = (
        sum(not flag for flag in claim_flags) / len(claim_flags) if claim_flags else 0.0
    )
    flip_rate, stability = _group_metrics(observations)
    return EvidenceEvalMetrics(
        requirement_extraction=_prf(
            _scoped(observations, "expected_requirement_ids"),
            _scoped(observations, "predicted_requirement_ids"),
        ),
        evidence_linking=_prf(
            _scoped(observations, "expected_link_ids"),
            _scoped(observations, "predicted_link_ids"),
        ),
        contradiction_detection=_prf(
            _scoped(observations, "expected_contradiction_ids"),
            _scoped(observations, "predicted_contradiction_ids"),
        ),
        partial_accuracy=_partial_accuracy(observations),
        span_exactness=_round(span_exactness),
        verdict_flip_rate=flip_rate,
        paraphrase_stability=stability,
        unsupported_resume_claim_rate=_round(unsupported_rate),
    )
