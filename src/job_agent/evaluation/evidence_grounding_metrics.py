from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean

from pydantic import Field

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    ExperimentVariant,
    PredictionRecord,
)
from job_agent.evaluation.evidence_grounding_validators import validate_span_membership
from job_agent.schemas import EvidenceLevel, StrictModel


_LEVEL_ORDER = {
    EvidenceLevel.NONE: -1,
    EvidenceLevel.C0: 0,
    EvidenceLevel.C1: 1,
    EvidenceLevel.C2: 2,
    EvidenceLevel.C3: 3,
}
_LABELS = list(EvidenceLevel)


class VariantMetrics(StrictModel):
    variant: ExperimentVariant
    case_count: int = Field(ge=0)
    scored_unit_count: int = Field(ge=0)
    pending_human_review: bool
    schema_valid_rate: float = Field(ge=0, le=1)
    execution_failure_rate: float = Field(ge=0, le=1)
    exact_requirement_coverage: float = Field(ge=0, le=1)
    exact_level_accuracy: float | None = Field(default=None, ge=0, le=1)
    fabricated_span_rate: float | None = Field(default=None, ge=0, le=1)
    strong_false_positive_rate: float | None = Field(default=None, ge=0, le=1)
    unsafe_upgrade_rate: float | None = Field(default=None, ge=0, le=1)
    macro_f1: float | None = Field(default=None, ge=0, le=1)
    weighted_kappa: float | None = Field(default=None, ge=-1, le=1)
    strong_support_precision: float | None = Field(default=None, ge=0, le=1)
    strong_support_recall: float | None = Field(default=None, ge=0, le=1)
    span_exact_match_f1: float | None = Field(default=None, ge=0, le=1)
    span_character_overlap_f1: float | None = Field(default=None, ge=0, le=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)


class BootstrapDifference(StrictModel):
    baseline: ExperimentVariant
    candidate: ExperimentVariant
    metric: str
    observed_difference: float
    ci_low: float
    ci_high: float
    clusters: int = Field(ge=1)
    resamples: int = Field(ge=1)


class GroundingEvaluationReport(StrictModel):
    result_status: str
    label_provenance: str
    review_status: str
    variants: list[VariantMetrics]
    bootstrap_differences: list[BootstrapDifference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def build_error_slice_rows(
    cases: list[DatasetCase],
    records: list[PredictionRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        gold_by_id = {unit.requirement_id: unit for unit in case.requirements}
        for record in records:
            if record.case_id != case.case_id:
                continue
            if not record.schema_valid:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "slices": "|".join(case.slices),
                        "requirement_id": "",
                        "variant": record.variant.value,
                        "gold_level": "",
                        "predicted_level": "",
                        "error_category": "schema_or_execution_failure",
                    }
                )
                continue
            predicted_ids = set()
            for item in record.items:
                predicted_ids.add(item.requirement_id)
                unit = gold_by_id.get(item.requirement_id)
                if unit is None:
                    category = "unknown_requirement"
                    gold_level = ""
                else:
                    gold_level = unit.gold_level.value
                    if not item.source_valid and item.spans:
                        category = "fabricated_span"
                    elif _LEVEL_ORDER[item.final_level] > _LEVEL_ORDER[unit.gold_level]:
                        category = (
                            "strong_false_positive"
                            if item.final_level in {EvidenceLevel.C2, EvidenceLevel.C3}
                            and unit.gold_level
                            not in {EvidenceLevel.C2, EvidenceLevel.C3}
                            else "unsafe_upgrade"
                        )
                    elif _LEVEL_ORDER[item.final_level] < _LEVEL_ORDER[unit.gold_level]:
                        category = "overconservative_downgrade"
                    else:
                        continue
                rows.append(
                    {
                        "case_id": case.case_id,
                        "slices": "|".join(case.slices),
                        "requirement_id": item.requirement_id,
                        "variant": record.variant.value,
                        "gold_level": gold_level,
                        "predicted_level": item.final_level.value,
                        "error_category": category,
                    }
                )
            for missing in set(gold_by_id) - predicted_ids:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "slices": "|".join(case.slices),
                        "requirement_id": missing,
                        "variant": record.variant.value,
                        "gold_level": gold_by_id[missing].gold_level.value,
                        "predicted_level": "",
                        "error_category": "missing_requirement",
                    }
                )
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _macro_f1(gold: list[EvidenceLevel], pred: list[EvidenceLevel]) -> float:
    scores = []
    for label in _LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        denominator = (2 * tp) + fp + fn
        scores.append((2 * tp / denominator) if denominator else 0.0)
    return mean(scores)


def _weighted_kappa(gold: list[EvidenceLevel], pred: list[EvidenceLevel]) -> float | None:
    if not gold:
        return None
    size = len(_LABELS)
    index = {label: position for position, label in enumerate(_LABELS)}
    observed = [[0 for _ in range(size)] for _ in range(size)]
    gold_count = [0 for _ in range(size)]
    pred_count = [0 for _ in range(size)]
    for gold_level, pred_level in zip(gold, pred):
        row, column = index[gold_level], index[pred_level]
        observed[row][column] += 1
        gold_count[row] += 1
        pred_count[column] += 1
    total = len(gold)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = max((size - 1) ** 2, 1)
    for row in range(size):
        for column in range(size):
            weight = ((row - column) ** 2) / denominator
            observed_disagreement += weight * observed[row][column] / total
            expected_disagreement += (
                weight * gold_count[row] * pred_count[column] / (total * total)
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def _f1(precision_numerator: int, predicted: int, gold: int) -> float | None:
    if predicted == 0 and gold == 0:
        return 1.0
    if predicted == 0 or gold == 0:
        return 0.0
    precision = precision_numerator / predicted
    recall = precision_numerator / gold
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _variant_metrics(
    variant: ExperimentVariant,
    cases: list[DatasetCase],
    records: list[PredictionRecord],
) -> VariantMetrics:
    by_case = {record.case_id: record for record in records if record.variant == variant}
    successful = [record for record in by_case.values() if record.schema_valid]
    expected_units = sum(len(case.requirements) for case in cases)
    coverage_hits = 0
    fabricated = 0
    predicted_spans = 0
    exact_span_hits = 0
    gold_span_count = 0
    overlap_hits = 0
    predicted_characters = 0
    gold_characters = 0
    gold_levels: list[EvidenceLevel] = []
    predicted_levels: list[EvidenceLevel] = []
    reason_codes: Counter[str] = Counter()

    for case in cases:
        record = by_case.get(case.case_id)
        if record is None or not record.schema_valid:
            continue
        item_by_id = {item.requirement_id: item for item in record.items}
        expected_ids = {unit.requirement_id for unit in case.requirements}
        if set(item_by_id) == expected_ids and len(item_by_id) == len(record.items):
            coverage_hits += len(expected_ids)
        for unit in case.requirements:
            item = item_by_id.get(unit.requirement_id)
            if item is None:
                continue
            gold_levels.append(unit.gold_level)
            predicted_levels.append(item.final_level)
            predicted_spans += len(item.spans)
            fabricated += sum(
                not validate_span_membership(
                    case.resume_text,
                    start=span.start_char,
                    end=span.end_char,
                    quote=span.quote,
                )
                for span in item.spans
            )
            gold_spans = case.materialized_gold_spans(unit.requirement_id)
            gold_span_count += len(gold_spans)
            gold_keys = {
                (span.start_char, span.end_char, span.quote) for span in gold_spans
            }
            exact_span_hits += sum(
                (span.start_char, span.end_char, span.quote) in gold_keys
                for span in item.spans
            )
            predicted_positions = {
                position
                for span in item.spans
                for position in range(span.start_char, span.end_char)
                if 0 <= position < len(case.resume_text)
            }
            gold_positions = {
                position
                for span in gold_spans
                for position in range(span.start_char, span.end_char)
            }
            overlap_hits += len(predicted_positions & gold_positions)
            predicted_characters += len(predicted_positions)
            gold_characters += len(gold_positions)
            reason_codes.update(item.reason_codes)

    strong_pred = {
        EvidenceLevel.C2,
        EvidenceLevel.C3,
    }
    non_strong_gold_count = sum(level not in strong_pred for level in gold_levels)
    strong_fp = sum(
        pred in strong_pred and gold not in strong_pred
        for gold, pred in zip(gold_levels, predicted_levels)
    )
    upgrades = sum(
        _LEVEL_ORDER[pred] > _LEVEL_ORDER[gold]
        for gold, pred in zip(gold_levels, predicted_levels)
    )
    strong_tp = sum(
        gold in strong_pred and pred in strong_pred
        for gold, pred in zip(gold_levels, predicted_levels)
    )
    strong_pred_count = sum(pred in strong_pred for pred in predicted_levels)
    strong_gold_count = sum(gold in strong_pred for gold in gold_levels)
    traces = [trace for record in by_case.values() for trace in record.traces]
    pending = any(case.review_status != "frozen" for case in cases)
    return VariantMetrics(
        variant=variant,
        case_count=len(cases),
        scored_unit_count=len(gold_levels),
        pending_human_review=pending,
        schema_valid_rate=len(successful) / len(cases),
        execution_failure_rate=(len(cases) - len(successful)) / len(cases),
        exact_requirement_coverage=coverage_hits / expected_units,
        exact_level_accuracy=_safe_ratio(
            sum(
                gold == predicted
                for gold, predicted in zip(gold_levels, predicted_levels)
            ),
            len(gold_levels),
        ),
        fabricated_span_rate=_safe_ratio(fabricated, predicted_spans),
        strong_false_positive_rate=_safe_ratio(strong_fp, non_strong_gold_count),
        unsafe_upgrade_rate=_safe_ratio(upgrades, len(gold_levels)),
        macro_f1=_macro_f1(gold_levels, predicted_levels) if gold_levels else None,
        weighted_kappa=_weighted_kappa(gold_levels, predicted_levels),
        strong_support_precision=_safe_ratio(strong_tp, strong_pred_count),
        strong_support_recall=_safe_ratio(strong_tp, strong_gold_count),
        span_exact_match_f1=_f1(exact_span_hits, predicted_spans, gold_span_count),
        span_character_overlap_f1=_f1(
            overlap_hits,
            predicted_characters,
            gold_characters,
        ),
        input_tokens=sum(trace.input_tokens or 0 for trace in traces),
        output_tokens=sum(trace.output_tokens or 0 for trace in traces),
        provider_calls=len(traces),
        latency_ms=sum(trace.latency_ms for trace in traces),
        reason_code_counts=dict(sorted(reason_codes.items())),
    )


def _case_metric(
    case: DatasetCase,
    record: PredictionRecord | None,
    *,
    metric: str,
) -> float:
    if record is None or not record.schema_valid:
        return 0.0 if metric == "exact_level_accuracy" else 1.0
    by_id = {item.requirement_id: item for item in record.items}
    values = []
    for unit in case.requirements:
        item = by_id.get(unit.requirement_id)
        if metric == "exact_level_accuracy":
            values.append(item is not None and item.final_level == unit.gold_level)
        elif metric == "unsafe_upgrade_rate":
            values.append(
                item is None
                or _LEVEL_ORDER[item.final_level] > _LEVEL_ORDER[unit.gold_level]
            )
        else:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
    return sum(values) / len(values)


def _cluster_bootstrap_difference(
    *,
    metric: str,
    baseline: ExperimentVariant,
    candidate: ExperimentVariant,
    cases: list[DatasetCase],
    records: list[PredictionRecord],
    resamples: int,
    seed: int,
) -> BootstrapDifference:
    lookup = {(record.case_id, record.variant): record for record in records}
    differences = [
        _case_metric(
            case,
            lookup.get((case.case_id, candidate)),
            metric=metric,
        )
        - _case_metric(
            case,
            lookup.get((case.case_id, baseline)),
            metric=metric,
        )
        for case in cases
    ]
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        samples.append(mean(rng.choice(differences) for _ in differences))
    samples.sort()
    low_index = max(int(0.025 * resamples) - 1, 0)
    high_index = min(int(0.975 * resamples), resamples - 1)
    return BootstrapDifference(
        baseline=baseline,
        candidate=candidate,
        metric=metric,
        observed_difference=mean(differences),
        ci_low=samples[low_index],
        ci_high=samples[high_index],
        clusters=len(cases),
        resamples=resamples,
    )


def evaluate_predictions(
    cases: list[DatasetCase],
    records: list[PredictionRecord],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 20260726,
) -> GroundingEvaluationReport:
    available = {record.variant for record in records}
    variants = [variant for variant in ExperimentVariant if variant in available]
    metrics = [_variant_metrics(variant, cases, records) for variant in variants]
    differences = []
    for candidate in (
        ExperimentVariant.B_EXTRACTIVE_PROMPT,
        ExperimentVariant.C_MEMBERSHIP_GUARD,
        ExperimentVariant.C_BLOCK,
        ExperimentVariant.C_BLOCK_ATTR,
        ExperimentVariant.D_ENTAILMENT_GATE,
        ExperimentVariant.C_ATOMIC,
    ):
        if {
            ExperimentVariant.A_CURRENT_V1,
            candidate,
        }.issubset(available):
            for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
                differences.append(
                    _cluster_bootstrap_difference(
                        metric=metric,
                        baseline=ExperimentVariant.A_CURRENT_V1,
                        candidate=candidate,
                        cases=cases,
                        records=records,
                        resamples=bootstrap_resamples,
                        seed=seed,
                    )
                )
    if {
        ExperimentVariant.JD_DIRECT,
        ExperimentVariant.C_BLOCK,
    }.issubset(available):
        for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
            differences.append(
                _cluster_bootstrap_difference(
                    metric=metric,
                    baseline=ExperimentVariant.JD_DIRECT,
                    candidate=ExperimentVariant.C_BLOCK,
                    cases=cases,
                    records=records,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            )
    if {
        ExperimentVariant.C_MEMBERSHIP_GUARD,
        ExperimentVariant.C_BLOCK,
    }.issubset(available):
        for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
            differences.append(
                _cluster_bootstrap_difference(
                    metric=metric,
                    baseline=ExperimentVariant.C_MEMBERSHIP_GUARD,
                    candidate=ExperimentVariant.C_BLOCK,
                    cases=cases,
                    records=records,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            )
    if {
        ExperimentVariant.C_BLOCK,
        ExperimentVariant.C_BLOCK_ATTR,
    }.issubset(available):
        for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
            differences.append(
                _cluster_bootstrap_difference(
                    metric=metric,
                    baseline=ExperimentVariant.C_BLOCK,
                    candidate=ExperimentVariant.C_BLOCK_ATTR,
                    cases=cases,
                    records=records,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            )
    if {
        ExperimentVariant.C_BLOCK,
        ExperimentVariant.C_ATOMIC,
    }.issubset(available):
        for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
            differences.append(
                _cluster_bootstrap_difference(
                    metric=metric,
                    baseline=ExperimentVariant.C_BLOCK,
                    candidate=ExperimentVariant.C_ATOMIC,
                    cases=cases,
                    records=records,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            )
    if {
        ExperimentVariant.C_MEMBERSHIP_GUARD,
        ExperimentVariant.C_ATOMIC,
    }.issubset(available):
        for metric in ("exact_level_accuracy", "unsafe_upgrade_rate"):
            differences.append(
                _cluster_bootstrap_difference(
                    metric=metric,
                    baseline=ExperimentVariant.C_MEMBERSHIP_GUARD,
                    candidate=ExperimentVariant.C_ATOMIC,
                    cases=cases,
                    records=records,
                    resamples=bootstrap_resamples,
                    seed=seed,
                )
            )
    pending = any(case.review_status != "frozen" for case in cases)
    warnings = []
    if pending:
        warnings.append(
            "All quality metrics use AI-draft labels pending human review; "
            "they are runner diagnostics, not benchmark conclusions."
        )
    return GroundingEvaluationReport(
        result_status="pilot_diagnostic_only" if pending else "frozen_evaluation",
        label_provenance=(
            "mixed" if len({case.label_provenance for case in cases}) > 1
            else cases[0].label_provenance
        ),
        review_status=(
            "mixed" if len({case.review_status for case in cases}) > 1
            else cases[0].review_status
        ),
        variants=metrics,
        bootstrap_differences=differences,
        warnings=warnings,
    )
