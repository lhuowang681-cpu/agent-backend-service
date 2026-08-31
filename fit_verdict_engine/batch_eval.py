from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from fit_verdict_engine.data_generator import SyntheticRecord, write_jsonl
from fit_verdict_engine.evaluator import evaluate_predictions
from job_agent.nodes.fit_verdict import _extract_json_object, evaluate_fit
from job_agent.schemas import FitVerdictResult, StrictModel, Verdict


class FitVerdictPrediction(StrictModel):
    record_id: str
    gold: Verdict
    pred: Verdict
    raw_model_output: str
    schema_valid: bool
    fallback_used: bool
    error_message: str | None = None


class RawModelOutputRecord(StrictModel):
    record_id: str
    gold: Verdict
    raw_model_output: str
    fallback_verdict: Verdict | None = None


def parse_prediction_record(
    *,
    record_id: str,
    gold: Verdict,
    raw_model_output: str,
    fallback_verdict: Verdict,
) -> FitVerdictPrediction:
    try:
        payload = _extract_json_object(raw_model_output)
        result = FitVerdictResult.model_validate_json(payload)
        return FitVerdictPrediction(
            record_id=record_id,
            gold=gold,
            pred=result.verdict,
            raw_model_output=raw_model_output,
            schema_valid=True,
            fallback_used=False,
        )
    except Exception as exc:
        return FitVerdictPrediction(
            record_id=record_id,
            gold=gold,
            pred=fallback_verdict,
            raw_model_output=raw_model_output,
            schema_valid=False,
            fallback_used=True,
            error_message=str(exc),
        )


def build_predictions_from_raw_outputs(rows: Iterable[RawModelOutputRecord]) -> list[FitVerdictPrediction]:
    return [
        parse_prediction_record(
            record_id=row.record_id,
            gold=row.gold,
            raw_model_output=row.raw_model_output,
            fallback_verdict=row.fallback_verdict or row.gold,
        )
        for row in rows
    ]


def load_raw_output_jsonl(path: Path) -> list[RawModelOutputRecord]:
    rows: list[RawModelOutputRecord] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rows.append(RawModelOutputRecord.model_validate(json.loads(line)))
    return rows


def build_rule_predictions(records: Iterable[SyntheticRecord]) -> list[FitVerdictPrediction]:
    predictions: list[FitVerdictPrediction] = []
    for record in records:
        result = evaluate_fit(record.fit_input)
        raw_model_output = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
        predictions.append(
            parse_prediction_record(
                record_id=record.record_id,
                gold=record.label,
                raw_model_output=raw_model_output,
                fallback_verdict=result.verdict,
            )
        )
    return predictions


def evaluate_prediction_records(records: Iterable[FitVerdictPrediction]) -> dict[str, float | int]:
    record_list = list(records)
    metrics = evaluate_predictions(
        gold=[record.gold for record in record_list],
        pred=[record.pred for record in record_list],
    )
    valid_records = [record for record in record_list if record.schema_valid]
    valid_metrics = evaluate_predictions(
        gold=[record.gold for record in valid_records],
        pred=[record.pred for record in valid_records],
    )
    fallback_count = sum(1 for record in record_list if record.fallback_used)
    count = len(record_list)

    return {
        **metrics,
        "valid_decision_accuracy": valid_metrics["decision_accuracy"],
        "valid_count": valid_metrics["count"],
        "schema_valid_count": len(valid_records),
        "schema_valid_rate": len(valid_records) / count if count else 0.0,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / count if count else 0.0,
    }


def write_prediction_jsonl(path: Path, records: Iterable[FitVerdictPrediction]) -> Path:
    return write_jsonl(path, (record.model_dump(mode="json") for record in records))
