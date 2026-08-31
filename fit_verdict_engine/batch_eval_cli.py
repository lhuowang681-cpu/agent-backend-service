from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fit_verdict_engine.batch_eval import (
    build_predictions_from_raw_outputs,
    build_rule_predictions,
    evaluate_prediction_records,
    load_raw_output_jsonl,
    write_prediction_jsonl,
)
from fit_verdict_engine.data_generator import generate_synthetic_records, load_eval_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Fit Verdict batch prediction/eval contract.")
    parser.add_argument(
        "--backend",
        choices=["rule"],
        default="rule",
        help="Prediction backend to use. The rule backend is the offline contract sanity loop.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=4,
        help="Synthetic records to generate per verdict class when --eval-input is omitted.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Fixed hand-authored eval JSONL. Overrides generated synthetic records.",
    )
    parser.add_argument(
        "--raw-output-input",
        type=Path,
        default=None,
        help=(
            "Optional JSONL of record_id, gold, raw_model_output, and fallback_verdict. "
            "When set, the CLI parses these raw model outputs instead of generating rule predictions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/fit_verdict_eval/fit_verdict_predictions.jsonl"),
        help="Path to write prediction JSONL rows.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path to write evaluation metrics JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.raw_output_input:
        predictions = build_predictions_from_raw_outputs(load_raw_output_jsonl(args.raw_output_input))
    else:
        records = load_eval_records(args.eval_input) if args.eval_input else generate_synthetic_records(per_class=args.per_class)
        predictions = build_rule_predictions(records)
    metrics = evaluate_prediction_records(predictions)

    write_prediction_jsonl(args.output, predictions)
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(predictions)} predictions to {args.output}")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
