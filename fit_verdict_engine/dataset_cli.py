from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fit_verdict_engine.data_generator import generate_synthetic_records, write_jsonl
from fit_verdict_engine.prompt_builder import EXPOSURES, build_training_rows
from fit_verdict_engine.v2_data import generate_v2_training_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic Fit Verdict SFT JSONL data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/fit_verdict_train.jsonl"),
        help="Path to write JSONL training rows.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=4,
        help="V1 synthetic records to generate per verdict class before exposure expansion.",
    )
    parser.add_argument(
        "--dataset-version",
        choices=["v1", "v2"],
        default="v1",
        help="Dataset generator version. V2 is a fixed 96-record candidate-training corpus.",
    )
    parser.add_argument(
        "--exposures",
        nargs="+",
        choices=EXPOSURES,
        default=list(EXPOSURES),
        help="Prompt exposure variants to include.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = generate_v2_training_records() if args.dataset_version == "v2" else generate_synthetic_records(args.per_class)
    rows = build_training_rows(records, exposures=tuple(args.exposures))
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
