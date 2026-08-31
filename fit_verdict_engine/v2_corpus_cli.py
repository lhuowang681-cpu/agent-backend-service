from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fit_verdict_engine.data_generator import write_jsonl
from fit_verdict_engine.v2_data import eval_rows, generate_v2_eval_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write fixed Fit Verdict v2 evaluation corpora.")
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = generate_v2_eval_records(args.split)
    write_jsonl(args.output, eval_rows(records))
    print(f"Wrote {len(records)} {args.split} v2 records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
