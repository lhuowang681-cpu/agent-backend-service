from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fit_verdict_engine.candidate_gate import evaluate_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Fit Verdict development candidate gate.")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection = evaluate_candidate(
        candidate_name=args.candidate_name,
        candidate_metrics_path=args.candidate_metrics,
        baseline_metrics_path=args.baseline_metrics,
        corpus_path=args.corpus,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {selection['status']} candidate selection to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
