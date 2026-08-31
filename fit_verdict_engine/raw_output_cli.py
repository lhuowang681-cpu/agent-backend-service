from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Sequence

from fit_verdict_engine.data_generator import generate_synthetic_records, load_eval_records
from fit_verdict_engine.raw_output import collect_raw_outputs, write_raw_output_jsonl
from job_agent.nodes.fit_verdict import LocalCausalLMVerdictRunner, LocalVerdictRunner


RunnerFactory = Callable[[str, str | None, int], LocalVerdictRunner]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect raw local_lora Fit Verdict outputs for batch eval.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=4,
        help="Synthetic eval records to generate per verdict class when --eval-input is omitted.",
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=None,
        help="Fixed hand-authored eval JSONL. Overrides generated synthetic records.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("FIT_VERDICT_MODEL", "fixtures/models/fixture-model"),
        help="Local base model path. Defaults to FIT_VERDICT_MODEL or the server Qwen path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("FIT_VERDICT_ADAPTER"),
        help="Optional converted PEFT adapter path. Defaults to FIT_VERDICT_ADAPTER.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per verdict.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/fit_verdict_eval/raw_model_outputs.jsonl"),
        help="Path to write raw-output JSONL rows.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    runner_factory: RunnerFactory = LocalCausalLMVerdictRunner,
) -> int:
    args = build_parser().parse_args(argv)
    runner = runner_factory(args.model_path, args.adapter_path, args.max_new_tokens)
    records = load_eval_records(args.eval_input) if args.eval_input else generate_synthetic_records(per_class=args.per_class)
    raw_rows = collect_raw_outputs(records, runner=runner)
    write_raw_output_jsonl(args.output, raw_rows)
    print(f"Wrote {len(raw_rows)} raw outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
