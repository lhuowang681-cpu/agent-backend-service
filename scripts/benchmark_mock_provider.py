from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from time import perf_counter

from pydantic import Field

from job_agent.llm.provider import TraceContext
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.schemas import StrictModel


class BenchmarkOutput(StrictModel):
    value: str = Field(min_length=1)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def invoke(index: int) -> tuple[float, bool]:
    provider = MockLLMProvider([{"value": f"fixture-{index}"}])
    started = perf_counter()
    result = provider.generate_structured(
        system_prompt="controlled benchmark",
        user_prompt="return the fixture",
        output_schema=BenchmarkOutput,
        tools=(),
        temperature=0.0,
        max_output_tokens=32,
        trace=TraceContext(
            session_id="mock-benchmark",
            run_id=f"mock-{index}",
            node_id="benchmark",
            skill_id="benchmark",
            skill_version="v1",
            prompt_version="v1",
        ),
    )
    return (perf_counter() - started) * 1000.0, result.schema_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.calls <= 0 or args.concurrency <= 0:
        raise SystemExit("calls and concurrency must be positive")

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(invoke, range(args.calls)))
    wall_seconds = perf_counter() - started
    latencies = [latency for latency, _ in results]
    failures = sum(not succeeded for _, succeeded in results)
    report = {
        "subject": "MockLLMProvider structured fixture path",
        "calls": args.calls,
        "concurrency": args.concurrency,
        "throughput_calls_per_second": args.calls / wall_seconds,
        "latency_ms": {
            "p50": median(latencies),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "errors": failures,
        "error_rate": failures / args.calls,
        "wall_seconds": wall_seconds,
        "disclaimer": "CPU-local Mock fixture benchmark; not real LLM API performance.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
