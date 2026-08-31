from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


@dataclass(frozen=True)
class Observation:
    status: int | None
    retry_after: str | None
    error_code: str | None
    transport_error: str | None
    latency_ms: float


def submit(base_url: str, index: int, timeout_seconds: float) -> Observation:
    unique = f"database-outage-{index}-{uuid4().hex}"
    payload = {
        "task_type": "semantic_job_flow",
        "session_id": unique,
        "input": {
            "selected_job": {
                "job_id": unique,
                "company": "Database Outage Fixture",
                "title": "Backend Agent Engineer",
                "desc": "Controlled PostgreSQL outage probe",
                "url": f"https://example.invalid/{unique}",
                "location": "Beijing",
            },
            "resume_ref": "artifact://demo/resume",
        },
        "provider_profile": "mock",
        "budget_profile": "quick",
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/runs",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-User-ID": "demo-user-a",
            "X-Request-ID": unique,
            "Idempotency-Key": unique,
        },
        method="POST",
    )
    started = perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read())
            return Observation(
                status=response.status,
                retry_after=response.headers.get("Retry-After"),
                error_code=(body.get("detail") or {}).get("code"),
                transport_error=None,
                latency_ms=(perf_counter() - started) * 1000,
            )
    except HTTPError as exc:
        body = json.loads(exc.read())
        return Observation(
            status=exc.code,
            retry_after=exc.headers.get("Retry-After"),
            error_code=(body.get("detail") or {}).get("code"),
            transport_error=None,
            latency_ms=(perf_counter() - started) * 1000,
        )
    except (TimeoutError, URLError) as exc:
        return Observation(
            status=None,
            retry_after=None,
            error_code=None,
            transport_error=type(exc).__name__,
            latency_ms=(perf_counter() - started) * 1000,
        )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("requests, concurrency and timeout must be positive")

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        observations = list(
            executor.map(
                lambda index: submit(args.base_url, index, args.timeout_seconds),
                range(args.requests),
            )
        )
    elapsed = perf_counter() - started
    latencies = [item.latency_ms for item in observations]
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_seconds": elapsed,
        "status_counts": dict(
            sorted(Counter(item.status for item in observations).items(), key=str)
        ),
        "transport_error_counts": dict(
            sorted(
                Counter(
                    item.transport_error
                    for item in observations
                    if item.transport_error is not None
                ).items()
            )
        ),
        "retry_after_counts": dict(
            sorted(Counter(item.retry_after for item in observations).items(), key=str)
        ),
        "error_code_counts": dict(
            sorted(Counter(item.error_code for item in observations).items(), key=str)
        ),
        "latency_ms": {
            "p50": median(latencies),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        },
    }
    print(json.dumps(report, ensure_ascii=False))
    expected = all(
        item.status == 503
        and item.retry_after == "1"
        and item.error_code == "database_unavailable"
        and item.transport_error is None
        for item in observations
    )
    return 0 if expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
