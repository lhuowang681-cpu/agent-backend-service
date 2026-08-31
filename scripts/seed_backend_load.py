from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4


def submit(
    base_url: str,
    index: int,
    *,
    batch_id: str,
    user_ids: tuple[str, ...],
) -> tuple[str, int]:
    user_id = user_ids[index % len(user_ids)]
    unique = f"{batch_id}-{index}-{uuid4().hex}"
    payload = {
        "task_type": "semantic_job_flow",
        "session_id": f"worker-seed-session-{unique}",
        "input": {
            "selected_job": {
                "job_id": f"worker-seed-job-{unique}",
                "company": "Worker Seed Fixture",
                "title": "Backend Agent Engineer",
                "desc": "Exact-count Worker drain benchmark",
                "url": f"https://example.invalid/worker-seed/{unique}",
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
            "X-User-ID": user_id,
            "X-Request-ID": f"worker-seed-request-{unique}",
            "Idempotency-Key": f"worker-seed-{batch_id}-{index}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            return user_id, response.status
    except HTTPError as exc:
        return user_id, exc.code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--batch-id", default=uuid4().hex)
    parser.add_argument("--users", default="demo-user-a,demo-user-b")
    parser.add_argument("--expected-accepted", type=int)
    args = parser.parse_args()
    if args.runs <= 0 or args.concurrency <= 0:
        raise SystemExit("runs and concurrency must be positive")
    user_ids = tuple(value.strip() for value in args.users.split(",") if value.strip())
    if not user_ids:
        raise SystemExit("at least one user is required")
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        outcomes = list(
            executor.map(
                lambda index: submit(
                    args.base_url,
                    index,
                    batch_id=args.batch_id,
                    user_ids=user_ids,
                ),
                range(args.runs),
            )
        )
    statuses = [status for _, status in outcomes]
    accepted = statuses.count(202)
    expected_accepted = (
        args.runs if args.expected_accepted is None else args.expected_accepted
    )
    if expected_accepted < 0 or expected_accepted > args.runs:
        raise SystemExit("expected accepted must be between zero and runs")
    report = {
        "batch_id": args.batch_id,
        "users": list(user_ids),
        "requested": args.runs,
        "accepted": accepted,
        "failed": args.runs - accepted,
        "status_counts": dict(sorted(Counter(statuses).items())),
        "accepted_by_user": {
            user_id: sum(
                status == 202
                for outcome_user_id, status in outcomes
                if outcome_user_id == user_id
            )
            for user_id in user_ids
        },
        "seed_seconds": perf_counter() - started,
    }
    print(json.dumps(report, ensure_ascii=False))
    if accepted != expected_accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
