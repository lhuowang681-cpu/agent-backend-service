from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

import psycopg
from psycopg.rows import dict_row


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED", "UNCERTAIN"}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": median(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def load_rows(database_url: str, batch_id: str) -> list[dict[str, object]]:
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        return list(
            connection.execute(
                """
                SELECT
                    r.run_id,
                    r.user_id,
                    r.status AS run_status,
                    r.retry_count,
                    r.created_at,
                    r.finished_at AS run_finished_at,
                    a.attempt_no,
                    a.worker_id,
                    a.status AS attempt_status,
                    a.error_code AS attempt_error_code,
                    a.recovery_kind,
                    a.started_at,
                    a.finished_at,
                    a.provider_call_count
                FROM runs r
                LEFT JOIN run_attempts a
                  ON a.user_id = r.user_id AND a.run_id = r.run_id
                WHERE r.idempotency_key LIKE %s
                ORDER BY r.created_at, a.attempt_no
                """,
                (f"worker-seed-{batch_id}-%",),
            ).fetchall()
        )


def wait_for_terminal(
    database_url: str,
    batch_id: str,
    *,
    expected_runs: int,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rows = load_rows(database_url, batch_id)
        statuses = {str(row["run_id"]): row["run_status"] for row in rows}
        if len(statuses) == expected_runs and all(
            status in TERMINAL_STATUSES for status in statuses.values()
        ):
            return rows
        time.sleep(0.25)
    raise TimeoutError(f"batch {batch_id} did not reach {expected_runs} terminal runs")


def seconds_between(later: datetime, earlier: datetime) -> float:
    return max((later - earlier).total_seconds(), 0.0)


def build_report(rows: list[dict[str, object]], batch_id: str) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    attempts: list[dict[str, object]] = []
    for row in rows:
        run_id = str(row["run_id"])
        runs.setdefault(run_id, row)
        if row["attempt_no"] is not None:
            attempts.append(row)

    terminal_runs = list(runs.values())
    successful = [row for row in terminal_runs if row["run_status"] == "SUCCEEDED"]
    status_counts = Counter(str(row["run_status"]) for row in terminal_runs)
    success_by_user = Counter(str(row["user_id"]) for row in successful)
    retry_count = sum(int(row["retry_count"]) for row in terminal_runs)

    finished_attempts = [
        row
        for row in attempts
        if row["started_at"] is not None and row["finished_at"] is not None
    ]
    attempt_seconds = [
        seconds_between(row["finished_at"], row["started_at"])
        for row in finished_attempts
    ]
    e2e_seconds = [
        seconds_between(row["run_finished_at"], row["created_at"])
        for row in terminal_runs
        if row["run_finished_at"] is not None
    ]

    first_attempt_by_run: dict[str, datetime] = {}
    for row in attempts:
        if row["started_at"] is None:
            continue
        run_id = str(row["run_id"])
        observed = first_attempt_by_run.get(run_id)
        if observed is None or row["started_at"] < observed:
            first_attempt_by_run[run_id] = row["started_at"]
    queue_wait_by_user: dict[str, list[float]] = defaultdict(list)
    for row in terminal_runs:
        started_at = first_attempt_by_run.get(str(row["run_id"]))
        if started_at is not None:
            queue_wait_by_user[str(row["user_id"])].append(
                seconds_between(started_at, row["created_at"])
            )

    execution_window = None
    throughput = None
    if finished_attempts:
        execution_window = seconds_between(
            max(row["finished_at"] for row in finished_attempts),
            min(row["started_at"] for row in finished_attempts),
        )
        if execution_window > 0:
            throughput = len(successful) / execution_window

    ordered_successes = sorted(
        (row for row in successful if row["run_finished_at"] is not None),
        key=lambda row: row["run_finished_at"],
    )
    prefix_counts: Counter[str] = Counter(
        {user_id: 0 for user_id in success_by_user}
    )
    max_prefix_skew = 0
    for row in ordered_successes:
        prefix_counts[str(row["user_id"])] += 1
        if prefix_counts:
            max_prefix_skew = max(
                max_prefix_skew,
                max(prefix_counts.values()) - min(prefix_counts.values()),
            )

    worker_counts = Counter(
        str(row["worker_id"])
        for row in finished_attempts
        if row["attempt_status"] == "SUCCEEDED"
    )
    attempt_error_counts = Counter(
        str(row["attempt_error_code"])
        for row in attempts
        if row["attempt_error_code"] is not None
    )
    admission_deferred_count = sum(
        row["recovery_kind"] == "provider_admission_deferred" for row in attempts
    )
    return {
        "batch_id": batch_id,
        "runs": len(terminal_runs),
        "status_counts": dict(sorted(status_counts.items())),
        "success_by_user": dict(sorted(success_by_user.items())),
        "retry_count": retry_count,
        "attempt_count": len(attempts),
        "admission_deferred_count": admission_deferred_count,
        "attempt_error_counts": dict(sorted(attempt_error_counts.items())),
        "worker_count_observed": len(worker_counts),
        "successful_attempts_by_worker": dict(sorted(worker_counts.items())),
        "execution_window_seconds": execution_window,
        "throughput_runs_per_second": throughput,
        "attempt_seconds": distribution(attempt_seconds),
        "queue_wait_seconds": distribution(
            [value for values in queue_wait_by_user.values() for value in values]
        ),
        "queue_wait_p95_by_user": {
            user_id: percentile(values, 0.95)
            for user_id, values in sorted(queue_wait_by_user.items())
        },
        "end_to_end_seconds": distribution(e2e_seconds),
        "max_completion_prefix_skew": max_prefix_skew,
        "provider_calls": sum(int(row["provider_call_count"]) for row in attempts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--database-url",
        default=os.environ.get(
            "JOB_AGENT_LOAD_DATABASE_URL",
            "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent_load",
        ),
    )
    args = parser.parse_args()
    if args.expected_runs <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("expected runs and timeout must be positive")
    rows = wait_for_terminal(
        args.database_url,
        args.batch_id,
        expected_runs=args.expected_runs,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(build_report(rows, args.batch_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
