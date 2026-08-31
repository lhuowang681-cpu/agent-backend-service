from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic, perf_counter
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from job_agent.backend_service.postgres_repository import PostgresRunRepository
from job_agent.backend_service.contracts import RunStatus


DEFAULT_LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
)
DEFAULT_REDIS_METRICS_TIMEOUT_SECONDS = 0.1
DEFAULT_REDIS_METRICS_FAILURE_COOLDOWN_SECONDS = 1.0


class MetricsRecorder(Protocol):
    def record_api_request(
        self, *, method: str, route: str, status_code: int, duration_seconds: float
    ) -> None: ...

    def record_worker_task(self, *, outcome: str, duration_seconds: float) -> None: ...

    def record_provider_call(
        self,
        *,
        provider: str,
        duration_seconds: float,
        succeeded: bool,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None: ...


class NullMetricsRecorder:
    def record_api_request(self, **_kwargs) -> None:
        return None

    def record_worker_task(self, **_kwargs) -> None:
        return None

    def record_provider_call(self, **_kwargs) -> None:
        return None


class RedisMetricsRecorder:
    """Low-cardinality, best-effort process metrics shared by API and Worker."""

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "job-agent:metrics",
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_SECONDS,
        redis_timeout_seconds: float = DEFAULT_REDIS_METRICS_TIMEOUT_SECONDS,
        failure_cooldown_seconds: float = (
            DEFAULT_REDIS_METRICS_FAILURE_COOLDOWN_SECONDS
        ),
    ) -> None:
        if redis_timeout_seconds <= 0 or failure_cooldown_seconds <= 0:
            raise ValueError("metrics_timeout_and_cooldown_must_be_positive")
        self._redis = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=redis_timeout_seconds,
            socket_timeout=redis_timeout_seconds,
            retry_on_timeout=False,
        )
        self.prefix = prefix
        self.buckets = buckets
        self.redis_timeout_seconds = redis_timeout_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self._suppressed_until = 0.0
        self._probe_lock = Lock()

    def close(self) -> None:
        self._redis.close()

    def record_api_request(
        self, *, method: str, route: str, status_code: int, duration_seconds: float
    ) -> None:
        status_class = f"{status_code // 100}xx"
        labels = f"{method.upper()}\t{route}\t{status_class}"

        def build(pipe) -> None:
            pipe.hincrby(f"{self.prefix}:api_requests", labels, 1)
            if status_code >= 400:
                pipe.hincrby(f"{self.prefix}:api_errors", labels, 1)
            if status_code >= 500:
                pipe.hincrby(f"{self.prefix}:api_availability_errors", labels, 1)
            elif status_code >= 400:
                pipe.hincrby(f"{self.prefix}:api_expected_rejections", labels, 1)
            self._observe(pipe, "api_latency", duration_seconds)

        self._execute_best_effort(build)

    def record_worker_task(self, *, outcome: str, duration_seconds: float) -> None:
        def build(pipe) -> None:
            pipe.hincrby(f"{self.prefix}:worker_tasks", outcome, 1)
            self._observe(pipe, "worker_latency", duration_seconds)

        self._execute_best_effort(build)

    def record_provider_call(
        self,
        *,
        provider: str,
        duration_seconds: float,
        succeeded: bool,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        safe_provider = (
            provider
            if provider in {"mock", "deepseek", "openai_compatible"}
            else "other"
        )
        outcome = "success" if succeeded else "error"

        def build(pipe) -> None:
            pipe.hincrby(
                f"{self.prefix}:provider_calls",
                f"{safe_provider}\t{outcome}",
                1,
            )
            if input_tokens is not None:
                pipe.hincrby(
                    f"{self.prefix}:provider_tokens",
                    f"{safe_provider}\tinput",
                    max(input_tokens, 0),
                )
            if output_tokens is not None:
                pipe.hincrby(
                    f"{self.prefix}:provider_tokens",
                    f"{safe_provider}\toutput",
                    max(output_tokens, 0),
                )
            self._observe(pipe, f"provider_latency:{safe_provider}", duration_seconds)

        self._execute_best_effort(build)

    def _execute_best_effort(self, build) -> None:
        if monotonic() < self._suppressed_until:
            return
        if not self._probe_lock.acquire(blocking=False):
            return
        try:
            if monotonic() < self._suppressed_until:
                return
            pipe = self._redis.pipeline(transaction=False)
            build(pipe)
            try:
                pipe.execute()
            except RedisError:
                self._suppressed_until = monotonic() + self.failure_cooldown_seconds
        finally:
            self._probe_lock.release()

    def snapshot(self) -> dict[str, dict[str, float]]:
        if self._probe_lock.locked() or monotonic() < self._suppressed_until:
            raise RedisError("metrics_redis_temporarily_unavailable")
        result: dict[str, dict[str, float]] = {}
        keys = {
            "api_requests": f"{self.prefix}:api_requests",
            "api_errors": f"{self.prefix}:api_errors",
            "api_availability_errors": (
                f"{self.prefix}:api_availability_errors"
            ),
            "api_expected_rejections": (
                f"{self.prefix}:api_expected_rejections"
            ),
            "worker_tasks": f"{self.prefix}:worker_tasks",
            "provider_calls": f"{self.prefix}:provider_calls",
            "provider_tokens": f"{self.prefix}:provider_tokens",
            "api_latency": f"{self.prefix}:hist:api_latency",
            "worker_latency": f"{self.prefix}:hist:worker_latency",
            "provider_latency_mock": f"{self.prefix}:hist:provider_latency:mock",
            "provider_latency_openai_compatible": (
                f"{self.prefix}:hist:provider_latency:openai_compatible"
            ),
            "provider_latency_deepseek": f"{self.prefix}:hist:provider_latency:deepseek",
            "provider_latency_other": f"{self.prefix}:hist:provider_latency:other",
        }
        try:
            pipe = self._redis.pipeline(transaction=False)
            for key in keys.values():
                pipe.hgetall(key)
            values = pipe.execute()
        except RedisError:
            raise
        for name, raw in zip(keys, values, strict=True):
            result[name] = {field: float(value) for field, value in raw.items()}
        return result

    def _observe(self, pipe, name: str, duration_seconds: float) -> None:
        value = max(duration_seconds, 0.0)
        key = f"{self.prefix}:hist:{name}"
        pipe.hincrby(key, "count", 1)
        pipe.hincrbyfloat(key, "sum", value)
        for bucket in self.buckets:
            if value <= bucket:
                pipe.hincrby(key, f"le:{bucket:g}", 1)


@dataclass(frozen=True)
class PrometheusMetricsService:
    repository: PostgresRunRepository
    recorder: RedisMetricsRecorder
    redis_url: str
    stream_name: str
    consumer_group: str
    buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_SECONDS
    redis_timeout_seconds: float = DEFAULT_REDIS_METRICS_TIMEOUT_SECONDS

    def render(self) -> str:
        lines: list[str] = []
        database = self.repository.observability_snapshot(self.buckets)
        self._render_database(lines, database)
        redis_available = 1
        try:
            transient = self.recorder.snapshot()
            self._render_transient(lines, transient)
            queue = Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=self.redis_timeout_seconds,
                socket_timeout=self.redis_timeout_seconds,
                retry_on_timeout=False,
            )
            try:
                retained = int(queue.xlen(self.stream_name))
                try:
                    pending = int(
                        queue.xpending(self.stream_name, self.consumer_group)["pending"]
                    )
                    groups = queue.xinfo_groups(self.stream_name)
                    matching = next(
                        (
                            group
                            for group in groups
                            if group["name"] == self.consumer_group
                        ),
                        None,
                    )
                    lag = int((matching or {}).get("lag") or 0)
                except ResponseError:
                    pending = 0
                    lag = retained
                depth = lag + pending
            finally:
                queue.close()
            lines.extend(
                [
                    "# TYPE job_agent_queue_depth gauge",
                    f"job_agent_queue_depth {depth}",
                    "# TYPE job_agent_queue_pending gauge",
                    f"job_agent_queue_pending {pending}",
                    "# TYPE job_agent_queue_retained_entries gauge",
                    f"job_agent_queue_retained_entries {retained}",
                ]
            )
        except RedisError:
            redis_available = 0
        lines.extend(
            [
                "# TYPE job_agent_metrics_redis_available gauge",
                f"job_agent_metrics_redis_available {redis_available}",
            ]
        )
        return "\n".join(lines) + "\n"

    def _render_database(self, lines: list[str], snapshot: dict[str, object]) -> None:
        lines.append("# TYPE job_agent_runs gauge")
        counts = snapshot["run_status_counts"]
        for status in RunStatus:
            lines.append(
                f'job_agent_runs{{status="{status.value}"}} {counts.get(status.value, 0)}'
            )
        lines.extend(
            [
                "# TYPE job_agent_retries_total counter",
                f'job_agent_retries_total {snapshot["retry_count"]}',
                "# TYPE job_agent_recoveries_total counter",
                f'job_agent_recoveries_total {snapshot["recovery_count"]}',
                "# TYPE job_agent_approvals_waiting gauge",
                f'job_agent_approvals_waiting {snapshot["approvals_waiting"]}',
            ]
        )
        lines.append("# TYPE job_agent_provider_admission_deferrals_total counter")
        allowed_reasons = {
            "provider_circuit_open",
            "provider_global_backpressure",
            "provider_limiter_unavailable",
            "provider_user_backpressure",
            "provider_user_cooldown",
        }
        deferrals: dict[str, int] = snapshot["provider_admission_deferrals"]
        normalized: dict[str, int] = {}
        for reason, count in deferrals.items():
            safe_reason = reason if reason in allowed_reasons else "other"
            normalized[safe_reason] = normalized.get(safe_reason, 0) + count
        for reason, count in sorted(normalized.items()):
            lines.append(
                f'job_agent_provider_admission_deferrals_total{{reason="{reason}"}} {count}'
            )
        self._render_histogram(lines, "job_agent_run_end_to_end_seconds", snapshot["run_latency"])
        self._render_histogram(
            lines,
            "job_agent_approval_wait_seconds",
            snapshot["approval_wait_latency"],
        )

    def _render_transient(
        self, lines: list[str], snapshot: dict[str, dict[str, float]]
    ) -> None:
        for metric, key in (
            ("job_agent_http_requests_total", "api_requests"),
            ("job_agent_http_errors_total", "api_errors"),
            (
                "job_agent_http_availability_errors_total",
                "api_availability_errors",
            ),
            (
                "job_agent_http_expected_rejections_total",
                "api_expected_rejections",
            ),
        ):
            lines.append(f"# TYPE {metric} counter")
            for labels, value in sorted(snapshot[key].items()):
                method, route, status_class = labels.split("\t", 2)
                lines.append(
                    f'{metric}{{method="{method}",route="{route}",status_class="{status_class}"}} {value:g}'
                )
        lines.append("# TYPE job_agent_worker_tasks_total counter")
        for outcome, value in sorted(snapshot["worker_tasks"].items()):
            lines.append(f'job_agent_worker_tasks_total{{outcome="{outcome}"}} {value:g}')
        lines.append("# TYPE job_agent_provider_calls_total counter")
        for labels, value in sorted(snapshot["provider_calls"].items()):
            provider, outcome = labels.split("\t", 1)
            lines.append(
                f'job_agent_provider_calls_total{{provider="{provider}",outcome="{outcome}"}} {value:g}'
            )
        lines.append("# TYPE job_agent_provider_tokens_total counter")
        for labels, value in sorted(snapshot["provider_tokens"].items()):
            provider, direction = labels.split("\t", 1)
            lines.append(
                f'job_agent_provider_tokens_total{{provider="{provider}",direction="{direction}"}} {value:g}'
            )
        self._render_histogram(lines, "job_agent_http_request_seconds", snapshot["api_latency"])
        self._render_histogram(lines, "job_agent_worker_task_seconds", snapshot["worker_latency"])
        for provider in ("mock", "deepseek", "openai_compatible", "other"):
            self._render_histogram(
                lines,
                "job_agent_provider_call_seconds",
                snapshot[f"provider_latency_{provider}"],
                extra_label=("provider", provider),
            )

    @staticmethod
    def _render_histogram(
        lines: list[str],
        name: str,
        histogram: dict[str, object],
        *,
        extra_label: tuple[str, str] | None = None,
    ) -> None:
        lines.append(f"# TYPE {name} histogram")
        labels = ""
        if extra_label is not None:
            labels = f'{extra_label[0]}="{extra_label[1]}",'
        buckets = histogram.get("buckets")
        if buckets is None:
            buckets = {
                key.removeprefix("le:"): value
                for key, value in histogram.items()
                if key.startswith("le:")
            }
            buckets["+Inf"] = histogram.get("count", 0)
        for boundary, count in buckets.items():
            lines.append(f'{name}_bucket{{{labels}le="{boundary}"}} {count:g}')
        count = float(histogram.get("count", 0))
        total = float(histogram.get("sum", 0.0))
        suffix = ""
        if extra_label is not None:
            suffix = f'{{{extra_label[0]}="{extra_label[1]}"}}'
        lines.append(f"{name}_count{suffix} {count:g}")
        lines.append(f"{name}_sum{suffix} {total:g}")


class Timer:
    def __init__(self) -> None:
        self.started = perf_counter()

    def elapsed(self) -> float:
        return max(perf_counter() - self.started, 0.0)
