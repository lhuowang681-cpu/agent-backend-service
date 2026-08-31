from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True)
class BackendSettings:
    database_url: str
    redis_url: str
    allowed_user_ids: tuple[str, ...]
    worker_id: str
    stream_name: str = "job-agent:runs"
    consumer_group: str = "agent-workers"
    lease_seconds: float = 30.0
    publisher_interval_seconds: float = 0.25
    recovery_interval_seconds: float = 5.0
    dispatch_reconcile_grace_seconds: float = 30.0
    max_attempts: int = 3
    max_queued_runs_global: int = 1000
    max_queued_runs_per_user: int = 100
    provider_global_limit: int = 4
    provider_per_user_limit: int = 4
    provider_permit_ttl_seconds: float = 60.0
    provider_circuit_threshold: int = 3
    provider_circuit_cooldown_seconds: float = 10.0
    metrics_prefix: str = "job-agent:metrics"
    backend_workspace_root: str = "output/backend_service"
    deepseek_api_key_env: str = "API_KEY"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.dispatch_reconcile_grace_seconds < 0:
            raise ValueError("dispatch_reconcile_grace_seconds_must_not_be_negative")
        if self.max_queued_runs_global < 1 or self.max_queued_runs_per_user < 1:
            raise ValueError("queued_run_limits_must_be_positive")
        if self.provider_permit_ttl_seconds <= self.deepseek_timeout_seconds:
            raise ValueError("provider_permit_ttl_must_exceed_provider_timeout")

    @classmethod
    def from_env(cls) -> "BackendSettings":
        raw_users = os.environ.get(
            "JOB_AGENT_ALLOWED_USER_IDS",
            "demo-user-a,demo-user-b",
        )
        user_ids = tuple(value.strip() for value in raw_users.split(",") if value.strip())
        if not user_ids:
            raise ValueError("at_least_one_allowed_user_is_required")
        return cls(
            database_url=os.environ.get(
                "JOB_AGENT_DATABASE_URL",
                "postgresql://job_agent:job_agent_dev@127.0.0.1:54329/job_agent",
            ),
            redis_url=os.environ.get(
                "JOB_AGENT_REDIS_URL",
                "redis://127.0.0.1:63799/0",
            ),
            allowed_user_ids=user_ids,
            worker_id=os.environ.get(
                "JOB_AGENT_WORKER_ID",
                f"{socket.gethostname()}-{uuid4()}",
            ),
            stream_name=os.environ.get("JOB_AGENT_STREAM_NAME", "job-agent:runs"),
            consumer_group=os.environ.get("JOB_AGENT_CONSUMER_GROUP", "agent-workers"),
            lease_seconds=float(os.environ.get("JOB_AGENT_LEASE_SECONDS", "30")),
            publisher_interval_seconds=float(
                os.environ.get("JOB_AGENT_PUBLISHER_INTERVAL_SECONDS", "0.25")
            ),
            recovery_interval_seconds=float(
                os.environ.get("JOB_AGENT_RECOVERY_INTERVAL_SECONDS", "5")
            ),
            dispatch_reconcile_grace_seconds=float(
                os.environ.get(
                    "JOB_AGENT_DISPATCH_RECONCILE_GRACE_SECONDS",
                    "30",
                )
            ),
            max_attempts=int(os.environ.get("JOB_AGENT_MAX_ATTEMPTS", "3")),
            max_queued_runs_global=int(
                os.environ.get("JOB_AGENT_MAX_QUEUED_RUNS_GLOBAL", "1000")
            ),
            max_queued_runs_per_user=int(
                os.environ.get("JOB_AGENT_MAX_QUEUED_RUNS_PER_USER", "100")
            ),
            provider_global_limit=int(
                os.environ.get("JOB_AGENT_PROVIDER_GLOBAL_LIMIT", "4")
            ),
            provider_per_user_limit=int(
                os.environ.get("JOB_AGENT_PROVIDER_PER_USER_LIMIT", "4")
            ),
            provider_permit_ttl_seconds=float(
                os.environ.get("JOB_AGENT_PROVIDER_PERMIT_TTL_SECONDS", "60")
            ),
            provider_circuit_threshold=int(
                os.environ.get("JOB_AGENT_PROVIDER_CIRCUIT_THRESHOLD", "3")
            ),
            provider_circuit_cooldown_seconds=float(
                os.environ.get("JOB_AGENT_PROVIDER_CIRCUIT_COOLDOWN_SECONDS", "10")
            ),
            metrics_prefix=os.environ.get(
                "JOB_AGENT_METRICS_PREFIX", "job-agent:metrics"
            ),
            backend_workspace_root=os.environ.get(
                "JOB_AGENT_BACKEND_WORKSPACE_ROOT",
                "output/backend_service",
            ),
            deepseek_api_key_env=os.environ.get(
                "JOB_AGENT_DEEPSEEK_API_KEY_ENV", "API_KEY"
            ),
            deepseek_base_url=os.environ.get(
                "JOB_AGENT_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
            deepseek_model=os.environ.get(
                "JOB_AGENT_DEEPSEEK_MODEL", "deepseek-chat"
            ),
            deepseek_timeout_seconds=float(
                os.environ.get("JOB_AGENT_DEEPSEEK_TIMEOUT_SECONDS", "30")
            ),
        )
