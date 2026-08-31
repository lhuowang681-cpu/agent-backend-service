CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    external_issuer TEXT,
    external_subject TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (external_issuer, external_subject)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    created_request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('QUEUED', 'RUNNING', 'WAITING_APPROVAL', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'UNCERTAIN')
    ),
    status_version BIGINT NOT NULL DEFAULT 0 CHECK (status_version >= 0),
    dispatch_generation BIGINT NOT NULL DEFAULT 1 CHECK (dispatch_generation >= 1),
    idempotency_key TEXT NOT NULL,
    request_hash CHAR(64) NOT NULL,
    request_payload JSONB NOT NULL,
    current_stage TEXT,
    progress JSONB,
    result_payload JSONB,
    error_code TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    UNIQUE (user_id, run_id),
    UNIQUE (user_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_runs_dispatchable
    ON runs (next_attempt_at, created_at)
    WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_runs_user_session_status
    ON runs (user_id, session_id, status);

CREATE TABLE IF NOT EXISTS run_attempts (
    attempt_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id UUID NOT NULL,
    attempt_no INTEGER NOT NULL CHECK (attempt_no >= 1),
    message_id UUID NOT NULL,
    dispatch_generation BIGINT NOT NULL,
    worker_id TEXT NOT NULL,
    lease_token UUID NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'SUCCEEDED', 'FAILED', 'EXPIRED', 'UNCERTAIN')),
    execution_phase TEXT NOT NULL DEFAULT 'CLAIMED' CHECK (
        execution_phase IN ('CLAIMED', 'AGENT_ACTIVE', 'FINALIZING', 'TERMINAL')
    ),
    provider_call_count INTEGER NOT NULL DEFAULT 0 CHECK (provider_call_count >= 0),
    error_code TEXT,
    recovery_kind TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    UNIQUE (user_id, attempt_id),
    UNIQUE (user_id, run_id, attempt_no),
    FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_run_attempts_expired_active
    ON run_attempts (lease_expires_at)
    WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS run_approvals (
    approval_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id UUID NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    action_digest CHAR(64) NOT NULL,
    pending_action_json JSONB NOT NULL,
    decision TEXT CHECK (decision IN ('approved', 'rejected')),
    reason TEXT,
    version BIGINT NOT NULL DEFAULT 0,
    resolved_by_user_id TEXT REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    UNIQUE (user_id, approval_id),
    UNIQUE (user_id, run_id, request_id),
    FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id),
    CHECK (resolved_by_user_id IS NULL OR resolved_by_user_id = user_id)
);

CREATE TABLE IF NOT EXISTS run_checkpoints (
    user_id TEXT NOT NULL,
    run_id UUID NOT NULL,
    checkpoint_kind TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint_version BIGINT NOT NULL,
    payload_private JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (user_id, run_id),
    FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
);

CREATE TABLE IF NOT EXISTS run_events (
    event_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id UUID NOT NULL,
    attempt_id UUID,
    actor_user_id TEXT REFERENCES users(user_id),
    sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    payload_sanitized JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, event_id),
    UNIQUE (user_id, run_id, sequence),
    FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id),
    FOREIGN KEY (user_id, attempt_id) REFERENCES run_attempts(user_id, attempt_id),
    CHECK (actor_user_id IS NULL OR actor_user_id = user_id)
);

CREATE TABLE IF NOT EXISTS outbox_messages (
    message_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    aggregate_id UUID NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type = 'run.dispatch'),
    dispatch_generation BIGINT NOT NULL CHECK (dispatch_generation >= 1),
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ,
    publish_attempts INTEGER NOT NULL DEFAULT 0 CHECK (publish_attempts >= 0),
    last_error_code TEXT,
    UNIQUE (user_id, message_id),
    FOREIGN KEY (user_id, aggregate_id) REFERENCES runs(user_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_outbox_unpublished
    ON outbox_messages (created_at)
    WHERE published_at IS NULL;
