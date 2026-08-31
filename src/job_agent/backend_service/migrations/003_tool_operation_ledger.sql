CREATE TABLE tool_operations (
    operation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    run_id UUID NOT NULL,
    attempt_id UUID,
    tool_name TEXT NOT NULL,
    action_digest CHAR(64) NOT NULL,
    request_hash CHAR(64) NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('PREPARED', 'INFLIGHT', 'SUCCEEDED', 'FAILED', 'UNCERTAIN')
    ),
    receipt_private JSONB,
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, run_id, operation_id),
    FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id),
    FOREIGN KEY (user_id, attempt_id) REFERENCES run_attempts(user_id, attempt_id),
    CHECK (action_digest ~ '^[a-f0-9]{64}$'),
    CHECK (request_hash ~ '^[a-f0-9]{64}$'),
    CHECK (receipt_private IS NULL OR jsonb_typeof(receipt_private) = 'object')
);

CREATE INDEX idx_tool_operations_run
    ON tool_operations(user_id, run_id, created_at);
