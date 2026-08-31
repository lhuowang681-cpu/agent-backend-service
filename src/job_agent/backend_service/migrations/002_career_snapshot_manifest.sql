CREATE TABLE career_snapshots (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    snapshot_revision TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    payload_private JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, snapshot_revision),
    CHECK (snapshot_revision ~ '^snapshot-[a-f0-9]{16,64}$'),
    CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    CHECK (jsonb_typeof(payload_private) = 'object')
);

ALTER TABLE runs
    ADD COLUMN execution_manifest JSONB;

ALTER TABLE runs
    ADD CONSTRAINT runs_execution_manifest_object
    CHECK (
        execution_manifest IS NULL
        OR jsonb_typeof(execution_manifest) = 'object'
    );
