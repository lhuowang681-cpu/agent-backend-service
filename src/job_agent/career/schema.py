SCHEMA_VERSION = 3

DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS companies (
 id TEXT PRIMARY KEY, normalized_name TEXT NOT NULL UNIQUE, display_name TEXT NOT NULL,
 priority TEXT NOT NULL DEFAULT 'normal', website_url TEXT, notes TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS company_tags (company_id TEXT NOT NULL REFERENCES companies(id) ON DELETE CASCADE, tag TEXT NOT NULL, PRIMARY KEY(company_id, tag));
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, company_id TEXT REFERENCES companies(id), legacy_job_id TEXT,
 title TEXT NOT NULL, city TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '', source TEXT NOT NULL,
 current_state TEXT NOT NULL, session_dir TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_legacy ON jobs(legacy_job_id) WHERE legacy_job_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS stage_events (
 id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, state TEXT NOT NULL,
 occurred_at TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', origin TEXT NOT NULL, source_key TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_stage_events_job_time ON stage_events(job_id, occurred_at);
CREATE TABLE IF NOT EXISTS interviews (
 id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), company_id TEXT REFERENCES companies(id), kind TEXT NOT NULL,
 round_label TEXT NOT NULL DEFAULT '', occurred_at TEXT, session_dir TEXT NOT NULL, artifact_path TEXT NOT NULL,
 ai_review_path TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_interviews_artifact ON interviews(session_dir, artifact_path);
CREATE TABLE IF NOT EXISTS artifact_links (
 id TEXT PRIMARY KEY, job_id TEXT REFERENCES jobs(id), session_dir TEXT NOT NULL, artifact_type TEXT NOT NULL,
 relative_path TEXT NOT NULL, content_hash TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_unique ON artifact_links(job_id, session_dir, relative_path);
CREATE TABLE IF NOT EXISTS migration_runs (
 id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT, source_root TEXT NOT NULL, summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS migration_items (
 run_id TEXT NOT NULL REFERENCES migration_runs(id) ON DELETE CASCADE, source_key TEXT NOT NULL,
 source_path TEXT NOT NULL, content_hash TEXT NOT NULL, entity_type TEXT NOT NULL, entity_id TEXT,
 decision TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(run_id, source_key)
);
DROP INDEX IF EXISTS idx_migration_success;
CREATE INDEX IF NOT EXISTS idx_migration_source_hash ON migration_items(source_key, content_hash);
CREATE TABLE IF NOT EXISTS migration_issues (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES migration_runs(id) ON DELETE CASCADE, kind TEXT NOT NULL,
 source_key TEXT NOT NULL, payload_json TEXT NOT NULL, resolution TEXT
);
"""
