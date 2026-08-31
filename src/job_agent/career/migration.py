from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .models import CompanyDraft, JobDraft, StageEventDraft
from .store import CareerStore, normalize_company_name


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(path: Path) -> str:
    try: return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError: return "missing"


@dataclass(frozen=True)
class MigrationIssue:
    kind: str
    source_key: str
    payload: dict
    resolution: str | None = None


@dataclass(frozen=True)
class MigrationPreview:
    source_root: Path
    item_count: int
    issues: tuple[MigrationIssue, ...] = ()


@dataclass(frozen=True)
class MigrationResult:
    run_id: str
    imported: int
    merged: int
    skipped: int
    pending: int
    issues: tuple[MigrationIssue, ...] = ()
    summary: dict = field(default_factory=dict)


class LegacyCareerMigration:
    def __init__(self, store: CareerStore | None = None):
        self.store = store

    def _store(self, output_root: Path) -> CareerStore:
        return self.store or CareerStore.open(output_root)

    def _sources(self, output_root: Path) -> list[tuple[str, Path, dict]]:
        sources: list[tuple[str, Path, dict]] = []
        tracker = output_root / "tracker.json"
        if tracker.exists():
            try: payload = json.loads(tracker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): payload = {"_error": "invalid_json"}
            sources.append(("tracker:canonical", tracker, payload))
        for path in sorted((output_root / "sessions").glob("*/selected_job.json")) if (output_root / "sessions").exists() else []:
            try: payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError): payload = {"_error": "invalid_json"}
            sources.append((f"session:{path.parent.name}:selected_job", path, payload))
        return sources

    def inspect(self, output_root: Path) -> MigrationPreview:
        issues: list[MigrationIssue] = []
        for key, path, payload in self._sources(Path(output_root)):
            if payload.get("_error"):
                issues.append(MigrationIssue("corrupt_source", key, {"path": str(path), "error": payload["_error"]}))
            if key.startswith("tracker"):
                for app in payload.get("applications", []) if isinstance(payload, dict) else []:
                    company = str(app.get("company") or "").strip()
                    if not company or normalize_company_name(company) in {"unknown", "user jd"}:
                        issues.append(MigrationIssue("unknown_company", f"{key}:{app.get('id','')}", app))
        return MigrationPreview(Path(output_root), len(self._sources(Path(output_root))), tuple(issues))

    def apply(self, output_root: Path, *, run_id: str | None = None) -> MigrationResult:
        output_root = Path(output_root); store = self._store(output_root); run_id = run_id or f"mig-{uuid.uuid4().hex[:12]}"
        started = _now(); imported = merged = skipped = pending = 0; issues: list[MigrationIssue] = []
        sources = self._sources(output_root)
        with store.transaction() as conn:
            conn.execute("INSERT INTO migration_runs(id, started_at, source_root, summary_json) VALUES (?,?,?,?)", (run_id, started, str(output_root), "{}"))
            for source_key, path, payload in sources:
                content_hash = _hash(path)
                previous = conn.execute("SELECT 1 FROM migration_items WHERE source_key=? AND content_hash=? AND decision IN ('imported','merged','skipped','pending')", (source_key, content_hash)).fetchone()
                if previous:
                    skipped += 1; decision = "skipped"; reason = "already imported with same content"
                    conn.execute("INSERT INTO migration_items VALUES (?,?,?,?,?,?,?,?)", (run_id, source_key, str(path), content_hash, "source", None, decision, reason)); continue
                if payload.get("_error"):
                    pending += 1; issue = MigrationIssue("corrupt_source", source_key, {"path": str(path), "error": payload["_error"]}); issues.append(issue)
                    conn.execute("INSERT INTO migration_issues VALUES (?,?,?,?,?,?)", (f"issue-{uuid.uuid4().hex[:12]}", run_id, issue.kind, issue.source_key, json.dumps(issue.payload, ensure_ascii=False), None))
                    conn.execute("INSERT INTO migration_items VALUES (?,?,?,?,?,?,?,?)", (run_id, source_key, str(path), content_hash, "source", None, "pending", "corrupt source")); continue
                if source_key.startswith("tracker"):
                    for app in payload.get("applications", []):
                        company_name = str(app.get("company") or "").strip(); normalized = normalize_company_name(company_name)
                        company_id = None
                        if company_name and normalized not in {"unknown", "user jd"}:
                            # use store methods through the same connection-independent API; transaction is re-entrant unsafe,
                            # so issue raw SQL here and keep the operation atomic.
                            row = conn.execute("SELECT id FROM companies WHERE normalized_name=?", (normalized,)).fetchone()
                            if row: company_id = row[0]; merged += 1
                            else:
                                company_id = f"co-{uuid.uuid4().hex[:12]}"; now = _now(); conn.execute("INSERT INTO companies VALUES (?,?,?,?,?,?,?,?)", (company_id, normalized, company_name, "normal", None, "", now, now)); imported += 1
                        else:
                            pending += 1; issue = MigrationIssue("unknown_company", f"{source_key}:{app.get('id','')}", app); issues.append(issue)
                            conn.execute("INSERT INTO migration_issues VALUES (?,?,?,?,?,?)", (f"issue-{uuid.uuid4().hex[:12]}", run_id, issue.kind, issue.source_key, json.dumps(issue.payload, ensure_ascii=False), None))
                        title = str(app.get("title") or "Untitled job")
                        jid = f"job-{uuid.uuid4().hex[:12]}"; legacy_id = app.get("job_id")
                        existing = conn.execute("SELECT id FROM jobs WHERE legacy_job_id=?", (legacy_id,)).fetchone() if legacy_id else None
                        if existing: jid = existing[0]; merged += 1
                        else:
                            now = _now(); conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (jid, company_id, legacy_id, title, str(app.get("city") or ""), str(app.get("url") or ""), str(app.get("source") or "legacy"), str(app.get("current_state") or "to_apply"), app.get("session_dir"), now, now)); imported += 1
                        for index, event in enumerate(app.get("state_history") or []):
                            source = f"{source_key}:{app.get('id','')}:{index}:{event.get('timestamp','')}"
                            if conn.execute("SELECT 1 FROM stage_events WHERE source_key=?", (source,)).fetchone(): continue
                            conn.execute("INSERT INTO stage_events VALUES (?,?,?,?,?,?,?)", (f"evt-{uuid.uuid4().hex[:12]}", jid, str(event.get("state") or app.get("current_state") or "to_apply"), str(event.get("timestamp") or _now()), str(event.get("notes") or ""), "legacy", source))
                        conn.execute("INSERT INTO migration_items VALUES (?,?,?,?,?,?,?,?)", (run_id, f"{source_key}:{app.get('id','')}", str(path), content_hash, "job", jid, "pending" if company_id is None else "imported", "unknown company" if company_id is None else "legacy tracker"))
                    conn.execute("INSERT INTO migration_items VALUES (?,?,?,?,?,?,?,?)", (run_id, source_key, str(path), content_hash, "source", None, "imported", "canonical tracker scanned"))
                else:
                    job = payload; company_name = str(job.get("company") or "").strip(); normalized = normalize_company_name(company_name)
                    row = conn.execute("SELECT id FROM companies WHERE normalized_name=?", (normalized,)).fetchone() if normalized else None
                    company_id = row[0] if row else None
                    if company_id is None and company_name and normalized not in {"unknown", "user jd"}:
                        company_id = f"co-{uuid.uuid4().hex[:12]}"; now = _now(); conn.execute("INSERT INTO companies VALUES (?,?,?,?,?,?,?,?)", (company_id, normalized, company_name, "normal", None, "", now, now)); imported += 1
                    if company_id is None:
                        pending += 1; issue = MigrationIssue("unknown_company", source_key, job); issues.append(issue); conn.execute("INSERT INTO migration_issues VALUES (?,?,?,?,?,?)", (f"issue-{uuid.uuid4().hex[:12]}", run_id, issue.kind, issue.source_key, json.dumps(issue.payload, ensure_ascii=False), None))
                    jid = f"job-{uuid.uuid4().hex[:12]}"; legacy_id = job.get("job_id")
                    existing = conn.execute("SELECT id FROM jobs WHERE legacy_job_id=?", (legacy_id,)).fetchone() if legacy_id else None
                    if existing: jid = existing[0]; merged += 1
                    else:
                        now = _now(); conn.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (jid, company_id, legacy_id, str(job.get("title") or "Untitled job"), str(job.get("location") or ""), str(job.get("url") or ""), str(job.get("source") or "legacy"), "to_apply", str(path.parent), now, now)); imported += 1
                    for artifact in path.parent.iterdir():
                        if not artifact.is_file():
                            continue
                        relative = artifact.name
                        link_id = f"art-{uuid.uuid4().hex[:12]}"
                        conn.execute("INSERT OR IGNORE INTO artifact_links VALUES (?,?,?,?,?,?)", (link_id, jid, str(path.parent), artifact.suffix.lstrip(".") or "file", relative, _hash(artifact)))
                        if artifact.name.startswith("10_mock_interview_run") or artifact.name.startswith("mock_answers"):
                            exists_interview = conn.execute("SELECT 1 FROM interviews WHERE job_id=? AND artifact_path=?", (jid, relative)).fetchone()
                            if not exists_interview:
                                review = "15_mock_interview_debrief.md" if (path.parent / "15_mock_interview_debrief.md").exists() else None
                                conn.execute("INSERT INTO interviews VALUES (?,?,?,?,?,?,?,?,?)", (f"int-{uuid.uuid4().hex[:12]}", jid, company_id, "mock", "mock interview", None, str(path.parent), relative, review))
                    conn.execute("INSERT INTO migration_items VALUES (?,?,?,?,?,?,?,?)", (run_id, source_key, str(path), content_hash, "job", jid, "pending" if company_id is None else "imported", "unknown company" if company_id is None else "session selected job"))
        completed = _now(); summary = {"imported": imported, "merged": merged, "skipped": skipped, "pending": pending, "issues": len(issues)}; store.record_migration_run(run_id, started, completed, output_root, summary)
        return MigrationResult(run_id, imported, merged, skipped, pending, tuple(issues), summary)

    def resolve_issue(self, issue_id: str, resolution: str) -> None:
        store = self.store
        if store is None: raise ValueError("migration must be initialized with a CareerStore to resolve issues")
        with store.transaction() as conn:
            conn.execute("UPDATE migration_issues SET resolution=? WHERE id=?", (resolution, issue_id))
