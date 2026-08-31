from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import ArtifactLink, Company, CompanyDetail, CompanyDraft, CompanySummary, Interview, Job, JobDraft, StageEvent, StageEventDraft
from .schema import DDL, SCHEMA_VERSION


def normalize_company_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().casefold()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CareerStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def open(cls, output_root: Path) -> "CareerStore":
        store = cls(Path(output_root) / "career.sqlite3")
        store.initialize()
        return store

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.executescript(DDL)
        if conn.execute("SELECT 1 FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)).fetchone() is None:
            conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (SCHEMA_VERSION, _now()))
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @contextmanager
    def transaction(self):
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close(); self._conn = None

    @staticmethod
    def _company(row, tags=()) -> Company:
        return Company(id=row["id"], normalized_name=row["normalized_name"], display_name=row["display_name"], priority=row["priority"], website_url=row["website_url"], notes=row["notes"], tags=tuple(tags), created_at=row["created_at"], updated_at=row["updated_at"])

    def save_company(self, draft: CompanyDraft, *, company_id: str | None = None) -> Company:
        name = draft.display_name.strip()
        normalized = normalize_company_name(name)
        if not normalized:
            raise ValueError("company display_name is required")
        now = _now(); cid = company_id or f"co-{uuid.uuid4().hex[:12]}"
        with self.transaction() as c:
            existing = c.execute("SELECT * FROM companies WHERE normalized_name=?", (normalized,)).fetchone()
            if existing and (company_id is None or existing["id"] != company_id):
                if company_id is not None:
                    raise ValueError("company name conflicts with an existing company; merge or choose another name")
                cid = existing["id"]
            if existing:
                cid = existing["id"]
                c.execute("UPDATE companies SET display_name=?, priority=?, website_url=?, notes=?, updated_at=? WHERE id=?", (name, draft.priority, draft.website_url, draft.notes, now, cid))
            else:
                c.execute("INSERT INTO companies VALUES (?,?,?,?,?,?,?,?)", (cid, normalized, name, draft.priority, draft.website_url, draft.notes, now, now))
            c.execute("DELETE FROM company_tags WHERE company_id=?", (cid,))
            c.executemany("INSERT INTO company_tags(company_id, tag) VALUES (?,?)", [(cid, t.strip()) for t in draft.tags if t.strip()])
            row = c.execute("SELECT * FROM companies WHERE id=?", (cid,)).fetchone()
            tags = [r[0] for r in c.execute("SELECT tag FROM company_tags WHERE company_id=? ORDER BY tag", (cid,))]
        return self._company(row, tags)

    def list_companies(self, query: str = "", priority: str | None = None, tags: set[str] | None = None) -> list[CompanySummary]:
        tags = tags or set(); conn = self._connect(); params = []; clauses = []
        if query.strip(): clauses.append("(c.display_name LIKE ? OR c.normalized_name LIKE ?)"); params += [f"%{query.strip()}%", f"%{normalize_company_name(query)}%"]
        if priority: clauses.append("c.priority=?"); params.append(priority)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute(f"SELECT c.*, COUNT(j.id) job_count, (SELECT state FROM stage_events se JOIN jobs sj ON sj.id=se.job_id WHERE sj.company_id=c.id ORDER BY se.occurred_at DESC LIMIT 1) latest_state, MAX(j.updated_at) latest_updated_at FROM companies c LEFT JOIN jobs j ON j.company_id=c.id{where} GROUP BY c.id ORDER BY c.updated_at DESC", params).fetchall()
        result = []
        for row in rows:
            row_tags = {r[0] for r in conn.execute("SELECT tag FROM company_tags WHERE company_id=?", (row["id"],))}
            if tags and not tags.issubset(row_tags): continue
            result.append(CompanySummary(self._company(row, sorted(row_tags)), row["job_count"], row["latest_state"], row["latest_updated_at"]))
        return result

    def list_tags(self) -> list[str]:
        return [row[0] for row in self._connect().execute("SELECT DISTINCT tag FROM company_tags ORDER BY tag")]

    def get_company_detail(self, company_id: str) -> CompanyDetail:
        conn = self._connect(); row = conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
        if row is None: raise KeyError(company_id)
        tags = [r[0] for r in conn.execute("SELECT tag FROM company_tags WHERE company_id=? ORDER BY tag", (company_id,))]
        jobs = tuple(Job(**dict(r)) for r in conn.execute("SELECT * FROM jobs WHERE company_id=? ORDER BY updated_at DESC", (company_id,)))
        interviews = tuple(Interview(**dict(r)) for r in conn.execute("SELECT * FROM interviews WHERE company_id=? ORDER BY occurred_at DESC", (company_id,)))
        latest = conn.execute("SELECT state, occurred_at FROM stage_events WHERE job_id IN (SELECT id FROM jobs WHERE company_id=?) ORDER BY occurred_at DESC LIMIT 1", (company_id,)).fetchone()
        return CompanyDetail(self._company(row, tags), jobs, interviews, latest["state"] if latest else None, latest["occurred_at"] if latest else None)

    def get_job(self, job_id: str) -> Job:
        row = self._connect().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return Job(**dict(row))

    def attach_job(self, job: JobDraft, *, session_dir: Path | None = None) -> Job:
        now = _now(); jid = f"job-{uuid.uuid4().hex[:12]}"
        with self.transaction() as c:
            existing = None
            if job.legacy_job_id: existing = c.execute("SELECT * FROM jobs WHERE legacy_job_id=?", (job.legacy_job_id,)).fetchone()
            if existing is None and job.company_id:
                existing = c.execute("SELECT * FROM jobs WHERE company_id=? AND lower(title)=lower(?) AND url=?", (job.company_id, job.title.strip(), job.url)).fetchone()
            if existing:
                jid = existing["id"]; c.execute("UPDATE jobs SET company_id=?, title=?, city=?, url=?, source=?, current_state=?, session_dir=?, updated_at=? WHERE id=?", (job.company_id, job.title.strip(), job.city, job.url, job.source, job.current_state, str(session_dir) if session_dir else existing["session_dir"], now, jid))
            else:
                c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (jid, job.company_id, job.legacy_job_id, job.title.strip(), job.city, job.url, job.source, job.current_state, str(session_dir) if session_dir else None, now, now))
                event = StageEventDraft(job.current_state, occurred_at=now, origin="attach", source_key=f"job:{jid}:initial")
                self._append_event(c, jid, event)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return Job(**dict(row))

    def _append_event(self, c, job_id: str, event: StageEventDraft) -> StageEvent:
        occurred = event.occurred_at or _now(); source = event.source_key or f"event:{job_id}:{uuid.uuid4().hex}"
        existing = c.execute("SELECT * FROM stage_events WHERE source_key=?", (source,)).fetchone()
        if existing: return StageEvent(**dict(existing))
        eid = f"evt-{uuid.uuid4().hex[:12]}"
        c.execute("INSERT INTO stage_events VALUES (?,?,?,?,?,?,?)", (eid, job_id, event.state, occurred, event.notes, event.origin, source))
        c.execute("UPDATE jobs SET current_state=?, updated_at=? WHERE id=?", (event.state, occurred, job_id))
        return StageEvent(eid, job_id, event.state, occurred, event.notes, event.origin, source)

    def append_stage_event(self, job_id: str, event: StageEventDraft) -> Job:
        with self.transaction() as c:
            current = c.execute("SELECT current_state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if current is None:
                raise KeyError(job_id)
            if event.origin != "legacy":
                from job_agent.schemas import ApplicationStatus
                from job_agent.tools.application_tracker import VALID_TRANSITIONS
                try:
                    old = ApplicationStatus(current["current_state"])
                    new = ApplicationStatus(event.state)
                except ValueError as exc:
                    raise ValueError(f"unknown application state: {event.state}") from exc
                if new != old and new not in VALID_TRANSITIONS[old]:
                    raise ValueError(f"invalid transition: {old.value} -> {new.value}")
            self._append_event(c, job_id, event)
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job(**dict(row))

    def add_artifact_link(self, *, job_id: str | None, session_dir: Path, artifact_type: str, relative_path: str, content_hash: str | None = None) -> ArtifactLink:
        aid = f"art-{uuid.uuid4().hex[:12]}"
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO artifact_links VALUES (?,?,?,?,?,?)", (aid, job_id, str(session_dir), artifact_type, relative_path, content_hash))
            row = c.execute("SELECT * FROM artifact_links WHERE job_id IS ? AND session_dir=? AND relative_path=?", (job_id, str(session_dir), relative_path)).fetchone()
        return ArtifactLink(**dict(row))

    def add_interview(self, *, job_id: str | None, company_id: str | None, kind: str, round_label: str, occurred_at: str | None, session_dir: Path, artifact_path: str, ai_review_path: str | None = None) -> Interview:
        iid = f"int-{uuid.uuid4().hex[:12]}"
        with self.transaction() as c:
            c.execute(
                """
                INSERT INTO interviews VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_dir, artifact_path) DO UPDATE SET
                    job_id=excluded.job_id,
                    company_id=excluded.company_id,
                    kind=excluded.kind,
                    round_label=excluded.round_label,
                    occurred_at=excluded.occurred_at,
                    ai_review_path=COALESCE(excluded.ai_review_path, interviews.ai_review_path)
                """,
                (iid, job_id, company_id, kind, round_label, occurred_at, str(session_dir), artifact_path, ai_review_path),
            )
            row = c.execute(
                "SELECT * FROM interviews WHERE session_dir=? AND artifact_path=?",
                (str(session_dir), artifact_path),
            ).fetchone()
        return Interview(**dict(row))

    def record_migration_run(self, run_id: str, started_at: str, completed_at: str, source_root: Path, summary: dict) -> None:
        with self.transaction() as c:
            c.execute("UPDATE migration_runs SET completed_at=?, source_root=?, summary_json=? WHERE id=?", (completed_at, str(source_root), json.dumps(summary, ensure_ascii=False), run_id))

    def connection(self) -> sqlite3.Connection:
        return self._connect()
