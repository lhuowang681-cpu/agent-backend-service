from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from job_agent.graph import JobAgentState
from job_agent.schemas import (
    ApplicationRecord,
    ApplicationStatus,
    ApplicationTracker,
    TrackerEvent,
    TrackerReminder,
    TrackerStats,
)


VALID_TRANSITIONS = {
    ApplicationStatus.TO_APPLY: {
        ApplicationStatus.APPLIED,
        ApplicationStatus.ABANDONED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.APPLIED: {
        ApplicationStatus.RESUME_SCREEN,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.RESUME_SCREEN: {
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.FIRST_INTERVIEW: {
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.SECOND_INTERVIEW: {
        ApplicationStatus.OTHER_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.OTHER_INTERVIEW: {
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.HR_INTERVIEW: {
        ApplicationStatus.OFFER,
        ApplicationStatus.REJECTED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.OFFER: {
        ApplicationStatus.ACCEPTED,
        ApplicationStatus.REJECTED,
        ApplicationStatus.ABANDONED,
        ApplicationStatus.CLOSED,
    },
    ApplicationStatus.ACCEPTED: set(),
    ApplicationStatus.REJECTED: set(),
    ApplicationStatus.ABANDONED: set(),
    ApplicationStatus.CLOSED: set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_tracker(timestamp: str) -> ApplicationTracker:
    return ApplicationTracker(
        version=1,
        created=timestamp,
        updated=timestamp,
        applications=[],
        stats=TrackerStats(),
    )


def load_tracker(path: Path) -> ApplicationTracker:
    payload = _read_json(path)
    if payload is None:
        return _empty_tracker(_now())
    return ApplicationTracker.model_validate(payload)


def _save_tracker(path: Path, tracker: ApplicationTracker) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(tracker.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)
    ApplicationTracker.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _next_application_id(tracker: ApplicationTracker) -> str:
    return f"app-{len(tracker.applications) + 1:03d}"


def _find_duplicate(tracker: ApplicationTracker, job_id: str | None, company: str, title: str) -> ApplicationRecord | None:
    company_key = company.casefold().strip()
    title_key = title.casefold().strip()
    for application in tracker.applications:
        if job_id and application.job_id == job_id:
            return application
        if application.company.casefold().strip() == company_key and application.title.casefold().strip() == title_key:
            return application
    return None


def _event(status: ApplicationStatus, timestamp: str, notes: str) -> TrackerEvent:
    return TrackerEvent(state=status, timestamp=timestamp, notes=notes)


def _artifact_paths(session_dir: Path) -> dict[str, str]:
    state_path = session_dir / "session_state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        artifact_paths = payload.get("artifact_paths", {})
        if isinstance(artifact_paths, dict):
            return {str(key): str(value) for key, value in artifact_paths.items()}
    return {path.stem: path.name for path in session_dir.iterdir() if path.is_file()}


def _recalculate_stats(applications: list[ApplicationRecord]) -> TrackerStats:
    by_state: dict[str, int] = {}
    offer_count = 0
    for application in applications:
        state = application.current_state.value
        by_state[state] = by_state.get(state, 0) + 1
        if application.current_state == ApplicationStatus.OFFER or any(
            event.state == ApplicationStatus.OFFER for event in application.state_history
        ):
            offer_count += 1
    submitted_states = {
        ApplicationStatus.APPLIED,
        ApplicationStatus.RESUME_SCREEN,
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.OTHER_INTERVIEW,
        ApplicationStatus.HR_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.ACCEPTED,
        ApplicationStatus.REJECTED,
    }
    submitted = sum(
        application.current_state in submitted_states
        or any(event.state in submitted_states for event in application.state_history)
        for application in applications
    )
    response_states = {
        ApplicationStatus.RESUME_SCREEN,
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.OTHER_INTERVIEW,
        ApplicationStatus.HR_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.ACCEPTED,
        ApplicationStatus.REJECTED,
    }
    interview_states = {
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.OTHER_INTERVIEW,
        ApplicationStatus.HR_INTERVIEW,
        ApplicationStatus.OFFER,
        ApplicationStatus.ACCEPTED,
    }
    responses = sum(
        any(event.state in response_states for event in application.state_history)
        or application.current_state in response_states
        for application in applications
    )
    interviews = sum(
        any(event.state in interview_states for event in application.state_history)
        or application.current_state in interview_states
        for application in applications
    )
    return TrackerStats(
        total=len(applications),
        by_state=by_state,
        offer_count=offer_count,
        response_rate=f"{responses}/{submitted}",
        interview_rate=f"{interviews}/{responses}",
        offer_rate=f"{offer_count}/{interviews}",
    )


def _refresh_tracker(tracker: ApplicationTracker, timestamp: str) -> ApplicationTracker:
    return tracker.model_copy(
        update={
            "updated": timestamp,
            "stats": _recalculate_stats(tracker.applications),
        }
    )


def add_application_from_state(
    tracker_path: Path,
    state: JobAgentState,
    session_dir: Path,
    status: ApplicationStatus = ApplicationStatus.TO_APPLY,
    notes: str = "",
    timestamp: str | None = None,
) -> ApplicationRecord:
    timestamp = timestamp or _now()
    tracker = load_tracker(tracker_path)
    job = state["selected_job"]
    fit_result = state["fit_result"]
    existing = _find_duplicate(tracker, job.job_id, job.company, job.title)
    event = _event(status, timestamp, notes)

    if existing is None:
        record = ApplicationRecord(
            id=_next_application_id(tracker),
            job_id=job.job_id,
            company=job.company,
            title=job.title,
            city=job.location,
            url=job.url,
            source=job.source,
            current_state=status,
            verdict=fit_result.verdict,
            fit_score=fit_result.score,
            session_dir=str(session_dir),
            artifact_paths=_artifact_paths(session_dir),
            notes=notes,
            state_history=[event],
        )
        tracker.applications.append(record)
    else:
        record = existing.model_copy(
            update={
                "current_state": status,
                "verdict": fit_result.verdict,
                "fit_score": fit_result.score,
                "session_dir": str(session_dir),
                "artifact_paths": _artifact_paths(session_dir),
                "notes": notes or existing.notes,
                "state_history": [*existing.state_history, event],
            }
        )
        index = tracker.applications.index(existing)
        tracker.applications[index] = record

    tracker = _refresh_tracker(tracker, timestamp)
    _save_tracker(tracker_path, tracker)
    return record


def update_application_state(
    tracker_path: Path,
    application_id: str,
    new_state: ApplicationStatus,
    notes: str = "",
    timestamp: str | None = None,
) -> ApplicationRecord:
    timestamp = timestamp or _now()
    tracker = load_tracker(tracker_path)
    for index, application in enumerate(tracker.applications):
        if application.id != application_id:
            continue
        allowed = VALID_TRANSITIONS[application.current_state]
        if new_state not in allowed and new_state != application.current_state:
            raise ValueError(f"invalid transition: {application.current_state.value} -> {new_state.value}")
        event = _event(new_state, timestamp, notes)
        updated = application.model_copy(
            update={
                "current_state": new_state,
                "notes": notes or application.notes,
                "state_history": [*application.state_history, event],
            }
        )
        tracker.applications[index] = updated
        tracker = _refresh_tracker(tracker, timestamp)
        _save_tracker(tracker_path, tracker)
        return updated
    raise ValueError(f"application not found: {application_id}")


def render_tracker_dashboard(tracker: ApplicationTracker) -> str:
    lines = [
        "# Application Tracker Dashboard",
        "",
        f"- Total applications: {tracker.stats.total}",
        f"- Offers: {tracker.stats.offer_count}",
        "",
        "## By State",
        "",
    ]
    if tracker.stats.by_state:
        lines.extend(f"- {state}: {count}" for state, count in sorted(tracker.stats.by_state.items()))
    else:
        lines.append("- No applications yet.")
    lines.extend(["", "## Applications", ""])
    if not tracker.applications:
        lines.append("- No tracked applications.")
    for application in tracker.applications:
        lines.append(
            f"- {application.id}: {application.company} - {application.title} "
            f"[{application.current_state.value}; verdict={application.verdict.value if application.verdict else 'unknown'}]"
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _build_reminders(applications: list[ApplicationRecord], now: datetime) -> list[TrackerReminder]:
    reminders: list[TrackerReminder] = []
    interview_states = {
        ApplicationStatus.FIRST_INTERVIEW,
        ApplicationStatus.SECOND_INTERVIEW,
        ApplicationStatus.HR_INTERVIEW,
    }
    stuck_states = {ApplicationStatus.APPLIED, ApplicationStatus.RESUME_SCREEN, *interview_states}
    for application in applications:
        if application.next_deadline:
            deadline = _parse_timestamp(application.next_deadline)
            days_until = (deadline - now).total_seconds() / 86400
            if application.current_state in interview_states and 0 <= days_until <= 3:
                reminders.append(
                    TrackerReminder(
                        kind="interview_due",
                        application_id=application.id,
                        due_at=application.next_deadline,
                        message=f"Prepare for the upcoming interview for {application.company}.",
                    )
                )
        if not application.state_history:
            continue
        last_update = _parse_timestamp(application.state_history[-1].timestamp)
        age_days = (now - last_update).total_seconds() / 86400
        threshold = 14 if application.current_state == ApplicationStatus.TO_APPLY else 7
        if (
            application.current_state == ApplicationStatus.TO_APPLY
            or application.current_state in stuck_states
        ) and age_days > threshold:
            reminders.append(
                TrackerReminder(
                    kind="stuck",
                    application_id=application.id,
                    message=f"Application {application.id} has been in {application.current_state.value} for too long.",
                )
            )
    return reminders


class ApplicationTrackerStore:
    """Canonical user-level tracker with atomic writes and session snapshots."""

    def __init__(self, canonical_path: Path | str, *, now=lambda: datetime.now(UTC)) -> None:
        self.canonical_path = Path(canonical_path)
        self._now = now

    def load(self, *, user_id: str) -> ApplicationTracker:
        tracker = load_tracker(self.canonical_path)
        if tracker.applications and tracker.user_id != user_id:
            raise ValueError(f"tracker belongs to another user: {tracker.user_id}")
        if tracker.user_id != user_id:
            tracker = tracker.model_copy(update={"user_id": user_id})
        return tracker

    def current_timestamp(self) -> str:
        """Return the store clock as an ISO timestamp for coordinated mutations."""
        return self._now().isoformat()

    def add_application(
        self,
        *,
        user_id: str,
        application: ApplicationRecord,
        user_confirmed: bool,
    ) -> ApplicationRecord:
        if not user_confirmed:
            raise ValueError("user confirmation is required before changing tracker")
        tracker = self.load(user_id=user_id)
        existing = _find_duplicate(tracker, application.job_id, application.company, application.title)
        if existing is None:
            record = application
            tracker.applications.append(record)
        else:
            record = existing.model_copy(
                update={
                    "verdict": application.verdict or existing.verdict,
                    "fit_score": application.fit_score if application.fit_score is not None else existing.fit_score,
                    "session_dir": application.session_dir or existing.session_dir,
                    "artifact_paths": application.artifact_paths or existing.artifact_paths,
                    "notes": application.notes or existing.notes,
                    "state_history": [*existing.state_history, *application.state_history],
                }
            )
            tracker.applications[tracker.applications.index(existing)] = record
        self._persist(tracker)
        return self._find_required(record.id, user_id=user_id)

    def add_from_state(
        self,
        *,
        user_id: str,
        state: JobAgentState,
        session_dir: Path,
        status: ApplicationStatus = ApplicationStatus.TO_APPLY,
        notes: str = "",
        timestamp: str | None = None,
        user_confirmed: bool,
    ) -> ApplicationRecord:
        tracker = self.load(user_id=user_id)
        job = state["selected_job"]
        fit_result = state["fit_result"]
        existing = _find_duplicate(tracker, job.job_id, job.company, job.title)
        application_id = existing.id if existing is not None else _next_application_id(tracker)
        event = _event(status, timestamp or _now(), notes)
        record = ApplicationRecord(
            id=application_id,
            job_id=job.job_id,
            company=job.company,
            title=job.title,
            city=job.location,
            url=job.url,
            source=job.source,
            current_state=status,
            verdict=fit_result.verdict,
            fit_score=fit_result.score,
            session_dir=str(session_dir),
            artifact_paths=_artifact_paths(session_dir),
            notes=notes,
            state_history=[event],
        )
        return self.add_application(
            user_id=user_id,
            application=record,
            user_confirmed=user_confirmed,
        )

    def update_status(
        self,
        *,
        user_id: str,
        application_id: str,
        new_status: ApplicationStatus,
        notes: str,
        user_confirmed: bool,
        timestamp: str | None = None,
    ) -> ApplicationRecord:
        if not user_confirmed:
            raise ValueError("user confirmation is required before changing tracker")
        tracker = self.load(user_id=user_id)
        for index, application in enumerate(tracker.applications):
            if application.id != application_id:
                continue
            allowed = VALID_TRANSITIONS[application.current_state]
            if new_status not in allowed and new_status != application.current_state:
                raise ValueError(f"invalid transition: {application.current_state.value} -> {new_status.value}")
            event = _event(new_status, timestamp or _now(), notes)
            updated = application.model_copy(
                update={
                    "current_state": new_status,
                    "notes": notes or application.notes,
                    "state_history": [*application.state_history, event],
                }
            )
            tracker.applications[index] = updated
            self._persist(tracker)
            return self._find_required(application_id, user_id=user_id)
        raise ValueError(f"application not found: {application_id}")

    def write_session_snapshot(
        self,
        *,
        user_id: str,
        session_dir: Path,
        application_id: str,
    ) -> Path:
        tracker = self.load(user_id=user_id)
        if not any(application.id == application_id for application in tracker.applications):
            raise ValueError(f"application not found: {application_id}")
        snapshot = tracker.model_copy(
            update={
                "tracker_revision": tracker.revision,
                "application_id": application_id,
                "snapshot_of": str(self.canonical_path),
            }
        )
        snapshot_path = session_dir / "tracker.json"
        _save_tracker(snapshot_path, snapshot)
        return snapshot_path

    def _persist(self, tracker: ApplicationTracker) -> None:
        timestamp = self._now().isoformat()
        refreshed = tracker.model_copy(
            update={
                "revision": tracker.revision + 1,
                "updated": timestamp,
                "stats": _recalculate_stats(tracker.applications),
                "reminders": _build_reminders(tracker.applications, self._now()),
                "tracker_revision": None,
                "application_id": None,
                "snapshot_of": None,
            }
        )
        _save_tracker(self.canonical_path, refreshed)

    def _find_required(self, application_id: str, *, user_id: str) -> ApplicationRecord:
        tracker = self.load(user_id=user_id)
        for application in tracker.applications:
            if application.id == application_id:
                return application
        raise ValueError(f"application not found: {application_id}")
