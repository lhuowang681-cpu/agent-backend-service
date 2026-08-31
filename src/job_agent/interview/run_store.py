from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from job_agent.interview.contracts import (
    AdaptiveInterviewRun,
    InterviewBlueprint,
    InterviewContextSnapshot,
    InterviewTranscriptTurn,
    InterviewTurnDecision,
)
from job_agent.interview.paths import InterviewArtifactPaths


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class AdaptiveInterviewRunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def create(
        self,
        context: InterviewContextSnapshot,
        blueprint: InterviewBlueprint,
    ) -> AdaptiveInterviewRun:
        paths = InterviewArtifactPaths.for_run(self.root, context.run_id)
        if paths.run.exists():
            raise FileExistsError(context.run_id)
        now = _now()
        run = AdaptiveInterviewRun(
            run_id=context.run_id,
            kind=context.kind,
            status="waiting",
            context_path="context.json",
            blueprint=blueprint,
            started_at=now,
            updated_at=now,
        )
        _atomic_json(paths.context, context.model_dump(mode="json"))
        _atomic_json(paths.blueprint, blueprint.model_dump(mode="json"))
        _atomic_json(paths.run, run.model_dump(mode="json"))
        return run

    def load(self, run_id: str) -> AdaptiveInterviewRun:
        path = InterviewArtifactPaths.for_run(self.root, run_id).run
        return AdaptiveInterviewRun.model_validate_json(path.read_text(encoding="utf-8"))

    def commit_turn(
        self,
        run: AdaptiveInterviewRun,
        decision: InterviewTurnDecision,
        *,
        answer: str | None,
        elapsed_seconds: int,
    ) -> AdaptiveInterviewRun:
        current = self.load(run.run_id)
        existing_decision = next(
            (item for item in current.decision_log if item.turn_id == decision.turn_id),
            None,
        )
        if existing_decision is not None:
            if existing_decision != decision:
                raise ValueError("interview turn idempotency conflict")
            if run.pending_turn is not None:
                committed = next(
                    (item for item in current.transcript if item.turn_id == run.pending_turn.turn_id),
                    None,
                )
                if committed is None or committed.answer != answer or committed.elapsed_seconds != elapsed_seconds:
                    raise ValueError("interview answer idempotency conflict")
            return current
        if current.revision != run.revision:
            raise ValueError("stale interview run revision")
        transcript = list(current.transcript)
        decision_log = list(current.decision_log)
        if current.pending_turn is None:
            if answer is not None:
                raise ValueError("cannot answer without a pending interview question")
        else:
            previous = current.pending_turn
            if any(item.turn_id == previous.turn_id for item in transcript):
                return current
            if answer is None:
                raise ValueError("pending interview question requires an answer")
            transcript.append(
                InterviewTranscriptTurn(
                    turn_id=previous.turn_id,
                    phase=previous.phase,
                    target_id=previous.target_id,
                    question=previous.public_message,
                    answer=answer,
                    elapsed_seconds=elapsed_seconds,
                    question_origin=previous.question_origin,
                    source_refs=previous.source_refs,
                )
            )
        status = "completed" if decision.action == "finish" else "waiting"
        decision_log.append(decision)
        updated = current.model_copy(
            update={
                "revision": current.revision + 1,
                "status": status,
                "transcript": transcript,
                "decision_log": decision_log,
                "pending_turn": None if status == "completed" else decision,
                "active_seconds": current.active_seconds + elapsed_seconds,
                "turn_count": len(transcript),
                "updated_at": _now(),
                "failure_code": None,
            }
        )
        paths = InterviewArtifactPaths.for_run(self.root, run.run_id)
        _atomic_json(paths.run, updated.model_dump(mode="json"))
        if status == "completed":
            self._export_transcript(paths, updated)
        return updated

    def set_status(self, run_id: str, status: str) -> AdaptiveInterviewRun:
        current = self.load(run_id)
        if status not in {"waiting", "paused", "ended_by_user", "failed"}:
            raise ValueError("unsupported manual interview status")
        updated = current.model_copy(
            update={
                "revision": current.revision + 1,
                "status": status,
                "updated_at": _now(),
            }
        )
        paths = InterviewArtifactPaths.for_run(self.root, run_id)
        _atomic_json(paths.run, updated.model_dump(mode="json"))
        if status == "ended_by_user":
            self._export_transcript(paths, updated)
        return updated

    @staticmethod
    def _export_transcript(paths: InterviewArtifactPaths, run: AdaptiveInterviewRun) -> None:
        _atomic_json(
            paths.transcript,
            {
                "schema_version": run.schema_version,
                "run_id": run.run_id,
                "run_revision": run.revision,
                "status": run.status,
                "turns": [turn.model_dump(mode="json") for turn in run.transcript],
            },
        )
