from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from job_agent.atomic_io import atomic_write_json


REAL_INTERVIEW_STAGES = (
    "一面",
    "二面",
    "三面",
    "HR 面",
    "终面",
    "其他",
)
_ROUND_FILE = re.compile(
    r"^13_real_interview_round_"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{6})"
    r"(?:_(?P<suffix>\d+))?\.json$"
)
_NOTE_HEADERS = {
    "## Captured Interview Notes",
    "## 本次面试记录",
    "## 面试原始记录",
}


@dataclass(frozen=True)
class RealInterviewQuestionRecord:
    question_id: str
    question: str
    answer: str = ""
    interviewer_follow_up: str = ""
    interviewer_feedback: str = ""
    self_assessment: str = "未判断"


class _QuestionPayload(BaseModel):
    question_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(default="", max_length=12000)
    interviewer_follow_up: str = Field(default="", max_length=4000)
    interviewer_feedback: str = Field(default="", max_length=4000)
    self_assessment: str = Field(default="未判断", max_length=40)


@dataclass(frozen=True)
class RealInterviewRound:
    round_id: str
    stage: str
    created_at: str
    notes: str
    path: Path
    questions: tuple[RealInterviewQuestionRecord, ...] = ()
    schema_version: int = 1


def _legacy_notes(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    captured: list[str] = []
    in_notes = False
    for line in lines:
        stripped = line.strip()
        if stripped in _NOTE_HEADERS:
            in_notes = True
            continue
        if in_notes and stripped.startswith("## "):
            break
        if in_notes and stripped:
            captured.append(stripped.removeprefix("- ").strip())
    if not captured and not any(line.strip() in _NOTE_HEADERS for line in lines):
        captured = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return "\n".join(captured).strip()


def _round_path(session_dir: Path, created_at: str) -> Path:
    parsed = datetime.fromisoformat(created_at)
    timestamp = parsed.strftime("%Y-%m-%dT%H%M%S")
    candidate = session_dir / f"13_real_interview_round_{timestamp}.json"
    suffix = 2
    while candidate.exists():
        candidate = session_dir / (
            f"13_real_interview_round_{timestamp}_{suffix}.json"
        )
        suffix += 1
    return candidate


def _write_round(
    session_dir: Path,
    *,
    stage: str,
    notes: str,
    created_at: str,
    questions: list[RealInterviewQuestionRecord] | None = None,
) -> RealInterviewRound:
    cleaned_stage = stage.strip()
    cleaned_notes = notes.strip()
    if not cleaned_stage:
        raise ValueError("real interview stage is required")
    if len(cleaned_stage) > 40:
        raise ValueError("real interview stage is too long")
    cleaned_questions = [
        _QuestionPayload.model_validate(
            question.__dict__
            if isinstance(question, RealInterviewQuestionRecord)
            else question
        )
        for question in (questions or [])
    ]
    if len(cleaned_notes) < 10 and not cleaned_questions:
        raise ValueError("real interview notes are too short")
    path = _round_path(Path(session_dir), created_at)
    payload = {
        "schema_version": 2,
        "round_id": path.stem,
        "stage": cleaned_stage,
        "created_at": created_at,
        "notes": cleaned_notes,
        "questions": [
            question.model_dump(mode="json")
            for question in cleaned_questions
        ],
    }
    atomic_write_json(path, payload)
    return RealInterviewRound(
        round_id=payload["round_id"],
        stage=cleaned_stage,
        created_at=created_at,
        notes=cleaned_notes,
        path=path,
        questions=tuple(
            RealInterviewQuestionRecord(**question.model_dump())
            for question in cleaned_questions
        ),
        schema_version=2,
    )


def list_real_interview_rounds(
    session_dir: Path,
) -> list[RealInterviewRound]:
    rounds: list[RealInterviewRound] = []
    for path in Path(session_dir).glob("13_real_interview_round_*.json"):
        if _ROUND_FILE.match(path.name) is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            question_payloads = payload.get("questions", [])
            if not isinstance(question_payloads, list):
                question_payloads = []
            questions = tuple(
                RealInterviewQuestionRecord(
                    **_QuestionPayload.model_validate(question).model_dump()
                )
                for question in question_payloads
            )
            item = RealInterviewRound(
                round_id=str(payload["round_id"]),
                stage=str(payload["stage"]),
                created_at=str(payload["created_at"]),
                notes=str(payload.get("notes", "")),
                path=path,
                questions=questions,
                schema_version=int(payload.get("schema_version", 1)),
            )
            datetime.fromisoformat(item.created_at)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if not item.stage.strip() or (not item.notes.strip() and not item.questions):
            continue
        rounds.append(item)
    return sorted(
        rounds,
        key=lambda item: (item.created_at, item.path.name),
        reverse=True,
    )


def archive_legacy_real_interview(
    session_dir: Path,
) -> RealInterviewRound | None:
    session_dir = Path(session_dir)
    if list_real_interview_rounds(session_dir):
        return None
    legacy_path = session_dir / "13_post_interview_review.md"
    notes = _legacy_notes(legacy_path)
    if len(notes) < 10:
        return None
    created_at = datetime.fromtimestamp(
        legacy_path.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")
    return _write_round(
        session_dir,
        stage="历史记录",
        notes=notes,
        created_at=created_at,
    )


def save_real_interview_round(
    session_dir: Path,
    *,
    stage: str,
    notes: str,
    questions: list[RealInterviewQuestionRecord] | None = None,
    now: Callable[[], datetime] | None = None,
) -> RealInterviewRound:
    current = (now or (lambda: datetime.now().astimezone()))()
    return _write_round(
        Path(session_dir),
        stage=stage,
        notes=notes,
        created_at=current.isoformat(timespec="seconds"),
        questions=questions,
    )


def suggested_real_interview_stage(session_dir: Path) -> str:
    existing = [
        item
        for item in reversed(list_real_interview_rounds(session_dir))
        if item.stage != "历史记录"
    ]
    sequence = ("一面", "二面", "三面", "终面")
    return sequence[min(len(existing), len(sequence) - 1)]
