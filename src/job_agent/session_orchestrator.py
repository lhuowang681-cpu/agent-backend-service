from __future__ import annotations

import json
import os
import re
from pathlib import Path

from job_agent.atomic_io import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_text_batch,
)
from job_agent.schemas import (
    ApplicationStatus,
    ArtifactFreshnessRecord,
    ArtifactFreshnessStatus,
    AnswerCard,
    AnswerCardDeck,
    EvidenceItem,
    EvidenceLevel,
    FitVerdictResult,
    MockInterviewAnswer,
    MockInterviewAnswerSet,
    MockInterviewPlan,
    MockInterviewQuestion,
    OrchestratorDecision,
    OrchestratorExecutionResult,
    RawJob,
    ResumeBullet,
    SessionIntent,
    SessionState,
    SelectionGateHandoff,
    StructuredJD,
    TargetedResume,
    Verdict,
)
from job_agent.nodes.answer_cards import build_answer_cards, render_answer_cards
from job_agent.nodes.interview_prep import prepare_interview, render_interview_prep
from job_agent.nodes.mock_debrief import render_mock_debrief, score_mock_interview
from job_agent.nodes.mock_interview import build_mock_interview_plan, render_mock_interview_plan
from job_agent.nodes.resume_tailoring import render_targeted_resume, tailor_resume
from job_agent.tools.application_tracker import (
    ApplicationTrackerStore,
    VALID_TRANSITIONS,
)


ARTIFACT_FILES = {
    "selected_job": "selected_job.json",
    "structured_jd": "01_jd_structured.json",
    "evidence_mapping": "02_evidence_mapping.json",
    "fit_verdict": "03_fit_verdict.json",
    "action_suggestion": "04_action_suggestion.md",
    "verdict_route": "05_verdict_route.json",
    "selection_handoff": "selection_handoff.json",
    "targeted_resume": "06_targeted_resume.md",
    "interview_grilling": "07_interview_grilling.md",
    "answer_cards": "08_answer_cards.md",
    "mock_interview_plan": "09_mock_interview_plan.md",
    "mock_interview_live": "10_mock_interview_live.md",
    "mock_answers": "mock_answers.json",
    "mock_debrief": "15_mock_interview_debrief.md",
    "post_interview_review": "13_post_interview_review.md",
    "tracker": "tracker.json",
}


RESUME_DOWNSTREAM_ARTIFACTS = [
    "interview_grilling",
    "answer_cards",
    "mock_interview_plan",
    "mock_interview_live",
    "mock_answers",
    "mock_debrief",
]


INTENT_KEYWORDS = {
    SessionIntent.MOCK_ANSWER_CAPTURE: [
        "answer:",
        "answer\uff1a",
        "\u56de\u7b54:",
        "\u56de\u7b54\uff1a",
        "\u5019\u9009\u4eba\u56de\u7b54:",
        "\u5019\u9009\u4eba\u56de\u7b54\uff1a",
    ],
    SessionIntent.MOCK_INTERVIEW: [
        "继续模拟面试",
        "模拟面试",
        "面试模拟",
        "mock interview",
        "\u5f00\u59cb\u7ec3",
        "\u9762\u8bd5\u7ec3\u4e60",
        "\u7ee7\u7eed\u95ee\u6211",
        "\u95ee\u6211\u95ee\u9898",
        "mock",
    ],
    SessionIntent.REVIEW: [
        "复盘面试",
        "面试复盘",
        "复盘",
        "review",
        "\u590d\u76d8\u521a\u624d",
        "\u9762\u8bd5\u53cd\u9988",
        "\u521a\u624d\u8868\u73b0",
        "\u6253\u5206",
        "debrief",
    ],
    SessionIntent.POST_INTERVIEW_REVIEW: [
        "interview notes:",
        "interview notes\uff1a",
        "real interview notes:",
        "real interview notes\uff1a",
        "post interview notes:",
        "post interview notes\uff1a",
        "\u9762\u8bd5\u8bb0\u5f55:",
        "\u9762\u8bd5\u8bb0\u5f55\uff1a",
        "\u771f\u5b9e\u9762\u8bd5\u590d\u76d8",
        "\u9762\u5b8c\u590d\u76d8",
    ],
    SessionIntent.RESUME_V2_FROM_REVIEW: [
        "revise resume from review",
        "resume v2 from review",
        "resume v2 from interview review",
        "resume from interview review",
        "review informed resume",
        "\u9762\u8bd5\u590d\u76d8\u540e\u6539\u7b80\u5386",
        "\u6839\u636e\u9762\u8bd5\u590d\u76d8\u6539\u7b80\u5386",
        "\u9762\u5b8c\u540e\u6539\u7b80\u5386",
    ],
    SessionIntent.INTERVIEW_PREP: [
        "准备面试",
        "面试准备",
        "答题卡",
        "interview prep",
        "prepare interview",
        "\u51c6\u5907\u56de\u7b54",
        "\u9762\u8bd5\u9898",
        "\u51c6\u5907\u7b54\u9898",
        "answer card",
    ],
    SessionIntent.RESUME_REVISION: [
        "修改简历",
        "改简历",
        "简历",
        "\u4f18\u5316\u7b80\u5386",
        "\u6da6\u8272\u7b80\u5386",
        "\u6539\u4e00\u4e0b\u7b80\u5386",
        "resume",
    ],
    SessionIntent.APPLICATION_STATE_UPDATE: [
        "mark applied",
        "mark as applied",
        "status applied",
        "application applied",
        "mark resume screen",
        "status resume screen",
        "application resume screen",
        "mark first interview",
        "status first interview",
        "application first interview",
        "mark rejected",
        "status rejected",
        "application rejected",
        "update application status",
        "\u5df2\u6295\u9012",
        "\u6807\u8bb0\u5df2\u6295\u9012",
        "\u66f4\u65b0\u4e3a\u5df2\u6295\u9012",
        "\u7b80\u5386\u7b5b\u9009",
        "\u4e00\u9762",
        "\u5df2\u62d2",
    ],
    SessionIntent.TRACK_APPLICATION: [
        "加入投递",
        "投递跟踪",
        "申请跟踪",
        "track",
        "\u8bb0\u5f55\u6295\u9012",
        "\u6295\u9012\u72b6\u6001",
        "\u66f4\u65b0\u6295\u9012",
        "tracker",
        "application",
    ],
    SessionIntent.SWITCH_JOB: [
        "换一个岗位",
        "换岗位",
        "重新选",
        "另一个岗位",
        "\u6362\u4e2a\u516c\u53f8",
        "\u770b\u4e0b\u4e00\u4e2a",
        "\u4e0b\u4e00\u4e2a\u5c97\u4f4d",
        "switch job",
        "another job",
    ],
}


GENERIC_CONTINUE_KEYWORDS = [
    "\u7ee7\u7eed\u4e0b\u4e00\u6b65",
    "\u63a5\u7740\u6765",
    "\u4e0b\u4e00\u6b65",
    "\u5f80\u4e0b",
    "\u7ee7\u7eed",
    "continue",
    "next step",
]


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_artifacts(session_dir: Path) -> dict[str, str]:
    return {name: filename for name, filename in ARTIFACT_FILES.items() if (session_dir / filename).exists()}


def _available_actions(artifact_paths: dict[str, str]) -> list[SessionIntent]:
    actions: list[SessionIntent] = []
    if {"structured_jd", "evidence_mapping"}.issubset(artifact_paths):
        actions.append(SessionIntent.RESUME_REVISION)
    if {"structured_jd", "evidence_mapping", "post_interview_review"}.issubset(artifact_paths):
        actions.append(SessionIntent.RESUME_V2_FROM_REVIEW)
    if {"structured_jd", "evidence_mapping", "fit_verdict", "targeted_resume"}.issubset(artifact_paths):
        actions.append(SessionIntent.INTERVIEW_PREP)
    if {"interview_grilling", "answer_cards"}.issubset(artifact_paths):
        actions.append(SessionIntent.MOCK_INTERVIEW)
    if "mock_interview_plan" in artifact_paths:
        actions.append(SessionIntent.MOCK_ANSWER_CAPTURE)
    if {"selected_job", "tracker"}.issubset(artifact_paths):
        actions.append(SessionIntent.APPLICATION_STATE_UPDATE)
    if {"selected_job", "fit_verdict"}.issubset(artifact_paths):
        actions.append(SessionIntent.TRACK_APPLICATION)
    if "selected_job" in artifact_paths:
        actions.append(SessionIntent.POST_INTERVIEW_REVIEW)
        actions.append(SessionIntent.SWITCH_JOB)
    return actions


def _pending_evidence_gaps(session_dir: Path) -> list[str]:
    evidence = _read_json(session_dir / ARTIFACT_FILES["evidence_mapping"])
    if not isinstance(evidence, list):
        return []
    gaps: list[str] = []
    for item in evidence:
        if item.get("level") in {"C0", "None"}:
            claim = item.get("claim", "unknown claim")
            risk = item.get("risk", "missing evidence")
            gaps.append(f"{claim}: {risk}")
    return gaps


def _load_artifact_freshness(session_dir: Path) -> dict[str, ArtifactFreshnessRecord]:
    payload = _read_json(session_dir / "session_state.json") or {}
    raw_records = payload.get("artifact_freshness", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_records, dict):
        return {}
    records: dict[str, ArtifactFreshnessRecord] = {}
    for artifact_name, record_payload in raw_records.items():
        if artifact_name not in ARTIFACT_FILES:
            continue
        try:
            records[artifact_name] = ArtifactFreshnessRecord.model_validate(record_payload)
        except (TypeError, ValueError):
            continue
    return records


def _build_artifact_freshness(
    session_dir: Path,
    artifact_paths: dict[str, str],
    *,
    fresh_artifacts: dict[str, str] | None = None,
    invalidated_artifacts: dict[str, str] | None = None,
    updated_by: str = "session_state_scan",
) -> dict[str, ArtifactFreshnessRecord]:
    fresh_artifacts = fresh_artifacts or {}
    invalidated_artifacts = invalidated_artifacts or {}
    mutation_names = set(fresh_artifacts) | set(invalidated_artifacts)
    unknown_names = mutation_names - set(ARTIFACT_FILES)
    if unknown_names:
        raise ValueError(f"Unknown artifact freshness keys: {', '.join(sorted(unknown_names))}.")
    overlap = set(fresh_artifacts) & set(invalidated_artifacts)
    if overlap:
        raise ValueError(f"Artifacts cannot be fresh and invalidated together: {', '.join(sorted(overlap))}.")

    previous = _load_artifact_freshness(session_dir)
    records = dict(previous)
    for artifact_name, path in artifact_paths.items():
        prior = previous.get(artifact_name)
        if prior is None:
            records[artifact_name] = ArtifactFreshnessRecord(
                path=path,
                revision=1,
                status=ArtifactFreshnessStatus.FRESH,
                updated_by="session_state_scan",
                note="Artifact present when session state was built.",
            )
        elif prior.status == ArtifactFreshnessStatus.INVALIDATED and artifact_name not in fresh_artifacts:
            records[artifact_name] = ArtifactFreshnessRecord(
                path=path,
                revision=prior.revision + 1,
                status=ArtifactFreshnessStatus.FRESH,
                updated_by="artifact_write_detected",
                note="Artifact reappeared after an invalidated state.",
            )

    for artifact_name, prior in previous.items():
        if artifact_name not in artifact_paths and prior.status == ArtifactFreshnessStatus.FRESH:
            records[artifact_name] = ArtifactFreshnessRecord(
                path=prior.path,
                revision=prior.revision,
                status=ArtifactFreshnessStatus.INVALIDATED,
                updated_by="artifact_missing",
                note="Artifact is no longer present in the session directory.",
            )

    for artifact_name, note in fresh_artifacts.items():
        prior = previous.get(artifact_name)
        records[artifact_name] = ArtifactFreshnessRecord(
            path=ARTIFACT_FILES[artifact_name],
            revision=prior.revision + 1 if prior is not None else 1,
            status=ArtifactFreshnessStatus.FRESH,
            updated_by=updated_by,
            note=note,
        )
    for artifact_name, note in invalidated_artifacts.items():
        prior = previous.get(artifact_name)
        records[artifact_name] = ArtifactFreshnessRecord(
            path=ARTIFACT_FILES[artifact_name],
            revision=prior.revision if prior is not None else 1,
            status=ArtifactFreshnessStatus.INVALIDATED,
            updated_by=updated_by,
            note=note,
        )
    return records


def build_session_state(
    session_dir: Path,
    *,
    fresh_artifacts: dict[str, str] | None = None,
    invalidated_artifacts: dict[str, str] | None = None,
    updated_by: str = "session_state_scan",
) -> SessionState:
    artifact_paths = _existing_artifacts(session_dir)
    selected_job = _read_json(session_dir / ARTIFACT_FILES["selected_job"]) or {}
    fit_verdict = _read_json(session_dir / ARTIFACT_FILES["fit_verdict"]) or {}
    verdict_value = fit_verdict.get("verdict")

    return SessionState(
        session_id=session_dir.name,
        job_id=selected_job.get("job_id"),
        company=selected_job.get("company"),
        title=selected_job.get("title"),
        version=1,
        latest_verdict=Verdict(verdict_value) if verdict_value else None,
        artifact_paths=artifact_paths,
        artifact_freshness=_build_artifact_freshness(
            session_dir,
            artifact_paths,
            fresh_artifacts=fresh_artifacts,
            invalidated_artifacts=invalidated_artifacts,
            updated_by=updated_by,
        ),
        available_actions=_available_actions(artifact_paths),
        pending_evidence_gaps=_pending_evidence_gaps(session_dir),
    )


def write_session_state(
    session_dir: Path,
    *,
    fresh_artifacts: dict[str, str] | None = None,
    invalidated_artifacts: dict[str, str] | None = None,
    updated_by: str = "session_state_scan",
) -> Path:
    state = build_session_state(
        session_dir,
        fresh_artifacts=fresh_artifacts,
        invalidated_artifacts=invalidated_artifacts,
        updated_by=updated_by,
    )
    path = session_dir / "session_state.json"
    return atomic_write_json(path, state.model_dump(mode="json"))


def infer_session_intent(user_request: str) -> SessionIntent:
    text = user_request.lower()
    normalized_text = text.replace("_", " ").replace("-", " ")
    post_interview_keywords = INTENT_KEYWORDS[SessionIntent.POST_INTERVIEW_REVIEW]
    if any(keyword.lower() in text for keyword in post_interview_keywords):
        return SessionIntent.POST_INTERVIEW_REVIEW
    resume_v2_keywords = INTENT_KEYWORDS[SessionIntent.RESUME_V2_FROM_REVIEW]
    if any(keyword.lower().replace("_", " ").replace("-", " ") in normalized_text for keyword in resume_v2_keywords):
        return SessionIntent.RESUME_V2_FROM_REVIEW
    application_state_keywords = INTENT_KEYWORDS[SessionIntent.APPLICATION_STATE_UPDATE]
    if any(keyword.lower().replace("_", " ").replace("-", " ") in normalized_text for keyword in application_state_keywords):
        return SessionIntent.APPLICATION_STATE_UPDATE
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return intent
    return SessionIntent.UNKNOWN


def _is_generic_continue_request(user_request: str) -> bool:
    text = user_request.lower()
    return any(keyword.lower() in text for keyword in GENERIC_CONTINUE_KEYWORDS)


def _default_intent_from_artifacts(artifact_paths: dict[str, str]) -> SessionIntent:
    if {"interview_grilling", "answer_cards"}.issubset(artifact_paths):
        return SessionIntent.MOCK_INTERVIEW
    if {"structured_jd", "evidence_mapping", "targeted_resume"}.issubset(artifact_paths):
        return SessionIntent.INTERVIEW_PREP
    if {"structured_jd", "evidence_mapping"}.issubset(artifact_paths):
        return SessionIntent.RESUME_REVISION
    if {"selected_job", "fit_verdict"}.issubset(artifact_paths):
        return SessionIntent.TRACK_APPLICATION
    if "selected_job" in artifact_paths:
        return SessionIntent.SWITCH_JOB
    return SessionIntent.UNKNOWN


ROUTING_REQUIREMENTS = {
    SessionIntent.RESUME_REVISION: ["01_jd_structured.json", "02_evidence_mapping.json"],
    SessionIntent.RESUME_V2_FROM_REVIEW: [
        "01_jd_structured.json",
        "02_evidence_mapping.json",
        "13_post_interview_review.md",
    ],
    SessionIntent.INTERVIEW_PREP: [
        "01_jd_structured.json",
        "02_evidence_mapping.json",
        "03_fit_verdict.json",
        "06_targeted_resume.md",
    ],
    SessionIntent.MOCK_INTERVIEW: ["07_interview_grilling.md", "08_answer_cards.md"],
    SessionIntent.MOCK_ANSWER_CAPTURE: ["09_mock_interview_plan.md"],
    SessionIntent.REVIEW: ["09_mock_interview_plan.md", "08_answer_cards.md", "mock_answers.json"],
    SessionIntent.POST_INTERVIEW_REVIEW: ["selected_job.json"],
    SessionIntent.APPLICATION_STATE_UPDATE: ["tracker.json", "selected_job.json"],
    SessionIntent.TRACK_APPLICATION: ["selected_job.json", "03_fit_verdict.json"],
    SessionIntent.SWITCH_JOB: ["selected_job.json"],
    SessionIntent.UNKNOWN: [],
}


NEXT_STEPS = {
    SessionIntent.RESUME_REVISION: "run_resume_tailoring",
    SessionIntent.RESUME_V2_FROM_REVIEW: "run_resume_v2_from_review",
    SessionIntent.INTERVIEW_PREP: "run_interview_prep",
    SessionIntent.MOCK_INTERVIEW: "run_mock_interview",
    SessionIntent.MOCK_ANSWER_CAPTURE: "capture_mock_answer",
    SessionIntent.REVIEW: "run_mock_debrief",
    SessionIntent.POST_INTERVIEW_REVIEW: "run_post_interview_review",
    SessionIntent.APPLICATION_STATE_UPDATE: "update_application_state",
    SessionIntent.TRACK_APPLICATION: "update_tracker",
    SessionIntent.SWITCH_JOB: "return_to_selection_gate",
    SessionIntent.UNKNOWN: "ask_clarifying_question",
}


def _missing_artifacts(session_dir: Path, required: list[str]) -> list[str]:
    return [filename for filename in required if not (session_dir / filename).exists()]


def _repair_step(intent: SessionIntent, missing: list[str]) -> str:
    if intent == SessionIntent.MOCK_INTERVIEW and missing == ["08_answer_cards.md"]:
        return "run_answer_cards_first"
    if intent == SessionIntent.MOCK_INTERVIEW:
        return "run_interview_prep_first"
    if intent == SessionIntent.MOCK_ANSWER_CAPTURE:
        return "run_mock_interview_first"
    if intent == SessionIntent.INTERVIEW_PREP:
        if missing == ["03_fit_verdict.json"]:
            return "run_fit_verdict_first"
        return "run_resume_tailoring_first"
    if intent == SessionIntent.RESUME_V2_FROM_REVIEW:
        if "13_post_interview_review.md" in missing:
            return "run_post_interview_review_first"
        return "run_jd_and_evidence_mapping_first"
    if intent == SessionIntent.RESUME_REVISION:
        return "run_jd_and_evidence_mapping_first"
    if intent == SessionIntent.APPLICATION_STATE_UPDATE:
        if "tracker.json" in missing:
            return "track_application_first"
        return "restore_selected_job_first"
    if intent == SessionIntent.TRACK_APPLICATION:
        return "restore_selected_job_or_verdict_first"
    if intent == SessionIntent.POST_INTERVIEW_REVIEW:
        return "restore_selected_job_first"
    if intent == SessionIntent.SWITCH_JOB:
        return "restore_selected_job_first"
    if intent == SessionIntent.REVIEW:
        return "collect_interview_transcript_first"
    return NEXT_STEPS[intent]


def plan_next_action(session_dir: Path, user_request: str) -> OrchestratorDecision:
    intent = infer_session_intent(user_request)
    if intent == SessionIntent.UNKNOWN and _is_generic_continue_request(user_request):
        intent = _default_intent_from_artifacts(_existing_artifacts(session_dir))
    required = ROUTING_REQUIREMENTS[intent]
    missing = _missing_artifacts(session_dir, required)
    runnable = not missing and intent != SessionIntent.UNKNOWN
    next_step = NEXT_STEPS[intent] if runnable else _repair_step(intent, missing)
    if intent == SessionIntent.UNKNOWN:
        message = "I could not infer the requested session action."
    elif runnable:
        message = f"Ready to execute {next_step}."
    else:
        message = f"Missing artifacts before {intent.value}: {', '.join(missing)}."
    return OrchestratorDecision(
        intent=intent,
        required_artifacts=required,
        missing_artifacts=missing,
        runnable=runnable,
        next_step=next_step,
        message=message,
    )


def _first_question_from_markdown(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            return stripped.removeprefix("### ").strip()
    return "Use the first question from 07_interview_grilling.md."


def _render_mock_interview_live(session_dir: Path) -> str:
    plan_path = session_dir / ARTIFACT_FILES["mock_interview_plan"]
    prep_path = session_dir / ARTIFACT_FILES["interview_grilling"]
    answer_path = session_dir / ARTIFACT_FILES["answer_cards"]
    source_path = plan_path if plan_path.exists() else prep_path
    source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    first_question = _first_question_from_markdown(source_text)

    return "\n".join(
        [
            "# Mock Interview Live Round",
            "",
            f"- Source plan: `{source_path.name}`",
            f"- Answer cards: `{answer_path.name}`",
            "- Mode: explicit orchestrator execution",
            "",
            "## Start",
            "",
            "Ask this question first:",
            "",
            f"> {first_question}",
            "",
            "## Live Rules",
            "",
            "- Ask one question at a time.",
            "- Do not reveal answer-card content during the live round.",
            "- Record candidate answers separately before writing a debrief.",
            "- Write `15_mock_interview_debrief.md` only after answers are collected and scored.",
        ]
    ).rstrip() + "\n"


def _execute_mock_interview(session_dir: Path) -> list[str]:
    filename = ARTIFACT_FILES["mock_interview_live"]
    write_session_state(session_dir)
    atomic_write_text(
        session_dir / filename,
        _render_mock_interview_live(session_dir),
    )
    write_session_state(
        session_dir,
        fresh_artifacts={"mock_interview_live": "Regenerated by the mock interview action."},
        updated_by="run_mock_interview",
    )
    return [filename]


def _write_regenerated_resume(
    session_dir: Path,
    markdown: str,
    *,
    updated_by: str,
    note: str,
) -> list[str]:
    filename = ARTIFACT_FILES["targeted_resume"]
    resume_path = session_dir / filename
    previous_markdown = resume_path.read_text(encoding="utf-8") if resume_path.exists() else None
    write_session_state(session_dir)
    atomic_write_text(resume_path, markdown)
    invalidated_artifacts: dict[str, str] = {}
    if previous_markdown != markdown:
        for artifact_name in RESUME_DOWNSTREAM_ARTIFACTS:
            stale_path = session_dir / ARTIFACT_FILES[artifact_name]
            if stale_path.exists():
                stale_path.unlink()
            invalidated_artifacts[artifact_name] = (
                f"Invalidated because {updated_by} produced a different targeted resume."
            )
    write_session_state(
        session_dir,
        fresh_artifacts={"targeted_resume": note},
        invalidated_artifacts=invalidated_artifacts,
        updated_by=updated_by,
    )
    return [filename]


def _execute_resume_revision(session_dir: Path) -> list[str]:
    structured_jd = StructuredJD.model_validate(_read_json(session_dir / ARTIFACT_FILES["structured_jd"]))
    evidence_payload = _read_json(session_dir / ARTIFACT_FILES["evidence_mapping"]) or []
    evidence = [EvidenceItem.model_validate(item) for item in evidence_payload]
    resume = tailor_resume(structured_jd=structured_jd, evidence=evidence)
    return _write_regenerated_resume(
        session_dir,
        render_targeted_resume(resume),
        updated_by="run_resume_tailoring",
        note="Regenerated from the current structured JD and evidence mapping.",
    )


def _extract_post_interview_review_notes(markdown: str) -> list[str]:
    notes: list[str] = []
    in_notes = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped in {"## Captured Interview Notes", "## 本次面试记录"}:
            in_notes = True
            continue
        if in_notes and stripped.startswith("## "):
            break
        if in_notes and stripped:
            notes.append(stripped)
    return notes or ["No captured interview notes were available."]


def _render_review_informed_resume(tailoring: TargetedResume, review_markdown: str) -> str:
    lines = [
        render_targeted_resume(tailoring).rstrip(),
        "",
        "## Post-Interview Review Signals",
        "",
        f"- Source artifact: `{ARTIFACT_FILES['post_interview_review']}`",
    ]
    for note in _extract_post_interview_review_notes(review_markdown):
        lines.append(f"- Interview signal: {note}")
    lines.extend(
        [
            "- Resume action: prioritize existing C1+ evidence that answers these signals before adding new claims.",
            "- Truth boundary: unsupported claims must stay out of the resume until `02_evidence_mapping.json` is upgraded.",
            "- Outcome boundary: this revision uses interview notes for emphasis only; it does not infer pass/fail or offer probability.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _execute_resume_v2_from_review(session_dir: Path) -> list[str]:
    structured_jd = StructuredJD.model_validate(_read_json(session_dir / ARTIFACT_FILES["structured_jd"]))
    evidence_payload = _read_json(session_dir / ARTIFACT_FILES["evidence_mapping"]) or []
    evidence = [EvidenceItem.model_validate(item) for item in evidence_payload]
    resume = tailor_resume(structured_jd=structured_jd, evidence=evidence)
    resume = resume.model_copy(
        update={
            "strategy_summary": (
                f"{resume.strategy_summary} Use post-interview notes as prioritization signals "
                "without weakening evidence boundaries."
            )
        }
    )
    review_markdown = (session_dir / ARTIFACT_FILES["post_interview_review"]).read_text(encoding="utf-8")
    return _write_regenerated_resume(
        session_dir,
        _render_review_informed_resume(resume, review_markdown),
        updated_by="run_resume_v2_from_review",
        note="Regenerated from the current JD, evidence mapping, and post-interview review.",
    )


def _section_lines(markdown: str, section_title: str) -> list[str]:
    lines: list[str] = []
    in_section = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == f"## {section_title}":
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped:
            lines.append(stripped)
    return lines


def _resume_bullet_from_markdown(line: str, fallback_level: EvidenceLevel) -> ResumeBullet | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    text = stripped.removeprefix("- ").strip()
    if text.startswith("No "):
        return None
    evidence_level = fallback_level
    evidence_summary = "Parsed from targeted resume artifact."
    risk = "Keep answer within the current resume artifact."
    match = re.search(r"\[evidence=([^;\]]+);\s*proof=([^;\]]*);\s*risk=([^\]]*)\]$", text)
    if match:
        try:
            evidence_level = EvidenceLevel(match.group(1).strip())
        except ValueError:
            evidence_level = fallback_level
        evidence_summary = match.group(2).strip() or evidence_summary
        risk = match.group(3).strip() or risk
        text = text[: match.start()].strip()
    return ResumeBullet(
        requirement_id="unknown",
        text=text,
        evidence_level=evidence_level,
        evidence_summary=evidence_summary,
        risk=risk,
    )


def _resume_bullets_from_section(markdown: str, section_title: str, fallback_level: EvidenceLevel) -> list[ResumeBullet]:
    bullets: list[ResumeBullet] = []
    for line in _section_lines(markdown, section_title):
        bullet = _resume_bullet_from_markdown(line, fallback_level)
        if bullet is not None:
            bullets.append(bullet)
    return bullets


def _resume_strategy_from_markdown(markdown: str) -> str:
    for line in _section_lines(markdown, "Resume Strategy"):
        stripped = line.strip()
        if stripped.startswith("- "):
            return stripped.removeprefix("- ").strip()
    return "Use the current targeted resume artifact as the interview-prep input."


def _parse_targeted_resume_markdown(markdown: str, structured_jd: StructuredJD) -> TargetedResume:
    claims_to_remove = [
        line.removeprefix("- ").strip()
        for line in _section_lines(markdown, "Claims To Remove Or Downgrade")
        if line.startswith("- ") and not line.removeprefix("- ").startswith("No ")
    ]
    return TargetedResume(
        company=structured_jd.company,
        title=structured_jd.title,
        strategy_summary=_resume_strategy_from_markdown(markdown),
        conservative_bullets=_resume_bullets_from_section(markdown, "Conservative Bullets", EvidenceLevel.C2),
        standard_bullets=_resume_bullets_from_section(markdown, "Standard Bullets", EvidenceLevel.C1),
        stronger_after_evidence=_resume_bullets_from_section(markdown, "Stronger After Evidence", EvidenceLevel.C0),
        claims_to_remove=claims_to_remove,
    )


def _execute_interview_prep(session_dir: Path) -> list[str]:
    structured_jd = StructuredJD.model_validate(_read_json(session_dir / ARTIFACT_FILES["structured_jd"]))
    evidence_payload = _read_json(session_dir / ARTIFACT_FILES["evidence_mapping"]) or []
    evidence = [EvidenceItem.model_validate(item) for item in evidence_payload]
    fit_result = FitVerdictResult.model_validate(_read_json(session_dir / ARTIFACT_FILES["fit_verdict"]))
    targeted_resume = _parse_targeted_resume_markdown(
        (session_dir / ARTIFACT_FILES["targeted_resume"]).read_text(encoding="utf-8"),
        structured_jd,
    )
    interview_prep = prepare_interview(
        structured_jd=structured_jd,
        evidence=evidence,
        fit_result=fit_result,
        targeted_resume=targeted_resume,
    )
    answer_cards = build_answer_cards(
        interview_prep=interview_prep,
        evidence=evidence,
        targeted_resume=targeted_resume,
    )
    mock_plan = build_mock_interview_plan(interview_prep=interview_prep, answer_cards=answer_cards)
    prep_filename = ARTIFACT_FILES["interview_grilling"]
    cards_filename = ARTIFACT_FILES["answer_cards"]
    plan_filename = ARTIFACT_FILES["mock_interview_plan"]
    prep_markdown = render_interview_prep(interview_prep)
    cards_markdown = render_answer_cards(answer_cards)
    plan_markdown = render_mock_interview_plan(mock_plan)
    plan_path = session_dir / plan_filename
    previous_plan_markdown = plan_path.read_text(encoding="utf-8") if plan_path.exists() else None
    write_session_state(session_dir)
    atomic_write_text_batch(
        {
            session_dir / prep_filename: prep_markdown,
            session_dir / cards_filename: cards_markdown,
            plan_path: plan_markdown,
        }
    )
    invalidated_artifacts: dict[str, str] = {}
    if previous_plan_markdown != plan_markdown:
        for artifact_name in ["mock_interview_live", "mock_answers", "mock_debrief"]:
            stale_path = session_dir / ARTIFACT_FILES[artifact_name]
            if stale_path.exists():
                stale_path.unlink()
                invalidated_artifacts[artifact_name] = (
                    "Invalidated because run_interview_prep produced a different mock plan."
                )
    write_session_state(
        session_dir,
        fresh_artifacts={
            "interview_grilling": "Regenerated from the current JD, evidence, fit verdict, and targeted resume.",
            "answer_cards": "Regenerated from the current interview prep, evidence, and targeted resume.",
            "mock_interview_plan": "Regenerated from the current interview prep and answer cards.",
        },
        invalidated_artifacts=invalidated_artifacts,
        updated_by="run_interview_prep",
    )
    return [prep_filename, cards_filename, plan_filename]


def _mock_plan_metadata(markdown: str) -> tuple[str, str, str, str]:
    mode = "technical"
    company = "unknown"
    title = "unknown"
    first_question_id = "Q1"
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Mode:"):
            mode = stripped.removeprefix("- Mode:").strip() or mode
        elif stripped.startswith("- Company:"):
            company = stripped.removeprefix("- Company:").strip() or company
        elif stripped.startswith("- Role:"):
            title = stripped.removeprefix("- Role:").strip() or title
        elif stripped.startswith("### "):
            match = re.match(r"###\s+(Q\d+)\.", stripped, flags=re.IGNORECASE)
            if match:
                first_question_id = match.group(1).upper()
                break
    return mode, company, title, first_question_id


def _parse_mock_answer_request(user_request: str, default_question_id: str) -> tuple[str, str]:
    question_match = re.search(r"\b(Q\d+)\b", user_request, flags=re.IGNORECASE)
    question_id = question_match.group(1).upper() if question_match else default_question_id
    delimiters = [
        "\u5019\u9009\u4eba\u56de\u7b54\uff1a",
        "\u5019\u9009\u4eba\u56de\u7b54:",
        "\u56de\u7b54\uff1a",
        "\u56de\u7b54:",
        "answer\uff1a",
        "answer:",
    ]
    folded = user_request.casefold()
    for delimiter in delimiters:
        index = folded.find(delimiter.casefold())
        if index >= 0:
            return question_id, user_request[index + len(delimiter) :].strip()
    return question_id, ""


def _restore_mock_answers_after_capture_failure(
    session_dir: Path,
    answer_path: Path,
    previous_answer_text: str | None,
) -> None:
    try:
        if previous_answer_text is None:
            answer_path.unlink()
        else:
            atomic_write_text(answer_path, previous_answer_text)
    except OSError:
        write_session_state(
            session_dir,
            invalidated_artifacts={
                "mock_answers": "Invalidated because mock answer recovery failed after a capture write error.",
                "mock_debrief": "Invalidated because mock answer recovery failed after a capture write error.",
            },
            updated_by="capture_mock_answer",
        )
        raise


def _execute_mock_answer_capture(session_dir: Path, user_request: str) -> list[str]:
    plan_text = (session_dir / ARTIFACT_FILES["mock_interview_plan"]).read_text(encoding="utf-8")
    mode, company, title, default_question_id = _mock_plan_metadata(plan_text)
    question_id, answer_text = _parse_mock_answer_request(user_request, default_question_id)
    if not answer_text:
        raise ValueError("mock answer text is required after `answer:` or `回答：`.")

    filename = ARTIFACT_FILES["mock_answers"]
    answer_path = session_dir / filename
    answer = MockInterviewAnswer(question_id=question_id, answer=answer_text)
    if answer_path.exists():
        previous_answer_text = answer_path.read_text(encoding="utf-8")
        existing_answer_set = MockInterviewAnswerSet.model_validate(json.loads(previous_answer_text))
        previous_answer_payload = json.dumps(existing_answer_set.model_dump(mode="json"), ensure_ascii=False, indent=2)
        answer_set = existing_answer_set
        updated_answers: list[MockInterviewAnswer] = []
        replaced = False
        for existing in answer_set.answers:
            if existing.question_id == question_id:
                if not replaced:
                    updated_answers.append(answer)
                    replaced = True
                continue
            updated_answers.append(existing)
        if not replaced:
            updated_answers.append(answer)
        answer_set = answer_set.model_copy(
            update={
                "company": company,
                "title": title,
                "mode": mode,
                "answers": updated_answers,
            }
        )
    else:
        answer_set = MockInterviewAnswerSet(
            company=company,
            title=title,
            mode=mode,  # type: ignore[arg-type]
            answers=[answer],
        )
        previous_answer_text = None
        previous_answer_payload = None

    answer_payload = json.dumps(answer_set.model_dump(mode="json"), ensure_ascii=False, indent=2)
    write_session_state(session_dir)

    invalidated_artifacts: dict[str, str] = {}
    try:
        atomic_write_text(answer_path, answer_payload)
    except OSError:
        _restore_mock_answers_after_capture_failure(session_dir, answer_path, previous_answer_text)
        raise
    if previous_answer_payload != answer_payload:
        debrief_path = session_dir / ARTIFACT_FILES["mock_debrief"]
        if debrief_path.exists():
            try:
                debrief_path.unlink()
            except OSError:
                _restore_mock_answers_after_capture_failure(session_dir, answer_path, previous_answer_text)
                raise
        invalidated_artifacts["mock_debrief"] = "Invalidated because mock interview answers changed."
    write_session_state(
        session_dir,
        fresh_artifacts={"mock_answers": "Captured mock interview answer."},
        invalidated_artifacts=invalidated_artifacts,
        updated_by="capture_mock_answer",
    )
    return [filename]


def _clean_inline_value(value: str) -> str:
    return value.strip().strip("`").strip()


def _bullet_value(line: str, label: str) -> str | None:
    prefix = f"- {label}:"
    stripped = line.strip()
    if stripped.startswith(prefix):
        return _clean_inline_value(stripped.removeprefix(prefix))
    return None


def _section_bullets(markdown: str, section_title: str) -> list[str]:
    bullets: list[str] = []
    in_section = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == f"## {section_title}":
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped.startswith("- "):
            bullets.append(stripped.removeprefix("- ").strip())
    return bullets


def _parse_mock_plan(markdown: str) -> MockInterviewPlan:
    mode, company, title, _ = _mock_plan_metadata(markdown)
    persona = "fair senior engineer"
    difficulty = "realistic"
    questions: list[MockInterviewQuestion] = []
    current: dict[str, object] | None = None
    collecting_followups = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Persona:"):
            persona = stripped.removeprefix("- Persona:").strip() or persona
        elif stripped.startswith("- Difficulty:"):
            difficulty = stripped.removeprefix("- Difficulty:").strip() or difficulty
        elif stripped.startswith("### "):
            if current is not None:
                questions.append(_mock_question_from_fields(current))
            match = re.match(r"###\s+(Q\d+)\.\s+(.+)", stripped, flags=re.IGNORECASE)
            if match:
                current = {
                    "question_id": match.group(1).upper(),
                    "prompt": match.group(2).strip(),
                    "requirement_id": match.group(1).upper(),
                    "focus": "general",
                    "time_limit_seconds": 180,
                    "risk_flags": [],
                    "follow_ups": [],
                }
                collecting_followups = False
        elif current is not None:
            requirement_id = _bullet_value(stripped, "Requirement ID")
            focus = _bullet_value(stripped, "Focus")
            time_limit = _bullet_value(stripped, "Time limit")
            risk_flags = _bullet_value(stripped, "Risk flags")
            if requirement_id is not None:
                current["requirement_id"] = requirement_id
                collecting_followups = False
            elif focus is not None:
                current["focus"] = focus
                collecting_followups = False
            elif time_limit is not None:
                seconds = re.search(r"\d+", time_limit)
                current["time_limit_seconds"] = int(seconds.group(0)) if seconds else 180
                collecting_followups = False
            elif risk_flags is not None:
                current["risk_flags"] = [item.strip() for item in risk_flags.split(",") if item.strip()]
                collecting_followups = False
            elif stripped == "- Follow-ups:":
                collecting_followups = True
            elif collecting_followups and stripped.startswith("- "):
                followups = current.get("follow_ups")
                if isinstance(followups, list):
                    followups.append(stripped.removeprefix("- ").strip())

    if current is not None:
        questions.append(_mock_question_from_fields(current))
    if not questions:
        questions.append(
            MockInterviewQuestion(
                question_id="Q1",
                requirement_id="unknown",
                prompt="Use the first mock interview question.",
                focus="general",
            )
        )
    scoring_dimensions = _section_bullets(markdown, "Scoring Dimensions") or ["evidence_quality", "ownership_clarity"]
    live_rules = _section_bullets(markdown, "Live Rules") or ["Score captured answers against answer cards."]
    return MockInterviewPlan(
        company=company,
        title=title,
        mode=mode,  # type: ignore[arg-type]
        persona=persona,
        difficulty=difficulty,  # type: ignore[arg-type]
        questions=questions,
        scoring_dimensions=scoring_dimensions,
        live_rules=live_rules,
    )


def _mock_question_from_fields(fields: dict[str, object]) -> MockInterviewQuestion:
    return MockInterviewQuestion(
        question_id=str(fields["question_id"]),
        requirement_id=str(fields["requirement_id"]),
        prompt=str(fields["prompt"]),
        follow_ups=[str(item) for item in fields.get("follow_ups", [])],
        time_limit_seconds=int(fields.get("time_limit_seconds", 180)),
        focus=str(fields.get("focus", "general")),
        risk_flags=[str(item) for item in fields.get("risk_flags", [])],
    )


def _parse_answer_cards(markdown: str) -> AnswerCardDeck:
    company = "unknown"
    title = "unknown"
    cards: list[AnswerCard] = []
    current: dict[str, object] | None = None
    collecting_prompts = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Company:"):
            company = stripped.removeprefix("- Company:").strip() or company
        elif stripped.startswith("- Title:"):
            title = stripped.removeprefix("- Title:").strip() or title
        elif stripped.startswith("### "):
            if current is not None:
                cards.append(_answer_card_from_fields(current))
            question = re.sub(r"^###\s+\d+\.\s*", "", stripped).strip()
            current = {
                "requirement_id": "unknown",
                "question": question,
                "evidence_level": EvidenceLevel.NONE,
                "short_answer": "",
                "supporting_evidence": "No resume-safe supporting claim.",
                "boundary": "Do not overclaim unsupported evidence.",
                "practice_prompts": [],
            }
            collecting_prompts = False
        elif current is not None:
            requirement_id = _bullet_value(stripped, "Requirement ID")
            evidence_level = _bullet_value(stripped, "Evidence level")
            short_answer = _bullet_value(stripped, "Short answer")
            supporting_evidence = _bullet_value(stripped, "Supporting evidence")
            boundary = _bullet_value(stripped, "Truth boundary")
            if requirement_id is not None:
                current["requirement_id"] = requirement_id
                collecting_prompts = False
            elif evidence_level is not None:
                current["evidence_level"] = EvidenceLevel(evidence_level)
                collecting_prompts = False
            elif short_answer is not None:
                current["short_answer"] = short_answer
                collecting_prompts = False
            elif supporting_evidence is not None:
                current["supporting_evidence"] = supporting_evidence
                collecting_prompts = False
            elif boundary is not None:
                current["boundary"] = boundary
                collecting_prompts = False
            elif stripped == "- Practice prompts:":
                collecting_prompts = True
            elif collecting_prompts and stripped.startswith("- "):
                prompts = current.get("practice_prompts")
                if isinstance(prompts, list):
                    prompts.append(stripped.removeprefix("- ").strip())

    if current is not None:
        cards.append(_answer_card_from_fields(current))
    if not cards:
        cards.append(
            AnswerCard(
                requirement_id="unknown",
                question="Use the first mock interview question.",
                short_answer="No answer card was available.",
                evidence_level=EvidenceLevel.NONE,
                supporting_evidence="No resume-safe supporting claim.",
                boundary="Do not overclaim unsupported evidence.",
            )
        )
    return AnswerCardDeck(company=company, title=title, cards=cards)


def _answer_card_from_fields(fields: dict[str, object]) -> AnswerCard:
    return AnswerCard(
        requirement_id=str(fields["requirement_id"]),
        question=str(fields["question"]),
        short_answer=str(fields["short_answer"]),
        evidence_level=fields["evidence_level"],  # type: ignore[arg-type]
        supporting_evidence=str(fields["supporting_evidence"]),
        boundary=str(fields["boundary"]),
        practice_prompts=[str(item) for item in fields.get("practice_prompts", [])],
    )


def _execute_mock_debrief(session_dir: Path) -> list[str]:
    plan = _parse_mock_plan((session_dir / ARTIFACT_FILES["mock_interview_plan"]).read_text(encoding="utf-8"))
    answer_cards = _parse_answer_cards((session_dir / ARTIFACT_FILES["answer_cards"]).read_text(encoding="utf-8"))
    answer_set = MockInterviewAnswerSet.model_validate(_read_json(session_dir / ARTIFACT_FILES["mock_answers"]))
    debrief = score_mock_interview(plan=plan, answer_set=answer_set, answer_cards=answer_cards)
    filename = ARTIFACT_FILES["mock_debrief"]
    write_session_state(session_dir)
    atomic_write_text(session_dir / filename, render_mock_debrief(debrief))
    write_session_state(
        session_dir,
        fresh_artifacts={"mock_debrief": "Regenerated by the mock debrief action."},
        updated_by="run_mock_debrief",
    )
    return [filename]


def _parse_post_interview_notes(user_request: str) -> str:
    delimiters = [
        "real interview notes\uff1a",
        "real interview notes:",
        "post interview notes\uff1a",
        "post interview notes:",
        "interview notes\uff1a",
        "interview notes:",
        "\u9762\u8bd5\u8bb0\u5f55\uff1a",
        "\u9762\u8bd5\u8bb0\u5f55:",
    ]
    folded = user_request.casefold()
    for delimiter in delimiters:
        index = folded.find(delimiter.casefold())
        if index >= 0:
            return user_request[index + len(delimiter) :].strip()
    return user_request.strip()


def _render_post_interview_review(selected_job: RawJob, notes: str) -> str:
    return "\n".join(
        [
            f"# 真实面试复盘：{selected_job.company} · {selected_job.title}",
            "",
            "## 岗位信息",
            "",
            f"- 公司：{selected_job.company}",
            f"- 岗位：{selected_job.title}",
            f"- 地点：{selected_job.location if selected_job.location != 'unknown' else '未填写'}",
            "",
            "## 本次面试记录",
            "",
            notes,
            "",
            "## 复盘检查",
            "",
            "- 哪些回答有代码、日志、文档或实验结果支撑？",
            "- 哪些问题暴露了知识盲区、表达卡顿或个人贡献说不清？",
            "- 面试官的追问是否说明简历强调方向与岗位关注点不一致？",
            "",
            "## 下一步动作",
            "",
            "- 为最薄弱的问题补一份 60 秒回答和可验证依据。",
            "- 如果面试官持续关注简历中不突出的主题，再调整岗位版简历。",
            "- 下次模拟面试优先重练本次卡顿或不会的问题。",
            "",
            "说明：这里只做复盘整理，不推断面试结果或通过概率。",
        ]
    ).rstrip() + "\n"


def _execute_post_interview_review(session_dir: Path, user_request: str) -> list[str]:
    notes = _parse_post_interview_notes(user_request)
    if not notes:
        raise ValueError("post-interview notes are required after `interview notes:`.")
    selected_job = RawJob.model_validate(_read_json(session_dir / ARTIFACT_FILES["selected_job"]))
    filename = ARTIFACT_FILES["post_interview_review"]
    write_session_state(session_dir)
    atomic_write_text(
        session_dir / filename,
        _render_post_interview_review(selected_job, notes),
    )
    write_session_state(
        session_dir,
        fresh_artifacts={"post_interview_review": "Regenerated by the post-interview review action."},
        updated_by="run_post_interview_review",
    )
    return [filename]


def _restore_tracker_after_freshness_failure(tracker_path: Path, previous_tracker_bytes: bytes | None) -> None:
    if previous_tracker_bytes is None:
        if tracker_path.exists():
            tracker_path.unlink()
        return
    restore_path = tracker_path.with_suffix(tracker_path.suffix + ".restore.tmp")
    try:
        restore_path.write_bytes(previous_tracker_bytes)
        restore_path.replace(tracker_path)
    finally:
        if restore_path.exists():
            restore_path.unlink()


def _canonical_tracker_path(session_dir: Path) -> Path:
    if session_dir.parent.name == "sessions":
        return session_dir.parent.parent / ARTIFACT_FILES["tracker"]
    return session_dir.parent / ARTIFACT_FILES["tracker"]


def _ensure_canonical_tracker(session_dir: Path) -> Path:
    canonical_path = _canonical_tracker_path(session_dir)
    snapshot_path = session_dir / ARTIFACT_FILES["tracker"]
    if not canonical_path.exists() and snapshot_path.exists():
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = canonical_path.with_suffix(canonical_path.suffix + ".migration.tmp")
        try:
            temp_path.write_bytes(snapshot_path.read_bytes())
            temp_path.replace(canonical_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return canonical_path


def _execute_tracker_update(session_dir: Path) -> list[str]:
    tracker_path = session_dir / ARTIFACT_FILES["tracker"]
    canonical_path = _ensure_canonical_tracker(session_dir)
    previous_tracker_bytes = tracker_path.read_bytes() if tracker_path.exists() else None
    previous_canonical_bytes = canonical_path.read_bytes() if canonical_path.exists() else None
    selected_job = RawJob.model_validate(_read_json(session_dir / ARTIFACT_FILES["selected_job"]))
    fit_result = FitVerdictResult.model_validate(_read_json(session_dir / ARTIFACT_FILES["fit_verdict"]))
    write_session_state(session_dir)
    store = ApplicationTrackerStore(canonical_path)
    try:
        record = store.add_from_state(
            user_id="default-user",
            state={"selected_job": selected_job, "fit_result": fit_result},
            session_dir=session_dir,
            status=ApplicationStatus.TO_APPLY,
            notes="tracked by Session Orchestrator",
            user_confirmed=True,
        )
        store.write_session_snapshot(
            user_id="default-user",
            session_dir=session_dir,
            application_id=record.id,
        )
        write_session_state(
            session_dir,
            fresh_artifacts={"tracker": "Updated by the tracker action."},
            updated_by="update_tracker",
        )
    except OSError:
        _restore_tracker_after_freshness_failure(tracker_path, previous_tracker_bytes)
        _restore_tracker_after_freshness_failure(canonical_path, previous_canonical_bytes)
        raise
    return [ARTIFACT_FILES["tracker"]]


def _application_state_from_request(user_request: str) -> ApplicationStatus:
    text = user_request.lower()
    normalized_text = text.replace("_", " ").replace("-", " ")
    if any(keyword in normalized_text for keyword in ["resume screen", "\u7b80\u5386\u7b5b\u9009"]):
        return ApplicationStatus.RESUME_SCREEN
    if any(keyword in normalized_text for keyword in ["first interview", "\u4e00\u9762"]):
        return ApplicationStatus.FIRST_INTERVIEW
    if any(keyword in normalized_text for keyword in ["rejected", "\u5df2\u62d2"]):
        return ApplicationStatus.REJECTED
    if any(
        keyword in text
        for keyword in [
            "mark applied",
            "mark as applied",
            "status applied",
            "application applied",
            "applied",
            "\u5df2\u6295\u9012",
            "\u6807\u8bb0\u5df2\u6295\u9012",
            "\u66f4\u65b0\u4e3a\u5df2\u6295\u9012",
        ]
    ):
        return ApplicationStatus.APPLIED
    raise ValueError("Unsupported application state update. Currently supported: applied, resume_screen, first_interview, rejected.")


def _find_application_for_selected_job(tracker, selected_job: RawJob):
    if selected_job.job_id:
        for application in tracker.applications:
            if application.job_id == selected_job.job_id:
                return application
        return None
    company_key = selected_job.company.casefold().strip()
    title_key = selected_job.title.casefold().strip()
    for application in tracker.applications:
        if application.company.casefold().strip() == company_key and application.title.casefold().strip() == title_key:
            return application
    return None


def _validate_application_state_transition(application, new_state: ApplicationStatus) -> None:
    allowed = VALID_TRANSITIONS[application.current_state]
    if new_state not in allowed and new_state != application.current_state:
        raise ValueError(f"invalid transition: {application.current_state.value} -> {new_state.value}")


def _execute_application_state_update(session_dir: Path, user_request: str) -> list[str]:
    tracker_path = session_dir / ARTIFACT_FILES["tracker"]
    canonical_path = _ensure_canonical_tracker(session_dir)
    selected_job = RawJob.model_validate(_read_json(session_dir / ARTIFACT_FILES["selected_job"]))
    store = ApplicationTrackerStore(canonical_path)
    tracker = store.load(user_id="default-user")
    application = _find_application_for_selected_job(tracker, selected_job)
    if application is None:
        raise ValueError("No tracked application found for selected job. Run track application first.")
    new_state = _application_state_from_request(user_request)
    _validate_application_state_transition(application, new_state)
    previous_tracker_bytes = tracker_path.read_bytes() if tracker_path.exists() else None
    previous_canonical_bytes = canonical_path.read_bytes() if canonical_path.exists() else None
    write_session_state(session_dir)
    try:
        updated = store.update_status(
            user_id="default-user",
            application_id=application.id,
            new_status=new_state,
            notes=f"marked {new_state.value} by Session Orchestrator",
            user_confirmed=True,
        )
        store.write_session_snapshot(
            user_id="default-user",
            session_dir=session_dir,
            application_id=updated.id,
        )
        write_session_state(
            session_dir,
            fresh_artifacts={"tracker": "Updated by the application state action."},
            updated_by="update_application_state",
        )
    except OSError:
        _restore_tracker_after_freshness_failure(tracker_path, previous_tracker_bytes)
        _restore_tracker_after_freshness_failure(canonical_path, previous_canonical_bytes)
        raise
    return [ARTIFACT_FILES["tracker"]]


def _execute_selection_handoff(session_dir: Path) -> list[str]:
    selected_job = RawJob.model_validate(_read_json(session_dir / ARTIFACT_FILES["selected_job"]))
    fit_payload = _read_json(session_dir / ARTIFACT_FILES["fit_verdict"]) or {}
    source_verdict = Verdict(fit_payload["verdict"]) if fit_payload.get("verdict") else None
    funnel_report = session_dir.parent.parent / "00_job_funnel_report.md"
    preserved_artifacts = sorted(
        path.relative_to(session_dir).as_posix()
        for path in session_dir.rglob("*")
        if path.is_file() and path.name != ARTIFACT_FILES["selection_handoff"]
    )
    handoff = SelectionGateHandoff(
        handoff_type="selection_gate",
        source_session_id=session_dir.name,
        source_job_id=selected_job.job_id,
        source_company=selected_job.company,
        source_title=selected_job.title,
        source_verdict=source_verdict,
        funnel_report_path=(
            Path(os.path.relpath(funnel_report, session_dir)).as_posix() if funnel_report.exists() else None
        ),
        preserved_artifacts=preserved_artifacts,
        next_step="select_job",
        message="Current session and artifacts are preserved. Select another job at the Selection Gate.",
    )
    write_session_state(session_dir)
    handoff_path = session_dir / ARTIFACT_FILES["selection_handoff"]
    atomic_write_json(handoff_path, handoff.model_dump(mode="json"))
    write_session_state(
        session_dir,
        fresh_artifacts={"selection_handoff": "Wrote a preserved-session handoff for the Selection Gate."},
        updated_by="return_to_selection_gate",
    )
    return [ARTIFACT_FILES["selection_handoff"]]


def run_next_action(session_dir: Path, user_request: str) -> OrchestratorExecutionResult:
    decision = plan_next_action(session_dir, user_request)
    if not decision.runnable:
        write_session_state(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=False,
            written_artifacts=[],
            message=decision.message,
        )

    if decision.next_step == "run_mock_interview":
        written_artifacts = _execute_mock_interview(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_mock_interview.",
        )
    if decision.next_step == "run_resume_tailoring":
        written_artifacts = _execute_resume_revision(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_resume_tailoring.",
        )
    if decision.next_step == "run_resume_v2_from_review":
        written_artifacts = _execute_resume_v2_from_review(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_resume_v2_from_review.",
        )
    if decision.next_step == "run_interview_prep":
        written_artifacts = _execute_interview_prep(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_interview_prep.",
        )
    if decision.next_step == "capture_mock_answer":
        try:
            written_artifacts = _execute_mock_answer_capture(session_dir, user_request)
        except ValueError as exc:
            return OrchestratorExecutionResult(
                decision=decision,
                executed=False,
                written_artifacts=[],
                message=str(exc),
            )
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed capture_mock_answer.",
        )
    if decision.next_step == "run_mock_debrief":
        written_artifacts = _execute_mock_debrief(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_mock_debrief.",
        )
    if decision.next_step == "run_post_interview_review":
        try:
            written_artifacts = _execute_post_interview_review(session_dir, user_request)
        except ValueError as exc:
            return OrchestratorExecutionResult(
                decision=decision,
                executed=False,
                written_artifacts=[],
                message=str(exc),
            )
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed run_post_interview_review.",
        )
    if decision.next_step == "update_application_state":
        try:
            written_artifacts = _execute_application_state_update(session_dir, user_request)
        except ValueError as exc:
            return OrchestratorExecutionResult(
                decision=decision,
                executed=False,
                written_artifacts=[],
                message=str(exc),
            )
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed update_application_state.",
        )
    if decision.next_step == "update_tracker":
        written_artifacts = _execute_tracker_update(session_dir)
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed update_tracker.",
        )
    if decision.next_step == "return_to_selection_gate":
        try:
            written_artifacts = _execute_selection_handoff(session_dir)
        except (TypeError, ValueError) as exc:
            return OrchestratorExecutionResult(
                decision=decision,
                executed=False,
                written_artifacts=[],
                message=f"Cannot write selection handoff: {exc}",
            )
        return OrchestratorExecutionResult(
            decision=decision,
            executed=True,
            written_artifacts=written_artifacts,
            message="Executed return_to_selection_gate.",
        )

    return OrchestratorExecutionResult(
        decision=decision,
        executed=False,
        written_artifacts=[],
        message=f"Execution is not implemented for {decision.next_step}.",
    )
