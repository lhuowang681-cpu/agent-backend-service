from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from job_agent.atomic_io import atomic_write_json, atomic_write_text
from job_agent.graph import JobAgentState
from job_agent.nodes.answer_cards import render_answer_cards
from job_agent.nodes.interview_prep import render_interview_prep
from job_agent.nodes.mock_debrief import render_mock_debrief, score_mock_interview
from job_agent.nodes.mock_interview import render_mock_interview_plan
from job_agent.nodes.resume_tailoring import render_targeted_resume
from job_agent.reports import render_funnel_report
from job_agent.session_orchestrator import write_session_state
from job_agent.schemas import AnswerCardDeck, MockInterviewAnswerSet, MockInterviewPlan


@dataclass(frozen=True)
class SessionOutputPaths:
    output_dir: Path
    funnel_report: Path
    session_dir: Path


def _slugify(value: str) -> str:
    text = value.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "job"


def _session_slug(state: JobAgentState) -> str:
    job = state["selected_job"]
    url_text = job.url.lower()
    readable_role_text = " ".join([job.title, job.desc])
    force_posttraining = "posttraining" in url_text or "后训练" in readable_role_text
    role_slug = "posttraining" if force_posttraining else _slugify(job.title)
    return f"{_slugify(job.company)}_{role_slug}"


def _write_json(path: Path, payload) -> None:
    atomic_write_json(path, payload)


def _render_action_markdown(state: JobAgentState) -> str:
    action = state["action"]
    lines = [
        "# Action Suggestion",
        "",
        f"- Next action: `{action.next_action}`",
        f"- Summary: {action.summary}",
        "",
        "## Actions",
        "",
    ]
    lines.extend(f"- {item}" for item in action.actions)
    return "\n".join(lines).rstrip() + "\n"


def write_funnel_report(state: JobAgentState, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    funnel_report_path = output_dir / "00_job_funnel_report.md"
    return atomic_write_text(
        funnel_report_path,
        render_funnel_report(state["leads"]),
    )


def write_agent_runtime_failure_audit(
    output_dir: Path,
    *,
    runtime_mode: str,
    harness,
    error: Exception,
) -> Path:
    """Persist sanitized traces for a failed graph run without prompts or raw output."""

    internal_dir = output_dir / ".internal"
    internal_dir.mkdir(parents=True, exist_ok=True)
    audit_path = internal_dir / "agent_runtime_failure_audit.json"
    _write_json(
        audit_path,
        {
            "status": "failed",
            "runtime_mode": runtime_mode,
            "error": {
                "type": type(error).__name__,
                "error_code": getattr(error, "error_code", "agent_runtime_error"),
                "message": str(error),
            },
            "llm_call_count": len(harness.traces),
            "traces": [trace.model_dump(mode="json") for trace in harness.traces],
            "calls": [event.model_dump(mode="json") for event in harness.audit_events],
        },
    )
    return audit_path


def write_session_outputs(state: JobAgentState, output_dir: Path) -> SessionOutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)

    funnel_report_path = output_dir / "00_job_funnel_report.md"
    if "leads" in state:
        write_funnel_report(state, output_dir)

    session_dir = output_dir / "sessions" / _session_slug(state)
    session_dir.mkdir(parents=True, exist_ok=True)

    _write_json(session_dir / "selected_job.json", state["selected_job"].model_dump())
    _write_json(session_dir / "01_jd_structured.json", state["structured_jd"].model_dump())
    _write_json(session_dir / "02_evidence_mapping.json", state["fit_input"].model_dump()["evidence"])
    fit_verdict_path = session_dir / "03_fit_verdict.json"
    _write_json(fit_verdict_path, state["fit_result"].model_dump())
    if "fit_audit" in state:
        internal_dir = session_dir / ".internal"
        internal_dir.mkdir(parents=True, exist_ok=True)
        audit = state["fit_audit"].model_copy(update={"final_verdict_path": fit_verdict_path.name})
        _write_json(internal_dir / "03_fit_verdict_audit.json", audit.model_dump())
    if state.get("provider_traces"):
        internal_dir = session_dir / ".internal"
        internal_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            internal_dir / "agent_runtime_audit.json",
            {
                "runtime_mode": state.get("runtime_mode"),
                "llm_call_count": state.get("llm_call_count", len(state["provider_traces"])),
                "queue_wait_ms": state.get("queue_wait_ms", 0),
                "cache_hit": state.get("cache_hit", False),
                "traces": [trace.model_dump(mode="json") for trace in state["provider_traces"]],
                "calls": [
                    event.model_dump(mode="json")
                    for event in state.get("agent_audit_events", [])
                ],
            },
        )
    atomic_write_text(
        session_dir / "04_action_suggestion.md",
        _render_action_markdown(state),
    )
    _write_json(session_dir / "05_verdict_route.json", state["verdict_route"].model_dump(mode="json"))
    if "resume_tailoring" in state:
        atomic_write_text(
            session_dir / "06_targeted_resume.md",
            render_targeted_resume(state["resume_tailoring"]),
        )
    if "interview_prep" in state:
        atomic_write_text(
            session_dir / "07_interview_grilling.md",
            render_interview_prep(state["interview_prep"]),
        )
    if "answer_cards" in state:
        atomic_write_text(
            session_dir / "08_answer_cards.md",
            render_answer_cards(state["answer_cards"]),
        )
    if "mock_interview_plan" in state:
        atomic_write_text(
            session_dir / "09_mock_interview_plan.md",
            render_mock_interview_plan(state["mock_interview_plan"]),
        )
    write_session_state(session_dir)

    return SessionOutputPaths(
        output_dir=output_dir,
        funnel_report=funnel_report_path,
        session_dir=session_dir,
    )


def write_mock_debrief(
    session_dir: Path,
    plan: MockInterviewPlan,
    answer_set: MockInterviewAnswerSet,
    answer_cards: AnswerCardDeck,
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_json(session_dir / "mock_answers.json", answer_set.model_dump(mode="json"))
    debrief = score_mock_interview(plan=plan, answer_set=answer_set, answer_cards=answer_cards)
    debrief_path = session_dir / "15_mock_interview_debrief.md"
    atomic_write_text(debrief_path, render_mock_debrief(debrief))
    write_session_state(session_dir)
    return debrief_path
