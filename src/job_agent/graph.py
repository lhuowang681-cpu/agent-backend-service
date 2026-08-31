from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from job_agent.nodes.answer_cards import build_answer_cards
from job_agent.nodes.action_suggestion import suggest_action
from job_agent.nodes.evidence_mapping import map_evidence
from job_agent.nodes.fit_verdict import FitVerdictBackend, RuleFitVerdictBackend
from job_agent.nodes.interview_prep import prepare_interview
from job_agent.nodes.jd_structurer import structure_jd
from job_agent.nodes.job_funnel import score_jobs
from job_agent.nodes.job_scout import scout_jobs
from job_agent.nodes.mock_interview import build_mock_interview_plan
from job_agent.nodes.resume_tailoring import tailor_resume
from job_agent.nodes.search_intent import build_search_intent
from job_agent.nodes.verdict_route import build_verdict_route
from job_agent.schemas import (
    ActionSuggestion,
    AnswerCardDeck,
    FitInput,
    FitVerdictAudit,
    FitVerdictResult,
    VerdictRouteDecision,
    InterviewPrep,
    JobLead,
    JobScoutResult,
    JobSourceKind,
    MockInterviewPlan,
    RawJob,
    SearchIntent,
    StructuredJD,
    TargetedResume,
)
from job_agent.llm.provider import ProviderTrace
from job_agent.llm.harness import InvocationAuditEvent
from job_agent.tools.job_search import build_manual_job


class JobAgentState(TypedDict, total=False):
    user_request: str
    search_intent: SearchIntent
    job_scout: JobScoutResult
    jobs: list[RawJob]
    leads: list[JobLead]
    selected_job: RawJob
    structured_jd: StructuredJD
    fit_input: FitInput
    fit_result: FitVerdictResult
    fit_audit: FitVerdictAudit
    action: ActionSuggestion
    verdict_route: VerdictRouteDecision
    resume_tailoring: TargetedResume
    interview_prep: InterviewPrep
    answer_cards: AnswerCardDeck
    mock_interview_plan: MockInterviewPlan
    provider_traces: list[ProviderTrace]
    agent_audit_events: list[InvocationAuditEvent]
    runtime_mode: str
    llm_call_count: int
    queue_wait_ms: int
    cache_hit: bool
    resumed_from_checkpoint: bool
    checkpoint_id: str


def _select_job(leads: list[JobLead], selected_job_id: str | None) -> RawJob:
    if selected_job_id is None:
        if not leads:
            raise ValueError("cannot auto-select a job from an empty lead list")
        return leads[0].job
    for lead in leads:
        if lead.job.job_id == selected_job_id:
            return lead.job
    raise ValueError(f"selected_job_id not found: {selected_job_id}")


def _run_selected_job_flow(
    user_request: str,
    selected_job: RawJob,
    resume_path: Path,
    fit_backend: FitVerdictBackend | None = None,
) -> JobAgentState:
    backend = fit_backend or RuleFitVerdictBackend()
    structured_jd = structure_jd(selected_job)
    evidence = map_evidence(resume_path, structured_jd.must_have)
    fit_input = FitInput(
        role_type=structured_jd.role_type,
        requirements=structured_jd.must_have,
        evidence=evidence,
        toy_signals=[],
    )
    fit_result = backend.evaluate(fit_input)
    action = suggest_action(fit_result)
    verdict_route = build_verdict_route(fit_result)
    resume_tailoring = tailor_resume(structured_jd=structured_jd, evidence=evidence)
    interview_prep = prepare_interview(
        structured_jd=structured_jd,
        evidence=evidence,
        fit_result=fit_result,
        targeted_resume=resume_tailoring,
    )
    answer_cards = build_answer_cards(
        interview_prep=interview_prep,
        evidence=evidence,
        targeted_resume=resume_tailoring,
    )
    mock_interview_plan = build_mock_interview_plan(
        interview_prep=interview_prep,
        answer_cards=answer_cards,
    )
    state: JobAgentState = {
        "user_request": user_request,
        "selected_job": selected_job,
        "structured_jd": structured_jd,
        "fit_input": fit_input,
        "fit_result": fit_result,
        "action": action,
        "verdict_route": verdict_route,
        "resume_tailoring": resume_tailoring,
        "interview_prep": interview_prep,
        "answer_cards": answer_cards,
        "mock_interview_plan": mock_interview_plan,
    }
    fit_audit = getattr(backend, "last_audit", None)
    if fit_audit is not None:
        state["fit_audit"] = fit_audit
    return state


def run_fixture_flow(
    user_request: str,
    job_fixture: Path,
    resume_path: Path,
    selected_job_id: str | None = None,
    fit_backend: FitVerdictBackend | None = None,
) -> JobAgentState:
    intent = build_search_intent(user_request)
    job_scout = scout_jobs(
        intent=intent,
        source_kind=JobSourceKind.FIXTURE,
        source_path=job_fixture,
    )
    return run_jobs_flow(
        user_request=user_request,
        jobs=job_scout.jobs,
        resume_path=resume_path,
        selected_job_id=selected_job_id,
        fit_backend=fit_backend,
        search_intent=job_scout.intent,
        job_scout=job_scout,
    )


def run_jobs_flow(
    user_request: str,
    jobs: list[RawJob],
    resume_path: Path,
    selected_job_id: str | None = None,
    fit_backend: FitVerdictBackend | None = None,
    search_intent: SearchIntent | None = None,
    job_scout: JobScoutResult | None = None,
) -> JobAgentState:
    intent = search_intent or build_search_intent(user_request)
    leads = score_jobs(jobs)
    selected_job = _select_job(leads, selected_job_id)
    state = _run_selected_job_flow(user_request, selected_job, resume_path, fit_backend)
    state.update(
        {
            "search_intent": intent,
            "jobs": jobs,
            "leads": leads,
        }
    )
    if job_scout is not None:
        state["job_scout"] = job_scout
    return state


def run_funnel_only(
    user_request: str,
    jobs: list[RawJob],
    search_intent: SearchIntent | None = None,
    job_scout: JobScoutResult | None = None,
) -> JobAgentState:
    intent = search_intent or build_search_intent(user_request)
    leads = score_jobs(jobs)
    state: JobAgentState = {
        "user_request": user_request,
        "search_intent": intent,
        "jobs": jobs,
        "leads": leads,
    }
    if job_scout is not None:
        state["job_scout"] = job_scout
    return state


def run_manual_jd_flow(
    jd_text: str,
    resume_path: Path,
    company: str = "Manual",
    title: str = "Manual JD",
    location: str = "unknown",
    fit_backend: FitVerdictBackend | None = None,
) -> JobAgentState:
    selected_job = build_manual_job(jd_text=jd_text, company=company, title=title, location=location)
    return _run_selected_job_flow(
        user_request="manual_jd",
        selected_job=selected_job,
        resume_path=resume_path,
        fit_backend=fit_backend,
    )


def run_semantic_job_flow(
    *,
    user_request: str,
    selected_job: RawJob,
    resume_path: Path,
    registry,
    harness,
    fit_backend: FitVerdictBackend | None = None,
    policy=None,
    runtime_mode=None,
    session_id: str | None = None,
    run_id: str | None = None,
    checkpoint_store=None,
    resume: bool = False,
    forbidden_output_strings: tuple[str, ...] = (),
    progress_callback=None,
) -> JobAgentState:
    """Run the API-driven semantic pipeline through the guarded graph runtime.

    This is deliberately separate from :func:`run_fixture_flow`: callers must
    provide an explicit registry and provider harness, while the runtime keeps
    schema validation, evidence guards, budgets, idempotency and audit policy.
    The function is a small public seam so CLI, notebooks and live harnesses do
    not reimplement the semantic-node ordering themselves.
    """
    from job_agent.runtime.graph_runtime import AgentGraphRuntime
    from job_agent.runtime.modes import RuntimeMode

    runtime = AgentGraphRuntime(
        registry=registry,
        harness=harness,
        fit_backend=fit_backend,
        policy=policy,
        runtime_mode=runtime_mode or RuntimeMode.AGENT_API,
        checkpoint_store=checkpoint_store,
        forbidden_output_strings=forbidden_output_strings,
        progress_callback=progress_callback,
    )
    state = runtime.run_selected_job(
        selected_job=selected_job,
        resume_text=resume_path.read_text(encoding="utf-8"),
        session_id=session_id or f"{selected_job.company}:{selected_job.job_id}",
        run_id=run_id or "semantic-flow",
        resume=resume,
    )
    state["user_request"] = user_request
    return state
