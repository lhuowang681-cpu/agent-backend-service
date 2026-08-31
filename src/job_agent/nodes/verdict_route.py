from __future__ import annotations

from job_agent.schemas import FitVerdictResult, Verdict, VerdictRouteDecision


def build_verdict_route(result: FitVerdictResult) -> VerdictRouteDecision:
    if result.verdict == Verdict.STRONG:
        return VerdictRouteDecision(
            verdict=result.verdict,
            route="apply_and_interview",
            gate="continue",
            next_actions=["apply", "prepare_interview"],
            rationale="Evidence coverage is sufficient for application and targeted interview preparation.",
        )
    if result.verdict == Verdict.WEAK:
        return VerdictRouteDecision(
            verdict=result.verdict,
            route="upgrade_evidence_then_apply",
            gate="upgrade_required",
            next_actions=["revise_resume", "prepare_interview"],
            rationale="The role remains plausible, but missing evidence should be upgraded before application.",
        )
    if result.verdict == Verdict.RISKY:
        return VerdictRouteDecision(
            verdict=result.verdict,
            route="evidence_upgrade_before_apply",
            gate="human_review",
            next_actions=["revise_resume", "prepare_interview"],
            rationale="High-risk evidence requires human review and focused preparation before application.",
        )
    return VerdictRouteDecision(
        verdict=result.verdict,
        route="return_to_selection_gate",
        gate="stop_and_reselect",
        next_actions=["switch_job"],
        rationale="The current role is not recommended; return to the selection gate and preserve this session.",
    )
