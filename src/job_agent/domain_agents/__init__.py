"""Domain-level tool-using agents."""

from job_agent.domain_agents.application_material import (
    ApplicationMaterialAgent,
    ApplicationMaterialGoal,
    ApplicationMaterialResult,
    ClaimAuditEntry,
    ClaimAuditStatus,
    ResumeClaimCandidate,
)
from job_agent.domain_agents.application_ops import (
    ApplicationOpsAgent,
    ApplicationOpsGoal,
    ApplicationOpsResult,
)
from job_agent.domain_agents.opportunity_research import (
    OpportunityResearchAgent,
    OpportunityResearchGoal,
    OpportunityResearchResult,
    OpportunityResearchVerifier,
    build_opportunity_tool_registry,
)
from job_agent.domain_agents.interview_coach import (
    InterviewCoachAgent,
    InterviewCoachGoal,
    InterviewCoachResult,
)

__all__ = [
    "ApplicationMaterialAgent",
    "ApplicationMaterialGoal",
    "ApplicationMaterialResult",
    "ApplicationOpsAgent",
    "ApplicationOpsGoal",
    "ApplicationOpsResult",
    "ClaimAuditEntry",
    "ClaimAuditStatus",
    "InterviewCoachAgent",
    "InterviewCoachGoal",
    "InterviewCoachResult",
    "OpportunityResearchAgent",
    "OpportunityResearchGoal",
    "OpportunityResearchResult",
    "OpportunityResearchVerifier",
    "ResumeClaimCandidate",
    "build_opportunity_tool_registry",
]
