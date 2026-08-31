from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Sequence

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.checkpoint import CheckpointStore
from job_agent.agent_runtime.contracts import (
    AgentFinish,
    AgentRunResult,
    AgentStepRecord,
    ToolContext,
    ToolEffect,
    ToolObservationStatus,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, AgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.schema_diagnostics import safe_validation_error_detail
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.nodes.fit_verdict import evaluate_fit
from job_agent.schemas import (
    EvidenceItem,
    EvidenceLevel,
    FitInput,
    FitVerdictResult,
    StrictModel,
    StructuredJD,
)


READ_CONTEXT_TOOL = "materials.read_context"
AUDIT_CLAIMS_TOOL = "materials.audit_claims"
EVALUATE_FIT_TOOL = "materials.evaluate_fit"
APPLICATION_MATERIAL_TOOLS = (
    READ_CONTEXT_TOOL,
    AUDIT_CLAIMS_TOOL,
    EVALUATE_FIT_TOOL,
)


class ClaimAuditStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NEEDS_CONFIRMATION = "needs_confirmation"


class ResumeClaimCandidate(StrictModel):
    claim_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ClaimAuditEntry(StrictModel):
    claim: ResumeClaimCandidate
    status: ClaimAuditStatus
    reason_code: str
    evidence_refs: list[str] = Field(default_factory=list)
    ungrounded_facts: list[str] = Field(default_factory=list)


class ApplicationMaterialGoal(StrictModel):
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = "Produce an evidence-grounded resume patch and deterministic fit assessment."


class ReadMaterialContextInput(StrictModel):
    include_resume: bool = True


class ReadMaterialContextOutput(StrictModel):
    structured_jd: StructuredJD
    evidence: list[EvidenceItem] = Field(min_length=1)
    resume_text: str = ""


class AuditClaimsInput(StrictModel):
    claims: list[ResumeClaimCandidate] = Field(min_length=1)


class AuditClaimsOutput(StrictModel):
    entries: list[ClaimAuditEntry] = Field(min_length=1)


class EvaluateFitInput(StrictModel):
    toy_signals: list[str] = Field(default_factory=list)


class EvaluateFitOutput(StrictModel):
    fit: FitVerdictResult
    evidence_refs: list[str] = Field(default_factory=list)


class ApplicationMaterialResult(StrictModel):
    company: str
    title: str
    fit: FitVerdictResult
    fit_evidence_refs: list[str] = Field(min_length=1)
    resume_patch: list[ResumeClaimCandidate] = Field(default_factory=list)
    claim_audit: list[ClaimAuditEntry] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)


_FACT_PATTERN = re.compile(
    r"(?i)(?<![\w.])\d+(?:\.\d+)?%?|\b(?:19|20)\d{2}\b|"
    r"\b(?:python|pytorch|tensorflow|jax|lora|qlora|rlhf|dpo|grpo|ppo|"
    r"langchain|langgraph|docker|kubernetes|redis|postgresql|mysql|cuda)\b"
)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _claim_facts(text: str, *, company: str) -> list[str]:
    facts = [match.group(0) for match in _FACT_PATTERN.finditer(text)]
    if company and _normalized_text(company) in _normalized_text(text):
        facts.append(company)
    unique: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        normalized = _normalized_text(fact)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(fact)
    return unique


def audit_resume_claims(
    claims: Sequence[ResumeClaimCandidate],
    *,
    evidence: Sequence[EvidenceItem],
    company: str,
) -> list[ClaimAuditEntry]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    entries: list[ClaimAuditEntry] = []
    for claim in claims:
        sources = [evidence_by_id[item_id] for item_id in claim.evidence_ids if item_id in evidence_by_id]
        missing_ids = sorted(set(claim.evidence_ids) - set(evidence_by_id))
        if missing_ids:
            entries.append(
                ClaimAuditEntry(
                    claim=claim,
                    status=ClaimAuditStatus.UNSUPPORTED,
                    reason_code="unknown_evidence_id",
                    ungrounded_facts=missing_ids,
                )
            )
            continue
        if any(item.requirement_id != claim.requirement_id for item in sources):
            entries.append(
                ClaimAuditEntry(
                    claim=claim,
                    status=ClaimAuditStatus.UNSUPPORTED,
                    reason_code="cross_requirement_evidence",
                    evidence_refs=[f"evidence:{item.evidence_id}" for item in sources],
                )
            )
            continue
        if any(item.level in {EvidenceLevel.C0, EvidenceLevel.NONE} for item in sources):
            entries.append(
                ClaimAuditEntry(
                    claim=claim,
                    status=ClaimAuditStatus.UNSUPPORTED,
                    reason_code="insufficient_evidence_level",
                    evidence_refs=[f"evidence:{item.evidence_id}" for item in sources],
                )
            )
            continue

        source_text = _normalized_text(" ".join(f"{item.claim} {item.proof}" for item in sources))
        facts = _claim_facts(claim.text, company=company)
        ungrounded = [fact for fact in facts if _normalized_text(fact) not in source_text]
        if ungrounded:
            entries.append(
                ClaimAuditEntry(
                    claim=claim,
                    status=ClaimAuditStatus.UNSUPPORTED,
                    reason_code="ungrounded_high_risk_fact",
                    evidence_refs=[f"evidence:{item.evidence_id}" for item in sources],
                    ungrounded_facts=ungrounded,
                )
            )
            continue

        if any(item.level == EvidenceLevel.C1 for item in sources):
            status = ClaimAuditStatus.NEEDS_CONFIRMATION
            reason_code = "c1_requires_confirmation"
        else:
            status = ClaimAuditStatus.SUPPORTED
            reason_code = "grounded_in_c2_c3_evidence"
        entries.append(
            ClaimAuditEntry(
                claim=claim,
                status=status,
                reason_code=reason_code,
                evidence_refs=[f"evidence:{item.evidence_id}" for item in sources],
            )
        )
    return entries


def build_application_material_registry(
    structured_jd: StructuredJD,
    evidence: Sequence[EvidenceItem],
    resume_text: str,
) -> ToolRegistry:
    evidence_items = list(evidence)
    registry = ToolRegistry()

    def read_context(value: ReadMaterialContextInput, context: ToolContext) -> ReadMaterialContextOutput:
        return ReadMaterialContextOutput(
            structured_jd=structured_jd,
            evidence=evidence_items,
            resume_text=resume_text if value.include_resume else "",
        )

    def audit_claims(value: AuditClaimsInput, context: ToolContext) -> AuditClaimsOutput:
        return AuditClaimsOutput(
            entries=audit_resume_claims(
                value.claims,
                evidence=evidence_items,
                company=structured_jd.company,
            )
        )

    def evaluate_material_fit(value: EvaluateFitInput, context: ToolContext) -> EvaluateFitOutput:
        fit = evaluate_fit(
            FitInput(
                role_type=structured_jd.role_type,
                requirements=structured_jd.must_have,
                evidence=evidence_items,
                toy_signals=value.toy_signals,
            )
        )
        return EvaluateFitOutput(
            fit=fit,
            evidence_refs=[f"evidence:{item.evidence_id}" for item in evidence_items],
        )

    registry.register(
        name=READ_CONTEXT_TOOL,
        description="Read the approved JD, evidence map, and resume text for this material run.",
        input_model=ReadMaterialContextInput,
        output_model=ReadMaterialContextOutput,
        handler=read_context,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=AUDIT_CLAIMS_TOOL,
        description="Audit proposed resume claims against exact evidence ids and high-risk fact anchors.",
        input_model=AuditClaimsInput,
        output_model=AuditClaimsOutput,
        handler=audit_claims,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=EVALUATE_FIT_TOOL,
        description="Compute the deterministic fit verdict from the approved JD and evidence map.",
        input_model=EvaluateFitInput,
        output_model=EvaluateFitOutput,
        handler=evaluate_material_fit,
        effect=ToolEffect.READ_ONLY,
    )
    return registry


class ApplicationMaterialVerifier:
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: Sequence[AgentStepRecord],
    ) -> VerificationResult:
        try:
            typed_goal = ApplicationMaterialGoal.model_validate(goal)
            result = ApplicationMaterialResult.model_validate(finish.result)
        except ValidationError as exc:
            detail = safe_validation_error_detail(
                exc,
                allowed_fields={
                    "company",
                    "title",
                    "fit",
                    "fit_evidence_refs",
                    "resume_patch",
                    "claim_audit",
                    "unresolved_gaps",
                    "claim",
                    "status",
                    "evidence_refs",
                },
            )
            return VerificationResult(
                passed=False,
                reason_code=f"application_material_schema_error:{detail}",
                feedback=(
                    "Return a valid material result with deterministic fit and claim audit fields. "
                    f"Shape diagnostic: {detail}"
                ),
            )
        except Exception:
            return VerificationResult(
                passed=False,
                reason_code="application_material_schema_error",
                feedback="Return a valid material result with deterministic fit and claim audit fields.",
            )
        if (result.company, result.title) != (typed_goal.company, typed_goal.title):
            return VerificationResult(
                passed=False,
                reason_code="material_target_mismatch",
                feedback="Keep company and title identical to the approved goal.",
            )

        read_seen = False
        observed_audits: dict[str, ClaimAuditEntry] = {}
        observed_fit: EvaluateFitOutput | None = None
        for step in steps:
            if step.observation.status not in {ToolObservationStatus.OK, ToolObservationStatus.EMPTY}:
                continue
            if step.action.tool_name == READ_CONTEXT_TOOL:
                read_seen = True
            elif step.action.tool_name == AUDIT_CLAIMS_TOOL:
                try:
                    audit_output = AuditClaimsOutput.model_validate(step.observation.data)
                except Exception:
                    continue
                observed_audits.update({entry.claim.claim_id: entry for entry in audit_output.entries})
            elif step.action.tool_name == EVALUATE_FIT_TOOL:
                try:
                    observed_fit = EvaluateFitOutput.model_validate(step.observation.data)
                except Exception:
                    continue

        if not read_seen:
            return VerificationResult(
                passed=False,
                reason_code="material_context_not_read",
                feedback="Read the approved material context before finishing.",
            )
        if observed_fit is None or result.fit != observed_fit.fit:
            return VerificationResult(
                passed=False,
                reason_code="fit_not_from_deterministic_tool",
                feedback="Use the exact fit result returned by materials.evaluate_fit.",
            )
        if set(result.fit_evidence_refs) != set(observed_fit.evidence_refs):
            return VerificationResult(
                passed=False,
                reason_code="fit_evidence_ref_mismatch",
                feedback="Use the exact evidence refs returned with the deterministic fit result.",
            )

        patch_ids = [claim.claim_id for claim in result.resume_patch]
        if len(patch_ids) != len(set(patch_ids)):
            return VerificationResult(
                passed=False,
                reason_code="duplicate_patch_claim_id",
                feedback="Use each audited claim at most once in the resume patch.",
            )
        for claim in result.resume_patch:
            audit = observed_audits.get(claim.claim_id)
            if audit is None or audit.claim != claim:
                return VerificationResult(
                    passed=False,
                    reason_code="claim_not_audited",
                    feedback=f"Claim {claim.claim_id} must exactly match an observed audit input.",
                )
            if audit.status != ClaimAuditStatus.SUPPORTED:
                return VerificationResult(
                    passed=False,
                    reason_code="unsupported_claim_in_patch",
                    feedback=f"Remove or reground claim {claim.claim_id}; its audit status is {audit.status.value}.",
                )

        expected_audit = [observed_audits[claim_id] for claim_id in sorted(observed_audits)]
        if sorted(result.claim_audit, key=lambda item: item.claim.claim_id) != expected_audit:
            return VerificationResult(
                passed=False,
                reason_code="claim_audit_mismatch",
                feedback="Copy the latest observed claim audit entries without changing them.",
            )
        evidence_refs = [
            ref
            for claim in result.resume_patch
            for ref in observed_audits[claim.claim_id].evidence_refs
        ]
        return VerificationResult(
            passed=True,
            reason_code="application_material_complete",
            feedback="All formal resume patch claims passed deterministic provenance checks.",
            evidence_refs=sorted(set([*result.fit_evidence_refs, *evidence_refs])),
        )


class ApplicationMaterialAgent:
    def __init__(
        self,
        *,
        model: AgentModel,
        structured_jd: StructuredJD,
        evidence: Sequence[EvidenceItem],
        resume_text: str,
        budget: AgentBudget | None = None,
        trajectory=None,
        checkpoint_store: CheckpointStore | None = None,
        max_same_verifier_reason: int | None = None,
    ) -> None:
        self.registry = build_application_material_registry(structured_jd, evidence, resume_text)
        policy_context = PolicyContext(
            skill_allowed_tools=APPLICATION_MATERIAL_TOOLS,
            agent_allowed_tools=APPLICATION_MATERIAL_TOOLS,
            runtime_allowed_tools=APPLICATION_MATERIAL_TOOLS,
        )
        self.loop = AgentLoop(
            agent_id="application-material",
            model=model,
            registry=self.registry,
            policy=PolicyEngine(),
            policy_context=policy_context,
            budget=BudgetManager(budget or AgentBudget.for_profile(BudgetProfile.STANDARD)),
            verifier=ApplicationMaterialVerifier(),
            trajectory=trajectory,
            checkpoint_store=checkpoint_store,
            skill_version="application-material-v1",
            prompt_version="application-material-controller-v1",
            max_same_verifier_reason=max_same_verifier_reason,
        )

    def run(
        self,
        goal: ApplicationMaterialGoal,
        *,
        session_id: str,
        run_id: str,
        workspace_root: Path | str,
        resume: bool = False,
    ) -> AgentRunResult:
        return self.loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal.model_dump(mode="json"),
            tool_context=ToolContext(
                session_id=session_id,
                run_id=run_id,
                agent_id="application-material",
                workspace_root=str(Path(workspace_root).resolve()),
            ),
            resume=resume,
        )
