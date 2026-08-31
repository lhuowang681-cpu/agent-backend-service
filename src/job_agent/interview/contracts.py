from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import StrictModel


ProjectFactKind = Literal["implementation", "ownership", "metric", "interpretation", "boundary"]
ProjectFactStatus = Literal["verified", "needs_confirmation", "confirmed", "rejected", "stale"]
InterviewKind = Literal["project_grill", "full_mock"]
InterviewPhase = Literal[
    "opening",
    "resume_project",
    "jd_technical",
    "fundamentals",
    "scenario_behavioral",
    "closing",
]
TurnAction = Literal["follow_up", "challenge", "switch_target", "transition_phase", "finish"]
QuestionOrigin = Literal[
    "resume_generated",
    "jd_generated",
    "project_generated",
    "answer_follow_up",
    "bank_adapted",
    "closing",
]


class ProjectFact(StrictModel):
    fact_id: str = Field(min_length=1)
    kind: ProjectFactKind
    statement: str = Field(min_length=1)
    status: ProjectFactStatus
    evidence_refs: list[str] = Field(default_factory=list)
    source_hashes: dict[str, str] = Field(default_factory=dict)


class ProjectDossier(StrictModel):
    schema_version: Literal[1] = 1
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source_fingerprint: str = Field(min_length=1)
    revision: int = Field(ge=1)
    facts: list[ProjectFact] = Field(min_length=1)
    confirmed_at: str | None = None

    @property
    def ready_for_interview(self) -> bool:
        return bool(self.confirmed_at) and any(
            fact.kind == "implementation" and fact.status in {"verified", "confirmed"}
            for fact in self.facts
        ) and not any(fact.status in {"needs_confirmation", "stale"} for fact in self.facts)


class InterviewSourceRef(StrictModel):
    ref_id: str = Field(min_length=1)
    kind: Literal["jd", "resume", "jd_evidence", "project_fact", "question_anchor", "transcript"]
    artifact_path: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)


class InterviewContextSnapshot(StrictModel):
    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1)
    kind: InterviewKind
    job_id: str | None = None
    company_id: str | None = None
    created_at: str = Field(min_length=1)
    source_refs: list[InterviewSourceRef] = Field(min_length=1)
    jd: dict | None = None
    resume_text: str | None = None
    jd_evidence: list[dict] = Field(default_factory=list)
    project_dossier: ProjectDossier | None = None

    @model_validator(mode="after")
    def validate_kind_sources(self):
        ref_ids = [ref.ref_id for ref in self.source_refs]
        if len(ref_ids) != len(set(ref_ids)):
            raise ValueError("source ref ids must be unique")
        if self.kind == "project_grill":
            if self.project_dossier is None or not self.project_dossier.ready_for_interview:
                raise ValueError("project grill requires a ready project dossier")
        elif (
            self.job_id is None
            or self.jd is None
            or not self.resume_text
            or not any(ref.kind == "jd_evidence" for ref in self.source_refs)
        ):
            raise ValueError("full mock requires job-linked JD, resume, and JD evidence")
        return self


class VerificationTarget(StrictModel):
    target_id: str = Field(min_length=1)
    competency: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    priority: Literal["must", "should", "optional"]
    source_refs: list[str] = Field(min_length=1)
    status: Literal["unseen", "probing", "verified", "unclear", "risk"] = "unseen"


class InterviewPhaseBudget(StrictModel):
    phase: InterviewPhase
    target_seconds: int = Field(gt=0)
    required: bool = True


class InterviewBlueprint(StrictModel):
    duration_minutes: Literal[15, 30, 45, 60]
    pressure: Literal["normal", "high"] = "normal"
    focus: str | None = None
    phases: list[InterviewPhaseBudget] = Field(min_length=1)
    targets: list[VerificationTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_targets(self):
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("target ids must be unique")
        phases = [item.phase for item in self.phases]
        if len(phases) != len(set(phases)):
            raise ValueError("phase budgets must be unique")
        return self


class TurnAssessment(StrictModel):
    answered: bool
    correctness: Literal["supported", "partially_supported", "unsupported", "not_applicable"]
    specificity: Literal["concrete", "mixed", "vague"]
    ownership: Literal["clear", "mixed", "unclear", "not_applicable"]
    conflicts: list[str] = Field(default_factory=list)
    new_leads: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class InterviewTurnDecision(StrictModel):
    turn_id: str = Field(min_length=1)
    action: TurnAction
    phase: InterviewPhase
    target_id: str | None = None
    public_message: str = Field(min_length=1)
    question_origin: QuestionOrigin
    assessment: TurnAssessment
    target_updates: dict[str, Literal["unseen", "probing", "verified", "unclear", "risk"]] = Field(
        default_factory=dict
    )
    source_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_finish_shape(self):
        if self.action == "finish":
            if self.target_id is not None or self.question_origin != "closing":
                raise ValueError("finish must close without a target")
        elif self.target_id is None:
            raise ValueError("non-finish turns require a target")
        return self


class InterviewTranscriptTurn(StrictModel):
    turn_id: str = Field(min_length=1)
    phase: InterviewPhase
    target_id: str | None = None
    question: str = Field(min_length=1)
    answer: str | None = None
    elapsed_seconds: int = Field(default=0, ge=0)
    question_origin: QuestionOrigin
    source_refs: list[str] = Field(default_factory=list)


class AdaptiveInterviewRun(StrictModel):
    schema_version: Literal[2] = 2
    revision: int = Field(default=1, ge=1)
    run_id: str = Field(min_length=1)
    kind: InterviewKind
    status: Literal["waiting", "paused", "completed", "ended_by_user", "failed"]
    context_path: str = Field(min_length=1)
    blueprint: InterviewBlueprint
    transcript: list[InterviewTranscriptTurn] = Field(default_factory=list)
    decision_log: list[InterviewTurnDecision] = Field(default_factory=list)
    pending_turn: InterviewTurnDecision | None = None
    active_seconds: int = Field(default=0, ge=0)
    turn_count: int = Field(default=0, ge=0)
    started_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    failure_code: str | None = None


class DimensionScore(StrictModel):
    dimension: Literal[
        "technical_correctness",
        "project_depth",
        "ownership_clarity",
        "evidence_specificity",
        "communication_structure",
        "role_fit",
    ]
    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class AnswerImprovementCard(StrictModel):
    turn_id: str = Field(min_length=1)
    issue: str = Field(min_length=1)
    recommended_structure: list[str] = Field(min_length=1)
    usable_fact_refs: list[str] = Field(default_factory=list)
    honesty_boundary: str = Field(min_length=1)
    retry_question: str = Field(min_length=1)


class AdaptiveInterviewDebrief(StrictModel):
    run_id: str = Field(min_length=1)
    conclusion: Literal["lean_pass", "borderline", "lean_no_pass", "insufficient_coverage"]
    conclusion_rationale: str = Field(min_length=1)
    conclusion_evidence_refs: list[str] = Field(min_length=1)
    dimensions: list[DimensionScore] = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    improvement_cards: list[AnswerImprovementCard] = Field(default_factory=list)
    final_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_all_dimensions_once(self) -> "AdaptiveInterviewDebrief":
        required = {
            "technical_correctness",
            "project_depth",
            "ownership_clarity",
            "evidence_specificity",
            "communication_structure",
            "role_fit",
        }
        observed = [item.dimension for item in self.dimensions]
        if len(observed) != len(required) or set(observed) != required:
            raise ValueError("debrief must contain each of the six dimensions exactly once")
        return self
