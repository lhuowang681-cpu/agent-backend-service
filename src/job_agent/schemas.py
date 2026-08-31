from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Mode(str, Enum):
    SEARCH = "search"
    SINGLE_JD = "single_jd"


class RoleType(str, Enum):
    POSTTRAINING = "posttraining"
    AGENTIC_RL = "agentic_rl"
    UNKNOWN = "unknown"


class EvidenceLevel(str, Enum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    NONE = "None"


class Verdict(str, Enum):
    STRONG = "strong fit"
    WEAK = "weak fit"
    RISKY = "risky fit"
    NOT_RECOMMENDED = "not recommended"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FitBackendKind(str, Enum):
    RULE = "rule"
    API = "api"
    LOCAL_LORA = "local_lora"


class RuntimeProfileKind(str, Enum):
    LOCAL_DEV = "local_dev"
    SERVER_AGENT = "server_agent"


class JobSourceKind(str, Enum):
    FIXTURE = "fixture"
    RAW_JOBS = "raw_jobs"


class SessionIntent(str, Enum):
    RESUME_REVISION = "resume_revision"
    RESUME_V2_FROM_REVIEW = "resume_v2_from_review"
    INTERVIEW_PREP = "interview_prep"
    MOCK_INTERVIEW = "mock_interview"
    MOCK_ANSWER_CAPTURE = "mock_answer_capture"
    REVIEW = "review"
    POST_INTERVIEW_REVIEW = "post_interview_review"
    TRACK_APPLICATION = "track_application"
    APPLICATION_STATE_UPDATE = "application_state_update"
    SWITCH_JOB = "switch_job"
    UNKNOWN = "unknown"


class ApplicationStatus(str, Enum):
    TO_APPLY = "to_apply"
    APPLIED = "applied"
    RESUME_SCREEN = "resume_screen"
    FIRST_INTERVIEW = "first_interview"
    SECOND_INTERVIEW = "second_interview"
    OTHER_INTERVIEW = "other_interview"
    HR_INTERVIEW = "hr_interview"
    OFFER = "offer"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"
    CLOSED = "closed"


class ArtifactFreshnessStatus(str, Enum):
    FRESH = "fresh"
    INVALIDATED = "invalidated"


class RuntimeConfig(StrictModel):
    runtime_profile: RuntimeProfileKind = RuntimeProfileKind.LOCAL_DEV
    fit_backend: FitBackendKind = FitBackendKind.RULE
    server_base_url: str | None = None
    server_model: str | None = None
    server_api_key_env: str | None = None
    local_model_path: str | None = None
    local_adapter_path: str | None = None

    @model_validator(mode="after")
    def validate_runtime_profile(self):
        if self.fit_backend == FitBackendKind.LOCAL_LORA and self.runtime_profile != RuntimeProfileKind.SERVER_AGENT:
            raise ValueError("local_lora requires runtime_profile=server_agent")
        return self


class SearchIntent(StrictModel):
    target_role: Literal["posttraining_rlhf"]
    cities: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)


class RawJob(StrictModel):
    job_id: str
    company: str
    title: str
    desc: str
    url: str
    location: str
    salary: str | None = None
    source: str = "fixture"
    job_type: str = "intern"
    posted_date: str | None = None
    fetched_at: str | None = None
    lead_score: int | None = Field(default=None, ge=0, le=100)
    lead_reason: str | None = None
    risk_flags: list[str] = Field(default_factory=list)


class JobScoutResult(StrictModel):
    intent: SearchIntent
    source_kind: JobSourceKind
    source_path: str
    jobs: list[RawJob] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobLead(StrictModel):
    job: RawJob
    score: int = Field(ge=0, le=100)
    bucket: Literal["Strong Fit", "Good Fit", "Risky Fit", "Not Recommended"]
    reasons: list[str]
    risks: list[str]


class JobRequirement(StrictModel):
    id: str
    text: str
    required: bool = True
    probe: str


class StructuredJD(StrictModel):
    company: str
    title: str
    role_type: RoleType
    must_have: list[JobRequirement] = Field(min_length=1)
    nice_to_have: list[str] = Field(default_factory=list)
    raw_jd: str


class EvidenceItem(StrictModel):
    evidence_id: str
    requirement_id: str
    claim: str
    level: EvidenceLevel
    proof: str
    risk: str


class EvidenceMappingResult(StrictModel):
    items: list[EvidenceItem] = Field(min_length=1)


class FitInput(StrictModel):
    role_type: RoleType
    requirements: list[JobRequirement] = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    toy_signals: list[str] = Field(default_factory=list)


class FitVerdictResult(StrictModel):
    verdict: Verdict
    score: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    need_human_review: bool
    reason_codes: list[str] = Field(min_length=1)
    explanation: str


class FitVerdictAudit(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    backend: FitBackendKind
    runtime_profile: RuntimeProfileKind
    model_path: str | None = None
    adapter_path: str | None = None
    raw_model_output: str
    extracted_json: dict[str, object] = Field(default_factory=dict)
    schema_valid: bool
    fallback_used: bool
    guard_applied: bool
    guard_reason_codes: list[str] = Field(default_factory=list)
    final_verdict_path: str | None = None
    error_message: str | None = None


class ActionSuggestion(StrictModel):
    next_action: Literal["apply", "apply_after_upgrade", "upgrade_first", "choose_another_job"]
    summary: str
    actions: list[str] = Field(min_length=1)


class VerdictRouteDecision(StrictModel):
    verdict: Verdict
    route: Literal[
        "apply_and_interview",
        "upgrade_evidence_then_apply",
        "evidence_upgrade_before_apply",
        "return_to_selection_gate",
    ]
    gate: Literal["continue", "upgrade_required", "human_review", "stop_and_reselect"]
    next_actions: list[str] = Field(min_length=1)
    rationale: str


class SelectionGateHandoff(StrictModel):
    handoff_type: Literal["selection_gate"]
    source_session_id: str
    source_job_id: str
    source_company: str
    source_title: str
    source_verdict: Verdict | None = None
    funnel_report_path: str | None = None
    preserved_artifacts: list[str] = Field(default_factory=list)
    next_step: Literal["select_job"]
    message: str


class ResumeBullet(StrictModel):
    requirement_id: str
    text: str
    evidence_level: EvidenceLevel
    evidence_summary: str
    risk: str


class TargetedResume(StrictModel):
    company: str
    title: str
    strategy_summary: str
    conservative_bullets: list[ResumeBullet] = Field(default_factory=list)
    standard_bullets: list[ResumeBullet] = Field(default_factory=list)
    stronger_after_evidence: list[ResumeBullet] = Field(default_factory=list)
    claims_to_remove: list[str] = Field(default_factory=list)


class InterviewQuestion(StrictModel):
    requirement_id: str
    question: str
    intent: str
    evidence_level: EvidenceLevel
    risk_note: str
    follow_ups: list[str] = Field(default_factory=list)


class InterviewPrep(StrictModel):
    company: str
    title: str
    strategy_summary: str
    questions: list[InterviewQuestion] = Field(min_length=1)


class AnswerCard(StrictModel):
    requirement_id: str
    question: str
    short_answer: str
    evidence_level: EvidenceLevel
    supporting_evidence: str
    boundary: str
    practice_prompts: list[str] = Field(default_factory=list)


class AnswerCardDeck(StrictModel):
    company: str
    title: str
    cards: list[AnswerCard] = Field(min_length=1)


class PostInterviewQuestionInput(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = ""
    interviewer_follow_up: str = ""
    interviewer_feedback: str = ""
    self_assessment: str = "未判断"


class PostInterviewRoundInput(StrictModel):
    round_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    notes: str = ""
    questions: list[PostInterviewQuestionInput] = Field(default_factory=list)


class PostInterviewGap(StrictModel):
    category: Literal[
        "knowledge",
        "expression",
        "project_evidence",
        "truth_boundary",
    ]
    priority: Literal["high", "medium", "low"]
    finding: str
    basis_question_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    action: str


class PostInterviewQuestionDiagnosis(StrictModel):
    question_id: str
    status: Literal[
        "strong",
        "partial",
        "weak",
        "insufficient_information",
    ]
    strengths: list[str] = Field(default_factory=list)
    gaps: list[PostInterviewGap] = Field(default_factory=list)
    improved_answer: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    practice_action: str


class PostInterviewReviewResult(StrictModel):
    company: str
    title: str
    round_id: str
    summary: str
    question_diagnoses: list[PostInterviewQuestionDiagnosis] = Field(
        default_factory=list
    )
    priority_gaps: list[PostInterviewGap] = Field(default_factory=list)
    learning_plan: list[str] = Field(default_factory=list)
    next_mock_topics: list[str] = Field(default_factory=list)
    resume_adjustments: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class MockInterviewQuestion(StrictModel):
    question_id: str
    requirement_id: str
    prompt: str
    follow_ups: list[str] = Field(default_factory=list)
    time_limit_seconds: int = Field(default=180, ge=30)
    focus: str
    risk_flags: list[str] = Field(default_factory=list)


class MockInterviewPlan(StrictModel):
    company: str
    title: str
    mode: Literal["technical", "behavioral", "mixed", "custom"] = "technical"
    persona: str
    difficulty: Literal["lenient", "realistic", "extra_tough"] = "realistic"
    questions: list[MockInterviewQuestion] = Field(min_length=1)
    scoring_dimensions: list[str] = Field(min_length=1)
    live_rules: list[str] = Field(min_length=1)


class MockInterviewAnswer(StrictModel):
    question_id: str
    answer: str
    elapsed_seconds: int = Field(default=0, ge=0)
    follow_up_answers: list[str] = Field(default_factory=list)


class MockInterviewAnswerSet(StrictModel):
    company: str
    title: str
    mode: Literal["technical", "behavioral", "mixed", "custom"] = "technical"
    answers: list[MockInterviewAnswer] = Field(min_length=1)


class RuleEvidenceAudit(StrictModel):
    """可复现的回答表面信号，不代表技术正确性或面试表现预测。"""

    question_id: str
    answer_present: bool
    answer_length_band: Literal["missing", "thin", "adequate"]
    evidence_signals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    ownership_clear: bool


class MockInterviewTranscriptItem(StrictModel):
    question_id: str
    requirement_id: str
    question: str
    answer: str = ""
    elapsed_seconds: int = Field(default=0, ge=0)


class MockAnswerEvaluation(StrictModel):
    question_id: str
    requirement_id: str
    status: Literal[
        "strong",
        "partial",
        "weak",
        "insufficient_information",
    ]
    relevance: int = Field(ge=1, le=5)
    technical_depth: int = Field(ge=1, le=5)
    reasoning_clarity: int = Field(ge=1, le=5)
    evidence_grounding: int = Field(ge=1, le=5)
    truth_boundary: int = Field(ge=1, le=5)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    improved_answer: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    next_focus: str = ""


class MockInterviewEvaluationResult(StrictModel):
    company: str
    title: str
    run_id: str
    question_evaluations: list[MockAnswerEvaluation] = Field(min_length=1)
    overall_score: float = Field(ge=1.0, le=5.0)
    strengths: list[str] = Field(default_factory=list)
    priority_gaps: list[str] = Field(default_factory=list)
    next_mock_topics: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class MockInterviewScore(StrictModel):
    question_id: str
    question: str
    answer_summary: str
    score: int = Field(ge=1, le=5)
    diagnosis: str
    risk_flags: list[str] = Field(default_factory=list)
    improvement: str


class MockInterviewDebrief(StrictModel):
    company: str
    title: str
    mode: Literal["technical", "behavioral", "mixed", "custom"] = "technical"
    average_score: float = Field(ge=1.0, le=5.0)
    readiness: Literal["ready", "needs targeted practice", "high risk", "not ready"]
    scores: list[MockInterviewScore] = Field(min_length=1)
    strengths_to_keep: list[str] = Field(default_factory=list)
    gaps_to_fix: list[str] = Field(default_factory=list)
    action_checklist: list[str] = Field(default_factory=list)
    final_note: str


class ArtifactFreshnessRecord(StrictModel):
    path: str
    revision: int = Field(ge=1)
    status: ArtifactFreshnessStatus
    updated_by: str
    note: str = ""


class SessionState(StrictModel):
    session_id: str
    job_id: str | None = None
    company: str | None = None
    title: str | None = None
    version: int = Field(default=1, ge=1)
    latest_verdict: Verdict | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_freshness: dict[str, ArtifactFreshnessRecord] = Field(default_factory=dict)
    available_actions: list[SessionIntent] = Field(default_factory=list)
    pending_evidence_gaps: list[str] = Field(default_factory=list)


class OrchestratorDecision(StrictModel):
    intent: SessionIntent
    required_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    runnable: bool
    next_step: str
    message: str


class OrchestratorExecutionResult(StrictModel):
    decision: OrchestratorDecision
    executed: bool
    written_artifacts: list[str] = Field(default_factory=list)
    message: str


class TrackerEvent(StrictModel):
    state: ApplicationStatus
    timestamp: str
    notes: str = ""


class ApplicationRecord(StrictModel):
    id: str
    job_id: str | None = None
    company: str
    title: str
    city: str = "unknown"
    url: str = ""
    source: str = ""
    current_state: ApplicationStatus
    verdict: Verdict | None = None
    fit_score: float | None = Field(default=None, ge=0.0, le=1.0)
    session_dir: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    next_deadline: str | None = None
    state_history: list[TrackerEvent] = Field(default_factory=list)


class TrackerReminder(StrictModel):
    kind: Literal["interview_due", "stuck", "stale_tracker", "missing_artifact"]
    message: str
    application_id: str | None = None
    due_at: str | None = None


class TrackerStats(StrictModel):
    total: int = 0
    by_state: dict[str, int] = Field(default_factory=dict)
    offer_count: int = 0
    response_rate: str = "0/0"
    interview_rate: str = "0/0"
    offer_rate: str = "0/0"


class ApplicationTracker(StrictModel):
    version: int = 1
    revision: int = Field(default=0, ge=0)
    user_id: str = "default-user"
    created: str
    updated: str
    applications: list[ApplicationRecord] = Field(default_factory=list)
    stats: TrackerStats = Field(default_factory=TrackerStats)
    reminders: list[TrackerReminder] = Field(default_factory=list)
    tracker_revision: int | None = Field(default=None, ge=0)
    application_id: str | None = None
    snapshot_of: str | None = None
