from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

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
    UserInputResponse,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, AgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.schema_diagnostics import safe_validation_error_detail
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.nodes.mock_debrief import score_mock_interview
from job_agent.prompts import get_prompt
from job_agent.schemas import (
    AnswerCardDeck,
    MockInterviewAnswer,
    MockInterviewAnswerSet,
    MockInterviewPlan,
    MockInterviewScore,
    StrictModel,
)


LOAD_INTERVIEW_TOOL = "interview.load_plan"
GRADE_ANSWER_TOOL = "interview.grade_answer"
INTERVIEW_COACH_TOOLS = (LOAD_INTERVIEW_TOOL,)
_PROMPT = get_prompt("domain.interview-coach")


class InterviewCoachGoal(StrictModel):
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    max_questions: int = Field(default=3, ge=1, le=20)
    target_focus: list[str] = Field(default_factory=list)


class LoadInterviewInput(StrictModel):
    include_answer_cards: bool = True


class LoadInterviewOutput(StrictModel):
    plan: MockInterviewPlan
    answer_cards: AnswerCardDeck | None = None


class GradeAnswerInput(StrictModel):
    request_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)


class GradeAnswerOutput(StrictModel):
    score: MockInterviewScore
    next_focus: str
    follow_up_recommended: bool


class InterviewCoachResult(StrictModel):
    company: str
    title: str
    completed_question_ids: list[str] = Field(min_length=1)
    transcript_complete: bool = True
    final_note: str


def build_interview_coach_registry(
    plan: MockInterviewPlan,
    answer_cards: AnswerCardDeck,
) -> ToolRegistry:
    questions = {question.question_id: question for question in plan.questions}
    registry = ToolRegistry()

    def load_plan(value: LoadInterviewInput, context: ToolContext) -> LoadInterviewOutput:
        return LoadInterviewOutput(
            plan=plan,
            answer_cards=answer_cards if value.include_answer_cards else None,
        )

    def grade_answer(value: GradeAnswerInput, context: ToolContext) -> GradeAnswerOutput:
        response = context.user_inputs.get(value.request_id)
        if response is None:
            raise ValueError("approved user input not found")
        response_question_id = response.value.get("question_id")
        if response_question_id != value.question_id:
            raise ValueError("user input question mismatch")
        question = questions.get(value.question_id)
        if question is None:
            requirement_id = response.value.get("requirement_id")
            question_prompt = response.value.get("question_prompt")
            if (
                not isinstance(requirement_id, str)
                or not requirement_id
                or not isinstance(question_prompt, str)
                or not question_prompt
            ):
                raise ValueError("adaptive interview question metadata missing")
            base_question = next(
                (
                    item
                    for item in plan.questions
                    if item.requirement_id == requirement_id
                ),
                plan.questions[0],
            )
            question = base_question.model_copy(
                update={
                    "question_id": value.question_id,
                    "requirement_id": requirement_id,
                    "prompt": question_prompt,
                    "focus": response.value.get("focus") or base_question.focus,
                }
            )
        else:
            question_prompt = response.value.get("question_prompt")
            if isinstance(question_prompt, str) and question_prompt:
                question = question.model_copy(update={"prompt": question_prompt})
        answer = response.value.get("answer", "")
        if not isinstance(answer, str):
            raise ValueError("answer must be a string")
        elapsed_seconds = response.value.get("elapsed_seconds", 0)
        if not isinstance(elapsed_seconds, int) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be a non-negative integer")
        effective_plan = plan.model_copy(update={"questions": [question]})
        answer_set = MockInterviewAnswerSet(
            company=plan.company,
            title=plan.title,
            mode=plan.mode,
            answers=[
                MockInterviewAnswer(
                    question_id=value.question_id,
                    answer="" if response.cancelled else answer,
                    elapsed_seconds=elapsed_seconds,
                )
            ],
        )
        debrief = score_mock_interview(effective_plan, answer_set, answer_cards)
        score = debrief.scores[0]
        return GradeAnswerOutput(
            score=score,
            next_focus=score.improvement,
            follow_up_recommended=score.score < 4 or bool(score.risk_flags),
        )

    registry.register(
        name=LOAD_INTERVIEW_TOOL,
        description="Load the approved mock interview plan and evidence-bounded answer cards.",
        input_model=LoadInterviewInput,
        output_model=LoadInterviewOutput,
        handler=load_plan,
        effect=ToolEffect.READ_ONLY,
    )
    registry.register(
        name=GRADE_ANSWER_TOOL,
        description="Grade one exact user answer using the deterministic mock interview rubric.",
        input_model=GradeAnswerInput,
        output_model=GradeAnswerOutput,
        handler=grade_answer,
        effect=ToolEffect.READ_ONLY,
    )
    return registry


class InterviewCoachVerifier:
    def verify(
        self,
        finish: AgentFinish,
        *,
        goal: dict,
        steps: Sequence[AgentStepRecord],
    ) -> VerificationResult:
        try:
            typed_goal = InterviewCoachGoal.model_validate(goal)
            result = InterviewCoachResult.model_validate(finish.result)
        except ValidationError as exc:
            detail = safe_validation_error_detail(
                exc,
                allowed_fields={
                    "company",
                    "title",
                    "completed_question_ids",
                    "transcript_complete",
                    "final_note",
                },
            )
            return VerificationResult(
                passed=False,
                reason_code=f"interview_result_schema_error:{detail}",
                feedback=(
                    "Return a valid transcript-completion result. "
                    f"Shape diagnostic: {detail}"
                ),
            )
        except Exception:
            return VerificationResult(
                passed=False,
                reason_code="interview_result_schema_error",
                feedback="Return a valid transcript-completion result.",
            )
        if (result.company, result.title) != (typed_goal.company, typed_goal.title):
            return VerificationResult(
                passed=False,
                reason_code="interview_target_mismatch",
                feedback="Keep company and title identical to the approved goal.",
            )
        question_ids = result.completed_question_ids
        if (
            len(question_ids) != typed_goal.max_questions
            or len(question_ids) != len(set(question_ids))
        ):
            return VerificationResult(
                passed=False,
                reason_code="interview_question_count_mismatch",
                feedback=(
                    "Collect exactly max_questions unique answers before finishing."
                ),
            )
        if not result.transcript_complete:
            return VerificationResult(
                passed=False,
                reason_code="interview_transcript_incomplete",
                feedback="Set transcript_complete only after all answers are collected.",
            )
        return VerificationResult(
            passed=True,
            reason_code="interview_coach_complete",
            feedback="Interview transcript is complete; semantic evaluation is a separate step.",
            evidence_refs=[f"question:{question_id}" for question_id in question_ids],
        )


class InterviewCoachAgent:
    def __init__(
        self,
        *,
        model: AgentModel,
        plan: MockInterviewPlan,
        answer_cards: AnswerCardDeck,
        budget: AgentBudget | None = None,
        trajectory=None,
        checkpoint_store: CheckpointStore | None = None,
        max_same_verifier_reason: int | None = None,
    ) -> None:
        self.registry = build_interview_coach_registry(plan, answer_cards)
        policy_context = PolicyContext(
            skill_allowed_tools=INTERVIEW_COACH_TOOLS,
            agent_allowed_tools=INTERVIEW_COACH_TOOLS,
            runtime_allowed_tools=INTERVIEW_COACH_TOOLS,
        )
        self.loop = AgentLoop(
            agent_id="interview-coach",
            model=model,
            registry=self.registry,
            policy=PolicyEngine(),
            policy_context=policy_context,
            budget=BudgetManager(budget or AgentBudget.for_profile(BudgetProfile.STANDARD)),
            verifier=InterviewCoachVerifier(),
            trajectory=trajectory,
            checkpoint_store=checkpoint_store,
            skill_version="interview-coach-v2",
            prompt_version=_PROMPT.identity,
            max_same_verifier_reason=max_same_verifier_reason,
        )

    def run(
        self,
        goal: InterviewCoachGoal,
        *,
        session_id: str,
        run_id: str,
        workspace_root: Path | str,
        user_inputs: Mapping[str, UserInputResponse] | None = None,
        resume: bool = False,
    ) -> AgentRunResult:
        return self.loop.run(
            session_id=session_id,
            run_id=run_id,
            goal=goal.model_dump(mode="json"),
            tool_context=ToolContext(
                session_id=session_id,
                run_id=run_id,
                agent_id="interview-coach",
                workspace_root=str(Path(workspace_root).resolve()),
            ),
            user_inputs=user_inputs,
            resume=resume,
        )
