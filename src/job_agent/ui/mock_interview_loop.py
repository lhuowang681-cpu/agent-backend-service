from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import ValidationError

from job_agent.agent_runtime.checkpoint import JsonCheckpointStore
from job_agent.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text_batch,
)
from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentInputRequest,
    AgentModelContext,
    AgentModelResponse,
    AgentRunStatus,
    UserInputRequest,
    UserInputResponse,
)
from job_agent.agent_runtime.live_agent_model import build_live_agent_model
from job_agent.domain_agents.interview_coach import (
    INTERVIEW_COACH_TOOLS,
    InterviewCoachAgent,
    InterviewCoachGoal,
    build_interview_coach_registry,
)
from job_agent.nodes.mock_debrief import audit_mock_interview
from job_agent.nodes.mock_interview_evaluation import render_rule_evidence_audit
from job_agent.schemas import (
    AnswerCardDeck,
    MockInterviewAnswer,
    MockInterviewAnswerSet,
    MockInterviewPlan,
    MockInterviewQuestion,
)
from job_agent.session_orchestrator import _parse_answer_cards, _parse_mock_plan


@dataclass(frozen=True)
class MockInterviewState:
    current_question: MockInterviewQuestion | None
    question_index: int
    history: list[dict]            # [{"question_id","prompt","answer","score","improvement"}]
    finished: bool
    final_result: dict
    pending_request_id: str | None
    total_questions: int = 0
    adaptive: bool = False


class MockInterviewRuntimeError(RuntimeError):
    def __init__(self, error_code: str):
        self.error_code = error_code
        super().__init__(f"mock interview runtime failed: {error_code}")


def _load_plan_and_cards(session_dir: Path) -> tuple[MockInterviewPlan, AnswerCardDeck]:
    """从 session 的 09/08 markdown 还原 MockInterviewPlan + AnswerCardDeck 对象。

    复用 session_orchestrator 的私有 parser（与 resume_loop._write_regenerated_resume 同模式）。
    """
    plan = _parse_mock_plan((session_dir / "09_mock_interview_plan.md").read_text(encoding="utf-8"))
    answer_cards = _parse_answer_cards((session_dir / "08_answer_cards.md").read_text(encoding="utf-8"))
    return plan, answer_cards


class OfflineInterviewModel:
    """offline/mock 的确定性 model（非 ScriptedAgentModel 盲回放）。

    hybrid 决策：
    - 空 context（单元测试 / 首次 decide）：走内部状态机
      load→ask all→finish
    - 非空 context（loop resume / 已有 steps）：从 context.steps + context.user_inputs
      推导当前阶段。resume 时 model 是新实例，内部 _phase/_q_index 不可用。
    """

    def __init__(self, plan: MockInterviewPlan) -> None:
        self.plan = plan
        self._phase = "load"
        self._q_index = 0

    def decide(self, context: AgentModelContext) -> AgentModelResponse:
        # resume 或 loop 二次 decide 时 context 有 steps/user_inputs → 从 context 推导
        if context.steps or context.user_inputs:
            return self._decide_from_context(context)
        return self._decide_stateful(context)

    def _decide_stateful(self, context: AgentModelContext) -> AgentModelResponse:
        """空 context 路径：内部状态机推进（单元测试 / 首次 decide）。"""
        question_limit = self._question_limit(context)
        if self._phase == "load":
            self._phase = "ask"
            return AgentModelResponse(
                decision=AgentAction(
                    action_id="load-plan",
                    tool_name="interview.load_plan",
                    tool_arguments={"include_answer_cards": True},
                    expected_observation="plan loaded",
                    progress_claim="Load approved interview plan.",
                ),
                provider="offline", model="offline-interview",
            )
        if self._phase == "ask":
            question = self.plan.questions[self._q_index]
            self._q_index += 1
            if self._q_index >= question_limit:
                self._phase = "finish"
            return AgentModelResponse(
                decision=AgentInputRequest(
                    request=UserInputRequest(
                        request_id=f"answer-{question.question_id}",
                        prompt=question.prompt,
                        response_schema={},
                        context_summary=question.focus,
                    ),
                    progress_claim=f"Ask {question.question_id}.",
                ),
                provider="offline", model="offline-interview",
            )
        return self._build_finish_response(
            [question.question_id for question in self.plan.questions[:question_limit]]
        )

    def _decide_from_context(self, context: AgentModelContext) -> AgentModelResponse:
        """非空 context 路径（resume / loop 二次 decide）：从 steps + user_inputs 推导。

        推导规则：按计划顺序收集全部 user_input，中途不评分；全部答案
        收齐后只提交 transcript 完成事实，语义评价由独立 Evaluator 执行。
        """
        question_limit = self._question_limit(context)
        answer_records = {}
        for record in context.user_inputs:
            value = record.response.value
            if not isinstance(value, dict):
                continue
            qid = value.get("question_id")
            if isinstance(qid, str) and qid:
                answer_records[qid] = record

        questions = self.plan.questions[:question_limit]
        for question in questions:
            if question.question_id not in answer_records:
                return AgentModelResponse(
                    decision=AgentInputRequest(
                        request=UserInputRequest(
                            request_id=f"answer-{question.question_id}",
                            prompt=question.prompt,
                            response_schema={},
                            context_summary=question.focus,
                        ),
                        progress_claim=f"Ask {question.question_id}.",
                    ),
                    provider="offline", model="offline-interview",
                )

        return self._build_finish_response(
            [question.question_id for question in questions]
        )

    def _question_limit(self, context: AgentModelContext) -> int:
        raw_limit = context.goal.get("max_questions", len(self.plan.questions))
        try:
            requested = int(raw_limit)
        except (TypeError, ValueError):
            requested = len(self.plan.questions)
        return min(max(requested, 1), len(self.plan.questions))

    def _build_finish_response(
        self,
        question_ids: list[str],
    ) -> AgentModelResponse:
        result = {
            "company": self.plan.company,
            "title": self.plan.title,
            "completed_question_ids": question_ids,
            "transcript_complete": True,
            "final_note": (
                "离线模拟面试已完成答题记录；尚未执行 AI 语义评价。"
            ),
        }
        return AgentModelResponse(
            decision=AgentFinish(
                result=result,
                completion_evidence=[
                    f"question:{question_id}" for question_id in question_ids
                ],
                confidence=0.8,
            ),
            provider="offline", model="offline-interview",
        )


def _build_offline_model(plan: MockInterviewPlan) -> OfflineInterviewModel:
    return OfflineInterviewModel(plan)


def _build_live_model(
    *,
    runtime,
    plan: MockInterviewPlan,
    answer_cards: AnswerCardDeck,
    session_id: str,
    run_id: str,
):
    """live：构造 InterviewCoachAgent 的 ToolUseDecisionModel（provider 出题，带 retry）。

    复用 build_interview_coach_registry 提供只读计划工具，
    再由 build_live_agent_model 包装成 tool_use_v2 decision model（submit_interview_result）。
    """
    registry = build_interview_coach_registry(plan, answer_cards)
    return build_live_agent_model(
        provider=runtime.provider,
        registry=registry,
        allowed_tools=INTERVIEW_COACH_TOOLS,
        agent_id="interview-coach",
        session_id=session_id,
        run_id=run_id,
        allow_user_input=True,
    )


def _build_model(
    *,
    mode: str,
    runtime,
    plan: MockInterviewPlan,
    answer_cards: AnswerCardDeck,
    session_id: str,
):
    """Choose the controller that matches the requested interview semantics.

    Offline mode replays the approved preparation plan deterministically.
    Live mode uses the Domain Agent loop so each answer becomes an observation
    for choosing the next follow-up. Both modes delay grading until all answers
    have been collected.
    """
    if mode == "agent_api_live":
        if runtime is None or getattr(runtime, "provider", None) is None:
            raise MockInterviewRuntimeError("live_runtime_unavailable")
        return _build_live_model(
            runtime=runtime,
            plan=plan,
            answer_cards=answer_cards,
            session_id=session_id,
            run_id="interview-run-1",
        )
    return _build_offline_model(plan)


def _question_from_request(
    request: UserInputRequest,
    plan: MockInterviewPlan,
    *,
    question_index: int,
) -> MockInterviewQuestion:
    """Turn a live input request into the exact question shown and graded."""
    raw_id = request.request_id.removeprefix("answer-").strip()
    question_id = raw_id or f"adaptive-{question_index + 1}"
    base_question = next(
        (
            question
            for question in plan.questions
            if question_id == question.question_id
            or question_id.startswith(f"{question.question_id}-")
        ),
        plan.questions[min(question_index, len(plan.questions) - 1)],
    )
    context = request.context_summary.strip()
    return base_question.model_copy(
        update={
            "question_id": question_id,
            "prompt": request.prompt,
            "focus": context or base_question.focus,
        }
    )


def _effective_plan_from_history(
    plan: MockInterviewPlan,
    history: Sequence[dict],
) -> MockInterviewPlan:
    questions: list[MockInterviewQuestion] = []
    for index, item in enumerate(history):
        base_question = next(
            (
                question
                for question in plan.questions
                if question.requirement_id == item.get("requirement_id")
            ),
            plan.questions[min(index, len(plan.questions) - 1)],
        )
        questions.append(
            base_question.model_copy(
                update={
                    "question_id": item["question_id"],
                    "requirement_id": item.get(
                        "requirement_id",
                        base_question.requirement_id,
                    ),
                    "prompt": item["prompt"],
                    "focus": item.get("focus") or base_question.focus,
                }
            )
        )
    return plan.model_copy(update={"questions": questions or plan.questions})


def _write_run_result(
    session_dir: Path,
    plan: MockInterviewPlan,
    state: MockInterviewState,
) -> Path:
    """先持久化 transcript 和规则审计；AI 评价使用独立原子 artifact。"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    run_path = session_dir / f"10_mock_interview_run_{timestamp}.json"
    suffix = 2
    while run_path.exists():
        run_path = session_dir / (
            f"10_mock_interview_run_{timestamp}_{suffix}.json"
        )
        suffix += 1
    effective_plan = _effective_plan_from_history(plan, state.history)
    answer_set = MockInterviewAnswerSet(
        company=plan.company,
        title=plan.title,
        mode=plan.mode,
        answers=[
            MockInterviewAnswer(
                question_id=h["question_id"],
                answer=h["answer"],
                elapsed_seconds=h.get("elapsed_seconds", 0),
            )
            for h in state.history
        ],
    )
    audits = audit_mock_interview(effective_plan, answer_set)
    payload = {
        "schema_version": 2,
        "result": state.final_result,
        "history": state.history,
        "rule_evidence_audit": [
            item.model_dump(mode="json") for item in audits
        ],
        "evaluation_status": "pending_ai" if state.adaptive else "offline_no_ai",
    }
    atomic_write_json(run_path, payload)
    atomic_write_text_batch(
        {
            session_dir / "15_mock_interview_debrief.md": (
                render_rule_evidence_audit(audits)
                + (
                    "\n## AI 综合评价\n\n"
                    "离线模式未调用模型，因此本轮尚无 AI 语义评价。\n"
                    if not state.adaptive
                    else "\n## AI 综合评价\n\n正在生成独立的整轮 AI 评价。\n"
                )
            )
        }
    )
    return run_path


def start_mock_interview(
    session_dir: Path,
    *,
    max_questions: int = 3,
    mode: str = "offline_rule",
    runtime=None,
) -> MockInterviewState:
    """Start one interview and pause at the first input request.

    Live mode lets the InterviewCoach Agent choose one question at a time.
    Offline mode follows the approved plan. Scoring is delayed in both modes.
    """
    plan, answer_cards = _load_plan_and_cards(session_dir)
    from job_agent.ui.post_interview_review_loop import (
        NEXT_INTERVIEW_FOCUS_FILE,
        load_next_interview_focus,
    )

    review_focus = load_next_interview_focus(session_dir)
    if review_focus and mode != "agent_api_live":
        terms = {
            term.casefold()
            for topic in review_focus
            for term in re.split(r"[\s，,。；;：:/]+", topic)
            if len(term.strip()) >= 2
        }

        def focus_score(question: MockInterviewQuestion) -> int:
            haystack = f"{question.prompt} {question.focus}".casefold()
            return sum(term in haystack for term in terms)

        plan = plan.model_copy(
            update={
                "questions": sorted(
                    plan.questions,
                    key=focus_score,
                    reverse=True,
                )
            }
        )
    session_id = f"workbench-interview:{session_dir.name}"
    adaptive = mode == "agent_api_live"
    effective_max_questions = (
        max_questions
        if adaptive
        else min(max_questions, len(plan.questions))
    )
    model = _build_model(
        mode=mode, runtime=runtime, plan=plan,
        answer_cards=answer_cards, session_id=session_id,
    )
    checkpoint_store = JsonCheckpointStore(session_dir / "interview-checkpoint.json")
    agent = InterviewCoachAgent(
        model=model,
        plan=plan,
        answer_cards=answer_cards,
        checkpoint_store=checkpoint_store,
    )
    goal = InterviewCoachGoal(
        company=plan.company,
        title=plan.title,
        max_questions=effective_max_questions,
        target_focus=(
            ["adaptive_live", *review_focus]
            if adaptive
            else review_focus
        ),
    )
    run_result = agent.run(
        goal=goal,
        session_id=session_id,
        run_id="interview-run-1",
        workspace_root=session_dir,
    )
    # agent 走到 input_request(q1) → state WAITING_FOR_USER
    if run_result.state.status != AgentRunStatus.WAITING_FOR_USER:
        raise MockInterviewRuntimeError(
            run_result.error_code
            or f"unexpected_status:{run_result.state.status.value}"
        )
    if run_result.pending_user_input is None:
        raise MockInterviewRuntimeError("missing_pending_user_input")
    focus_path = session_dir / NEXT_INTERVIEW_FOCUS_FILE
    if review_focus and focus_path.exists():
        focus_path.unlink()
    request = run_result.pending_user_input
    question = _question_from_request(
        request,
        plan,
        question_index=0,
    )
    return MockInterviewState(
        current_question=question,
        question_index=0,
        history=[],
        finished=False,
        final_result={},
        pending_request_id=request.request_id,
        total_questions=effective_max_questions,
        adaptive=adaptive,
    )


def restore_mock_interview_state(session_dir: Path) -> MockInterviewState | None:
    """从持久化 checkpoint 恢复 Streamlit 展示状态。

    Agent checkpoint 是运行事实源；这个函数只重建 UI 所需的当前题目和已答摘要，
    不修改 checkpoint，也不推进 agent。
    """
    try:
        plan, _ = _load_plan_and_cards(session_dir)
        checkpoint = JsonCheckpointStore(session_dir / "interview-checkpoint.json").load()
    except Exception:
        return None
    pending = checkpoint.pending_user_input
    if pending is None:
        return None

    question_index = len(checkpoint.user_inputs)
    current_question = _question_from_request(
        pending,
        plan,
        question_index=question_index,
    )

    history: list[dict] = []
    for index, record in enumerate(checkpoint.user_inputs):
        value = record.response.value
        if not isinstance(value, dict):
            continue
        qid = value.get("question_id")
        if not isinstance(qid, str):
            continue
        history.append(
            {
                "question_id": qid,
                "requirement_id": value.get("requirement_id")
                or _question_from_request(
                    record.request,
                    plan,
                    question_index=index,
                ).requirement_id,
                "prompt": value.get("question_prompt")
                or record.request.prompt,
                "focus": value.get("focus")
                or record.request.context_summary,
                "answer": value.get("answer", ""),
                "score": None,
                "improvement": "",
                "elapsed_seconds": value.get("elapsed_seconds", 0),
            }
        )

    adaptive = "adaptive_live" in checkpoint.state.goal.get(
        "target_focus",
        [],
    )
    requested_questions = int(
        checkpoint.state.goal.get("max_questions", len(plan.questions))
    )
    return MockInterviewState(
        current_question=current_question,
        question_index=question_index,
        history=history,
        finished=False,
        final_result={},
        pending_request_id=pending.request_id,
        total_questions=(
            requested_questions
            if adaptive
            else min(requested_questions, len(plan.questions))
        ),
        adaptive=adaptive,
    )


def _build_answer_response(
    state: MockInterviewState,
    answer: str,
    elapsed_seconds: int,
) -> UserInputResponse:
    """把用户的答案文本 + 当前 pending request 包装成 UserInputResponse。"""
    return UserInputResponse(
        request_id=state.pending_request_id or "",
        value={
            "question_id": state.current_question.question_id if state.current_question else "",
            "requirement_id": (
                state.current_question.requirement_id
                if state.current_question
                else ""
            ),
            "question_prompt": (
                state.current_question.prompt if state.current_question else ""
            ),
            "focus": state.current_question.focus if state.current_question else "",
            "answer": answer,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def answer_question(
    session_dir: Path,
    state: MockInterviewState,
    answer: str,
    *,
    elapsed_seconds: int = 0,
    max_questions: int = 3,     # 仅 checkpoint 恢复失败时作 fallback（正常路径 goal 从 checkpoint 读）
    mode: str = "offline_rule",
    runtime=None,
) -> MockInterviewState:
    """Resume with one answer, then adaptively ask or finish.

    Live mode gives the exact answer to the InterviewCoach Agent as an
    observation so it can generate one follow-up or switch topic. Offline mode
    advances through the approved plan. Both modes collect every answer before
    the deterministic scoring tools run.

    注意：goal（含 max_questions）从 checkpoint 恢复，保证与 start_mock_interview
    一致；参数 max_questions 仅在 checkpoint 缺失时作 fallback（不应发生）。
    """
    plan, answer_cards = _load_plan_and_cards(session_dir)
    session_id = f"workbench-interview:{session_dir.name}"
    model = _build_model(
        mode=mode, runtime=runtime, plan=plan,
        answer_cards=answer_cards, session_id=session_id,
    )
    checkpoint_store = JsonCheckpointStore(session_dir / "interview-checkpoint.json")
    checkpoint_snapshot = (
        checkpoint_store.path.read_bytes()
        if checkpoint_store.path.is_file()
        else None
    )
    # 从 checkpoint 恢复原始 goal，避免 max_questions 等 field 与 start 不一致 → goal_mismatch
    try:
        checkpoint = checkpoint_store.load()
        goal = InterviewCoachGoal.model_validate(checkpoint.state.goal)
    except (OSError, json.JSONDecodeError, ValidationError):
        effective_max_questions = (
            max_questions
            if state.adaptive
            else min(max_questions, len(plan.questions))
        )
        goal = InterviewCoachGoal(
            company=plan.company,
            title=plan.title,
            max_questions=effective_max_questions,
            target_focus=["adaptive_live"] if state.adaptive else [],
        )
    agent = InterviewCoachAgent(
        model=model,
        plan=plan,
        answer_cards=answer_cards,
        checkpoint_store=checkpoint_store,
    )
    response = _build_answer_response(state, answer, elapsed_seconds)
    try:
        run_result = agent.run(
            goal=goal,
            session_id=session_id,
            run_id="interview-run-1",
            workspace_root=session_dir,
            user_inputs={response.request_id: response},
            resume=True,
        )
    except Exception as exc:
        if checkpoint_snapshot is not None:
            atomic_write_bytes(checkpoint_store.path, checkpoint_snapshot)
        raise MockInterviewRuntimeError("interview_turn_failed") from exc
    # 答题阶段只累积原始答案；全部收齐后才统一回填评分。
    new_history = list(state.history)
    if state.current_question is not None:
        new_history.append({
            "question_id": state.current_question.question_id,
            "requirement_id": state.current_question.requirement_id,
            "prompt": state.current_question.prompt,
            "focus": state.current_question.focus,
            "answer": answer,
            "score": None,
            "improvement": "",
            "elapsed_seconds": elapsed_seconds,
        })
    if run_result.state.status == AgentRunStatus.COMPLETED:
        finished_state = MockInterviewState(
            current_question=None,
            question_index=state.question_index + 1,
            history=new_history,
            finished=True,
            final_result=run_result.result,
            pending_request_id=None,
            total_questions=(
                getattr(state, "total_questions", 0)
                or (
                    goal.max_questions
                    if state.adaptive
                    else min(goal.max_questions, len(plan.questions))
                )
            ),
            adaptive=state.adaptive,
        )
        run_path = _write_run_result(session_dir, plan, finished_state)
        if mode == "agent_api_live" and runtime is not None:
            try:
                from job_agent.ui.mock_interview_evaluation_loop import (
                    generate_mock_interview_evaluation_live,
                )

                evaluation = generate_mock_interview_evaluation_live(
                    session_dir,
                    run_path,
                    plan=_effective_plan_from_history(plan, new_history),
                    history=new_history,
                    answer_cards=answer_cards,
                    runtime=runtime,
                )
                finished_state = replace(
                    finished_state,
                    final_result={
                        **finished_state.final_result,
                        "ai_evaluation": evaluation.model_dump(mode="json"),
                    },
                )
            except Exception:
                try:
                    from job_agent.ui.mock_interview_evaluation_loop import (
                        update_run_evaluation_status,
                    )

                    update_run_evaluation_status(
                        run_path,
                        status="ai_failed",
                        error_code="ai_evaluation_failed",
                    )
                except Exception:
                    pass
                finished_state = replace(
                    finished_state,
                    final_result={
                        **finished_state.final_result,
                        "evaluation_error_code": "ai_evaluation_failed",
                    },
                )
        return finished_state
    # 否则推进到下一题
    if run_result.state.status != AgentRunStatus.WAITING_FOR_USER:
        if checkpoint_snapshot is not None:
            atomic_write_bytes(checkpoint_store.path, checkpoint_snapshot)
        raise MockInterviewRuntimeError(
            run_result.error_code
            or f"unexpected_status:{run_result.state.status.value}"
        )
    if run_result.pending_user_input is None:
        if checkpoint_snapshot is not None:
            atomic_write_bytes(checkpoint_store.path, checkpoint_snapshot)
        raise MockInterviewRuntimeError("missing_pending_user_input")
    next_index = state.question_index + 1
    next_question = _question_from_request(
        run_result.pending_user_input,
        plan,
        question_index=next_index,
    )
    return MockInterviewState(
        current_question=next_question,
        question_index=next_index,
        history=new_history,
        finished=False,
        final_result={},
        pending_request_id=run_result.pending_user_input.request_id,
        total_questions=(
            getattr(state, "total_questions", 0)
            or (
                goal.max_questions
                if state.adaptive
                else min(goal.max_questions, len(plan.questions))
            )
        ),
        adaptive=state.adaptive,
    )
