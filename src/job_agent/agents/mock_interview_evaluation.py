from __future__ import annotations

import json
import re
from statistics import mean

from job_agent.agents.base import (
    AgentGuardError,
    GuardFeedback,
    SkillLookup,
    invoke_with_guard_repair,
    reject_forbidden_output,
)
from job_agent.llm.harness import LLMHarness, NodePolicy
from job_agent.llm.provider import AgentResult, TraceContext
from job_agent.prompts import get_prompt
from job_agent.schemas import (
    AnswerCardDeck,
    EvidenceItem,
    MockInterviewEvaluationResult,
    MockInterviewPlan,
    MockInterviewTranscriptItem,
)


SKILL_ID = "mock-interview-evaluation"
_PROMPT = get_prompt("semantic.mock-interview-evaluation")
PROMPT_VERSION = _PROMPT.identity
_PASS_PREDICTION = re.compile(
    r"(?:通过率|录用概率|offer\s*概率|hire\s*probability)\s*[:：]?\s*\d",
    re.IGNORECASE,
)


def _overall_score(value: MockInterviewEvaluationResult) -> float:
    dimensions: list[int] = []
    for item in value.question_evaluations:
        dimensions.extend(
            [
                item.relevance,
                item.technical_depth,
                item.reasoning_clarity,
                item.evidence_grounding,
                item.truth_boundary,
            ]
        )
    return round(mean(dimensions), 2)


class MockInterviewEvaluationAgent:
    """整轮结束后调用一次 LLM，并用确定性 Guard 约束评价合同。"""

    def __init__(
        self,
        *,
        registry: SkillLookup,
        harness: LLMHarness,
        policy: NodePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.harness = harness
        self.policy = policy or NodePolicy(
            allow_rule_fallback=False,
            max_output_tokens=8192,
        )

    def run(
        self,
        plan: MockInterviewPlan,
        *,
        transcript: list[MockInterviewTranscriptItem],
        answer_cards: AnswerCardDeck,
        evidence: list[EvidenceItem],
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[MockInterviewEvaluationResult]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="mock_interview_evaluation",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(
            guard_feedback: GuardFeedback | None,
        ) -> AgentResult[MockInterviewEvaluationResult]:
            task_context = {
                "interview_contract": {
                    "company": plan.company,
                    "title": plan.title,
                    "run_id": run_id,
                    "question_ids": [item.question_id for item in transcript],
                    "question_requirements": {
                        item.question_id: item.requirement_id
                        for item in transcript
                    },
                },
                "verified_evidence": [
                    item.model_dump(mode="json") for item in evidence
                ],
                "answer_cards": answer_cards.model_dump(mode="json"),
                "instruction": _PROMPT.task_instruction,
            }
            if guard_feedback is not None:
                task_context["guard_feedback"] = guard_feedback
            if forbidden_output_strings:
                task_context["forbidden_output_strings"] = list(
                    forbidden_output_strings
                )
            return self.harness.invoke_structured(
                skill=skill,
                task_context=task_context,
                untrusted_inputs={
                    "mock_interview_transcript": json.dumps(
                        [item.model_dump(mode="json") for item in transcript],
                        ensure_ascii=False,
                    )
                },
                output_schema=MockInterviewEvaluationResult,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        result = invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda candidate: self._guard(
                plan,
                transcript,
                evidence,
                candidate,
                run_id=run_id,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )
        normalized = result.value.model_copy(
            update={"overall_score": _overall_score(result.value)}
        )
        return result.model_copy(update={"value": normalized})

    @staticmethod
    def _guard(
        plan: MockInterviewPlan,
        transcript: list[MockInterviewTranscriptItem],
        evidence: list[EvidenceItem],
        result: AgentResult[MockInterviewEvaluationResult],
        *,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        value = result.value
        node_id = "mock_interview_evaluation"
        if value.company != plan.company:
            raise AgentGuardError(
                "company_mismatch", result.trace, node_id=node_id
            )
        if value.title != plan.title:
            raise AgentGuardError("title_mismatch", result.trace, node_id=node_id)
        if value.run_id != run_id:
            raise AgentGuardError("run_id_mismatch", result.trace, node_id=node_id)

        transcript_by_id = {item.question_id: item for item in transcript}
        evaluation_ids = [
            item.question_id for item in value.question_evaluations
        ]
        if (
            len(transcript_by_id) != len(transcript)
            or len(evaluation_ids) != len(set(evaluation_ids))
            or set(evaluation_ids) != set(transcript_by_id)
        ):
            raise AgentGuardError(
                "question_coverage_mismatch", result.trace, node_id=node_id
            )

        evidence_by_id = {item.evidence_id: item for item in evidence}
        for evaluation in value.question_evaluations:
            source = transcript_by_id[evaluation.question_id]
            if evaluation.requirement_id != source.requirement_id:
                raise AgentGuardError(
                    "requirement_id_mismatch", result.trace, node_id=node_id
                )
            if not set(evaluation.evidence_ids).issubset(evidence_by_id):
                raise AgentGuardError(
                    "unknown_evidence_id", result.trace, node_id=node_id
                )
            if any(
                evidence_by_id[evidence_id].requirement_id
                != source.requirement_id
                for evidence_id in evaluation.evidence_ids
            ):
                raise AgentGuardError(
                    "cross_requirement_evidence",
                    result.trace,
                    node_id=node_id,
                )
            if evaluation.improved_answer.strip() and any(
                evidence_by_id[evidence_id].level.value in {"C0", "None"}
                for evidence_id in evaluation.evidence_ids
            ):
                raise AgentGuardError(
                    "weak_evidence_used_as_answer",
                    result.trace,
                    node_id=node_id,
                )
            if not source.answer.strip():
                dimensions = (
                    evaluation.relevance,
                    evaluation.technical_depth,
                    evaluation.reasoning_clarity,
                    evaluation.evidence_grounding,
                    evaluation.truth_boundary,
                )
                if (
                    evaluation.status != "insufficient_information"
                    or dimensions != (1, 1, 1, 1, 1)
                    or evaluation.improved_answer.strip()
                ):
                    raise AgentGuardError(
                        "missing_answer_not_abstained",
                        result.trace,
                        node_id=node_id,
                    )

        rendered = value.model_dump_json()
        if _PASS_PREDICTION.search(rendered):
            raise AgentGuardError(
                "pass_probability_prediction", result.trace, node_id=node_id
            )
        reject_forbidden_output(
            rendered,
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id=node_id,
        )
