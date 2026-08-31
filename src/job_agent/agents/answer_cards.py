from __future__ import annotations

import re

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
    InterviewPrep,
    TargetedResume,
)


SKILL_ID = "answer-cards"
_PROMPT = get_prompt("semantic.answer-cards")
PROMPT_VERSION = _PROMPT.identity
_CJK_TEXT = re.compile(r"[\u3400-\u9fff]")


class AnswerCardsAgent:
    def __init__(
        self,
        *,
        registry: SkillLookup,
        harness: LLMHarness,
        policy: NodePolicy | None = None,
    ) -> None:
        self.registry = registry
        self.harness = harness
        self.policy = policy or NodePolicy(allow_rule_fallback=False)

    def run(
        self,
        interview_prep: InterviewPrep,
        *,
        evidence: list[EvidenceItem],
        targeted_resume: TargetedResume,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[AnswerCardDeck]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="answer_cards",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(
            guard_feedback: GuardFeedback | None,
        ) -> AgentResult[AnswerCardDeck]:
            task_context = {
                "interview_prep": interview_prep.model_dump(mode="json"),
                "verified_evidence": [
                    item.model_dump(mode="json") for item in evidence
                ],
                "targeted_resume": targeted_resume.model_dump(mode="json"),
                "instruction": _PROMPT.task_instruction,
            }
            if forbidden_output_strings:
                task_context["forbidden_output_strings"] = list(
                    forbidden_output_strings
                )
            if guard_feedback is not None:
                task_context["guard_feedback"] = guard_feedback
            return self.harness.invoke_structured(
                skill=skill,
                task_context=task_context,
                untrusted_inputs={},
                output_schema=AnswerCardDeck,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        return invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda result: self._guard(
                interview_prep,
                evidence,
                result,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )

    @staticmethod
    def _guard(
        interview_prep: InterviewPrep,
        evidence: list[EvidenceItem],
        result: AgentResult[AnswerCardDeck],
        *,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        deck = result.value
        if deck.company != interview_prep.company:
            raise AgentGuardError(
                "company_mismatch", result.trace, node_id="answer_cards"
            )
        if deck.title != interview_prep.title:
            raise AgentGuardError(
                "title_mismatch", result.trace, node_id="answer_cards"
            )
        questions = {
            item.requirement_id: item for item in interview_prep.questions
        }
        evidence_map = {item.requirement_id: item for item in evidence}
        card_ids = [item.requirement_id for item in deck.cards]
        if len(card_ids) != len(set(card_ids)) or set(card_ids) != set(questions):
            raise AgentGuardError(
                "requirement_coverage_mismatch",
                result.trace,
                node_id="answer_cards",
            )
        for card in deck.cards:
            question = questions[card.requirement_id]
            source = evidence_map.get(card.requirement_id)
            if (
                source is None
                or card.question != question.question
                or card.evidence_level != question.evidence_level
                or card.supporting_evidence != source.proof
                or card.boundary != source.risk
            ):
                raise AgentGuardError(
                    "answer_card_grounding_mismatch",
                    result.trace,
                    node_id="answer_cards",
                )
            if (
                not _CJK_TEXT.search(card.short_answer)
                or not card.practice_prompts
                or any(
                    not _CJK_TEXT.search(prompt)
                    for prompt in card.practice_prompts
                )
            ):
                raise AgentGuardError(
                    "answer_card_language_mismatch",
                    result.trace,
                    node_id="answer_cards",
                )
        reject_forbidden_output(
            deck.model_dump_json(),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id="answer_cards",
        )
