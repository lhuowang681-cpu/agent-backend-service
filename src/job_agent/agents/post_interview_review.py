from __future__ import annotations

import json
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
    EvidenceItem,
    PostInterviewReviewResult,
    PostInterviewRoundInput,
    StructuredJD,
    TargetedResume,
)


SKILL_ID = "post-interview-review"
_PROMPT = get_prompt("semantic.post-interview-review")
PROMPT_VERSION = _PROMPT.identity
_PASS_PREDICTION = re.compile(
    r"(?:通过率|录用概率|offer\s*概率|hire\s*probability)\s*[:：]?\s*\d",
    re.IGNORECASE,
)


class PostInterviewReviewAgent:
    """对已保存的真实面试记录做一次 evidence-aware 结构化诊断。"""

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
        structured_jd: StructuredJD,
        *,
        evidence: list[EvidenceItem],
        targeted_resume: TargetedResume,
        interview_round: PostInterviewRoundInput,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[PostInterviewReviewResult]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="post_interview_review",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )
        question_ids = [item.question_id for item in interview_round.questions]

        def invoke(
            guard_feedback: GuardFeedback | None,
        ) -> AgentResult[PostInterviewReviewResult]:
            task_context = {
                "job": structured_jd.model_dump(mode="json", exclude={"raw_jd"}),
                "verified_evidence": [
                    item.model_dump(mode="json") for item in evidence
                ],
                "targeted_resume": targeted_resume.model_dump(mode="json"),
                "round_contract": {
                    "round_id": interview_round.round_id,
                    "stage": interview_round.stage,
                    "question_ids": question_ids,
                },
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
                    "real_interview_record": json.dumps(
                        interview_round.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                },
                output_schema=PostInterviewReviewResult,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        return invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda result: self._guard(
                structured_jd,
                evidence,
                interview_round,
                result,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )

    @staticmethod
    def _guard(
        structured_jd: StructuredJD,
        evidence: list[EvidenceItem],
        interview_round: PostInterviewRoundInput,
        result: AgentResult[PostInterviewReviewResult],
        *,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        value = result.value
        node_id = "post_interview_review"
        if value.company != structured_jd.company:
            raise AgentGuardError(
                "company_mismatch", result.trace, node_id=node_id
            )
        if value.title != structured_jd.title:
            raise AgentGuardError("title_mismatch", result.trace, node_id=node_id)
        if value.round_id != interview_round.round_id:
            raise AgentGuardError(
                "round_id_mismatch", result.trace, node_id=node_id
            )

        question_map = {
            item.question_id: item for item in interview_round.questions
        }
        diagnosis_ids = [
            item.question_id for item in value.question_diagnoses
        ]
        if len(diagnosis_ids) != len(set(diagnosis_ids)) or set(
            diagnosis_ids
        ) != set(question_map):
            raise AgentGuardError(
                "question_coverage_mismatch", result.trace, node_id=node_id
            )

        requirement_ids = {
            item.id for item in structured_jd.must_have
        }
        evidence_by_id = {item.evidence_id: item for item in evidence}
        evidence_ids = set(evidence_by_id)
        all_gaps = list(value.priority_gaps)
        for diagnosis in value.question_diagnoses:
            source = question_map[diagnosis.question_id]
            if (
                not source.answer.strip()
                and not source.interviewer_feedback.strip()
                and diagnosis.status != "insufficient_information"
            ):
                raise AgentGuardError(
                    "missing_answer_not_abstained",
                    result.trace,
                    node_id=node_id,
                )
            if not set(diagnosis.evidence_ids).issubset(evidence_ids):
                raise AgentGuardError(
                    "unknown_evidence_id", result.trace, node_id=node_id
                )
            if diagnosis.improved_answer.strip() and any(
                evidence_by_id[evidence_id].level.value in {"C0", "None"}
                for evidence_id in diagnosis.evidence_ids
            ):
                raise AgentGuardError(
                    "weak_evidence_used_as_answer",
                    result.trace,
                    node_id=node_id,
                )
            all_gaps.extend(diagnosis.gaps)

        for gap in all_gaps:
            if not set(gap.basis_question_ids).issubset(question_map):
                raise AgentGuardError(
                    "unknown_basis_question_id",
                    result.trace,
                    node_id=node_id,
                )
            if not set(gap.requirement_ids).issubset(requirement_ids):
                raise AgentGuardError(
                    "unknown_requirement_id", result.trace, node_id=node_id
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
