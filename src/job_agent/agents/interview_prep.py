from __future__ import annotations

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
from job_agent.schemas import EvidenceItem, FitVerdictResult, InterviewPrep, StructuredJD, TargetedResume


SKILL_ID = "interview-grilling"
_PROMPT = get_prompt("semantic.interview-prep")
PROMPT_VERSION = _PROMPT.identity


class InterviewPrepAgent:
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
        structured_jd: StructuredJD,
        *,
        evidence: list[EvidenceItem],
        fit_result: FitVerdictResult,
        targeted_resume: TargetedResume,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[InterviewPrep]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="interview_prep",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(guard_feedback: GuardFeedback | None) -> AgentResult[InterviewPrep]:
            task_context = {
                "structured_jd": structured_jd.model_dump(mode="json", exclude={"raw_jd"}),
                "verified_evidence": [item.model_dump(mode="json") for item in evidence],
                "fit_verdict": fit_result.model_dump(mode="json"),
                "targeted_resume": targeted_resume.model_dump(mode="json"),
                "instruction": _PROMPT.task_instruction,
            }
            if forbidden_output_strings:
                task_context["forbidden_output_strings"] = list(forbidden_output_strings)
            if guard_feedback is not None:
                task_context["guard_feedback"] = guard_feedback
            return self.harness.invoke_structured(
                skill=skill,
                task_context=task_context,
                untrusted_inputs={},
                output_schema=InterviewPrep,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        return invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda result: self._guard(
                structured_jd,
                evidence,
                result,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )

    @staticmethod
    def _guard(
        structured_jd: StructuredJD,
        evidence: list[EvidenceItem],
        result: AgentResult[InterviewPrep],
        *,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        if result.value.company != structured_jd.company:
            raise AgentGuardError("company_mismatch", result.trace, node_id="interview_prep")
        if result.value.title != structured_jd.title:
            raise AgentGuardError("title_mismatch", result.trace, node_id="interview_prep")
        required_ids = {requirement.id for requirement in structured_jd.must_have}
        question_ids = [question.requirement_id for question in result.value.questions]
        if set(question_ids) != required_ids or len(question_ids) != len(required_ids):
            raise AgentGuardError("requirement_coverage_mismatch", result.trace, node_id="interview_prep")
        evidence_by_requirement = {item.requirement_id: item for item in evidence}
        for question in result.value.questions:
            source = evidence_by_requirement.get(question.requirement_id)
            if source is None or question.evidence_level != source.level:
                raise AgentGuardError("evidence_level_mismatch", result.trace, node_id="interview_prep")
        reject_forbidden_output(
            result.value.model_dump_json(),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id="interview_prep",
        )
