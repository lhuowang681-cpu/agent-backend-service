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
from job_agent.schemas import EvidenceMappingResult, StructuredJD


SKILL_ID = "evidence-contract"
_PROMPT = get_prompt("semantic.evidence-mapping")
PROMPT_VERSION = _PROMPT.identity


class EvidenceMappingAgent:
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
        resume_text: str,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[EvidenceMappingResult]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="evidence_mapping",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(guard_feedback: GuardFeedback | None) -> AgentResult[EvidenceMappingResult]:
            task_context = {
                "structured_jd": structured_jd.model_dump(mode="json", exclude={"raw_jd"}),
                "instruction": _PROMPT.task_instruction,
            }
            if forbidden_output_strings:
                task_context["forbidden_output_strings"] = list(forbidden_output_strings)
            if guard_feedback is not None:
                task_context["guard_feedback"] = guard_feedback
            return self.harness.invoke_structured(
                skill=skill,
                task_context=task_context,
                untrusted_inputs={"resume": resume_text},
                output_schema=EvidenceMappingResult,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        return invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda result: self._guard(
                structured_jd,
                result,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )

    @staticmethod
    def _guard(
        structured_jd: StructuredJD,
        result: AgentResult[EvidenceMappingResult],
        *,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        required_ids = {requirement.id for requirement in structured_jd.must_have}
        mapped_ids = [item.requirement_id for item in result.value.items]
        if set(mapped_ids) != required_ids or len(mapped_ids) != len(required_ids):
            raise AgentGuardError(
                "requirement_coverage_mismatch",
                result.trace,
                node_id="evidence_mapping",
            )
        evidence_ids = [item.evidence_id for item in result.value.items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise AgentGuardError("duplicate_evidence_id", result.trace, node_id="evidence_mapping")
        reject_forbidden_output(
            result.value.model_dump_json(),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id="evidence_mapping",
        )
