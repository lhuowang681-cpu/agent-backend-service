from __future__ import annotations

import json

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
from job_agent.schemas import RawJob, StructuredJD


SKILL_ID = "jd-analysis"
_PROMPT = get_prompt("semantic.jd-structurer")
PROMPT_VERSION = _PROMPT.identity


class JDStructurerAgent:
    """Semantic JD extraction with immutable job identity and source-text guards."""

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
        job: RawJob,
        *,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> AgentResult[StructuredJD]:
        skill = self.registry.get(SKILL_ID)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="jd_structurer",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )
        def invoke(guard_feedback: GuardFeedback | None) -> AgentResult[StructuredJD]:
            task_context = {
                "job_id": job.job_id,
                "company": job.company,
                "title": job.title,
                "location": job.location,
                "instruction": _PROMPT.task_instruction,
            }
            if forbidden_output_strings:
                task_context["forbidden_output_strings"] = list(forbidden_output_strings)
            if guard_feedback is not None:
                task_context["guard_feedback"] = guard_feedback
            return self.harness.invoke_structured(
                skill=skill,
                task_context=task_context,
                untrusted_inputs={"jd": job.desc},
                output_schema=StructuredJD,
                tools=[],
                policy=self.policy,
                trace=trace,
            )

        return invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda result: self._guard(
                job,
                result,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=_PROMPT.repair_instruction,
        )

    @staticmethod
    def _guard(
        job: RawJob,
        result: AgentResult[StructuredJD],
        *,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> None:
        structured = result.value
        if structured.company != job.company:
            raise AgentGuardError("company_mismatch", result.trace, node_id="jd_structurer")
        if structured.title != job.title:
            raise AgentGuardError("title_mismatch", result.trace, node_id="jd_structurer")
        if structured.raw_jd != job.desc:
            raise AgentGuardError("raw_jd_mismatch", result.trace, node_id="jd_structurer")
        requirement_ids = [requirement.id for requirement in structured.must_have]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise AgentGuardError("duplicate_requirement_id", result.trace, node_id="jd_structurer")
        reject_forbidden_output(
            json.dumps(
                structured.model_dump(mode="json", exclude={"raw_jd"}),
                ensure_ascii=False,
            ),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id="jd_structurer",
        )
