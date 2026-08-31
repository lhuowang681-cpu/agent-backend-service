from __future__ import annotations

from job_agent.agents.base import (
    AgentGuardError,
    GuardFeedback,
    SkillLookup,
    invoke_with_guard_repair,
    reject_forbidden_output,
)
from job_agent.evidence.contracts import (
    EvidenceV2Error,
    ParentRequirement,
    RequirementAnalysis,
    RequirementAnalysisCandidate,
    RequirementAtom,
    SourceCatalog,
    SourceDocumentRef,
)
from job_agent.evidence.semantic_skill import with_output_schema
from job_agent.evidence.sources import EvidenceSourceStore
from job_agent.llm.harness import LLMHarness, NodePolicy
from job_agent.llm.provider import AgentResult, TraceContext


SKILL_ID = "jd-analysis"
PROMPT_VERSION = "evidence-requirements-v2.1-quote-locator"


class RequirementAnalyzer:
    def __init__(
        self,
        *,
        registry: SkillLookup,
        harness: LLMHarness,
        source_store: EvidenceSourceStore,
    ) -> None:
        self.registry = registry
        self.harness = harness
        self.source_store = source_store
        self.policy = NodePolicy(max_retries=0, allow_rule_fallback=False, max_output_tokens=4096)

    def analyze(
        self,
        jd_source: SourceDocumentRef,
        *,
        catalog: SourceCatalog,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> RequirementAnalysis:
        if jd_source.kind != "raw_jd":
            raise ValueError("requirement analysis requires raw_jd source")
        jd_text = self.source_store.read(catalog, jd_source.source_id)
        skill = with_output_schema(self.registry.get(SKILL_ID), RequirementAnalysisCandidate)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="evidence_requirements_v2",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(feedback: GuardFeedback | None) -> AgentResult[RequirementAnalysisCandidate]:
            context: dict[str, object] = {
                "source_id": jd_source.source_id,
                "instruction": (
                    "Extract only explicit JD requirements. Cite each parent and atom using source_id plus an "
                    "exact quote; use prefix/suffix anchors only when the quote repeats. Never emit hashes or "
                    "numeric offsets. Split a parent "
                    "into 1-4 independently verifiable atoms, never add role-common requirements. "
                    "Return an empty analysis with insufficient_jd_detail when no explicit requirement exists."
                ),
            }
            if feedback:
                context["guard_feedback"] = feedback
            result = self.harness.invoke_structured(
                skill=skill,
                task_context=context,
                untrusted_inputs={"raw_jd": jd_text},
                output_schema=RequirementAnalysisCandidate,
                tools=[],
                policy=self.policy,
                trace=trace,
            )
            return result

        result = invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda value: self._guard(
                value,
                catalog=catalog,
                jd_source=jd_source,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=(
                "Regenerate the complete analysis with source_id and exact quotes from the raw JD, adding "
                "prefix/suffix anchors or a longer quote when text repeats. Never return hashes or offsets. "
                "Keep 1-4 atoms per parent, no inferred requirements, and valid unique ids."
            ),
        )
        analysis = self._materialize(result.value, catalog)
        if len(analysis.atoms) < 3 and "insufficient_jd_detail" not in analysis.warnings:
            analysis = analysis.model_copy(update={"warnings": [*analysis.warnings, "insufficient_jd_detail"]})
        return analysis

    def _materialize(
        self, candidate: RequirementAnalysisCandidate, catalog: SourceCatalog
    ) -> RequirementAnalysis:
        parents = [
            ParentRequirement(
                requirement_id=item.requirement_id,
                text=item.text,
                importance=item.importance,
                hard_gate=item.hard_gate,
                source_span=self.source_store.resolve_locator(catalog, item.source_locator),
                atom_ids=item.atom_ids,
            )
            for item in candidate.parents
        ]
        atoms = [
            RequirementAtom(
                atom_id=item.atom_id,
                parent_requirement_id=item.parent_requirement_id,
                text=item.text,
                source_span=self.source_store.resolve_locator(catalog, item.source_locator),
                verification_signals=item.verification_signals,
            )
            for item in candidate.atoms
        ]
        return RequirementAnalysis(parents=parents, atoms=atoms, warnings=candidate.warnings)

    def _guard(
        self,
        result: AgentResult[RequirementAnalysisCandidate],
        *,
        catalog: SourceCatalog,
        jd_source: SourceDocumentRef,
        forbidden_output_strings: tuple[str, ...],
    ) -> None:
        for parent in result.value.parents:
            if parent.source_locator.source_id != jd_source.source_id:
                raise AgentGuardError("requirement_source_mismatch", result.trace, node_id=trace_node())
        for atom in result.value.atoms:
            if atom.source_locator.source_id != jd_source.source_id:
                raise AgentGuardError("requirement_source_mismatch", result.trace, node_id=trace_node())
        try:
            materialized = self._materialize(result.value, catalog)
        except EvidenceV2Error as exc:
            raise AgentGuardError(exc.error_code, result.trace, node_id=trace_node()) from exc
        except ValueError as exc:
            raise AgentGuardError("invalid_requirement_graph", result.trace, node_id=trace_node()) from exc
        for parent in materialized.parents:
            if parent.source_span.source_id != jd_source.source_id:
                raise AgentGuardError("requirement_source_mismatch", result.trace, node_id=trace_node())
            self._validate_span(result, catalog, parent.source_span)
        for atom in materialized.atoms:
            self._validate_span(result, catalog, atom.source_span)
        reject_forbidden_output(
            result.value.model_dump_json(),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id=trace_node(),
        )

    def _validate_span(self, result: AgentResult, catalog: SourceCatalog, span) -> None:
        try:
            self.source_store.validate_span(catalog, span)
        except ValueError as exc:
            raise AgentGuardError("invalid_requirement_span", result.trace, node_id=trace_node()) from exc


def trace_node() -> str:
    return "evidence_requirements_v2"
