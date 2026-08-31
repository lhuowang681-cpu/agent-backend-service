from __future__ import annotations

import json

from job_agent.agents.base import (
    AgentGuardError,
    GuardFeedback,
    SkillLookup,
    invoke_with_guard_repair,
    reject_forbidden_output,
)
from job_agent.evidence.contracts import (
    AtomEvidenceResult,
    EvidenceLink,
    EvidenceLevel,
    EvidenceMappingCandidate,
    EvidenceMappingV2,
    EvidenceV2Error,
    RequirementAnalysis,
    SourceCatalog,
)
from job_agent.evidence.semantic_skill import with_output_schema
from job_agent.evidence.level_policy import EvidenceLevelPolicy
from job_agent.evidence.sources import EvidenceSourceStore
from job_agent.llm.harness import LLMHarness, NodePolicy
from job_agent.llm.provider import AgentResult, TraceContext


SKILL_ID = "evidence-contract"
PROMPT_VERSION = "evidence-mapping-v2.2-support-level-separation"
_VERIFIED_LEVELS = {EvidenceLevel.C2, EvidenceLevel.C3}


class EvidenceMappingService:
    def __init__(
        self,
        *,
        registry: SkillLookup,
        harness: LLMHarness,
        source_store: EvidenceSourceStore,
        level_policy: EvidenceLevelPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.harness = harness
        self.source_store = source_store
        self.level_policy = level_policy or EvidenceLevelPolicy()
        self.policy = NodePolicy(max_retries=0, allow_rule_fallback=False, max_output_tokens=6144)

    def map(
        self,
        analysis: RequirementAnalysis,
        *,
        catalog: SourceCatalog,
        session_id: str,
        run_id: str,
        forbidden_output_strings: tuple[str, ...] = (),
    ) -> EvidenceMappingV2:
        if not analysis.atoms:
            return EvidenceMappingV2()
        candidate_documents = [item for item in catalog.documents if item.kind != "raw_jd"]
        sources = {
            item.source_id: self.source_store.read(catalog, item.source_id)
            for item in candidate_documents
        }
        skill = with_output_schema(self.registry.get(SKILL_ID), EvidenceMappingCandidate)
        trace = TraceContext(
            session_id=session_id,
            run_id=run_id,
            node_id="evidence_mapping_v2",
            skill_id=skill.skill_id,
            skill_version=skill.version,
            prompt_version=PROMPT_VERSION,
        )

        def invoke(feedback: GuardFeedback | None) -> AgentResult[EvidenceMappingCandidate]:
            context: dict[str, object] = {
                "requirements": analysis.model_dump(mode="json"),
                "candidate_sources": [item.model_dump(mode="json") for item in candidate_documents],
                "instruction": (
                    "Map every atom exactly once. Cite links using source_id plus an exact quote and optional "
                    "prefix/suffix anchors when text repeats; never emit hashes or numeric offsets. Preserve "
                    "partial and contradictory evidence. Respect each candidate source artifact_type: original "
                    "resume/unknown/code/README/task note max C1; experiment/comparison/bad-case/design record "
                    "max C2; metric/production log/rollout/incident report max C3. Judge semantic coverage "
                    "independently from evidence level: an explicit complete statement may be semantically "
                    "supported at C1 while still remaining an unverified lead. Do not create links for unrelated "
                    "source text; use an empty link list for a gap."
                ),
            }
            if feedback:
                context["guard_feedback"] = feedback
            result = self.harness.invoke_structured(
                skill=skill,
                task_context=context,
                untrusted_inputs={"candidate_source_texts": json.dumps(sources, ensure_ascii=False)},
                output_schema=EvidenceMappingCandidate,
                tools=[],
                policy=self.policy,
                trace=trace,
            )
            return result

        result = invoke_with_guard_repair(
            invoke=invoke,
            guard=lambda value: self._guard(
                value,
                analysis=analysis,
                catalog=catalog,
                forbidden_output_strings=forbidden_output_strings,
            ),
            repair_instruction=(
                "Regenerate the complete mapping with one result per atom and source_id plus exact quotes. "
                "Use anchors or longer quotes when text repeats; never emit hashes or offsets. Cite no raw JD, "
                "respect artifact_type level caps, and preserve all contradictions."
            ),
        )
        return self._materialize(result.value, catalog)

    def _materialize(
        self, mapping: EvidenceMappingCandidate, catalog: SourceCatalog
    ) -> EvidenceMappingV2:
        documents = {item.source_id: item for item in catalog.documents}
        results: list[AtomEvidenceResult] = []
        for result in mapping.atom_results:
            links: list[EvidenceLink] = []
            for link in result.links:
                span = self.source_store.resolve_locator(catalog, link.source_locator)
                document = documents[span.source_id]
                self.level_policy.validate(document, link.level)
                links.append(
                    EvidenceLink(
                        link_id=link.link_id,
                        atom_id=link.atom_id,
                        source_span=span,
                        claim=link.claim,
                        support_status=link.support_status,
                        level=link.level,
                        confidence=link.confidence,
                        independence_group=self.source_store.independence_group(span),
                        needs_confirmation=link.needs_confirmation,
                    )
                )
            results.append(
                AtomEvidenceResult(
                    atom_id=result.atom_id,
                    links=links,
                    has_contradiction=any(link.support_status == "contradictory" for link in links),
                    match_status=self._match_status(links),
                    rationale=result.rationale,
                )
            )
        return EvidenceMappingV2(atom_results=results, warnings=mapping.warnings)

    def _guard(
        self,
        result: AgentResult[EvidenceMappingCandidate],
        *,
        analysis: RequirementAnalysis,
        catalog: SourceCatalog,
        forbidden_output_strings: tuple[str, ...],
    ) -> None:
        expected = {item.atom_id for item in analysis.atoms}
        observed = {item.atom_id for item in result.value.atom_results}
        if expected != observed or len(observed) != len(result.value.atom_results):
            raise AgentGuardError("mapping_atom_coverage_mismatch", result.trace, node_id="evidence_mapping_v2")
        documents = {item.source_id: item for item in catalog.documents}
        all_link_ids: list[str] = []
        for atom_result in result.value.atom_results:
            for link in atom_result.links:
                all_link_ids.append(link.link_id)
                document = documents.get(link.source_locator.source_id)
                if document is None or document.kind == "raw_jd":
                    raise AgentGuardError("forbidden_evidence_source", result.trace, node_id="evidence_mapping_v2")
        if len(all_link_ids) != len(set(all_link_ids)):
            raise AgentGuardError("duplicate_evidence_link_id", result.trace, node_id="evidence_mapping_v2")
        try:
            materialized = self._materialize(result.value, catalog)
        except EvidenceV2Error as exc:
            raise AgentGuardError(exc.error_code, result.trace, node_id="evidence_mapping_v2") from exc
        except ValueError as exc:
            raise AgentGuardError("invalid_evidence_mapping", result.trace, node_id="evidence_mapping_v2") from exc
        for atom_result in materialized.atom_results:
            for link in atom_result.links:
                try:
                    self.source_store.validate_span(catalog, link.source_span)
                except ValueError as exc:
                    raise AgentGuardError("invalid_evidence_span", result.trace, node_id="evidence_mapping_v2") from exc
        reject_forbidden_output(
            result.value.model_dump_json(),
            forbidden_output_strings=forbidden_output_strings,
            trace=result.trace,
            node_id="evidence_mapping_v2",
        )

    @staticmethod
    def _match_status(links) -> str:
        positive = [item for item in links if item.support_status in {"supported", "partial"}]
        if not positive:
            return "unsupported" if links else "gap"
        verified = [item for item in positive if item.level in _VERIFIED_LEVELS]
        if any(item.support_status == "supported" for item in verified):
            return "supported"
        if verified:
            return "partial"
        return "unverified_lead"
