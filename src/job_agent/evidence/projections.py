from __future__ import annotations

from dataclasses import dataclass

from job_agent.evidence.contracts import (
    EvidenceBundle,
    EvidenceDisplayItem,
    EvidenceDisplayView,
    EvidenceLevel,
    InterviewEvidenceAtom,
    InterviewEvidenceView,
    ResumeEvidenceClaim,
    ResumeEvidenceView,
)
from job_agent.schemas import EvidenceItem, JobRequirement, RoleType, StructuredJD


@dataclass(frozen=True)
class ResumeAgentInputs:
    """Narrow compatibility input for the existing application-material agent."""

    structured_jd: StructuredJD
    evidence: list[EvidenceItem]


class EvidenceProjectionService:
    def for_resume(self, bundle: EvidenceBundle) -> ResumeEvidenceView:
        claims: list[ResumeEvidenceClaim] = []
        for result in bundle.mapping.atom_results:
            eligible = [
                link
                for link in result.links
                if link.level in {EvidenceLevel.C2, EvidenceLevel.C3}
                and link.support_status in {"supported", "partial"}
            ]
            if not eligible:
                continue
            best = max(eligible, key=lambda item: (item.level == EvidenceLevel.C3, item.confidence))
            claims.append(
                ResumeEvidenceClaim(
                    atom_id=result.atom_id,
                    claim=best.claim,
                    level=best.level,
                    link_ids=[item.link_id for item in eligible],
                    boundary="仅部分支持，简历表述必须保留范围边界"
                    if best.support_status == "partial"
                    else None,
                )
            )
        return ResumeEvidenceView(run_id=bundle.manifest.run_id, claims=claims)

    def for_interview(self, bundle: EvidenceBundle) -> InterviewEvidenceView:
        parents = {parent.requirement_id: parent for parent in bundle.requirements.parents}
        atoms = {atom.atom_id: atom for atom in bundle.requirements.atoms}
        items: list[InterviewEvidenceAtom] = []
        for result in bundle.mapping.atom_results:
            atom = atoms[result.atom_id]
            parent = parents[atom.parent_requirement_id]
            items.append(
                InterviewEvidenceAtom(
                    atom_id=atom.atom_id,
                    requirement_text=parent.text,
                    atom_text=atom.text,
                    importance=parent.importance,
                    match_status=result.match_status,
                    has_contradiction=result.has_contradiction,
                    links=result.links,
                )
            )
        return InterviewEvidenceView(run_id=bundle.manifest.run_id, atoms=items)

    def for_resume_agent(
        self,
        bundle: EvidenceBundle,
        *,
        company: str,
        title: str,
        raw_jd: str,
    ) -> ResumeAgentInputs:
        """Adapt only the verified Resume projection to the legacy agent contract."""
        parents = {item.requirement_id: item for item in bundle.requirements.parents}
        atoms = {item.atom_id: item for item in bundle.requirements.atoms}
        results = {item.atom_id: item for item in bundle.mapping.atom_results}
        resume_view = self.for_resume(bundle)
        evidence: list[EvidenceItem] = []
        for claim in resume_view.claims:
            links = {item.link_id: item for item in results[claim.atom_id].links}
            for link_id in claim.link_ids:
                link = links[link_id]
                if link.support_status != "supported":
                    continue
                evidence.append(
                    EvidenceItem(
                        evidence_id=link.link_id,
                        requirement_id=claim.atom_id,
                        claim=link.claim,
                        level=link.level,
                        proof=link.source_span.exact_quote,
                        risk="来自 Evidence v2 已校验且完整支持的 source span",
                    )
                )
        structured_jd = StructuredJD(
            company=company,
            title=title,
            role_type=RoleType.UNKNOWN,
            must_have=[
                JobRequirement(
                    id=atom.atom_id,
                    text=atom.text,
                    required=parents[atom.parent_requirement_id].importance == "must",
                    probe=f"验证：{atom.text}",
                )
                for atom in bundle.requirements.atoms
            ],
            nice_to_have=[item.text for item in bundle.requirements.parents if item.importance == "nice"],
            raw_jd=raw_jd,
        )
        return ResumeAgentInputs(structured_jd=structured_jd, evidence=evidence)

    def for_ui(self, bundle: EvidenceBundle) -> EvidenceDisplayView:
        atoms = {atom.atom_id: atom for atom in bundle.requirements.atoms}
        items = [
            EvidenceDisplayItem(
                atom_id=result.atom_id,
                label=atoms[result.atom_id].text,
                jd_quote=atoms[result.atom_id].source_span.exact_quote,
                evidence_quotes=[link.source_span.exact_quote for link in result.links],
            )
            for result in bundle.mapping.atom_results
        ]
        return EvidenceDisplayView(
            run_id=bundle.manifest.run_id,
            summary=bundle.assessment.display_summary,
            items=items,
        )
