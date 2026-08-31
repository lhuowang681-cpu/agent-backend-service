from pathlib import Path

from job_agent.evidence.contracts import (
    EvidenceLevel,
    EvidenceMappingV2,
    ParentRequirement,
    RequirementAnalysis,
    RequirementAtom,
    SourceInput,
)
from job_agent.evidence.mapping import EvidenceMappingService
from job_agent.evidence.sources import EvidenceSourceStore
from job_agent.llm.harness import LLMHarness
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import EvidenceMappingResult


class _Registry:
    def get(self, skill_id: str) -> SkillSpec:
        return SkillSpec(
            skill_id=skill_id,
            version="fixture-v1",
            instructions="Map evidence without fabrication.",
            reference_paths=(Path("skill-references/evidence-contract.md"),),
            output_schema=EvidenceMappingResult,
        )


def test_mapping_normalizes_c1_and_preserves_contradiction(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [
            SourceInput(kind="raw_jd", display_label="JD", text="模型评测"),
            SourceInput(kind="original_resume", display_label="Resume", text="参与模型评测，但未独立负责"),
        ],
    )
    jd_doc, resume_doc = catalog.documents
    jd_span = store.make_span(catalog, jd_doc.source_id, 0, len("模型评测"))
    analysis = RequirementAnalysis(
        parents=[
            ParentRequirement(
                requirement_id="r1",
                text="模型评测",
                importance="must",
                source_span=jd_span,
                atom_ids=["a1"],
            )
        ],
        atoms=[
            RequirementAtom(
                atom_id="a1", parent_requirement_id="r1", text="独立评测", source_span=jd_span
            )
        ],
    )
    quote = "参与模型评测"
    evidence_span = store.make_span(catalog, resume_doc.source_id, 0, len(quote))
    payload = {
        "atom_results": [
            {
                "atom_id": "a1",
                "links": [
                    {
                        "link_id": "l1",
                        "atom_id": "a1",
                        "source_locator": {"source_id": resume_doc.source_id, "exact_quote": quote},
                        "claim": quote,
                        "support_status": "partial",
                        "level": "C1",
                        "confidence": 0.6,
                        "needs_confirmation": True,
                    },
                    {
                        "link_id": "l2",
                        "atom_id": "a1",
                        "source_locator": {"source_id": resume_doc.source_id, "exact_quote": "未独立负责"},
                        "claim": "未独立负责",
                        "support_status": "contradictory",
                        "level": "C1",
                        "confidence": 0.9,
                        "needs_confirmation": True,
                    },
                ],
                "rationale": "related but ownership conflicts",
            }
        ],
    }
    provider = MockLLMProvider([payload])
    mapping = EvidenceMappingService(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).map(analysis, catalog=catalog, session_id="s1", run_id="run-1")
    result = mapping.atom_results[0]
    assert result.match_status == "unverified_lead"
    assert result.has_contradiction is True
    assert all(link.independence_group for link in result.links)
    assert all(link.level == EvidenceLevel.C1 for link in result.links)
    assert result.links[0].source_span == evidence_span


def test_mapping_repairs_level_above_source_cap(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1",
        [
            SourceInput(kind="raw_jd", display_label="JD", text="独立评测"),
            SourceInput(
                kind="project_source",
                display_label="实验",
                text="独立完成离线评测",
                artifact_type="experiment_record",
            ),
        ],
    )
    jd_doc, evidence_doc = catalog.documents
    jd_span = store.make_span(catalog, jd_doc.source_id, 0, len("独立评测"))
    analysis = RequirementAnalysis(
        parents=[ParentRequirement(requirement_id="r1", text="独立评测", importance="must", source_span=jd_span, atom_ids=["a1"])],
        atoms=[RequirementAtom(atom_id="a1", parent_requirement_id="r1", text="独立评测", source_span=jd_span)],
    )

    def payload(level: str) -> dict:
        return {
            "atom_results": [
                {
                    "atom_id": "a1",
                    "links": [
                        {
                            "link_id": "l1",
                            "atom_id": "a1",
                            "source_locator": {"source_id": evidence_doc.source_id, "exact_quote": "独立完成离线评测"},
                            "claim": "独立完成离线评测",
                            "support_status": "supported",
                            "level": level,
                            "confidence": 0.9,
                            "needs_confirmation": False,
                        }
                    ],
                    "rationale": "有实验记录",
                }
            ],
            "warnings": [],
        }

    provider = MockLLMProvider([payload("C3"), payload("C2")])
    mapping = EvidenceMappingService(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).map(analysis, catalog=catalog, session_id="s1", run_id="run-1")
    assert mapping.atom_results[0].links[0].level == EvidenceLevel.C2
    assert len(provider.calls) == 2
    assert "evidence_level_exceeds_source_cap" in provider.calls[1]["user_prompt"]


def test_empty_analysis_skips_provider(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot("run-1", [SourceInput(kind="raw_jd", display_label="JD", text="介绍")])
    provider = MockLLMProvider([])
    mapping = EvidenceMappingService(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).map(RequirementAnalysis(), catalog=catalog, session_id="s1", run_id="run-1")
    assert mapping == EvidenceMappingV2()
    assert provider.calls == []
