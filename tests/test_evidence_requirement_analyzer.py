from pathlib import Path

from job_agent.evidence.contracts import SourceInput
from job_agent.evidence.requirements import RequirementAnalyzer
from job_agent.evidence.sources import EvidenceSourceStore
from job_agent.llm.harness import LLMHarness
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import StructuredJD


class _Registry:
    def get(self, skill_id: str) -> SkillSpec:
        return SkillSpec(
            skill_id=skill_id,
            version="fixture-v1",
            instructions="Extract explicit requirements only.",
            reference_paths=(Path("skill-references/jd-analysis.md"),),
            output_schema=StructuredJD,
        )


def test_requirement_analyzer_builds_bounded_atoms_with_exact_jd_span(tmp_path: Path):
    jd = "独立完成训练和评测"
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot("run-1", [SourceInput(kind="raw_jd", display_label="JD", text=jd)])
    document = catalog.documents[0]
    locator = {"source_id": document.source_id, "exact_quote": jd}
    provider = MockLLMProvider(
        [
            {
                "parents": [
                    {
                        "requirement_id": "r1",
                        "text": jd,
                        "importance": "must",
                        "hard_gate": True,
                        "source_locator": locator,
                        "atom_ids": ["a1", "a2"],
                    }
                ],
                "atoms": [
                    {"atom_id": "a1", "parent_requirement_id": "r1", "text": "训练", "source_locator": locator, "verification_signals": ["训练流程"]},
                    {"atom_id": "a2", "parent_requirement_id": "r1", "text": "评测", "source_locator": locator, "verification_signals": ["评测设计"]},
                ],
                "warnings": [],
            }
        ]
    )
    analysis = RequirementAnalyzer(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).analyze(document, catalog=catalog, session_id="s1", run_id="run-1")
    assert [item.text for item in analysis.atoms] == ["训练", "评测"]
    assert analysis.warnings == ["insufficient_jd_detail"]
    assert len(provider.calls) == 1


def test_requirement_analyzer_accepts_explicit_zero_requirement_result(tmp_path: Path):
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot("run-1", [SourceInput(kind="raw_jd", display_label="JD", text="岗位介绍")])
    provider = MockLLMProvider([{"parents": [], "atoms": [], "warnings": []}])
    result = RequirementAnalyzer(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).analyze(catalog.documents[0], catalog=catalog, session_id="s1", run_id="run-1")
    assert result.atoms == []
    assert "insufficient_jd_detail" in result.warnings


def test_requirement_analyzer_materializes_quote_only_candidate(tmp_path: Path):
    jd = "前缀 独立评测 后缀"
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot(
        "run-1", [SourceInput(kind="raw_jd", display_label="JD", text=jd)]
    )
    document = catalog.documents[0]
    locator = {"source_id": document.source_id, "exact_quote": "独立评测"}
    provider = MockLLMProvider(
        [
            {
                "parents": [
                    {
                        "requirement_id": "r1",
                        "text": "独立评测",
                        "importance": "must",
                        "hard_gate": False,
                        "source_locator": locator,
                        "atom_ids": ["a1"],
                    }
                ],
                "atoms": [
                    {
                        "atom_id": "a1",
                        "parent_requirement_id": "r1",
                        "text": "独立评测",
                        "source_locator": locator,
                        "verification_signals": [],
                    }
                ],
                "warnings": [],
            }
        ]
    )
    analysis = RequirementAnalyzer(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).analyze(document, catalog=catalog, session_id="s1", run_id="run-1")
    assert analysis.atoms[0].source_span.start_offset == 3
    assert len(provider.calls) == 1


def test_requirement_analyzer_repairs_ambiguous_quote_with_anchor(tmp_path: Path):
    jd = "要求：Python。加分：Python。"
    store = EvidenceSourceStore(tmp_path)
    catalog = store.snapshot("run-1", [SourceInput(kind="raw_jd", display_label="JD", text=jd)])
    source_id = catalog.documents[0].source_id
    base = {
        "parents": [
            {
                "requirement_id": "r1",
                "text": "Python",
                "importance": "must",
                "hard_gate": False,
                "source_locator": {"source_id": source_id, "exact_quote": "Python"},
                "atom_ids": ["a1"],
            }
        ],
        "atoms": [
            {
                "atom_id": "a1",
                "parent_requirement_id": "r1",
                "text": "Python",
                "source_locator": {"source_id": source_id, "exact_quote": "Python"},
                "verification_signals": [],
            }
        ],
        "warnings": [],
    }
    repaired = {
        **base,
        "parents": [{**base["parents"][0], "source_locator": {"source_id": source_id, "exact_quote": "Python", "prefix_anchor": "要求："}}],
        "atoms": [{**base["atoms"][0], "source_locator": {"source_id": source_id, "exact_quote": "Python", "prefix_anchor": "要求："}}],
    }
    provider = MockLLMProvider([base, repaired])
    result = RequirementAnalyzer(
        registry=_Registry(), harness=LLMHarness(provider), source_store=store
    ).analyze(catalog.documents[0], catalog=catalog, session_id="s1", run_id="run-1")
    assert result.parents[0].source_span.start_offset == len("要求：")
    assert len(provider.calls) == 2
    assert "ambiguous_source_quote" in provider.calls[1]["user_prompt"]
