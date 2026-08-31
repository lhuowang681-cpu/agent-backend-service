from pathlib import Path

from job_agent.evidence.evaluation.contracts import LiveEvaluationApproval
from job_agent.evidence.evaluation.live_contracts import LiveEvaluationReport, LiveVariantPrediction
from job_agent.evidence.evaluation.live_runner import (
    default_corpus_path,
    load_corpus,
    run_live_evaluation,
    rescore_live_report,
    score_prediction,
)
from job_agent.evidence.evaluation.runner import default_manifest_path, load_manifest
from job_agent.evidence.evaluation.pipeline_live_runner import run_full_pipeline_evaluation
from job_agent.llm.skill_registry import SkillSpec
from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.schemas import EvidenceMappingResult, StructuredJD


def _span(source: str, text: str, quote: str) -> dict:
    start = text.index(quote)
    return {
        "source": source,
        "start_offset": start,
        "end_offset": start + len(quote),
        "exact_quote": quote,
    }


def _gold_prediction(case) -> dict:
    source_texts = {
        "resume": case.resume_text,
        **{item.source_key: item.text for item in case.evidence_sources},
    }
    requirements = [
        {
            "atom_key": atom.atom_id,
            "text": atom.jd_quote,
            "importance": atom.importance,
            "hard_gate": atom.hard_gate,
            "source_span": _span("jd", case.jd_text, atom.jd_quote),
        }
        for atom in case.gold_atoms
    ]
    evidence_links = [
        {
            "link_key": link.link_id,
            "atom_key": link.atom_id,
            "source_span": _span(
                link.source_key, source_texts[link.source_key], link.resume_quote
            ),
            "support_status": link.support_status,
            "level": link.level.value,
        }
        for link in case.gold_links
    ]
    resume_claims = [
        {
            "atom_key": link.atom_id,
            "claim": link.resume_quote,
            "evidence_span": _span(
                link.source_key, source_texts[link.source_key], link.resume_quote
            ),
        }
        for link in case.gold_links
        if link.support_status == "supported" and link.level.value in {"C2", "C3"}
    ]
    return {
        "requirements": requirements,
        "evidence_links": evidence_links,
        "estimated_fit_band": case.expected_fit_band,
        "resume_claims": resume_claims,
    }


def _fixture():
    manifest = load_manifest(default_manifest_path())
    corpus = load_corpus(default_corpus_path(), manifest)
    return manifest, corpus


def _approval(case_count: int = 60, max_tokens: int = 10_000_000):
    return LiveEvaluationApproval(
        approved=True,
        endpoint="https://example.test",
        model="glm-5.2",
        case_count=case_count,
        max_total_tokens=max_tokens,
    )


def test_live_dry_run_executes_all_three_variants_with_same_cases(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    responses = [
        _gold_prediction(case)
        for _variant in manifest.variants
        for case in corpus.cases
    ]
    provider = MockLLMProvider(
        responses,
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(),
        provider=provider,
    )
    assert len(provider.calls) == 180
    assert len(report.rows) == 180
    assert all(item.failed_rows == 0 for item in report.variants)
    assert all(item.fit_band_accuracy == 1.0 for item in report.variants)
    assert all(item.metrics.span_exactness == 1.0 for item in report.variants)
    assert report.resume_quality_claim_allowed is False


def test_live_budget_gate_blocks_calls_before_provider_use(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    provider = MockLLMProvider(
        [], provider_name="anthropic_compatible", model="glm-5.2"
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(max_tokens=1),
        provider=provider,
    )
    assert provider.calls == []
    assert len(report.rows) == 180
    assert all(row.error_code == "token_budget_exhausted" for row in report.rows)


def test_canary_preserves_schema_failure_row(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    provider = MockLLMProvider(
        ["not json"], provider_name="anthropic_compatible", model="glm-5.2"
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(),
        provider=provider,
        canary_only=True,
    )
    assert len(report.rows) == 1
    assert report.rows[0].status == "failed"
    assert report.rows[0].error_code == "invalid_json"


def test_scorer_separates_invalid_span_from_semantics_and_rejects_unverified_claim():
    _, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-03")
    payload = _gold_prediction(case)
    payload["requirements"][0]["source_span"]["exact_quote"] = "伪造"
    partial = next(
        link for link in case.gold_links if link.support_status == "partial"
    )
    payload["resume_claims"].append(
        {
            "atom_key": partial.atom_id,
            "claim": partial.resume_quote,
            "evidence_span": _span("resume", case.resume_text, partial.resume_quote),
        }
    )
    observation, _ = score_prediction(case, LiveVariantPrediction.model_validate(payload))
    assert "sft_training" in observation.predicted_requirement_ids
    assert "invalid_jd_span_0" in observation.predicted_span_ids
    assert observation.resume_claim_supported[-1] is False


def test_unknown_claim_reference_is_scored_instead_of_rejecting_row():
    _, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-02")
    payload = _gold_prediction(case)
    link = next(item for item in case.gold_links if item.support_status == "supported")
    payload["resume_claims"].append(
        {
            "atom_key": "claim_generated_key",
            "claim": link.resume_quote,
            "evidence_span": _span("resume", case.resume_text, link.resume_quote),
        }
    )
    prediction = LiveVariantPrediction.model_validate(payload)
    observation, _ = score_prediction(case, prediction)
    assert observation.resume_claim_supported[0] is False


def test_requirement_concept_matches_unique_containing_span():
    _, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-01")
    payload = _gold_prediction(case)
    quote = "要求独立完成 SFT 训练"
    payload["requirements"][0]["source_span"] = _span("jd", case.jd_text, quote)
    prediction = LiveVariantPrediction.model_validate(payload)
    observation, _ = score_prediction(case, prediction)
    assert "sft_training" in observation.predicted_requirement_ids
    assert f"jd:0:{len(quote)}" in observation.predicted_span_ids


def test_semantic_score_and_exact_span_are_separate_axes():
    _, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-01")
    payload = _gold_prediction(case)
    payload["requirements"][0]["source_span"]["end_offset"] -= 1
    prediction = LiveVariantPrediction.model_validate(payload)
    observation, _ = score_prediction(case, prediction)
    assert "sft_training" in observation.predicted_requirement_ids
    assert "invalid_jd_span_0" in observation.predicted_span_ids


def test_atom_semantics_match_when_pipeline_cites_shared_parent_span():
    _, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-01")
    payload = _gold_prediction(case)
    parent_span = _span("jd", case.jd_text, case.jd_text)
    for requirement in payload["requirements"]:
        requirement["source_span"] = parent_span
    prediction = LiveVariantPrediction.model_validate(payload)
    observation, _ = score_prediction(case, prediction)
    assert set(observation.predicted_requirement_ids) == {
        "sft_training",
        "offline_eval",
        "inference_deploy",
    }
    assert observation.expected_span_ids != observation.predicted_span_ids


def test_existing_rows_can_be_rescored_without_provider_calls(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    responses = [
        _gold_prediction(case)
        for _variant in manifest.variants
        for case in corpus.cases
    ]
    provider = MockLLMProvider(
        responses, provider_name="anthropic_compatible", model="glm-5.2"
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(),
        provider=provider,
    )
    calls_before = len(provider.calls)
    rescored = rescore_live_report(report, corpus)
    assert len(provider.calls) == calls_before
    assert all(item.metrics.requirement_extraction.f1 == 1.0 for item in rescored.variants)


def test_checkpoint_resume_skips_completed_rows_without_duplicate_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    cases = corpus.cases[:2]
    checkpoint = tmp_path / "checkpoint.json"

    class InterruptedProvider(MockLLMProvider):
        def generate_structured(self, **kwargs):
            if len(self.calls) == 1:
                raise RuntimeError("simulated process interruption")
            return super().generate_structured(**kwargs)

    interrupted = InterruptedProvider(
        [_gold_prediction(cases[0])],
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    import pytest

    with pytest.raises(RuntimeError, match="interruption"):
        run_live_evaluation(
            manifest=manifest,
            corpus=corpus,
            approval=_approval(case_count=2),
            provider=interrupted,
            checkpoint_path=checkpoint,
        )
    partial = LiveEvaluationReport.model_validate_json(checkpoint.read_text(encoding="utf-8"))
    assert len(partial.rows) == 1

    remaining = [
        _gold_prediction(case)
        for variant_index, _variant in enumerate(manifest.variants)
        for case_index, case in enumerate(cases)
        if not (variant_index == 0 and case_index == 0)
    ]
    resumed_provider = MockLLMProvider(
        remaining,
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=2),
        provider=resumed_provider,
        checkpoint_path=checkpoint,
        resume_report=partial,
    )
    assert len(resumed_provider.calls) == 5
    assert len(report.rows) == 6
    assert len({(item.variant_id, item.case_id) for item in report.rows}) == 6


def test_failed_checkpoint_row_requires_explicit_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    checkpoint = tmp_path / "failed.json"
    failed_provider = MockLLMProvider(
        ["not json"], provider_name="anthropic_compatible", model="glm-5.2"
    )
    first = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=1),
        provider=failed_provider,
        canary_only=True,
        checkpoint_path=checkpoint,
    )
    skipped_provider = MockLLMProvider(
        [_gold_prediction(corpus.cases[0])],
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    skipped = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=1),
        provider=skipped_provider,
        canary_only=True,
        resume_report=first,
    )
    assert skipped_provider.calls == []
    assert skipped.rows[0].status == "failed"

    retry_provider = MockLLMProvider(
        [_gold_prediction(corpus.cases[0])],
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    retried = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=1),
        provider=retry_provider,
        canary_only=True,
        resume_report=first,
        retry_failed=True,
    )
    assert len(retry_provider.calls) == 1
    assert retried.rows[0].status == "passed"


def test_production_pipeline_live_runner_handles_sparse_case(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()

    class Registry:
        def get(self, skill_id: str) -> SkillSpec:
            output = StructuredJD if skill_id == "jd-analysis" else EvidenceMappingResult
            return SkillSpec(
                skill_id=skill_id,
                version="fixture-v1",
                instructions="Use exact source spans and do not fabricate.",
                reference_paths=(Path("skill-references/fixture.md"),),
                output_schema=output,
            )

    provider = MockLLMProvider(
        [{"parents": [], "atoms": [], "warnings": []}],
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    report = run_full_pipeline_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=1),
        provider=provider,
        registry=Registry(),
    )
    assert len(provider.calls) == 1
    assert report.variants[0].passed_rows == 1
    assert report.variants[0].fit_band_accuracy == 1.0


def test_production_pipeline_checkpoint_resume_avoids_repeat_calls(monkeypatch, tmp_path):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()

    class Registry:
        def get(self, skill_id: str) -> SkillSpec:
            output = StructuredJD if skill_id == "jd-analysis" else EvidenceMappingResult
            return SkillSpec(
                skill_id=skill_id,
                version="fixture-v1",
                instructions="Use exact source spans and do not fabricate.",
                reference_paths=(Path("skill-references/fixture.md"),),
                output_schema=output,
            )

    empty = {"parents": [], "atoms": [], "warnings": []}
    checkpoint = tmp_path / "production-checkpoint.json"

    class SimulatedInterrupt(BaseException):
        pass

    class InterruptedProvider(MockLLMProvider):
        def generate_structured(self, **kwargs):
            if len(self.calls) == 1:
                raise SimulatedInterrupt("simulated production interruption")
            return super().generate_structured(**kwargs)

    interrupted = InterruptedProvider(
        [empty], provider_name="anthropic_compatible", model="glm-5.2"
    )
    import pytest

    with pytest.raises(SimulatedInterrupt, match="interruption"):
        run_full_pipeline_evaluation(
            manifest=manifest,
            corpus=corpus,
            approval=_approval(case_count=2),
            provider=interrupted,
            registry=Registry(),
            checkpoint_path=checkpoint,
        )
    partial = LiveEvaluationReport.model_validate_json(checkpoint.read_text(encoding="utf-8"))
    assert len(partial.rows) == 1

    resumed_provider = MockLLMProvider(
        [empty], provider_name="anthropic_compatible", model="glm-5.2"
    )
    report = run_full_pipeline_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=_approval(case_count=2),
        provider=resumed_provider,
        registry=Registry(),
        resume_report=partial,
    )
    assert len(resumed_provider.calls) == 1
    assert len(report.rows) == 2
    assert all(row.status == "passed" for row in report.rows)


def test_production_pipeline_maps_independent_c2_artifact_sources(monkeypatch):
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "test-key")
    manifest, corpus = _fixture()
    case = next(item for item in corpus.cases if item.case_id == "dev-compound_parent-01")
    case = case.model_copy(
        update={
            "evidence_sources": [
                source.model_copy(update={"artifact_type": "experiment_record"})
                for source in case.evidence_sources
            ]
        }
    )
    one_case_corpus = corpus.model_copy(update={"cases": [case]})

    class Registry:
        def get(self, skill_id: str) -> SkillSpec:
            output = StructuredJD if skill_id == "jd-analysis" else EvidenceMappingResult
            return SkillSpec(
                skill_id=skill_id,
                version="fixture-v1",
                instructions="Use exact source spans and do not fabricate.",
                reference_paths=(Path("skill-references/fixture.md"),),
                output_schema=output,
            )

    def locator(source_id: str, quote: str) -> dict:
        return {"source_id": source_id, "exact_quote": quote}

    parents = []
    atoms = []
    for index, gold in enumerate(case.gold_atoms, start=1):
        source_locator = locator("src_001_raw_jd", gold.jd_quote)
        parents.append(
            {
                "requirement_id": f"parent-{index}",
                "text": gold.jd_quote,
                "importance": gold.importance,
                "hard_gate": gold.hard_gate,
                "source_locator": source_locator,
                "atom_ids": [gold.atom_id],
            }
        )
        atoms.append(
            {
                "atom_id": gold.atom_id,
                "parent_requirement_id": f"parent-{index}",
                "text": gold.jd_quote,
                "source_locator": source_locator,
            }
        )
    source_by_key = {
        source.source_key: (f"src_{index:03d}_user_artifact", source.text)
        for index, source in enumerate(case.evidence_sources, start=3)
    }
    atom_results = []
    for gold in case.gold_atoms:
        links = []
        for link in (item for item in case.gold_links if item.atom_id == gold.atom_id):
            source_id, text = source_by_key[link.source_key]
            links.append(
                {
                    "link_id": link.link_id,
                    "atom_id": link.atom_id,
                    "source_locator": locator(source_id, link.resume_quote),
                    "claim": link.resume_quote,
                    "support_status": link.support_status,
                    "level": link.level.value,
                    "confidence": 0.9,
                }
            )
        atom_results.append(
            {
                "atom_id": gold.atom_id,
                "links": links,
                "rationale": "independent artifact fixture",
            }
        )
    provider = MockLLMProvider(
        [
            {"parents": parents, "atoms": atoms, "warnings": []},
            {"atom_results": atom_results, "warnings": []},
        ],
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    report = run_full_pipeline_evaluation(
        manifest=manifest,
        corpus=one_case_corpus,
        approval=_approval(case_count=1),
        provider=provider,
        registry=Registry(),
    )
    assert len(provider.calls) == 2
    assert report.rows[0].status == "passed"
    assert report.variants[0].metrics.requirement_extraction.f1 == 1.0
    assert report.variants[0].metrics.evidence_linking.f1 == 1.0
    assert {
        link.source_span.source for link in report.rows[0].prediction.evidence_links
    } == {item.source_key for item in case.evidence_sources}
