import json
from pathlib import Path

from job_agent.evidence.evaluation.live_runner import load_corpus
from job_agent.evidence.evaluation.mock_runner import run_mock_suite
from job_agent.evidence.evaluation.runner import load_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval" / "evidence_v2"


def _load(short: str):
    manifest = load_manifest(DATA / f"manifest_{short}_v4.json")
    corpus = load_corpus(DATA / f"cases_{short}_v4.json", manifest)
    return manifest, corpus


def test_frozen_builder_is_reproducible(tmp_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_evidence_eval_corpus.py"), "--output-dir", str(tmp_path)],
        check=True,
    )
    for name in (
        "cases_dev_v4.json",
        "manifest_dev_v4.json",
        "cases_held_out_v4.json",
        "manifest_held_out_v4.json",
        "corpus_100_audit_v4.json",
    ):
        assert (tmp_path / name).read_bytes() == (DATA / name).read_bytes()


def test_100_cases_use_family_level_60_40_split_without_leakage():
    dev_manifest, dev = _load("dev")
    held_manifest, held = _load("held_out")
    assert (len(dev.cases), len(held.cases)) == (60, 40)
    dev_families = {item.family_id for item in dev.cases}
    held_families = {item.family_id for item in held.cases}
    assert (len(dev_families), len(held_families)) == (12, 8)
    assert dev_families.isdisjoint(held_families)
    assert all(sum(case.family_id == family for case in dev.cases) == 5 for family in dev_families)
    assert all(sum(case.family_id == family for case in held.cases) == 5 for family in held_families)
    assert dev_manifest.held_out_frozen is False
    assert held_manifest.held_out_frozen is True


def test_corpus_gold_spans_levels_and_edge_slices_are_auditable():
    cases = [*_load("dev")[1].cases, *_load("held_out")[1].cases]
    assert {case.expected_fit_band for case in cases} == {
        "high",
        "medium",
        "low",
        "insufficient_information",
    }
    assert {link.level.value for case in cases for link in case.gold_links} == {"C1", "C2", "C3"}
    assert all(
        link.source_key != "resume"
        for case in cases
        for link in case.gold_links
        if link.level.value in {"C2", "C3"}
    )
    assert all(
        link.source_key == "resume"
        for case in cases
        for link in case.gold_links
        if link.level.value == "C1"
    )
    verified_quotes = [
        link.resume_quote
        for case in cases
        for link in case.gold_links
        if link.level.value in {"C2", "C3"} and link.support_status == "supported"
    ]
    assert all(
        any(signal in quote for signal in ("baseline", "失败样本", "p95", "回滚"))
        for quote in verified_quotes
    )
    tags = {tag for case in cases for tag in case.slice_tags}
    assert {
        "sparse_jd",
        "compound_parent",
        "hard_gate",
        "partial",
        "contradiction",
        "multi_link",
        "generated_source_rejection",
        "unsupported_metric",
        "unicode_offsets",
        "resume_claim_gate",
        "repeated_quote",
    }.issubset(tags)
    for case in cases:
        source_texts = {
            "resume": case.resume_text,
            **{item.source_key: item.text for item in case.evidence_sources},
        }
        for atom in case.gold_atoms:
            assert case.jd_text.count(atom.jd_quote) == 1
        for link in case.gold_links:
            assert source_texts[link.source_key].count(link.resume_quote) == 1


def test_no_network_mock_executes_all_300_variant_case_rows():
    reports = [
        run_mock_suite(
            manifest_path=DATA / f"manifest_{short}_v4.json",
            corpus_path=DATA / f"cases_{short}_v4.json",
        )
        for short in ("dev", "held_out")
    ]
    assert sum(len(report.rows) for report in reports) == 300
    assert all(row.status == "passed" for report in reports for row in report.rows)
    assert all(
        variant.metrics.requirement_extraction.f1 == 1.0
        and variant.metrics.evidence_linking.f1 == 1.0
        and variant.metrics.span_exactness == 1.0
        and variant.fit_band_accuracy == 1.0
        for report in reports
        for variant in report.variants
    )


def test_audit_records_frozen_hashes_before_live_evaluation():
    audit = json.loads((DATA / "corpus_100_audit_v4.json").read_text(encoding="utf-8"))
    assert audit["synthetic_only"] is True
    assert audit["quality_claim_allowed"] is False
    assert sum(item["case_count"] for item in audit["splits"].values()) == 100
    assert all(item["frozen_before_live_run"] for item in audit["splits"].values())
