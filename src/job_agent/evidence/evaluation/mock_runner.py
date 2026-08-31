from __future__ import annotations

import argparse
from pathlib import Path

from job_agent.evidence.evaluation.contracts import LiveEvaluationApproval
from job_agent.evidence.evaluation.live_contracts import LiveEvalCase
from job_agent.evidence.evaluation.live_runner import _atomic_json, load_corpus, run_live_evaluation
from job_agent.evidence.evaluation.runner import load_manifest
from job_agent.llm.providers.mock import MockLLMProvider


def _span(source: str, text: str, quote: str) -> dict:
    start = text.index(quote)
    return {
        "source": source,
        "start_offset": start,
        "end_offset": start + len(quote),
        "exact_quote": quote,
    }


def gold_prediction(case: LiveEvalCase) -> dict:
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


def run_mock_suite(*, manifest_path: Path, corpus_path: Path):
    manifest = load_manifest(manifest_path)
    corpus = load_corpus(corpus_path, manifest)
    responses = [
        gold_prediction(case)
        for _variant in manifest.variants
        for case in corpus.cases
    ]
    provider = MockLLMProvider(
        responses,
        provider_name="anthropic_compatible",
        model="glm-5.2",
    )
    approval = LiveEvaluationApproval(
        approved=True,
        endpoint="mock://no-network",
        model="glm-5.2",
        case_count=len(corpus.cases),
        max_total_tokens=10_000_000,
    )
    report = run_live_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=approval,
        provider=provider,
        enforce_live_readiness=False,
    )
    if len(provider.calls) != len(corpus.cases) * len(manifest.variants):
        raise AssertionError("mock runner did not execute every variant-case pair")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 300-row no-network Evidence v2 mock suite")
    root = Path(__file__).resolve().parents[4] / "data" / "eval" / "evidence_v2"
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", choices=["v4", "v5"], default="v4")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for short in ("dev", "held_out"):
        report = run_mock_suite(
            manifest_path=args.data_dir / f"manifest_{short}_{args.revision}.json",
            corpus_path=args.data_dir / f"cases_{short}_{args.revision}.json",
        )
        total_rows += len(report.rows)
        _atomic_json(
            args.output_dir / f"evidence-v2-{short}-mock-{args.revision}.json",
            report.model_dump(mode="json"),
        )
    if total_rows != 300:
        raise AssertionError(f"expected 300 mock rows, got {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
