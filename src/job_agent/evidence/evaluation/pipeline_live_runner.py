from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from job_agent.evidence.evaluation.contracts import LiveEvaluationApproval
from job_agent.evidence.evaluation.live_contracts import (
    LiveEvalCorpus,
    LiveEvalRow,
    LiveEvaluationReport,
    LiveVariantPrediction,
    PredictedEvidenceLink,
    PredictedRequirement,
    PredictedResumeClaim,
    PredictedSpan,
)
from job_agent.evidence.evaluation.live_runner import (
    _atomic_json,
    _report_from_rows,
    _resume_rows,
    default_corpus_path,
    load_corpus,
)
from job_agent.evidence.evaluation.runner import default_manifest_path, load_manifest, require_live_approval
from job_agent.evidence.contracts import SourceInput
from job_agent.evidence.pipeline import EvidenceV2Pipeline
from job_agent.llm.harness import LLMHarness
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.skill_registry import SkillRegistry


def _prediction_from_pipeline(result) -> LiveVariantPrediction:
    bundle = result.bundle
    source_keys = {
        item.source_id: (
            "jd"
            if item.kind == "raw_jd"
            else "resume"
            if item.kind == "original_resume"
            else item.display_label.split("::", 1)[0]
        )
        for item in bundle.sources.documents
    }
    requirements = [
        PredictedRequirement(
            atom_key=atom.atom_id,
            text=atom.text,
            importance=next(
                parent.importance
                for parent in bundle.requirements.parents
                if parent.requirement_id == atom.parent_requirement_id
            ),
            hard_gate=next(
                parent.hard_gate
                for parent in bundle.requirements.parents
                if parent.requirement_id == atom.parent_requirement_id
            ),
            source_span=PredictedSpan(
                source="jd",
                start_offset=atom.source_span.start_offset,
                end_offset=atom.source_span.end_offset,
                exact_quote=atom.source_span.exact_quote,
            ),
        )
        for atom in bundle.requirements.atoms
    ]
    links = [
        PredictedEvidenceLink(
            link_key=link.link_id,
            atom_key=item.atom_id,
            source_span=PredictedSpan(
                source=source_keys[link.source_span.source_id],
                start_offset=link.source_span.start_offset,
                end_offset=link.source_span.end_offset,
                exact_quote=link.source_span.exact_quote,
            ),
            support_status=link.support_status,
            level=link.level,
        )
        for item in bundle.mapping.atom_results
        for link in item.links
    ]
    links_by_id = {item.link_key: item for item in links}
    claims = []
    for claim in result.resume_view.claims:
        if not claim.link_ids:
            continue
        link = links_by_id[claim.link_ids[0]]
        claims.append(
            PredictedResumeClaim(
                atom_key=claim.atom_id,
                claim=claim.claim,
                evidence_span=link.source_span,
            )
        )
    return LiveVariantPrediction(
        requirements=requirements,
        evidence_links=links,
        estimated_fit_band=bundle.assessment.estimated_fit_band,
        resume_claims=claims,
    )


def run_full_pipeline_evaluation(
    *,
    manifest,
    corpus: LiveEvalCorpus,
    approval: LiveEvaluationApproval,
    provider,
    registry,
    checkpoint_path: Path | None = None,
    resume_report: LiveEvaluationReport | None = None,
    retry_failed: bool = False,
) -> LiveEvaluationReport:
    require_live_approval(approval)
    if approval.case_count > len(corpus.cases):
        raise ValueError("approved case count exceeds frozen corpus")
    variant = next(item for item in manifest.variants if item.variant_id == "full_evidence_v2")
    if (variant.provider, variant.model) != (
        str(getattr(provider, "provider_name", "unknown")),
        str(getattr(provider, "model", "unknown")),
    ):
        raise ValueError("provider identity does not match frozen full-v2 manifest")
    cases = corpus.cases[: approval.case_count]
    allowed_keys = {("full_evidence_v2", case.case_id) for case in cases}
    rows = _resume_rows(
        report=resume_report,
        manifest=manifest,
        corpus=corpus,
        provider=provider,
        allowed_keys=allowed_keys,
        retry_failed=retry_failed,
    )
    completed = {(item.variant_id, item.case_id) for item in rows}
    budget_used = sum(
        item.budget_tokens
        or (item.input_tokens or 0) + (item.output_tokens or 0)
        for item in rows
    )

    def checkpoint() -> None:
        if checkpoint_path is None:
            return
        report = _report_from_rows(
            manifest=manifest,
            corpus=corpus,
            approval=approval,
            provider=provider,
            rows=rows,
        )
        _atomic_json(checkpoint_path, report.model_dump(mode="json"))

    for case in cases:
        key = ("full_evidence_v2", case.case_id)
        if key in completed:
            continue
        # Production v2 permits one semantic repair in each of its two stages.
        source_bytes = case.jd_text + case.resume_text + "".join(
            item.text for item in case.evidence_sources
        )
        reserved = 2 * 4096 + 2 * 6144 + len(source_bytes.encode("utf-8"))
        if budget_used + reserved > approval.max_total_tokens:
            row = LiveEvalRow(
                case_id=case.case_id,
                variant_id="full_evidence_v2",
                status="failed",
                error_code="token_budget_exhausted",
                provider=str(getattr(provider, "provider_name", "unknown")),
                model=str(getattr(provider, "model", "unknown")),
                latency_ms=0,
            )
        else:
            harness = LLMHarness(provider)
            try:
                with tempfile.TemporaryDirectory(prefix="evidence-v2-live-eval-") as temporary:
                    result = EvidenceV2Pipeline(registry=registry, harness=harness).run(
                        session_dir=Path(temporary),
                        raw_jd=case.jd_text,
                        original_resume=case.resume_text,
                        session_id=f"evidence-v2-eval:{case.case_id}",
                        run_id=f"live-{case.case_id}",
                        supporting_sources=tuple(
                            SourceInput(
                                kind=source.kind,
                                display_label=f"{source.source_key}::{source.display_label}",
                                text=source.text,
                                artifact_type=source.artifact_type,
                            )
                            for source in case.evidence_sources
                        ),
                    )
                prediction = _prediction_from_pipeline(result)
                row = LiveEvalRow(
                    case_id=case.case_id,
                    variant_id="full_evidence_v2",
                    status="passed",
                    prediction=prediction,
                    provider=str(getattr(provider, "provider_name", "unknown")),
                    model=str(getattr(provider, "model", "unknown")),
                    latency_ms=sum(item.latency_ms for item in harness.traces),
                    input_tokens=sum(item.input_tokens or 0 for item in harness.traces),
                    output_tokens=sum(item.output_tokens or 0 for item in harness.traces),
                    budget_tokens=(
                        sum((item.input_tokens or 0) + (item.output_tokens or 0) for item in harness.traces)
                        or reserved
                    ),
                    normalization_warnings=sorted(
                        set(result.bundle.requirements.warnings)
                        | set(result.bundle.mapping.warnings)
                    ),
                )
            except Exception as exc:
                prediction = None
                row = LiveEvalRow(
                    case_id=case.case_id,
                    variant_id="full_evidence_v2",
                    status="failed",
                    error_code=str(getattr(exc, "error_code", type(exc).__name__)),
                    provider=str(getattr(provider, "provider_name", "unknown")),
                    model=str(getattr(provider, "model", "unknown")),
                    latency_ms=sum(item.latency_ms for item in harness.traces),
                    input_tokens=sum(item.input_tokens or 0 for item in harness.traces),
                    output_tokens=sum(item.output_tokens or 0 for item in harness.traces),
                    budget_tokens=(
                        sum((item.input_tokens or 0) + (item.output_tokens or 0) for item in harness.traces)
                        or reserved
                    ),
                )
            budget_used += row.budget_tokens
        rows.append(row)
        completed.add(key)
        checkpoint()
    return _report_from_rows(
        manifest=manifest,
        corpus=corpus,
        approval=approval,
        provider=provider,
        rows=rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the production Evidence v2 pipeline evaluation")
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--corpus", type=Path, default=default_corpus_path())
    parser.add_argument("--endpoint", default="https://open.bigmodel.cn/api/anthropic")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--case-count", type=int)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--skill-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    corpus = load_corpus(args.corpus, manifest)
    case_count = args.case_count if args.case_count is not None else len(corpus.cases)
    approval = LiveEvaluationApproval(
        approved=True,
        endpoint=args.endpoint,
        model=args.model,
        case_count=case_count,
        max_total_tokens=args.max_total_tokens,
    )
    require_live_approval(approval)
    provider = AnthropicCompatibleProvider(
        base_url=args.endpoint,
        model=args.model,
        api_key=os.environ["JOB_AGENT_LIVE_API_KEY"],
        timeout_s=120.0,
    )
    registry = SkillRegistry.from_sources(skill_root=args.skill_root)
    resume_report = None
    if args.resume_report and args.resume_report.exists():
        resume_report = LiveEvaluationReport.model_validate_json(
            args.resume_report.read_text(encoding="utf-8")
        )
    report = run_full_pipeline_evaluation(
        manifest=manifest,
        corpus=corpus,
        approval=approval,
        provider=provider,
        registry=registry,
        checkpoint_path=args.resume_report or args.report,
        resume_report=resume_report,
        retry_failed=args.retry_failed,
    )
    _atomic_json(args.report, report.model_dump(mode="json"))
    return 0 if report.variants[0].failed_rows == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
