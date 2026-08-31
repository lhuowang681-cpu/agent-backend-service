from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from job_agent.evidence.evaluation.live_contracts import LiveEvaluationReport
from job_agent.evidence.evaluation.live_runner import load_corpus, rescore_live_report, score_prediction
from job_agent.evidence.evaluation.metrics import evaluate_observations
from job_agent.evidence.evaluation.runner import load_manifest


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return round(center - margin, 6), round(center + margin, 6)


def _load_report(path: Path) -> LiveEvaluationReport:
    return LiveEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _variant_rows(report: LiveEvaluationReport) -> list[dict]:
    values = []
    for variant in report.variants:
        total = variant.passed_rows + variant.failed_rows
        low, high = _wilson(variant.passed_rows, total)
        values.append(
            {
                "variant_id": variant.variant_id,
                "passed_rows": variant.passed_rows,
                "failed_rows": variant.failed_rows,
                "completion_rate": round(variant.passed_rows / total, 6),
                "completion_wilson_95": [low, high],
                "fit_band_accuracy": variant.fit_band_accuracy,
                "requirement_f1": variant.metrics.requirement_extraction.f1,
                "link_f1": variant.metrics.evidence_linking.f1,
                "contradiction_f1": variant.metrics.contradiction_detection.f1,
                "partial_accuracy": variant.metrics.partial_accuracy,
                "span_exactness": variant.metrics.span_exactness,
                "unsupported_resume_claim_rate": variant.metrics.unsupported_resume_claim_rate,
            }
        )
    return values


def _family_metrics(report, corpus) -> list[dict]:
    cases = {case.case_id: case for case in corpus.cases}
    grouped: dict[str, list] = {}
    for row in report.rows:
        grouped.setdefault(cases[row.case_id].family_id or "unknown", []).append(row)
    output = []
    for family_id, rows in sorted(grouped.items()):
        observations = []
        fit_correct = 0
        for row in rows:
            observation, correct = score_prediction(cases[row.case_id], row.prediction)
            observations.append(observation)
            fit_correct += correct
        metrics = evaluate_observations(observations)
        output.append(
            {
                "family_id": family_id,
                "case_count": len(rows),
                "passed": sum(row.status == "passed" for row in rows),
                "failed": sum(row.status == "failed" for row in rows),
                "fit_band_accuracy": round(fit_correct / len(rows), 6),
                "requirement_f1": metrics.requirement_extraction.f1,
                "link_f1": metrics.evidence_linking.f1,
                "contradiction_f1": metrics.contradiction_detection.f1,
                "span_exactness": metrics.span_exactness,
                "unsupported_resume_claim_rate": metrics.unsupported_resume_claim_rate,
            }
        )
    return output


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Evidence v2 100-case evaluation summary")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "eval" / "evidence_v2")
    parser.add_argument("--report-dir", type=Path, default=root / "output" / "private" / "evidence_eval")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--campaign-paid-tokens", type=int, required=True)
    parser.add_argument("--approved-token-budget", type=int, default=10_000_000)
    args = parser.parse_args(argv)

    split_reports = {}
    family_rows = {}
    official_tokens = 0
    canonicalized = {}
    errors = {}
    corpus_hashes = {}
    for split, data_short, report_short in (
        ("development", "dev", "dev"),
        ("held_out", "held_out", "held-out"),
    ):
        manifest = load_manifest(args.data_dir / f"manifest_{data_short}_v4.json")
        corpus = load_corpus(args.data_dir / f"cases_{data_short}_v4.json", manifest)
        baseline = _load_report(args.report_dir / f"glm-5.2-{report_short}-baselines-100-v4.json")
        production = _load_report(args.report_dir / f"glm-5.2-{report_short}-production-100-v4.json")
        if baseline.variants != rescore_live_report(baseline, corpus).variants:
            raise ValueError(f"{split} baseline rescore mismatch")
        if production.variants != rescore_live_report(production, corpus).variants:
            raise ValueError(f"{split} production rescore mismatch")
        split_reports[split] = [*_variant_rows(baseline), *_variant_rows(production)]
        family_rows[split] = _family_metrics(production, corpus)
        official_tokens += (
            baseline.observed_input_tokens
            + baseline.observed_output_tokens
            + production.observed_input_tokens
            + production.observed_output_tokens
        )
        canonicalized[split] = {
            "rows": sum(
                "source_span_offsets_canonicalized" in row.normalization_warnings
                for row in production.rows
            ),
            "total": len(production.rows),
        }
        errors[split] = dict(
            Counter(row.error_code for row in production.rows if row.error_code)
        )
        corpus_hashes[split] = manifest.corpus_hash

    mock_rows = 0
    for short in ("dev", "held_out"):
        mock = _load_report(args.report_dir / "mock-v4" / f"evidence-v2-{short}-mock-v4.json")
        if any(row.status != "passed" for row in mock.rows):
            raise ValueError("mock report contains failed rows")
        mock_rows += len(mock.rows)
    if mock_rows != 300:
        raise ValueError("mock suite must contain exactly 300 variant-case rows")

    payload = {
        "schema_version": 1,
        "report_id": "evidence-v2-100-case-glm-5.2-v4",
        "corpus": {
            "case_count": 100,
            "family_count": 20,
            "development_cases": 60,
            "held_out_cases": 40,
            "family_level_split": True,
            "synthetic_only": True,
            "hashes": corpus_hashes,
        },
        "mock_correctness": {
            "variant_case_rows": mock_rows,
            "passed": mock_rows,
            "network_calls": 0,
            "model_quality_claim_allowed": False,
        },
        "live_quality": split_reports,
        "production_family_metrics": family_rows,
        "operational": {
            "official_v4_paid_tokens": official_tokens,
            "campaign_paid_tokens_including_invalidated_pilots": args.campaign_paid_tokens,
            "approved_token_budget": args.approved_token_budget,
            "campaign_budget_utilization": round(
                args.campaign_paid_tokens / args.approved_token_budget, 6
            ),
            "production_error_codes": errors,
            "offset_canonicalization": canonicalized,
        },
        "invalidated_development_revisions": ["v2", "v3"],
        "disclaimers": [
            "The corpus is synthetic and does not estimate hiring success or real-resume accuracy.",
            "Mock correctness tests evaluator wiring only; it is not model quality.",
            "Failed live rows remain in metric denominators and were not automatically retried.",
            "Provider price was not available, so no currency cost is claimed.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Evidence v2 — 100-Case GLM-5.2 Evaluation (v4)",
        "",
        "> Synthetic benchmark only. It measures this evaluator/pipeline, not hiring success or real-resume accuracy.",
        "",
        "## Corpus and execution integrity",
        "",
        "- 20 scenario families × 5 variants = 100 cases; family-level 60 development / 40 frozen held-out split.",
        f"- 300 no-network mock variant-case rows passed; this is contract correctness, not model quality.",
        f"- Official v4 live tokens: {official_tokens:,}; full campaign including invalidated development pilots: {args.campaign_paid_tokens:,} / {args.approved_token_budget:,}.",
        "- V2 and v3 development artifacts are retained as label-audit history and excluded from quality claims; neither held-out revision was called.",
        "",
        "## Live semantic and operational results",
        "",
        "| Split | Variant | Passed | Completion | Fit acc. | Req F1 | Link F1 | Contrad. F1 | Partial acc. | Span exact | Unsupported claim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "held_out"):
        for item in split_reports[split]:
            total = item["passed_rows"] + item["failed_rows"]
            lines.append(
                f"| {split} | `{item['variant_id']}` | {item['passed_rows']}/{total} | "
                f"{_percent(item['completion_rate'])} | {item['fit_band_accuracy']:.3f} | "
                f"{item['requirement_f1']:.3f} | {item['link_f1']:.3f} | "
                f"{item['contradiction_f1']:.3f} | {item['partial_accuracy']:.3f} | "
                f"{item['span_exactness']:.3f} | {_percent(item['unsupported_resume_claim_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Operational findings",
            "",
            f"- Development production errors: {errors['development']}; held-out: {errors['held_out']}.",
            f"- Offset canonicalization: {canonicalized['development']['rows']}/{canonicalized['development']['total']} development and {canonicalized['held_out']['rows']}/{canonicalized['held_out']['total']} held-out production rows.",
            "- Held-out production completion was 37/40 (92.5%); failed rows were two schema errors and one invalid evidence span.",
            "- Held-out atomization improved requirement F1 from 0.708 (direct) to 1.000, but single-link evidence F1 remained 0.368.",
            "- Full-v2 held-out fit accuracy was 0.525 and link F1 0.374; unsupported emitted resume claims fell from 100% in both baselines to 11.6%.",
            "- Full-v2 did not dominate every metric: held-out partial accuracy (0.383) was slightly below atomized-single-link (0.392), and completion was lower.",
            "",
            "## Resume-safe claim drafts",
            "",
            "**Agent engineering:** Built an auditable Evidence v2 pipeline with immutable source spans, fail-closed guards, atomic checkpoint/resume, and independent C1/C2/C3 source handling; validated 300 mock rows plus 300 GLM-5.2 live variant-case observations on a 100-case synthetic benchmark, with 92.5% held-out production completion and zero hidden fallback.",
            "",
            "**Algorithm/evaluation:** Designed a 20-family counterfactual evidence benchmark with family-level 60/40 split; on 40 frozen held-out cases, requirement atomization improved F1 from 0.708 to 1.000, while full-v2 reduced unsupported emitted resume claims from 100% to 11.6% and raised fit-band accuracy to 52.5%, exposing remaining span and structured-output failure modes.",
            "",
            "These drafts must retain the words `synthetic benchmark` when used externally.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
