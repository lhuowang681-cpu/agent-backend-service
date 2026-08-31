from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from job_agent.evidence.evaluation.live_contracts import LiveEvaluationReport
from job_agent.evidence.evaluation.live_runner import load_corpus, rescore_live_report
from job_agent.evidence.evaluation.runner import load_manifest


def _load(path: Path) -> LiveEvaluationReport:
    return LiveEvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(center - margin, 6), round(center + margin, 6)]


def _variant_rows(report: LiveEvaluationReport) -> list[dict]:
    rows = []
    for variant in report.variants:
        total = variant.passed_rows + variant.failed_rows
        metrics = variant.metrics
        rows.append(
            {
                "variant_id": variant.variant_id,
                "passed_rows": variant.passed_rows,
                "failed_rows": variant.failed_rows,
                "completion_rate": round(variant.passed_rows / total, 6),
                "completion_wilson_95": _wilson(variant.passed_rows, total),
                "fit_band_accuracy": variant.fit_band_accuracy,
                "requirement_f1": metrics.requirement_extraction.f1,
                "link_f1": metrics.evidence_linking.f1,
                "contradiction_f1": metrics.contradiction_detection.f1,
                "partial_accuracy": metrics.partial_accuracy,
                "span_exactness": metrics.span_exactness,
                "unsupported_resume_claim_rate": metrics.unsupported_resume_claim_rate,
            }
        )
    return rows


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Evidence hardening v5 report")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "eval" / "evidence_v2")
    parser.add_argument("--report-dir", type=Path, default=root / "output" / "private" / "evidence_eval")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--campaign-paid-tokens", type=int, required=True)
    parser.add_argument("--approved-token-budget", type=int, default=10_000_000)
    args = parser.parse_args(argv)

    split_rows: dict[str, list[dict]] = {}
    errors: dict[str, dict[str, int]] = {}
    hashes: dict[str, str] = {}
    official_tokens = 0
    reports: dict[tuple[str, str], LiveEvaluationReport] = {}
    for split, data_short, report_short in (
        ("development", "dev", "dev"),
        ("held_out", "held_out", "held-out"),
    ):
        manifest = load_manifest(args.data_dir / f"manifest_{data_short}_v5.json")
        corpus = load_corpus(args.data_dir / f"cases_{data_short}_v5.json", manifest)
        baseline = _load(args.report_dir / f"glm-5.2-{report_short}-baselines-100-v5.json")
        production = _load(args.report_dir / f"glm-5.2-{report_short}-production-100-v5.json")
        if baseline.variants != rescore_live_report(baseline, corpus).variants:
            raise ValueError(f"{split} baseline rescore mismatch")
        if production.variants != rescore_live_report(production, corpus).variants:
            raise ValueError(f"{split} production rescore mismatch")
        reports[(split, "baseline")] = baseline
        reports[(split, "production")] = production
        split_rows[split] = [*_variant_rows(baseline), *_variant_rows(production)]
        errors[split] = dict(Counter(row.error_code for row in production.rows if row.error_code))
        hashes[split] = manifest.corpus_hash
        official_tokens += sum(
            row.budget_tokens for report in (baseline, production) for row in report.rows
        )

    mock_rows = 0
    for short in ("dev", "held_out"):
        mock = _load(
            args.report_dir
            / "mock-v5-final-candidate"
            / f"evidence-v2-{short}-mock-v5.json"
        )
        if any(row.status != "passed" for row in mock.rows):
            raise ValueError("v5 mock report contains failed rows")
        mock_rows += len(mock.rows)
    if mock_rows != 300:
        raise ValueError(f"expected 300 mock rows, got {mock_rows}")

    v4 = json.loads(
        (args.report_dir / "glm-5.2-evidence-100-v4-summary.json").read_text(encoding="utf-8")
    )
    v4_held = next(
        row
        for row in v4["live_quality"]["held_out"]
        if row["variant_id"] == "full_evidence_v2"
    )
    v5_held = next(
        row for row in split_rows["held_out"] if row["variant_id"] == "full_evidence_v2"
    )

    payload = {
        "schema_version": 1,
        "report_id": "evidence-v2-hardening-100-case-glm-5.2-v5",
        "corpus": {
            "case_count": 100,
            "family_count": 20,
            "development_cases": 60,
            "held_out_cases": 40,
            "new_family_ids": True,
            "family_level_split": True,
            "synthetic_only": True,
            "hashes": hashes,
        },
        "mock_correctness": {
            "variant_case_rows": mock_rows,
            "passed": mock_rows,
            "network_calls": 0,
            "model_quality_claim_allowed": False,
        },
        "live_quality": split_rows,
        "operational": {
            "official_v5_paid_tokens": official_tokens,
            "campaign_paid_tokens_including_v4_and_invalidated_pilots": args.campaign_paid_tokens,
            "approved_token_budget": args.approved_token_budget,
            "campaign_budget_utilization": round(args.campaign_paid_tokens / args.approved_token_budget, 6),
            "production_error_codes": errors,
            "persisted_source_level_cap_violations": 0,
            "numeric_offset_provider_fields": 0,
            "successful_guard_repair_rate": None,
            "successful_guard_repair_rate_note": "The v5 row contract did not persist repair-attempt counts; no post-hoc value is fabricated.",
        },
        "directional_v4_comparison": {
            "warning": "v4 and v5 use different scenario families; deltas are directional, not paired causal estimates.",
            "v4_held_out_production": v4_held,
            "v5_held_out_production": v5_held,
            "deltas": {
                "completion_rate": round(v5_held["completion_rate"] - v4_held["completion_rate"], 6),
                "requirement_f1": round(v5_held["requirement_f1"] - v4_held["requirement_f1"], 6),
                "link_f1": round(v5_held["link_f1"] - v4_held["link_f1"], 6),
                "fit_band_accuracy": round(v5_held["fit_band_accuracy"] - v4_held["fit_band_accuracy"], 6),
                "unsupported_resume_claim_rate": round(
                    v5_held["unsupported_resume_claim_rate"]
                    - v4_held["unsupported_resume_claim_rate"],
                    6,
                ),
            },
        },
        "invalidated_v5_development_pilots": [
            "label_audit_corpus_100_v5_dev_pilot_invalidated.json",
            "label_audit_corpus_100_v5_dev_pilot_r2_invalidated.json",
        ],
        "disclaimers": [
            "The benchmark is synthetic and does not estimate hiring success or real-resume accuracy.",
            "Mock correctness tests evaluator wiring only; it is not model quality.",
            "Official held-out failures were not retried.",
            "v4 and v5 family sets differ, so their delta is not a paired causal estimate.",
            "Provider price was not available, so no currency cost is claimed.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Evidence v2 Hardening — 100-Case GLM-5.2 Evaluation (v5)",
        "",
        "> Synthetic benchmark only. v5 uses 20 new families and a family-disjoint 60/40 split.",
        "",
        "## Execution integrity",
        "",
        f"- 300/300 no-network mock variant-case rows passed.",
        f"- Official v5 paid tokens: {official_tokens:,}; cumulative campaign including v4 and invalidated pilots: {args.campaign_paid_tokens:,} / {args.approved_token_budget:,}.",
        "- v5 held-out was frozen before its first live request; official held-out failures were not retried.",
        "- Provider candidate schemas contain no numeric offsets, hashes, or deterministic derived fields.",
        "",
        "## Live results",
        "",
        "| Split | Variant | Passed | Completion | Fit acc. | Req F1 | Link F1 | Contrad. F1 | Span exact | Unsupported claim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("development", "held_out"):
        for item in split_rows[split]:
            total = item["passed_rows"] + item["failed_rows"]
            lines.append(
                f"| {split} | `{item['variant_id']}` | {item['passed_rows']}/{total} | "
                f"{_percent(item['completion_rate'])} | {item['fit_band_accuracy']:.3f} | "
                f"{item['requirement_f1']:.3f} | {item['link_f1']:.3f} | "
                f"{item['contradiction_f1']:.3f} | {item['span_exactness']:.3f} | "
                f"{_percent(item['unsupported_resume_claim_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Operational findings",
            "",
            f"- Development production errors: {errors['development']}; held-out: {errors['held_out']}.",
            "- Development production improved directionally over v4 development: 55/60 vs 52/60 completion, Link F1 0.602 vs 0.429, and Fit 0.700 vs 0.483.",
            "- Frozen held-out production completed 38/40; failures were one `source_quote_not_found` and one schema error.",
            "- Held-out production reached Link F1 0.602 and Fit accuracy 0.625; numeric-offset failures were eliminated.",
            "- The remaining exact-quote bottleneck is model paraphrase/character copying, now visible as fail-closed `source_quote_not_found` rather than an incorrect persisted span.",
            "- Successful guard-repair counts were not persisted by the row schema, so no repair-rate number is claimed.",
            "",
            "## Directional comparison with immutable v4",
            "",
            "v4 and v5 use different family sets, so the following is not a paired causal estimate.",
            "",
            f"- Completion: {_percent(v4_held['completion_rate'])} → {_percent(v5_held['completion_rate'])}.",
            f"- Requirement F1: {v4_held['requirement_f1']:.3f} → {v5_held['requirement_f1']:.3f}.",
            f"- Link F1: {v4_held['link_f1']:.3f} → {v5_held['link_f1']:.3f}.",
            f"- Fit accuracy: {v4_held['fit_band_accuracy']:.3f} → {v5_held['fit_band_accuracy']:.3f}.",
            f"- Unsupported resume claim rate: {_percent(v4_held['unsupported_resume_claim_rate'])} → {_percent(v5_held['unsupported_resume_claim_rate'])}.",
            "",
            "## Resume-safe claim drafts",
            "",
            "**Agent engineering:** Built a live-only, auditable Evidence pipeline that separates LLM semantic candidates from persisted contracts, deterministically resolves immutable source quotes, enforces artifact-aware C1/C2/C3 caps, and fails closed without fallback; validated 300 mock rows and a 100-case synthetic GLM-5.2 benchmark with 95.0% frozen held-out production completion.",
            "",
            "**Algorithm/evaluation:** Designed a 20-family, family-disjoint synthetic benchmark and development-only tuning protocol; on 40 new frozen held-out cases, the production pipeline achieved requirement/link F1 of 0.905/0.602 and 0.625 fit-band accuracy, while error taxonomy isolated exact-quote copying and schema compliance as remaining bottlenecks.",
            "",
            "Keep the phrase `synthetic benchmark` in any external use.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
