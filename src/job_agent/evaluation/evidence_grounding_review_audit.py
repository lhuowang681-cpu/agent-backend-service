from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from job_agent.evaluation.evidence_grounding_contracts import (
    load_jsonl_dataset,
    sha256_file,
)
from job_agent.schemas import EvidenceLevel, StrictModel


_LEVEL_ORDER = [
    EvidenceLevel.NONE,
    EvidenceLevel.C0,
    EvidenceLevel.C1,
    EvidenceLevel.C2,
    EvidenceLevel.C3,
]
_STRONG_LEVELS = {EvidenceLevel.C2, EvidenceLevel.C3}


class AISecondReviewRecord(StrictModel):
    case_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    original_level: EvidenceLevel
    reviewed_level: EvidenceLevel
    decision: Literal["keep", "change"]
    confidence: Literal["low", "medium", "high"]
    quote_verified: bool
    issue_codes: list[str]
    review_note: str = Field(min_length=1)
    reviewer_provenance: Literal["ai_second_pass"]
    review_status: Literal["pending_human_spot_check"]
    guide_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self):
        changed = self.original_level != self.reviewed_level
        if changed != (self.decision == "change"):
            raise ValueError("AI review decision does not match level change")
        return self


class ReviewDisagreement(StrictModel):
    case_id: str
    requirement_id: str
    ai_level: EvidenceLevel
    human_level: EvidenceLevel
    crosses_strong_boundary: bool
    human_reason: str


class WilsonInterval(StrictModel):
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    confidence: Literal[0.95] = 0.95


class EvidenceGroundingReviewAudit(StrictModel):
    schema_version: Literal[
        "evidence-grounding-review-audit-v1"
    ] = "evidence-grounding-review-audit-v1"
    result_status: Literal["NOT_PROVEN"] = "NOT_PROVEN"
    evidence_grade: Literal[
        "ai_reviewed_with_partial_human_spot_check"
    ] = "ai_reviewed_with_partial_human_spot_check"
    sampling_design: Literal[
        "representative_spot_check_not_proven_random"
    ] = "representative_spot_check_not_proven_random"
    base_file: str
    base_sha256: str
    ai_review_ledger_file: str
    ai_review_ledger_sha256: str
    human_adjudication_file: str
    human_adjudication_sha256: str
    ai_reviewed_units: int = Field(ge=0)
    human_checked_units: int = Field(ge=0)
    human_unchecked_units: int = Field(ge=0)
    exact_level_agreements: int = Field(ge=0)
    exact_level_disagreements: int = Field(ge=0)
    exact_level_agreement_rate: float = Field(ge=0, le=1)
    exact_level_agreement_wilson_95: WilsonInterval
    strong_boundary_agreements: int = Field(ge=0)
    strong_boundary_agreement_rate: float = Field(ge=0, le=1)
    quadratic_weighted_kappa: float
    ai_confidence_counts: dict[str, int]
    disagreements: list[ReviewDisagreement]
    allowed_claims: list[str]
    forbidden_claims: list[str]
    next_gate: str


def _load_ai_review_ledger(path: Path) -> list[AISecondReviewRecord]:
    records = [
        AISecondReviewRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = [(item.case_id, item.requirement_id) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate unit in AI second-review ledger")
    return records


def _wilson_interval(successes: int, total: int) -> WilsonInterval:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + (z * z / total)
    centre = proportion + (z * z / (2 * total))
    margin = z * math.sqrt(
        (proportion * (1 - proportion) / total)
        + (z * z / (4 * total * total))
    )
    return WilsonInterval(
        lower=max(0.0, (centre - margin) / denominator),
        upper=min(1.0, (centre + margin) / denominator),
    )


def _quadratic_weighted_kappa(
    first: list[EvidenceLevel],
    second: list[EvidenceLevel],
) -> float:
    if not first or len(first) != len(second):
        raise ValueError("weighted kappa requires paired non-empty labels")
    size = len(_LEVEL_ORDER)
    index = {level: position for position, level in enumerate(_LEVEL_ORDER)}
    observed = [[0 for _ in range(size)] for _ in range(size)]
    first_count = [0 for _ in range(size)]
    second_count = [0 for _ in range(size)]
    for first_level, second_level in zip(first, second):
        row = index[first_level]
        column = index[second_level]
        observed[row][column] += 1
        first_count[row] += 1
        second_count[column] += 1
    total = len(first)
    denominator = (size - 1) ** 2
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for row in range(size):
        for column in range(size):
            weight = ((row - column) ** 2) / denominator
            observed_disagreement += weight * observed[row][column] / total
            expected_disagreement += (
                weight
                * first_count[row]
                * second_count[column]
                / (total * total)
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - (observed_disagreement / expected_disagreement)


def build_evidence_grounding_review_audit(
    *,
    base_path: Path | str,
    ai_review_ledger_path: Path | str,
    human_adjudication_path: Path | str,
    manifest_path: Path | str,
) -> EvidenceGroundingReviewAudit:
    base = Path(base_path)
    ledger = Path(ai_review_ledger_path)
    adjudication_file = Path(human_adjudication_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    adjudication = json.loads(adjudication_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(adjudication, dict):
        raise ValueError("review audit inputs must be JSON objects")

    base_hash = sha256_file(base)
    ledger_hash = sha256_file(ledger)
    adjudication_hash = sha256_file(adjudication_file)
    if manifest.get("pilot_sha256") != base_hash:
        raise ValueError("manifest pilot hash mismatch")
    ai_manifest = manifest.get("ai_second_review", {})
    human_manifest = manifest.get("human_spot_check", {})
    if ai_manifest.get("ledger_sha256") != ledger_hash:
        raise ValueError("manifest AI review ledger hash mismatch")
    if human_manifest.get("adjudication_sha256") != adjudication_hash:
        raise ValueError("manifest human adjudication hash mismatch")
    if adjudication.get("base_sha256") != base_hash:
        raise ValueError("human adjudication base hash mismatch")

    dataset = load_jsonl_dataset(base)
    expected = {
        (case.case_id, unit.requirement_id): unit.gold_level
        for case in dataset.cases
        for unit in case.requirements
    }
    records = _load_ai_review_ledger(ledger)
    by_key = {
        (item.case_id, item.requirement_id): item for item in records
    }
    if set(by_key) != set(expected):
        raise ValueError("AI second-review ledger does not cover the base dataset")
    for key, record in by_key.items():
        if record.original_level != expected[key]:
            raise ValueError(
                "AI second-review original level mismatch: "
                f"{key[0]}/{key[1]}"
            )
        if not record.quote_verified:
            raise ValueError(
                "AI second-review contains an unverified quote: "
                f"{key[0]}/{key[1]}"
            )

    decisions = adjudication.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("human adjudication decisions must be a list")
    human_keys: list[tuple[str, str]] = []
    ai_levels: list[EvidenceLevel] = []
    human_levels: list[EvidenceLevel] = []
    disagreements: list[ReviewDisagreement] = []
    strong_agreements = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("human adjudication decision must be an object")
        key = (decision.get("case_id"), decision.get("requirement_id"))
        if not all(isinstance(part, str) and part for part in key):
            raise ValueError("human adjudication unit key is invalid")
        if key not in by_key:
            raise ValueError("human adjudication references an unknown AI review unit")
        human_keys.append(key)
        ai_level = by_key[key].reviewed_level
        traced_ai_level = EvidenceLevel(decision.get("ai_second_review_level"))
        if traced_ai_level != ai_level:
            raise ValueError(
                "human adjudication AI level trace mismatch: "
                f"{key[0]}/{key[1]}"
            )
        human_level = EvidenceLevel(decision.get("human_level"))
        ai_levels.append(ai_level)
        human_levels.append(human_level)
        ai_strong = ai_level in _STRONG_LEVELS
        human_strong = human_level in _STRONG_LEVELS
        strong_agreements += ai_strong == human_strong
        if ai_level != human_level:
            disagreements.append(
                ReviewDisagreement(
                    case_id=key[0],
                    requirement_id=key[1],
                    ai_level=ai_level,
                    human_level=human_level,
                    crosses_strong_boundary=ai_strong != human_strong,
                    human_reason=str(decision.get("reason", "")),
                )
            )
    if len(human_keys) != len(set(human_keys)):
        raise ValueError("duplicate unit in human adjudication")
    checked = len(human_keys)
    if adjudication.get("checked_units") != checked:
        raise ValueError("human adjudication checked-unit count mismatch")
    if adjudication.get("unchecked_units") != len(expected) - checked:
        raise ValueError("human adjudication unchecked-unit count mismatch")

    agreements = checked - len(disagreements)
    return EvidenceGroundingReviewAudit(
        base_file=base.name,
        base_sha256=base_hash,
        ai_review_ledger_file=ledger.name,
        ai_review_ledger_sha256=ledger_hash,
        human_adjudication_file=adjudication_file.name,
        human_adjudication_sha256=adjudication_hash,
        ai_reviewed_units=len(records),
        human_checked_units=checked,
        human_unchecked_units=len(expected) - checked,
        exact_level_agreements=agreements,
        exact_level_disagreements=len(disagreements),
        exact_level_agreement_rate=agreements / checked,
        exact_level_agreement_wilson_95=_wilson_interval(agreements, checked),
        strong_boundary_agreements=strong_agreements,
        strong_boundary_agreement_rate=strong_agreements / checked,
        quadratic_weighted_kappa=_quadratic_weighted_kappa(
            ai_levels,
            human_levels,
        ),
        ai_confidence_counts=dict(
            sorted(Counter(item.confidence for item in records).items())
        ),
        disagreements=disagreements,
        allowed_claims=[
            "30/30 units received an AI second-pass review",
            "6/30 units received a human spot-check",
            "AI-human agreement is an exploratory partial-audit diagnostic",
        ],
        forbidden_claims=[
            "human_gold_benchmark",
            "inter_rater_agreement",
            "frozen_heldout_result",
            "production_quality_improvement",
        ],
        next_gate=(
            "Use stratified random human spot-checking plus all changed, "
            "medium/low-confidence, and unsafe-boundary units; do not promote "
            "to human gold."
        ),
    )


def render_evidence_grounding_review_audit_markdown(
    audit: EvidenceGroundingReviewAudit,
) -> str:
    interval = audit.exact_level_agreement_wilson_95
    lines = [
        "# Evidence Mapping AI Review + Human Spot-check Audit",
        "",
        f"- Result status: `{audit.result_status}`",
        f"- Evidence grade: `{audit.evidence_grade}`",
        f"- Base SHA-256: `{audit.base_sha256}`",
        f"- AI reviewed: `{audit.ai_reviewed_units}/"
        f"{audit.ai_reviewed_units}`",
        f"- Human spot-check: `{audit.human_checked_units}/"
        f"{audit.ai_reviewed_units}`",
        f"- Exact level agreement: `{audit.exact_level_agreements}/"
        f"{audit.human_checked_units}` "
        f"(`{audit.exact_level_agreement_rate:.3f}`)",
        "- Exact agreement Wilson 95% CI: "
        f"`[{interval.lower:.3f}, {interval.upper:.3f}]`",
        f"- Sampling design: `{audit.sampling_design}`",
        "- Strong/non-strong boundary agreement: "
        f"`{audit.strong_boundary_agreements}/"
        f"{audit.human_checked_units}` "
        f"(`{audit.strong_boundary_agreement_rate:.3f}`)",
        f"- Quadratic weighted kappa: "
        f"`{audit.quadratic_weighted_kappa:.3f}`",
        "- New API calls: `0`",
        "",
        "## Disagreements",
        "",
    ]
    if not audit.disagreements:
        lines.append("- None.")
    for item in audit.disagreements:
        lines.append(
            f"- `{item.case_id}/{item.requirement_id}`: "
            f"AI `{item.ai_level.value}` → human `{item.human_level.value}`; "
            f"crosses strong boundary = `{str(item.crosses_strong_boundary).lower()}`."
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "这证明 30/30 units 已经过第二遍 AI rubric review，并有 6/30 人工抽检；"
            "它不构成 human gold、inter-rater agreement 或 frozen heldout。",
            "现有 6 条没有可证明的随机抽样设计，因此 Wilson 区间只作描述性敏感度"
            "展示，不能当总体置信区间；加上样本很小，正式质量结论保持 "
            "`NOT_PROVEN`。",
            "",
            "## Scale-up protocol",
            "",
            "扩到 100–数百 units 时：",
            "",
            "1. AI 按冻结 rubric 全量复核；",
            "2. deterministic validator 检查 level、span、offset、hash 与 unit 覆盖；",
            "3. 人工必查所有 change、medium/low confidence 和 unsafe-boundary units；",
            "4. 再从其余 high-confidence units 按 slice/source group 分层随机抽样；",
            "5. 随机抽样至少 60 条且无 critical disagreement，才能用 rule-of-three "
            "把 95% 错误率上界压到约 5%；",
            "6. 有分歧时记录 Wilson 上界、修订 rubric，并在独立新样本上重新审计；",
            "7. 这种产物仍叫 AI-reviewed dataset with human audit，不叫 human gold。",
            "",
        ]
    )
    return "\n".join(lines)
