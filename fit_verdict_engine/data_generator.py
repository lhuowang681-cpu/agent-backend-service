from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from job_agent.schemas import EvidenceItem, EvidenceLevel, FitInput, JobRequirement, RoleType, Verdict


@dataclass(frozen=True)
class SyntheticRecord:
    record_id: str
    fit_input: FitInput
    label: Verdict
    scenario_id: str


def _requirements(count: int = 3) -> list[JobRequirement]:
    pool = [
        JobRequirement(id="req_sft_lora", text="SFT / LoRA training workflow", required=True, probe="training config"),
        JobRequirement(id="req_alignment", text="RLHF / DPO / GRPO alignment basics", required=True, probe="preference data"),
        JobRequirement(id="req_eval", text="LLM eval and ablation experience", required=True, probe="eval metrics"),
        JobRequirement(id="req_data", text="preference data cleaning and labeling", required=True, probe="data pipeline"),
        JobRequirement(id="req_serving", text="adapter deployment and inference guardrails", required=True, probe="serving"),
    ]
    return pool[:count]


def _evidence(requirement_id: str, level: EvidenceLevel, scenario_id: str = "baseline") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev_{scenario_id}_{requirement_id}_{level.value}",
        requirement_id=requirement_id,
        claim=f"{requirement_id} evidence for {scenario_id}",
        level=level,
        proof="synthetic proof",
        risk="synthetic risk",
    )


def _record(record_id: str, label: Verdict, variant_index: int = 0) -> SyntheticRecord:
    if label == Verdict.STRONG:
        scenarios = [
            (3, [(0, EvidenceLevel.C2), (1, EvidenceLevel.C1), (2, EvidenceLevel.C3)], []),
            (4, [(0, EvidenceLevel.C1), (1, EvidenceLevel.C2), (2, EvidenceLevel.C1)], []),
            (5, [(0, EvidenceLevel.C2), (1, EvidenceLevel.C2), (2, EvidenceLevel.C1), (3, EvidenceLevel.C1)], []),
            (3, [(0, EvidenceLevel.C3), (1, EvidenceLevel.C2), (2, EvidenceLevel.C1)], []),
        ]
    elif label == Verdict.WEAK:
        scenarios = [
            (3, [(0, EvidenceLevel.C1), (1, EvidenceLevel.C1)], []),
            (4, [(0, EvidenceLevel.C2), (2, EvidenceLevel.C1)], []),
            (5, [(0, EvidenceLevel.C1), (3, EvidenceLevel.C1)], []),
            (2, [(0, EvidenceLevel.C2)], []),
        ]
    elif label == Verdict.RISKY:
        scenarios = [
            (3, [(0, EvidenceLevel.C1)], []),
            (4, [(0, EvidenceLevel.C2)], []),
            (3, [(0, EvidenceLevel.C2), (1, EvidenceLevel.C2), (2, EvidenceLevel.C2)], ["toy_alignment_claim"]),
            (5, [(0, EvidenceLevel.C1)], ["says RLHF but no preference data or reward loop"]),
        ]
    else:
        scenarios = [
            (3, [], []),
            (4, [], []),
            (3, [(0, EvidenceLevel.C0), (1, EvidenceLevel.C0)], []),
            (5, [(4, EvidenceLevel.C0)], []),
        ]

    scenario_index = variant_index % len(scenarios)
    requirement_count, evidence_specs, toy_signals = scenarios[scenario_index]
    scenario_id = f"{label.value.replace(' ', '_')}_scenario_{scenario_index}"
    requirements = _requirements(requirement_count)
    evidence = [_evidence(requirements[index].id, level, scenario_id) for index, level in evidence_specs]

    fit_input = FitInput(
        role_type=RoleType.POSTTRAINING,
        requirements=requirements,
        evidence=evidence,
        toy_signals=toy_signals,
    )
    return SyntheticRecord(record_id=record_id, fit_input=fit_input, label=label, scenario_id=scenario_id)


def generate_synthetic_records(per_class: int) -> list[SyntheticRecord]:
    if per_class < 1:
        raise ValueError("per_class must be at least 1")

    labels = [Verdict.STRONG, Verdict.WEAK, Verdict.RISKY, Verdict.NOT_RECOMMENDED]
    records: list[SyntheticRecord] = []
    for label in labels:
        safe_label = label.value.replace(" ", "_")
        for index in range(per_class):
            records.append(_record(f"{safe_label}_{index:03d}", label, variant_index=index))
    return records


def load_eval_records(path: Path) -> list[SyntheticRecord]:
    """Load hand-authored Fit Verdict evaluation records from a JSONL corpus."""
    records: list[SyntheticRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("row must be a JSON object")
            record_id = row["record_id"]
            scenario_id = row["scenario_id"]
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("record_id must be a non-empty string")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError("scenario_id must be a non-empty string")
            records.append(
                SyntheticRecord(
                    record_id=record_id,
                    fit_input=FitInput.model_validate(row["fit_input"]),
                    label=Verdict(row["gold"]),
                    scenario_id=scenario_id,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid eval record at {path}:{line_number}: {exc}") from exc

    if not records:
        raise ValueError(f"Eval input contains no records: {path}")
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
