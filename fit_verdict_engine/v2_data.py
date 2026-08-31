from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from fit_verdict_engine.data_generator import SyntheticRecord
from fit_verdict_engine.rule_engine import label_fit_input
from job_agent.schemas import EvidenceItem, EvidenceLevel, FitInput, JobRequirement, RoleType, Verdict


ROLE_TOPICS: dict[RoleType, tuple[str, ...]] = {
    RoleType.POSTTRAINING: (
        "supervised recipe reproducibility",
        "preference data curation",
        "safety refusal evaluation",
        "checkpoint ablation review",
        "adapter serving rollout",
        "evaluation governance",
        "training data lineage",
        "inference latency analysis",
        "reward model assessment",
        "release decision review",
        "model regression triage",
        "annotation quality audit",
        "post-training experiment tracking",
    ),
    RoleType.AGENTIC_RL: (
        "tool boundary design",
        "trace replay evaluation",
        "agent reward checks",
        "policy iteration analysis",
        "tool safety review",
        "agent benchmark construction",
        "simulator scenario design",
        "failure recovery routing",
        "multi-step planning analysis",
        "feedback signal diagnosis",
        "preference loop operations",
        "red-team trace review",
        "agent rollout governance",
    ),
    RoleType.UNKNOWN: (
        "typed data contract design",
        "integration test strategy",
        "operational runbook authoring",
        "incident analysis",
        "service observability",
        "privacy control review",
        "migration delivery",
        "on-call response",
        "forecasting analysis",
        "compliance review",
        "security control design",
        "release coordination",
        "cross-team delivery planning",
    ),
}


def _safe_label(label: Verdict) -> str:
    return label.value.replace(" ", "_")


def _requirements(role: RoleType, variant: int, count: int) -> list[JobRequirement]:
    topics = ROLE_TOPICS[role]
    return [
        JobRequirement(
            id=f"{role.value}_{variant}_{index}",
            text=f"Demonstrate {topics[(variant + index) % len(topics)]}",
            required=True,
            probe=f"Explain evidence for {topics[(variant + index) % len(topics)]}",
        )
        for index in range(count)
    ]


def _evidence(requirement: JobRequirement, level: EvidenceLevel, scenario_id: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"evidence_{scenario_id}_{requirement.id}",
        requirement_id=requirement.id,
        claim=f"Completed a scoped {requirement.text.lower()} artifact",
        level=level,
        proof="Versioned implementation, evaluation output, or review record",
        risk="Evidence is limited to the documented scope",
    )


def _record(prefix: str, label: Verdict, role: RoleType, variant: int) -> SyntheticRecord:
    sparse_risky = label == Verdict.RISKY and variant % 2 == 0
    count = 4 if sparse_risky else 3
    requirements = _requirements(role, variant, count)
    scenario_id = f"{prefix}_{role.value}_{_safe_label(label)}_{variant:02d}"

    if label == Verdict.STRONG:
        evidence = [
            _evidence(requirements[0], EvidenceLevel.C2, scenario_id),
            _evidence(requirements[1], EvidenceLevel.C1, scenario_id),
            _evidence(requirements[2], EvidenceLevel.C3, scenario_id),
        ]
        toy_signals: list[str] = []
    elif label == Verdict.WEAK:
        evidence = [
            _evidence(requirements[0], EvidenceLevel.C1, scenario_id),
            _evidence(requirements[1], EvidenceLevel.C2, scenario_id),
        ]
        toy_signals = []
    elif sparse_risky:
        evidence = [_evidence(requirements[0], EvidenceLevel.C1, scenario_id)]
        toy_signals = []
    elif label == Verdict.RISKY:
        evidence = [_evidence(requirement, EvidenceLevel.C1, scenario_id) for requirement in requirements]
        toy_signals = ["claims high-stakes ownership without reproducible boundary evidence"]
    elif variant % 2 == 0:
        evidence = []
        toy_signals = []
    else:
        evidence = [
            _evidence(requirements[0], EvidenceLevel.C0, scenario_id),
            _evidence(requirements[1], EvidenceLevel.C0, scenario_id),
        ]
        toy_signals = []

    return SyntheticRecord(
        record_id=f"{prefix}_{_safe_label(label)}_{role.value}_{variant:02d}",
        fit_input=FitInput(role_type=role, requirements=requirements, evidence=evidence, toy_signals=toy_signals),
        label=label,
        scenario_id=scenario_id,
    )


def _records(prefix: str, variants_per_role: int, variant_offset: int) -> list[SyntheticRecord]:
    records: list[SyntheticRecord] = []
    for label in Verdict:
        for role in RoleType:
            for index in range(variants_per_role):
                records.append(_record(prefix, label, role, variant_offset + index))
    return records


def _validate_records(records: list[SyntheticRecord], expected_per_role: int) -> list[SyntheticRecord]:
    if len({record.scenario_id for record in records}) != len(records):
        raise ValueError("V2 records must have unique scenario IDs")
    if any(label_fit_input(record.fit_input) != record.label for record in records):
        raise ValueError("V2 records must agree with the production rule label")
    counts = Counter((record.label, record.fit_input.role_type) for record in records)
    if any(counts[(label, role)] != expected_per_role for label in Verdict for role in RoleType):
        raise ValueError("V2 records must be balanced by verdict and role")
    return records


def generate_v2_training_records() -> list[SyntheticRecord]:
    return _validate_records(_records(prefix="train_v2", variants_per_role=8, variant_offset=0), expected_per_role=8)


def generate_v2_eval_records(split: str) -> list[SyntheticRecord]:
    if split == "dev":
        return _validate_records(_records(prefix="dev_v2", variants_per_role=3, variant_offset=8), expected_per_role=3)
    if split == "heldout":
        return _validate_records(_records(prefix="heldout_v2", variants_per_role=2, variant_offset=11), expected_per_role=2)
    raise ValueError("split must be 'dev' or 'heldout'")


def eval_rows(records: Iterable[SyntheticRecord]) -> list[dict]:
    return [
        {
            "record_id": record.record_id,
            "gold": record.label.value,
            "scenario_id": record.scenario_id,
            "fit_input": record.fit_input.model_dump(mode="json"),
        }
        for record in records
    ]
