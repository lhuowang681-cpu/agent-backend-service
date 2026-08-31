from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    EvidenceGroundingDataset,
    load_jsonl_dataset,
    sha256_file,
)
from job_agent.schemas import EvidenceLevel, StrictModel


class LabelChange(StrictModel):
    case_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    field: Literal["gold_level"]
    from_level: EvidenceLevel = Field(alias="from")
    to_level: EvidenceLevel = Field(alias="to")

    model_config = {
        **StrictModel.model_config,
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def validate_change(self):
        if self.from_level == self.to_level:
            raise ValueError("label overlay change must modify the level")
        return self


class ConfirmedUnchanged(StrictModel):
    case_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    level: EvidenceLevel


class HumanSpotCheckOverlay(StrictModel):
    schema_version: Literal["evidence-grounding-label-overlay-v1"]
    base_file: str
    base_sha256: str = Field(min_length=64, max_length=64)
    adjudication_file: str
    status: Literal["next_pilot_candidate_pending_full_human_review"]
    changes: list[LabelChange] = Field(min_length=1)
    confirmed_unchanged: list[ConfirmedUnchanged] = Field(default_factory=list)
    application_rule: str

    @model_validator(mode="after")
    def validate_unique_units(self):
        changed = [(item.case_id, item.requirement_id) for item in self.changes]
        unchanged = [
            (item.case_id, item.requirement_id) for item in self.confirmed_unchanged
        ]
        if len(changed) != len(set(changed)):
            raise ValueError("duplicate label change")
        if len(unchanged) != len(set(unchanged)):
            raise ValueError("duplicate confirmed-unchanged unit")
        if set(changed) & set(unchanged):
            raise ValueError("unit cannot be both changed and unchanged")
        return self


def load_human_spot_check_overlay(path: Path | str) -> HumanSpotCheckOverlay:
    return HumanSpotCheckOverlay.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def apply_human_spot_check_overlay(
    dataset: EvidenceGroundingDataset,
    overlay: HumanSpotCheckOverlay,
) -> EvidenceGroundingDataset:
    changes = {
        (item.case_id, item.requirement_id): item for item in overlay.changes
    }
    unchanged = {
        (item.case_id, item.requirement_id): item
        for item in overlay.confirmed_unchanged
    }
    seen_changes: set[tuple[str, str]] = set()
    seen_unchanged: set[tuple[str, str]] = set()
    cases: list[DatasetCase] = []
    for case in dataset.cases:
        units = []
        for unit in case.requirements:
            key = (case.case_id, unit.requirement_id)
            if key in changes:
                change = changes[key]
                if unit.gold_level != change.from_level:
                    raise ValueError(
                        f"overlay source level mismatch: {case.case_id}/{unit.requirement_id}"
                    )
                unit = unit.model_copy(update={"gold_level": change.to_level})
                seen_changes.add(key)
            if key in unchanged:
                confirmation = unchanged[key]
                if unit.gold_level != confirmation.level:
                    raise ValueError(
                        f"confirmed level mismatch: {case.case_id}/{unit.requirement_id}"
                    )
                seen_unchanged.add(key)
            units.append(unit)
        cases.append(case.model_copy(update={"requirements": units}))
    missing_changes = set(changes) - seen_changes
    missing_unchanged = set(unchanged) - seen_unchanged
    if missing_changes or missing_unchanged:
        raise ValueError("overlay references an unknown case/requirement")
    return EvidenceGroundingDataset(
        dataset_version="pilot-v1.1-human-spot-checked-candidate",
        cases=cases,
    )


def materialize_candidate(
    *,
    base_path: Path | str,
    overlay_path: Path | str,
) -> tuple[EvidenceGroundingDataset, str]:
    base = Path(base_path)
    overlay = load_human_spot_check_overlay(overlay_path)
    base_hash = sha256_file(base)
    if base_hash != overlay.base_sha256:
        raise ValueError("overlay base hash mismatch")
    return apply_human_spot_check_overlay(load_jsonl_dataset(base), overlay), base_hash


def render_dataset_jsonl(dataset: EvidenceGroundingDataset) -> str:
    return (
        "\n".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
            for case in dataset.cases
        )
        + "\n"
    )
