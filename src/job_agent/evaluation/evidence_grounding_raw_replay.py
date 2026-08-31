from __future__ import annotations

import json
from pathlib import Path

from job_agent.atomic_io import atomic_write_json
from job_agent.evaluation.evidence_grounding_atomic import (
    validate_atomic_evidence_graph,
)
from job_agent.evaluation.evidence_grounding_atomic_runner import (
    AtomicEvidenceGraphDraft,
    AtomicEvidenceGroundingRunner,
    AtomicValidatedArtifact,
    canonicalize_atomic_graph_draft,
)
from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    ExperimentVariant,
    PredictionRecord,
    stable_protocol_hash,
)
from job_agent.evaluation.evidence_grounding_runner import (
    BlockMappingOutput,
    EvidenceGroundingRunner,
)


BLOCK_REPLAY_VERSION = "c-block-raw-replay-v1.1"
ATOMIC_REPLAY_VERSION = "c-atomic-draft-replay-v1.1"


def _raw_wrapper(
    raw_root: Path,
    *,
    case_id: str,
    variant: ExperimentVariant,
) -> dict:
    if variant == ExperimentVariant.C_BLOCK:
        candidates = [
            raw_root / f"{case_id}_{variant.value}_repair.json",
            raw_root / f"{case_id}_{variant.value}.json",
        ]
    else:
        candidates = [
            raw_root / f"{case_id}_{variant.value}_repair_1.json",
            raw_root / f"{case_id}_{variant.value}_initial.json",
        ]
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        raise ValueError(f"missing raw replay source: {case_id}/{variant.value}")
    wrapper = json.loads(selected.read_text(encoding="utf-8"))
    if not wrapper.get("raw_output"):
        raise ValueError(f"raw replay source has no output: {case_id}/{variant.value}")
    return wrapper


def replay_block_records(
    *,
    cases: list[DatasetCase],
    source_prediction_root: Path | str,
    source_raw_root: Path | str,
    output_prediction_root: Path | str,
    dataset_hash: str,
) -> list[PredictionRecord]:
    source_prediction_root = Path(source_prediction_root)
    source_raw_root = Path(source_raw_root)
    output_prediction_root = Path(output_prediction_root)
    output_prediction_root.mkdir(parents=True, exist_ok=True)
    protocol_hash = stable_protocol_hash(
        {
            "replay_version": BLOCK_REPLAY_VERSION,
            "dataset_hash": dataset_hash,
            "source_variant": ExperimentVariant.C_BLOCK.value,
            "new_api_calls": 0,
        }
    )
    records: list[PredictionRecord] = []
    for case in cases:
        source_record = PredictionRecord.model_validate_json(
            (
                source_prediction_root
                / f"{case.case_id}_{ExperimentVariant.C_BLOCK.value}.json"
            ).read_text(encoding="utf-8")
        )
        try:
            wrapper = _raw_wrapper(
                source_raw_root,
                case_id=case.case_id,
                variant=ExperimentVariant.C_BLOCK,
            )
            parsed = BlockMappingOutput.model_validate_json(
                wrapper["raw_output"]
            )
            items = EvidenceGroundingRunner._validate_block_items(
                case,
                parsed.items,
            )
            EvidenceGroundingRunner._validate_requirement_coverage(
                case,
                items,
            )
            items = [
                item.model_copy(
                    update={
                        "reason_codes": item.reason_codes
                        + ["raw_schema_replayed_v1_1"]
                    }
                )
                for item in items
            ]
            record = PredictionRecord(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                variant=ExperimentVariant.C_BLOCK,
                schema_valid=True,
                items=items,
                traces=source_record.traces,
            )
        except (ValueError, OSError) as exc:
            record = PredictionRecord(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                variant=ExperimentVariant.C_BLOCK,
                schema_valid=False,
                execution_error=f"raw_replay_{type(exc).__name__}",
                traces=source_record.traces,
            )
        atomic_write_json(
            output_prediction_root
            / f"{case.case_id}_{ExperimentVariant.C_BLOCK.value}.json",
            record.model_dump(mode="json"),
        )
        records.append(record)
    return records


def replay_atomic_records(
    *,
    cases: list[DatasetCase],
    source_prediction_root: Path | str,
    source_raw_root: Path | str,
    output_prediction_root: Path | str,
    output_graph_root: Path | str,
    dataset_hash: str,
) -> list[PredictionRecord]:
    source_prediction_root = Path(source_prediction_root)
    source_raw_root = Path(source_raw_root)
    output_prediction_root = Path(output_prediction_root)
    output_graph_root = Path(output_graph_root)
    output_prediction_root.mkdir(parents=True, exist_ok=True)
    output_graph_root.mkdir(parents=True, exist_ok=True)
    protocol_hash = stable_protocol_hash(
        {
            "replay_version": ATOMIC_REPLAY_VERSION,
            "dataset_hash": dataset_hash,
            "source_variant": ExperimentVariant.C_ATOMIC.value,
            "new_api_calls": 0,
        }
    )
    records: list[PredictionRecord] = []
    for case in cases:
        source_record = PredictionRecord.model_validate_json(
            (
                source_prediction_root
                / f"{case.case_id}_{ExperimentVariant.C_ATOMIC.value}.json"
            ).read_text(encoding="utf-8")
        )
        try:
            wrapper = _raw_wrapper(
                source_raw_root,
                case_id=case.case_id,
                variant=ExperimentVariant.C_ATOMIC,
            )
            draft = AtomicEvidenceGraphDraft.model_validate_json(
                wrapper["raw_output"]
            )
            prediction, canonicalization = canonicalize_atomic_graph_draft(
                draft
            )
            validated = validate_atomic_evidence_graph(
                resume_text=case.resume_text,
                prediction=prediction,
                expected_requirement_ids={
                    unit.requirement_id for unit in case.requirements
                },
            )
            record = AtomicEvidenceGroundingRunner._to_prediction_record(
                case=case,
                prediction=prediction,
                validated=validated,
                dataset_hash=dataset_hash,
                protocol_hash=protocol_hash,
                traces=source_record.traces,
            )
            record = record.model_copy(
                update={
                    "items": [
                        item.model_copy(
                            update={
                                "reason_codes": item.reason_codes
                                + ["raw_schema_replayed_v1_1"]
                            }
                        )
                        for item in record.items
                    ]
                }
            )
            artifact = AtomicValidatedArtifact(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                graph=validated,
                draft_canonicalization=canonicalization,
            )
            atomic_write_json(
                output_graph_root
                / f"{case.case_id}_{ExperimentVariant.C_ATOMIC.value}.json",
                artifact.model_dump(mode="json"),
            )
        except (ValueError, OSError) as exc:
            record = PredictionRecord(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                variant=ExperimentVariant.C_ATOMIC,
                schema_valid=False,
                execution_error=f"raw_replay_{type(exc).__name__}",
                traces=source_record.traces,
            )
        atomic_write_json(
            output_prediction_root
            / f"{case.case_id}_{ExperimentVariant.C_ATOMIC.value}.json",
            record.model_dump(mode="json"),
        )
        records.append(record)
    return records
