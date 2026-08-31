from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fit_verdict_engine.batch_eval import RawModelOutputRecord
from fit_verdict_engine.data_generator import SyntheticRecord, write_jsonl
from job_agent.nodes.fit_verdict import LocalVerdictRunner, _build_local_lora_prompt


def collect_raw_outputs(
    records: Iterable[SyntheticRecord],
    runner: LocalVerdictRunner,
) -> list[RawModelOutputRecord]:
    rows: list[RawModelOutputRecord] = []
    for record in records:
        prompt = _build_local_lora_prompt(record.fit_input)
        raw_model_output = runner.generate(prompt)
        rows.append(
            RawModelOutputRecord(
                record_id=record.record_id,
                gold=record.label,
                raw_model_output=raw_model_output,
                fallback_verdict=record.label,
            )
        )
    return rows


def write_raw_output_jsonl(path: Path, rows: Iterable[RawModelOutputRecord]) -> Path:
    return write_jsonl(path, (row.model_dump(mode="json") for row in rows))
