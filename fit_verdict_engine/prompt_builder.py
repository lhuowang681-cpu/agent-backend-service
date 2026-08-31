from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from fit_verdict_engine.data_generator import SyntheticRecord
from fit_verdict_engine.prompt_contract import FIT_VERDICT_SYSTEM_PROMPT, build_fit_verdict_user_prompt
from job_agent.nodes.fit_verdict import evaluate_fit


RULE_BLOCK = """
规则:
- strong fit: 70% 以上 must-have 有 C1+ 证据，且没有明显 toy signal
- weak fit: 40%-70% must-have 有证据，或多为 C0/C1 但可解释
- risky fit: 证据覆盖低于 40%，或高危 toy signal 命中
- not recommended: 核心要求完全无证据，且一周内无法补齐
"""

SCHEMA_BLOCK = """
输出 JSON 字段: verdict, score, coverage, risk_level, need_human_review, reason_codes, explanation
"""

SYSTEM_PROMPT = FIT_VERDICT_SYSTEM_PROMPT
EXPOSURES = ("runtime",)


def _payload(record: SyntheticRecord) -> str:
    return json.dumps(record.fit_input.model_dump(mode="json"), ensure_ascii=False, indent=2)


def build_prompt_variants(record: SyntheticRecord) -> dict[str, str]:
    payload = _payload(record)
    task = "判断候选人与岗位的 fit verdict。"
    return {
        "runtime": build_fit_verdict_user_prompt(record.fit_input),
        "full": f"{task}\n{SCHEMA_BLOCK}\n{RULE_BLOCK}\n输入:\n{payload}",
        "partial": f"{task}\n{SCHEMA_BLOCK}\n规则摘要: 根据 must-have 覆盖率和 toy signal 判断。\n输入:\n{payload}",
        "minimal": f"{task}\n{SCHEMA_BLOCK}\n输入:\n{payload}",
        "no_skill": f"输入:\n{payload}\n请输出 JSON。",
    }


def _assistant_payload(record: SyntheticRecord) -> str:
    result = evaluate_fit(record.fit_input)
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)


def build_training_rows(
    records: Iterable[SyntheticRecord],
    exposures: Sequence[str] = EXPOSURES,
) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        variants = build_prompt_variants(record)
        answer = _assistant_payload(record)
        for exposure in exposures:
            if exposure not in variants:
                raise ValueError(f"unknown exposure: {exposure}")
            rows.append(
                {
                    "record_id": f"{record.record_id}:{exposure}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": variants[exposure]},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {
                        "label": record.label.value,
                        "scenario_id": record.scenario_id,
                        "exposure": exposure,
                        "requirement_count": len(record.fit_input.requirements),
                        "evidence_count": len(record.fit_input.evidence),
                        "toy_signal_count": len(record.fit_input.toy_signals),
                    },
                }
            )
    return rows
