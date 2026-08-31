from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, model_validator

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    load_jsonl_dataset,
)
from job_agent.schemas import StrictModel


DIRECT_JD_DATASET_VERSION = "direct-jd-synthetic-sidecar-v1"
DIRECT_JD_CASE_COUNT = 40
DIRECT_JD_UNIT_COUNT = 120


class DirectJDRequirementLink(StrictModel):
    requirement_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_char <= self.start_char:
            raise ValueError("direct JD requirement end must be greater than start")
        return self


class DirectJDCase(StrictModel):
    case_id: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    raw_jd_text: str = Field(min_length=1)
    requirement_links: list[DirectJDRequirementLink] = Field(min_length=1)
    construction: str = "deterministic_synthetic_raw_jd"

    @model_validator(mode="after")
    def validate_links(self):
        ids = [item.requirement_id for item in self.requirement_links]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate direct JD requirement_id")
        quotes = [item.quote for item in self.requirement_links]
        if len(quotes) != len(set(quotes)):
            raise ValueError("duplicate direct JD requirement quote")
        for item in self.requirement_links:
            if self.raw_jd_text[item.start_char : item.end_char] != item.quote:
                raise ValueError("direct JD requirement quote offset mismatch")
            if self.raw_jd_text.count(item.quote) != 1:
                raise ValueError("direct JD requirement quote must be unique")
        return self


class DirectJDSidecar(StrictModel):
    dataset_version: str = DIRECT_JD_DATASET_VERSION
    cases: list[DirectJDCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self):
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate direct JD case_id")
        groups = [item.source_group for item in self.cases]
        if len(groups) != len(set(groups)):
            raise ValueError("duplicate direct JD source_group")
        return self


def _format_raw_jd(requirements: list[str], style: str) -> str:
    background = (
        "团队背景：团队技术栈覆盖 Python、LLM 与 Agent，"
        "致力于建设智能招聘与评测平台。"
    )
    footer = "其他信息：工作地点、薪酬与福利以正式招聘页面为准。"
    if style == "markdown":
        lines = [
            "# Agent 算法岗位",
            background,
            "## 核心要求",
            *[f"- {item}" for item in requirements],
            footer,
        ]
        return "\n".join(lines)
    if style == "latex_like":
        lines = [
            "\\section{Agent 算法岗位}",
            background,
            "\\subsection{核心要求}",
            *[f"\\item {item}" for item in requirements],
            footer,
        ]
        return "\n".join(lines)
    if style == "crlf":
        lines = [
            "Agent 算法岗位",
            background,
            "核心要求：",
            *[f"{index}. {item}" for index, item in enumerate(requirements, start=1)],
            footer,
        ]
        return "\r\n".join(lines)
    lines = [
        "Agent 算法岗位",
        background,
        "核心要求：",
        *[f"{index}. {item}" for index, item in enumerate(requirements, start=1)],
        footer,
    ]
    return "\n".join(lines)


def build_direct_jd_cases(cases: list[DatasetCase]) -> list[DirectJDCase]:
    direct_cases: list[DirectJDCase] = []
    for case in cases:
        requirements = [item.requirement for item in case.requirements]
        style = case.slices[1]
        raw_jd = _format_raw_jd(requirements, style)
        links = []
        for unit in case.requirements:
            start = raw_jd.index(unit.requirement)
            links.append(
                DirectJDRequirementLink(
                    requirement_id=unit.requirement_id,
                    quote=unit.requirement,
                    start_char=start,
                    end_char=start + len(unit.requirement),
                )
            )
        direct_cases.append(
            DirectJDCase(
                case_id=case.case_id,
                source_group=case.source_group,
                raw_jd_text=raw_jd,
                requirement_links=links,
            )
        )
    return direct_cases


def validate_sidecar_alignment(
    cases: list[DatasetCase],
    sidecar: DirectJDSidecar,
) -> None:
    case_by_id = {case.case_id: case for case in cases}
    sidecar_by_id = {case.case_id: case for case in sidecar.cases}
    if set(case_by_id) != set(sidecar_by_id):
        raise ValueError("direct JD sidecar case coverage mismatch")
    for case_id, case in case_by_id.items():
        direct = sidecar_by_id[case_id]
        if direct.source_group != case.source_group:
            raise ValueError("direct JD source_group mismatch")
        expected = {
            unit.requirement_id: unit.requirement for unit in case.requirements
        }
        actual = {
            item.requirement_id: item.quote
            for item in direct.requirement_links
        }
        if actual != expected:
            raise ValueError("direct JD requirement alignment mismatch")


def load_direct_jd_sidecar(
    path: Path | str,
    *,
    grounding_dataset_path: Path | str | None = None,
) -> DirectJDSidecar:
    source = Path(path)
    sidecar = DirectJDSidecar(
        cases=[
            DirectJDCase.model_validate(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if grounding_dataset_path is not None:
        dataset = load_jsonl_dataset(grounding_dataset_path)
        validate_sidecar_alignment(dataset.cases, sidecar)
    return sidecar
