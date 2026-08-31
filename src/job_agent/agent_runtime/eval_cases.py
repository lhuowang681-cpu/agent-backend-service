from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from job_agent.schemas import ApplicationStatus, StrictModel


class HarnessEvalCase(StrictModel):
    case_id: str = Field(min_length=1)
    role_family: Literal["agent_algorithm", "llm_application", "backend_data"]
    target_role: str = Field(min_length=1)
    cities: list[str] = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    min_candidates: int = Field(default=1, gt=0)
    resume_fixture: str = Field(min_length=1)
    mock_answers: list[str] = Field(min_length=1)
    desired_tracker_status: ApplicationStatus = ApplicationStatus.TO_APPLY
    expected_semantic_outcome: Literal["continue", "stop_and_reselect"] = "continue"


_CASE_LIST = TypeAdapter(list[HarnessEvalCase])
_REQUIRED_FAMILIES = {"agent_algorithm", "llm_application", "backend_data"}


def load_harness_eval_cases(path: Path | str) -> list[HarnessEvalCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = _CASE_LIST.validate_python(payload)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate harness eval case id")
    families = {case.role_family for case in cases}
    if families != _REQUIRED_FAMILIES:
        raise ValueError(
            "harness eval cases must contain exactly agent_algorithm, llm_application, backend_data"
        )
    return cases
