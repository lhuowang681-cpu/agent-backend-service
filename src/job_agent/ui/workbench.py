from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    requires: tuple[str, ...]  # 该环节"已完成"所需存在的 artifact 文件名


# 环节定义与 ARTIFACT_FILES 对齐（见 session_orchestrator.ARTIFACT_FILES）
STAGES: list[Stage] = [
    Stage("jd_structured", "JD 结构化", ("01_jd_structured.json",)),
    Stage("evidence", "证据映射", ("01_jd_structured.json", "02_evidence_mapping.json")),
    Stage("fit", "Fit 判定", ("01_jd_structured.json", "02_evidence_mapping.json", "03_fit_verdict.json")),
    Stage("resume", "改简历", ("01_jd_structured.json", "02_evidence_mapping.json", "06_targeted_resume.md")),
    Stage(
        "interview",
        "面试准备",
        ("01_jd_structured.json", "02_evidence_mapping.json", "03_fit_verdict.json", "06_targeted_resume.md", "07_interview_grilling.md"),
    ),
    Stage("answer_cards", "答题卡", ("07_interview_grilling.md", "08_answer_cards.md")),
    Stage("mock", "模拟面试", ("08_answer_cards.md", "09_mock_interview_plan.md")),
]


@dataclass(frozen=True)
class StageStatus:
    stage_id: str
    name: str
    status: str  # "done" | "locked"


def compute_stage_statuses(session_dir: Path) -> list[StageStatus]:
    """根据 session 目录里已存在的 artifact 文件，算出每个环节的状态。"""
    if (session_dir / "evidence_current.json").exists():
        resume_ready = (session_dir / "06_targeted_resume.md").exists()
        return [
            StageStatus("jd_structured", "JD 结构化", "done"),
            StageStatus("evidence", "证据映射", "done"),
            StageStatus("fit", "Fit 判定", "done"),
            StageStatus("resume", "改简历", "done" if resume_ready else "locked"),
            StageStatus("interview", "面试准备", "done" if resume_ready else "locked"),
            StageStatus("answer_cards", "答题卡（旧版）", "locked"),
            StageStatus("mock", "模拟面试（旧版）", "locked"),
        ]
    return [
        StageStatus(
            stage_id=stage.stage_id,
            name=stage.name,
            status="done" if all((session_dir / req).exists() for req in stage.requires) else "locked",
        )
        for stage in STAGES
    ]
