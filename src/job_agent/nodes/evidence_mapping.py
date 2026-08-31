from __future__ import annotations

import re
from pathlib import Path

from job_agent.nodes.jd_structurer import requirement_keywords
from job_agent.schemas import EvidenceItem, EvidenceLevel, JobRequirement


_LEARNING_MARKERS = ("了解", "熟悉", "学习", "阅读", "课程", "概念", "exposure")
_ACTION_MARKERS = (
    "负责",
    "实现",
    "完成",
    "搭建",
    "开发",
    "构建",
    "设计",
    "维护",
    "复现",
    "评测",
    "测试",
    "分析",
    "记录",
    "优化",
    "上线",
)
_ARTIFACT_MARKERS = (
    "checkpoint",
    "trajectory",
    "pytest",
    "日志",
    "报告",
    "指标",
    "配置",
    "测试集",
    "评测集",
    "baseline",
)


def _matching_lines(resume_text: str, requirement: JobRequirement) -> list[str]:
    keywords = requirement_keywords(requirement.id, requirement.text)
    if not keywords:
        return []
    matches: list[str] = []
    for raw_line in resume_text.splitlines():
        clean = raw_line.strip().lstrip("-*• ").strip()
        lowered = clean.casefold()
        if clean and any(keyword.casefold() in lowered for keyword in keywords):
            matches.append(clean)
    return matches


def _level_for_lines(lines: list[str]) -> EvidenceLevel:
    if not lines:
        return EvidenceLevel.NONE
    combined = " ".join(lines).casefold()
    has_action = any(marker in combined for marker in _ACTION_MARKERS)
    has_artifact = any(marker in combined for marker in _ARTIFACT_MARKERS)
    learning_only = any(marker in combined for marker in _LEARNING_MARKERS)
    if (has_action or has_artifact) and re.search(r"\d+(?:\.\d+)?%", combined):
        return EvidenceLevel.C3
    if has_action or has_artifact:
        return EvidenceLevel.C2
    if learning_only:
        return EvidenceLevel.C1
    return EvidenceLevel.C0


def map_evidence(resume_path: Path, requirements: list[JobRequirement]) -> list[EvidenceItem]:
    resume_text = resume_path.read_text(encoding="utf-8")
    evidence = []
    for index, requirement in enumerate(requirements, start=1):
        matches = _matching_lines(resume_text, requirement)
        level = _level_for_lines(matches)
        proof = (
            "简历原文：" + "；".join(matches[:2])
            if matches
            else "无直接证据"
        )
        risk = (
            "需要准备可验证的项目细节和个人贡献"
            if level in {EvidenceLevel.C2, EvidenceLevel.C3}
            else "只能谨慎描述学习经历，不能声称项目所有权"
            if level in {EvidenceLevel.C0, EvidenceLevel.C1}
            else "核心要求缺少简历证据"
        )
        evidence.append(
            EvidenceItem(
                evidence_id=f"ev_{index:03d}",
                requirement_id=requirement.id,
                claim=requirement.text,
                level=level,
                proof=proof,
                risk=risk,
            )
        )
    return evidence
