from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from job_agent.schemas import EvidenceLevel


@dataclass(frozen=True)
class EvidencePresentation:
    label: str
    explanation: str
    advice: str
    is_supported: bool


_PRESENTATIONS = {
    EvidenceLevel.C3: EvidencePresentation(
        label="匹配充分",
        explanation="简历原文包含实际行动或产物，并提供了量化结果。",
        advice="可以重点保留；面试时准备解释数据口径、个人贡献和复现方式。",
        is_supported=True,
    ),
    EvidenceLevel.C2: EvidencePresentation(
        label="直接匹配",
        explanation="简历原文包含实际行动、项目产物或评测记录，但量化结果不充分。",
        advice="可以写入简历；最好补充规模、指标、对比结果或个人负责范围。",
        is_supported=True,
    ),
    EvidenceLevel.C1: EvidencePresentation(
        label="部分匹配",
        explanation=(
            "简历中存在相关经历或能力线索，但关键行动、产物、负责范围或结果不完整。"
        ),
        advice=(
            "可以作为相关经历谨慎表达；补充你实际做了什么、如何验证和个人贡献，"
            "不要写出超出原文证据的结论。"
        ),
        is_supported=False,
    ),
    EvidenceLevel.C0: EvidencePresentation(
        label="证据较弱",
        explanation="找到了相关词，但缺少明确行动、产物或可验证结果。",
        advice="暂时不要作为核心经历；先补充具体做法、产物和结果。",
        is_supported=False,
    ),
    EvidenceLevel.NONE: EvidencePresentation(
        label="未找到匹配",
        explanation="当前简历中没有找到能够支持这条岗位要求的原文。",
        advice="不要根据 JD 补写不存在的经历；有真实项目时再补充证据。",
        is_supported=False,
    ),
}


def presentation_for(level: EvidenceLevel | str) -> EvidencePresentation:
    resolved = level if isinstance(level, EvidenceLevel) else EvidenceLevel(level)
    return _PRESENTATIONS[resolved]


def clean_proof(proof: str) -> str:
    cleaned = proof.strip()
    if cleaned.startswith("简历原文："):
        cleaned = cleaned.removeprefix("简历原文：").strip()
    if cleaned in {"", "无直接证据"}:
        return "未在简历中找到对应原文"
    return cleaned


def match_counts(levels: Iterable[EvidenceLevel | str]) -> dict[str, int]:
    counts = {"supported": 0, "partial": 0, "missing": 0}
    for value in levels:
        level = value if isinstance(value, EvidenceLevel) else EvidenceLevel(value)
        if level in {EvidenceLevel.C2, EvidenceLevel.C3}:
            counts["supported"] += 1
        elif level in {EvidenceLevel.C0, EvidenceLevel.C1}:
            counts["partial"] += 1
        else:
            counts["missing"] += 1
    return counts


_AUDIT_BULLET = re.compile(
    r"^- (?:(?:Evidence-backed bullet|Cautious bullet after 1-day evidence check|"
    r"Stronger bullet after adding concrete proof|Only after evidence exists): )?"
    r"(?P<claim>.*?) \[evidence=(?P<level>C[0-3]|None); "
    r"proof=(?P<proof>.*?); risk=(?P<risk>.*?)\]$"
)

_HEADING_TRANSLATIONS = {
    "## Resume Strategy": "## 修改原则",
    "## Conservative Bullets": "## 可以安全使用的内容",
    "## Standard Bullets": "## 需要谨慎表达的内容",
    "## Stronger After Evidence": "## 补充证据后再使用",
    "## Claims To Remove Or Downgrade": "## 不应直接写入的内容",
    "## Suggested Claims (evidence-grounded)": "## 可以考虑写入的内容",
    "## Claim Audit": "## 表述安全检查",
}

_EMPTY_LINES = {
    "- No C2/C3 evidence-backed bullet is ready yet.",
    "- No C1 cautious bullet is available.",
    "- No evidence-upgrade bullet is needed.",
}


def user_facing_resume_markdown(markdown: str) -> str:
    """Translate the internal resume audit artifact into a Chinese user view."""

    rendered: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("# Targeted Resume Tailoring:"):
            rendered.append("# 简历修改建议")
            continue
        if line.startswith("# Resume Patch"):
            rendered.append("# AI 生成的简历修改建议")
            continue
        if line in _HEADING_TRANSLATIONS:
            rendered.append(_HEADING_TRANSLATIONS[line])
            continue
        if line in _EMPTY_LINES:
            continue
        if line == "- No unsupported claim detected.":
            rendered.append("- 暂未发现必须删除的无证据表述。")
            continue
        if line.startswith("- Use C2/C3 evidence directly"):
            rendered.append(
                "- 优先使用简历中有明确行动、产物或结果支撑的内容；"
                "证据不足的内容保持谨慎，不根据 JD 编造经历。"
            )
            continue
        match = _AUDIT_BULLET.match(line)
        if match:
            presentation = presentation_for(match.group("level"))
            rendered.extend(
                [
                    f"- **对应岗位要求：{match.group('claim')}**",
                    f"  - 简历依据：{clean_proof(match.group('proof'))}",
                    f"  - 修改建议：{presentation.advice}",
                ]
            )
            continue
        rendered.append(raw_line)

    result = "\n".join(rendered).strip()
    return result or "当前没有可展示的简历修改建议。"
