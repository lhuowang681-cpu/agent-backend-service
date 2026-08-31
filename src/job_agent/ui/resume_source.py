from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from job_agent.atomic_io import atomic_write_text, atomic_write_text_batch
from job_agent.schemas import EvidenceLevel
from job_agent.ui.resume_match import clean_proof


ALLOWED_RESUME_SUFFIXES = {".txt", ".md", ".tex"}
MAX_RESUME_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ResumeSource:
    filename: str
    suffix: str
    raw_text: str
    normalized_markdown: str
    byte_size: int


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "resume.md"
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("简历文件不是可识别的 UTF-8 或 GB18030 文本。")


def _strip_latex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        lines.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(lines)


def normalize_latex_to_markdown(source: str) -> str:
    """Extract readable resume text from common LaTeX constructs."""

    text = _strip_latex_comments(source)
    text = re.sub(
        r"\\(?:documentclass|usepackage)(?:\[[^\]]*\])?\{[^{}]*\}",
        "",
        text,
    )
    text = re.sub(
        r"\\(?:pagestyle|thispagestyle|geometry)\{[^{}]*\}",
        "",
        text,
    )
    text = re.sub(
        r"\\(?:section|section\*|cvsection)\{([^{}]*)\}",
        lambda match: f"\n## {match.group(1)}\n",
        text,
    )
    text = re.sub(
        r"\\(?:subsection|subsection\*|cvsubsection)\{([^{}]*)\}",
        lambda match: f"\n### {match.group(1)}\n",
        text,
    )
    for _ in range(4):
        updated = re.sub(
            r"\\(?:textbf|textit|emph|underline|href)\{([^{}]*)\}",
            r"\1",
            text,
        )
        if updated == text:
            break
        text = updated
    text = re.sub(r"\\begin\{[^{}]+\}(?:\[[^\]]*\])?", "\n", text)
    text = re.sub(r"\\end\{[^{}]+\}", "\n", text)
    text = re.sub(r"\\item(?:\[[^\]]*\])?\s*", "\n- ", text)
    text = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace(r"\&", "&").replace(r"\%", "%").replace(r"\_", "_")
    text = text.replace(r"\#", "#").replace(r"\$", "$")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_resume_source(filename: str, data: bytes) -> ResumeSource:
    if not data:
        raise ValueError("简历文件不能为空。")
    if len(data) > MAX_RESUME_BYTES:
        raise ValueError("简历文件不能超过 2 MB。")
    safe_name = _safe_filename(filename)
    suffix = Path(safe_name).suffix.casefold()
    if suffix not in ALLOWED_RESUME_SUFFIXES:
        raise ValueError("只支持 .txt、.md 和 .tex 简历。")
    raw_text = _decode_text(data)
    normalized = (
        normalize_latex_to_markdown(raw_text)
        if suffix == ".tex"
        else raw_text.strip()
    )
    if not normalized:
        raise ValueError("简历解析后没有可用于匹配的文字。")
    return ResumeSource(
        filename=safe_name,
        suffix=suffix,
        raw_text=raw_text,
        normalized_markdown=normalized,
        byte_size=len(data),
    )


def build_pasted_resume_source(text: str) -> ResumeSource:
    value = text.strip()
    if not value:
        raise ValueError("简历内容不能为空。")
    data = value.encode("utf-8")
    return ResumeSource(
        filename="pasted_resume.md",
        suffix=".md",
        raw_text=value,
        normalized_markdown=value,
        byte_size=len(data),
    )


def persist_resume_source(session_dir: Path, source: ResumeSource) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    original_path = session_dir / f"00_original_resume{source.suffix}"
    atomic_write_text_batch(
        {
            original_path: source.raw_text,
            session_dir / "00_resume_normalized.md": (
                source.normalized_markdown
            ),
            session_dir / "00_resume_source.json": json.dumps(
                {
                    "filename": source.filename,
                    "suffix": source.suffix,
                    "byte_size": source.byte_size,
                    "original_path": original_path.name,
                    "normalized_path": "00_resume_normalized.md",
                },
                ensure_ascii=False,
                indent=2,
            ),
        }
    )
    return original_path


def load_resume_source_metadata(session_dir: Path) -> dict:
    path = Path(session_dir) / "00_resume_source.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_targeted_resume_draft(session_dir: Path) -> str:
    session_dir = Path(session_dir)
    original_path = session_dir / "00_resume_normalized.md"
    if not original_path.exists():
        legacy_path = session_dir / "00_original_resume.md"
        if legacy_path.exists():
            original_path = legacy_path
        else:
            return ""
    original = original_path.read_text(encoding="utf-8").strip()
    try:
        jd = json.loads((session_dir / "01_jd_structured.json").read_text(encoding="utf-8"))
        evidence_payload = json.loads(
            (session_dir / "02_evidence_mapping.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return original
    evidence = (
        evidence_payload.get("items", [])
        if isinstance(evidence_payload, dict)
        else evidence_payload
    )
    supported: list[str] = []
    cautious: list[tuple[EvidenceLevel, str]] = []
    requirement_by_id = {
        str(requirement.get("id", "")): str(requirement.get("text", ""))
        for requirement in jd.get("must_have", [])
        if isinstance(requirement, dict)
    }
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            level = EvidenceLevel(item.get("level", "None"))
        except ValueError:
            continue
        proof = clean_proof(str(item.get("proof", "")))
        if level in {EvidenceLevel.C2, EvidenceLevel.C3}:
            supported.append(proof)
        elif level in {EvidenceLevel.C0, EvidenceLevel.C1}:
            requirement = requirement_by_id.get(
                str(item.get("requirement_id", "")),
                str(item.get("claim", "")),
            )
            cautious.append((level, requirement))

    company = str(jd.get("company", "目标公司"))
    title = str(jd.get("title", "目标岗位"))
    lines = [
        f"# {company} · {title} 定向简历草稿",
        "",
        "> 本草稿保留原始简历正文，只把已有证据对应的内容放到岗位重点区；"
        "请人工确认后再投递。",
        "",
        "## 针对岗位建议优先展示",
        "",
    ]
    if supported:
        for proof in dict.fromkeys(supported):
            lines.append(f"- {proof}")
    else:
        lines.append("- 当前没有足够证据支撑的岗位重点，请先补充真实项目材料。")
    if cautious:
        lines.extend(
            [
                "",
                "## 需要谨慎处理的岗位要求",
                "",
                *[
                    (
                        f"- {claim}（已有相关线索；保留真实事实，并补充具体做法、"
                        "产物、结果和个人贡献）"
                        if level == EvidenceLevel.C1
                        else f"- {claim}（当前证据较弱，不要补写尚未发生的经历）"
                    )
                    for level, claim in dict.fromkeys(cautious)
                    if claim
                ],
            ]
        )
    lines.extend(["", "---", "", "## 原始简历正文", "", original])
    return "\n".join(lines).rstrip() + "\n"


def write_targeted_resume_draft(session_dir: Path) -> Path | None:
    draft = build_targeted_resume_draft(session_dir)
    if not draft:
        return None
    path = Path(session_dir) / "06_targeted_resume_draft.md"
    return atomic_write_text(path, draft)
