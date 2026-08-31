from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from job_agent.schemas import RawJob

EXPLICIT_JD_DELIMITER = "===== 新岗位 ====="


class JobDraft(BaseModel):
    """工作台中的一个显式岗位草稿；正文内部空行永远属于当前 JD。"""

    draft_id: str = Field(min_length=1)
    company: str = ""
    title: str = ""
    location: str = ""
    jd_text: str = ""


def _split_segments(text: str) -> list[str]:
    # 三个以上换行（含空行）切段；再多空行也只算一个分隔
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _guess_title(segment: str) -> str:
    first_line = segment.splitlines()[0].strip()
    for separator in ("｜", "|", "·"):
        if separator in first_line:
            left, right = (part.strip() for part in first_line.split(separator, 1))
            if left and right:
                return right[:60]
    # 取第一行的标题部分（到第一个句号/冒号/职责关键字），避免含职责描述导致
    # 与 jd_structurer LLM 提取的 title 不一致（guard title_mismatch）
    for sep in ("。", "：", "，", "；", "职责", "要求"):
        idx = first_line.find(sep)
        if idx > 0:
            first_line = first_line[:idx].strip()
            break
    return first_line[:60] or "User JD"


def _guess_company(segment: str) -> str | None:
    first_line = segment.splitlines()[0].strip()
    for separator in ("｜", "|", "·"):
        if separator in first_line:
            left, right = (part.strip() for part in first_line.split(separator, 1))
            if left and right:
                return left[:80]
    return None


def _guess_location(segment: str) -> str:
    match = re.search(
        r"(?:地点|工作地点|城市|location)\s*[：:]\s*([^\n，,；;。]+)",
        segment,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else "unknown"


def parse_jd_pool(
    text: str,
    company: str | None = None,
    location: str | None = None,
) -> list[RawJob]:
    """把用户粘贴的多段 JD 文本解析成 RawJob 列表（空行分段）。

    company：可选公司名，若提供则所有段共用。live 模式下用来与 LLM 从 JD 提取的
    公司对齐，避免 jd_structurer guard 的 company_mismatch；留空则默认 "User JD {index}"。
    location：可选地点覆盖；留空时尝试从每段 JD 的“地点/城市”字段中提取。
    """
    jobs: list[RawJob] = []
    for index, segment in enumerate(_split_segments(text), start=1):
        job_id = f"user_jd_{index}"
        jobs.append(
            RawJob(
                job_id=job_id,
                company=company or _guess_company(segment) or f"User JD {index}",
                title=_guess_title(segment),
                desc=segment,
                url=f"manual://{job_id}",
                location=(location or "").strip() or _guess_location(segment),
                source="user_provided",
            )
        )
    return jobs


def build_jobs_from_drafts(drafts: list[JobDraft]) -> list[RawJob]:
    """把显式岗位卡片转换为 RawJob；一个 draft 严格对应一个岗位。"""

    jobs: list[RawJob] = []
    seen_ids: set[str] = set()
    for draft in drafts:
        text = draft.jd_text.strip()
        if not text:
            continue
        normalized_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", draft.draft_id).strip("-")
        if not normalized_id:
            raise ValueError("岗位草稿 ID 无效")
        job_id = f"user_jd_{normalized_id}"
        if job_id in seen_ids:
            raise ValueError(f"岗位草稿 ID 重复：{draft.draft_id}")
        seen_ids.add(job_id)
        jobs.append(
            RawJob(
                job_id=job_id,
                company=draft.company.strip()
                or _guess_company(text)
                or f"User JD {len(jobs) + 1}",
                title=draft.title.strip() or _guess_title(text),
                desc=text,
                url=f"manual://{job_id}",
                location=draft.location.strip() or _guess_location(text),
                source="user_provided",
            )
        )
    return jobs


def parse_explicit_jd_import(text: str) -> list[str]:
    """按明确分隔符拆分批量文本；普通空行不会切分岗位。"""

    return [
        segment.strip()
        for segment in text.split(EXPLICIT_JD_DELIMITER)
        if segment.strip()
    ]


def save_pool(jobs: list[RawJob], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([j.model_dump(mode="json") for j in jobs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_pool(path: Path) -> list[RawJob]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RawJob.model_validate(item) for item in data]
