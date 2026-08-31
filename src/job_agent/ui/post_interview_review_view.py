from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


_NOTE_HEADERS = {
    "## Captured Interview Notes",
    "## 本次面试记录",
    "## 面试原始记录",
}


@dataclass(frozen=True)
class PostInterviewReviewView:
    notes: str
    round_id: str = ""
    questions: tuple[dict[str, Any], ...] = ()

    @property
    def is_too_short(self) -> bool:
        return len(self.notes.strip()) < 10 and not self.questions


def load_post_interview_review(path: Path) -> PostInterviewReviewView:
    path = Path(path)
    if not path.is_file():
        return PostInterviewReviewView(notes="")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            notes = payload.get("notes", "")
            questions = payload.get("questions", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            return PostInterviewReviewView(notes="")
        return PostInterviewReviewView(
            notes=notes.strip() if isinstance(notes, str) else "",
            round_id=str(payload.get("round_id", "")),
            questions=tuple(
                item for item in questions if isinstance(item, dict)
            )
            if isinstance(questions, list)
            else (),
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return PostInterviewReviewView(notes="")
    captured: list[str] = []
    in_notes = False
    for line in lines:
        stripped = line.strip()
        if stripped in _NOTE_HEADERS:
            in_notes = True
            continue
        if in_notes and stripped.startswith("## "):
            break
        if in_notes and stripped:
            captured.append(stripped.removeprefix("- ").strip())
    if not captured and lines and not any(
        line.strip() in _NOTE_HEADERS for line in lines
    ):
        captured = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return PostInterviewReviewView(notes="\n".join(captured).strip())


def render_post_interview_review(path: Path) -> None:
    review = load_post_interview_review(path)
    if not review.notes and not review.questions:
        st.info("这次真实面试还没有可读取的记录。")
        return
    st.markdown("#### 本次面试记录")
    if review.notes:
        with st.container(border=True):
            st.write(review.notes)
    if review.questions:
        st.markdown("#### 本轮问题与回答")
        for index, item in enumerate(review.questions, start=1):
            with st.expander(
                f"问题 {index}：{str(item.get('question', ''))[:80]}",
                expanded=index == 1,
            ):
                st.markdown(f"**问题**\n\n{item.get('question', '—')}")
                st.markdown(
                    f"**我的回答**\n\n{item.get('answer') or '未记录'}"
                )
                st.markdown(
                    f"**面试官追问**\n\n"
                    f"{item.get('interviewer_follow_up') or '未记录'}"
                )
                st.markdown(
                    f"**面试官反馈**\n\n"
                    f"{item.get('interviewer_feedback') or '未记录'}"
                )
                st.caption(
                    f"现场自评：{item.get('self_assessment', '未判断')}"
                )
    if review.is_too_short:
        st.warning(
            "当前记录太短，无法形成有效复盘。请至少补充一道面试题、你的回答表现，"
            "以及面试官的反馈或追问。"
        )
        return
    st.markdown("#### 建议复盘顺序")
    st.write("1. 列出面试官实际问到的问题，不要只写知识点名称。")
    st.write("2. 标记每道题是“答得清楚、回答卡顿、不会回答”中的哪一种。")
    st.write("3. 为卡顿或不会的问题补一份真实依据、60 秒答案和下一次练习动作。")
    st.caption(
        "这份记录只帮助整理问题和改进动作，不推断面试通过率，也不替你编造反馈。"
    )


def render_ai_review_result(review) -> None:
    st.markdown("#### AI 复盘结论")
    st.write(review.summary)
    for diagnosis in review.question_diagnoses:
        with st.expander(
            f"{diagnosis.question_id} · {diagnosis.status}",
            expanded=diagnosis.status in {"weak", "insufficient_information"},
        ):
            if diagnosis.strengths:
                st.markdown("**可以保留**")
                for item in diagnosis.strengths:
                    st.write(f"- {item}")
            if diagnosis.improved_answer:
                st.markdown(
                    f"**改进版回答**\n\n{diagnosis.improved_answer}"
                )
            st.markdown(f"**练习动作**\n\n{diagnosis.practice_action}")
    st.markdown("#### 优先补齐")
    if review.priority_gaps:
        for gap in review.priority_gaps:
            st.write(
                f"- [{gap.priority}] {gap.finding} → {gap.action}"
            )
    else:
        st.write("- 当前记录不足以形成可靠优先级。")
    st.markdown("#### 下一轮练习")
    for item in review.learning_plan:
        st.write(f"- {item}")
    if review.next_mock_topics:
        st.markdown("**建议模拟面试主题**")
        for item in review.next_mock_topics:
            st.write(f"- {item}")
    if review.resume_adjustments:
        st.markdown("**简历调整建议**")
        for item in review.resume_adjustments:
            st.write(f"- {item}")
    if review.limitations:
        st.markdown("**信息不足与事实边界**")
        for item in review.limitations:
            st.write(f"- {item}")
    st.caption(
        "AI 复盘用于发现准备缺口，不判断面试是否通过，也不预测录用概率。"
    )
