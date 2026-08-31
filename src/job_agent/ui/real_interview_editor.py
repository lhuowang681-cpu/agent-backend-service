from __future__ import annotations

import streamlit as st

from job_agent.ui.real_interview_rounds import RealInterviewQuestionRecord


def _ids_key(ui_scope: str) -> str:
    return f"real_interview_question_ids::{ui_scope}"


def _counter_key(ui_scope: str) -> str:
    return f"real_interview_question_counter::{ui_scope}"


def _field_key(ui_scope: str, question_id: str, field: str) -> str:
    return f"real_interview_{field}::{ui_scope}::{question_id}"


def _ensure_ids(ui_scope: str) -> list[str]:
    key = _ids_key(ui_scope)
    if key not in st.session_state:
        st.session_state[key] = ["q1"]
        st.session_state[_counter_key(ui_scope)] = 1
    return list(st.session_state[key])


def _add_question(ui_scope: str) -> None:
    counter = int(st.session_state.get(_counter_key(ui_scope), 1)) + 1
    st.session_state[_counter_key(ui_scope)] = counter
    ids = list(st.session_state.get(_ids_key(ui_scope), ["q1"]))
    ids.append(f"q{counter}")
    st.session_state[_ids_key(ui_scope)] = ids


def _delete_question(ui_scope: str, question_id: str) -> None:
    ids = list(st.session_state.get(_ids_key(ui_scope), ["q1"]))
    if len(ids) <= 1:
        return
    st.session_state[_ids_key(ui_scope)] = [
        item for item in ids if item != question_id
    ]
    for field in (
        "question",
        "answer",
        "follow_up",
        "feedback",
        "assessment",
    ):
        st.session_state.pop(
            _field_key(ui_scope, question_id, field),
            None,
        )


def reset_real_interview_editor(ui_scope: str) -> None:
    ids = list(st.session_state.get(_ids_key(ui_scope), []))
    for question_id in ids:
        for field in (
            "question",
            "answer",
            "follow_up",
            "feedback",
            "assessment",
        ):
            st.session_state.pop(
                _field_key(ui_scope, question_id, field),
                None,
            )
    st.session_state.pop(_ids_key(ui_scope), None)
    st.session_state.pop(_counter_key(ui_scope), None)


def render_real_interview_question_editor(
    ui_scope: str,
) -> list[RealInterviewQuestionRecord]:
    ids = _ensure_ids(ui_scope)
    records: list[RealInterviewQuestionRecord] = []
    st.markdown("##### 面试问题与回答（建议填写）")
    st.caption(
        "记录得越具体，AI 越能区分知识、表达、项目证据和事实边界缺口。"
    )
    for index, question_id in enumerate(ids, start=1):
        with st.container(border=True):
            title_col, delete_col = st.columns([5, 1])
            with title_col:
                st.markdown(f"**问题 {index}**")
            with delete_col:
                st.button(
                    "删除",
                    key=f"delete_real_question::{ui_scope}::{question_id}",
                    disabled=len(ids) <= 1,
                    on_click=_delete_question,
                    args=(ui_scope, question_id),
                    use_container_width=True,
                )
            question = st.text_area(
                "面试官问题（可选）",
                key=_field_key(ui_scope, question_id, "question"),
                height=90,
            )
            answer = st.text_area(
                "我的回答（可选）",
                key=_field_key(ui_scope, question_id, "answer"),
                height=120,
            )
            follow_up, feedback = st.columns(2)
            with follow_up:
                interviewer_follow_up = st.text_area(
                    "面试官追问（可选）",
                    key=_field_key(ui_scope, question_id, "follow_up"),
                    height=90,
                )
            with feedback:
                interviewer_feedback = st.text_area(
                    "面试官反馈（可选）",
                    key=_field_key(ui_scope, question_id, "feedback"),
                    height=90,
                )
            assessment = st.selectbox(
                "我的现场自评",
                ("未判断", "答得清楚", "回答卡顿", "不会回答"),
                key=_field_key(ui_scope, question_id, "assessment"),
            )
            if question.strip():
                records.append(
                    RealInterviewQuestionRecord(
                        question_id=question_id,
                        question=question.strip(),
                        answer=answer.strip(),
                        interviewer_follow_up=interviewer_follow_up.strip(),
                        interviewer_feedback=interviewer_feedback.strip(),
                        self_assessment=assessment,
                    )
                )
    st.button(
        "添加一道面试题",
        key=f"add_real_question::{ui_scope}",
        on_click=_add_question,
        args=(ui_scope,),
    )
    return records
