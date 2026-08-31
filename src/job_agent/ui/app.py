from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from job_agent.career import CareerStore, LegacyCareerMigration
from job_agent.ui.local_config import load_local_runtime_defaults
from job_agent.ui.views.adaptive_interview_page import (
    render_adaptive_interview_page,
)
from job_agent.ui.views.company_library_page import (
    render_company_library_page,
)
from job_agent.ui.views.interview_log_page import render_interview_log_page
from job_agent.ui.views.workbench_page import render_workbench_page


def _output_root() -> Path:
    return Path(os.environ.get("JOB_AGENT_OUTPUT_ROOT", "output/workbench"))


def _career_store(*, migrate: bool = False) -> CareerStore:
    output_root = _output_root()
    store = CareerStore.open(output_root)
    if migrate and not st.session_state.get("career_migrated"):
        result = LegacyCareerMigration(store).apply(output_root)
        st.session_state["career_migrated"] = True
        st.session_state["career_migration_summary"] = result.summary
    return store


def _render_company_library() -> None:
    output_root = _output_root()
    render_company_library_page(
        output_root=output_root,
        store=_career_store(migrate=True),
    )


def _render_adaptive_interview() -> None:
    render_adaptive_interview_page(
        output_root=_output_root(),
        repo_root=Path.cwd(),
        store=_career_store(),
        defaults=load_local_runtime_defaults(),
    )


def main() -> None:
    st.set_page_config(
        page_title="求职工作台",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="auto",
    )
    workbench_page = st.Page(
        render_workbench_page,
        title="工作台",
        icon=":material/work:",
    )
    company_library_page = st.Page(
        _render_company_library,
        title="公司库",
        icon=":material/domain:",
        default=True,
    )
    adaptive_interview_page = st.Page(
        _render_adaptive_interview,
        title="面试训练",
        icon=":material/record_voice_over:",
    )
    interview_log_page = st.Page(
        render_interview_log_page,
        title="面试日志",
        icon=":material/history:",
    )
    st.session_state["_workbench_page"] = workbench_page
    st.session_state["_company_library_page"] = company_library_page
    st.session_state["_adaptive_interview_page"] = adaptive_interview_page
    navigation = st.navigation(
        [
            company_library_page,
            workbench_page,
            adaptive_interview_page,
            interview_log_page,
        ]
    )
    navigation.run()


if __name__ == "__main__":
    main()
