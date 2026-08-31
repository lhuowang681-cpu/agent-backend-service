from __future__ import annotations

from pathlib import Path

import streamlit as st

from job_agent.evidence.projections import EvidenceProjectionService
from job_agent.evidence.repository import EvidenceArtifactRepository


def render_evidence_v2_view(session_dir: Path) -> None:
    bundle = EvidenceArtifactRepository().load_current(session_dir)
    if bundle is None:
        legacy = EvidenceArtifactRepository().load_legacy_view(session_dir)
        if legacy is not None:
            st.info("旧版证据：来源位置未校验。重新分析后会创建 Evidence v2，不改写旧文件。")
        return
    view = EvidenceProjectionService().for_ui(bundle)
    st.subheader("岗位匹配证据")
    col_fit, col_confidence = st.columns(2)
    col_fit.metric("匹配判断", view.summary.verdict_label)
    col_confidence.metric("证据可信度", view.summary.confidence_label)
    if view.summary.strengths:
        st.markdown("**核心优势**")
        for item in view.summary.strengths:
            st.write(f"- {item}")
    if view.summary.needs_confirmation:
        st.markdown("**需要确认**")
        for item in view.summary.needs_confirmation:
            st.write(f"- {item}")
    if view.summary.critical_gaps:
        st.markdown("**关键缺口**")
        for item in view.summary.critical_gaps:
            st.write(f"- {item}")
    for item in view.items:
        with st.expander(item.label):
            st.caption("JD 原文")
            st.write(item.jd_quote)
            if item.evidence_quotes:
                st.caption("原始材料引用")
                for quote in item.evidence_quotes:
                    st.write(f"> {quote}")
            else:
                st.write("未发现可定位的原始材料证据。")
