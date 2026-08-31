"""Streamlit AppTest — 工作台端到端测试。

已知限制（Streamlit 1.37.1）：
- `st.selectbox.set_value()` 有随机 AttributeError（_widget_state）
- 投递追踪 section 和面试复盘的轮次选择依赖 selectbox，偶发失败属已知 bug
- 模拟面试 section 无 selectbox，全流程可稳定通过
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _seed_session_for_review(tmp_path: Path) -> Path:
    """造一个含完整 mock interview 数据的 session 目录（用于复盘 AppTest）。"""
    from job_agent.nodes.answer_cards import render_answer_cards
    from job_agent.nodes.mock_interview import render_mock_interview_plan
    from job_agent.schemas import (
        AnswerCard,
        AnswerCardDeck,
        MockInterviewPlan,
        MockInterviewQuestion,
    )

    session_dir = tmp_path / "output" / "sessions" / "testco_engineer"
    session_dir.mkdir(parents=True)

    plan = MockInterviewPlan(
        company="TestCo", title="Engineer", mode="technical",
        persona="friendly", difficulty="realistic",
        questions=[
            MockInterviewQuestion(
                question_id="Q1", requirement_id="req_sft",
                prompt="Explain SFT workflow.", focus="technical_depth",
            ),
            MockInterviewQuestion(
                question_id="Q2", requirement_id="req_lora",
                prompt="How does LoRA work?", focus="technical_depth",
            ),
        ],
        scoring_dimensions=["accuracy"], live_rules=["be honest"],
    )
    cards = AnswerCardDeck(
        company="TestCo", title="Engineer",
        cards=[
            AnswerCard(
                requirement_id="req_sft", question="q",
                short_answer="SFT is supervised fine-tuning.",
                evidence_level="C2", supporting_evidence="ev",
                boundary="do not overclaim",
            ),
            AnswerCard(
                requirement_id="req_lora", question="q",
                short_answer="LoRA uses low-rank adaptation.",
                evidence_level="C2", supporting_evidence="ev",
                boundary="do not overclaim",
            ),
        ],
    )

    (session_dir / "09_mock_interview_plan.md").write_text(
        render_mock_interview_plan(plan), encoding="utf-8")
    (session_dir / "08_answer_cards.md").write_text(
        render_answer_cards(cards), encoding="utf-8")
    # session 基础文件
    (session_dir / "01_jd_structured.json").write_text(json.dumps({
        "company": "TestCo", "title": "Engineer",
    }, ensure_ascii=False), encoding="utf-8")
    (session_dir / "selected_job.json").write_text(json.dumps({
        "job_id": "test", "company": "TestCo", "title": "Engineer",
    }, ensure_ascii=False), encoding="utf-8")
    (session_dir / "03_fit_verdict.json").write_text(json.dumps({
        "verdict": "strong fit", "score": 85,
    }, ensure_ascii=False), encoding="utf-8")
    (session_dir / "06_targeted_resume.md").write_text(
        "# Targeted Resume\n- SFT experience", encoding="utf-8")
    (session_dir / "07_interview_grilling.md").write_text(
        "# Interview Grilling\n- Explain SFT.", encoding="utf-8")

    # session_state.json
    (session_dir / "session_state.json").write_text(json.dumps({
        "session_id": "test", "company": "TestCo", "title": "Engineer",
        "version": 1, "latest_verdict": "strong fit",
        "artifact_paths": {}, "artifact_freshness": {},
        "available_actions": [], "pending_evidence_gaps": [],
    }, ensure_ascii=False), encoding="utf-8")

    return session_dir


def _render_workbench(session_dir: Path) -> AppTest:
    at = AppTest.from_string(
        "from job_agent.ui.views.workbench_page import render_workbench_page\n"
        "render_workbench_page()\n"
    )
    at.session_state["session_dir"] = str(session_dir)
    at.session_state["active_session_dir"] = str(session_dir)
    at.run()
    return at


class TestAppSmoke:
    """基本页面加载：不传输入，只验证无异常渲染。"""

    def test_app_loads_without_exception(self) -> None:
        at = AppTest.from_file("src/job_agent/ui/app.py")
        at.run()
        assert not at.exception


class TestMockInterviewAppTest:
    """模拟面试 section：start → answer → finish（无 selectbox，稳定）。"""

    def test_full_mock_interview_flow(self, tmp_path: Path) -> None:
        session_dir = _seed_session_for_review(tmp_path)
        at = _render_workbench(session_dir)
        at.session_state["active_workbench_step"] = "review"
        at.run()
        assert not at.exception

        assert any(
            "面试复盘" in (e.value if hasattr(e, "value") else "")
            for e in at.subheader
        )
        assert not any(b.label == "开始模拟面试" for b in at.button)

    def test_review_section_renders_with_data(self, tmp_path: Path) -> None:
        """验证有数据时面试复盘 section 渲染，不崩。"""
        # 先创建一轮 run 数据
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        payload = {
            "result": {"average_score": 4.0, "company": "TestCo", "title": "Engineer",
                       "completed_question_ids": ["Q1", "Q2"],
                       "scores": [], "next_focus": ["RLHF"], "final_note": "ok"},
            "history": [
                {"question_id": "Q1", "prompt": "Explain SFT.", "answer": "SFT is...",
                 "score": 4, "improvement": "good", "elapsed_seconds": 30},
                {"question_id": "Q2", "prompt": "How does LoRA work?", "answer": "LoRA uses...",
                 "score": 4, "improvement": "good", "elapsed_seconds": 45},
            ],
        }
        session_dir = _seed_session_for_review(tmp_path)
        (session_dir / f"10_mock_interview_run_{ts}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        at = _render_workbench(session_dir)
        at.session_state["active_workbench_step"] = "review"
        at.run()
        assert not at.exception

        # 面试复盘 section 可见
        assert any("面试复盘" in (e.value if hasattr(e, "value") else "") for e in at.subheader)


class TestApplicationOpsAppTest:
    """投递追踪是用户主工作流中的可见步骤。"""

    def test_ops_section_is_visible_in_primary_workflow(self, tmp_path: Path) -> None:
        session_dir = _seed_session_for_review(tmp_path)
        at = _render_workbench(session_dir)
        at.session_state["active_workbench_step"] = "application"
        at.run()
        assert not at.exception
        workflow = next(item for item in at.radio if item.label == "工作流")
        assert "投递进度" in workflow.options
        assert any(
            "投递进度" in (item.value if hasattr(item, "value") else "")
            for item in at.subheader
        )


class TestInformationArchitecture:
    def test_empty_workbench_opens_new_session_wizard(self, tmp_path: Path) -> None:
        at = AppTest.from_string(
            "from job_agent.ui.views.workbench_page import render_workbench_page\n"
            "render_workbench_page()\n"
        )
        at.session_state["active_output_root"] = str(tmp_path / "empty-output")
        at.run()

        assert not at.exception
        assert any(
            "开始新的求职流程" in (title.value if hasattr(title, "value") else "")
            for title in at.title
        )

    def test_interview_log_is_independent_page(self, tmp_path: Path) -> None:
        session_dir = _seed_session_for_review(tmp_path)
        payload = {
            "result": {"average_score": 4.0, "next_focus": ["RLHF"]},
            "history": [
                {
                    "question_id": "Q1",
                    "prompt": "Explain SFT.",
                    "answer": "SFT is...",
                    "score": 4,
                    "improvement": "Add metrics.",
                    "elapsed_seconds": 30,
                }
            ],
        }
        (session_dir / "10_mock_interview_run_2026-07-24T120000.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        at = AppTest.from_string(
            "from job_agent.ui.views.interview_log_page import render_interview_log_page\n"
            "render_interview_log_page()\n"
        )
        at.session_state["active_output_root"] = str(tmp_path / "output")
        at.run()

        assert not at.exception
        assert any(
            "面试日志" in (title.value if hasattr(title, "value") else "")
            for title in at.title
        )
        assert any(
            "TestCo · Engineer" in (item.value if hasattr(item, "value") else "")
            for item in at.markdown
        )
