from pathlib import Path
from unittest.mock import patch, MagicMock

from job_agent.evidence.contracts import EvidenceLevel
from job_agent.schemas import RawJob
from job_agent.ui.resume_loop import (
    ReviseResult,
    _read_session_context,
    render_resume_latex,
    revise_resume_live,
    revise_resume_offline,
)
from tests.evidence_v2_fixtures import commit_single_atom_evidence_v2


def test_render_resume_latex_escapes_and_itemizes():
    from job_agent.domain_agents.application_material import ResumeClaimCandidate
    patch_claims = [
        ResumeClaimCandidate(claim_id="c1", requirement_id="r1", text="提升 100% 吞吐 & 降低延迟", evidence_ids=["ev1"]),
        ResumeClaimCandidate(claim_id="c2", requirement_id="r2", text="SFT/LoRA #1", evidence_ids=["ev2"]),
    ]
    latex = render_resume_latex(patch_claims)
    assert latex.startswith("\\begin{itemize}")
    assert latex.endswith("\\end{itemize}")
    assert "\\item" in latex
    assert "100\\%" in latex          # % 转义
    assert "\\&" in latex             # & 转义
    assert "\\#" in latex             # # 转义


def _make_session(tmp_path: Path) -> Path:
    """造一个有 06 的 session（用 offline refill 生成完整 session）。"""
    from job_agent.ui.jd_pool import parse_jd_pool
    from job_agent.ui.runner import run_initial
    from job_agent.ui.modes import WorkbenchMode
    jobs = parse_jd_pool("做 SFT/LoRA。", company="TestCo")
    return run_initial(jobs, "# 简历\n- 会 SFT", tmp_path, mode=WorkbenchMode.OFFLINE_RULE)


def test_revise_resume_offline_regenerates_06(tmp_path: Path):
    session_dir = _make_session(tmp_path)
    before = (session_dir / "06_targeted_resume.md").read_text(encoding="utf-8")
    result = revise_resume_offline(session_dir)
    assert isinstance(result, ReviseResult)
    assert result.resume_md  # 非空
    assert (session_dir / "06_targeted_resume.md").exists()


def test_revise_resume_offline_honors_removal_feedback(tmp_path: Path):
    from job_agent.ui.jd_pool import parse_jd_pool
    from job_agent.ui.modes import WorkbenchMode
    from job_agent.ui.runner import run_initial

    session_dir = run_initial(
        parse_jd_pool(
            "星河科技｜AI 应用工程师\n"
            "负责 LLM Agent 工作流、RAG 评测与自动化测试。"
        ),
        "# 简历\n- Agent checkpoint\n- RAG 离线评测\n- pytest 自动化测试",
        tmp_path / "workbench",
        mode=WorkbenchMode.OFFLINE_RULE,
    )
    result = revise_resume_offline(
        session_dir,
        "只保留有证据的 Agent、RAG 和自动化测试经历，"
        "删除 SFT、LoRA、RLHF 相关表述。全篇使用中文。",
    )

    assert "Agent checkpoint" in result.resume_md
    assert "RAG 离线评测" in result.resume_md
    assert "pytest 自动化测试" in result.resume_md
    assert "SFT" not in result.resume_md
    assert "LoRA" not in result.resume_md
    assert "RLHF" not in result.resume_md


def test_revise_resume_live_writes_06_and_returns_claims(tmp_path: Path):
    session_dir = _make_session(tmp_path)
    from job_agent.domain_agents.application_material import (
        ApplicationMaterialResult, ClaimAuditEntry, ResumeClaimCandidate, ClaimAuditStatus,
    )
    # NOTE: 原始 brief 用 fit=MagicMock()，但 ApplicationMaterialResult 是 StrictModel，
    # pydantic 会拒绝 MagicMock；改为最小化的合法 FitVerdictResult 构造（已在 schemas 中校验）。
    from job_agent.schemas import FitVerdictResult, RiskLevel, Verdict
    fake_fit = FitVerdictResult(
        verdict=Verdict.STRONG, score=0.8, coverage=0.7, risk_level=RiskLevel.LOW,
        need_human_review=False, reason_codes=["ok"], explanation="fake fit",
    )
    fake_result = ApplicationMaterialResult(
        company="TestCo", title="User JD 1", fit=fake_fit,
        fit_evidence_refs=["ev1"],
        resume_patch=[ResumeClaimCandidate(claim_id="c1", requirement_id="req_sft_lora", text="会 SFT", evidence_ids=["ev1"])],
        claim_audit=[ClaimAuditEntry(
            claim=ResumeClaimCandidate(claim_id="c1", requirement_id="req_sft_lora", text="会 SFT", evidence_ids=["ev1"]),
            status=ClaimAuditStatus.SUPPORTED, reason_code="ok", evidence_refs=["ev1"], ungrounded_facts=[])],
        unresolved_gaps=[],
    )
    fake_run = MagicMock()
    fake_run.result = fake_result.model_dump(mode="json")
    fake_runtime = MagicMock()
    with patch("job_agent.ui.resume_loop.ApplicationMaterialAgent") as fake_agent_cls, \
         patch("job_agent.ui.resume_loop.build_live_agent_model") as fake_model:
        fake_agent_cls.return_value.run.return_value = fake_run
        result = revise_resume_live(session_dir, "强调 RLHF", fake_runtime)
    assert result.resume_md
    assert result.latex
    assert result.claims and result.claims[0]["status"] == "supported"
    assert (session_dir / "06_targeted_resume.md").exists()


def _write_selected_job(session_dir: Path) -> None:
    job = RawJob(
        job_id="job-1",
        company="Acme",
        title="Agent Engineer",
        desc="需要独立设计评测",
        url="",
        location="",
    )
    (session_dir / "selected_job.json").write_text(
        job.model_dump_json(),
        encoding="utf-8",
    )


def test_v2_resume_context_reads_current_verified_projection(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    commit_single_atom_evidence_v2(session, level=EvidenceLevel.C2)
    _write_selected_job(session)
    (session / "06_targeted_resume.md").write_text(
        "CURRENT TARGET",
        encoding="utf-8",
    )

    structured, evidence, resume = _read_session_context(session)

    assert structured.company == "Acme"
    assert [item.evidence_id for item in evidence] == ["l1"]
    assert resume == "CURRENT TARGET"


def test_v2_resume_revision_does_not_turn_c1_into_claim(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    commit_single_atom_evidence_v2(session, level=EvidenceLevel.C1)
    _write_selected_job(session)
    resume_path = session / "06_targeted_resume.md"
    resume_path.write_text("UNCHANGED TARGET", encoding="utf-8")

    result = revise_resume_live(
        session,
        "加入新事实",
        runtime=object(),
    )

    assert result.resume_md == "UNCHANGED TARGET"
    assert result.claims == []
    assert resume_path.read_text(encoding="utf-8") == "UNCHANGED TARGET"
