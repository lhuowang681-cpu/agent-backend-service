from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from job_agent.agent_runtime.live_agent_model import build_live_agent_model
from job_agent.domain_agents.application_material import (
    APPLICATION_MATERIAL_TOOLS,
    ApplicationMaterialAgent,
    ApplicationMaterialGoal,
    ApplicationMaterialResult,
    ResumeClaimCandidate,
    build_application_material_registry,
)
from job_agent.schemas import EvidenceItem, EvidenceLevel, StructuredJD
from job_agent.session_orchestrator import _write_regenerated_resume
from job_agent.ui.resume_source import write_targeted_resume_draft


# LaTeX 特殊字符转义表（% & _ # $ { }）
_LATEX_SPECIAL = str.maketrans({"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"})


@dataclass(frozen=True)
class ReviseResult:
    resume_md: str
    latex: str
    claims: list[dict]        # ClaimAuditEntry 的 dict（status/reason_code/...）
    invalidated: list[str]    # 失效的下游 artifact 文件名


def render_resume_latex(resume_patch: list[ResumeClaimCandidate]) -> str:
    """把建议 claims 渲染成 LaTeX itemize 片段（转义特殊字符）。"""
    if not resume_patch:
        return "\\begin{itemize}\n\\end{itemize}"
    lines = ["\\begin{itemize}"]
    for claim in resume_patch:
        text = claim.text.translate(_LATEX_SPECIAL)
        lines.append(f"  \\item {text}")
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def _render_patch_markdown(result: ApplicationMaterialResult) -> str:
    """把 ApplicationMaterialResult 渲染成 06 的 markdown（建议 claims + 审计摘要）。"""
    lines = [f"# Resume Patch — {result.company} / {result.title}", ""]
    lines.append("## Suggested Claims (evidence-grounded)")
    if result.resume_patch:
        for claim in result.resume_patch:
            lines.append(f"- [{claim.requirement_id}] {claim.text} (evidence: {', '.join(claim.evidence_ids)})")
    else:
        lines.append("- (no suggested claims)")
    lines.extend(["", "## Claim Audit"])
    for entry in result.claim_audit:
        lines.append(f"- {entry.claim.text}: **{entry.status.value}** ({entry.reason_code})")
    return "\n".join(lines)


def _read_session_context(session_dir: Path) -> tuple[StructuredJD, list[EvidenceItem], str]:
    """Read the current Evidence v2 projection, with legacy compatibility."""
    from job_agent.evidence.projections import EvidenceProjectionService
    from job_agent.evidence.repository import EvidenceArtifactRepository
    from job_agent.evidence.sources import EvidenceSourceStore
    from job_agent.schemas import RawJob

    bundle = EvidenceArtifactRepository().load_current(session_dir)
    if bundle is not None:
        selected = RawJob.model_validate(
            json.loads(
                (session_dir / "selected_job.json").read_text(encoding="utf-8")
            )
        )
        jd_source = next(
            item for item in bundle.sources.documents if item.kind == "raw_jd"
        )
        raw_jd = EvidenceSourceStore(session_dir).read(
            bundle.sources,
            jd_source.source_id,
        )
        inputs = EvidenceProjectionService().for_resume_agent(
            bundle,
            company=selected.company,
            title=selected.title,
            raw_jd=raw_jd,
        )
        resume_path = session_dir / "06_targeted_resume.md"
        resume_text = (
            resume_path.read_text(encoding="utf-8")
            if resume_path.exists()
            else ""
        )
        return inputs.structured_jd, inputs.evidence, resume_text

    jd = StructuredJD.model_validate(json.loads((session_dir / "01_jd_structured.json").read_text(encoding="utf-8")))
    ev_payload = json.loads((session_dir / "02_evidence_mapping.json").read_text(encoding="utf-8"))
    evidence = [EvidenceItem.model_validate(item) for item in ev_payload]
    normalized_path = session_dir / "00_resume_normalized.md"
    resume_path = (
        normalized_path
        if normalized_path.exists()
        else session_dir / "06_targeted_resume.md"
    )
    resume_text = resume_path.read_text(encoding="utf-8") if resume_path.exists() else ""
    return jd, evidence, resume_text


def revise_resume_live(session_dir: Path, feedback: str, runtime) -> ReviseResult:
    """live 模式：ApplicationMaterialAgent 按反馈（拼 objective）产出 claim 审计 + 建议 claims，
    渲染 markdown 写回 06 + 失效下游，返回 ReviseResult（含 latex 片段）。"""
    from job_agent.evidence.projections import EvidenceProjectionService
    from job_agent.evidence.repository import EvidenceArtifactRepository

    current = EvidenceArtifactRepository().load_current(session_dir)
    if current is not None and (
        not current.requirements.atoms
        or not EvidenceProjectionService().for_resume(current).claims
    ):
        resume_path = session_dir / "06_targeted_resume.md"
        return ReviseResult(
            resume_md=(
                resume_path.read_text(encoding="utf-8")
                if resume_path.exists()
                else ""
            ),
            latex="",
            claims=[],
            invalidated=[],
        )

    structured_jd, evidence, resume_text = _read_session_context(session_dir)
    if not evidence:
        return ReviseResult(
            resume_md=resume_text,
            latex="",
            claims=[],
            invalidated=[],
        )
    material_registry = build_application_material_registry(structured_jd, evidence, resume_text)
    model = build_live_agent_model(
        provider=runtime.provider, registry=material_registry,
        allowed_tools=APPLICATION_MATERIAL_TOOLS, agent_id="application-material",
        session_id=f"workbench-resume:{session_dir.name}", run_id="resume-revise-1",
    )
    agent = ApplicationMaterialAgent(
        model=model, structured_jd=structured_jd, evidence=evidence, resume_text=resume_text,
    )
    goal = ApplicationMaterialGoal(
        company=structured_jd.company, title=structured_jd.title,
        objective=(
            "用中文生成有证据支撑的简历修改建议，不得编造经历；"
            f"用户补充要求：{feedback}"
        ),
    )
    run_result = agent.run(
        goal=goal, session_id=f"workbench-resume:{session_dir.name}",
        run_id="resume-revise-1", workspace_root=session_dir,
    )
    material = ApplicationMaterialResult.model_validate(run_result.result)
    resume_md = _render_patch_markdown(material)
    latex = render_resume_latex(material.resume_patch)
    invalidated = _write_regenerated_resume(
        session_dir, resume_md, updated_by="resume_loop_live", note="Revised by ApplicationMaterialAgent (live).",
    )
    write_targeted_resume_draft(session_dir)
    return ReviseResult(
        resume_md=resume_md, latex=latex,
        claims=[entry.model_dump(mode="json") for entry in material.claim_audit],
        invalidated=invalidated,
    )


_TOPIC_ALIASES = {
    "SFT": ("sft",),
    "LoRA": ("lora",),
    "RLHF": ("rlhf",),
    "DPO": ("dpo",),
    "GRPO": ("grpo",),
}


def _excluded_topics(feedback: str) -> set[str]:
    normalized = feedback.casefold()
    if not any(marker in normalized for marker in ("删除", "去掉", "移除", "不要", "排除")):
        return set()
    return {
        label
        for label, aliases in _TOPIC_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    }


def _render_offline_revision(
    structured_jd: StructuredJD,
    evidence: list[EvidenceItem],
    feedback: str,
) -> str:
    excluded = _excluded_topics(feedback)
    supported_only = any(
        marker in feedback
        for marker in ("只保留有证据", "仅保留有证据", "只写有证据", "不要无证据")
    )

    def is_excluded(item: EvidenceItem) -> bool:
        lowered = item.claim.casefold()
        return any(
            label in excluded
            and any(alias in lowered for alias in _TOPIC_ALIASES[label])
            for label in excluded
        )

    visible = [item for item in evidence if not is_excluded(item)]
    supported = [
        item
        for item in visible
        if item.level in {EvidenceLevel.C1, EvidenceLevel.C2, EvidenceLevel.C3}
    ]
    gaps = [
        item
        for item in visible
        if item.level in {EvidenceLevel.C0, EvidenceLevel.NONE}
    ]
    lines = [
        f"# 目标简历：{structured_jd.company} · {structured_jd.title}",
        "",
        "## 修改策略",
        "",
        "- 仅使用当前简历中可定位的证据，不补写未提供的项目经历。",
    ]
    if excluded:
        lines.append("- 已按要求删除用户指定的无证据主题。")
    lines.extend(["", "## 有证据的经历", ""])
    if supported:
        for item in supported:
            proof = item.proof.removeprefix("简历原文：")
            lines.append(f"- {proof}（对应岗位要求：{item.claim}）")
    else:
        lines.append("- 当前简历中没有可安全写入的匹配经历。")
    if not supported_only:
        lines.extend(["", "## 需要补充证据的岗位要求", ""])
        if gaps:
            lines.extend(f"- {item.claim}：{item.risk}" for item in gaps)
        else:
            lines.append("- 暂无。")
    return "\n".join(lines).rstrip() + "\n"


def revise_resume_offline(session_dir: Path, feedback: str = "") -> ReviseResult:
    """offline/mock 模式：按用户反馈重建 evidence-grounded 简历。"""
    structured_jd, evidence, _ = _read_session_context(session_dir)
    resume_md = _render_offline_revision(structured_jd, evidence, feedback)
    invalidated = _write_regenerated_resume(
        session_dir,
        resume_md,
        updated_by="resume_loop_offline",
        note="Revised from deterministic evidence mapping and user feedback.",
    )
    write_targeted_resume_draft(session_dir)
    return ReviseResult(
        resume_md=resume_md,
        latex="",
        claims=[],
        invalidated=invalidated,
    )
