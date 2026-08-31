from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable
from uuid import uuid4

from job_agent.graph import run_jobs_flow, run_manual_jd_flow, run_semantic_job_flow
from job_agent.outputs import write_session_outputs
from job_agent.schemas import RawJob
from job_agent.ui.resume_source import (
    build_pasted_resume_source,
    build_resume_source,
    persist_resume_source,
    write_targeted_resume_draft,
)


def _v2_session_slug(job: RawJob) -> str:
    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        f"{job.company}_{job.title}".casefold(),
    ).strip("_") or "job"
    return f"{text}_{uuid4().hex[:8]}"


def run_initial_v2_live(
    jobs: list[RawJob],
    resume_text: str,
    output_dir: Path,
    *,
    runtime,
    selected_job_id: str | None = None,
    resume_source_name: str | None = None,
    resume_source_text: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Create one live-only Evidence v2 session without legacy fallback artifacts."""
    from job_agent.agents.resume_tailoring import ResumeTailoringAgent
    from job_agent.evidence.pipeline import EvidenceV2Pipeline
    from job_agent.evidence.projections import EvidenceProjectionService
    from job_agent.evidence.sources import EvidenceSourceStore
    from job_agent.nodes.resume_tailoring import render_targeted_resume
    from job_agent.session_orchestrator import write_session_state

    selected = _select_lead(jobs, selected_job_id)
    sessions_root = Path(output_dir) / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    session_name = _v2_session_slug(selected)
    session_dir = sessions_root / session_name
    staging_dir = sessions_root / f".{session_name}.staging"
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        (staging_dir / "selected_job.json").write_text(
            json.dumps(
                selected.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if progress_callback is not None:
            progress_callback("evidence_v2")
        pipeline = EvidenceV2Pipeline(
            registry=runtime.registry,
            harness=runtime.harness,
        )
        result = pipeline.run(
            session_dir=staging_dir,
            raw_jd=selected.desc,
            original_resume=resume_text,
            session_id=f"workbench:{session_name}",
        )
        analysis = result.bundle.requirements
        if not analysis.atoms or not result.resume_view.claims:
            resume_markdown = (
                f"# Targeted Resume: {selected.company} - {selected.title}\n\n"
                "> 当前没有可用于新增简历事实的 C2/C3 证据；以下保留用户原始简历。\n\n"
                f"{resume_text.strip()}\n"
            )
        else:
            raw_jd_source = next(
                item
                for item in result.bundle.sources.documents
                if item.kind == "raw_jd"
            )
            raw_jd = EvidenceSourceStore(staging_dir).read(
                result.bundle.sources,
                raw_jd_source.source_id,
            )
            agent_inputs = EvidenceProjectionService().for_resume_agent(
                result.bundle,
                company=selected.company,
                title=selected.title,
                raw_jd=raw_jd,
            )
            if not agent_inputs.evidence:
                resume_markdown = (
                    f"# Targeted Resume: {selected.company} - {selected.title}\n\n"
                    "> 当前证据仅部分支持岗位要求，未生成新的简历事实；以下保留用户原始简历。\n\n"
                    f"{resume_text.strip()}\n"
                )
            else:
                if progress_callback is not None:
                    progress_callback("resume_tailoring")
                tailored = ResumeTailoringAgent(
                    registry=runtime.registry,
                    harness=runtime.harness,
                ).run(
                    agent_inputs.structured_jd,
                    evidence=agent_inputs.evidence,
                    resume_text=resume_text,
                    session_id=f"workbench:{session_name}",
                    run_id=f"resume-v2-{uuid4().hex}",
                ).value
                resume_markdown = render_targeted_resume(tailored)
        (staging_dir / "06_targeted_resume.md").write_text(
            resume_markdown,
            encoding="utf-8",
        )
        source = (
            build_resume_source(
                resume_source_name,
                (resume_source_text or resume_text).encode("utf-8"),
            )
            if resume_source_name
            else build_pasted_resume_source(resume_text)
        )
        persist_resume_source(staging_dir, source)
        write_targeted_resume_draft(staging_dir)
        write_session_state(staging_dir)
        os.replace(staging_dir, session_dir)
        return session_dir
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def run_initial_base(jobs: list[RawJob], resume_text: str, output_dir: Path) -> Path:
    """offline 跑完整 DAG：把 resume_text 落盘，调 graph，写 session outputs，返回 session_dir。

    单 JD 走 run_manual_jd_flow；多 JD 走 run_jobs_flow（Job Funnel 打分，自动选 lead 最高的）。
    全程 offline，不调 LLM。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "_resume.md"
    resume_path.write_text(resume_text, encoding="utf-8")

    if len(jobs) == 1:
        job = jobs[0]
        state = run_manual_jd_flow(
            jd_text=job.desc,
            resume_path=resume_path,
            company=job.company,
            title=job.title,
            location=job.location or "unknown",
        )
    else:
        # run_jobs_flow 内部用 score_jobs 打分并选 selected_job_id（None 时取 lead 最高）
        state = run_jobs_flow(
            user_request="user-provided jd pool",
            jobs=jobs,
            resume_path=resume_path,
            selected_job_id=None,
        )

    paths = write_session_outputs(state, output_dir)
    paths.session_dir.mkdir(parents=True, exist_ok=True)
    persist_resume_source(paths.session_dir, build_pasted_resume_source(resume_text))
    write_targeted_resume_draft(paths.session_dir)
    return paths.session_dir


def _select_lead(jobs: list[RawJob], selected_job_id: str | None) -> RawJob:
    if selected_job_id is not None:
        for j in jobs:
            if j.job_id == selected_job_id:
                return j
    return jobs[0]


def _safe_model_dump(obj):
    """Pydantic 对象走 model_dump；测试 stub（object()）回退原值。"""
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _offline_mock_responses(state: dict) -> list[dict]:
    """从 offline state 构造 semantic 5 节点的 mock 响应。"""
    evidence = state.get("fit_input")
    evidence_payload = (
        {"items": [item.model_dump(mode="json") for item in evidence.evidence]}
        if evidence is not None and hasattr(evidence, "evidence")
        else {"items": []}
    )
    return [
        _safe_model_dump(state["structured_jd"]),
        evidence_payload,
        _safe_model_dump(state["resume_tailoring"]),
        _safe_model_dump(state["interview_prep"]),
        _safe_model_dump(state["answer_cards"]),
    ]


def run_initial(
    jobs: list[RawJob],
    resume_text: str,
    output_dir: Path,
    *,
    mode,
    runtime=None,
    skill_root: str | None = None,
    selected_job_id: str | None = None,
    user_request: str = "workbench-initial",
    resume_source_name: str | None = None,
    resume_source_text: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """三模式初始流程分发，返回 session_dir。

    - 多 JD：本函数假定调用方已选定 lead（selected_job_id）；若 None 取第一个。
      （多 JD 的 Research 候选选择在 app 层做，选完后把 selected_job_id 传进来。）
    - OFFLINE_RULE：run_manual_jd_flow（阶段 0 逻辑）。
    - AGENT_API_MOCK：用 skill_root，先 offline 跑取 mock_responses，再 build mock runtime，
      再 run_semantic_job_flow（验证管线，不烧真 token）。
    - AGENT_API_LIVE：用 runtime（app 层 build，含 key 的 provider/harness/registry）
      直接 run_semantic_job_flow。
    """
    from job_agent.ui.modes import WorkbenchMode, build_runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "_resume.md"
    resume_path.write_text(resume_text, encoding="utf-8")
    selected = _select_lead(jobs, selected_job_id)

    if mode == WorkbenchMode.OFFLINE_RULE:
        if progress_callback is not None:
            progress_callback("offline_processing")
        state = run_manual_jd_flow(
            jd_text=selected.desc,
            resume_path=resume_path,
            company=selected.company,
            title=selected.title,
            location=selected.location or "unknown",
        )
    elif mode == WorkbenchMode.AGENT_API_MOCK:
        if not skill_root:
            raise ValueError("AGENT_API_MOCK 模式需要 skill_root")
        # 先 offline 跑拿 4 节点 mock 响应，再 build mock runtime，再 semantic 跑
        base = run_manual_jd_flow(
            jd_text=selected.desc, resume_path=resume_path,
            company=selected.company, title=selected.title,
            location=selected.location or "unknown",
        )
        mock_runtime = build_runtime(
            WorkbenchMode.AGENT_API_MOCK,
            skill_root=skill_root,
            mock_responses=_offline_mock_responses(base),
        )
        state = run_semantic_job_flow(
            user_request=user_request, selected_job=selected, resume_path=resume_path,
            registry=mock_runtime.registry, harness=mock_runtime.harness,
            progress_callback=progress_callback,
        )
    elif mode == WorkbenchMode.AGENT_API_LIVE:
        if runtime is None:
            raise ValueError("AGENT_API_LIVE 模式需要 runtime（含 provider/harness/registry）")
        state = run_semantic_job_flow(
            user_request=user_request, selected_job=selected, resume_path=resume_path,
            registry=runtime.registry, harness=runtime.harness,
            progress_callback=progress_callback,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    paths = write_session_outputs(state, output_dir)
    paths.session_dir.mkdir(parents=True, exist_ok=True)
    source = (
        build_resume_source(
            resume_source_name,
            (resume_source_text or resume_text).encode("utf-8"),
        )
        if resume_source_name
        else build_pasted_resume_source(resume_text)
    )
    persist_resume_source(paths.session_dir, source)
    write_targeted_resume_draft(paths.session_dir)
    return paths.session_dir
