from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from job_agent.agent_runtime.contracts import AgentAction, AgentFinish
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.domain_agents.opportunity_research import (
    OpportunityResearchAgent,
    OpportunityResearchGoal,
)
from job_agent.schemas import RawJob
from job_agent.ui.jd_pool import save_pool


@dataclass(frozen=True)
class ResearchCandidate:
    job_id: str
    title: str
    company: str
    score: float
    reason: str


def _offline_replay_script(jobs: list[RawJob]) -> list:
    """通用 offline 回放轨迹：search all → read each → dedupe → finish。

    ScriptedAgentModel 决定"调什么 tool"（固定序列），tool 真读 JD 池文件返回真数据。
    evidence_refs 必须用 verifier 期望的 step:<N>:<tool_name> 格式（step_index 从 1 起递增）。
    """
    job_ids = [j.job_id for j in jobs]
    # step 1: search；step 2..N+1: read 每个 JD；step N+2: dedupe
    evidence_refs = ["step:1:jobs.search"]
    for index in range(len(job_ids)):
        evidence_refs.append(f"step:{index + 2}:jobs.read")
    evidence_refs.append(f"step:{len(job_ids) + 2}:jobs.deduplicate")

    actions: list = [
        AgentAction(
            action_id="search-1",
            tool_name="jobs.search",
            tool_arguments={"keywords": list({kw for j in jobs for kw in (j.title,)}),
                             "cities": []},
            expected_observation=f"返回 {len(jobs)} 个候选 JD",
            progress_claim="已用关键词检索到全部用户 JD",
        ),
    ]
    for index, j in enumerate(job_ids, start=1):
        actions.append(
            AgentAction(
                action_id=f"read-{index}",
                tool_name="jobs.read",
                tool_arguments={"job_id": j},
                expected_observation=f"读取 {j} 全文",
                progress_claim=f"已读取 {j}",
            )
        )
    actions.append(
        AgentAction(
            action_id="dedupe-1",
            tool_name="jobs.deduplicate",
            tool_arguments={"job_ids": job_ids},
            expected_observation="去重后保留全部候选",
            progress_claim="已对候选去重",
        )
    )
    actions.append(
        AgentFinish(
            result={
                "search_queries": ["user-provided jd pool"],
                "candidate_job_ids": job_ids,
                "evidence_refs": evidence_refs,
                "unresolved_gaps": [],
            },
            completion_evidence=evidence_refs,
            unresolved_items=[],
            confidence=1.0,
        )
    )
    return actions


def run_research(
    jobs: list[RawJob],
    goal: OpportunityResearchGoal,
    *,
    workspace_root: Path,
    use_live: bool = False,
    decision_model=None,
) -> list[ResearchCandidate]:
    """对多 JD 池跑 Research agent，返回候选列表。

    offline（默认）：ScriptedAgentModel 回放通用轨迹，tool 真读 JD 池。
    live：调用方传 decision_model（ToolUseDecisionModel），走真 LLM tool-use。
    返回所有候选（offline 下等分占位 score=1.0），UI 让用户选 lead。
    """
    workspace_root = Path(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    pool_path = save_pool(jobs, workspace_root / "_research_pool.json")

    if use_live and decision_model is not None:
        model = decision_model
    else:
        model = ScriptedAgentModel(_offline_replay_script(jobs))

    agent = OpportunityResearchAgent(model=model, source_path=pool_path)
    run_result = agent.run(
        goal=goal,
        session_id="workbench-research",
        run_id="research-1",
        workspace_root=workspace_root,
    )

    candidate_ids = _extract_candidate_ids(run_result, jobs)
    by_id = {j.job_id: j for j in jobs}
    candidates: list[ResearchCandidate] = []
    for job_id in candidate_ids:
        job = by_id.get(job_id)
        if job is None:
            continue
        candidates.append(
            ResearchCandidate(
                job_id=job.job_id,
                title=job.title,
                company=job.company,
                score=1.0,
                reason="offline scripted 回放候选（用户粘贴 JD，未做语义打分）",
            )
        )
    return candidates


def _extract_candidate_ids(run_result, jobs: list[RawJob]) -> list[str]:
    """从 AgentRunResult 提取候选 job_id；兜底返回全部（offline 回放总返回全部）。

    AgentRunResult.result 是 dict（直接来自 AgentFinish.result），不是 AgentFinish 对象。
    """
    payload = getattr(run_result, "result", None)
    if isinstance(payload, dict) and payload.get("candidate_job_ids"):
        return list(payload["candidate_job_ids"])
    # 兜底：result 取不到 candidate_job_ids 时，返回全部 job_id（offline 回放的预期）
    return [j.job_id for j in jobs]
