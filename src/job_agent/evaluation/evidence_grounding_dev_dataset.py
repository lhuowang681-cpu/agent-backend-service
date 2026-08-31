from __future__ import annotations

from collections import Counter

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    GoldRequirementUnit,
    GoldSpanDraft,
)
from job_agent.schemas import EvidenceLevel


DEV_DATASET_VERSION = "dev-synthetic-template-draft-v1"
DEV_CASE_COUNT = 40
DEV_UNIT_COUNT = 120

_STYLES = ("plain", "markdown", "latex_like", "crlf")
_COHORTS = (
    {
        "artifact": "Agent 工具调度器",
        "topic": "GRPO",
        "platform": "LLM 评测平台",
        "component": "日志解析器",
        "metric": "任务成功率",
        "keyword": "MCP",
    },
    {
        "artifact": "RAG 检索评测器",
        "topic": "DPO",
        "platform": "检索增强平台",
        "component": "重排模块",
        "metric": "检索命中率",
        "keyword": "LangGraph",
    },
    {
        "artifact": "多轮面试 Agent",
        "topic": "PPO",
        "platform": "对话 Agent 平台",
        "component": "状态机模块",
        "metric": "任务完成率",
        "keyword": "Toolformer",
    },
    {
        "artifact": "奖励数据流水线",
        "topic": "RLHF",
        "platform": "后训练数据平台",
        "component": "数据清洗模块",
        "metric": "有效样本率",
        "keyword": "LoRA",
    },
)


def _format_resume(lines: list[str], style: str) -> str:
    if style == "markdown":
        return "\n".join(f"- {line}" for line in lines)
    if style == "latex_like":
        return "\n".join(f"\\item {line}" for line in lines)
    if style == "crlf":
        return "\r\n".join(lines)
    return "\n".join(lines)


def _span(resume: str, quote: str, *, occurrence: int = 0) -> GoldSpanDraft:
    positions: list[int] = []
    start = 0
    while True:
        position = resume.find(quote, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    if occurrence >= len(positions):
        raise ValueError(f"gold quote occurrence missing: {quote}")
    start_char = positions[occurrence]
    return GoldSpanDraft(
        quote=quote,
        start_char=start_char,
        end_char=start_char + len(quote),
    )


def _unit(
    *,
    requirement_id: str,
    requirement: str,
    level: EvidenceLevel,
    resume: str,
    quotes: list[tuple[str, int]] | None = None,
    rationale: str,
) -> GoldRequirementUnit:
    return GoldRequirementUnit(
        requirement_id=requirement_id,
        requirement=requirement,
        gold_level=level,
        gold_spans=[
            _span(resume, quote, occurrence=occurrence)
            for quote, occurrence in (quotes or [])
        ],
        rationale=rationale,
    )


def _case(
    *,
    family: str,
    cohort_index: int,
    lines: list[str],
    build_units,
) -> DatasetCase:
    style = _STYLES[cohort_index]
    resume = _format_resume(lines, style)
    return DatasetCase(
        case_id=f"dev-{family}-{cohort_index + 1:02d}",
        split="dev",
        source_group=f"dev-source-{family}-{cohort_index + 1:02d}",
        source_kind="synthetic",
        slices=[family, style, "synthetic_template_draft"],
        resume_text=resume,
        requirements=build_units(resume),
        label_provenance="ai_draft",
        review_status="pending_human_review",
    )


def _explicit_action(index: int) -> DatasetCase:
    values = _COHORTS[index]
    result = 11 + index * 3
    action = (
        f"独立设计并实现{values['artifact']}，"
        f"将{values['metric']}提升{result}%。"
    )
    keyword = f"技能：{values['keyword']}。"
    lines = [action, keyword, "未参与 Kubernetes 生产部署。"]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_primary",
                requirement=f"{values['artifact']}开发与效果验证",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(action, 0)],
                rationale="candidate action + artifact + direct result",
            ),
            _unit(
                requirement_id="req_keyword",
                requirement=f"{values['keyword']} 实践",
                level=EvidenceLevel.C0,
                resume=resume,
                quotes=[(keyword, 0)],
                rationale="keyword list only",
            ),
            _unit(
                requirement_id="req_deploy",
                requirement="Kubernetes 生产部署",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation",
            ),
        ]

    return _case(
        family="explicit_action",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _cross_sentence(index: int) -> DatasetCase:
    values = _COHORTS[index]
    action = f"负责构建{values['artifact']}并完成离线验收。"
    result = f"上线试运行后，{values['metric']}提升{8 + index * 2}%。"
    negative = "没有负责线上集群运维。"
    lines = [action, result, negative]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_joint",
                requirement=f"{values['artifact']}及量化结果",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(action, 0), (result, 0)],
                rationale="cross-sentence action/artifact/result",
            ),
            _unit(
                requirement_id="req_build",
                requirement=f"构建{values['artifact']}",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(action, 0)],
                rationale="action + artifact without requiring result",
            ),
            _unit(
                requirement_id="req_ops",
                requirement="线上集群运维",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation",
            ),
        ]

    return _case(
        family="cross_sentence",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _learning_reproduction(index: int) -> DatasetCase:
    values = _COHORTS[index]
    learning = f"学习{values['topic']}课程并阅读两篇相关论文。"
    reproduction = (
        f"复现{values['topic']}最小训练脚本，产出可运行 notebook。"
    )
    negative = "尚未完成多机分布式训练。"
    lines = [learning, reproduction, negative]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_learning",
                requirement=f"{values['topic']}知识",
                level=EvidenceLevel.C1,
                resume=resume,
                quotes=[(learning, 0)],
                rationale="learning evidence",
            ),
            _unit(
                requirement_id="req_reproduction",
                requirement=f"{values['topic']}最小可运行复现",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(reproduction, 0)],
                rationale="reproduction + concrete artifact under rubric v1.1",
            ),
            _unit(
                requirement_id="req_distributed",
                requirement="多机分布式训练",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation",
            ),
        ]

    return _case(
        family="learning_reproduction",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _team_ownership(index: int) -> DatasetCase:
    values = _COHORTS[index]
    participation = (
        f"参与团队搭建{values['platform']}，负责其中的{values['component']}。"
    )
    result = f"团队整体{values['metric']}提升{9 + index}% 。"
    lines = [participation, result, "未声明对整个平台独立负责。"]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_platform_owner",
                requirement=f"独立负责{values['platform']}",
                level=EvidenceLevel.C1,
                resume=resume,
                quotes=[(participation, 0)],
                rationale="limited participation, not platform ownership",
            ),
            _unit(
                requirement_id="req_component",
                requirement=f"实现{values['component']}",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(participation, 0)],
                rationale="candidate owns a concrete component",
            ),
            _unit(
                requirement_id="req_team_result",
                requirement=f"候选人独立带来{values['metric']}提升",
                level=EvidenceLevel.C0,
                resume=resume,
                quotes=[(result, 0)],
                rationale="team result without candidate attribution",
            ),
        ]

    return _case(
        family="team_ownership",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _negation_plan(index: int) -> DatasetCase:
    values = _COHORTS[index]
    negative = f"未实现{values['artifact']}，仅阅读接口文档。"
    plan = f"计划下季度部署{values['platform']}。"
    actual = f"使用{values['keyword']}编写离线评测脚本。"
    lines = [negative, plan, actual]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_negated",
                requirement=f"实现{values['artifact']}",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation is not positive evidence",
            ),
            _unit(
                requirement_id="req_planned",
                requirement=f"部署{values['platform']}",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="future plan is not completed work",
            ),
            _unit(
                requirement_id="req_actual",
                requirement=f"{values['keyword']}离线评测脚本",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(actual, 0)],
                rationale="candidate action + concrete script",
            ),
        ]

    return _case(
        family="negation_plan",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _numeric_distractor(index: int) -> DatasetCase:
    values = _COHORTS[index]
    other = (
        f"在推荐系统中实现特征交叉模块，将 AUC 提升{2 + index}%。"
    )
    agent = f"实现{values['artifact']}并完成单元测试。"
    lines = [other, agent, "Agent 项目未记录准确率提升。"]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_recommender",
                requirement="推荐系统特征交叉与 AUC 提升",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(other, 0)],
                rationale="action + artifact + result in recommender task",
            ),
            _unit(
                requirement_id="req_agent",
                requirement=f"{values['artifact']}与测试",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(agent, 0)],
                rationale="action + artifact",
            ),
            _unit(
                requirement_id="req_agent_metric",
                requirement="Agent 准确率量化提升",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="numeric result belongs to another task",
            ),
        ]

    return _case(
        family="numeric_distractor",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _repeated_quote(index: int) -> DatasetCase:
    values = _COHORTS[index]
    repeated = (
        f"Agent 规划器调用 Agent 工具；候选人实现{values['artifact']}并完成回归。"
    )
    lines = [repeated, f"相关框架：{values['keyword']}。", "未部署在线服务。"]
    first_agent = ("Agent", 0)

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_agent_keyword",
                requirement="Agent 概念接触",
                level=EvidenceLevel.C0,
                resume=resume,
                quotes=[first_agent],
                rationale="repeated minimal quote with explicit occurrence",
            ),
            _unit(
                requirement_id="req_tool",
                requirement=f"实现{values['artifact']}与工具调用",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(repeated, 0)],
                rationale="action + artifact in a repeated-quote line",
            ),
            _unit(
                requirement_id="req_online",
                requirement="在线服务部署",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation",
            ),
        ]

    return _case(
        family="repeated_quote",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _shared_span(index: int) -> DatasetCase:
    values = _COHORTS[index]
    shared = (
        f"独立实现{values['platform']}的检索、重排和评测模块，"
        f"{values['metric']}提升{7 + index * 2}%。"
    )
    lines = [shared, "所有模块共用同一条项目描述。", "没有移动端交付。"]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_retrieval",
                requirement=f"{values['platform']}检索与量化结果",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(shared, 0)],
                rationale="shared span supports retrieval requirement",
            ),
            _unit(
                requirement_id="req_evaluation",
                requirement=f"{values['platform']}评测与量化结果",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(shared, 0)],
                rationale="same source span supports a second requirement",
            ),
            _unit(
                requirement_id="req_mobile",
                requirement="移动端交付",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="explicit negation",
            ),
        ]

    return _case(
        family="shared_span",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _multi_claim_sentence(index: int) -> DatasetCase:
    values = _COHORTS[index]
    sentence = (
        f"阅读{values['topic']}论文，同时独立实现{values['artifact']}并编写测试。"
    )
    lines = [sentence, "两个活动发生在同一阶段。", "未记录业务指标。"]
    learning_quote = f"阅读{values['topic']}论文"
    action_quote = f"独立实现{values['artifact']}并编写测试"

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_learning",
                requirement=f"{values['topic']}学习",
                level=EvidenceLevel.C1,
                resume=resume,
                quotes=[(learning_quote, 0)],
                rationale="one sentence contains a learning claim",
            ),
            _unit(
                requirement_id="req_implementation",
                requirement=f"{values['artifact']}实现",
                level=EvidenceLevel.C2,
                resume=resume,
                quotes=[(action_quote, 0)],
                rationale="same sentence contains an independent implementation claim",
            ),
            _unit(
                requirement_id="req_business_result",
                requirement=f"{values['artifact']}业务指标提升",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="no result evidence",
            ),
        ]

    return _case(
        family="multi_claim_sentence",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


def _cross_requirement(index: int) -> DatasetCase:
    values = _COHORTS[index]
    agent = (
        f"为客服 Agent 实现{values['component']}，"
        f"任务成功率提升{6 + index}% 。"
    )
    recommender = (
        f"为推荐系统实现召回模块，召回率提升{12 + index}% 。"
    )
    lines = [agent, recommender, "两项指标分别统计，不能交叉归因。"]

    def units(resume: str):
        return [
            _unit(
                requirement_id="req_agent",
                requirement=f"客服 Agent {values['component']}与结果",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(agent, 0)],
                rationale="agent-specific action/artifact/result",
            ),
            _unit(
                requirement_id="req_recommender",
                requirement="推荐系统召回模块与结果",
                level=EvidenceLevel.C3,
                resume=resume,
                quotes=[(recommender, 0)],
                rationale="recommender-specific action/artifact/result",
            ),
            _unit(
                requirement_id="req_agent_recall",
                requirement="客服 Agent 召回率提升",
                level=EvidenceLevel.NONE,
                resume=resume,
                rationale="recall metric belongs to recommender requirement",
            ),
        ]

    return _case(
        family="cross_requirement",
        cohort_index=index,
        lines=lines,
        build_units=units,
    )


_BUILDERS = (
    _explicit_action,
    _cross_sentence,
    _learning_reproduction,
    _team_ownership,
    _negation_plan,
    _numeric_distractor,
    _repeated_quote,
    _shared_span,
    _multi_claim_sentence,
    _cross_requirement,
)


def build_dev_cases() -> list[DatasetCase]:
    cases = [
        builder(cohort_index)
        for builder in _BUILDERS
        for cohort_index in range(len(_COHORTS))
    ]
    if len(cases) != DEV_CASE_COUNT:
        raise AssertionError("dev case count drift")
    if sum(len(case.requirements) for case in cases) != DEV_UNIT_COUNT:
        raise AssertionError("dev unit count drift")
    return cases


def build_dev_slice_counts(cases: list[DatasetCase]) -> dict[str, int]:
    counts = Counter(
        case.slices[0]
        for case in cases
    )
    return dict(sorted(counts.items()))
