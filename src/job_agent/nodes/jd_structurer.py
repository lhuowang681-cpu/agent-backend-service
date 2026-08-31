from __future__ import annotations

from job_agent.schemas import JobRequirement, RawJob, RoleType, StructuredJD


_REQUIREMENT_SPECS = [
    (
        "req_sft_lora",
        "熟悉 SFT / LoRA 微调流程",
        "追问数据格式、训练配置、loss 与 eval 的关系",
        ("sft", "lora", "instruction tuning", "指令微调"),
    ),
    (
        "req_alignment",
        "了解 RLHF / DPO / GRPO / reward model 基本方法",
        "追问偏好数据、reward/verifier 设计和算法差异",
        ("rlhf", "rlaif", "dpo", "grpo", "reward model", "偏好对齐", "偏好数据"),
    ),
    (
        "req_training_ops",
        "具备训练流程、实验配置和 checkpoint 管理经验",
        "追问训练配置、日志、checkpoint 和失败恢复",
        ("训练流程", "训练日志", "实验日志", "training pipeline", "pipeline"),
    ),
    (
        "req_agent",
        "具备 LLM Agent 工作流、工具调用和状态恢复经验",
        "追问工作流编排、tool use、checkpoint、trajectory 和失败恢复",
        (
            "agent",
            "agent 工作流",
            "agent workflow",
            "tool use",
            "工具调用",
            "checkpoint",
            "trajectory",
        ),
    ),
    (
        "req_rag",
        "具备 RAG 检索、评测和 bad case 分析经验",
        "追问检索方案、评测集、指标、失败样例和改进闭环",
        ("rag", "检索增强", "向量检索", "召回", "rerank"),
    ),
    (
        "req_python",
        "熟练使用 Python 进行工程开发",
        "追问负责模块、工程结构、异常处理和可维护性",
        ("python",),
    ),
    (
        "req_testing",
        "具备自动化测试和质量保障经验",
        "追问测试分层、失败用例、回归策略和覆盖范围",
        ("自动化测试", "pytest", "单元测试", "集成测试", "端到端测试"),
    ),
    (
        "req_eval",
        "具备大模型评测、消融实验和 bad case 分析经验",
        "追问 baseline、指标定义、失败样例和局限性",
        ("评测", "消融实验", "bad case", "evaluation", "evaluate"),
    ),
]


def requirement_keywords(requirement_id: str, requirement_text: str = "") -> tuple[str, ...]:
    for spec_id, _, _, keywords in _REQUIREMENT_SPECS:
        if spec_id == requirement_id:
            return keywords
    tokens = tuple(
        token.casefold()
        for token in requirement_text.replace("/", " ").replace("、", " ").split()
        if len(token.strip()) >= 2
    )
    return tokens


def structure_jd(job: RawJob) -> StructuredJD:
    desc = job.desc.casefold()
    requirements = [
        JobRequirement(id=req_id, text=text, required=True, probe=probe)
        for req_id, text, probe, keywords in _REQUIREMENT_SPECS
        if any(keyword.casefold() in desc for keyword in keywords)
    ]
    if not requirements:
        summary = next(
            (line.strip() for line in job.desc.splitlines() if line.strip()),
            job.title,
        )
        requirements = [
            JobRequirement(
                id="req_general",
                text=summary[:120],
                required=True,
                probe="追问具体职责、个人贡献、证据和结果",
            )
        ]
    posttraining_ids = {"req_sft_lora", "req_alignment", "req_training_ops"}
    role_type = (
        RoleType.POSTTRAINING
        if any(item.id in posttraining_ids for item in requirements)
        else RoleType.UNKNOWN
    )
    return StructuredJD(
        company=job.company,
        title=job.title,
        role_type=role_type,
        must_have=requirements,
        nice_to_have=[],
        raw_jd=job.desc,
    )
