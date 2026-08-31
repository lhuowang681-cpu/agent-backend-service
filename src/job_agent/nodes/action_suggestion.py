from __future__ import annotations

from job_agent.schemas import ActionSuggestion, FitVerdictResult, Verdict


def suggest_action(result: FitVerdictResult) -> ActionSuggestion:
    if result.verdict == Verdict.STRONG:
        return ActionSuggestion(
            next_action="apply",
            summary="建议优先投递，并准备围绕训练配置、评测指标和 bad case 的面试追问。",
            actions=["保存 JD", "使用当前证据准备项目讲解", "优先准备 SFT/LoRA 和 exposure eval 追问"],
        )
    if result.verdict == Verdict.WEAK:
        return ActionSuggestion(
            next_action="apply_after_upgrade",
            summary="可以投递，但建议先补 1-2 个证据点。",
            actions=["补齐训练配置说明", "整理 3 个 bad case", "准备 DPO/GRPO 基础问答"],
        )
    if result.verdict == Verdict.RISKY:
        return ActionSuggestion(
            next_action="upgrade_first",
            summary="当前面试被问穿风险较高，先补证据再投递。",
            actions=["补一个最小 LoRA 复现实验", "保存 loss 日志", "写出 eval 对比表"],
        )
    return ActionSuggestion(
        next_action="choose_another_job",
        summary="不建议当前投递，优先回到岗位漏斗选择更匹配岗位。",
        actions=["回到岗位列表", "选择 must-have 更贴近已有项目的 JD"],
    )
