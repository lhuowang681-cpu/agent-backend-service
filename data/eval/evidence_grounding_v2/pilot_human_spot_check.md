# Evidence Grounding Pilot：人工抽检表

状态：已完成 6/30 条人工抽检
二审来源：`ai_second_pass`
事实边界：不是 `human_gold`，不能冻结 heldout。

## 1. 二审汇总

| 结论 | 数量 |
| --- | ---: |
| 保留原标签 | 29 |
| 建议修改 | 1 |
| 高置信度 | 26 |
| 中等置信度、建议重点抽检 | 4 |
| 总计 | 30 |

人工抽检裁决：

| 抽检结果 | 数量 |
| --- | ---: |
| 同意保留原标签 | 4 |
| 确认修改标签 | 2 |
| 尚未逐条人工复核 | 24 |

确认修改：

```text
pilot-tool-005 / req_classifier：C3 → C1
pilot-learning-002 / req_grpo：C1 → C2
```

唯一建议修改：

```text
pilot-tool-005 / req_classifier
C3 → C1
```

理由：原文“另一个分类模型在离线集上达到 91% 准确率”有结果，但没有说明候选人
训练、实现或评估了什么。按严格 attribution，缺少个人行动和责任范围。如果原意是
候选人亲自评估，应先把原文补成“我评估该模型，并在离线集得到 91%”，然后才给 C3。

## 2. 建议你优先抽检的 6 条

你不必从头读 30 条，先检查下面 6 条即可覆盖最主要的分界：

### 抽检 1：责任归属缺失

- ID：`pilot-tool-005 / req_classifier`
- Requirement：具备分类模型评估经验
- 原文：另一个分类模型在离线集上达到 91% 准确率。
- 原标签：C3
- 二审建议：C1
- 你需要决定：是否把无主语的简历 bullet 默认归因给候选人？

建议：不要默认。面试项目强调 truth boundary，缺少动作时保守降级更一致。

### 抽检 2：规模指标能否算 C3

- ID：`pilot-negation-003 / req_data_pipeline`
- Requirement：具备大规模数据处理经验
- 原文：在数据清洗任务中处理了 10 万条记录，准确率检查与 Agent 部署无关。
- 原标签：C3
- 二审建议：保留 C3，中等置信度
- 你需要决定：处理规模是否算“可验证结果或指标”？

建议：算。10 万条与大规模数据处理直接对应；如果你规定 C3 必须是质量或业务效果，
则应统一降为 C2，并同步修改标注指南。

### 抽检 3：基础日志算不算可观测性

- ID：`pilot-rpa-008 / req_observability`
- Requirement：具备失败日志和可观测性实践
- 原文：负责 RPA 报备链路，串联取数、字段映射、校验和填报，并保存失败日志。
- 原标签：C2
- 二审建议：保留 C2，中等置信度
- 你需要决定：保存失败日志是否满足“可观测性实践”？

建议：可以支持基础 C2，但只能说“有失败日志”，不能扩展成 metrics、trace、告警或
成熟 observability platform。

### 抽检 4：共享 span 的归因

- ID：`pilot-posttrain-009 / req_data_generation`
- Requirement：具备训练数据构造经验
- 原文：构建 rule engine、data generator、prompt builder、evaluator 的 SFT/GRPO
  实验闭环，并用 2000 条标注样本完成流程验证。
- 原标签：C3
- 二审建议：保留 C3，中等置信度
- 你需要决定：2000 条样本是否明确由 data generator 产生？

建议：当前语义基本成立；若要做正式 gold，最好把文本改成“data generator 生成并整理
2000 条标注样本”，消除共享 span 歧义。

### 抽检 5：学习与实践分界

- ID：`pilot-learning-002 / req_grpo`
- Requirement：具备 GRPO 后训练实践经验
- 原文：完成了 RLHF 与 GRPO 在线课程，并复现了课程中的最小训练示例。
- 原标签：C1
- 二审建议：保留 C1，高置信度
- 你需要决定：复现最小示例能否算 C2？

建议：不能。C2 会暗示自主实现或具体产物所有权，课程复现应保持 C1。

### 抽检 6：否定与 ownership

- ID：`pilot-ownership-010 / req_agent_ownership`
- Requirement：独立负责 Agent planner 核心模块
- 原文：参与团队的 Agent 原型讨论，负责会议记录；核心 planner 由另一位同学实现。
- 原标签：None
- 二审建议：保留 None，高置信度
- 你需要决定：出现 Agent/planner 关键词能否给 C0？

建议：不能。requirement 要求“独立负责核心模块”，而原文直接否定 ownership。

## 3. 全部 30 条二审结果

| # | Case / Requirement | 原标签 | 二审 | 置信度 | 结论摘要 |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | agent-001 / agent_loop | C3 | C3 | 高 | 独立实现 + checkpoint + 20/20 |
| 2 | agent-001 / python | C2 | C2 | 高 | Python 行动和产物，无结果 |
| 3 | agent-001 / rag | None | None | 高 | 无 RAG 证据 |
| 4 | learning-002 / grpo | C1 | C1 | 高 | 课程与最小复现 |
| 5 | learning-002 / rag | C0 | C0 | 高 | 只了解概念 |
| 6 | learning-002 / deploy | None | None | 高 | 无部署证据 |
| 7 | negation-003 / agent_deploy | None | None | 高 | 明示尚未做过 |
| 8 | negation-003 / kubernetes | None | None | 高 | 只是未来计划 |
| 9 | negation-003 / data_pipeline | C3 | C3 | 中 | 10 万条为直接规模结果 |
| 10 | eval-004 / evaluator | C2 | C2 | 高 | 设计指标和报告，无效果值 |
| 11 | eval-004 / preference_data | C1 | C1 | 高 | 协助整理，ownership 有限 |
| 12 | eval-004 / reward_model | None | None | 高 | 偏好数据不等于 RM 训练 |
| 13 | tool-005 / tool_use | C3 | C3 | 高 | 6 工具 + 48 回归测试 |
| 14 | tool-005 / classifier | C3 | **C1** | 中 | 有 91% 结果但缺个人动作 |
| 15 | tool-005 / tool_accuracy | None | None | 高 | 91% 属于分类模型 |
| 16 | sparse-006 / python | C0 | C0 | 高 | 仅技能关键词 |
| 17 | sparse-006 / langgraph | C0 | C0 | 高 | 仅技能关键词 |
| 18 | sparse-006 / lora | C0 | C0 | 高 | 仅技能关键词 |
| 19 | latex-007 / architecture | C2 | C2 | 高 | 实现完整链路，无结果 |
| 20 | latex-007 / testing | C2 | C2 | 高 | 编写测试，无覆盖或结果 |
| 21 | latex-007 / online_service | None | None | 高 | 无在线稳定性证据 |
| 22 | rpa-008 / workflow | C2 | C2 | 高 | 负责完整自动化链路 |
| 23 | rpa-008 / observability | C2 | C2 | 中 | 仅支持基础失败日志 |
| 24 | rpa-008 / llm_tool_choice | None | None | 高 | 明示未使用 LLM |
| 25 | posttrain-009 / posttrain_loop | C3 | C3 | 高 | 闭环 + 2000 条验证 |
| 26 | posttrain-009 / data_generation | C3 | C3 | 中 | data generator + 2000 条 |
| 27 | posttrain-009 / reward_model | None | None | 高 | 明示未训练 RM |
| 28 | ownership-010 / agent_ownership | None | None | 高 | planner 由他人实现 |
| 29 | ownership-010 / collaboration | C1 | C1 | 高 | 仅讨论和会议记录 |
| 30 | ownership-010 / dpo | C1 | C1 | 高 | 只复现公开 notebook |

## 4. 你如何记录抽检

在下面填写即可，不要直接把数据改成 `human_gold`：

| 抽检 ID | 同意二审？ | 你的最终等级 | 备注 |
| --- | --- | --- | --- |
| tool-005 / classifier | 是 | C1 | 同意二审降级 |
| negation-003 / data_pipeline | 是 | C3 | 认可规模指标 |
| rpa-008 / observability | 是 | C2 | 失败日志满足基础可观测性 |
| posttrain-009 / data_generation | 是 | C3 | data generator 与 2000 条关联明确 |
| learning-002 / grpo | 否 | C2 | 人工裁决：成功最小复现算有限实践 |
| ownership-010 / agent_ownership | 是 | None | 保持 None |

本次结果已记录为 `human_spot_checked_partial`。剩余 24 条仍是 AI 标签，整个 Pilot
仍不能称为 `human_gold / frozen`。正式 heldout 需要人类逐条复核全部 units。

原 Pilot 数据和 hash 不改写；下一版变更记录在
`pilot_human_spot_check_overlay.json`，规则修订见
`annotation_guide_v1_1_amendment.md`。
