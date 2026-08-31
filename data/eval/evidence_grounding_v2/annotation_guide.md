# Evidence Grounding V2 标注指南

版本：`v1`
状态：`frozen_for_pilot_schema`
适用范围：Pilot 的标注演练与人工复核，不代表 heldout 已冻结。

## 1. 标注对象

最小标注单元是：

```text
一个 JD requirement + 同一份简历
```

标注者需要给出：

1. `gold_level`：`C0 / C1 / C2 / C3 / None`；
2. 一个或多个能支持该等级的简历原文 quote；
3. 简短判定理由；
4. case slice。

不能根据常识补全候选人的经历，也不能把 JD 文本、模型解释或面试话术当作简历证据。

## 2. 等级判定

| 等级 | 可操作定义 | 可以怎么说 | 不能怎么说 |
| --- | --- | --- | --- |
| `C3` | 原文同时包含明确个人行动、可定位产物/职责，以及可验证结果或指标 | “我实现并评测过，产物/结果是……” | 把不相关的数字当成本 requirement 成果 |
| `C2` | 原文包含明确个人行动和具体产物/职责，但缺少可验证结果指标 | “我实现/负责过……” | 声称达到未记录的业务或模型效果 |
| `C1` | 原文只证明学习、复现、辅助参与或有限 exposure | “我学习/复现/参与过……” | 声称独立设计、生产落地或拥有完整模块 |
| `C0` | 只有关键词、邻近概念或无法归因的弱相关信息 | “我接触过相关概念” | 将关键词匹配升级为项目能力 |
| `None` | 没有支持证据，或原文明示没有做过 | “简历中没有证据” | 用相似项目或未来计划代替证据 |

等级是“当前简历允许的主张强度”，不是候选人的真实能力上限。

## 3. 高频分歧

### C0 与 C1

- “了解 RAG 概念”是 C0；
- “完成 RAG 课程并复现检索 demo”是 C1；
- 单独出现技术名词但没有学习动作时，不得给 C1。

### C1 与 C2

- “协助整理评测数据”通常是 C1；
- “设计评测集并实现 evaluator”是 C2；
- “复现开源项目”默认 C1，除非原文还清楚证明了自主设计或新增产物。

### C2 与 C3

- “实现 checkpoint 恢复模块”是 C2；
- “实现 checkpoint 恢复模块，故障恢复测试 20/20 通过”是 C3；
- 数字必须属于该 requirement 的行动和产物，不能从同一段的另一项工作借用。

### None 与负向表述

“未做过生产部署”“计划学习”“尚未实现”都不能产生正向证据。即使句子中含有
“部署”“RAG”“GRPO”等关键词，也标为 `None`。

## 4. Span 规则

1. quote 必须逐字来自 `resume_text`；
2. quote 应是支持判定所需的最小完整片段；
3. 不要只截取技术名词，必须尽量保留行动、产物和结果；
4. 多个不连续片段可以标多个 spans；
5. 同一 quote 在简历中出现多次时，人工冻结前必须补齐 offset；
6. Python 字符 offset 使用半开区间 `[start_char, end_char)`；
7. `None` 不得包含 supporting span；
8. AI 草拟数据可以暂缺 offset，但 quote 必须在该简历中唯一出现；冻结为 human gold
   前必须物化并人工复核 offset。

## 5. Slice 判定

- `explicit_strong`：明确行动、产物、结果；
- `explicit_action`：明确行动和产物但无结果；
- `learning_only`：课程、学习、复现、有限参与；
- `keyword_only`：只有相关名词；
- `missing`：完全无证据；
- `negation`：原文明示未做过；
- `numeric_distractor`：有数字但属于其他任务；
- `cross_requirement`：同段包含多个能力，容易错配；
- `sparse_resume`：材料极少；
- `latex_like`：包含 LaTeX 或转义文本。

## 6. 双遍复核流程

第一遍只看 requirement 和简历，独立标 level 与 spans。第二遍至少间隔一次工作会话，
打乱 unit 顺序后重新标注。分歧按以下顺序解决：

1. quote 是否属于原文；
2. 行动是否能归因给候选人；
3. 产物是否与 requirement 对应；
4. 指标是否属于该产物；
5. 选择满足事实边界的最低充分等级。

达到以下条件后才允许把 `review_status` 改为 `frozen`：

- 两遍 level 一致，或分歧已记录并裁决；
- span offset 已物化并复核；
- `label_provenance=human_gold`；
- source-group split 和文件 hash 已重新生成；
- heldout 未被任何 Prompt 调优过程查看。

## 7. Pilot 的事实边界

当前 `pilot.jsonl` 由 AI 按本指南草拟，因此每条数据固定为：

```text
label_provenance = ai_draft
review_status = pending_human_review
```

其指标只用于验证 schema、runner、membership guard、指标实现和错误切片。不得在简历、
README 或面试中表述为“人工 benchmark 提升”或“heldout 实验结论”。
