# Evidence Grounding 标注指南 v1.1 修订说明

状态：`adjudicated_for_next_pilot_candidate`
基础版本：`annotation_guide.md` v1
来源：2026-07-26 人工抽检第 5 条

## 修订原因

v1 把“课程学习和最小复现”默认归为 C1。人工抽检裁决认为：

```text
完成课程，并成功复现一个可运行的 GRPO 最小训练示例
```

可以算有限的 hands-on practice，因此标为 C2。

## v1.1 的 C1/C2 分界

### C1

- 只阅读课程、论文或文档；
- 只理解流程，没有执行结果；
- 复制代码但没有成功运行；
- 只观察他人完成的训练；
- 无法展示代码、配置、日志或输出等复现产物。

### C2

- 成功运行课程或开源项目中的最小训练示例；
- 能展示对应代码、配置、日志、checkpoint 或输出中的至少一项；
- 可以准确说明复现步骤、输入输出和已知限制。

C2 只表示“做过有限实践”，不允许升级成：

- 独立设计 GRPO 算法；
- 自主构造完整训练数据；
- 完成系统消融或取得泛化提升；
- 具有生产后训练经验。

### C3

除 C2 的可运行产物外，还必须有直接对应、可核验的结果或指标，例如完整实验对照、
任务效果、稳定性结果或明确规模验证。仅“成功跑通 demo”不自动成为 C3。

## 适用范围

本修订只用于构建下一版 Pilot candidate。已经完成真实 API 调用的
`pilot.jsonl`、原 dataset hash 和 Pilot 报告保持不变，以保证实验可复现。

下一版应用的标签 overlay：

```text
pilot_human_spot_check_overlay.json
```

由于目前只有 6/30 units 经过人工抽检，整个数据集仍是
`ai_draft / pending_human_review`，不能标记为 `human_gold / frozen`。
