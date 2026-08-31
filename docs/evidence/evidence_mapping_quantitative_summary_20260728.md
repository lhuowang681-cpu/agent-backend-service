# Job Agent 量化证据总表

日期：2026-07-28

用途：面试 Agent 算法开发 / 大模型算法。事实优先级为当前代码与 JSON、当前状态、
本总表、历史报告。

## 1. Reliability Fault Injection

```text
PASS         10
FAIL          0
NOT_PROVEN    1
```

覆盖 provider/API、schema、semantic guard、repair budget、atomic commit，以及
targeted resume、interview prep、mock evaluation。证明失败合同和恢复路径，
不证明生产 SLA。

## 2. Evidence Mapping Pilot Review

```text
AI second review       30/30
human spot-check        6/30
exact agreement          5/6
quadratic weighted κ   0.957
```

6 条不是已证明的随机样本，formal semantic quality 为 `NOT_PROVEN`。

## 3. C-Atomic Pilot

| Metric | C | C_ATOMIC |
| --- | ---: | ---: |
| Exact accuracy | 0.833 | 0.800 |
| Ordinal overprediction | 0.067 | 0.033 |
| Strong precision | 0.909 | 1.000 |
| Strong recall | 0.833 | 0.833 |
| Span exact F1 | 0.429 | 0.488 |

C-Atomic 更保守但总体 accuracy 未提高，因此没有迁移。

## 4. Synthetic Dev120 Post-hoc Comparison

| Metric | C | C_BLOCK | C_ATOMIC |
| --- | ---: | ---: | ---: |
| Exact accuracy | 0.800 | 0.875 | 0.708 |
| Macro F1 | 0.760 | 0.820 | 0.668 |
| Ordinal overprediction | 0.158 | 0.050 | 0.058 |
| Strong precision | 0.966 | 1.000 | 1.000 |
| Strong recall | 1.000 | 1.000 | 0.804 |
| Span exact F1 | 0.220 | 0.687 | 0.742 |

该结果来自 failure-driven post-hoc contract replay，实际 provider invocation
traces `181`，只能用于候选选择。

## 5. C_BLOCK+ATTR 预注册淘汰实验

| Metric | C_BLOCK | C_BLOCK+ATTR |
| --- | ---: | ---: |
| Exact accuracy | 0.917 | 0.817 |
| Macro F1 | 0.877 | 0.768 |
| Ordinal overprediction | 0.025 | 0.117 |
| Strong recall | 1.000 | 0.982 |

```text
accuracy delta       -0.100, 95% CI [-0.175, -0.033]
ordinal upward delta +0.092, 95% CI [+0.033, +0.158]
decision             DO_NOT_ADVANCE
```

Guard 只改变 `1/120`，主要失败信号来自多任务 Prompt 扰动 level judgment。

## 6. Raw-JD Direct Baseline

| Metric | JD_DIRECT | C_BLOCK |
| --- | ---: | ---: |
| End-to-end accuracy | 0.825 | 0.908 |
| Macro F1 | 0.710 | 0.867 |
| Ordinal overprediction | 0.125 | 0.033 |
| Weak→strong unsafe | 2/120 | 2/120 |
| Span exact F1 | 0.630 | 0.583 |
| Schema valid | 40/40 | 40/40 |

```text
accuracy delta       +0.083, 95% CI [+0.025, +0.150]
ordinal upward delta -0.092, 95% CI [-0.150, -0.033]
requirement P/R       1.000 / 1.000
provider traces       80
```

支持 structured pipeline bundle 的 synthetic directional advantage；严格安全
改善、纯 JD 分条因果、真实 JD 泛化均为 `NOT_PROVEN`。

## 7. 不可跨实验直接比较

C_BLOCK 在不同单次运行中的 accuracy：

```text
0.875
0.917
0.908
```

Prompt/协议生命周期和模型运行不同。只能使用同一次 paired run 的 delta 与 CI，
不能挑最高值作为“最终准确率”。

## 8. 面试 Claim Ledger

| Claim | 状态 | 安全说法 |
| --- | --- | --- |
| Fault contracts 可恢复 | PROVEN_BY_TEST | 10 PASS，另有 1 NOT_PROVEN |
| Block pipeline 优于 raw-JD one-shot | SYNTHETIC_DIRECTIONAL | paired accuracy +0.083 |
| Block pipeline 严格安全性更高 | NOT_PROVEN | weak→strong 为 2/120 vs 2/120 |
| C_BLOCK+ATTR 更好 | REJECTED | accuracy -0.100，门禁拒绝 |
| Human-gold semantic accuracy | NOT_PROVEN | 人工 Dev120 复核 0/120 |
| Real-JD generalization | NOT_PROVEN | 当前 raw JD 为 synthetic sidecar |
| Production accuracy/SLA | NOT_PROVEN | 没有生产流量与供应商账单 |

## 9. 当前开放审计项

1. 失败缓存恢复后重试；
2. C_BLOCK protocol 绑定完整 contract hash；
3. extraction audit 绑定当前 repair provenance；
4. 重复模型运行或 frozen heldout；
5. 分层随机人工复核。

## 10. 当前验证

```text
Evidence Grounding focused  100 passed
full repository             708 passed, 2 warnings
credential exact matches      0
```

两条 warning 为既有 protobuf/Python 3.14 deprecation。
