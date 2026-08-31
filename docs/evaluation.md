# Evaluation / 评测口径

本项目把工程合同、模型行为和业务结果分开报告。测试通过表示当前 fixture 与运行环境中的
合同得到验证，不等价于 LLM 语义准确率、招聘效果、并发能力或生产 SLA。

## 2026-08-01 可复现结果

| 层级 | 结果 | 解释 |
| --- | ---: | --- |
| 公开仓库全量回归 | `244 passed, 2 warnings` | 当前公开代码、CareerStore、Evidence v2、动态面试与 Runtime tests |
| 开发工作区升级相关回归 | `135 passed` | 合并后与本次升级 source commit 相关的测试 |
| 开发工作区全量回归 | `821 passed / 0 failed / 2 warnings` | native LangGraph execution schema 已修复 |
| Reliability Fault Injection | `10 PASS / 0 FAIL / 1 NOT_PROVEN` | `82/82` applicable assertions |
| Evidence v5 no-network mock | `300/300` variant-case rows | 验证 corpus/runner/contracts，不代表 live 语义质量 |

公开仓库复现命令：

```powershell
python -m pip install -e ".[ui,dev]"
python -m pytest -q
```

测试输出中的 2 个 warning 是 protobuf 在 Python 3.14 的 deprecation warning。

## Evidence v5

v5 使用 20 个新 family、100 个 synthetic cases；按 family 做 60 development / 40
held-out 划分，并在正式 held-out 调用前冻结 corpus hash。

| Campaign | Pipeline completion | Req F1 | Link F1 | Fit | Unsupported claim rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.2 development | 55/60 | 0.952 | 0.602 | 0.700 | 未在发布摘要记录 |
| GLM-5.2 frozen held-out | 38/40 | 0.905 | 0.602 | 0.625 | 15.4% |

指标语义：

- `pipeline completion`：case 完成 requirement → evidence → fit 全链路，不是准确率；
- `Req F1`：requirement extraction 的匹配质量；
- `Link F1`：requirement–evidence link 的匹配质量；
- `Fit`：当前 frozen rubric 下的 fit decision agreement；
- `unsupported claim rate`：最终投影中缺少足够证据支撑的 claim 比例。

数据是 synthetic benchmark，且是一次模型 campaign；结果不覆盖真实简历泛化、模型运行
方差、人工标注一致性、供应商账单或生产表现。已观测的 v5 held-out 不再用于后续调参或重跑。

## 评测分层

1. **Contract tests**：Schema、非法 ID、quote resolution、provenance cap、tool allowlist、transaction 与 fail-closed。
2. **Migration/transaction tests**：legacy 去重、幂等重跑、未知公司待整理、rollback 与事件一致性。
3. **Fixture/replay E2E**：在 Mock/fixture 输入上验证完整 DAG、CareerStore 和动态 Interview 轨迹。
4. **Fault injection**：注入 API、Schema、Guard、写入和预算失败，检查旧 artifact 与状态不被破坏。
5. **Live semantic campaign**：冻结数据与指标后调用真实模型，单独报告 completion、accuracy-like metrics 与失败 slice。

## LangGraph 兼容修复

历史 15 项失败来自当前 LangGraph 将 `checkpoint_id` 视为保留 channel。修复将 native
execution schema 与应用 checkpoint metadata 分层，并补齐所有 Semantic DAG 输出 channel；
现已用 `engine_name=langgraph` 的 `28` 个 graph/full-E2E 聚焦测试验证。旧 checkpoint 的
公开 `checkpoint_id` 字段不变，不需要数据迁移。

## 必须同时说明的非主张

- `244 passed` 与 `821 passed` 都是工程回归，不是模型排行榜或 SLA；
- v5 synthetic 结果不是 human-gold benchmark；
- Mock Interview debrief 尚无专家 gold calibration，分数不是录用概率；
- Reliability Pack 有 1 项 `NOT_PROVEN`，不能写成 11/11 全部证明；
- Fit Verdict LoRA adapter 冻结 held-out `0.25` 低于 base `0.50`，quality gate 失败；
- 没有证明真实招聘提升、自动投递可靠性、多租户隔离或线上高并发。

历史 Evidence Grounding 与 fault-injection 报告保留在 [evidence/](evidence/) 供审计，
但旧实验数字不覆盖本页的当前发布口径。
