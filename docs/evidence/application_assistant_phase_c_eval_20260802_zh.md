# Application Assistant Phase C 确定性评测报告

日期：2026-08-02  
评测器：`application-assistant-evaluator-v1`  
Provider：确定性 Mock controller（未调用真实 API）

## 1. 评测目标

本评测回答 Agent 应用问题，不测传统 HTTP 吞吐：

- Agent 是否只选择预期 Tool，且参数符合 Registry Schema；
- sandbox write 是否必经 Approval，审批前是否零副作用；
- Result 是否绑定 Run 的职位与 CareerSnapshot，是否泄漏 `resume_text`；
- Worker/Loop 恢复时是否跳过已完成只读节点；
- 非幂等 Tool 已执行但 receipt 未返回时，是否停止重放并分类为 `UNCERTAIN`；
- 一次场景消耗多少 Mock 模型决策和逻辑 Tool 调用。

## 2. 场景集

`data/eval/application_assistant_phase_c_cases.json` 固化 24 条脱敏场景：

| 类型 | 数量 | 预期 |
| --- | ---: | --- |
| 正常 Approval + Checkpoint resume | 20 | `SUCCEEDED` |
| Tool 已执行、receipt 窗口崩溃 | 4 | `UNCERTAIN` |
| 含不可信简历指令标记 | 5（与上面重叠） | 标记不进入草稿或公开 Result |

场景覆盖 `agent_algorithm`、`llm_application`、`backend_data` 三类岗位，以及 email/form 两种
sandbox draft channel。数据是 synthetic fixture，不代表真实用户分布。

## 3. 运行合同

每条正常场景实际执行：

```text
ApplicationAssistantMockModel
-> existing AgentLoop
-> career.read_snapshot
-> Policy requires Approval for application.write_sandbox_draft
-> JsonCheckpointStore pause
-> new AgentLoop resume with digest-bound ApprovalDecision
-> sandbox draft -> verifier -> COMPLETED
```

故障场景在 write Tool 返回 observation 前注入崩溃。Tool 文件已经生成，但 Checkpoint 仍保留
`inflight_non_idempotent_key`；新的 AgentLoop 恢复后必须返回
`non_idempotent_execution_uncertain`，不能再次执行 Tool。PostgreSQL Run/Tool Ledger 同步进入
`UNCERTAIN` 的部署态证据由
`tests/test_backend_service_reliability.py::test_application_tool_crash_recovers_to_run_and_ledger_uncertain_without_replay`
单独覆盖。

## 4. 结果

运行命令：

```powershell
$env:PYTHONPATH='.;src'
python scripts/evaluate_application_assistant.py `
  --output-dir output/backend_service/application_evaluation/phase_c_20260802_r2
```

| 指标 | 结果 |
| --- | ---: |
| 场景通过 | 24 / 24 |
| 正常任务成功率 | 1.000 |
| Tool selection precision / recall | 1.000 / 1.000 |
| Tool 参数 Schema-valid rate | 1.000 |
| Approval-required recall | 1.000 |
| 审批前副作用数 | 0 |
| CareerSnapshot grounding pass rate | 1.000 |
| Checkpoint resume success rate | 1.000 |
| 已完成只读节点重放数 | 0 |
| duplicate side-effect rate | 0.000 |
| `UNCERTAIN` classification accuracy | 1.000（4 / 4） |
| Mock model decisions | 68 |
| 逻辑 Tool calls | 48 |
| 本机评测延迟 p50 / p95 / p99 | 37.85 / 47.54 / 50.82 ms |

## 5. 解释边界

- 1.000 是 24 条确定性 fixture 上的合同通过率，不是真实用户准确率、招聘效果或 SLA；
- 延迟只包含本机 Mock AgentLoop、Checkpoint 和 sandbox 文件 I/O，不是 HTTP 控制面、Redis Worker、
  DeepSeek API 或真实模型延迟；
- Mock controller 的 Tool 选择是确定性的，这组评测主要用于检测合同和恢复回归，不能证明真实模型
  在开放输入上的 Tool 选择能力；
- 4 条故障场景证明 AgentLoop 的不确定结果分类；PostgreSQL Ledger 原子收敛由独立集成测试证明；
- sandbox 文件不是邮件、表单提交或真实外部副作用；第三方 API 仍需要 operation adapter、查询接口和
  人工 reconciliation 设计；
- 本轮未调用真实 API。真实 Application Assistant Canary 和 HTTP/Compose 手工验证按计划最后执行。

## 6. 代码与复现位置

| 位置 | 作用 |
| --- | --- |
| `src/job_agent/backend_service/application_evaluation.py` | 场景合同、真实 Loop 执行、故障注入和指标聚合 |
| `data/eval/application_assistant_phase_c_cases.json` | 24 条脱敏、版本化场景 |
| `scripts/evaluate_application_assistant.py` | 离线复现入口和 JSON 报告输出 |
| `tests/test_application_assistant_evaluation.py` | 覆盖门禁和 24 场景回归 |
| `src/job_agent/backend_service/application_assistant.py` | 被评测的 Mock controller、Registry、Verifier 与 Loop 装配 |
| `tests/test_backend_service_reliability.py` | PostgreSQL/Redis 部署态 Approval、restart、Ledger 和 UNCERTAIN 证据 |
