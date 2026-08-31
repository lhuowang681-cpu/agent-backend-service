# Reliability Fault Injection Evidence Pack

日期：2026-07-28
范围：Job Agent 本地 Reliability 合同
机器结果：
[`reliability_fault_injection_results_20260728.json`](reliability_fault_injection_results_20260728.json)

## 结论

本证据包执行了 11 个 case：

- `10 PASS`
- `0 FAIL`
- `1 NOT_PROVEN`
- `82 / 82` 个 applicable assertions 成立
- `10 / 10` 个 applicable cases 得到证明

这里的 `case_proof_rate = 1.0` 和 `assertion_proof_rate = 1.0`
只描述确定性 fixture 合同，不是生产故障概率、SLA、真实招聘效果或 LLM
语义质量指标。

唯一 `NOT_PROVEN` 是：
`format_repair_budget_action_integration`。仓库已经有独立
`BudgetManager.record_format_repair` 合同，但当前 Semantic Agent / Domain Agent
action runtime 没有把实际 schema/format repair 次数传入这个预算计数器。本轮没有
为了得到全绿结果而改写架构、放宽 Guard 或伪造 checkpoint 集成。

## 执行记录

- Evidence Pack 自动化：`17 passed in 1.40s`
- Reliability 相关回归：`51 passed, 2 warnings in 8.41s`
- 全量工程回归：`630 passed, 2 warnings in 46.13s`
- 机器 JSON SHA-256：
  `7CAC7AF682FA0D051DBAFB6E53563B4BF7F0E61CB9E746018022F4D4F775FA4A`
- 两个 warning 都是既有 protobuf / Python 3.14 deprecation warning。

## 审计边界

- 根目录没有 `AGENTS.md`；已记录该缺口，并按仓库实际存在的 `CLAUDE.md`
  与 `docs/evidence/current_status.md` 执行。
- 全部 case 使用 `pytest`、mock provider、fixture、`tmp_path`/临时目录和
  synthetic secret。
- `real_api_executed = false`。
- `credential_files_read = false`。
- 没有读取 credential 文件，没有调用真实 API，也没有网络依赖。
- 没有修改现有 Guard、budget 默认值或 production action 架构。
- `ProviderTimeoutError` / `ProviderRateLimitError` 两个 UI case 在 operation
  边界注入，证明错误分类和 last-known-good containment，不声称完成了真实网络
  端到端验证。
- `unhandled_traceback = false` 的结论只覆盖本证据包执行到的组件/UI action
  边界。
- secret leak 检查使用合成 Key 和合成本地路径；没有拿真实 credential 作为
  扫描词。

## 故障矩阵

| Case | Action | 注入层 | 终态 / error_code | 结果 |
| --- | --- | --- | --- | --- |
| `resume_provider_timeout` | `regenerate_resume` | provider error at operation boundary | `failed / timeout` | PASS |
| `resume_schema_error_destructive_rollback` | `regenerate_resume` | partial write + delete + Schema error | `failed / schema_error` | PASS |
| `resume_semantic_guard_rejection` | `regenerate_resume` | semantic Guard | `failed / ungrounded_high_risk_fact` | PASS |
| `interview_prep_provider_rate_limit` | `regenerate_interview_prep` | provider error at operation boundary | `failed / rate_limit` | PASS |
| `interview_prep_atomic_batch_commit_error` | `regenerate_interview_prep` | second `os.replace` | `failed / artifact_write_error` | PASS |
| `mock_evaluation_schema_repair_exhausted` | `evaluate_mock_interview` | `LLMHarness` schema repair | `failed / invalid_json` | PASS |
| `mock_evaluation_guard_repair_exhausted` | `evaluate_mock_interview` | deterministic semantic Guard | `failed / unknown_evidence_id` | PASS |
| `mock_interview_model_budget_resume` | `continue_mock_interview` | model-call budget | `BUDGET_EXCEEDED / model_call_budget_exceeded` | PASS |
| `mock_interview_tool_budget_resume` | `continue_mock_interview` | tool-call budget | `BUDGET_EXCEEDED / tool_call_budget_exceeded` | PASS |
| `format_repair_budget_manager_contract` | `budget_contract` | standalone `BudgetManager` | `BUDGET_EXCEEDED / format_repair_budget_exceeded` | PASS |
| `format_repair_budget_action_integration` | `evaluate_mock_interview` | action integration | `NOT_PROVEN` | NOT_PROVEN |

完整的逐 artifact before/after SHA-256、assertion、budget usage、N/A 和
NOT_PROVEN 字段见机器 JSON。

## 已证明的 Reliability 合同

### Last-known-good 与原子恢复

9 个包含受保护 artifact 的 applicable cases 都记录了 before/after SHA-256，
故障终止后的映射完全一致。

其中 Schema destructive case 在 operation 中：

1. 覆盖 `06_targeted_resume.md` 为半成品；
2. 删除 `07_interview_grilling.md`；
3. 抛出 `LLMInvocationError("schema_error")`；
4. 验证两个文件与 `session_state.json` 的字节和 SHA-256 完全恢复；
5. 解除故障后重试成功。

Atomic batch case 在第二次 `os.replace` 抛出 `OSError`，并额外验证
`atomic_write_text_batch` 在外层 UI wrapper 介入前已经恢复旧文件。

### Schema 与 Guard repair 有界

- Mock Evaluation 连续两次返回 invalid JSON：
  `attempt_count = 2`、`schema_repair_count = 1`，随后以 `invalid_json`
  失败，旧 evaluation JSON/Markdown、run status、debrief 和
  `session_state.json` 保持不变。
- Mock Evaluation 连续两次返回 schema-valid 但引用未知 evidence ID 的结果：
  两次 provider attempt 后以 `unknown_evidence_id` 失败；
  `schema_repair_count = 0`，说明失败来自 deterministic Guard，而不是把 Guard
  失败改写成 Schema 成功。
- 两类失败解除后都能重新生成 evaluation，并将 run status 更新为
  `ai_complete`。

### Budget exceeded、用户等待和 checkpoint

Model-call 与 tool-call 两个 Mock Interview case 都执行了相同恢复协议：

1. Agent 先进入 `WAITING_FOR_USER` 并保存 checkpoint；
2. synthetic active time 先推进 3 秒；
3. 用户等待推进 100 秒；
4. 提交一份回答后触发对应 budget exceeded；
5. checkpoint 中仍存在完整回答，`active_seconds` 仍为 `3.0`；
6. 保持 `BudgetProfile.EVAL` 不变，只调高对应 limit；
7. 从 checkpoint 继续，回答进入 recovery model context；
8. run 完成后才删除 checkpoint。

因此本证据包证明的是：

- `2 / 2` action-integrated budget cases 保留用户回答与 checkpoint；
- `2 / 2` 的 100 秒用户等待没有计入 active wall time；
- `2 / 2` 在同 profile 调高 limit 后恢复成功。

Standalone format-repair budget case也验证了固定 limit、稳定 error code 和提高
limit 后继续；但它没有 action checkpoint 集成，因此不能用来填补
`format_repair_budget_action_integration = NOT_PROVEN`。

### Traceback 与诊断泄漏

- 所有 10 个 applicable cases 都没有未处理 traceback。
- 7 个产生 UI diagnostic 的 failure cases 中，合成 Key、合成本地私有路径和
  raw/parsed provider 内容均未出现在 user message 或 diagnostic。
- 这只能证明当前自动化覆盖到的 sanitized boundary，不能外推为全仓库
  credential 安全审计。

## 可复制命令

仅运行本证据包测试：

```powershell
python -m pytest tests/test_reliability_fault_injection.py -q
```

重新生成机器 JSON：

```powershell
python scripts/run_reliability_fault_injection.py `
  --output docs/evidence/reliability_fault_injection_results_20260728.json
```

运行相关既有回归：

```powershell
python -m pytest `
  tests/test_agent_budgets.py `
  tests/test_agent_checkpoint.py `
  tests/test_atomic_io.py `
  tests/test_ui_live_action.py `
  tests/test_ui_workbench_journey.py `
  tests/test_reliability_fault_injection.py -q
```

全量工程回归：

```powershell
python -m pytest -q
```

以上命令默认不读取 credential，也不执行真实 API。

## STAR 故障故事草稿 1：Schema 失败与 artifact 回滚

**Situation**

Job Agent 的 targeted resume 和 interview prep 会更新多个下游 artifact。如果模型
返回 Schema error，或者 commit 在中途失败，用户可能看到半成品，甚至丢失上一成功
版本。

**Task**

我要证明失败不会污染 last-known-good artifact，并且不能只比较“文件存在”，而要
比较原始字节和 SHA-256。

**Action**

我复用了 UI live-action snapshot、`atomic_write_text_batch`、mock exception 和临时
目录。一个 case 先覆盖目标简历、删除面试准备文件，再抛出 `schema_error`；另一个
case 在第二次 `os.replace` 注入 `OSError`。测试同时记录 protected artifacts 与
`session_state.json` 的 before/after SHA-256，并在解除故障后执行 recovery retry。

**Result**

这两个 case 都得到 PASS：失败后的 SHA-256 与上一成功版本完全一致，分别返回稳定的
`schema_error` 和 `artifact_write_error`，没有未处理 traceback；解除故障后的重试
成功。这个结果证明本地事务边界的 fault containment，不证明生产分布式事务或 SLA。

## STAR 故障故事草稿 2：Budget exceeded 后 checkpoint 恢复

**Situation**

逐题 Mock Interview 会在 `WAITING_FOR_USER` 时保存 checkpoint。风险是用户回答刚提交，
Agent 就耗尽 model/tool budget，导致回答丢失，或者把用户思考时间错误算入 active
wall time。

**Task**

我要证明 model-call 和 tool-call 两类 budget exceeded 都能保留回答和 checkpoint，
并能在不更换 budget profile 的前提下恢复。

**Action**

我用 FakeClock 先累计 3 秒 active time，再模拟 100 秒用户等待；提交回答后分别触发
`model_call_budget_exceeded` 和 `tool_call_budget_exceeded`。随后读取 checkpoint
确认回答存在、`active_seconds` 仍为 3.0，再保持 `BudgetProfile.EVAL` 不变，只提高
对应 limit 并 resume。

**Result**

`2 / 2` case 都以预期 `BUDGET_EXCEEDED` 终止，回答和 checkpoint 均保留，100 秒等待
没有计入 active wall time；提高 limit 后 `2 / 2` 都从 checkpoint 完成，回答进入
recovery model context，checkpoint 只在完成后删除。该结果不包含真实 API 延迟或
生产进程崩溃恢复。

## 未证明与下一步

`NOT_PROVEN`：

- action runtime 中的真实 schema/format repair 次数尚未接入
  `AgentBudget.format_repairs` 与 checkpoint。

本证据包不证明：

- LLM evaluation 准确率；
- Evidence Mapping 语义质量；
- 真实 provider 可用性；
- 并发写入、进程强杀、分布式事务、HA 或 SLA；
- 生产事故率。

后续 Evidence Mapping Evaluation 应作为独立 Semantic Quality Evidence Pack，
不能用本报告的 Reliability proof rate 代替模型指标。
