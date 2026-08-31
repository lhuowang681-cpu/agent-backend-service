# Job Agent 应用优化计划

状态：`PHASE_A_D_COMPLETE / SMALL_SMOKE_COMPLETE / LARGE_MATRIX_COMPLETE / A5_FAULT_LOAD_COMPLETE`  
目标岗位：Agent 应用开发 / AI Backend / Agent Infrastructure  
日期：2026-08-02

## 1. 目标

下一轮不继续扩建传统 SaaS 能力，而是完成一个可证明的 Agent 应用纵向闭环：

```text
Versioned CareerSnapshot
-> Existing Domain AgentLoop
-> Tool Registry / Policy
-> Approval pause
-> Checkpoint + Worker restart
-> Tool operation receipt
-> Resume without duplicate side effect
-> Agent-level evaluation
```

完成后应能用代码和自动化测试回答：Agent 如何获得上下文、为什么选择某个 Tool、何时必须审批、
崩溃后如何恢复、如何避免重复副作用、如何评估任务是否真的完成。

## 2. 优先级原则

1. Agent 状态与副作用正确性优先于吞吐和中间件。
2. 复用现有 `AgentLoop`、Policy、Approval、Checkpoint、CareerStore，不建立平行实现。
3. 先做一个真实任务纵向闭环，再扩任务类型。
4. Mock 用于确定性故障与评测；真实 Provider 只做小规模 Canary。
5. 不把 `per_user_limit` 等配置参数写成 Agent 业务语义。

## 3. Phase A：恢复与并发合同加固

### A1. Session claim 串行化

问题：当前两个事务可分别锁不同 Run，同时认为同一 Session 没有 `RUNNING` Run。

改造：

- claim 时对 `(user_id, session_id)` 获取 PostgreSQL transaction advisory lock，或使用显式 Session lease 行；
- 保留不同 Session 并行；
- busy Run 必须被有界重新调度，不能固定滞留 PEL。

验收：

- 两线程同时 claim 同一 Session，只有一个成功；
- 同一用户不同 Session 可并行；
- 不同用户同名 Session 可并行。

### A2. Checkpoint fencing

问题：Run 终态提交有 lease fencing，但 Checkpoint save/delete 只校验 user/run，旧 Worker 可能覆盖新恢复点。

改造：

- Checkpoint Store 绑定 `attempt_id + lease_token`；
- save/delete 在同一数据库事务验证 Attempt 仍为 active 且 lease 未过期；
- WAITING_APPROVAL 的最终 Checkpoint 与 Run/Approval 转移仍保持原子合同。

验收：旧 Worker 失去 lease 后不能保存或删除 Checkpoint，新 Attempt 的 Checkpoint 保持不变。

### A3. Redis-loss reconciliation

问题：当前只在整个 Stream lag/PEL 都为零时重建缺失唤醒；持续 backlog 可能让个别 QUEUED Run 长期饥饿。

改造：

- 每轮执行有界 reconcile；
- 只处理超过 grace period、且没有未发布 Outbox 的 ready Run；PostgreSQL 无法精确证明某条
  Redis 消息仍存在，因此选择有界重复 wake-up，而不是伪装 exactly-once；
- generation 继续过滤重复消息。

验收：删除某个 Run 的 Redis 消息，同时保留其他 backlog，该 Run 仍在有界时间内恢复执行。

### A4. Provider admission 与 circuit 语义

问题：当前 permit 包住整次 Agent Run，固定 TTL 不续租；本地输入/Checkpoint/凭据错误也可能污染 Provider circuit。

改造优先方案：

- Provider 并发 permit 在每次真实 Provider call 前获取、调用后释放；
- Run 级并发如仍需要，使用独立命名和独立配额，不能冒充 Provider call concurrency；
- 只有 timeout、transport、5xx、明确 Provider 协议失败计入 circuit；
- 429 进入明确 cooldown，不计为本地 Agent 错误；
- `per_user_limit` 由部署配置决定，不固定解释为 1。

验收：长 Run 超过旧 TTL 不突破 Provider cap；三次本地合同错误不会阻断其他用户的 Provider 调用。

### A5. API 错误率与故障流量合同

问题：第五阶段只证明正常控制面负载下 2,703 个请求为 0 错误；尚未形成 PostgreSQL、Redis、
Worker 故障和过载条件下的 API 错误率报告。当前 HTTP 4xx、Run 失败和 Provider 错误也不能混为
一个“错误率”。

指标必须分层：

```text
API availability error rate
  = unexpected 5xx / valid HTTP requests

API expected rejection rate
  = expected 4xx + 409 + 429 / total HTTP requests

Run failure rate
  = FAILED + UNCERTAIN / terminal Runs

Provider error rate
  = timeout + transport + 429 + provider 5xx / Provider calls
```

改造：

- HTTP 指标区分预期拒绝和意外 5xx，不能把合法的 409/429 当服务崩溃；
- PostgreSQL 不可用或连接池耗尽时返回稳定、脱敏的 `503`；
- Redis 不可用时 Run 创建仍以 PostgreSQL + Outbox 为权威，符合条件时继续返回 `202`；
- Worker 全停时 API 可以接单，但必须用 queue age/depth、Worker heartbeat 或 degraded readiness 表达积压；
- 增加 per-user/global QUEUED admission，超限返回 `429 + Retry-After`，且不创建 Run；
- 使用 Mock Provider 运行正常、混合非法请求、数据库故障、Redis 故障、Worker 停止与恢复场景；
- 报告请求量、预期拒绝率、5xx、p50/p95/p99、恢复时间和资源数据。

验收矩阵：

| 场景 | 预期 |
| --- | --- |
| 正常创建 | `202` |
| 相同 Idempotency-Key 与相同请求 | `202`，相同 `run_id` |
| 相同 Key 与不同请求 | `409` |
| 非法身份 | `401` |
| 跨用户查询 | `404` |
| PostgreSQL 不可用 | `503`，无异常细节泄漏 |
| Redis 暂时不可用 | POST 可 `202`，恢复后 Outbox 唤醒 |
| Worker 全停 | POST 可 `202`，积压/降级信号可见 |
| Queue 超限 | `429 + Retry-After`，不创建 Run |
| Provider 429/timeout | HTTP 查询仍可用，Run 按失败分类转移 |

故障实验必须把“HTTP 控制面可用”与“Agent 最终成功”分开报告，且不得用真实模型做压力测试。

### Phase A 实施记录（2026-08-02）

已完成并由自动化测试证明：

- A1：`claim()` 使用 `(user_id, session_id)` transaction advisory lock；busy Run 生成下一代
  Outbox，不滞留原 PEL；
- A2：Domain/Semantic Checkpoint Store 绑定 `ClaimedRun`，save/delete 在事务内校验 active
  `attempt_id + lease_token + lease_expires_at`；
- A3：Recovery 每轮有界 reconcile，不再依赖全队列为空；grace period 之后为当前 generation
  补一个重复 wake-up，并刷新下一检查窗口。重复消息仍由 PostgreSQL claim fence，且不会让队首旧消息失效；
- A4：Provider permit 下沉到每次 `generate_structured()`；429 只进入 user cooldown，
  transport/timeout/明确协议故障才影响 circuit，本地 Agent 错误只释放 permit；
- A5：运行时 PostgreSQL 故障返回脱敏 `503 + Retry-After`；QUEUED per-user/global admission
  返回 `429 + Retry-After` 且不创建 Run；幂等重复在容量检查前返回原 Run；HTTP 指标分开记录
  availability 5xx 与 expected rejection 4xx。

新增关键测试：

- 旧 Attempt 不能覆盖或删除新 Checkpoint；
- 非空 Redis backlog 中丢失单条消息仍会恢复，且同 generation 不会无限补发；
- Provider permit 是单调用作用域，本地异常不会打开其他用户共享的 circuit；
- queue cap、脱敏 503、Redis 不可用仍 202、Provider 429/timeout 状态转移。

最终矩阵完成：400 请求混合合同矩阵为 0 unexpected、0% availability error、75% 预期拒绝；
Worker 全停后 40 个 Run 保持 QUEUED 并在恢复后全部成功；Redis 停机时修复指标写入阻塞后
40/40 继续 202，Outbox 恢复后清零；PostgreSQL 停机时收敛 1 秒连接池等待后 20/20 返回脱敏
`503 + Retry-After`；显式 queue cap 得到 20×202、4×429 且只创建 20 个 Run/Outbox。

实验同时修复了 Mock Checkpoint resume 的顺序 fixture Schema 错位：单用户 4 Worker、9 次 provider
deferral 后 50/50 成功，之后 1,838 个控制面 Run 无新增 `schema_error`。完整 p50/p95/p99、恢复时间、
资源和未证明边界见 `docs/performance/backend_service_final_fault_matrix_20260802_zh.md`。

最终补充验证：

- Backend/Agent/LLM/Application focused：`82 passed`；
- 工作区全量：`283 passed / 19 failed / 2 warnings`；
- 19 项失败全部是审计前已存在的冻结 Evidence CRLF/hash 不匹配；本轮未修改冻结数据或
  821 项回归口径；
- `compileall`、`pip check`、`docker compose config --quiet`、`git diff --check` 通过。

## 4. Phase B：Domain Agent 应用纵向闭环（Mock Controller）

### B1. 新任务合同

新增一个受控任务类型，例如 `application_assistant_flow`。现有 `semantic_job_flow` 保持兼容。

任务至少包含：

- 读取职位与 CareerSnapshot；
- 规划申请材料；
- 调用一个只读 Tool 和一个需要审批的 sandbox write Tool；
- 输出脱敏、结构化 Result。

### B2. Versioned CareerSnapshot Adapter

- 从现有 CareerStore/Artifact Repository 读取用户拥有的数据；
- Run 创建时绑定不可变 `snapshot_revision`；
- Worker 重启或用户后续修改简历，都不改变运行中 Run 的输入；
- 禁止继续为所有用户注入同一硬编码简历。

### B3. Execution Manifest

Run 持久记录：

```text
provider / model
skill_version / prompt_version
tool_registry_version / policy_version
career_snapshot_revision
checkpoint_schema_version
```

恢复前验证兼容性；不允许一个长 Run 静默混用两套 Prompt、Tool Schema 或 Policy。

### B4. Tool operation ledger

增加 Tool 操作状态合同：

```text
PREPARED -> INFLIGHT -> SUCCEEDED / FAILED / UNCERTAIN
```

关键字段：稳定 `operation_id`、`run_id`、Tool 名称、action digest、request hash、receipt、时间戳。

恢复规则：

- 已有成功 receipt：返回历史结果，不重复执行；
- 明确尚未执行：允许执行；
- 已 INFLIGHT 且没有 receipt：进入 `UNCERTAIN`，禁止盲目重试。

### B5. 部署 Worker 接入现有 DomainAgent

- `worker_main` 按持久化 task/profile 路由到 `DomainAgentExecutionAdapter`；
- sandbox Tool 必须复用现有 Registry、Policy、Approval digest 和 Checkpoint；
- 不执行真实邮件、表单或自动投递。

纵向验收：

```text
POST -> QUEUED -> RUNNING -> WAITING_APPROVAL
restart API/Worker
approve with matching digest/version
-> QUEUED -> RUNNING -> SUCCEEDED
Tool receipt exactly one
```

同时验证 reject、digest mismatch、重复 approve、模型完成后崩溃、Tool INFLIGHT 无 receipt。

### Phase B 实施记录（2026-08-02）

已完成的代码合同：

- B1：新增 `application_assistant_flow` 判别式请求合同；请求绑定职位、`career_snapshot_revision`、
  draft channel/target。当前只允许确定性 `mock` controller，真实 DeepSeek 接入留到最后的低并发 Canary；
- B2：新增 PostgreSQL `career_snapshots` 与 `VersionedCareerSnapshotStore`。revision 由内容 hash 生成，
  `(user_id, snapshot_revision)` 隔离且不可变；Run 创建前验证 owner 与 revision；
- B3：Run/Event/Outbox 创建事务同时冻结 `ExecutionManifest`；Worker 恢复前 fail-closed 校验
  Provider/Model、Skill、Prompt、Tool Registry、Policy、Snapshot 与 Checkpoint Schema 版本；
- B4：新增 `tool_operations` ledger 和 `LedgeredToolExecutor`。非幂等 Tool 在执行前落 `INFLIGHT`，
  成功/确定失败保存完整 observation receipt；恢复发现无 receipt 时不重放。Run 转为 `UNCERTAIN` 时，
  同一事务把该 Run 的 `INFLIGHT` operation 一起转为 `UNCERTAIN`；
- B5：部署 Worker 已按持久化 `task_type` 路由：`semantic_job_flow` 保持原路径，
  `application_assistant_flow` 进入现有 `DomainAgentExecutionAdapter -> AgentLoop`。sandbox Tool 复用
  Registry、Policy、Approval digest、Checkpoint 与 Tool Ledger，工作目录按 `user_id/run_id` 隔离。

自动化纵向证据：

```text
Repository create -> Worker -> WAITING_APPROVAL
-> approve -> new Repository + new Worker -> resume -> SUCCEEDED
-> exactly one sandbox draft + exactly one SUCCEEDED operation receipt
```

另一个故障用例在 Tool 已写 sandbox、receipt 尚未落库时模拟进程崩溃；lease 恢复后 Checkpoint
阻止重放，最终 Run 与 operation 均为 `UNCERTAIN`，sandbox draft 仍只有一个。相关 Backend、
Agent Runtime、LLM 与 Career 回归为 `82 passed`；`compileall` 和 `docker compose config --quiet` 通过。
工作区全量为 `278 passed / 19 failed / 2 warnings`；19 项仍全部来自既有冻结 Evidence
CRLF/hash 不匹配，本阶段未修改冻结数据或 821 项回归口径。

准确边界：这证明的是部署 Worker 代码路径和 PostgreSQL/Redis 自动化集成，不是本轮已手工调用的
HTTP/Compose E2E；CareerSnapshot 目前由内部 Store/import seam 预置，尚无公开写入 API；sandbox draft
不是真实邮件、表单或投递；Ledger 只覆盖已接入的 Tool，不能宣称任意外部系统 exactly-once。

## 5. Phase C：Agent 级评测

建立 20–30 个脱敏、可控场景，至少覆盖：

| 维度 | 指标 |
| --- | --- |
| 任务完成 | task success rate |
| Tool 选择 | precision / recall |
| Tool 参数 | schema-valid rate |
| 审批安全 | approval-required recall、pre-approval side-effect count |
| Grounding | Evidence/CareerSnapshot grounding pass rate |
| 恢复 | checkpoint resume success、completed-node replay count |
| 副作用 | duplicate side-effect rate、UNCERTAIN classification accuracy |
| 成本 | calls/tokens per Run、按节点分布 |
| 延迟 | node latency、Run E2E p50/p95/p99 |

评测默认使用确定性 Mock/fixture；真实 Provider 只跑小规模 Canary，结果与 Mock 分开报告。

### Phase C 实施记录（2026-08-02）

新增 `application-assistant-evaluator-v1` 与 24 条版本化 synthetic 场景：20 条正常 Approval +
Checkpoint resume，4 条 Tool receipt 窗口崩溃，其中 5 条带不可信简历指令标记。评测直接执行与部署
路径相同的 Mock controller、AgentLoop、Registry、Policy、Approval、Checkpoint 和 sandbox Tool；
PostgreSQL Ledger 的原子收敛仍由可靠性集成测试单独证明。

结果：`24/24` 通过；task success、Tool precision/recall、参数 Schema、Approval recall、Grounding、
Checkpoint resume、UNCERTAIN classification 均为 `1.000`；审批前副作用、completed-node replay 和
duplicate side effect 均为 0。总 Mock model decisions 为 68，逻辑 Tool calls 为 48。Mock 不返回 token
usage，因此 token 成本标记为未测，不伪造 0。

本机评测 p50/p95/p99 为 `37.85 / 47.54 / 50.82 ms`，只代表离线 Mock AgentLoop + 文件 I/O，
不是 HTTP、Redis Worker、DeepSeek 或真实模型延迟。完整口径见
`docs/evidence/application_assistant_phase_c_eval_20260802_zh.md`。

Phase C 完成后的 Backend/Agent/LLM/Career 相关回归为 `84 passed`；工作区全量为
`280 passed / 19 failed / 2 warnings`。19 项仍是既有冻结 Evidence CRLF/hash 失败。

## 6. Phase D：Agent 可调试性

- 记录 request/run/attempt/node/tool/operation 关联 ID；
- 保存脱敏的节点状态转移、版本、token、latency、error code；
- 不保存 Chain-of-Thought、API Key、私人简历正文或原始模型响应；
- 能回答“在哪个节点失败、使用哪个 Prompt/Tool 版本、恢复跳过了哪些节点”。

### Phase D 实施记录（2026-08-02）

新增 `GET /api/v1/runs/{run_id}/debug`。PostgreSQL 在 repeatable-read、read-only 事务中关联 Run、
Manifest、Attempt、Approval、Checkpoint metadata、Tool operation 与脱敏 Event。成功路径能显示审批暂停
和恢复的两个 Attempt；Tool receipt 窗口故障能显示：

```text
attempt 1: approval_pause
attempt 2: lease_expired_safe_replay
attempt 3: non_idempotent_execution_uncertain
tool operation: UNCERTAIN, linked to attempt 2
```

投影不返回 request、Snapshot/Checkpoint private payload、Tool arguments/receipt、lease token、内部 Worker
名称、原始 Provider 内容或 Chain-of-Thought；跨用户仍为 404，Event 最多返回最近 500 条并标记截断。

在不运行大矩阵的前提下完成两个 Smoke：一个 Mock HTTP Run `202 -> SUCCEEDED`；一个真实 DeepSeek
Semantic Run `202 -> SUCCEEDED`，1 Attempt、2 Provider calls，Manifest 为
`deepseek_flash / deepseek-chat`。完整边界见
`docs/evidence/backend_service_phase_d_debug_smoke_20260802_zh.md`。

## 7. 非目标

当前不优先实现：Kafka、tenant shard、Kubernetes、登录页面、计费、复杂 RBAC、每用户独立队列、
万级 QPS、真实自动投递或邮件发送。

JWT、readiness、retention、Docker hardening 和 CI 属于基础工程改进，但不得挤占 Phase A–C 的 Agent
正确性与评测工作。

## 8. 阶段顺序与完成定义

```text
Phase A correctness
-> Phase B real Agent vertical slice
-> Phase C evaluation
-> Phase D debugging/provenance
```

每阶段都必须：

1. 展示修改文件与新增合同；
2. 先跑聚焦测试，再跑 Backend 相关回归；
3. 不修改冻结 Evidence 数据与既有评测口径；
4. 更新代码地图和失败边界；
5. 标注哪些能力是部署态 E2E、集成测试或仍未证明；
6. 不自动修改简历、不自动 Push。
