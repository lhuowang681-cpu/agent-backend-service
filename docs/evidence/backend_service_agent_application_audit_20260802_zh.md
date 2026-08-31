# Backend Service Agent 应用方向审计

日期：2026-08-02  
方式：三路独立只读审查 + 主审交叉核对  
结论：`CONDITIONAL PASS`

更新：2026-08-02 已完成 Phase A1–A5、Phase B Application Assistant Mock 纵向闭环、Phase C
确定性 Agent 评测、Phase D 脱敏 provenance，以及最终大规模 HTTP/Worker/故障矩阵。真实 DeepSeek
仍只做小规模 Smoke/Canary，不参与压力测试。
以下原始 finding 保留用于审计，状态以各条的“当前状态”为准。

当前相关验证：Backend/Agent/LLM/Application focused `82 passed`；Snapshot、Manifest、Tool Ledger、
Approval restart、Tool crash-to-UNCERTAIN 与 24 场景评测均通过。当前全量为
`283 passed / 19 failed / 2 warnings`，19 项为既有冻结 Evidence CRLF/hash 环境失败。

## 1. 定位结论

当前实现有真实的异步控制面、PostgreSQL 幂等/Outbox、Redis 至少一次唤醒、Worker lease/fencing、
Approval/Checkpoint Adapter、真实 Provider Canary 和分层性能证据，不是包装空壳。

准确定位是：

> 可靠性导向的 Agent Backend 学习与验证平台。

尚不能称为生产级多租户 Agent SaaS。当前已用现有 Domain AgentLoop、Approval、Checkpoint 和一个
sandbox 非幂等 Tool 闭合 Mock controller 路径；下一轮最高价值是建立 Agent Evaluation，而不是继续
堆中间件。

## 2. 已确认强项

| 能力 | 证据 |
| --- | --- |
| HTTP 202 与长任务解耦 | `tests/test_backend_service.py` 的异步创建测试 |
| user-scoped durable idempotency | PostgreSQL `(user_id, idempotency_key)` 唯一约束、request hash、并发测试 |
| transactional Outbox | Run/Event/Outbox 同事务创建 |
| at-least-once + execution ownership | Redis Stream、DB claim、lease、heartbeat、generation、fencing |
| Approval/Checkpoint 复用 | `DomainAgentExecutionAdapter` 与真实 AgentLoop 集成测试 |
| 性能口径 | HTTP、Worker、Mock、DeepSeek Canary 分开报告 |
| 调试闭环 | 第七阶段由 p99/PEL 定位 delayed Outbox，并以 ready-time filter 修复 |
| Agent provenance | owner-scoped debug 投影关联 Manifest/Attempt/Approval/Checkpoint/Tool operation/Event，且不返回 private payload |

这些能力可以作为 Agent 应用岗位的可靠性底座，但必须使用“已部署”“集成测试已证明”“未证明”三种
措辞区分证据等级。

## 3. Confirmed findings

### P1-1：同 Session claim 存在 TOCTOU

当前状态：`RESOLVED`

`PostgresRunRepository.claim()` 只锁当前 Run，再普通查询同 Session 的 `RUNNING` Run。两个事务可分别
锁不同 Run，并同时看到 Session 空闲。

影响：共享对话历史、Memory 或 Session Artifact 可能并发修改。

计划映射：Phase A1。

修补证据：`pg_advisory_xact_lock(hashtextextended(user_id + session_id))` 串行化同 Session
claim；同 Session 并发与不同 Session 并发测试均通过。

### P1-2：Checkpoint 写入缺少 Attempt fencing

当前状态：`RESOLVED`

Checkpoint save/delete 只携带 user/run，不验证 active `attempt_id + lease_token`。旧 Worker 失去 lease
后仍可能覆盖新 Attempt 的恢复点。

计划映射：Phase A2。

修补证据：Checkpoint save/delete 必须携带 `ClaimedRun`，并在同一 PostgreSQL 事务调用
`_require_active_attempt()`；旧 Attempt 覆盖和删除测试均被拒绝。

### P1-3：非空 backlog 下 Redis-loss reconcile 可能饥饿

当前状态：`RESOLVED`

Recovery 只在整个 Stream lag 与 PEL 都为空时重建缺失唤醒。持续负载下，某个 Redis 消息丢失的
QUEUED Run 可能长期不被扫描。

计划映射：Phase A3。

修补证据：Recovery 每轮执行有界 reconcile；grace period 后为当前 generation 补发 wake-up，
并刷新下一检查窗口。重复消息由 DB claim fence，且不会因 generation 前移让 backlog 中的原消息失效。
测试在保留另一条 backlog 时删除目标消息，目标 Run 仍重新进入 Stream。

### P1-4：Provider permit TTL 不覆盖长 Run

当前状态：`RESOLVED FOR DEPLOYED SEMANTIC PROVIDER PATH`

Provider permit 使用固定 TTL，Run lease 有 heartbeat，但 permit 不续租。执行时间超过 TTL 后，旧调用
尚未结束，新 Worker 已可能获取新 permit。

更根本的问题是 permit 包住整次 Agent Run，却被命名为 Provider 并发限制。

计划映射：Phase A4，优先改为每次真实 Provider call admission。

修补证据：`AdmissionControlledProvider` 在每次出站调用前 acquire，调用后立即记录并 release；
permit TTL 配置必须大于 Provider timeout，因此不再覆盖整个长 Run。

### P1-5：部署态 Approval/UNCERTAIN 纵向链路未闭合

当前状态：`RESOLVED FOR APPLICATION_ASSISTANT MOCK WORKER PATH`

当前 `worker_main` 只装配 `SemanticJobExecutionAdapter`，公开任务合同也只允许
`semantic_job_flow`。Approval、DomainAgent 和 Tool once 主要由测试装配证明，Compose 请求无法真实进入
`WAITING_APPROVAL` 或 Tool receipt 驱动的 `UNCERTAIN`。

计划映射：Phase B1–B5。

修补证据：公开合同已允许 `application_assistant_flow`，`worker_main` 按持久化 `task_type` 路由到
`DomainAgentExecutionAdapter`。自动化集成从 Repository/Outbox/Redis Worker 进入
`WAITING_APPROVAL`，审批后由新 Repository 与新 Worker 恢复到 `SUCCEEDED`。本轮没有执行手工
HTTP/Compose 调用，因此不能把该证据表述成已完成真实 API E2E。

### P1-6：非幂等 Tool crash receipt 尚未证明

当前状态：`RESOLVED FOR SANDBOX DRAFT TOOL`

现有 `UNCERTAIN` 测试会注入已经分类好的错误，但没有完整模拟“请求已发出、外部系统可能执行、receipt
未落库、Worker 崩溃”。在 DomainAgent 部署前，不能把 model-only expired attempt replay 直接推广到
非幂等 Tool。

计划映射：Phase B4–B5。

修补证据：`LedgeredToolExecutor` 在 Tool 前写 `INFLIGHT`，成功后写 observation receipt。故障测试在
sandbox 文件已生成、receipt 未落库处崩溃；lease 恢复后 Agent Checkpoint 阻止再次调用，Run 与
operation 在同一事务进入 `UNCERTAIN`，文件数仍为 1。该结论不能外推为任意第三方 API exactly-once。

### P2-1：本地 Agent 错误可能污染 Provider circuit

当前状态：`RESOLVED FOR DEPLOYED SEMANTIC PROVIDER PATH`

credential、input、checkpoint、runtime 等非 Provider 错误可能进入全局 Provider failure counter，导致
坏请求阻断其他用户。

计划映射：Phase A4。

修补证据：只有 `ProviderRateLimitError`、`ProviderTransportError` 和明确
`invalid_response` 在 Provider wrapper 内更新 cooldown/circuit；本地异常只释放 permit。

### P2-2：`per_user_limit=1` 被过度解释

当前状态：`RESOLVED`

它是未经证明的保守容量默认值，不是“同一用户必须串行”或“用户必须绑定 Worker”的业务语义。
共享 Redis Stream + 多消费者仍是当前正确默认方案。

计划映射：Phase A4；同时发布第七阶段解释更正。

修补证据：默认 `provider_per_user_limit` 不再固定为 1，而是可部署配置；共享 Stream 与任意
消费者 claim 的模型未改变。

### P2-3：真实 Career 数据路径尚未接入

当前状态：`RESOLVED FOR VERSIONED APPLICATION SNAPSHOT PATH`

Worker 为受控用户注入相同硬编码简历，证明了 owner-scoped Run 结构，但没有证明多用户 CareerSnapshot
读取和版本冻结。

计划映射：Phase B2。

修补证据：Application Run 必须绑定 PostgreSQL 中 owner-scoped、内容寻址且不可变的
`career_snapshot_revision`；不同用户不能解析同一 revision，Worker 重启仍读取 Run Manifest 冻结的
版本。现有 Semantic tracer 的硬编码 resume 未改，且尚无公开 Snapshot 写入 API，因此不能声称已经把
桌面 SQLite CareerStore 直接改成共享生产 CareerStore。

### P2-4：Agent 评测不足

当前状态：`RESOLVED FOR APPLICATION_ASSISTANT MOCK PATH`

现有 Evidence 评测很强，Backend 可靠性测试也较完整，但缺少 Tool selection、Approval correctness、
resume replay、duplicate side-effect、grounding 和 cost 的统一 Agent 任务评测。

计划映射：Phase C。

修补证据：`application-assistant-evaluator-v1` 固化 24 条 synthetic 场景，直接执行现有 AgentLoop、
Tool、Policy、Approval 和 Checkpoint。它聚合 task success、Tool precision/recall、参数 Schema、
Approval recall、Grounding、resume replay、duplicate side effect、UNCERTAIN classification、调用次数与
本地延迟。结果为 24/24，但只代表确定性 Mock 合同回归；真实模型开放输入质量仍未证明。

### P2-5：API 错误率只覆盖正常负载

当前状态：`RESOLVED FOR CONTROLLED MOCK LOAD ENVIRONMENT`

第五阶段记录了正常 POST 负载的 0 错误率，也有 HTTP counter 和功能性 4xx/409 测试，但没有系统
验证 PostgreSQL 不可用、Redis 不可用、Worker 全停、Queue 超限和恢复期间的 5xx/拒绝率。API 错误、
Run 失败和 Provider 错误尚未形成分层 SLI。

计划映射：Phase A5。

已完成：脱敏 503、Redis down 仍 202、queue cap 429、Provider 429/timeout、幂等/409/401/404
故障矩阵；Prometheus counter 已区分 availability 5xx 与 expected rejection 4xx。

负载证据：400 请求混合矩阵为 0 unexpected、0% availability error；Worker 全停时 40/40 请求仍
202 且恢复后全部成功；Redis 停机修复 best-effort 指标阻塞后 40/40 继续 202，liveness 50 ms，
Outbox 恢复后清零；PostgreSQL 停机时修复连接池等待后 20/20 返回脱敏 `503 + Retry-After`，
无客户端 timeout；queue cap 为 20×202、4×429 且拒绝请求不创建 Run。

矩阵还发现并修复 Mock Provider 在 Checkpoint resume 后的 fixture Schema 错位；单用户 4 Worker、
9 次 admission deferral 后 50/50 成功，随后 1,838 Run 无新增 `schema_error`。完整延迟、资源、恢复
时间和限制见 `docs/performance/backend_service_final_fault_matrix_20260802_zh.md`。

## 4. 传统后端类非阻塞项

以下问题真实存在，但对当前 Agent 应用岗位不是最高优先级：

- unsigned Header identity，不是生产登录；
- Stream/Outbox/Event/Checkpoint retention；
- readiness、Worker heartbeat、结构化日志；
- Migration checksum、Docker non-root、dependency lock；
- CI 中依赖缺失时集成测试会 skip；
- 全量测试存在 19 个冻结 Evidence CRLF/hash 环境失败。

这些项目进入后续基础工程 Backlog，不应先于 Phase A–C。

## 5. 文档与主张审计

必须更正：第七阶段证明的是“Worker 并发高于配置 permit 时产生 deferral 与写放大”，不能推导为：

- 同一用户必须固定交给同一 Worker；
- 当前需要 tenant-aware shard/dispatcher；
- `per_user_limit=1` 是多用户 Agent 的语义合同。

必须保留：delayed Outbox 过早进入 PEL 的根因、ready-time filter 修复、p99 改善和持久 deferral 指标，
这些都有代码、数据库和复测证据。

## 6. 审计门禁

Phase B 完成后的证据门禁：

- 可以说“Application Assistant Mock Worker 的 Approval/Checkpoint/AgentLoop 部署代码路径已通过自动化集成测试”；
- 不可以说“本轮已手工完成 HTTP/Compose E2E 或真实 DeepSeek Application Assistant 调用”；
- 可以说“已接入的 sandbox draft Tool 在 receipt 丢失窗口会停止重放并进入 UNCERTAIN”；
- 不可以说“任意非幂等 Tool 或第三方系统已实现 exactly-once”；
- 可以说“实现 owner-scoped 多用户隔离 seam”；
- 不可以说“已实现生产身份或共享多租户 CareerStore”。

## 7. 计划决策

审计结果全部映射到 `docs/tech-specs/agent_application_optimization_plan_zh.md`。Phase D 已完成；
大规模自动化矩阵和 A5 混合故障负载已经完成。下一步只应在明确面试收益或部署需求时选择：补
readiness/Worker heartbeat 与 retention，或扩展真实 Agent 任务评测；真实 Provider 继续只保留低并发
Canary，不做压力测试。当前不启动 Kafka、tenant shard 或 Kubernetes。
