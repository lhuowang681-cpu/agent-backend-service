# 从零理解 Job Agent 异步 Backend Service

状态：第六阶段已完成 DeepSeek Flash 小规模真实 Canary；这证明适配链路可用，不证明真实模型容量或 SLA，也不把 Mock 数据冒充真实 LLM 结果。

## 1. 为什么需要异步任务

### 直觉

普通 HTTP handler 最适合“很快算完并返回”。Agent Run 可能经过五次模型调用、多个 Tool、审批等待和重试，耗时从秒到小时。让 HTTP 连接一直等待，会把客户端断线、网关超时和 Agent 是否继续执行错误地绑在一起。

### 合同 / 公式

```text
POST Run = 接受命令 + 持久化身份 + 排队
HTTP latency != Agent end-to-end latency

POST /runs -> 202 Accepted + run_id
GET /runs/{run_id} -> 当前权威状态
```

`202` 表示服务器接受了任务，不表示 Agent 已成功。成功只能由后续 `SUCCEEDED` 状态表达。

### 代码

- API：`src/job_agent/backend_service/api.py`
- 应用服务：`src/job_agent/backend_service/service.py`
- 异步 Worker：`src/job_agent/backend_service/worker.py`
- “POST 不等待”测试：`tests/test_backend_service.py::test_post_returns_202_without_waiting_for_worker_completion`

## 2. Idempotency Key

### 直觉

客户端可能因为超时不知道 POST 是否成功，于是重发。如果每次重发都创建 Run，同一份简历可能被重复处理、重复调用模型，未来甚至重复执行 Tool。

### 合同 / 公式

```text
request_hash = SHA-256(canonical_request_json)
idempotency_identity = (user_id, Idempotency-Key)

same identity + same hash      -> same run_id
same identity + different hash -> 409 conflict
different user + same key      -> different run_id
```

不能只用进程内“最近请求 cache”作为最终保证；第二阶段用内存唯一映射证明语义，第三阶段必须交给 PostgreSQL 唯一约束。

### 代码

- canonical hash：`src/job_agent/backend_service/service.py`
- 原子 create-or-get：`src/job_agent/backend_service/repository.py`
- 重复/冲突/跨用户测试：`tests/test_backend_service.py`

## 3. At-least-once Delivery

### 直觉

可靠队列宁愿把一条消息再送一次，也不能在 Worker 崩溃时永远丢掉。于是消费者必须假设“我看到的消息可能以前见过”。

### 合同 / 公式

```text
delivery_count >= 1
message delivery != permission to execute

execute only if:
  persisted owner matches message.user_id
  status == QUEUED
  dispatch_generation matches
  claim succeeds
```

至少一次投递不等于至少一次副作用。消息可重复，业务执行必须靠数据库 claim、generation、checkpoint 和 Tool 幂等收敛。

### 代码

- 第二阶段 envelope/claim：`contracts.py`、`repository.py`
- 第二阶段进程内队列：`queueing.py`
- 第三阶段实现：`redis_streams.py` 使用 consumer group、PEL、`XACK`、`XAUTOCLAIM`；PostgreSQL claim 仍决定是否执行。
- Phase B 非幂等边界：`tool_operations.py` 用稳定 operation ID 和持久化 receipt 收敛副作用；
  `AgentLoop` Checkpoint 在无 receipt 的崩溃窗口阻止重放。这里的正确结果是 `UNCERTAIN`，不是猜测成功或失败。

## 4. Outbox

### 直觉

如果先写数据库再发队列，进程可能在两者之间崩溃：Run 已是 `QUEUED`，消息却没有。反过来先发消息，Worker 可能找不到 Run。

### 合同 / 公式

```text
DB transaction:
  INSERT run
  INSERT first_event
  INSERT outbox_message
COMMIT

publisher: outbox -> broker
```

Outbox 不提供 Tool exactly-once；它只关闭“业务事务成功但唤醒消息丢失”的 dual-write 窗口。

### 代码

- 第二阶段已知窗口：`RunService.create_run()` 先写内存 Repository、后 publish。
- 第三阶段实现：`postgres_repository.py` 在同一事务创建 `runs + run_events + outbox_messages`，`outbox.py` 独立发布。

## 5. Backpressure

### 直觉

队列不是无限仓库。到达速度长期大于处理速度时，增加 Worker 只会把压力转移给 Provider，最终形成 429、超时和重试风暴。

### 合同 / 公式

```text
global queue cap
per-user queue/in-flight cap
global provider concurrency cap
per-user RPM / TPM / retry budget

capacity exceeded -> 429 + Retry-After
```

多用户系统必须防 noisy neighbor：用户 A 的大量任务不能把用户 B 永久饿死。

### 代码

- 第二阶段：只实现 user-scoped ownership，没有声称容量治理。
- Phase A 加固：`RunService + PostgresRunRepository` 在创建事务内执行 per-user/global QUEUED
  admission，超限返回 `429 + Retry-After` 且不创建 Run；幂等重复先返回原 Run。
- `AdmissionControlledProvider` 把 Redis permit 限定在一次真实 Provider 调用，而不是整个
  Agent Run。per-user 数值是部署容量参数，不是“同一用户只能串行”的业务语义。
- Redis 在 acquire 前不可用时 fail closed，不绕过限流直接打 Provider；调用完成后的
  release/circuit 记录是 best effort，permit 仍由 TTL 有界回收。

## 6. Timeout / Retry / Circuit Breaker

### 直觉

- Timeout：一次调用最多等多久。
- Retry：某类暂时失败能否再试，以及最多试几次。
- Circuit Breaker：供应商明显异常时，停止所有 Worker 同时撞墙。

三者解决不同问题，不能用一个无限重试循环替代。

### 合同 / 公式

```text
delay = min(cap, base * 2^attempt) * jitter

retry only if:
  failure is classified retryable
  retry budget remains
  operation is idempotent or has no uncertain side effect
```

429、transport timeout 可能重试；Schema 错误、Approval digest mismatch、Checkpoint identity mismatch 不应盲重试；非幂等 Tool 结果不确定进入 `UNCERTAIN`。

### 代码

- 现有 Provider 分类：`src/job_agent/llm/harness.py`、`provider.py`
- 第二阶段 Adapter 只捕获已知合同失败并映射安全错误码：`backend_service/execution.py`
- `provider_admission.py` 已实现用户级 429 cooldown 与 Provider 级 circuit。只有
  timeout/transport/明确协议故障影响 circuit；本地 input、Checkpoint、credential 错误不计入。
- `failures.py` 实现 Run attempt 的有界 retry budget；`tool_operations.py` 已为接入的 sandbox
  非幂等 Tool 实现 operation ledger。它不自动覆盖其他 Tool，更不代表第三方系统 exactly-once。

## 7. 数据库事务

### 直觉

一次状态变化往往同时影响 Run、Event、Attempt、Approval 和 Outbox。如果只成功一半，GET、Worker 和恢复器会看到互相矛盾的世界。

### 合同 / 公式

```text
atomic transition:
  compare status_version / fencing token
  update run
  append event
  maybe insert outbox
  commit all or rollback all
```

数据库连接在 COMMIT 附近断开时，不能直接重放非幂等操作；先按稳定 operation ID 查询事务是否已经提交。

### 代码

- 第二阶段：`InMemoryRunRepository` 用 `RLock` 原子证明状态合同。
- 第三阶段：`postgres_repository.py` 已实现 PostgreSQL transaction、唯一约束、lease/fencing；
  租约时间统一由数据库计算。
- Phase A 加固：Checkpoint save/delete 也必须在同一事务验证 active Attempt 的
  `attempt_id + lease_token + lease_expires_at`，防止旧 Worker 覆盖新恢复点。
- Phase B 加固：`mark_uncertain()` 在一个事务中同时提交 Run、Attempt、Event 和该 Run 的
  `INFLIGHT -> UNCERTAIN` Tool operation，避免 API 已显示不确定但 Ledger 仍误报执行中的分裂状态。

## 8. Checkpoint 与业务状态的区别

### 直觉

业务状态回答“用户现在看到什么”：`RUNNING`、`WAITING_APPROVAL`、`FAILED`。Checkpoint 回答“Agent 内部已可靠完成到哪一步、恢复时能否跳过”。二者不能互相冒充。

### 合同 / 公式

```text
Run status      = service control-plane truth
Agent checkpoint = resumable runtime execution truth

WAITING_APPROVAL requires:
  durable Run status
  durable pending Approval
  compatible Agent checkpoint
```

只有 `RUNNING` 没有 checkpoint，不能证明某个 Tool 没执行；只有 checkpoint 没有业务状态，API 也不知道应不应该继续调度。

### 代码

- 现有 semantic checkpoint：`src/job_agent/runtime/semantic_checkpoint.py`
- 现有 domain checkpoint：`src/job_agent/agent_runtime/checkpoint.py`
- 第二阶段服务只保存业务状态；第三阶段 `checkpoint_adapter.py` 将现有 Domain/Semantic Checkpoint Schema 持久化为 user/run-scoped private JSONB，恢复验证仍由现有 Runtime 完成。
- Phase B 的 `application_assistant.py` 证明二者如何协作：审批暂停时 Run 是
  `WAITING_APPROVAL`，Checkpoint 保存 pending action；Tool 执行中崩溃时 Checkpoint 保存
  inflight key，服务恢复后据此把 Run 和 Ledger 收敛到 `UNCERTAIN`。

## 9. Little's Law

### 直觉

长任务即使每秒只有少量新请求，也可能积累大量在途 Run。HTTP QPS 低不等于系统压力低。

### 合同 / 公式

```text
L = lambda * W

L      = 平均在途任务数
lambda = 平均到达率
W      = 平均端到端耗时
```

例如每分钟 10 个 Run、平均耗时 30 分钟：平均约有 `10 * 30 = 300` 个 Run 在途。容量报告必须同时展示 arrival rate、queue depth、in-flight 和 Run latency。

### 代码

- 第二阶段尚未做指标与压测。
- 第五阶段已区分 HTTP 控制面、Worker、Mock Provider 和真实 Provider 吞吐；Phase A 又把
  HTTP availability 5xx、expected rejection 4xx、Run failure 和 Provider error 分开定义。

## 10. p50 / p95 / p99

### 直觉

平均值会掩盖尾部慢请求。p95 表示 95% 的样本不超过该值，剩余 5% 更慢；p99 更关注极端尾部。

### 合同 / 公式

```text
p50 = 中位数
p95 = 排序后第 95 百分位
p99 = 排序后第 99 百分位
```

必须分别统计 HTTP latency、queue wait、Provider latency 和 Run end-to-end latency，不能把其中一个数字冒充整个系统性能。

### 代码

- 第五阶段已使用 Locust + 可控 Mock Provider，并报告 p50/p95/p99、错误率与资源使用。
- 最终矩阵已把正常 HTTP、混合预期拒绝、Worker 停止、Redis 停机、PostgreSQL 停机和 queue cap
  分开报告。`503` 是数据库不可用时的 availability failure；`401/404/409/429` 是本实验预期拒绝，
  不能合并成一个错误率。
- 故障实验发现 metrics 虽是 best-effort，若在 event loop 内同步连接 Redis，仍会反向阻塞业务；
  因此指标记录必须移出响应关键路径，并使用短 timeout 与本地 cooldown。

## 11. Agent 长任务与传统 Web 请求

### 直觉

传统 CRUD 请求通常是一次数据库事务；Agent Run 是一个跨模型、Tool、审批和恢复点的长生命周期状态机。

### 合同 / 公式

| 维度 | 普通 Web 请求 | 长耗时 Agent Run |
| --- | --- | --- |
| 生命周期 | 毫秒到秒 | 秒到小时/天 |
| 客户端连接 | 通常覆盖整个处理 | 只负责创建/查询 |
| 中断 | 请求失败即可 | Worker 可死但 Run 必须可恢复 |
| 副作用 | 常在单事务内 | 可能跨外部 Tool，结果会不确定 |
| 人工等待 | 很少 | Approval 可长期暂停 |
| 幂等 | HTTP/DB 层为主 | HTTP、消息、Run、Tool 多层 |

### 代码

- 状态合同：`backend_service/contracts.py`
- HTTP/Worker 分离：`api.py`、`worker.py`
- 未来 Approval/Checkpoint 恢复复用现有 `agent_runtime`，不另建平行语义。

## 12. 当前代码地图

| 文件 | 必须理解的核心问题 |
| --- | --- |
| `contracts.py` | 哪些字段公开，哪些内部字段不能泄漏 |
| `service.py` | 为什么只有首次 create-or-get 才发布消息 |
| `repository.py` | 为什么每个用户查询都必须先带 `user_id` |
| `queueing.py` | 为什么它只是 Tracer，不能假装 Redis 可靠性 |
| `worker.py` | 为什么消息到达不等于获得执行权 |
| `execution.py` | 如何复用现有 Runtime，为什么结果要做安全投影 |
| `api.py` | 为什么身份不能从 JSON body 获取 |
| `tests/test_backend_service.py` | 哪些多用户、幂等和异步合同已经被机器证明 |
| `migrations/001_backend_service.sql` | 哪些唯一约束、复合外键和状态检查由数据库强制 |
| `postgres_repository.py` | Run/Event/Attempt/Approval/Outbox 如何在事务中保持一致 |
| `execution.py` 的 `DomainAgentExecutionAdapter` | 如何把已有 `AgentLoop`、Checkpoint 和 Approval 接到服务层，而不重新实现 Tool/Policy |
| `failures.py` | 哪些错误允许有界重试，哪些失败关闭，哪些进入 `UNCERTAIN` |
| `provider_admission.py` | 为什么 permit 必须绑定一次 Provider call，以及 429/local/transport 错误如何分类 |
| `recovery.py` | 非空 backlog 下如何按 grace window 补 wake-up，并由 DB claim 过滤重复 |
| `reliable_worker.py` | lease/fencing、审批暂停、重试分类和 failpoint 的组合边界 |
| `metrics.py` | HTTP/Worker/Provider 短期直方图与 PostgreSQL 业务指标如何汇总为 Prometheus text |
| `redis_streams.py` | 为什么 consumer-group lag、PEL 和 retained entries 是三个不同的 Queue 指标 |
| `load_tests/locustfile.py` | 如何只测 HTTP 控制面而不混入 Worker/LLM 吞吐 |
| `scripts/benchmark_mock_provider.py` | 如何建立 Mock Provider 微基准并明确其外推边界 |
| `docs/performance/backend_service_phase5_report_zh.md` | p50/p95/p99、资源数据、瓶颈和 `NOT_PROVEN` 结论 |
| `src/job_agent/llm/providers/deepseek_compatible.py` | 为什么真实 Provider 必须固定 endpoint/model 边界并保留 Schema 校验 |
| `worker_main.py` 的 `RoutedExecutionAdapter` | 为什么 Provider 选择来自持久化 Run，而不是 Redis 消息或 HTTP 进程内变量 |
| `metrics.py` 的 `job_agent_provider_tokens_total` | 如何只聚合 input/output token 而不泄漏 prompt、response 或用户标识 |
| `docs/performance/backend_service_phase6_deepseek_canary_zh.md` | 真实 Canary 的已测事实、不可追溯项和禁止外推边界 |
| `scripts/seed_backend_load.py` | 如何用两个用户和唯一 batch 构造可审计 backlog，而不是混用旧 Run |
| `scripts/report_backend_load.py` | 如何区分成功数、Attempt 写放大、admission deferral、queue wait 与 E2E |
| `postgres_repository.py::pending_dispatches` | 为什么延迟 Run 到期前不能进入 Redis Stream/Pending Entries List |
| `docs/performance/backend_service_phase7_worker_scaling_zh.md` | 为什么 4 Worker 可能比 2 Worker 慢，以及公平完成不等于高效调度 |
| `redis_streams.py`、`outbox.py` | 为什么 Redis 可重复、可丢失，而数据库仍能恢复唤醒 |
| `reliable_worker.py` | lease、heartbeat、fencing、ACK 的正确顺序 |
| `checkpoint_adapter.py` | 如何复用现有 Checkpoint Schema，并用 active Attempt fencing 拒绝旧 Worker 写入 |
| `tests/test_backend_service_persistence.py` | 真实 PostgreSQL/Redis 下哪些合同已被证明 |
| `career_snapshot.py`、`migrations/002_career_snapshot_manifest.sql` | 多用户 Snapshot 为什么需要 owner scope、内容寻址 revision 与不可变冻结 |
| `execution_manifest.py` | 长 Run 恢复前为什么必须校验 Provider/Prompt/Tool/Policy/Checkpoint 版本 |
| `tool_operations.py`、`migrations/003_tool_operation_ledger.sql` | 非幂等 Tool 的 intent、receipt 和不确定结果如何持久化 |
| `application_assistant.py` | 如何用现有 AgentLoop/Policy/Approval/Checkpoint 实现一个 sandbox-only 纵向任务 |
| `tests/test_backend_service_reliability.py::test_application_tool_crash_recovers_to_run_and_ledger_uncertain_without_replay` | Tool 已执行但 receipt 未写入时，为什么恢复必须停在 UNCERTAIN 且不能重放 |
| `application_evaluation.py` | 如何把 Tool、Approval、Grounding、Checkpoint、重复副作用和 UNCERTAIN 变成可计算 hard checks |
| `data/eval/application_assistant_phase_c_cases.json` | 为什么正常任务与预期故障必须分开标注，避免把安全停止算成普通失败 |
| `docs/evidence/application_assistant_phase_c_eval_20260802_zh.md` | 为什么 24/24 Mock 合同通过不能外推为真实 LLM 准确率或 API 性能 |
| `contracts.py::RunDebugView`、`postgres_repository.py::get_debug_view` | 如何在不暴露 private payload 的前提下关联 Run、Attempt、Approval、Checkpoint、Tool operation 和 Event |
| `docs/evidence/backend_service_phase_d_debug_smoke_20260802_zh.md` | 单请求 Smoke 能证明什么，以及为什么不能替代大规模故障矩阵 |
| `docs/performance/backend_service_final_fault_matrix_20260802_zh.md` | 如何分开报告正常吞吐、预期拒绝、组件停机、恢复时间和未证明边界 |
| `scripts/probe_backend_database_outage.py` | 为什么连接池 checkout timeout 必须小于客户端 timeout，以及如何验证脱敏 503 |
| `metrics.py::RedisMetricsRecorder`、`api.py::record_request_metrics` | 为什么 best-effort 指标也必须隔离出业务响应路径 |
| `execution.py::SchemaMappedMockLLMProvider` | 为什么可恢复 Agent 的 Mock 必须按当前节点 Schema 响应，不能依赖从头消费的顺序队列 |

## 13. 面试追问

1. 为什么创建 Run 返回 `202` 而不是 `200` 或 `201`？
2. `202` 能否证明任务最终一定执行？为什么？
3. 为什么 Idempotency-Key 必须和 `user_id` 组成作用域？
4. 同一个 key 对应不同 payload 时为什么不能直接返回旧 Run？
5. 进程内字典为什么不能提供部署后的最终幂等保证？
6. At-least-once delivery 与 exactly-once side effect 有什么区别？
7. 为什么 Worker 收到 Redis 消息后还必须去 PostgreSQL claim？
8. `dispatch_generation` 如何让旧消息失效？
9. Outbox 解决了什么窗口？它没有解决什么问题？
10. 模型返回后、状态写入前崩溃，为什么通常可以再次调用模型但可能重复计费？
11. 为什么非幂等 Tool timeout 后不能盲目重试？
12. 什么证据会让 Run 进入 `UNCERTAIN`？
13. lease 和 fencing token 分别解决什么问题？
14. Approval 为什么要绑定 session、run、request 和 action digest？
15. Checkpoint 与 Run status 为什么必须分别持久化？
16. Provider retry、Run retry、消息 redelivery 为什么要分层计数？
17. Circuit Breaker open、half-open、closed 各代表什么？
18. Redis 暂时不可用时为什么 Run 真相不能只存在 Redis？
19. PostgreSQL COMMIT 附近连接断开时，为什么不能直接重试整个事务？
20. 如何证明用户 A 无法通过 run_id 枚举用户 B 的任务？
21. 为什么跨用户资源访问返回 404 而不是 403？
22. per-user cap 与 global cap 为什么需要同时存在？
23. 用 Little's Law 如何估算长耗时 Run 的平均在途数量？
24. 为什么 Mock Provider 压测结果不能写成真实 LLM 吞吐？
25. HTTP p99、Provider p99 和 Run p99 为什么是三个不同指标？
26. 如果未来增加共享 workspace，哪些 user-owned 约束需要版本化迁移？
27. 为什么 Worker 数超过“活跃用户数 × per-user permit”后可能只增加 deferral？
28. `next_attempt_at` 在 PostgreSQL 中尚未到期时，为什么 Outbox 不应提前发布到 Redis？
29. Consumer 收到尚不可执行的消息却不 ACK，会如何形成 PEL 尾延迟？
30. 为什么 200/200 成功仍可能掩盖 688 次 admission deferral 和数据库写放大？
31. 为什么不能从 `per_user_limit=1` 的实验直接推出“同用户必须绑定 Worker”或 tenant shard？
32. 为什么 Redis 指标写入已经 catch `RedisError`，仍可能拖慢整个 API event loop？
33. 数据库连接池 checkout timeout、SQL statement timeout 和 HTTP client timeout 分别约束什么？
34. 为什么顺序 Mock response queue 在 Checkpoint 跳过已完成节点后会产生 Schema 错位？
35. PostgreSQL 停机时返回 503 为什么比让所有请求等满 30 秒更符合背压合同？
