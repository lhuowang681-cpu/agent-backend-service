# Backend Service 最终负载与故障矩阵

日期：2026-08-02  
环境：Windows + Docker Desktop，单 API，PostgreSQL 16，Redis 7，1/2/4 个 Worker  
数据隔离：`job_agent_load` 数据库、`job-agent:load-*` Redis namespace  
Provider：全部批量实验使用可控 Mock；真实 DeepSeek 不参与本报告压力测试

## 1. 结论

最终矩阵完成，并在实验中发现、修复和复测了三个真实问题：

1. Redis 停机时 best-effort 指标写入阻塞 API event loop；修复为响应路径外记录、短 socket timeout
   和进程内冷却熔断。修复后 40/40 创建请求返回 `202`，liveness 为 50 ms。
2. PostgreSQL 停机时连接池默认等待过长；修复为 1 秒 checkout timeout。修复后 20/20 请求返回
   脱敏 `503 + Retry-After: 1`，无客户端 timeout。
3. Mock Run 从 Checkpoint 恢复时，顺序响应队列可能与下一节点 Schema 错位；修复为按
   `output_schema` 选择 fixture。专项背压恢复 50/50 成功，9 次 deferral 后无 `schema_error`。

本报告没有证明生产 SLA、真实 LLM 吞吐、真实身份系统或任意第三方 Tool 的 exactly-once。

## 2. 指标口径

```text
API availability error rate
  = unexpected 5xx / valid HTTP requests

API expected rejection rate
  = expected 401 + 404 + 409 + 429 / total HTTP requests

Run failure rate
  = FAILED + UNCERTAIN / terminal Runs

Worker throughput
  = successful Runs / (last Attempt finish - first Attempt start)
```

HTTP QPS、Worker Run/s、Mock Provider calls/s 和真实 Provider 吞吐不能互相替代。`202` 只表示 Run
已被 PostgreSQL 接收，不表示 Agent 已完成。

## 3. HTTP 控制面

### 3.1 Locust 正常创建基线

配置：10 users、spawn rate 5/s、15 秒、每次唯一 user-scoped Idempotency-Key。

| 请求 | 失败 | 吞吐 | p50 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,838 | 0 | 124.03 req/s | 46 ms | 61 ms | 71 ms | 121 ms |

原始数据：`final_control_plane_fixed_20260802_*.csv`。

负载中一次 `docker stats` 采样：

| 组件 | CPU | 内存 |
| --- | ---: | ---: |
| API | 57.68% | 57.21 MiB |
| PostgreSQL | 57.67% | 85.64 MiB |
| Redis | 7.85% | 12.64 MiB |
| 每个 Worker | 21.63%–24.39% | 75.27–84.37 MiB |

瓶颈首先在 API + PostgreSQL 原子事务路径，不在本地 Mock Provider。每个创建请求都提交 Run、Event
和 Outbox；QUEUED admission 的事务锁也会限制同一控制面实例的扩展。

### 3.2 400 请求混合合同矩阵

`matrix_id=d2e1807ef58a46afadbfc1e7dfa2197f`，concurrency=16。

| 类别 | 数量 | 实际状态 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| valid create | 100 | 100×202 | 100.08 ms | 124.11 ms | 142.76 ms |
| unauthorized | 100 | 100×401 | 20.02 ms | 36.28 ms | 38.05 ms |
| cross-user read | 100 | 100×404 | 38.55 ms | 59.99 ms | 88.10 ms |
| idempotency conflict | 100 | 100×409 | 93.27 ms | 122.31 ms | 126.38 ms |

- 总吞吐：249.58 req/s；
- unexpected response：0；
- availability error rate：0%；
- expected rejection rate：75%。

这里的 75% 是故意构造的合法拒绝，不能写成“服务错误率 75%”。

## 4. Worker 1/2/4 扩展矩阵

每批 200 个 Run，两个用户各 100；Provider global cap=4。该对照批次使用保守的
per-user cap=1 作为碰撞实验，不是业务语义；普通 load 配置已经收敛为 per-user cap=2。

| Worker | 成功 | Run/s | Attempts | Deferrals | queue wait p50/p95/p99 | E2E p50/p95/p99 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 200/200 | 17.27 | 200 | 0 | 16.87/21.20/21.58 s | 16.92/21.25/21.63 s |
| 2 | 200/200 | 30.34 | 200 | 0 | 8.16/10.16/10.37 s | 8.21/10.22/10.42 s |
| 4 | 200/200 | 40.74 | 210 | 10 | 7.06/8.24/8.31 s | 7.20/8.39/9.42 s |

2 Worker 分工为 99/101；4 Worker 分工为 47/49/49/55。两个用户都完成 100，最大完成前缀偏斜
分别为 3、3、7，没有观察到饥饿。共享 Redis Stream 的多个消费者可以领取同一用户的不同 Run；
per-user Provider cap 只限制同时出站调用，不绑定 Worker。

4 Worker 的 10 次 deferral 说明 Worker 扩容接近 Provider cap 后会产生 Attempt/数据库写放大，但本轮
仍提升了短 Mock Run 吞吐。该结论不能外推到长耗时真实 LLM。

## 5. 故障矩阵

### 5.1 Worker 全停

- 停止全部 Worker 后提交 40 个 Run：40/40 返回 `202`；
- 等待 3 秒后数据库仍为 `QUEUED=40`；
- 恢复 4 Worker 后：40/40 `SUCCEEDED`，41 Attempts，1 次可恢复 provider deferral；
- Compose 报告 Worker started 后，终态轮询在 0.544 秒内观察到批次已完成。

最后一个数字不等于严格生产 RTO：容器启动命令与轮询不是同一个高精度计时区间。它只证明持久积压
可恢复，API 接收能力与 Worker 可用性是两个 SLI。

### 5.2 Redis 停机

首轮故障暴露指标写入阻塞问题；15 个已提交 Run 与 15 条 Outbox 都已在 PostgreSQL 保留，没有数据
丢失。修复后重新执行独立批次：

- Redis 完全停止时 40/40 返回 `202`，提交 0.412 秒；
- liveness：200，50 ms；
- 数据库：`QUEUED=40`、`outbox_pending=40`；
- `/metrics`：200、`job_agent_metrics_redis_available 0`，首次探测 3.356 秒；
- 恢复 Redis 和 4 Worker 后：40/40 `SUCCEEDED`、0 retry，Outbox pending 总数回到 0；
- 从 Redis start、Worker rebuild/start 到目标批次终态：7.213 秒；
- 四个 Worker 各完成 10 个 Run。

`/metrics` 首次 Redis 故障探测仍会受 Docker DNS 失败延迟影响，但它运行在独立线程池，不再阻塞
Run API 或 liveness。这是已知的观测面降级，不应伪装为零影响。

### 5.3 PostgreSQL 停机

修复前：20 个并发请求仅 8 个返回正确 503，12 个在 15 秒客户端 timeout。根因是 8 连接池槽位
耗尽后仍使用默认长 checkout wait。

修复后：

| 请求 | 503 | timeout | Retry-After=1 | p50 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 20 | 0 | 20 | 1,030.45 ms | 1,110.29 ms | 1,110.29 ms | 1,112.50 ms |

响应只含 `database_unavailable`，不含连接串或数据库异常详情。数据库恢复后进行有界、每秒一次探测，
第 3 次恢复为 `202`，从启动 PostgreSQL 到成功接收共 5.439 秒。

### 5.4 Queue cap

显式设置 global=20、per-user=10，停止 Worker 后由两个用户各提交 12 个请求：

- 20×202、4×429；
- 两个用户各创建 10 个 Run；
- 只生成 20 个 Run 和 20 条 Outbox；
- 额外探针返回 `429 + Retry-After: 1`；
- 恢复 Worker 后已接收的 20 个 Run 全部成功。

拒绝发生在创建事务内，不存在“返回 429 但暗中创建 Run”的分裂结果。

## 6. Checkpoint + Provider 背压专项

负载中曾出现 5 个历史 `schema_error`：Run 在第一个节点保存 Checkpoint 后因 provider admission
deferral 重新入队，新 Mock Provider 又从第一条顺序 fixture 开始，导致下一节点 Schema 错位。

修复为 `SchemaMappedMockLLMProvider` 后，使用单用户、4 Worker、临时 per-user cap=1 强制制造恢复：

- 50/50 `SUCCEEDED`；
- 59 Attempts、9 次 `provider_user_backpressure`；
- 4 Worker 完成 12/13/12/13；
- 0 个新 `schema_error`；
- 随后的 1,838 个 Locust Run 全部成功，load 数据库 `new_schema_errors=0`。

旧的 5 条失败记录保留为诊断证据，没有清理或改写历史。

## 7. 原始十项可靠性合同

| 合同 | 证据 |
| --- | --- |
| 相同 Idempotency-Key 重复提交 | 并发 API/Repository 测试：相同 payload 返回同一 Run；不同 payload 为 409 |
| Worker claim 前崩溃 | Redis PEL reclaim 自动化测试 |
| 模型完成、状态写入前崩溃 | failpoint + lease recovery 测试；模型调用可重放但不冒充零重复计费 |
| WAITING_APPROVAL 后重启 | 新 Repository/Worker 恢复到 SUCCEEDED 的 Domain Agent 集成测试 |
| Approval digest 不匹配 | owner/session/run/action digest + status version 返回 409 |
| Provider 429 | 有界 retry/cooldown 分类测试 |
| Provider timeout | 有界 retry/circuit 分类测试 |
| Redis 暂时不可用 | 本报告 40 请求停机实验 + Outbox 自动排空 |
| PostgreSQL 事务回滚 | Run/Event/Outbox 原子回滚故障注入测试 |
| 非幂等 Tool 结果不确定 | Tool INFLIGHT 无 receipt 转 UNCERTAIN，sandbox side effect 不重放 |

不能对非幂等 Tool 盲目重试，因为“请求超时”不能证明外部系统没有执行。当前 Ledger 只证明已接入的
sandbox draft Tool；任意第三方邮件、表单或投递系统仍需 reconciliation/API-specific idempotency。

## 8. 修改与验证

关键修改：

- `api.py`：API 指标记录移出 HTTP 响应关键路径；
- `metrics.py`：Redis 短 timeout、本地 cooldown 与非阻塞 probe；
- `postgres_repository.py`：连接池 checkout timeout=1 秒；
- `execution.py`：Mock fixture 按输出 Schema 路由，兼容 Checkpoint resume；
- `compose.backend.load.yml`：普通 Mock load 的 per-user cap 改为 2；
- `seed_backend_load.py`：输出 status counts，支持显式 expected accepted；
- `probe_backend_database_outage.py`：可复现 503、Retry-After、延迟和 transport error 报告；
- `tests/test_backend_service.py`、`tests/test_backend_service_persistence.py`：新增回归合同。

验证结果：

```text
Backend/Agent/LLM/Application focused: 82 passed
Full workspace: 283 passed / 19 failed / 2 warnings
Ordinary Compose final Mock smoke: health 200, QUEUED -> SUCCEEDED
```

19 项失败仍全部属于冻结 Evidence corpus 的 CRLF/byte-hash 差异。本阶段没有修改冻结数据、Manifest、
Evidence 结果或 821 项回归口径。

## 9. 未证明与风险

- 未做长时间 soak、Redis retention/trim、磁盘写满、网络分区或跨主机部署；
- unsigned `X-User-ID` 只是开发身份 seam，不是生产认证；
- 没有 readiness/Worker heartbeat，Worker 停机只能由 queue age/depth 与外部编排识别；
- PostgreSQL 是单实例，没有验证 HA/failover；
- `/metrics` 在首次 Redis DNS 故障时可慢约 3.36 秒，但业务 API 已隔离；
- Mock Provider 极短，绝对 Worker 吞吐不能外推到 DeepSeek；
- 真实 DeepSeek 只完成单任务 Canary，未测真实并发、429 比例、成本上限或 SLA；
- 没有生产级共享 CareerStore 写入 API、计费、RBAC 或第三方 Tool reconciliation。

因此准确表述是：当前项目已经完成一个可部署、可测试、owner-scoped 的多用户 Agent Backend 学习闭环，
并验证了核心异步、幂等、审批、恢复与背压合同；它不是生产级多租户 SaaS，也没有证明真实 LLM 高并发。
