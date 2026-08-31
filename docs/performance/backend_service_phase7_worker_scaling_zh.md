# Backend Service 第七阶段：多 Worker 扩容与公平性报告

日期：2026-08-02

## 1. 结论

> 解释更正（2026-08-02 审计后）：本实验中的 `per_user cap=1` 是保守配置，不是同一用户必须串行、
> 绑定 Worker 或使用 tenant shard 的业务合同。数据只证明 Worker 并发高于当时配置 permit 时会产生
> deferral 与写放大；共享 Redis Stream + 多消费者仍是当前默认设计。
>
> 最终矩阵更新：普通 load 配置已改为 per-user cap=2；cap=1 只保留为背压/恢复故障注入。最新
> 1/2/4 Worker、Checkpoint resume 与组件停机结果见
> `backend_service_final_fault_matrix_20260802_zh.md`。

多 Worker 执行能力有效，但 Worker 数不能脱离 Provider permit 和用户调度独立扩张。在本次两个活跃
用户、per-user cap=1 的 Mock 工作负载下：

- 2 Worker 是有效池大小，修复后达到 38.99 runs/s；
- 4 Worker 因同用户 permit 碰撞产生 688 次 admission deferral，下降到 30.01 runs/s；
- 两个用户最终各成功 100 Run，没有观察到饥饿；
- 扩容实验发现并修复了 delayed Outbox 过早进入 Redis PEL 导致的约 30 秒尾延迟；
- 本报告只代表可控 Mock、当前机器和短批次，不代表真实 LLM 吞吐。

## 2. 实验合同

- 数据库：独立 `job_agent_load`。
- Redis：独立 `job-agent:load-*` Stream/group/metrics namespace。
- 工作负载：每批 200 个 `semantic_job_flow`，两个用户轮询，各 100。
- Provider：Mock，每个成功 Run 固定 2 次调用，不访问网络。
- 每轮先停止 Worker 形成 backlog，再以 1/2/4 副本排空。
- Run 通过唯一 batch 的 Idempotency-Key 查询；不清理或混入旧批次。
- throughput window：最早 Attempt start 到最晚 Attempt finish。
- queue wait：Run create 到首次 Attempt start。
- E2E：Run create 到 terminal finish。
- deferral：`recovery_kind=provider_admission_deferred` 的持久 Attempt。

## 3. 首轮扩容基线

默认 Provider global cap=4、per-user cap=1：

| Workers | Success | Throughput | Attempts | Deferrals | E2E p50 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 200/200 | 17.84/s | 200 | 0 | 8.69 s | 12.98 s | 13.36 s | 13.45 s |
| 2 | 200/200 | 4.79/s | 280 | 80 | 6.81 s | 9.96 s | 44.13 s | 44.88 s |
| 4 | 200/200 | 4.70/s | 736 | 536 | 9.12 s | 45.19 s | 45.78 s | 45.81 s |

这个结果不能解释成“Python 多进程更慢”：2/4 Worker 的绝大多数 Attempt 本身仍在几十毫秒完成，
但少数 Run 接近 45 秒。数据库审计确认所有额外 Attempt 的错误都是
`provider_user_backpressure`。

## 4. 根因：Delayed Outbox 过早发布

Provider permit 拒绝后，Repository 会把 Run 重新置为 `QUEUED`，设置未来的 `next_attempt_at`，
并在同一事务插入下一 generation 的 Outbox。这部分是正确的。

错误在 Publisher 查询：它只检查 `published_at IS NULL`，没有检查 Run 是否已经到
`next_attempt_at`。于是延迟消息立即进入 Redis；Worker 领取时数据库拒绝 claim，消息没有 ACK，
留在 PEL，直到约 30 秒后 `XAUTOCLAIM`。这解释了 p99/max 的阶跃尾巴。

修复后查询同时要求：

```sql
o.published_at IS NULL AND r.next_attempt_at <= now()
```

PostgreSQL 继续保存延迟唤醒的权威状态，Redis 只接收已经可执行的消息。没有引入新的延迟队列或
中间件。

## 5. 修复后复测

| Workers | Success | Throughput | Attempts | Deferrals | E2E p50 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 200/200 | 38.99/s | 298 | 98 | 4.58 s | 6.55 s | 7.30 s | 7.75 s |
| 4 | 200/200 | 30.01/s | 888 | 688 | 5.51 s | 7.70 s | 8.33 s | 9.30 s |

约 30 秒尾巴已经消失。4 Worker 仍不如 2 Worker，因为只有两个用户 permit：额外 Worker 反复
claim、写 Attempt/Event/Outbox 后被 admission 拒绝，形成 4.44 attempts/Run 的写放大。

## 6. 扩容能力对照

为排除 Worker 实现本身无法并行，只在 Mock 实验中把 per-user cap 临时调到 2：

| Workers | per-user cap | Throughput | Deferrals | E2E p95/max |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 34.33/s | 0 | 8.39 / 8.50 s |
| 4 | 2 | 5.45/s | 64 | 4.69 / 39.28 s |

该结果证明两个 Worker 可以并行排空。它不构成放宽生产限流的依据：真实 Provider 配额、成本和
noisy-neighbor 隔离都要求保留 per-user cap。

## 7. 多用户公平性

所有批次均为两个用户各提交 100、各成功 100、retry=0。修复后的用户 queue-wait p95：

| Workers | user A | user B | 最大完成前缀偏斜 |
| ---: | ---: | ---: | ---: |
| 2 | 4.971 s | 4.978 s | 4 Runs |
| 4 | 3.205 s | 3.205 s | 5 Runs |

因此没有观察到最终饥饿或明显用户尾延迟差异。但公平性不能只看最终成功数：4 Worker 的 688 次
deferral 是调度低效和数据库写放大的证据。

## 8. 新增可观测合同

```text
job_agent_provider_admission_deferrals_total{reason="provider_user_backpressure"}
job_agent_worker_tasks_total{outcome="backpressured"}
```

前者来自 PostgreSQL 持久 Attempt，重启后仍可审计；后者来自 Redis 短期 Worker 指标。Reason 使用
固定 allowlist，未知值归一为 `other`。部署态复核时，Load 数据库累计 1,466 次历史 deferral；重建
后两组 Worker 短期指标记录 786 次，分别对应 98 + 688。

## 9. 扩容决策

当前不能使用“queue depth 高就无限增加 Worker”的规则。对本次固定配置，可用下式解释测量结果：

```text
desired_worker_concurrency
  <= min(global_provider_limit, sum(active_user_available_permits))
```

它不是通用 autoscaling 公式。实际部署必须按 Provider/profile 配额、真实调用粒度和活跃 Run 测量。
同时观察 deferral rate、Attempt/Run 比、数据库写入、queue depth 和 E2E tail。对于本实验的两个活跃
用户和 `per_user cap=1`，2 Worker 只是该批次的较优配置，不是多用户架构上限。

当前不引入 tenant shard 或公平 dispatcher。只有未来在合理 permit 配置下仍持续观察到配额碰撞与
饥饿，才重新评估更复杂调度；不能从本次 `cap=1` 实验直接推出该需求。

## 10. 修改文件

- `compose.backend.yml`：移除共享 Worker ID。
- `compose.backend.load.yml`：只用于实验的 Provider cap override。
- `postgres_repository.py`：Outbox ready-time filter 和持久 deferral aggregate。
- `reliable_worker.py`：记录 backpressured Worker outcome。
- `metrics.py`：导出 admission deferral counter。
- `scripts/seed_backend_load.py`：多用户、唯一 batch 种子。
- `scripts/report_backend_load.py`：Worker/用户/Attempt/延迟报告。
- `tests/test_backend_service.py`、`tests/test_backend_service_persistence.py`：报告与延迟 Outbox 回归。

## 11. 未证明

- 没有真实模型并发调用，也没有产生额外真实 API 费用。
- 没有超过两个用户，因此未建立用户数量到最佳 Worker 数的完整曲线。
- 没有长时间 soak、超过两个用户或热点用户分布。
- Mock Run 很短，绝对吞吐不能外推到真实 Agent；本实验最可信的是状态、deferral 和尾延迟关系。

## 12. 验证结果

```text
LLM + Backend focused: 61 passed
Related regression excluding frozen Evidence modules: 260 passed, 2 warnings
Full regression: 262 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
compileall: PASS
pip check: PASS
git diff --check: PASS
broad exception / unbounded-loop scan: no matches
Docker Compose API/Worker rebuild: PASS
```

本阶段共执行七个精确 200-Run 批次，合计 1,400/1,400 `SUCCEEDED`、0 retry。全量回归的
19 个失败仍全部属于冻结 Evidence CRLF/byte-hash 差异，没有修改其合同或数据。

## 13. 批次索引

| Purpose | Batch ID |
| --- | --- |
| 1 Worker baseline | `phase7-w1-ac194eae` |
| 2 Worker before fix | `phase7-w2-d96b4de3` |
| 4 Worker before fix | `phase7-w4-cf25d00a` |
| 2 Worker / cap=2 control | `phase7-w2cap2-33b0a799` |
| 4 Worker / cap=2 diagnostic | `phase7-w4cap2-7068585f` |
| 2 Worker after fix | `phase7fix-w2-05d2168c` |
| 4 Worker after fix | `phase7fix-w4-eeac42b3` |

这些批次只包含合成职位与受控 demo 用户；数据库卷保留以便复核，没有导出用户私人内容。
