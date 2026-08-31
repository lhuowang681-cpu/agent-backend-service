# Backend Service 第五阶段压测报告

日期：2026-08-02

## 1. 结论先行

本轮先测量，没有预设“万级 QPS”。当前单机、单 API、单 Worker、PostgreSQL/Redis Docker Compose 配置下：

| 口径 | 工作负载 | 吞吐 | p50 | p95 | p99 | 错误率 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HTTP 控制面 | 10 Locust users，15 秒，只创建 Run | 192.42 req/s | 15 ms | 30 ms | 38 ms | 0% |
| 单 Worker 完整任务 | 精确 200 个 Semantic Mock Run 排空 | 18.20 runs/s | 49.98 ms/attempt | 60.06 ms | 68.04 ms | 0% |
| Worker 内 Mock 调用 | 400 次 Provider call 随 200 个 Run 执行 | 36.40 calls/s | ≤5 ms | ≤5 ms | ≤5 ms | 0% |
| Mock Provider 微基准 | 50,000 次本地结构化 fixture，concurrency=4 | 28,581.56 calls/s | 0.010 ms | 0.030 ms | 0.048 ms | 0% |
| 真实模型 API | 未调用 | `NOT_PROVEN` | `NOT_PROVEN` | `NOT_PROVEN` | `NOT_PROVEN` | `NOT_PROVEN` |

这些数字不可互换。HTTP 控制面吞吐不代表 Agent 完成速度；Mock Provider 微基准只测本地 JSON/Pydantic fixture 路径，绝不是 LLM 性能。

## 2. 实验配置

- Windows 主机，Docker Desktop；容器报告的可用内存上限约 15.18 GiB。
- API：单 Uvicorn process。
- Worker：单 process，一次消费一个 Redis Stream delivery。
- PostgreSQL 16 Alpine；Redis 7 Alpine。
- Provider：`MockLLMProvider`，不访问网络。
- 控制面压测工具：Locust 2.46.3，位于独立 `.venv-load`。
- HTTP 场景只执行 `POST /api/v1/runs`，每次使用不同的 user-scoped Idempotency-Key。
- Worker 场景先形成 backlog，再启动单 Worker；处理窗口以数据库中 `min(started_at)` 到 `max(finished_at)` 计算，不把容器启动时间算进 Worker 吞吐。

## 3. HTTP 控制面

原始 Locust 摘要：

- 请求：2703；失败：0。
- 平均：16.51 ms；最小：7.01 ms；最大：67.69 ms。
- 吞吐：192.42 req/s。
- p50/p95/p99：15/30/38 ms。

资源采样峰值：

| 组件 | CPU 峰值 | 内存观测峰值 |
| --- | ---: | ---: |
| API | 72.37% | 52.62 MiB |
| PostgreSQL | 31.81% | 79.35 MiB |
| Redis | 4.56% | 10.75 MiB |

瓶颈首先出现在 API/Python 与 PostgreSQL 事务路径，而不是 Redis。这个场景每个 POST 都执行 Run + Event + Outbox 原子事务，并额外写一份短期请求指标。

## 4. Worker 与端到端延迟

- 使用 exact-count seed script 提交 200 个 Run，200 个均返回 `202`；启动 Worker 前数据库确认 `QUEUED=200`。
- 200 个 Run 全部 `SUCCEEDED`，0 retry，0 recovery。
- Worker 数据库处理窗口：10.988 秒，即 18.20 runs/s。
- 每个 fixture Run 走 `not recommended` 路由，因此只发生 2 次 Mock Provider call；总计 400 次。
- Attempt p50/p95/p99：49.98/60.06/68.04 ms。
- Run 端到端 p50/p95/p99：25.64/28.83/29.32 秒。

Run 端到端延迟明显大于 Attempt 延迟，是因为 backlog 在 Worker 启动前已经形成。它展示了排队等待，而不是 Provider 慢。按照 Little's Law，同样的单 Worker 吞吐下继续增加到达率会线性扩大在途 Run 和等待时间。

Worker 负载阶段资源峰值：

| 组件 | CPU 峰值 | 内存观测峰值 |
| --- | ---: | ---: |
| Worker | 54.40% | 77.28 MiB |
| PostgreSQL | 30.58% | 87.38 MiB |
| Redis | 2.61% | 10.93 MiB |

当前瓶颈是串行 Worker 的完整 Agent/数据库状态路径，不是 Mock Provider 自身。增加 Worker 前必须继续保留 per-user Provider cap、session serialization 和数据库 claim fencing；不能只看 CPU 就无条件水平扩容。

## 5. Queue depth 语义修正

指标首次运行时，原 `queue.depth()` 使用 `XLEN`，ACK 后仍显示 251。这不是 backlog，而是 Stream retained entries。第五阶段将其拆分为：

- `job_agent_queue_depth = consumer group lag + PEL pending`：待处理工作量；
- `job_agent_queue_pending`：已经投递、尚未 ACK；
- `job_agent_queue_retained_entries`：Stream 保留的历史 entry 数。

一次 251-entry 诊断批次排空后实测为 `depth=0`、`pending=0`、`retained_entries=251`。该修正也避免 RecoveryCoordinator 因历史 XLEN 非零而跳过 PostgreSQL dispatch reconciliation。

## 6. 可复现命令

```powershell
python -m venv .venv-load
.\.venv-load\Scripts\python.exe -m pip install -r requirements.loadtest.txt

.\.venv-load\Scripts\python.exe -m locust `
  -f load_tests\locustfile.py --headless `
  -u 10 -r 5 -t 15s --host http://127.0.0.1:8000 `
  --csv docs\performance\phase5_control_plane --only-summary

$env:PYTHONPATH = "src"
python scripts\benchmark_mock_provider.py --calls 50000 --concurrency 4
python scripts\seed_backend_load.py --runs 200 --concurrency 8
```

原始 Locust CSV 保存在本目录。专用 `job_agent_load` 数据库和 `job-agent:load-*` Redis namespace 用于隔离压测数据，不能对生产库执行清理命令。

## 7. 未证明与下一轮实验

- 没有真实模型调用，真实 Provider latency、429 比例、成本和吞吐均为 `NOT_PROVEN`。
- 没有多 Worker 横向扩展曲线；当前只证明单 Worker 基线。
- 没有长时间 soak test、Redis trim/retention、PostgreSQL I/O 饱和点或网络故障注入。
- 没有把开发 Header identity 当成生产认证压测。
- 下一轮应该逐级测试 1/2/4 Worker，并观察 PostgreSQL contention、per-user fairness 和 Provider cap，而不是直接宣称更高并发。
