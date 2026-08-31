# Backend Phase D Debug / Provider Smoke 记录

日期：2026-08-02  
范围：小规模 Smoke，不是负载测试或最终矩阵测评

## 1. Debug provenance 合同

新增 owner-scoped：

```text
GET /api/v1/runs/{run_id}/debug
```

PostgreSQL 使用一个 repeatable-read、read-only 事务投影：

- Run status/version/stage/error/retry；
- Execution Manifest；
- Attempt ID/no/status/phase/provider call count/error/recovery kind；
- Approval ID/request ID/action digest/Tool/decision/version；
- Checkpoint kind/ID/version/timestamp；
- Tool operation ID/Attempt ID/Tool/digest/state/error/timestamp；
- 最多 500 条脱敏 Event，并标记是否截断。

明确不返回：Run request payload、Idempotency-Key、简历正文、CareerSnapshot payload、Checkpoint payload、
Tool arguments、request hash、Tool receipt、lease token、内部 Worker 名称、Provider prompt/response 或
Chain-of-Thought。跨用户查询仍返回 404。

## 2. Mock HTTP Smoke

单个 `semantic_job_flow / mock` Run：

| 项目 | 结果 |
| --- | --- |
| POST | `202` |
| 终态 | `SUCCEEDED` |
| Attempt | 1 |
| Event | 8 |
| Manifest | present |
| debug request payload | absent |
| debug private checkpoint payload | absent |

该结果只证明单请求 HTTP→PostgreSQL/Outbox→Redis→Worker→GET/debug 闭环，不是吞吐或 SLA。

## 3. DeepSeek Smoke

系统环境 `API_KEY` 只注入临时 Worker 进程；未打印值、未写文件或 Compose。为避免无 Key Worker 抢占，
Smoke 期间暂停普通 Worker，完成后删除临时容器并恢复普通 Worker。

| 项目 | 结果 |
| --- | --- |
| Run 数 | 1 |
| POST | `202` |
| 终态 | `SUCCEEDED` |
| Attempt | 1 |
| Provider calls | 2 |
| Manifest provider/model | `deepseek_flash / deepseek-chat` |
| 原始响应保存 | 否 |

该 Smoke 复核了真实 Provider 路由、Schema Runtime 和 debug Manifest；没有测试开放输入质量、并发、
429、长尾延迟、成本稳定性或生产可用性。

## 4. 尚未执行

- 大规模 Application Assistant 场景矩阵；
- 混合 HTTP 4xx/5xx、PostgreSQL/Redis/Worker 故障负载；
- Locust 多用户控制面压测复跑；
- 真实 Provider 批量 Canary 或压力测试。

这些项目按用户要求在剩余代码合同完成后统一测评，Mock 与真实 Provider 结果继续分开报告。
