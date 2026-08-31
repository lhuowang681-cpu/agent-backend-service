# Backend Service 第六阶段：DeepSeek Flash Canary 报告

日期：2026-08-02

## 1. 结论

真实 DeepSeek Flash 已通过“直接 API 冒烟”和“Backend 完整单 Run / 小批量 Run”验证。完整链路为：

```text
POST 202
-> PostgreSQL Run + Outbox
-> Redis Stream
-> Worker claim / lease / fencing
-> Existing Semantic Agent Runtime
-> DeepSeek official API
-> PostgreSQL result
-> GET Run
```

这证明真实 Provider Adapter 能穿过现有异步服务纵向闭环。它不是压力测试，不证明生产吞吐、
可用性、SLA、成本或真实 LLM 的并发上限。四个 Run 的排队时间显示，当前单 Worker 串行执行是
最明显的系统瓶颈。

## 2. 安全与实验边界

- 凭据来自 Worker 子进程的系统环境 `API_KEY`，未输出值，未写入文件、日志或 Git diff。
- 只访问 allowlist 中的 `https://api.deepseek.com/v1/chat/completions`。
- 请求模型为 `deepseek-chat`；响应模型标识为 `deepseek-v4-flash`。
- 未记录直接冒烟的响应正文；只检查 JSON 合法性和必要元数据。
- 没有读取 credential 文件或 `output/private/**`。
- 完成 Canary 后已停止本地真实 Worker；没有继续执行真实模型压测。

## 3. 直接 API 冒烟

请求使用 `response_format={"type":"json_object"}` 与 `max_tokens=64`。结果：

| Metric | Value |
| --- | ---: |
| outcome | success |
| latency | 1,208 ms |
| prompt tokens | 47 |
| completion tokens | 11 |
| finish reason | stop |
| JSON valid | true |

该步骤只验证官方端点、凭据和 JSON Object 模式可用，不验证 Agent Runtime。

## 4. Backend 完整单 Run

Run 使用 `provider_profile=deepseek_flash`，经过 PostgreSQL、Outbox、Redis Stream 和独立 Worker。

| Metric | Value |
| --- | ---: |
| terminal status | SUCCEEDED |
| end-to-end latency | 4.621 s |
| LLM calls | 2 |
| retries | 0 |
| fit verdict | not recommended |
| provider latency sum | 3.017 s |

HTTP 返回 `202` 后没有等待模型完成；最终结果通过 GET 查询。这一结果证明控制面与执行面的分离
在真实 Provider 下仍成立。

## 5. 四 Run 小批量 Canary

四个 Run 在两个受控用户间交替提交，使用一个串行 Worker：

| Metric | Value |
| --- | ---: |
| Runs | 4 |
| SUCCEEDED / FAILED | 4 / 0 |
| total LLM calls | 20 |
| total retries | 0 |
| error codes | none |
| E2E min | 18.969 s |
| E2E p50 | 39.431 s |
| E2E p95 / p99 / max | 77.822 s |

单 Run 的模型调用数取决于现有 Semantic Flow 的 fit 分支，因此 4 Run 共调用 20 次。加上完整单
Run 的 2 次调用，Backend 指标观察到 22 次 DeepSeek 调用，累计 Provider latency 为 79.617 秒。

样本只有四个，p95/p99 实际落在最大样本附近，不应被当作稳定统计量。它的用途是暴露串行排队，
不是建立容量基线。

## 6. Token 与成本边界

`job_agent_provider_tokens_total{provider,direction}` 在上述 22 次 Backend 调用完成之后才接入，
所以历史总 token 和总成本不可追溯。不能根据 latency、调用次数或最大输出长度反推实际 token。

唯一已知 token 是直接 API 冒烟的 47 input / 11 output。后续运行会按 Provider 返回的 usage 分别
累计 input/output token；指标不包含 prompt、response、用户或 Run 标识。成本计算仍不是本阶段目标，
因为价格表是外部可变配置，而且本次历史 usage 不完整。

## 7. 故障与重试合同

- DeepSeek 缺 Key：`provider_credential_missing`，明确失败，不回退 Mock。
- 429 / timeout / transport：Provider Adapter 保留稳定错误类型，交给现有服务故障分类与背压逻辑。
- Canary 的节点 policy 设置 `max_retries=0`、`max_schema_repairs=0`、无规则 fallback，避免隐藏真实失败。
- JSON Object 只约束返回为 JSON；最终字段合同仍由现有 `LLMHarness` 和 Pydantic Schema 校验。
- 模型调用完成但持久状态写入前崩溃，可能再次调用并重复计费；这不等同于 Tool exactly-once。
- 非幂等 Tool 的结果不确定时仍进入 `UNCERTAIN`，不能因为 Provider 可重试就盲目重放 Tool。

## 8. 瓶颈与下一步

当前证据表明 HTTP 控制面不是这组真实 Canary 的主要耗时；单 Worker 串行处理多个长 Run 产生了
明显队列等待。合理的下一步不是对真实 API 加压，而是：

1. 用 Mock Provider 验证增加 Worker 数后的 claim、租约、per-user/global admission 是否仍正确；
2. 将 queue wait 与 execution time 拆成独立直方图，避免只看 E2E；
3. 在明确费用预算和 Provider 配额后，再做极小、阶梯式的真实并发 Canary；
4. 为 token/cost 建立从首次调用开始的持久账本后，才讨论单 Run 成本分布；
5. 生产部署使用 Secret 管理，不把宿主环境变量写进 Compose 或镜像。

## 9. 可复核代码与测试

- Provider：`src/job_agent/llm/providers/deepseek_compatible.py`
- 通用传输：`src/job_agent/llm/providers/openai_compatible.py`
- 路由与 Runtime 复用：`src/job_agent/backend_service/execution.py`
- Worker 配置：`src/job_agent/backend_service/worker_main.py`
- 指标：`src/job_agent/backend_service/metrics.py`
- Provider 测试：`tests/test_llm_agentization_runtime.py`
- 路由/故障测试：`tests/test_backend_service_reliability.py`
- 持久指标测试：`tests/test_backend_service_persistence.py`

真实 API 响应正文和凭据不属于可复核工件；自动化测试使用可控 transport / Mock，不产生费用。

## 10. 第六阶段验证结果

```text
LLM + Backend focused: 59 passed
Related regression excluding frozen Evidence modules: 258 passed, 2 warnings
Full regression: 260 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
compileall: PASS
pip check: PASS
git diff --check: PASS
broad exception / unbounded-loop scan: no matches
Docker Compose API/Worker rebuild: PASS
```

部署态还使用唯一 Idempotency-Key 进行了两项无费用检查：Mock 重复 POST 返回相同 `run_id` 且最终
`SUCCEEDED`；未注入 Key 的 `deepseek_flash` Run 最终为 `FAILED / provider_credential_missing`。

全量回归的 19 个失败全部属于既有冻结 Evidence 文件的 CRLF/byte-hash 差异。本阶段未修改这些文件、
manifest、Evidence 评测语义或回归口径；排除对应三个模块后相关回归全部通过。
