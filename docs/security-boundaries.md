# 安全与公开边界

## Credential 与本地配置

- 普通 UI 只从进程环境变量 `JOB_AGENT_LIVE_API_KEY` 读取 Key；
- 页面没有 API Key 输入框，Runtime 不读取 `glm.txt`，也没有 offline/mock fallback；
- Key 不进入 Streamlit state、artifact、SQLite、日志、diagnostic 或 model prompt；
- `.env`、`glm.txt`、`output/`、SQLite、checkpoint、cache、private capture 与日志均被 Git 忽略；
- `.env.example` 只列变量名与非秘密默认值，仓库不自动加载其中的真实凭据。

`raw_model_output` 与完整 prompt 属于内部审计数据。公开 trace 和报告只能保留 sanitized
metadata，不包含 Authorization header、完整简历或原始模型响应。

## CareerStore 与 legacy migration

- SQLite 位于本地 output root，不是共享服务或生产数据库；
- legacy session/tracker 只读迁移，不覆盖、不删除源文件；
- source hash 保证幂等；无法确定公司归属的记录进入待整理，不静默猜测；
- 状态更新和 event append 在同一 transaction 中完成，失败 rollback；
- artifact 表只存链接与 hash，不把私人正文复制进数据库。

这能提供本地一致性和审计，不代表分布式 ACID、备份恢复、访问控制或数据删除证明。

## Evidence 与 Interview 数据流

- Evidence 只来自允许的 JD/简历 source snapshot 与明确的 artifact 类型；
- 学习资料、题库文本、模拟面试回答和 debrief 都不能反向成为 runtime Evidence；
- Interview 对 Evidence 是单向只读依赖；transcript 不更新 evidence level 或 fit；
- exact quote 无法唯一定位、source 缺失、snapshot hash 变化或来源等级超 cap 时 fail closed。

## Tool 与副作用

- Tool Registry 为每个工具声明 input/output schema、effect、timeout、idempotency、approval 与 sandbox 属性；
- skill、agent 配置、runtime policy 三层 allowlist 取交集；模型不能通过 prompt 扩大权限；
- approval 绑定 action digest + session + run，参数变化会使旧批准失效；
- email、calendar、form 只有 disposable sandbox outbox 语义，没有真实 external write；
- repository/source 工具只读，并受路径、调用次数、token budget 与 timeout 约束。

## 失败语义

- API、网络、Schema、Guard 或写入失败会终止当前 live 动作，不生成离线替代 artifact；
- schema/Guard repair 有限，仍不合法则 fail closed；
- 原子写入与 last-known-good 防止半成品覆盖；
- 非幂等动作在 checkpoint 前崩溃时进入 uncertain outcome，不盲目重放；
- cooperative timeout 依赖 handler 配合 deadline，不能当作不可抢占的硬超时。

## 不作的承诺

项目不能证明完整的 prompt-injection 防护、生产级 secret management、跨租户隔离、真实平台
合规、线上高并发、HA、自动投递幂等或招聘效果。产品化前仍需 provider OAuth、KMS、访问
控制、审计日志、队列/outbox、receipt reconciliation、备份、限流、监控和独立安全评估。
