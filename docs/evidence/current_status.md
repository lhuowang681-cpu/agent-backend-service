# 当前事实与边界

日期：2026-08-01
版本状态：`personal-career-workbench phase-1 release candidate`

## 已实现

- SQLite `CareerStore`：公司、岗位、阶段事件、迁移审计、artifact link、面试索引；
- 非破坏、幂等 legacy migration；未知公司和冲突进入待整理；
- 公司库首页/详情、搜索、优先级/标签筛选、岗位与面试汇总、从公司预填新岗位；
- 普通 UI live-only；Key 只来自 `JOB_AGENT_LIVE_API_KEY`，无页面输入、无 `glm.txt`、无 fallback；
- Evidence v2：immutable source、quote resolver、requirement/link candidate、provenance level cap、projection；
- 动态模拟面试：一次一题、checkpoint、根据回答追问/换题、coverage 与 budget 控制；
- `ProjectDossier`、公开基础题 anchor 和允许的只读工具共同构成问题来源；
- 完整 transcript 后生成 debrief；Interview 不写入 Evidence；
- Schema、Guard、有限 repair、atomic write、last-known-good 与 sanitized diagnostics。

## 当前验证

- 公开仓库：`244 passed, 2 warnings`；
- native LangGraph + semantic full-E2E 聚焦回归：`28 passed`；
- 开发工作区升级相关：`135 passed`；
- 开发工作区全量：`821 passed / 0 failed / 2 warnings`；
- 历史 15 项 LangGraph 失败已通过 execution schema / application checkpoint metadata 分层修复；
- 旧 checkpoint 对外 `checkpoint_id` 字段保持兼容，无需迁移；
- Reliability Fault Injection：`10 PASS / 0 FAIL / 1 NOT_PROVEN`，`82/82` applicable assertions；
- Evidence v5 mock：`300/300` variant-case rows；
- GLM-5.2 development：55/60 completion，Req F1 0.952，Link F1 0.602，Fit 0.700；
- GLM-5.2 frozen held-out：38/40 completion，Req F1 0.905，Link F1 0.602，Fit 0.625，unsupported claim rate 15.4%。

`38/40` 是 pipeline completion，不是准确率。以上模型结果来自 100-case synthetic
benchmark，不是 human-gold，不证明真实简历泛化、招聘效果或 SLA。

## Evidence 准确语义

- C0/C1/C2/C3 是证据强度，不是能力分数；
- LLM 产生 exact quote 与语义候选，Python 解析 offset/hash、执行来源 cap 和持久化；
- candidate schema 不包含可确定计算的 offset/hash/派生状态；
- 模糊 quote、stale snapshot、未知 source 和 over-cap evidence 必须 repair 或 fail closed；
- Fit 是 Evidence projection；面试表现不能反向更改 Evidence 或 Fit。

## Interview 准确语义

```text
Frozen context snapshot
  = JD + Resume + Evidence + Project Dossier + public anchors

Adaptive interviewer
  = one turn / question + user answer observation + follow-up / switch / finish

Debrief
  = transcript complete 后的一次结构化复盘
```

题库 anchor 不是随机抽题池，Evidence 也不是唯一问题来源。模型仍可能生成质量一般或
过度贴近材料的问题，因此需要 coverage、duplicate、budget、tool allowlist 与最终人工判断。

## 未实现

- 第二阶段题库 CRUD、练习系统和导出；
- 语音、实时转写、coding sandbox、爬虫、云同步或移动端；
- 真实自动投递、邮件或表单 external write；
- 生产数据库、消息队列、多租户、HA、监控和 SLA；
- 人工 gold 的 Evidence / Mock Evaluator calibration；
- 已证明有效的 DPO/RLHF/GRPO 或 LoRA quality release。

## 可安全引用的表达

> 当前实现并验证了本地 CareerStore、可审计迁移、Evidence v2 确定性约束和动态面试
> Agent 链路；测试能证明工程合同与 synthetic benchmark 表现，但不能证明真实招聘提升、
> human-gold 准确率或生产 SLA。
