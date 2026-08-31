# 文档导航

- [architecture.md](architecture.md)：CareerStore、Evidence v2、Adaptive Interview、Runtime 与 UI 的职责边界。
- [evaluation.md](evaluation.md)：公开测试、故障注入、Evidence v5 与已知失败的正确口径。
- [security-boundaries.md](security-boundaries.md)：live credential、迁移、Evidence 单向依赖、Tool 与失败边界。
- [evidence/current_status.md](evidence/current_status.md)：2026-08-01 当前事实、数字、非主张与未实现范围。
- [tech-specs/agent_application_optimization_plan_zh.md](tech-specs/agent_application_optimization_plan_zh.md)：Agent 应用方向的分阶段优化计划与验收合同。
- [evidence/backend_service_agent_application_audit_20260802_zh.md](evidence/backend_service_agent_application_audit_20260802_zh.md)：异步可靠性、部署闭环与 Agent 岗位价值审计。
- [evidence/application_assistant_phase_c_eval_20260802_zh.md](evidence/application_assistant_phase_c_eval_20260802_zh.md)：24 条 Mock Agent 场景的 Tool、Approval、Grounding、恢复与副作用评测。
- [evidence/backend_service_phase_d_debug_smoke_20260802_zh.md](evidence/backend_service_phase_d_debug_smoke_20260802_zh.md)：脱敏 Run provenance API 与 Mock/DeepSeek 单请求 Smoke。
- [performance/backend_service_final_fault_matrix_20260802_zh.md](performance/backend_service_final_fault_matrix_20260802_zh.md)：最终 HTTP、Worker 扩展、背压与 Redis/PostgreSQL/Worker 故障恢复矩阵。

根目录 [README](../README.md) 提供安装、live-only 启动方式和项目总览。

本公开镜像只保留可复现的工程材料、synthetic fixture 和脱敏 evidence summary；不包含
`docs/learning`、真实简历、私人运行输出、模型原始响应、凭据或内部业务数据。学习资料和
面试输出也不会成为 Runtime Evidence source。
