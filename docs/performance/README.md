# Backend performance evidence

本目录只记录可复现实验与原始摘要。Mock Provider、HTTP 控制面、Worker 和真实模型 API 的口径必须分开；没有实测的数据标为 `NOT_PROVEN`。

- `backend_service_final_fault_matrix_20260802_zh.md`：最终 HTTP、1/2/4 Worker、Redis/PostgreSQL/Worker
  故障、queue cap、恢复时间和资源报告；
- `final_control_plane_fixed_20260802_*.csv`：最终 Locust 原始摘要；
- 真实 DeepSeek 仅见单任务 Canary 报告，不参与上述压力测试。
