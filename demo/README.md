# Offline CLI demo artifacts

`scripts/demo_local.ps1` 使用固定 fixture 与 deterministic rule backend，离线生成以下示例：

- 正向岗位的 session artifact；
- `plan-next-action` 只规划、不写入 artifact 的输出；
- `run-next-action` 的 mock-interview / local tracker 路径。

所有示例均是脱敏 fixture，不依赖真实模型 API。运行结果将写入 `demo/offline-run/`；该目录可安全重建。

该脚本用于 deterministic CLI fixture/replay 与工程调试。普通 Streamlit UI 固定走 live API，
不会把这条离线路径作为生成 fallback。
