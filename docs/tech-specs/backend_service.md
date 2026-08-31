# Job Agent AI Backend Service — Technical Spec

状态：第二阶段 Tracer Bullet 已实现并通过聚焦测试，等待阶段验收  
日期：2026-08-02  
目标分支：`feature/backend-service`  
基线：`6a457758349343b9d90c08282661370744b00df2`

## 1. Context

本规格把现有本地 Job Agent 包装成一个可部署、可测试、具备真实多用户隔离边界的异步 Backend Service，用于学习和验证长耗时 Agent 任务的异步执行、状态一致性、幂等、审批暂停、故障恢复和背压。工程形态参考常规分层后端：API/身份上下文、应用服务、Repository、队列/Worker、领域 Runtime Adapter 和基础设施 Adapter 各自承担清晰职责。

服务层只拥有调度、持久化和服务状态；它不能重新实现或绕过现有：

- `AgentLoop`；
- `PolicyEngine`；
- `ApprovalRequest` / `ApprovalDecision`；
- `AgentCheckpoint` / `SemanticGraphCheckpoint`；
- Tool effect、Verifier、Budget 和 Evidence 合同。

本规格建立在以下已核实事实之上：

- 发布版的 `agent_runtime/`、`runtime/`、`career/` 共 34 个文件与当前研发工作区逐字节一致；
- 当前工程事实基线是 `821 passed, 2 warnings`，不得改变其语义或统计口径；
- `AgentLoop` 已支持 approval scope、action digest、checkpoint/resume 和非幂等 inflight fail-closed；
- `AgentGraphRuntime` 已支持按 semantic node 保存 checkpoint，并校验 input、skill、prompt 和 state digest；
- `CareerStore` 是本地 SQLite 业务事实库，不是分布式 Run Repository；
- 当前 Tool timeout 是 cooperative timeout，不是可以杀死永久阻塞代码的 hard timeout。

## 2. Goals

1. `POST /api/v1/runs` 在持久化 Run 后立即返回 `202`，不等待模型完成。
2. HTTP 请求、任务消息、Worker Attempt 和现有 Agent execution 可以通过稳定 ID 关联。
3. 重复 HTTP 请求不创建两个 Run。
4. 任务消息至少一次投递时，不并发执行同一个 Run，也不盲重放非幂等 Tool。
5. Worker 崩溃后可以从已确认 checkpoint 恢复；副作用不确定时进入 `UNCERTAIN`。
6. Approval 在服务重启后仍然有效，并同时绑定 `session_id`、`run_id`、`request_id` 和 `action_digest`。
7. Provider 429、timeout、transport failure 使用有界重试、全局保护和用户级背压，不使用无限重试。
8. 测试分别证明控制面合同、Worker 合同、Mock Provider 行为和真实 Provider 边界。
9. 所有公开 Event、错误和指标默认脱敏，不公开 raw prompt、raw response、API Key 或简历正文。
10. 从 Tracer Bullet 开始支持多个彼此隔离的用户；Run、Event、Approval、Checkpoint、文件、队列配额和 Provider 预算不得跨用户混用。
11. 任意用户都不能读取、审批、拒绝、恢复或推断其他用户的 Run；跨用户资源访问统一按不存在处理。

## 3. Non-Goals

- 不提供真实自动投递、邮件发送或表单提交；
- 不承诺 exactly-once 消息、exactly-once Provider invocation 或 exactly-once 外部副作用；
- 不在第一版实现注册、登录、找回密码、SSO、组织/workspace/tenant、membership、复杂 RBAC、计费、HA 或生产 SLA；
- 不修改现有 Agent、Evidence、Policy、Approval、Checkpoint 和 Verifier 语义；
- 不把 Adaptive Interview 的普通用户回答伪装成 Approval；
- 不使用真实 LLM 做压测；
- 不把 Mock Provider 压测结果表述为真实 LLM 服务性能；
- 不为了展示技术栈引入 Kafka、Celery、Kubernetes 或分布式事务框架；
- 不自动修改简历、Push 或公开 GitHub。

## 4. Confirmed Decisions

1. Tracer Bullet 使用现有 `run_semantic_job_flow` 和 `MockLLMProvider`。
2. Service v1 只支持 Approval pause；普通 `AgentInputRequest` 暂不服务化。
3. Phase 3 使用 PostgreSQL Outbox + Redis Streams；PostgreSQL 是唯一权威状态源。
4. Redis 消息只携带调度标识，不携带完整简历、Prompt、Checkpoint 或模型原始输出。
5. `UNCERTAIN` 不能通过普通 `/resume` 自动离开，必须先人工核对副作用。
6. `user_id` 同时是 v1 的数据所有权、资源、配额、并发和操作审计边界；不引入 `tenant_id`、membership 或共享 workspace。
7. HTTP 请求体不能提交或覆盖 `user_id`；它由 `RequestContext` Adapter 从可信身份来源解析。
8. v1 的幂等身份固定为 `(user_id, Idempotency-Key)`；同名 `session_id` 的串行化范围是 `(user_id, session_id)`。
9. 多用户隔离与多节点扩容是两件事：初始 Compose 可以只有一个 API/Worker 实例但必须正确隔离多个用户；增加 Worker 节点前必须满足共享 PostgreSQL Run 状态和本规格中的 CareerStore 部署边界。

## 5. Architecture

```text
Client
  -> trusted identity source / dev-test identity fixture
  -> RequestContext(user_id, request_id)
  -> FastAPI Control Plane
  -> PostgreSQL transaction
       runs
       run_events
       outbox_messages
  -> Outbox Publisher
  -> Redis Stream
  -> Agent Worker
  -> Run claim + Attempt lease + fencing token
  -> AgentExecutionAdapter
  -> Existing AgentGraphRuntime / AgentLoop
  -> Existing Approval / Checkpoint / Policy / ToolExecutor
  -> PostgreSQL status + event + checkpoint
```

### 5.1 Source of Truth

PostgreSQL 是以下事实的唯一权威来源：

- Run 当前状态；
- 当前有效 dispatch generation；
- Attempt lease 和 fencing token；
- Approval 决定；
- user 归属与操作审计；
- Checkpoint；
- 公开 Event；
- 尚未发布或需要重发的 Outbox message。

Redis Streams 只是可重建的唤醒层。Redis 丢失数据后，Reconciler 可以从 PostgreSQL 中的 Outbox 和可执行 `QUEUED` Run 重建消息。

### 5.2 Identity Chain

```text
user_id + HTTP request_id
  + Idempotency-Key
  + request_hash
      -> run_id
      -> dispatch_generation
      -> user-scoped outbox message_id
      -> Redis message(user_id, run_id, generation)
      -> user-scoped attempt_id + lease_token
      -> existing session_id + run_id + approval digest
      -> user-scoped checkpoint/storage/event
```

`user_id` 必须贯穿 HTTP、数据库、Outbox、Redis message、Worker claim、Checkpoint 和文件路径。Redis 中的 `user_id` 只用于路由与一致性校验，数据库 Run 记录才是权威归属；二者不一致时拒绝执行并记录安全事件。`run_id` 贯穿服务和现有 Runtime。`attempt_id` 属于服务调度层，不写入现有 Agent 语义模型；Trajectory Adapter 在外层 Event envelope 中补充它。

### 5.3 Multi-user Boundary

```text
user_id = 数据所有权、资源、队列配额、并发、Provider 预算和操作审计边界
```

- 应用层每个用户用例都必须接收 `RequestContext`，Repository 每个用户数据查询都必须显式带 `user_id`；禁止先按全局 ID 查询后在内存里检查所有者。
- 跨用户 Run/Event/Approval/Checkpoint 查询返回 `404`，不返回 `403`，避免确认其他用户资源是否存在。
- 应用层 owner filter 是 v1 的强制安全边界；PostgreSQL RLS 只作为后续 defense-in-depth，不能代替应用查询约束。
- 生产部署必须由可信 IdP、API Gateway 或签名 Token 提供身份。仅开发/测试模式可启用受控 Header/fixture Adapter，且生产配置检测到该 Adapter 时必须拒绝启动。

### 5.4 Multi-user Alternatives

| Option | Decision |
| --- | --- |
| 只在表中保存 `user_id` | **采用**；准确表达当前“每个用户拥有自己的数据”，合同和测试最小完整 |
| 只保存 `tenant_id` | 不采用；当前没有组织、workspace 或成员关系，命名会制造并不存在的共享语义 |
| 同时保存 `tenant_id + user_id` | 不采用；v1 两者一对一是冗余模型，会扩大复合外键、配额和测试复杂度 |
| 让客户端在 body/Header 任意声明 user | 不采用；这只是可伪造参数，不是身份 |
| 本阶段实现完整登录/RBAC | 不采用；与异步可靠性主目标正交，由 `RequestIdentityAdapter` 保留可信接入 seam |
| 只依赖 PostgreSQL RLS | 不采用；RLS 可后补为 defense-in-depth，但应用 Repository 仍必须显式 user-scoped |

未来只有在出现“多个用户共享同一 Career workspace/Run/artifact”这一真实需求时，才新增 `workspaces`、`workspace_memberships`、角色与共享授权规则。该变化作为独立版本化迁移处理；v1 不预埋一个永远等于用户的伪 `tenant_id`。

## 6. Decomposition

- **RequestIdentityAdapter** — 把可信认证结果转换为不可由业务请求体覆盖的 `RequestContext`。
- **RunService** — 在 `RequestContext` 范围内创建、查询和驱动 Run 状态迁移的唯一应用入口。
- **RunRepository** — 封装 Run、Attempt、Approval、Checkpoint、Event 和 Outbox 的事务与 CAS。
- **DispatchQueue** — 发布和领取最小任务消息，不保存业务真相。
- **WorkerEngine** — claim、heartbeat、执行、恢复、ACK 和 fencing。
- **AgentExecutionAdapter** — 把服务 Run 转换成现有 Agent Runtime 调用并映射结果。
- **CareerContextAdapter** — 隔离现有 SQLite `CareerStore`，向 Worker 提供用户范围内的业务快照。
- **ProviderAdmissionController** — Provider 并发、rate limit、cooldown 和 circuit 状态。
- **RecoveryCoordinator** — 回收过期 lease、重发 Outbox、分类安全恢复与 `UNCERTAIN`。
- **ObservabilityRecorder** — 指标和 sanitized Event，不保存 private raw content。

## 7. Modules

### 7.1 RequestIdentityAdapter

**Responsibility:** 从可信身份载体构造请求上下文，使业务请求体永远不能选择自己的用户身份。

**Public interface:**

```python
class RequestContext(StrictModel):
    user_id: str
    request_id: str

class RequestIdentityAdapter(Protocol):
    def resolve(self, request: Request) -> RequestContext: ...
```

**Invariants:**

- `user_id` 不从 JSON body、path 或 query parameter 获取；
- 解析出的用户必须存在且处于可用状态；
- dev/test Adapter 只接受预先配置或测试夹具中的身份，不动态创建任意用户；
- production profile 禁止启用信任无签名 `X-User-ID` 的 Adapter；
- Worker 不重新信任 HTTP 身份载体，而是从 PostgreSQL Run 记录恢复权威 owner user。

**Failure modes:**

- 缺失或无效身份 -> `401 identity_required`；
- 已解析但无权使用目标资源 -> 对资源接口返回 `404 resource_not_found`；
- production 启用 dev identity Adapter -> 启动失败；
- 身份目录中的 user 不存在或被禁用 -> fail closed 并记录脱敏安全事件。

**Tests:**

- 请求体中的伪造 `user_id` 被 schema 拒绝；
- dev fixture 能解析两个相互隔离的用户；
- production profile 拒绝不可信 Header Adapter；
- 已禁用用户不能创建或读取 Run。

### 7.2 RunService

**Responsibility:** 在 `RequestContext` 范围内向 HTTP 层提供稳定的 Run、Approval 和 Resume 用例，不暴露数据库或队列细节。

**Public interface:**

```python
class RunService(Protocol):
    def create_run(
        self,
        context: RequestContext,
        command: CreateRunCommand,
        *,
        idempotency_key: str,
    ) -> CreateRunResult: ...

    def get_run(self, context: RequestContext, run_id: str) -> RunView: ...

    def decide_approval(
        self,
        context: RequestContext,
        command: DecideApprovalCommand,
    ) -> RunView: ...

    def resume_run(
        self,
        context: RequestContext,
        command: ResumeRunCommand,
    ) -> RunView: ...

    def list_events(
        self,
        context: RequestContext,
        run_id: str,
        *,
        after: int,
        limit: int,
    ) -> EventPage: ...
```

**Invariants:**

- 相同 user/key 和相同 request hash 永远返回同一 `run_id`；
- 相同 key、不同 hash 永远冲突；
- 每个查询和变更先按 `context.user_id` 限定资源；
- 创建 Run、首条 Event 和 Outbox message 在同一事务提交；
- HTTP 层不直接写 Run 状态；
- Approval 决定必须重新经过现有 `PolicyEngine` 校验。

**Failure modes:**

- 缺少 `Idempotency-Key` -> `400 idempotency_key_required`；
- key payload 冲突 -> `409 idempotency_key_payload_mismatch`；
- queue admission 超限 -> `429 queue_capacity_exceeded`，且不创建 Run；
- stale status version -> `409 stale_run_version`；
- Approval 已作相反决定 -> `409 approval_already_resolved`。

**Tests:**

- 并发提交相同 key 只产生一条 Run；
- 相同 key 不同 payload 冲突；
- Repository rollback 后 Run/Event/Outbox 均不存在；
- 重复相同 Approval 决定幂等；
- digest、scope、version 任一不匹配均不排队；
- user A 不能读取、审批、拒绝或恢复 user B 的 Run；
- 不同 user 使用相同 key 各自产生一个 Run。

### 7.3 RunRepository

**Responsibility:** 通过事务、唯一约束、CAS 和 fencing 维护服务层全部权威状态。

**Public interface:**

```python
class RunRepository(Protocol):
    def create_or_get(self, request: NewRunRequest) -> CreateOrGetResult: ...
    def claim(self, command: ClaimRunCommand) -> ClaimedAttempt | None: ...
    def heartbeat(self, command: HeartbeatCommand) -> bool: ...
    def transition(self, command: TransitionRunCommand) -> RunRecord: ...
    def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None: ...
    def load_checkpoint(self, user_id: str, run_id: str) -> StoredCheckpoint | None: ...
    def decide_approval(self, command: DecideApprovalCommand) -> ApprovalRecord: ...
    def append_runtime_event(self, event: StoredRuntimeEvent) -> int: ...
    def list_due_recoveries(self, *, limit: int) -> list[RecoveryCandidate]: ...
```

**Invariants:**

- 每次状态写入必须匹配 `status_version`；
- 所有用户数据读写必须以 `user_id` 开始查询条件并由 user-scoped foreign key 约束；
- Worker 写入必须额外匹配 `attempt_id + lease_token`；
- 同一 Run 同时至多一个有效 Attempt；
- 同一 `(user_id, session_id)` 同时至多一个处于 `RUNNING/WAITING_APPROVAL` 的可变更 Run；不同 user 的同名 session 不冲突；
- terminal Run 不再接受旧 generation 的任务消息；
- 状态迁移与对应服务 Event 同事务提交。

**Failure modes:**

- serialization/deadlock -> 有界 DB transaction retry；
- connection lost during commit -> 按稳定 ID 查询提交结果；
- fencing mismatch -> 丢弃旧 Worker 结果；
- checkpoint validation failure -> fail closed，不覆盖旧 checkpoint。

**Tests:**

- 两个 Worker 并发 claim 只有一个成功；
- lease 过期后新 Worker 获得新 token；
- 旧 Worker 返回时不能覆盖新状态；
- PostgreSQL rollback 不产生半个状态迁移；
- terminal 状态不可逆；
- owner user mismatch 的 claim、checkpoint、approval 和 transition 全部 fail closed。

### 7.4 DispatchQueue

**Responsibility:** 低延迟唤醒 Worker，并提供至少一次投递；不判断 Run 是否应该执行。

**Message schema:**

```python
class RunDispatchMessage(BaseModel):
    message_id: str
    user_id: str
    run_id: str
    dispatch_generation: int
    enqueued_at: datetime
```

**Invariants:**

- 消息不包含简历正文、Prompt、API Key、Approval arguments 或 Checkpoint；
- 消息携带 `user_id` 但它不是授权真相；Worker 必须与 PostgreSQL Run 的 owner user 交叉校验；
- Worker 必须先通过 PostgreSQL claim 才能调用 Agent；
- ACK 只能发生在状态事务提交后；
- stale generation 消息可以安全 ACK；
- Stream 被清空后可以由 PostgreSQL 重建。

**Failure modes:**

- Redis unavailable -> 保留 Outbox，延迟发布；
- Worker 在 `XREADGROUP` 后崩溃 -> PEL 中由 `XAUTOCLAIM` 回收；
- 发布成功、Outbox 未标记 -> 允许重复发布；
- ACK 失败 -> 重复消息由数据库状态去重。

**Tests:**

- 重复消息只产生一个有效 Attempt；
- 未 ACK 消息可被其他 consumer reclaim；
- Redis 重启后 Outbox 可重新发布；
- stale generation 不执行 Agent；
- Redis message user 与 Run owner 不一致时不执行 Agent，并产生 sanitized security event。

### 7.5 WorkerEngine

**Responsibility:** 把一条至少一次消息安全地转成至多一个有效 Agent Attempt。

**Public interface:**

```python
class WorkerEngine:
    def handle(self, message: RunDispatchMessage) -> HandleResult: ...
    def recover_expired(self, *, limit: int) -> RecoverySummary: ...
```

**Execution sequence:**

```text
read stream message
-> DB claim(user_id, run_id, generation)
-> verify message user == persisted Run owner
-> start heartbeat
-> acquire session/provider admission
-> build AgentExecutionAdapter
-> execute or resume existing Runtime
-> classify outcome
-> DB transition + Event + optional Outbox
-> ACK Redis message
```

**Invariants:**

- Agent 执行绝不发生在 claim 之前；
- FastAPI 进程不执行长耗时 Agent；
- heartbeat 丢失后 Worker 不再有提交权限；
- Provider retry、Run retry 和 message redelivery 分别计数；
- Worker 进程不能共享一个带持久内存 cache 的全局 `AgentGraphRuntime`。

**Failure modes:**

- crash before model -> lease recovery；
- crash after model before checkpoint -> 允许重新调用模型，记录可能重复计费；
- crash after idempotent Tool -> 使用业务 key 重放；
- crash around non-idempotent Tool -> `UNCERTAIN`；
- terminal commit 后 ACK 前 crash -> 重复消息读取终态并 ACK。

**Tests:**

- 在每个故障注入点杀死 Worker 并验证最终状态；
- heartbeat 过期和 fencing；
- terminal commit/ACK 窗口；
- 同 user/session 串行化；
- cooperative cancellation 边界。

### 7.6 AgentExecutionAdapter

**Responsibility:** 复用现有 Runtime，完成输入、checkpoint、event 和结果映射，不重新实现 Agent 语义。

**Public interface:**

```python
class AgentExecutionAdapter(Protocol):
    def execute(self, context: ServiceRunContext) -> AgentExecutionOutcome: ...
```

**Tracer implementation:**

```text
SemanticJobFlowAdapter
  -> existing run_semantic_job_flow
  -> existing MockLLMProvider
  -> existing SemanticGraphCheckpoint
```

**Later implementation:**

```text
DomainAgentAdapter
  -> existing AgentLoop.run
  -> existing ApprovalDecision
  -> existing AgentCheckpoint
```

**Status mapping:**

| Existing Runtime outcome | Service status |
| --- | --- |
| `COMPLETED` | `SUCCEEDED` |
| `WAITING_FOR_USER` + pending approval | `WAITING_APPROVAL` |
| `FAILED` + `non_idempotent_execution_uncertain` | `UNCERTAIN` |
| `FAILED` / `BUDGET_EXCEEDED` | `FAILED` |
| retryable Provider error + compatible checkpoint | `QUEUED` with `next_attempt_at` |

**Invariants:**

- 只通过公开 Runtime seam 调用现有 Agent；
- 不手工执行 Tool；
- 不在 Adapter 中判定 Approval 是否有效；
- 不把普通 `pending_user_input` 映射为 `WAITING_APPROVAL`；
- raw Provider output 不写服务 Event。

**Failure modes:**

- checkpoint identity/version mismatch -> `FAILED`；
- unsupported ordinary input request -> `FAILED unsupported_user_input_wait`；
- exhausted Provider retry -> 依据最后一条 sanitized ProviderTrace 分类；
- Runtime 返回无对应映射的新状态 -> fail closed。

**Tests:**

- Tracer Bullet 完整成功路径；
- semantic checkpoint 跳过已完成节点；
- approval 恢复不重新规划 pending action；
- non-idempotent uncertain 映射；
- Provider trace 不泄漏 raw output。

### 7.7 CareerContextAdapter

**Responsibility:** 在不迁移或重写 `CareerStore` 语义的前提下，为 Agent execution 提供用户隔离的只读业务上下文快照。

**Public interface:**

```python
class CareerContextAdapter(Protocol):
    def load_snapshot(
        self,
        context: ServiceRunContext,
        refs: CareerInputRefs,
    ) -> CareerContextSnapshot: ...
```

**Invariants:**

- Tracer Bullet 使用显式 fixture/snapshot，不把一个共享 SQLite 文件暴露给多个用户；
- 本地单节点模式如需读取现有 `CareerStore`，根目录必须由服务端按 user 派生，并对每个 user 串行访问；
- 客户端只能提交逻辑 artifact reference，不能提交绝对路径、`..` 或选择其他 user 根目录；
- `CareerStore` 仍是业务事实来源，不充当 Run Repository；本阶段不静默迁移、不重写其业务语义。

**Failure modes:**

- snapshot owner 与 Run owner 不匹配 -> fail closed；
- 路径规范化后越出用户根 -> `400 invalid_artifact_reference`；
- 多节点部署请求可变 CareerStore 工作流、但尚无共享 Repository -> 拒绝该部署模式，而不是假装一致；
- SQLite busy/corrupt -> 分类失败并保留 Run/Checkpoint 证据，不盲目重试写操作。

**Evolution:**

完整多节点、多用户 Career 工作流需要单独实现 PostgreSQL `CareerRepository` Adapter，或明确维持单节点用户分片模式。该迁移必须以现有 `CareerStore` 合同测试为准，不能夹带在 Tracer Bullet 中。

**Tests:**

- user A 无法引用 user B 的 career snapshot；
- 绝对路径与路径穿越被拒绝；
- 两个 user 的同名 artifact/session 不冲突；
- 本地 CareerStore Adapter 对同 user 串行、对不同 user 使用不同根目录。

### 7.8 ProviderAdmissionController

**Responsibility:** 在现有 Provider 调用之外控制共享并发和 cooldown，避免所有 Worker 同时撞击供应商。

**Public interface:**

```python
class ProviderAdmissionController(Protocol):
    def acquire(
        self,
        user_id: str,
        key: ProviderKey,
        *,
        deadline: datetime,
    ) -> Permit: ...
    def record_success(self, permit: Permit, usage: ProviderUsage) -> None: ...
    def record_rate_limit(self, permit: Permit, *, retry_after_s: float | None) -> None: ...
    def record_failure(self, permit: Permit, error_code: str) -> None: ...
```

**Invariants:**

- 同时执行全局 provider/model 硬上限和 per-user 公平配额；任何用户都不能耗尽其他用户的全部许可；
- 等待 admission 不占用 Agent active-time budget；
- existing Provider retry 和 service retry 都有独立上限；
- circuit open 时不继续制造同步重试风暴；
- 不根据 Mock Provider 吞吐设置真实 Provider 限额。

**Failure modes:**

- Redis limiter unavailable -> fail closed 或使用低并发本地安全值，不进入无限放行；
- 429 无 `Retry-After` -> bounded exponential backoff + full jitter；
- permit 泄漏 -> TTL 自动回收。

**Tests:**

- 并发上限；
- 429 cooldown；
- timeout 不永久占用 permit；
- circuit half-open 只允许少量探测；
- limiter 故障时不会无限放大调用；
- user A 的 429/cooldown 或配额耗尽不错误扣减 user B 的用户预算；provider 全局熔断仍对所有用户生效。

### 7.9 RecoveryCoordinator

**Responsibility:** 把过期 lease 和未发布 Outbox 转成可审计的恢复决定。

**Public interface:**

```python
class RecoveryCoordinator:
    def publish_outbox(self, *, limit: int) -> PublishSummary: ...
    def reconcile_queue(self, *, limit: int) -> ReconcileSummary: ...
    def recover_expired_attempts(self, *, limit: int) -> RecoverySummary: ...
```

**Recovery classification:**

| Condition | Decision |
| --- | --- |
| no checkpoint, no Tool began | requeue from start |
| compatible checkpoint, no inflight non-idempotent key | requeue with resume |
| idempotent Tool may repeat | requeue with same business key |
| non-idempotent inflight key exists | `UNCERTAIN` |
| checkpoint incompatible/corrupt | `FAILED` and preserve checkpoint |
| terminal Run has duplicate message | ACK only |

**Invariants:**

- 恢复决定写 Event；
- 同一 expired Attempt 只恢复一次；
- `UNCERTAIN` 不自动重新排队；
- Redis queue depth 不是恢复真相，PostgreSQL 才是。

### 7.10 ObservabilityRecorder

**Responsibility:** 记录服务可靠性和容量指标，并执行公开内容脱敏。

**Minimum metrics:**

- HTTP request count/error rate；
- Run status count；
- PostgreSQL executable backlog；
- Redis Stream length/Pending Entries；
- Provider latency、429、timeout；
- Run end-to-end latency；
- retry count，按 transport/provider/run/message 分层；
- Approval waiting time；
- recovery count；
- Worker queue wait、active time、lease expiration；
- process CPU、memory 和 DB/Redis connection usage。

**Invariants:**

- labels 不包含 `run_id`、`session_id`、用户文本等高基数字段；
- Prometheus labels 默认也不包含原始 `user_id`；per-user quota counter 用 Redis/数据库 key 实现，指标只暴露聚合分布，用户明细通过受控审计查询获取；
- 公开 Event 只保存 metadata 或 sanitized payload；
- Event/API 查询始终按 `RequestContext.user_id` 限定，观测能力不能成为跨用户旁路；
- `output/private` 和 full-private checkpoint 不进入日志。

## 8. Data Model

所有用户拥有或用户操作产生的表都保存 `user_id`。即使 `run_id` 等 ID 全局唯一，Repository 也不得省略 owner predicate；子表使用 user-scoped foreign key，阻止因代码错误把用户 A 的子记录挂到用户 B 的 Run。

### 8.1 `users`

```text
users:
  user_id PRIMARY KEY
  external_issuer NULLABLE
  external_subject NULLABLE
  status
  created_at / updated_at
  UNIQUE(external_issuer, external_subject)
```

开发/测试通过 seed/fixture 建立用户；不提供注册或登录 API。`external_issuer + external_subject` 是未来可信身份提供方的映射 seam，不允许业务请求自行填写。未来若出现共享 workspace，再新增 workspace/membership 表并设计迁移，不改变 v1 已有 `user_id` 所有权事实。

### 8.2 `runs`

```text
run_id UUID/ULID PRIMARY KEY
user_id NOT NULL
created_request_id NOT NULL
session_id TEXT NOT NULL
task_type TEXT NOT NULL
status TEXT NOT NULL
status_version BIGINT NOT NULL
dispatch_generation BIGINT NOT NULL
idempotency_key TEXT NOT NULL
request_hash CHAR(64) NOT NULL
request_payload JSONB / input_ref TEXT
current_stage TEXT
progress JSONB
result_payload JSONB / result_ref TEXT
error_code TEXT
retry_count INTEGER NOT NULL
next_attempt_at TIMESTAMPTZ
created_at / updated_at / finished_at TIMESTAMPTZ
```

约束：

```text
UNIQUE(user_id, run_id)
UNIQUE(user_id, idempotency_key)
FOREIGN KEY (user_id) REFERENCES users(user_id)
CHECK(valid status)
CHECK(status_version >= 0)
CHECK(dispatch_generation >= 1)
```

### 8.3 `run_attempts`

```text
attempt_id PRIMARY KEY
user_id NOT NULL
run_id NOT NULL
attempt_no
message_id
dispatch_generation
worker_id
lease_token
lease_expires_at
heartbeat_at
status
provider_call_count
error_code
recovery_kind
started_at / finished_at
UNIQUE(user_id, attempt_id)
FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
```

### 8.4 `run_approvals`

```text
approval_id PRIMARY KEY
user_id NOT NULL
run_id NOT NULL
session_id NOT NULL
request_id NOT NULL
action_digest NOT NULL
pending_action_json
decision NULL | approved | rejected
reason
version
resolved_by_user_id NULLABLE
created_at / resolved_at
UNIQUE(user_id, approval_id)
UNIQUE(user_id, run_id, request_id)
FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
FOREIGN KEY (resolved_by_user_id) REFERENCES users(user_id)
```

服务层 Approval envelope 绑定 `(user_id, session_id, run_id, request_id, action_digest)`；其中 session/run/digest 的语义验证仍委托现有 `PolicyEngine`，服务层只增加用户归属、操作人审计和 CAS。v1 中 `resolved_by_user_id` 必须等于 Run owner。

### 8.5 `run_events`

```text
event_id PRIMARY KEY
user_id NOT NULL
run_id NOT NULL
attempt_id NULLABLE
actor_user_id NULLABLE
sequence BIGINT
event_type
payload_sanitized JSONB
created_at
UNIQUE(user_id, event_id)
UNIQUE(user_id, run_id, sequence)
FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
FOREIGN KEY (user_id, attempt_id) REFERENCES run_attempts(user_id, attempt_id)
FOREIGN KEY (actor_user_id) REFERENCES users(user_id)
```

`actor_user_id` 为 null 表示系统/Worker 事件；非 null 时 v1 必须等于该 Event 的 owner `user_id`，不能借审计字段建立跨用户引用。

### 8.6 `run_checkpoints`

```text
user_id NOT NULL
run_id NOT NULL
checkpoint_kind
checkpoint_id
checkpoint_version
payload_private JSONB
updated_at
expires_at NULLABLE
PRIMARY KEY (user_id, run_id)
FOREIGN KEY (user_id, run_id) REFERENCES runs(user_id, run_id)
```

### 8.7 `outbox_messages`

```text
message_id PRIMARY KEY
user_id NOT NULL
aggregate_id = run_id
message_type = run.dispatch
dispatch_generation
payload JSONB
created_at
published_at NULLABLE
publish_attempts
last_error_code
UNIQUE(user_id, message_id)
FOREIGN KEY (user_id, aggregate_id) REFERENCES runs(user_id, run_id)
```

Outbox payload 只包含 `user_id`、`run_id`、`dispatch_generation` 和调度 metadata，不包含私人业务内容。Publisher 不按消息 payload 自行决定归属，而是用 `(user_id, aggregate_id)` 关联权威 Run。

### 8.8 User Ownership and Session Serialization

```text
idempotency identity       = (user_id, idempotency_key)
session serialization key  = (user_id, session_id)
resource lookup key        = (user_id, resource_id)
worker claim key           = (user_id, run_id, dispatch_generation)
```

“同 session 至多一个可变更 Run”可通过 user-scoped advisory lock、lock table 或可验证的部分唯一约束实现；最终实现前先用 PostgreSQL 集成测试证明并发行为。所有 Repository SQL 测试都必须检查生成的查询或实际结果包含 owner predicate。

### 8.9 Private Storage

```text
storage/users/{user_id}/runs/{run_id}/
storage/users/{user_id}/career/
```

- 路径只能由服务端从已验证 ID 派生；客户端不得提交绝对路径或物理文件路径。
- logical artifact reference 必须解析并规范化后仍位于 user root 下；symlink/reparse-point escape 也必须拒绝。
- Checkpoint、result 和 Career artifact 的 ACL/retention 均以 user 为范围；公开 Event 只保存脱敏引用。
- 对象存储替代本地文件系统时，object key 仍使用同一 user prefix 合同，不能依赖 bucket 名称隐式隔离。

## 9. HTTP API

所有 `/api/v1` 用例先由 `RequestIdentityAdapter` 产生 `RequestContext`。以下 JSON 示例故意不含 `user_id`；若客户端提交该字段，strict schema 返回 `422`，而不是静默忽略。开发/测试身份来自受控 fixture/配置，生产身份接入属于部署前置条件，不在本阶段增加登录端点。

### 9.1 Create Run

```http
POST /api/v1/runs
Idempotency-Key: required
```

```json
{
  "task_type": "semantic_job_flow",
  "session_id": "session-001",
  "input": {
    "selected_job": {},
    "resume_ref": "artifact://..."
  },
  "provider_profile": "mock",
  "budget_profile": "quick"
}
```

```http
202 Accepted
Location: /api/v1/runs/{run_id}
```

同一 key 和同一 canonical request hash 返回同一 `run_id`；同一 key、不同 hash 返回 `409`。
上述“同一 key”只在当前 `RequestContext.user_id` 内成立；不同 user 使用相同 key 各自创建 Run。

### 9.2 Get Run

```http
GET /api/v1/runs/{run_id}
```

返回 status、version、stage、progress、retry、Approval 摘要、sanitized error 和结果引用。
查询条件必须是 `(context.user_id, run_id)`；其他 user 的 `run_id` 返回与不存在资源相同的 `404 resource_not_found`。

### 9.3 Decide Approval

```http
POST /api/v1/runs/{run_id}/approve
POST /api/v1/runs/{run_id}/reject
```

```json
{
  "approval_id": "approval-...",
  "action_digest": "sha256...",
  "expected_status_version": 4,
  "reason": "optional"
}
```

Approve/Reject 先验证 Run owner、当前用户、`session_id + run_id + request_id + action_digest` 和状态版本，再持久化带 `resolved_by_user_id` 的 `ApprovalDecision`，最后创建新 dispatch generation。Reject 不直接伪造 Runtime 失败，而是让原有 AgentLoop 消费 denied observation。其他 user 对该 Run 的请求返回 `404`，不产生 Approval/Event/Outbox。

### 9.4 Resume

```http
POST /api/v1/runs/{run_id}/resume
```

仅用于已明确可安全恢复的 `FAILED` Run。`WAITING_APPROVAL` 使用 approve/reject；`UNCERTAIN` 返回 `409 uncertain_requires_reconciliation`。
只有 Run owner 可以调用；变更事件记录 `actor_user_id`。

### 9.5 Events

```http
GET /api/v1/runs/{run_id}/events?after=42&limit=100
```

Phase 2 使用游标分页；后续可在同一路径按 `Accept: text/event-stream` 增加 SSE。
Event 分页查询必须同时限定 `user_id + run_id`，游标只在该 Run 内有效；跨 user 复用游标不能泄漏事件数量或存在性。

## 10. State Machine

```text
QUEUED
  -> RUNNING
  -> CANCELLED

RUNNING
  -> SUCCEEDED
  -> WAITING_APPROVAL
  -> QUEUED             retryable + safe checkpoint
  -> FAILED
  -> CANCELLED
  -> UNCERTAIN

WAITING_APPROVAL
  -> QUEUED             approve/reject creates next generation

FAILED
  -> QUEUED             explicit safe resume

UNCERTAIN
  -> no automatic transition
```

所有状态迁移必须由一个集中式 transition policy 校验，Repository 同时执行 CAS。

## 11. Idempotency Model

### 11.1 HTTP Idempotency

```text
canonical_request = stable JSON(task_type, session_id, input refs, provider profile, budget)
request_hash = SHA-256(canonical_request)
identity = (RequestContext.user_id, Idempotency-Key)
```

数据库唯一约束而不是进程内 cache 提供最终保证。
`user_id` 进入 v1 唯一键，使不同用户可以安全复用同一客户端生成策略。未来若引入共享 workspace，是否增加新的幂等作用域必须通过版本化 API/迁移决策，不能静默改变现有 key 的含义。

### 11.2 Message Idempotency

每次从不可运行状态进入 `QUEUED`，`dispatch_generation += 1`。消息携带 generation；Worker claim 必须同时匹配：

```text
run_id
user_id
status == QUEUED
dispatch_generation
next_attempt_at <= now
```

旧消息即使再次投递，也无法 claim 新 generation。
若消息 `user_id` 与数据库 Run owner 不一致，即使 `run_id/generation` 匹配也不能 claim；该消息进入安全告警并 ACK/隔离，防止毒消息反复投递。

### 11.3 Agent and Tool Idempotency

- completed action ID 防止同一 checkpoint 内重复 action；
- 非幂等 Tool 使用 tool name + canonical arguments 生成 execution key；
- 真正幂等 Tool 必须拥有业务 idempotency key 或可验证的目标状态；
- optimistic revision 防止 tracker 并发覆盖；
- `RuntimeToolSpec.idempotent=True` 只是声明，不能自动提供 exactly-once。

## 12. Crash and Recovery Matrix

| Crash point | State evidence | Recovery |
| --- | --- | --- |
| before Redis delivery | Outbox exists | Publisher retries |
| after delivery, before DB claim | PEL entry | `XAUTOCLAIM`, then claim |
| after claim, before model | expired lease, no new checkpoint | requeue from prior checkpoint/start |
| during Provider call | lease/checkpoint before call | retry per Provider/Run budget |
| model returned, state not saved | no external side effect committed | repeat model; record possible duplicate billing |
| idempotent Tool returned, checkpoint not saved | business idempotency key | safe replay |
| non-idempotent Tool may have run | inflight key in checkpoint | `UNCERTAIN` |
| WAITING_APPROVAL committed | Approval + checkpoint durable | remain waiting after restart |
| terminal DB commit, before ACK | terminal Run | duplicate message ACK only |
| PostgreSQL connection lost during commit | stable operation IDs | query outcome before retry |

## 13. Retry Policy

### 13.1 Retryable

- `rate_limit`；
- `timeout`；
- `transport_error`；
- Redis unavailable；
- PostgreSQL serialization/deadlock；
- Worker crash before uncertain side effect；
- explicitly idempotent Tool transient failure。

### 13.2 Non-Retryable

- schema/contract failure；
- Tool input/output validation failure；
- policy denial；
- Approval digest/scope mismatch；
- checkpoint identity/version mismatch；
- verifier repeated failure；
- exhausted budget。

### 13.3 Uncertain

- non-idempotent inflight execution；
- external timeout with no idempotency key and no queryable receipt；
- side effect committed but local receipt outcome cannot be established。

### 13.4 Backoff

```text
delay = min(cap, base * 2^attempt) * random(0.5, 1.5)
```

同时限制：

- `max_attempts`；
- `max_retry_elapsed`；
- `next_attempt_at`；
- provider circuit cooldown。

不能用 `catch Exception` 或无限 retry 代替错误分类。

## 14. Queue Technology Decision — ADR-001

### 14.1 Decision

Phase 3 选择 **Redis Streams consumer group**，但 PostgreSQL 始终是 Run/Outbox 的权威来源。

选择 Redis Streams 不是因为它能提供 exactly-once，而是因为：

1. 当前任务流只有一个主要消费者角色：Agent Worker；
2. 任务是低吞吐、长耗时，瓶颈预计在 Provider 和 Tool，而不是 broker；
3. Redis 同时解决短期限流、provider cooldown 和近实时 queue/Pending depth；
4. consumer group、PEL、`XACK`、`XAUTOCLAIM` 已足以实现所需的至少一次唤醒；
5. PostgreSQL Outbox 和 Reconciler 使 Stream 可以丢失并重建；
6. 本项目目标是学习可靠性合同，而不是运维事件平台。

### 14.2 Why Not Kafka Now

Kafka 能提供更强的持久日志、分区扩展、多 consumer group 独立回放和成熟的跨服务事件平台，但当前没有对应需求：

- 没有多个独立下游团队需要消费同一 Run Event；
- 不要求以消息总线作为多年审计日志；
- 没有测量结果显示 Redis 或 PostgreSQL dispatch 成为吞吐瓶颈；
- 没有需要按大量 partition 扩展的短任务流；
- Kafka transaction/exactly-once 不能让任意 Tool 外部副作用 exactly-once，仍需业务幂等和 `UNCERTAIN`；
- Kafka/KRaft、partition、retention、rebalance、lag 和本地 Compose 运维会扩大当前学习面。

在此项目里提前使用 Kafka，主要增加的是运维和故障模式，不会消除 Run Repository、Outbox、fencing 或 Tool idempotency 的必要性。

### 14.3 Strongest Simpler Alternative: PostgreSQL Queue

可使用：

```sql
SELECT ...
FROM runs
WHERE status = 'QUEUED' AND next_attempt_at <= now()
FOR UPDATE SKIP LOCKED
LIMIT ...
```

优点：

- 一个数据系统；
- Run 创建和排队天然同事务；
- 无 Outbox dual-write；
- 对当前低吞吐长任务可能已经足够。

缺点：

- polling latency；
- 大量空轮询和锁竞争会消耗 DB connection/IO；
- provider limiter、cooldown 和短期计数仍需另一个协调层或数据库实现；
- queue backlog 与业务查询争用同一数据库。

如果 Tracer Bullet 后测得 PostgreSQL polling 已完全满足延迟和资源目标，真实产品可以不引入 Redis Queue；但本项目按已确认路线使用 Redis Streams，以学习至少一次消息、PEL reclaim 和 Redis 故障恢复。

### 14.4 Other Alternatives

| Option | Why not selected now |
| --- | --- |
| Redis List + `BRPOP` | pop 后到 DB claim 前需要额外可靠性协议；Streams 的 PEL 更直接 |
| Celery/RQ | 隐藏部分消息细节，不利于本项目学习 at-least-once、claim 和 recovery |
| RabbitMQ | 功能可行，但项目还需要 Redis limiter；会增加第二套 broker 运维 |
| Kafka | 当前缺少多订阅者、长期日志和已测量吞吐需求 |
| PostgreSQL only | 最简单且有效；保留为对照与可替代方案 |

### 14.5 Kafka Reconsideration Triggers

只有出现以下经测量需求时重新评估 Kafka：

- 多个独立 consumer group 必须按自己的 offset 重放 Run/Event；
- 消息日志本身需要长期 retention，并成为跨服务审计事实；
- 单 Redis Stream/consumer group 在目标硬件上成为已定位瓶颈；
- backlog 超出可接受 Redis memory 和恢复窗口；
- 需要跨多个服务和团队统一事件平台；
- 团队已拥有 Kafka 运维、监控和容量治理能力。

迁移 Kafka 时也不删除 PostgreSQL Run Repository、业务幂等、fencing 和 `UNCERTAIN`。

## 15. Backpressure

### 15.1 Admission

- 使用 PostgreSQL executable backlog 作为权威 admission 信号；
- 同时设置 global hard cap 和 per-user queue/in-flight cap；某个 user 的积压不能挤占其他 user 的保留容量；
- 当前 user 超过上限时返回 `429 + Retry-After`，不创建新 Run；
- 相同 Idempotency-Key 的已存在请求仍可返回已有 Run。

### 15.2 Worker

- 从小并发开始；
- 同 `(user_id, session_id)` 串行；不同 user 的同名 session 可以并行；
- Worker scheduler 使用 per-user in-flight 上限；per-user backlog cap 给其他 user 保留可进入队列的容量，使延迟受单 user 最大 backlog 约束；
- Worker 进程数与 Provider 并发独立配置；
- 队列长度高时不自动无限扩容 Provider calls。

v1 不声称实现严格 weighted fair queueing。若压测证明单 Stream 顺序仍造成不可接受的 user starvation，再在保持 PostgreSQL 权威 claim 的前提下评估 user lanes 或公平调度器；不能因为“多用户”就直接跳到 Kafka。

### 15.3 Provider

- global per-provider/model concurrency；
- per-user requests/minute、tokens/minute、并发和 retry budget；
- 429 cooldown；
- timeout、retry budget 和 circuit breaker；
- 等待许可不占 Agent active-time budget。

### 15.4 Little's Law

```text
L = lambda * W
```

长耗时 Run 即使 HTTP QPS 很低，也可能形成大量在途任务。容量规划必须同时报告 arrival rate、平均/分位 Run latency 和 in-flight backlog，不能只报 API QPS。

## 16. Test Matrix

| Required scenario | Injection point | Expected assertion |
| --- | --- | --- |
| duplicate Idempotency-Key | concurrent API calls | one Run, same response identity |
| Worker crash before claim | stream delivery | PEL reclaim, one Attempt |
| Worker crash before model | after DB claim | lease recovery, safe retry |
| model complete before state write | adapter failpoint | model may repeat, committed Tool does not |
| WAITING_APPROVAL restart | after checkpoint/status commit | still waiting, no model/tool call |
| Approval digest mismatch | API + Policy | `409`, no dispatch/tool |
| Provider 429 | scripted Provider | bounded retry/cooldown/metric |
| Provider timeout | scripted Provider | bounded safe retry |
| Redis unavailable | publish/consume | Outbox retained, later dispatch |
| PostgreSQL rollback | create/transition | no partial Run/Event/Outbox |
| non-idempotent uncertain | after handler before receipt | `UNCERTAIN`, no automatic replay |
| stale Worker result | after lease takeover | fencing rejects write |
| duplicate ACK/message | terminal Run | no extra Attempt/tool |
| checkpoint corruption | load | fail closed, preserve evidence |
| private output scan | API/Event/metrics | no raw/private/credential content |
| cross-user Run read | user A requests user B Run | `404`, no existence/detail leak |
| cross-user Event read | user A lists user B events | `404`, no event/count leak |
| cross-user mutation | A approve/reject/resume B Run | `404`, no state/Event/Outbox change |
| same key across users | A and B submit same Idempotency-Key | two isolated Runs |
| same session across users | A and B run same session ID | no serialization conflict |
| per-user quota isolation | A exhausts queue/provider budget | B remains admissible within global cap |
| queue owner mismatch | forged/stale Redis message | no claim/Agent call; security event |
| user storage escape | absolute/path traversal/symlink ref | request rejected outside user root |
| identity body spoofing | JSON includes `user_id` | strict schema rejection |
| production identity safety | dev Header Adapter in production profile | service refuses startup |

## 17. Delivery Sequence

### Phase 1 — Design

- 本规格；
- ADR-001；
- 不写服务代码。

### Phase 2 — Tracer Bullet

1. [x] `RequestContext`、dev/test identity fixture、Service Pydantic contracts and state transition policy；
2. [x] user-aware in-memory `RunRepository` contract implementation；
3. [x] In-process deterministic `DispatchQueue`；
4. [x] user fixture/snapshot `CareerContextAdapter` and `SemanticJobFlowAdapter` using existing Mock Provider；
5. [x] WorkerEngine happy path；
6. [x] FastAPI create/get/events endpoints；
7. [x] duplicate Idempotency-Key and mandatory cross-user isolation E2E。

Tracer Bullet 完成标志：两个受控测试用户各自创建真实现有 semantic Agent flow，经 Worker 执行并由 GET 返回结果；同一 user 重复 key 仍是同一 Run，不同 user 使用相同 key 是两个 Run，且双方不能读取对方 Run/Event。

### Phase 3 — Persistence and Queue

1. [x] user-scoped PostgreSQL schema/repository and identity seeds；
2. [x] Outbox Publisher；
3. [x] Redis Streams adapter；
4. [x] Worker lease/heartbeat/fencing；
5. [x] user-scoped PostgreSQL checkpoint and Approval persistence adapters；
6. [x] Docker Compose；
7. [x] repository/queue/storage user-isolation integration tests。

Phase 3 不自动迁移 `CareerStore`。若交付目标要求多个 Worker 节点共同修改 Career 业务数据，则在进入该部署形态前增加独立 PostgreSQL `CareerRepository` Adapter 阶段，并用现有 CareerStore 合同测试证明语义一致；否则明确限定 Career mutable workflow 为单节点 user-root 模式。

### Phase 4 — Reliability

1. Approval endpoints and Domain Agent adapter；
2. failure classifier；
3. RecoveryCoordinator；
4. provider admission/circuit；
5. ten required fault scenarios；
6. existing 821 regression。

### Phase 5 — Observability and Load

1. minimum metrics；
2. Mock Provider latency/error controls；
3. Locust or k6；
4. HTTP/control-plane report；
5. Worker throughput report；
6. Mock Provider report；
7. real Provider explicitly marked unmeasured or separately measured。

### Phase 6 — Learning Document

新增 `docs/learning/backend_service_from_zero_zh.md`，使用“直觉 -> 合同/公式 -> 代码”结构，并列出概念到代码的映射和至少 20 个面试追问。

## 18. Acceptance Criteria

每阶段必须：

1. 列出修改文件；
2. 运行聚焦测试；
3. 运行相关全量回归；
4. 说明新增合同；
5. 说明失败边界；
6. 更新架构/学习文档；
7. 标出用户必须亲自理解的代码位置；
8. 不自动修改简历；
9. 不自动 Push。

最终交付必须满足：

- 所有要求的 API 合同有自动化测试；
- 所有用户拥有资源的 Repository/API/queue/storage 路径通过跨用户隔离测试；
- 所有十个可靠性场景有明确 PASS/FAIL/NOT_PROVEN；
- 821 项既有回归语义与结果口径未被篡改；
- Docker Compose 可以启动 API、Worker、PostgreSQL 和 Redis；
- Mock 压测报告包含 p50/p95/p99、错误率、资源使用和明确免责声明；
- 学习文档完成并链接到实际代码；
- 只有在上述全部完成后才提出简历修改建议。

## 19. Risks

1. **Cooperative timeout**：永久阻塞的 in-process Tool 不能被安全终止；后续若接入不可信 Tool，必须使用可杀 Worker process。
2. **Private checkpoint**：PostgreSQL checkpoint 可能包含用户材料，需要访问控制、retention 和备份策略。
3. **Duplicate Provider billing**：模型返回后、checkpoint 前崩溃可能导致重复调用；服务不能声称 exactly-once billing。
4. **SQLite CareerStore concurrency**：只允许作为用户隔离的业务上下文来源；不得被多个 Worker 当作 Run Store。共享可变 Career 数据在多节点下没有 PostgreSQL Adapter 就不具备一致性保证。
5. **Session serialization**：Approval 长时间等待会阻塞同 user/session 后续可变更 Run，这是保护一致性的有意选择，不应阻塞其他 user 的同名 session。
6. **Retry layering**：existing Provider retry、service Run retry、Redis redelivery 必须分别计数，避免乘法放大。
7. **Redis memory**：Stream retention/trim 必须以已 ACK 且 PostgreSQL 可重建为前提。
8. **Scope creep**：普通用户输入、Adaptive Interview、取消 API、真实外部 Action 均不能偷偷进入 Tracer Bullet。
9. **身份伪造**：开发 Header Adapter 若误进生产，owner filter 也会建立在伪造上下文上；必须用 profile 启动检查和生产部署测试阻断。
10. **遗漏 owner predicate**：应用层漏写一次过滤就可能形成越权；user-scoped Repository interface、复合外键、集成测试和后续 RLS defense-in-depth 共同降低风险。
11. **Noisy neighbor**：仅做全局背压会让单个用户占满队列/Provider 许可；必须同时实施 per-user 与 global cap，并测试隔离。

## 20. Open Questions

以下问题不阻塞 Tracer Bullet，将在对应阶段用测量或安全策略回答：

1. PostgreSQL-only queue 与 Redis Streams 在目标机器上的 queue latency/DB resource 对比是多少？
2. Redis Stream 的 retention、`MAXLEN` 和 PEL reclaim interval 应根据何种实际 backlog 设置？
3. private checkpoint 的 retention 是否沿用 Semantic Checkpoint 的 30 天，还是按 Run 类型区分？
4. 是否增加独立 `/cancel` API；如果增加，哪些 Tool 支持 cooperative cancellation？
5. Provider transport 如何保留并利用 `Retry-After`，同时避免泄漏响应 header？
6. `WAITING_INPUT` 与 Adaptive Interview 服务化是否作为单独 PRD，而不是本项目尾部扩展？
7. 如果未来出现共享 Career workspace，采用 owner-only、workspace-shared 还是细粒度 RBAC？该问题不影响 v1，因为当前明确只有 user-owned 资源并统一拒绝跨用户访问。

## 21. Ambiguity Report

```text
Ambiguity Report:
  Goals:        0.0   ✓ clear
  Acceptance:   0.0   ✓ clear
  Boundaries:   0.0   ✓ clear
  Alternatives: 0.25  ✓ queue and user-only versus tenant identity alternatives and triggers recorded
  Assumptions:  0.25  ✓ load/retention/fairness values are measured later; trusted production identity and multi-node Career boundaries are explicit gates
  ──────────────────────────────
  Aggregate:    0.10  ✓ below threshold (0.2 spec)

Push lightly on: measured Redis-vs-PostgreSQL queue behavior, per-user scheduling fairness, and whether mutable Career workflows need multi-node deployment.
```

## 22. Phase 2 Implementation Record

### 22.1 Implemented Vertical Slice

```text
POST /api/v1/runs
  -> ConfiguredHeaderIdentityAdapter -> RequestContext(user_id, request_id)
  -> RunService -> InMemoryRunRepository.create_or_get
  -> InMemoryDispatchQueue
  -> WorkerEngine thread
  -> SemanticJobExecutionAdapter
  -> existing run_semantic_job_flow
  -> existing LLMHarness + MockLLMProvider
  -> InMemoryRunRepository status/events/result
  -> GET run / GET events
```

实现文件映射：

| Contract | File |
| --- | --- |
| API/Pydantic/Run state | `src/job_agent/backend_service/contracts.py` |
| user-scoped authoritative tracer state | `src/job_agent/backend_service/repository.py` |
| process-local wake-up queue | `src/job_agent/backend_service/queueing.py` |
| Career snapshot + existing Runtime adapter | `src/job_agent/backend_service/execution.py` |
| Idempotent application use cases | `src/job_agent/backend_service/service.py` |
| asynchronous Worker lifecycle | `src/job_agent/backend_service/worker.py` |
| FastAPI + controlled identity | `src/job_agent/backend_service/api.py` |
| composition and deployable Mock entrypoint | `src/job_agent/backend_service/bootstrap.py`, `main.py` |
| behavior tests | `tests/test_backend_service.py` |

### 22.2 New Contracts Proven in Phase 2

- `POST` returns `202 + run_id` without waiting for Worker completion.
- `(user_id, Idempotency-Key)` and canonical request hash provide process-local HTTP idempotency.
- Same user/key/same payload returns one Run and schedules Agent exactly once.
- Same user/key/different payload returns `409 idempotency_key_payload_mismatch`.
- Different users may reuse the same key and session ID without collision.
- Run/Event lookup is owner-scoped; cross-user access returns indistinguishable `404`.
- HTTP `request_id` is preserved as internal `created_request_id`; message/Agent correlation uses `run_id`.
- Redis-like message envelope contains only `user_id/run_id/generation/message_id`, never resume text.
- Worker verifies message user against persisted Run owner before Agent execution.
- Public Run/Event views exclude request payload, resume snapshot and raw Provider output.

### 22.3 Explicit Phase 2 Failure Boundaries

- Repository, queue, Career snapshots and Events are in memory; process restart loses all state.
- Run creation and queue publish are not atomic in this tracer; a publish failure can leave a queued Run without wake-up. Phase 3 Outbox closes this window.
- Queue delivery is process-local and does not yet implement PEL, reclaim, lease, heartbeat or fencing.
- No persisted Attempt, Approval, Checkpoint or crash recovery is claimed by the service layer yet.
- Only controlled Mock Provider execution is supported; no true Provider throughput or SLA is measured.
- Known Agent/Schema/Artifact/IO failures map to bounded error codes; unknown programming errors are not blindly retried.
- `ConfiguredHeaderIdentityAdapter` is dev/test-only and construction fails in production mode.
- Career snapshots are user-scoped fixtures; mutable multi-node Career workflows still require the later Repository adapter.

### 22.4 Verification Evidence

```text
python -m pytest -q tests/test_backend_service.py
10 passed

python -m pytest -q \
  tests/test_agent_graph_runtime.py \
  tests/test_semantic_agents.py \
  tests/test_llm_agentization_runtime.py
54 passed

python -m pytest -q \
  --ignore=tests/test_evidence_eval_corpus_v2.py \
  --ignore=tests/test_evidence_eval_corpus_v5.py \
  --ignore=tests/test_evidence_live_eval.py
233 passed, 2 existing protobuf warnings
```

未把全量回归写成 PASS：当前 Windows Worktree 中冻结 Evidence JSON 为 CRLF，manifest 记录的 corpus hash 对应 LF，导致 19 个 Evidence hash/下游测试失败。新增代码未修改 `data/`、Evidence 实现或这些测试；按安全要求不规范化 corpus、不更新 manifest，也不改变 821 项研发基线口径。

## 23. Phase 3 Implementation Record

### 23.1 Implemented Deployment Slice

```text
FastAPI API process
  -> PostgreSQL transaction: Run + first Event + Outbox
  -> Outbox Publisher
  -> Redis Stream + consumer group + PEL
  -> Worker DB claim
  -> Attempt lease + heartbeat + fencing token
  -> existing semantic Agent Runtime + Mock Provider
  -> PostgreSQL result/Event/Attempt terminal transaction
  -> XACK
```

PostgreSQL 是唯一权威来源；Redis payload 只有 `message_id/user_id/run_id/dispatch_generation/enqueued_at`，不含请求正文、简历或 Provider 输出。API 不依赖 Redis 可用性即可持久化并返回 `202`；Redis 恢复后 Outbox Publisher 再发布。

### 23.2 Code Map

| Contract | File |
| --- | --- |
| PostgreSQL tables and owner-scoped foreign keys | `src/job_agent/backend_service/migrations/001_backend_service.sql` |
| idempotent Run/Event/Attempt/Approval/Outbox transactions | `src/job_agent/backend_service/postgres_repository.py` |
| existing Checkpoint protocol adapters | `src/job_agent/backend_service/checkpoint_adapter.py` |
| Redis Streams group, PEL, ACK and reclaim | `src/job_agent/backend_service/redis_streams.py` |
| transactional Outbox publisher | `src/job_agent/backend_service/outbox.py` |
| lease heartbeat, fencing and duplicate-message handling | `src/job_agent/backend_service/reliable_worker.py` |
| persistent API/Worker composition | `persistent_bootstrap.py`, `persistent_main.py`, `worker_main.py` |
| deployable stack | `Dockerfile.backend`, `compose.backend.yml` |
| safe build context and dependency layer | `.dockerignore`, `requirements.backend.txt` |
| real PostgreSQL/Redis tests | `tests/test_backend_service_persistence.py` |

### 23.3 Proven Contracts

- `UNIQUE(user_id, idempotency_key)` is the cross-process HTTP idempotency authority; 8 concurrent submissions create one Run and one Outbox row.
- Run, initial Event and Outbox are one PostgreSQL transaction; injected failure after Run insert leaves all three absent.
- Duplicate Stream entries are allowed, but only one can claim matching `QUEUED + generation`; the Agent executes once.
- A claim creates an Attempt lease. Heartbeat and completion require the same `attempt_id + lease_token` and an unexpired DB-clock lease.
- Lease time is computed with PostgreSQL `now()`, not Worker local time; this avoids host/container clock skew.
- Expired Mock/model-only attempts are requeued with a new generation; stale Worker completion is fenced out.
- Recovery is bounded by `JOB_AGENT_MAX_ATTEMPTS` (default 3); exhaustion becomes `FAILED/worker_recovery_exhausted`, not infinite retry.
- Existing Runtime `compute_action_digest`, `ApprovalRequest`, `ApprovalDecision`, `AgentCheckpoint` and `SemanticGraphCheckpoint` remain semantic authorities. PostgreSQL adapters only persist and owner-scope them.
- Semantic Runtime checkpoints are private JSONB, user/run scoped, expire using the existing checkpoint timestamp, and are deleted by the existing Runtime after success.
- Same-user same-session serialization defers a still-valid Stream delivery without ACK; PEL reclaim retries it later. Stale/terminal deliveries are ACKed without execution.

### 23.4 Failure Boundaries After Phase 3

- The deployed vertical slice remains Mock Provider only. Provider 429/timeout admission, backoff and circuit breaking belong to Phase 4.
- Safe automatic lease recovery is enabled only for the current model-only semantic execution profile. A future non-idempotent Tool with an unresolved inflight key must transition to `UNCERTAIN`, never enter this replay path.
- Approval records and checkpoint adapters are durable, but approve/reject HTTP endpoints and `WAITING_APPROVAL -> QUEUED` orchestration remain Phase 4.
- Redis is a wake-up layer, not truth. Stream retention/trim policy and queue-depth metrics remain Phase 5 work.
- Mutable `CareerStore` is not migrated to shared PostgreSQL. The current Worker uses user-scoped Mock snapshots; multi-node mutable Career workflows remain forbidden until a contract-compatible CareerRepository exists.
- Dev/test `X-User-ID` remains allowlisted but unsigned; no registration/login/billing/RBAC is claimed.

### 23.5 Verification Evidence

```text
python -m pytest -q tests/test_backend_service.py tests/test_backend_service_persistence.py
20 passed

docker compose -f compose.backend.yml build api worker
PASS

docker compose -f compose.backend.yml up -d
API + Worker + PostgreSQL + Redis healthy/running

Container E2E:
two identical POST requests -> same run_id
terminal status -> SUCCEEDED
existing Agent Runtime -> Mock Provider calls recorded

python -m pytest -q \
  --ignore=tests/test_evidence_eval_corpus_v2.py \
  --ignore=tests/test_evidence_eval_corpus_v5.py \
  --ignore=tests/test_evidence_live_eval.py
243 passed, 2 existing protobuf warnings

python -m pytest -q
245 passed, 19 known frozen-Evidence CRLF/hash failures, 2 existing warnings
```

第一次镜像构建曾因 Docker Hub token endpoint 网络超时失败；重新拉取 `python:3.12-slim` 后构建通过。该环境波动没有被记作服务性能或可靠性结论。

## 24. Phase 4 Implementation Record

### 24.1 Implemented reliability slice

```text
Redis delivery / PostgreSQL reconciliation
  -> fenced Run claim + lease heartbeat
  -> per-user/global Provider admission
  -> existing AgentLoop through DomainAgentExecutionAdapter
  -> existing AgentCheckpoint + ApprovalDecision contracts
  -> SUCCEEDED / WAITING_APPROVAL / bounded RETRY / FAILED / UNCERTAIN
```

- `approve`、`reject` 和安全 `resume` 均要求 owner、状态版本与已有 Runtime approval 合同；相同决定幂等，相反决定冲突。
- `DomainAgentExecutionAdapter` 只负责装配已有 `AgentLoop`、PostgreSQL checkpoint adapter、已决 approval 和 ToolContext；action digest 校验、Policy、ToolExecutor、非幂等 inflight key 仍由原 Runtime 决定。
- Provider 429/timeout/transport failure 仅按显式 allowlist 有界重试；未知错误 fail closed；非幂等副作用结果不确定直接进入 `UNCERTAIN`。
- Redis Stream/PEL 都为空时，RecoveryCoordinator 从 PostgreSQL 的 ready `QUEUED` Run 重建 dispatch；Redis 始终不是 Run 真相。
- Provider admission rejection 只推迟 Run，不消耗 Provider retry budget；Redis limiter 不可用时拒绝准入，避免无保护地放大请求。

### 24.2 Reliability evidence matrix

| Scenario | Result | Proven boundary |
| --- | --- | --- |
| duplicate `Idempotency-Key` | PASS | PostgreSQL unique owner/key returns one Run |
| Worker crash before processing | PASS | Stream PEL reclaim processes the same generation |
| model completed before state commit crash | PASS | lease expiry requeues; stale completion is fenced |
| restart while `WAITING_APPROVAL` | PASS | checkpoint and approval survive repository restart |
| approval digest mismatch | PASS | `409`, no requeue or Tool call |
| Provider 429 | PASS | bounded retry plus user cooldown |
| Provider timeout | PASS | bounded retry; exhaustion is explicit failure |
| Redis temporarily unavailable | PASS | Outbox remains pending; PostgreSQL reconciliation republishes |
| PostgreSQL transaction rollback | PASS | Run/Event/Outbox do not partially commit |
| non-idempotent Tool outcome unknown | PASS | `UNCERTAIN`, no automatic replay/resume |
| real `AgentLoop` approval resume | PASS | same checkpoint/digest; Tool executes once |

`PASS` 只指可控 Mock Provider 和本地 PostgreSQL/Redis 集成测试；不证明真实 LLM 的可用性、吞吐或 exactly-once Provider billing。

### 24.3 Remaining boundaries

- `AgentLoop` 当前会把任意 model exception 归一为 `model_error`；生产 Provider adapter 必须在进入服务重试分类前保留稳定的 429/timeout/transport code。第四阶段测试证明服务策略，不冒充所有未来 Provider SDK 都已正确映射。
- 没有自动 reconciliation endpoint 把 `UNCERTAIN` 改回可运行状态；在存在 Tool-specific receipt/audit 合同前，这是有意限制。
- Redis Stream retention/trim、指标导出和负载测量仍属于第五阶段。
- 登录、RBAC 和计费仍不在范围内；开发 identity adapter 不是生产认证机制。

### 24.4 Verification evidence

```text
python -m pytest -q \
  tests/test_backend_service.py \
  tests/test_backend_service_persistence.py \
  tests/test_backend_service_reliability.py
31 passed

python -m pytest -q \
  --ignore=tests/test_evidence_eval_corpus_v2.py \
  --ignore=tests/test_evidence_eval_corpus_v5.py \
  --ignore=tests/test_evidence_live_eval.py
254 passed, 2 warnings

python -m pytest -q
256 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
```

Compose API/Worker 镜像重新构建成功；部署态相同 `Idempotency-Key` 两次 POST 返回同一个 `run_id`，最终状态为 `SUCCEEDED`。Compose 的 Semantic Mock 任务不触发 approval，因此不把它冒充部署态 approval E2E；approval/restart/真实 `AgentLoop` resume 由 PostgreSQL/Redis 集成测试证明。

短租约测试还揭示 `now()` 是 PostgreSQL transaction-stable timestamp，不适合作为长事务内的墙钟 fencing 判断。Attempt lease 创建、heartbeat、过期扫描和 active-attempt 校验统一使用 `clock_timestamp()`；Run/Event 的事务时间仍使用 `now()`。

## 25. Phase 5 Implementation Record

### 25.1 Metrics contract

- `/metrics` 输出 Prometheus text，不输出 user_id、run_id、Prompt、简历或 Provider raw output。
- PostgreSQL 在 scrape 时聚合 Run status、end-to-end latency、retry、approval waiting 和 recovery；这些是持久业务指标。
- API、Worker 和 Provider call 的短期 counter/histogram 写入 Redis；失败时 fail-open，不改变 Run 结果。
- Queue backlog 定义为 consumer-group lag + PEL pending；`XLEN` 单独暴露为 retained entries，不能冒充待处理任务数。
- Provider label 是受控低基数集合；HTTP route 使用 FastAPI route template，不使用真实 run_id，避免 cardinality explosion。

### 25.2 Load evidence

| Layer | Measured result |
| --- | --- |
| HTTP control plane | 2703 requests, 192.42 req/s, p50/p95/p99 15/30/38 ms, 0 errors |
| single Worker | exact 200 successful Runs, 18.20 runs/s, attempt p50/p95/p99 49.98/60.06/68.04 ms |
| Mock calls inside Worker | 400 calls, 36.40 calls/s, all observed in <=5 ms bucket |
| isolated Mock microbenchmark | 50,000 calls, 28,581.56 calls/s, p50/p95/p99 0.010/0.030/0.048 ms |
| real LLM API | `NOT_PROVEN` |

完整环境、资源峰值、原始 CSV 和限制见 `docs/performance/backend_service_phase5_report_zh.md`。上述数字不设为容量承诺，也不外推到生产或真实模型。

### 25.3 Phase 5 code map

| Concern | File |
| --- | --- |
| metrics recording and Prometheus rendering | `src/job_agent/backend_service/metrics.py` |
| database business aggregates | `src/job_agent/backend_service/postgres_repository.py` |
| request instrumentation | `src/job_agent/backend_service/api.py` |
| Worker and Provider trace instrumentation | `reliable_worker.py`, `execution.py`, `worker_main.py` |
| queue lag/PEL/retention semantics | `redis_streams.py`, `recovery.py` |
| Locust scenario | `load_tests/locustfile.py` |
| Mock microbenchmark | `scripts/benchmark_mock_provider.py` |
| exact Worker backlog seed | `scripts/seed_backend_load.py` |
| measured report | `docs/performance/backend_service_phase5_report_zh.md` |

### 25.4 Verification

```text
Backend focused: 32 passed
Related regression excluding known frozen Evidence modules: 255 passed, 2 warnings
Full regression: 257 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
Docker Compose build: PASS
/metrics integration and Queue lag/PEL/retention tests: PASS
pip check / compileall / git diff --check: PASS
```

## 26. Phase 6 DeepSeek Flash Canary

### 26.1 Provider routing contract

- `CreateRunRequest.provider_profile` 是持久化执行选择的一部分，当前只允许 `mock` 与
  `deepseek_flash`；因此相同 `(user_id, Idempotency-Key)` 改换 Provider 会产生 request-hash
  冲突，而不会悄悄复用不同语义的 Run。
- `RoutedExecutionAdapter` 只负责根据已持久化 profile 选择 Adapter；Semantic Flow、Checkpoint、
  Pydantic Schema 校验和结果投影仍复用现有 Runtime。
- DeepSeek 适配器固定官方 `/v1` endpoint allowlist，只允许 `deepseek-chat` / `deepseek-reasoner`，
  使用 `json_object` 传输模式；模型输出仍由 `LLMHarness` 按目标 Pydantic Schema 严格验证。
- DeepSeek 凭据只从 Worker 环境读取。缺 Key 时返回稳定的 `provider_credential_missing`，禁止
  fallback 到 Mock，避免把假结果写成真实结果。
- Canary policy 设为 Provider retry `0`、schema repair `0`、无 rule fallback。服务层仍保留它自己的
  有界故障分类，但非幂等 Tool 与未知外部副作用合同没有改变。

### 26.2 HTTP、消息和真实调用的关联

```text
HTTP request
  -> persisted Run(request.provider_profile = deepseek_flash)
  -> transactional Outbox(run_id, user_id, dispatch_generation)
  -> Redis Stream wake-up
  -> PostgreSQL claim + fenced Attempt
  -> RoutedExecutionAdapter
  -> DeepSeekCompatibleProvider
  -> existing Semantic Runtime / Checkpoint
  -> PostgreSQL terminal result
```

Redis 消息不携带 API Key，也不决定 Provider；Worker 必须重新读取 PostgreSQL 中已持久化的 Run。
这保持了“PostgreSQL 是业务真相、Redis 只是可恢复唤醒”的既有合同。

### 26.3 Measured evidence

2026-08-02 只执行了小规模真实 Canary：

| Scenario | Result |
| --- | --- |
| direct API smoke | 1.208 s；47 input / 11 output tokens；合法 JSON |
| requested / returned model | `deepseek-chat` / `deepseek-v4-flash` |
| full backend single Run | `SUCCEEDED`；4.621 s E2E；2 LLM calls；0 retries |
| four-Run two-user batch | 4/4 `SUCCEEDED`；20 LLM calls；0 retries |
| batch E2E | p50 39.431 s；p95/p99/max 77.822 s；min 18.969 s |
| accumulated DeepSeek metrics | 22 calls；79.617 provider-seconds |

Token 聚合指标是在上述 22 次调用之后接入，历史总 token/cost 不可追溯；只保留直接冒烟已知的
47/11 token，不做推算。4 Run 样本也不足以给出稳定分位数或容量结论。该实验不是压测，不能代表
真实模型吞吐、SLA 或生产成本。单 Worker 串行执行是当前批次最明显的排队瓶颈。

### 26.4 Observability and security

- `/metrics` 新增 `job_agent_provider_tokens_total{provider,direction}`；只聚合 token 数，不记录
  prompt、response、user_id、run_id 或 credential。
- Provider latency 使用受控低基数 `provider="deepseek"` label；请求模型与供应商返回模型只进入
  Canary 报告，不把任意模型字符串变成 Prometheus label。
- 真实响应正文未写入报告。API Key 未写入配置、日志、测试 fixture 或 Git diff。
- Compose 默认不注入真实 Key；生产环境应由部署平台的 Secret 机制注入 Worker。

### 26.5 Code and test map

| Concern | File |
| --- | --- |
| DeepSeek endpoint/model allowlist and JSON mode | `src/job_agent/llm/providers/deepseek_compatible.py` |
| OpenAI-compatible JSON Schema/Object transport | `src/job_agent/llm/providers/openai_compatible.py` |
| persisted profile contract | `src/job_agent/backend_service/contracts.py` |
| provider routing and semantic Runtime reuse | `src/job_agent/backend_service/execution.py` |
| Worker credential injection and strict policy | `src/job_agent/backend_service/worker_main.py` |
| Provider call/latency/token metrics | `src/job_agent/backend_service/metrics.py` |
| adapter and routing tests | `tests/test_llm_agentization_runtime.py`, `tests/test_backend_service_reliability.py` |
| persistent metric tests | `tests/test_backend_service_persistence.py` |
| measured report | `docs/performance/backend_service_phase6_deepseek_canary_zh.md` |

### 26.6 Verification

```text
LLM + Backend focused: 59 passed
Related regression excluding known frozen Evidence modules: 258 passed, 2 warnings
Full regression: 260 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
compileall / pip check / git diff --check: PASS
broad exception and unbounded-loop scan: no matches
Docker Compose API/Worker rebuild: PASS
deployment Mock idempotency + terminal Run: same run_id / SUCCEEDED
deployment DeepSeek without credential: FAILED / provider_credential_missing
```

部署验证没有向 Compose Worker 注入真实 Key，因此最后一项没有产生真实 API 调用。真实调用证据来自
26.3 的有界 Canary；部署态验证与真实调用验证刻意分开，防止在常规 E2E 中意外产生费用。

## 27. Phase 7 Multi-Worker Scaling and Fairness

### 27.1 Worker identity contract

Compose 不再固定 `JOB_AGENT_WORKER_ID`。未显式配置时，`BackendSettings` 使用容器 hostname + UUID，
因此每个副本拥有独立 Redis consumer name 和 PostgreSQL attempt `worker_id`。生产平台仍可显式注入
稳定实例 ID，但多个同时存活的副本不得共享同一身份。

### 27.2 Delayed Outbox root cause and fix

扩容基线发现 2/4 Worker 出现 39–46 秒尾延迟。根因链路为：

```text
provider admission reject
-> Run.next_attempt_at = now + 250 ms
-> Outbox inserted in same transaction
-> publisher ignored next_attempt_at and published immediately
-> Worker claim raised RunDispatchDeferredError
-> delivery stayed in PEL
-> XAUTOCLAIM after ~30 s
```

`pending_dispatches()` 现在只选择 `r.next_attempt_at <= now()` 的 Outbox。延迟窗口内消息留在
PostgreSQL，尚未交给 Redis；Publisher 轮询到期后再发布。该修复保留事务 Outbox 和 generation
fencing，不增加第二套延迟队列，也不把 sleep 放进 HTTP 或 Agent Runtime。

### 27.3 Measured scaling curve

工作负载均为 200 个 Semantic Mock Run，两个用户各 100 个，Provider 每用户 cap=1、全局 cap=4：

| Pool | State | Throughput | Deferrals | Attempts | E2E p50/p95/p99/max |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | baseline | 17.84 runs/s | 0 | 200 | 8.69 / 12.98 / 13.36 / 13.45 s |
| 2 | before fix | 4.79 runs/s | 80 | 280 | 6.81 / 9.96 / 44.13 / 44.88 s |
| 4 | before fix | 4.70 runs/s | 536 | 736 | 9.12 / 45.19 / 45.78 / 45.81 s |
| 2 | after fix | 38.99 runs/s | 98 | 298 | 4.58 / 6.55 / 7.30 / 7.75 s |
| 4 | after fix | 30.01 runs/s | 688 | 888 | 5.51 / 7.70 / 8.33 / 9.30 s |

修复消除了约 30 秒 PEL 尾巴。2 Worker 达到当前两用户 permit 的有效并发；4 Worker 产生 688 次
deferral，即 4.44 attempts/Run，并因 PostgreSQL/Event/Outbox 写放大反而比 2 Worker 慢。

额外的 Mock-only 对照把 per-user cap 临时调到 2：2 Worker 达到 34.33 runs/s 且 0 deferral，证明
Worker 横向执行本身有效。但放宽 cap 不是生产修复；真实 Provider 的用户配额与 noisy-neighbor
隔离优先级高于 Mock 吞吐。

### 27.4 Fairness and backpressure interpretation

审计更正：`per_user cap=1` 是本轮实验配置，不是同用户串行或 Worker affinity 的业务合同。

- 每组最终均为 200/200 `SUCCEEDED`、0 retry；两个用户各完成 100。
- 修复后 queue-wait p95：2 Worker 两用户为 4.971/4.978 秒；4 Worker 为 3.205/3.205 秒。
- 完成序列最大用户前缀偏斜为 4（2 Worker）和 5（4 Worker），没有观察到最终饥饿。
- 公平完成不等于调度高效。4 Worker 的 deferral write amplification 是明确容量信号。
- 自动扩容不能只看 queue depth；至少同时看实际 Provider call permit、admission deferral rate、
  DB 写入与 E2E tail。2 Worker 只是在本轮两用户、cap=1 Mock 批次中优于 4 Worker。

### 27.5 Observability additions

- PostgreSQL 累计 `provider_admission_deferred` Attempt，`/metrics` 暴露
  `job_agent_provider_admission_deferrals_total{reason}`。
- Worker 在 admission reject 路径记录 `job_agent_worker_tasks_total{outcome="backpressured"}`；
  该路径不再从 Worker 指标中消失。
- reason 经过固定 allowlist，未知错误合并为 `other`，避免标签基数失控。

### 27.6 Remaining boundary

共享 Redis Stream + 多消费者仍是当前正确默认方案，同一用户的不同 Run 可以由不同 Worker 并行执行。
当前修复解决“延迟消息过早进入 PEL”；剩余 deferral 表示 Worker 并发与当时配置 permit 不匹配，
不能直接推出 tenant/user shard 需求。只有在合理 Provider call 粒度与配额下仍持续出现饥饿，才重新
评估公平调度。

Agent 应用后续优化不继续扩展此调度实验，统一进入
`docs/tech-specs/agent_application_optimization_plan_zh.md`。

完整实验记录见 `docs/performance/backend_service_phase7_worker_scaling_zh.md`。

### 27.7 Verification

```text
LLM + Backend focused: 61 passed
Related regression excluding known frozen Evidence modules: 260 passed, 2 warnings
Full regression: 262 passed, 19 known frozen-Evidence CRLF/hash failures, 2 warnings
compileall / pip check / git diff --check: PASS
broad exception and unbounded-loop scan: no matches
Docker Compose API/Worker rebuild: PASS
seven exact 200-Run Mock batches: 1400/1400 SUCCEEDED, 0 retries
```

全量失败仍只来自冻结 Evidence 文件的 CRLF/byte-hash 差异；本阶段未修改 Evidence 数据、manifest、
评测语义或既有回归口径。
