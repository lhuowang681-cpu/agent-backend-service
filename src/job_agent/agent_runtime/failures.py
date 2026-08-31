from __future__ import annotations


class AgentRuntimeError(RuntimeError):
    error_code = "agent_runtime_error"
    retryable = False

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.error_code)


class AgentContractError(AgentRuntimeError):
    error_code = "agent_contract_error"


class ToolRegistryError(AgentRuntimeError):
    error_code = "tool_registry_error"


class ToolExecutionError(AgentRuntimeError):
    error_code = "tool_execution_error"


class ToolInputError(ToolExecutionError):
    error_code = "tool_input_error"


class ToolOutputError(ToolExecutionError):
    error_code = "tool_output_error"


class ToolRetryableError(ToolExecutionError):
    error_code = "tool_retryable_error"
    retryable = True


class ToolTimeoutError(ToolRetryableError):
    error_code = "tool_timeout"


class PolicyDeniedError(AgentRuntimeError):
    error_code = "policy_denied"


class BudgetExceededError(AgentRuntimeError):
    error_code = "budget_exceeded"

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class TrajectoryError(AgentRuntimeError):
    error_code = "trajectory_error"


class CheckpointError(AgentRuntimeError):
    error_code = "checkpoint_error"
