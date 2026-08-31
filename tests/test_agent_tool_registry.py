from __future__ import annotations

import pytest
from pydantic import Field

from job_agent.agent_runtime.contracts import AgentAction, ToolContext, ToolEffect, ToolObservationStatus
from job_agent.agent_runtime.failures import ToolRegistryError, ToolRetryableError, ToolTimeoutError
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_executor import ToolExecutor
from job_agent.agent_runtime.tool_registry import RegisteredTool, ToolRegistry
from job_agent.schemas import StrictModel


class SearchInput(StrictModel):
    query: str = Field(min_length=1)


class SearchOutput(StrictModel):
    jobs: list[str]


def _action(arguments=None) -> AgentAction:
    return AgentAction(
        action_id="a-1",
        tool_name="jobs.search",
        tool_arguments=arguments or {"query": "agent intern"},
        expected_observation="jobs",
        progress_claim="search jobs",
    )


def _context() -> ToolContext:
    return ToolContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="opportunity-research",
        workspace_root="fixtures/workspace",
        sandbox_root="fixtures/workspace/sandbox",
    )


def _policy_context() -> PolicyContext:
    return PolicyContext(
        skill_allowed_tools=("jobs.search",),
        agent_allowed_tools=("jobs.search",),
        runtime_allowed_tools=("jobs.search",),
    )


def test_registry_projects_one_schema_to_provider_contract() -> None:
    registry = ToolRegistry()
    registered = registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: SearchOutput(jobs=[value.query]),
    )

    provider_spec = registry.provider_specs()[0]
    assert provider_spec.input_schema == SearchInput.model_json_schema()
    assert registered.spec.output_schema == SearchOutput.model_json_schema()


def test_registry_rejects_duplicate_and_schema_drift() -> None:
    registry = ToolRegistry()
    tool = registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: SearchOutput(jobs=[]),
    )
    with pytest.raises(ToolRegistryError, match="duplicate"):
        registry.register_tool(tool)

    drifted = RegisteredTool(
        spec=tool.spec.model_copy(update={"input_schema": {"type": "string"}}),
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=tool.handler,
    )
    with pytest.raises(ToolRegistryError, match="schema drift"):
        ToolRegistry([drifted])


def test_executor_validates_authorization_input_and_output() -> None:
    registry = ToolRegistry()
    registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: SearchOutput(jobs=["job-1"]),
    )
    action = _action()
    auth = PolicyEngine().authorize(action, registry.get("jobs.search").spec, _policy_context())
    observation = ToolExecutor(registry).execute(action, auth, _context())
    assert observation.status == ToolObservationStatus.OK
    assert observation.data == {"jobs": ["job-1"]}

    invalid = _action({"unknown": "value"})
    invalid_auth = PolicyEngine().authorize(
        invalid,
        registry.get("jobs.search").spec,
        _policy_context(),
    )
    invalid_observation = ToolExecutor(registry).execute(invalid, invalid_auth, _context())
    assert invalid_observation.error_code == "tool_input_error"


def test_executor_classifies_retryable_and_output_errors() -> None:
    retry_registry = ToolRegistry()

    def retry_handler(value, context):
        raise ToolRetryableError("source unavailable")

    retry_registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=retry_handler,
    )
    action = _action()
    auth = PolicyEngine().authorize(
        action,
        retry_registry.get("jobs.search").spec,
        _policy_context(),
    )
    retry_observation = ToolExecutor(retry_registry).execute(action, auth, _context())
    assert retry_observation.status == ToolObservationStatus.RETRYABLE_ERROR

    invalid_registry = ToolRegistry()
    invalid_registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: {"wrong": "shape"},
    )
    invalid_auth = PolicyEngine().authorize(
        action,
        invalid_registry.get("jobs.search").spec,
        _policy_context(),
    )
    invalid_observation = ToolExecutor(invalid_registry).execute(action, invalid_auth, _context())
    assert invalid_observation.error_code == "tool_output_error"


def test_executor_does_not_run_handler_after_denial_or_argument_change() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: calls.append(value.query) or SearchOutput(jobs=[]),
    )
    action = _action()
    denied = PolicyEngine().authorize(
        action,
        registry.get("jobs.search").spec,
        _policy_context().model_copy(update={"runtime_allowed_tools": ()}),
    )
    denied_observation = ToolExecutor(registry).execute(action, denied, _context())
    assert denied_observation.error_code == "authorization_required"
    assert calls == []

    allowed = PolicyEngine().authorize(action, registry.get("jobs.search").spec, _policy_context())
    changed = action.model_copy(update={"tool_arguments": {"query": "changed"}})
    changed_observation = ToolExecutor(registry).execute(changed, allowed, _context())
    assert changed_observation.error_code == "authorization_digest_mismatch"
    assert calls == []


def test_executor_reports_handler_overrun_as_timeout() -> None:
    ticks = iter((0.0, 0.02))
    registry = ToolRegistry()
    registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: SearchOutput(jobs=[]),
        timeout_s=0.01,
    )
    action = _action()
    auth = PolicyEngine().authorize(action, registry.get("jobs.search").spec, _policy_context())
    observation = ToolExecutor(registry, clock=lambda: next(ticks)).execute(action, auth, _context())
    assert observation.status == ToolObservationStatus.RETRYABLE_ERROR
    assert observation.error_code == "tool_timeout"


def test_executor_injects_cooperative_deadline_into_handler_context() -> None:
    seen_deadlines = []
    registry = ToolRegistry()

    def cooperative_handler(value, context):
        seen_deadlines.append(context.deadline_monotonic)
        assert context.remaining_seconds() is not None
        raise ToolTimeoutError()

    registry.register(
        name="jobs.search",
        description="Search fixture jobs with cooperative cancellation",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=cooperative_handler,
        timeout_s=1.0,
    )
    action = _action()
    auth = PolicyEngine().authorize(action, registry.get("jobs.search").spec, _policy_context())
    observation = ToolExecutor(registry).execute(action, auth, _context())
    assert seen_deadlines[0] is not None
    assert observation.status == ToolObservationStatus.RETRYABLE_ERROR
    assert observation.error_code == "tool_timeout"


def test_runtime_tool_spec_declares_cooperative_timeout_contract() -> None:
    registry = ToolRegistry()
    tool = registry.register(
        name="jobs.search",
        description="Search fixture jobs",
        input_model=SearchInput,
        output_model=SearchOutput,
        handler=lambda value, context: SearchOutput(jobs=[]),
        timeout_s=1.0,
    )

    assert tool.spec.timeout_mode == "cooperative"
