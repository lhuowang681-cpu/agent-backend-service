from __future__ import annotations

from job_agent.agent_runtime.contracts import (
    AgentAction,
    ApprovalDecision,
    RuntimeToolSpec,
    ToolEffect,
)
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine


def _action(tool_name: str = "jobs.search", arguments=None) -> AgentAction:
    return AgentAction(
        action_id="a-1",
        tool_name=tool_name,
        tool_arguments=arguments or {"query": "agent intern"},
        expected_observation="result",
        progress_claim="make progress",
    )


def _spec(
    *,
    name: str = "jobs.search",
    effect: ToolEffect = ToolEffect.READ_ONLY,
    sandbox_only: bool = False,
) -> RuntimeToolSpec:
    approval = effect in {
        ToolEffect.LOCAL_STATE_MUTATION,
        ToolEffect.EXTERNAL_DRAFT,
        ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
        ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
        ToolEffect.SENSITIVE_READ,
    }
    return RuntimeToolSpec(
        name=name,
        description="test tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        effect=effect,
        requires_approval=approval,
        sandbox_only=sandbox_only,
    )


def _context(tool_name: str = "jobs.search", **updates) -> PolicyContext:
    values = dict(
        skill_allowed_tools=(tool_name,),
        agent_allowed_tools=(tool_name,),
        runtime_allowed_tools=(tool_name,),
        arguments_preview_keys=("query",),
    )
    values.update(updates)
    return PolicyContext(**values)


def _approve(result) -> ApprovalDecision:
    request = result.approval_request
    assert request is not None
    return ApprovalDecision(
        request_id=request.request_id,
        action_digest=request.action_digest,
        approved=True,
        session_id=request.session_id,
        run_id=request.run_id,
    )


def test_read_only_tool_requires_all_three_allowlists() -> None:
    engine = PolicyEngine()
    action = _action()
    allowed = engine.authorize(action, _spec(), _context())
    denied = engine.authorize(
        action,
        _spec(),
        _context(runtime_allowed_tools=()),
    )
    assert allowed.allowed is True
    assert denied.reason_code == "runtime_tool_denied"


def test_local_mutation_requires_matching_approval_digest() -> None:
    engine = PolicyEngine()
    action = _action("tracker.update", {"state": "applied"})
    spec = _spec(name="tracker.update", effect=ToolEffect.LOCAL_STATE_MUTATION)
    context = _context("tracker.update")
    pending = engine.authorize(action, spec, context)
    assert pending.requires_approval is True

    approved = engine.authorize(action, spec, context, approval=_approve(pending))
    assert approved.allowed is True

    changed = action.model_copy(update={"tool_arguments": {"state": "offer"}})
    rejected = engine.authorize(changed, spec, context, approval=_approve(pending))
    assert rejected.reason_code == "approval_digest_mismatch"

    request = pending.approval_request
    assert request is not None
    denied = engine.authorize(
        action,
        spec,
        context,
        approval=ApprovalDecision(
            request_id=request.request_id,
            action_digest=request.action_digest,
            approved=False,
            reason="user denied",
        ),
    )
    assert denied.reason_code == "approval_denied"


def test_sandbox_external_tool_can_run_only_after_approval() -> None:
    engine = PolicyEngine()
    action = _action("sandbox.email.send", {"to": "fixture@example.invalid"})
    spec = _spec(
        name="sandbox.email.send",
        effect=ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
        sandbox_only=True,
    )
    context = _context("sandbox.email.send")
    pending = engine.authorize(action, spec, context)
    approved = engine.authorize(action, spec, context, approval=_approve(pending))
    assert pending.reason_code == "approval_required"
    assert approved.allowed is True


def test_real_external_tool_is_denied_until_necessary_allowlist_is_explicit() -> None:
    engine = PolicyEngine()
    action = _action("calendar.create", {"title": "Interview"})
    spec = _spec(
        name="calendar.create",
        effect=ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
        sandbox_only=False,
    )
    denied = engine.authorize(action, spec, _context("calendar.create"))
    assert denied.reason_code == "real_external_tool_denied"

    necessary = _context(
        "calendar.create",
        necessary_real_tool_allowlist=("calendar.create",),
    )
    pending = engine.authorize(action, spec, necessary)
    approved = engine.authorize(action, spec, necessary, approval=_approve(pending))
    assert approved.allowed is True


def test_sensitive_read_is_fail_closed_even_with_allowlists() -> None:
    action = _action("credentials.read")
    result = PolicyEngine().authorize(
        action,
        _spec(name="credentials.read", effect=ToolEffect.SENSITIVE_READ),
        _context("credentials.read"),
    )
    assert result.reason_code == "sensitive_read_denied"
