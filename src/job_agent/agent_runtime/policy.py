from __future__ import annotations

import hashlib
import json
from job_agent.agent_runtime.contracts import (
    AgentAction,
    ApprovalDecision,
    ApprovalRequest,
    AuthorizationResult,
    RuntimeToolSpec,
    ToolEffect,
)
from job_agent.schemas import StrictModel


def compute_action_digest(action: AgentAction) -> str:
    payload = {
        "action_id": action.action_id,
        "tool_name": action.tool_name,
        "tool_arguments": action.tool_arguments,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class PolicyContext(StrictModel):
    skill_allowed_tools: tuple[str, ...]
    agent_allowed_tools: tuple[str, ...]
    runtime_allowed_tools: tuple[str, ...]
    necessary_real_tool_allowlist: tuple[str, ...] = ()
    target_summary: str = "agent tool action"
    arguments_preview_keys: tuple[str, ...] = ()


class PolicyEngine:
    _APPROVAL_EFFECTS = {
        ToolEffect.LOCAL_STATE_MUTATION,
        ToolEffect.EXTERNAL_DRAFT,
        ToolEffect.EXTERNAL_REVERSIBLE_WRITE,
        ToolEffect.EXTERNAL_IRREVERSIBLE_WRITE,
        ToolEffect.SENSITIVE_READ,
    }

    def authorize(
        self,
        action: AgentAction,
        spec: RuntimeToolSpec,
        context: PolicyContext,
        *,
        approval: ApprovalDecision | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AuthorizationResult:
        if (session_id is None) != (run_id is None):
            raise ValueError("approval scope requires both session_id and run_id")
        digest = compute_action_digest(action)
        denied_reason = self._deny_reason(action, spec, context)
        if denied_reason is not None:
            return AuthorizationResult(
                action_id=action.action_id,
                action_digest=digest,
                tool_name=action.tool_name,
                effect=spec.effect,
                allowed=False,
                requires_approval=False,
                reason_code=denied_reason,
            )

        requires_approval = spec.requires_approval or spec.effect in self._APPROVAL_EFFECTS
        request = (
            self._approval_request(
                action,
                spec,
                context,
                digest,
                session_id=session_id,
                run_id=run_id,
            )
            if requires_approval
            else None
        )
        if requires_approval and approval is None:
            return AuthorizationResult(
                action_id=action.action_id,
                action_digest=digest,
                tool_name=action.tool_name,
                effect=spec.effect,
                allowed=False,
                requires_approval=True,
                reason_code="approval_required",
                approval_request=request,
            )
        if approval is not None:
            if request is None:
                return AuthorizationResult(
                    action_id=action.action_id,
                    action_digest=digest,
                    tool_name=action.tool_name,
                    effect=spec.effect,
                    allowed=False,
                    requires_approval=False,
                    reason_code="unexpected_approval",
                )
            if approval.action_digest != digest:
                return AuthorizationResult(
                    action_id=action.action_id,
                    action_digest=digest,
                    tool_name=action.tool_name,
                    effect=spec.effect,
                    allowed=False,
                    requires_approval=False,
                    reason_code="approval_digest_mismatch",
                )
            if (
                approval.session_id != request.session_id
                or approval.run_id != request.run_id
            ):
                return AuthorizationResult(
                    action_id=action.action_id,
                    action_digest=digest,
                    tool_name=action.tool_name,
                    effect=spec.effect,
                    allowed=False,
                    requires_approval=False,
                    reason_code="approval_scope_mismatch",
                )
            if approval.request_id != request.request_id:
                return AuthorizationResult(
                    action_id=action.action_id,
                    action_digest=digest,
                    tool_name=action.tool_name,
                    effect=spec.effect,
                    allowed=False,
                    requires_approval=False,
                    reason_code="approval_request_mismatch",
                )
            if not approval.approved:
                return AuthorizationResult(
                    action_id=action.action_id,
                    action_digest=digest,
                    tool_name=action.tool_name,
                    effect=spec.effect,
                    allowed=False,
                    requires_approval=False,
                    reason_code="approval_denied",
                )

        return AuthorizationResult(
            action_id=action.action_id,
            action_digest=digest,
            tool_name=action.tool_name,
            effect=spec.effect,
            allowed=True,
            requires_approval=False,
            reason_code="allowed",
        )

    @staticmethod
    def _deny_reason(
        action: AgentAction,
        spec: RuntimeToolSpec,
        context: PolicyContext,
    ) -> str | None:
        if action.tool_name != spec.name:
            return "tool_spec_mismatch"
        for allowed, code in (
            (context.skill_allowed_tools, "skill_tool_denied"),
            (context.agent_allowed_tools, "agent_tool_denied"),
            (context.runtime_allowed_tools, "runtime_tool_denied"),
        ):
            if action.tool_name not in allowed:
                return code
        if spec.effect == ToolEffect.SENSITIVE_READ:
            return "sensitive_read_denied"
        if spec.effect.is_external and not spec.sandbox_only:
            if action.tool_name not in context.necessary_real_tool_allowlist:
                return "real_external_tool_denied"
        return None

    @staticmethod
    def _approval_request(
        action: AgentAction,
        spec: RuntimeToolSpec,
        context: PolicyContext,
        digest: str,
        *,
        session_id: str | None,
        run_id: str | None,
    ) -> ApprovalRequest:
        preview = {
            key: action.tool_arguments[key]
            for key in context.arguments_preview_keys
            if key in action.tool_arguments
        }
        request_material = digest
        if session_id is not None and run_id is not None:
            request_material = f"{session_id}\0{run_id}\0{digest}"
        request_digest = hashlib.sha256(request_material.encode("utf-8")).hexdigest()
        return ApprovalRequest(
            request_id=f"approval-{request_digest[:24]}",
            action_digest=digest,
            tool_name=action.tool_name,
            effect=spec.effect,
            target_summary=context.target_summary,
            arguments_preview=preview,
            session_id=session_id,
            run_id=run_id,
        )
