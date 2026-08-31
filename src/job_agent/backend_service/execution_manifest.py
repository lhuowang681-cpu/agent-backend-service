from __future__ import annotations

from collections.abc import Mapping

from job_agent.backend_service.contracts import (
    ApplicationAssistantInput,
    CreateRunRequest,
    ExecutionManifest,
)


def build_execution_manifest(
    request: CreateRunRequest,
    *,
    provider_models: Mapping[str, str],
) -> ExecutionManifest:
    try:
        model = provider_models[request.provider_profile]
    except KeyError as exc:
        raise ValueError("provider_profile_not_manifested") from exc
    if request.task_type == "application_assistant_flow":
        if not isinstance(request.input, ApplicationAssistantInput):
            raise ValueError("task_input_contract_mismatch")
        return ExecutionManifest(
            provider_profile=request.provider_profile,
            model=model,
            skill_version="application-assistant-v1",
            prompt_version="application-assistant-controller-v1",
            tool_registry_version="application-assistant-tools-v1",
            policy_version="agent-policy-v1",
            career_snapshot_revision=request.input.career_snapshot_revision,
            checkpoint_schema_version="domain-agent-checkpoint-v1",
        )
    return ExecutionManifest(
        provider_profile=request.provider_profile,
        model=model,
        skill_version="backend-semantic-flow-v1",
        prompt_version="semantic-graph-v1",
        tool_registry_version="backend-semantic-registry-v1",
        policy_version="node-policy-v1",
        checkpoint_schema_version="semantic-graph-checkpoint-v1",
    )
