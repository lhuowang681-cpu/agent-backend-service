from __future__ import annotations

import json
from time import sleep as default_sleep

import pytest

from job_agent.agent_runtime.contracts import (
    AgentAction,
    AgentFinish,
    AgentModelContext,
    AgentStepRecord,
    ToolObservation,
    ToolObservationStatus,
)
from job_agent.agent_runtime.tool_use_decision_model import ToolUseDecisionModel
from job_agent.domain_agents.opportunity_research import JobSearchInput, OpportunityResearchResult
from job_agent.llm.harness import NodePolicy
from job_agent.llm.provider import (
    ProviderContractError,
    ProviderTransportError,
    ToolSpec,
    ToolUseResult,
    TraceContext,
)
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.skill_registry import SkillSpec


def _trace() -> TraceContext:
    return TraceContext(
        session_id="session-v2",
        run_id="run-v2",
        node_id="opportunity:tool-use:1",
        skill_id="opportunity-v2",
        skill_version="v2",
        prompt_version="tool-use-v2",
    )


class QueueToolUseProvider:
    provider_name = "tool-use-mock"
    model = "tool-use-model"

    def __init__(self, outputs: list[tuple[str, dict] | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate_tool_use(self, **kwargs) -> ToolUseResult:
        self.calls.append(kwargs)
        next_output = self.outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        name, tool_input = next_output
        trace = kwargs["trace"]
        return ToolUseResult(
            provider=self.provider_name,
            model=self.model,
            skill_id=trace.skill_id,
            skill_version=trace.skill_version,
            prompt_version=trace.prompt_version,
            latency_ms=1,
            input_tokens=10,
            output_tokens=5,
            schema_valid=True,
            tool_use_id=f"tool-{len(self.calls)}",
            tool_name=name,
            tool_input=tool_input,
        )


def _model(
    provider: QueueToolUseProvider,
    *,
    max_retries: int = 0,
    max_runtime_tool_calls: int | None = None,
    sleep=default_sleep,
) -> ToolUseDecisionModel:
    search = ToolSpec(
        name="jobs.search",
        description="Search local jobs.",
        input_schema=JobSearchInput.model_json_schema(),
    )
    return ToolUseDecisionModel(
        provider=provider,
        skill=SkillSpec(
            skill_id="opportunity-v2",
            version="v2",
            instructions="Search before submitting the final opportunity result.",
            reference_paths=(),
            output_schema=OpportunityResearchResult,
            allowed_tools=("jobs.search",),
            guardrails=("Never invent a job id.",),
        ),
        tools=[search],
        result_schema=OpportunityResearchResult,
        submit_tool_name="submit_opportunity_result",
        session_id="session-v2",
        run_id="run-v2",
        agent_id="opportunity-research",
        prompt_version="tool-use-v2",
        max_runtime_tool_calls=max_runtime_tool_calls,
        policy=NodePolicy(max_retries=max_retries, max_output_tokens=1024),
        sleep=sleep,
    )


def test_anthropic_native_tool_use_builds_any_request_and_parses_one_tool() -> None:
    captured = {}

    def transport(http_request, timeout_s):
        captured["payload"] = json.loads(http_request.data.decode("utf-8"))
        return json.dumps(
            {
                "model": "glm-4.7",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-live-1",
                        "name": "jobs.search",
                        "input": {"keywords": ["Agent"], "cities": ["Shanghai"]},
                    }
                ],
                "usage": {"input_tokens": 21, "output_tokens": 8},
            }
        ).encode("utf-8")

    provider = AnthropicCompatibleProvider(
        base_url="https://example.invalid/api/anthropic",
        model="glm-4.7",
        api_key="fixture-key",
        transport=transport,
    )
    result = provider.generate_tool_use(
        system_prompt="trusted",
        messages=[{"role": "user", "content": "untrusted"}],
        tools=[
            ToolSpec(
                name="jobs.search",
                description="Search local jobs.",
                input_schema=JobSearchInput.model_json_schema(),
            )
        ],
        temperature=0.0,
        max_output_tokens=512,
        trace=_trace(),
    )

    assert captured["payload"]["tool_choice"] == {"type": "any"}
    assert captured["payload"]["tools"][0]["name"] == "jobs.search"
    assert result.tool_name == "jobs.search"
    assert result.tool_input["keywords"] == ["Agent"]
    assert result.input_tokens == 21


def test_anthropic_native_tool_use_rejects_text_only_response() -> None:
    provider = AnthropicCompatibleProvider(
        base_url="https://example.invalid/api/anthropic",
        model="glm-4.7",
        api_key="fixture-key",
        transport=lambda req, timeout: json.dumps(
            {"content": [{"type": "text", "text": "not a tool"}]}
        ).encode("utf-8"),
    )
    with pytest.raises(ProviderContractError, match="exactly one allowed tool_use"):
        provider.generate_tool_use(
            system_prompt="trusted",
            messages=[{"role": "user", "content": "untrusted"}],
            tools=[ToolSpec(name="jobs.search", description="Search", input_schema={})],
            temperature=0.0,
            max_output_tokens=512,
            trace=_trace(),
        )


def test_tool_use_decision_model_maps_runtime_tool_and_native_history() -> None:
    provider = QueueToolUseProvider(
        [("jobs.search", {"keywords": ["Agent"], "cities": ["Shanghai"]})]
    )
    response = _model(provider).decide(
        AgentModelContext(
            goal={"keywords": ["Agent"], "cities": ["Shanghai"], "min_candidates": 1},
        )
    )

    assert isinstance(response.decision, AgentAction)
    assert response.decision.tool_name == "jobs.search"
    assert response.decision.action_id == "tool-1"
    assert provider.calls[0]["tools"][-1].name == "submit_opportunity_result"


def test_tool_use_decision_model_maps_submit_result_after_tool_observation() -> None:
    final = {
        "search_queries": ["Agent"],
        "candidate_job_ids": ["job-1"],
        "evidence_refs": ["step:1:jobs.search"],
        "unresolved_gaps": [],
    }
    provider = QueueToolUseProvider([("submit_opportunity_result", final)])
    action = AgentAction(
        action_id="tool-search",
        tool_name="jobs.search",
        tool_arguments={"keywords": ["Agent"], "cities": ["Shanghai"]},
        expected_observation="jobs",
        progress_claim="search",
    )
    response = _model(provider).decide(
        AgentModelContext(
            goal={"keywords": ["Agent"], "cities": ["Shanghai"], "min_candidates": 1},
            steps=[
                AgentStepRecord(
                    step_index=1,
                    action=action,
                    observation=ToolObservation(
                        action_id="tool-search",
                        tool_name="jobs.search",
                        status=ToolObservationStatus.OK,
                        data={"jobs": [{"job_id": "job-1"}]},
                        duration_ms=1,
                    ),
                )
            ],
        )
    )

    assert isinstance(response.decision, AgentFinish)
    assert response.decision.result == final
    messages = provider.calls[0]["messages"]
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert "job-1" in messages[2]["content"][0]["content"]


def test_tool_use_decision_model_hides_runtime_tools_after_limit() -> None:
    final = {
        "search_queries": ["Agent"],
        "candidate_job_ids": ["job-1"],
        "evidence_refs": ["step:2:jobs.search"],
        "unresolved_gaps": [],
    }
    provider = QueueToolUseProvider([("submit_opportunity_result", final)])
    steps = []
    for index in (1, 2):
        action = AgentAction(
            action_id=f"tool-search-{index}",
            tool_name="jobs.search",
            tool_arguments={"keywords": ["Agent"], "cities": ["Shanghai"]},
            expected_observation="jobs",
            progress_claim="search",
        )
        steps.append(
            AgentStepRecord(
                step_index=index,
                action=action,
                observation=ToolObservation(
                    action_id=action.action_id,
                    tool_name=action.tool_name,
                    status=ToolObservationStatus.OK,
                    data={"jobs": [{"job_id": "job-1"}]},
                    duration_ms=1,
                ),
            )
        )
    response = _model(provider, max_runtime_tool_calls=2).decide(
        AgentModelContext(goal={"min_candidates": 1}, steps=steps)
    )
    assert isinstance(response.decision, AgentFinish)
    assert [tool.name for tool in provider.calls[0]["tools"]] == ["submit_opportunity_result"]


def test_tool_use_decision_model_retries_retryable_provider_error() -> None:
    provider = QueueToolUseProvider(
        [
            ProviderTransportError("transient outage"),
            ("jobs.search", {"keywords": ["Agent"], "cities": ["Shanghai"]}),
        ]
    )
    model = _model(provider, max_retries=2, sleep=lambda _seconds: None)
    response = model.decide(
        AgentModelContext(
            goal={"keywords": ["Agent"], "cities": ["Shanghai"], "min_candidates": 1},
        )
    )

    assert isinstance(response.decision, AgentAction)
    assert response.decision.tool_name == "jobs.search"
    assert len(provider.calls) == 2


def test_tool_use_decision_model_does_not_retry_non_retryable_error() -> None:
    provider = QueueToolUseProvider([ProviderContractError("deterministic failure")])
    model = _model(provider, max_retries=2, sleep=lambda _seconds: None)
    with pytest.raises(ProviderContractError):
        model.decide(
            AgentModelContext(
                goal={"keywords": ["Agent"], "cities": ["Shanghai"], "min_candidates": 1},
            )
        )
    assert len(provider.calls) == 1
