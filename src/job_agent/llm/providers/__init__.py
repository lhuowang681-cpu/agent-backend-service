"""Concrete structured-generation providers."""

from job_agent.llm.providers.mock import MockLLMProvider
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.llm.providers.openai_compatible import OpenAICompatibleProvider
from job_agent.llm.providers.deepseek_compatible import DeepSeekCompatibleProvider
from job_agent.llm.providers.transformers import LocalTransformersProvider

__all__ = [
    "AnthropicCompatibleProvider",
    "LocalTransformersProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "DeepSeekCompatibleProvider",
]
