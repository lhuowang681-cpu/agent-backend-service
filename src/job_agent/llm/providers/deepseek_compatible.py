from __future__ import annotations

from job_agent.llm.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekCompatibleProvider(OpenAICompatibleProvider):
    """DeepSeek JSON-mode transport; schema validation remains in LLMHarness."""

    provider_name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_s: float = 30.0,
    ) -> None:
        if base_url.rstrip("/") != "https://api.deepseek.com/v1":
            raise ValueError("deepseek_endpoint_not_allowlisted")
        if model not in {"deepseek-chat", "deepseek-reasoner"}:
            raise ValueError("deepseek_model_not_allowlisted")
        if not api_key:
            raise ValueError("deepseek_credential_missing")
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            response_format_mode="json_object",
        )
