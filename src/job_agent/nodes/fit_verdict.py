from __future__ import annotations

import json
import os
from typing import Protocol
from urllib import request

from fit_verdict_engine.prompt_contract import FIT_VERDICT_SYSTEM_PROMPT, build_fit_verdict_user_prompt
from job_agent.schemas import (
    EvidenceLevel,
    FitBackendKind,
    FitInput,
    FitVerdictAudit,
    FitVerdictResult,
    RiskLevel,
    RuntimeConfig,
    RuntimeProfileKind,
    Verdict,
)


SUPPORTED_LEVELS = {EvidenceLevel.C1, EvidenceLevel.C2, EvidenceLevel.C3}


class FitVerdictBackend(Protocol):
    def evaluate(self, fit_input: FitInput) -> FitVerdictResult:
        """Return a validated fit verdict for a structured requirement-evidence map."""


class LocalVerdictRunner(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a raw model response for a fit verdict prompt."""


def _coverage(fit_input: FitInput) -> float:
    required_ids = {requirement.id for requirement in fit_input.requirements if requirement.required}
    if not required_ids:
        return 0.0
    covered_ids = {
        item.requirement_id
        for item in fit_input.evidence
        if item.requirement_id in required_ids and item.level in SUPPORTED_LEVELS
    }
    return round(len(covered_ids) / len(required_ids), 4)


def evaluate_fit(fit_input: FitInput) -> FitVerdictResult:
    coverage = _coverage(fit_input)
    has_toy_signal = bool(fit_input.toy_signals)
    reason_codes: list[str] = []

    if has_toy_signal:
        verdict = Verdict.RISKY
        risk_level = RiskLevel.HIGH
        reason_codes.append("toy_signal_hit")
    elif coverage >= 0.7:
        verdict = Verdict.STRONG
        risk_level = RiskLevel.LOW
        reason_codes.extend(["coverage_ge_70", "has_c1_plus_evidence", "no_toy_signal"])
    elif coverage >= 0.4:
        verdict = Verdict.WEAK
        risk_level = RiskLevel.MEDIUM
        reason_codes.append("coverage_between_40_70")
    elif coverage > 0:
        verdict = Verdict.RISKY
        risk_level = RiskLevel.HIGH
        reason_codes.append("coverage_lt_40")
    else:
        verdict = Verdict.NOT_RECOMMENDED
        risk_level = RiskLevel.HIGH
        reason_codes.append("no_core_evidence")

    return FitVerdictResult(
        verdict=verdict,
        score=coverage,
        coverage=coverage,
        risk_level=risk_level,
        need_human_review=risk_level != RiskLevel.LOW,
        reason_codes=reason_codes,
        explanation=f"must-have 覆盖率为 {coverage:.0%}，verdict={verdict.value}。",
    )


class RuleFitVerdictBackend:
    def evaluate(self, fit_input: FitInput) -> FitVerdictResult:
        return evaluate_fit(fit_input)


class StaticApiFitVerdictBackend:
    """API-compatible backend for tests and future OpenAI-compatible adapter wiring."""

    def __init__(self, payload: dict):
        self._payload = payload

    def evaluate(self, fit_input: FitInput) -> FitVerdictResult:
        return FitVerdictResult.model_validate(self._payload)


def _default_http_post(url: str, payload: dict, api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=60) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


class OpenAICompatibleFitVerdictBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        http_post=_default_http_post,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._http_post = http_post

    @classmethod
    def from_config(cls, config: RuntimeConfig, http_post=_default_http_post) -> "OpenAICompatibleFitVerdictBackend":
        if not config.server_base_url:
            raise ValueError("server_base_url is required for OpenAI-compatible fit verdict backend")
        if not config.server_model:
            raise ValueError("server_model is required for OpenAI-compatible fit verdict backend")
        api_key = os.environ.get(config.server_api_key_env) if config.server_api_key_env else None
        return cls(
            base_url=config.server_base_url,
            model=config.server_model,
            api_key=api_key,
            http_post=http_post,
        )

    def evaluate(self, fit_input: FitInput) -> FitVerdictResult:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a fit verdict classifier. Return only JSON matching "
                        "verdict, score, coverage, risk_level, need_human_review, reason_codes, explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(fit_input.model_dump(), ensure_ascii=False),
                },
            ],
        }
        response = self._http_post(f"{self.base_url}/chat/completions", payload, self.api_key)
        content = response["choices"][0]["message"]["content"]
        return FitVerdictResult.model_validate_json(content)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("local_lora output did not contain a JSON object")
    return stripped[start : end + 1]


def _build_local_lora_prompt(fit_input: FitInput) -> str:
    return build_fit_verdict_user_prompt(fit_input)


def _append_reason(reason_codes: list[str], reason_code: str) -> list[str]:
    if reason_code in reason_codes:
        return reason_codes
    return [*reason_codes, reason_code]


def guard_local_fit_verdict(result: FitVerdictResult, fit_input: FitInput) -> FitVerdictResult:
    expected_coverage = _coverage(fit_input)
    updates = {}
    reason_codes = list(result.reason_codes)
    guard_notes = []
    need_human_review = result.need_human_review

    if abs(result.coverage - expected_coverage) > 0.05:
        updates["coverage"] = expected_coverage
        reason_codes = _append_reason(reason_codes, "guard_corrected_coverage")
        guard_notes.append(f"coverage corrected from {result.coverage:.0%} to {expected_coverage:.0%}")
        need_human_review = True

    effective_coverage = updates.get("coverage", result.coverage)
    if effective_coverage >= 0.7 and result.score < 0.5:
        reason_codes = _append_reason(reason_codes, "guard_score_coverage_mismatch")
        guard_notes.append(
            f"score {result.score:.0%} is low while evidence coverage is {effective_coverage:.0%}"
        )
        need_human_review = True

    if result.risk_level != RiskLevel.LOW and need_human_review is False:
        reason_codes = _append_reason(reason_codes, "guard_risk_requires_review")
        guard_notes.append(f"risk level {result.risk_level.value} requires human review")
        need_human_review = True

    if not guard_notes:
        return result

    explanation = f"{result.explanation} Guard: {'; '.join(guard_notes)}."
    return result.model_copy(
        update={
            **updates,
            "need_human_review": need_human_review,
            "reason_codes": reason_codes,
            "explanation": explanation,
        }
    )


class LocalCausalLMVerdictRunner:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("local_lora backend requires torch and transformers") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("local_lora adapter loading requires peft") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def generate(self, prompt: str) -> str:
        tokenizer, model = self._load()
        messages = [
            {
                "role": "system",
                "content": FIT_VERDICT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tokenizer(text, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)


class LocalLoraFitVerdictBackend:
    def __init__(
        self,
        runner: LocalVerdictRunner,
        fallback_backend: FitVerdictBackend | None = None,
        runtime_profile: RuntimeProfileKind = RuntimeProfileKind.SERVER_AGENT,
        model_path: str | None = None,
        adapter_path: str | None = None,
    ):
        self._runner = runner
        self._fallback_backend = fallback_backend or RuleFitVerdictBackend()
        self._runtime_profile = runtime_profile
        self._model_path = model_path or getattr(runner, "model_path", None)
        self._adapter_path = adapter_path or getattr(runner, "adapter_path", None)
        self.last_audit: FitVerdictAudit | None = None

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "LocalLoraFitVerdictBackend":
        if not config.local_model_path:
            raise ValueError("local_model_path is required for local_lora backend")
        return cls(
            runner=LocalCausalLMVerdictRunner(
                model_path=config.local_model_path,
                adapter_path=config.local_adapter_path,
            ),
            runtime_profile=config.runtime_profile,
            model_path=config.local_model_path,
            adapter_path=config.local_adapter_path,
        )

    def evaluate(self, fit_input: FitInput) -> FitVerdictResult:
        prompt = _build_local_lora_prompt(fit_input)
        raw = ""
        extracted_json: dict[str, object] = {}
        try:
            raw = self._runner.generate(prompt)
            payload = _extract_json_object(raw)
            parsed_payload = json.loads(payload)
            if not isinstance(parsed_payload, dict):
                raise ValueError("local_lora JSON payload must be an object")
            extracted_json = parsed_payload
            result = FitVerdictResult.model_validate(parsed_payload)
            guarded = guard_local_fit_verdict(result, fit_input)
            guard_reason_codes = [code for code in guarded.reason_codes if code.startswith("guard_")]
            self.last_audit = FitVerdictAudit(
                backend=FitBackendKind.LOCAL_LORA,
                runtime_profile=self._runtime_profile,
                model_path=self._model_path,
                adapter_path=self._adapter_path,
                raw_model_output=raw,
                extracted_json=extracted_json,
                schema_valid=True,
                fallback_used=False,
                guard_applied=True,
                guard_reason_codes=guard_reason_codes,
            )
            return guarded
        except Exception as exc:
            fallback = self._fallback_backend.evaluate(fit_input)
            reason_codes = list(fallback.reason_codes)
            if "local_lora_fallback" not in reason_codes:
                reason_codes.append("local_lora_fallback")
            self.last_audit = FitVerdictAudit(
                backend=FitBackendKind.LOCAL_LORA,
                runtime_profile=self._runtime_profile,
                model_path=self._model_path,
                adapter_path=self._adapter_path,
                raw_model_output=raw,
                extracted_json=extracted_json,
                schema_valid=False,
                fallback_used=True,
                guard_applied=False,
                guard_reason_codes=[],
                error_message=str(exc),
            )
            return fallback.model_copy(
                update={
                    "need_human_review": True,
                    "reason_codes": reason_codes,
                    "explanation": f"{fallback.explanation} local_lora fallback: {exc}",
                }
            )
