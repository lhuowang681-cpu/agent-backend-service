from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlparse

from pydantic import Field, ValidationError, model_validator

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    EvidencePrediction,
    EvidenceSpan,
    ExperimentVariant,
    PredictionRecord,
    SanitizedTrace,
    ValidatedEvidencePrediction,
    load_jsonl_dataset,
    sha256_file,
    stable_protocol_hash,
)
from job_agent.evaluation.evidence_grounding_direct_jd_dataset import (
    DirectJDCase,
    load_direct_jd_sidecar,
)
from job_agent.evaluation.evidence_grounding_runner import (
    EvidenceGroundingRunError,
    EvidenceGroundingRunner,
)
from job_agent.evaluation.evidence_grounding_validators import (
    validate_prediction_sources,
)
from job_agent.llm.provider import LLMProvider, ProviderResult
from job_agent.schemas import EvidenceLevel, StrictModel


DIRECT_JD_PROMPT_VERSION = "direct-jd-one-shot-v1.0"
DIRECT_JD_VALIDATOR_VERSION = "jd-quote+resume-quote-membership-v1.0"


class DirectJDResumeRef(StrictModel):
    quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)


class DirectJDEvidencePrediction(StrictModel):
    evidence_id: str = Field(min_length=1)
    jd_quote: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    proposed_level: EvidenceLevel
    evidence_refs: list[DirectJDResumeRef] = Field(default_factory=list)
    risk: str = ""
    abstain: bool = False

    @model_validator(mode="after")
    def validate_refs_for_level(self):
        if self.proposed_level != EvidenceLevel.NONE and not self.evidence_refs:
            raise ValueError("non-None direct JD predictions require evidence refs")
        return self


class DirectJDMappingOutput(StrictModel):
    items: list[DirectJDEvidencePrediction] = Field(min_length=1)


class DirectJDRunnerConfig(StrictModel):
    dataset_path: str = "data/eval/evidence_grounding_v3_dev/dev_120.jsonl"
    raw_jd_path: str = (
        "data/eval/evidence_grounding_v4_direct_jd/raw_jd_120.jsonl"
    )
    output_root: str = (
        "output/private/evidence_grounding_v4_direct_jd/direct_jd"
    )
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-4.7"
    api_key_env: str = "JOB_AGENT_LIVE_API_KEY"
    allow_network: bool = False
    stage: Literal["stage0", "pilot"] = "stage0"
    timeout_s: float = Field(default=90.0, gt=0, le=300)
    max_output_tokens: int = Field(default=4096, ge=512, le=8192)
    max_transport_retries: int = Field(default=1, ge=0, le=2)
    max_format_repairs: int = Field(default=1, ge=0, le=1)
    max_provider_calls: int = Field(default=80, ge=1, le=300)
    seed: int = 20260726
    expected_dataset_hash: str | None = Field(default=None, min_length=64, max_length=64)
    expected_raw_jd_hash: str | None = Field(default=None, min_length=64, max_length=64)
    allowed_hosts: tuple[str, ...] = ("open.bigmodel.cn",)

    @model_validator(mode="after")
    def validate_endpoint(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ValueError("direct JD runner endpoint is not allowlisted")
        return self


DirectProviderFactory = Callable[[DirectJDRunnerConfig, str], LLMProvider]


class DirectJDEvidenceGroundingRunner(EvidenceGroundingRunner):
    def __init__(
        self,
        *,
        provider_factory: DirectProviderFactory | None = None,
    ) -> None:
        super().__init__(provider_factory=provider_factory)

    def run(self, config: DirectJDRunnerConfig) -> list[PredictionRecord]:
        if not config.allow_network:
            raise EvidenceGroundingRunError("network_not_explicitly_enabled")
        dataset_path = Path(config.dataset_path)
        raw_jd_path = Path(config.raw_jd_path)
        dataset_hash = sha256_file(dataset_path)
        raw_jd_hash = sha256_file(raw_jd_path)
        if (
            config.expected_dataset_hash
            and dataset_hash != config.expected_dataset_hash
        ):
            raise EvidenceGroundingRunError("dataset_hash_mismatch")
        if (
            config.expected_raw_jd_hash
            and raw_jd_hash != config.expected_raw_jd_hash
        ):
            raise EvidenceGroundingRunError("raw_jd_hash_mismatch")
        composite_hash = stable_protocol_hash(
            {
                "grounding_dataset_sha256": dataset_hash,
                "raw_jd_sidecar_sha256": raw_jd_hash,
            }
        )
        dataset = load_jsonl_dataset(dataset_path)
        sidecar = load_direct_jd_sidecar(
            raw_jd_path,
            grounding_dataset_path=dataset_path,
        )
        raw_jd_by_case = {case.case_id: case for case in sidecar.cases}
        cases = list(
            dataset.cases[:3] if config.stage == "stage0" else dataset.cases
        )
        random.Random(config.seed).shuffle(cases)

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise EvidenceGroundingRunError("credential_not_injected")
        provider = self.provider_factory(config, api_key)
        del api_key

        contract = direct_jd_contract()
        protocol_payload = {
            "prompt_version": DIRECT_JD_PROMPT_VERSION,
            "validator_version": DIRECT_JD_VALIDATOR_VERSION,
            "contract_hash": stable_protocol_hash(contract),
            "model": config.model,
            "temperature": 0.0,
            "max_output_tokens": config.max_output_tokens,
            "max_format_repairs": config.max_format_repairs,
            "stage": config.stage,
            "variant": ExperimentVariant.JD_DIRECT.value,
        }
        protocol_hash = stable_protocol_hash(protocol_payload)
        output_root = Path(config.output_root)
        prediction_root = output_root / "predictions"
        raw_root = output_root / "raw"
        prediction_root.mkdir(parents=True, exist_ok=True)
        raw_root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            output_root / "run_manifest.json",
            {
                **protocol_payload,
                "protocol_hash": protocol_hash,
                "grounding_dataset_hash": dataset_hash,
                "raw_jd_hash": raw_jd_hash,
                "composite_dataset_hash": composite_hash,
                "case_ids": [case.case_id for case in cases],
                "label_provenance": "deterministic_synthetic_template_draft",
                "review_status": "pending_human_review",
                "credential_source": config.api_key_env,
                "raw_artifacts": "private_only",
            },
        )

        records = []
        for case in cases:
            records.append(
                self._load_or_run_direct(
                    case=case,
                    direct_case=raw_jd_by_case[case.case_id],
                    provider=provider,
                    config=config,
                    composite_hash=composite_hash,
                    protocol_hash=protocol_hash,
                    prediction_root=prediction_root,
                    raw_root=raw_root,
                )
            )
        return records

    def _load_or_run_direct(
        self,
        *,
        case: DatasetCase,
        direct_case: DirectJDCase,
        provider: LLMProvider,
        config: DirectJDRunnerConfig,
        composite_hash: str,
        protocol_hash: str,
        prediction_root: Path,
        raw_root: Path,
    ) -> PredictionRecord:
        variant = ExperimentVariant.JD_DIRECT
        target = prediction_root / f"{case.case_id}_{variant.value}.json"
        cached = self._load_cache(target, composite_hash, protocol_hash)
        if cached is not None:
            return cached
        result, traces, error = self._invoke_direct(
            case=case,
            direct_case=direct_case,
            provider=provider,
            config=config,
            repair=False,
        )
        self._write_raw(raw_root, case.case_id, variant.value, result, error)
        if result is None or error is not None:
            record = self._failure_record(
                case,
                variant,
                composite_hash,
                protocol_hash,
                traces,
                error or "provider_error",
            )
            self._write_record(target, record)
            return record

        try:
            items = self._parse_and_validate(case, direct_case, result)
            self._validate_requirement_coverage(case, items)
        except (ValidationError, ValueError) as first_error:
            if not config.max_format_repairs:
                record = self._failure_record(
                    case,
                    variant,
                    composite_hash,
                    protocol_hash,
                    traces,
                    self._safe_direct_error(first_error),
                )
                self._write_record(target, record)
                return record
            repair, repair_traces, repair_error = self._invoke_direct(
                case=case,
                direct_case=direct_case,
                provider=provider,
                config=config,
                repair=True,
            )
            traces.extend(repair_traces)
            self._write_raw(
                raw_root,
                case.case_id,
                f"{variant.value}_repair",
                repair,
                repair_error,
            )
            if repair is None or repair_error is not None:
                record = self._failure_record(
                    case,
                    variant,
                    composite_hash,
                    protocol_hash,
                    traces,
                    repair_error or "direct_jd_repair_failed",
                )
                self._write_record(target, record)
                return record
            try:
                items = self._parse_and_validate(case, direct_case, repair)
                self._validate_requirement_coverage(case, items)
            except (ValidationError, ValueError) as repair_contract_error:
                record = self._failure_record(
                    case,
                    variant,
                    composite_hash,
                    protocol_hash,
                    traces,
                    self._safe_direct_error(repair_contract_error),
                )
                self._write_record(target, record)
                return record

        record = PredictionRecord(
            protocol_hash=protocol_hash,
            dataset_hash=composite_hash,
            case_id=case.case_id,
            variant=variant,
            schema_valid=True,
            items=items,
            traces=traces,
        )
        self._write_record(target, record)
        return record

    def _invoke_direct(
        self,
        *,
        case: DatasetCase,
        direct_case: DirectJDCase,
        provider: LLMProvider,
        config: DirectJDRunnerConfig,
        repair: bool,
    ) -> tuple[ProviderResult | None, list[SanitizedTrace], str | None]:
        user_prompt = direct_jd_user_prompt(case, direct_case)
        if repair:
            user_prompt += (
                "\n\nThe previous output violated exact JD requirement coverage or "
                "verbatim resume quote membership. Regenerate the COMPLETE mapping. "
                "Return each Core Requirements line exactly once, copy jd_quote and "
                "resume evidence verbatim, and do not include Team Background or "
                "Other Information as requirements."
            )
        return self._invoke(
            provider=provider,
            config=config,
            output_schema=DirectJDMappingOutput,
            system_prompt=direct_jd_system_prompt(),
            user_prompt=user_prompt,
            case_id=case.case_id,
            node_id=(
                "grounding:JD_DIRECT:repair"
                if repair
                else "grounding:JD_DIRECT"
            ),
            prompt_version=DIRECT_JD_PROMPT_VERSION,
        )

    @staticmethod
    def _parse_and_validate(
        case: DatasetCase,
        direct_case: DirectJDCase,
        result: ProviderResult,
    ) -> list[ValidatedEvidencePrediction]:
        parsed = DirectJDMappingOutput.model_validate(result.parsed_output)
        return validate_direct_jd_items(case, direct_case, parsed.items)

    @staticmethod
    def _safe_direct_error(exc: Exception) -> str:
        text = str(exc).casefold()
        for known in (
            "direct_jd_requirement_coverage_mismatch",
            "direct_jd_unknown_requirement_quote",
            "duplicate_evidence_id",
            "duplicate_direct_jd_requirement_quote",
        ):
            if known in text:
                return known
        return "direct_jd_schema_or_contract_failure"


def _locate_global_quote(
    text: str,
    quote: str,
    occurrence_index: int | None,
) -> tuple[int | None, int | None, str | None]:
    positions = []
    start = 0
    while quote:
        position = text.find(quote, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    if not positions:
        return None, None, "quote_not_in_resume"
    if occurrence_index is None:
        if len(positions) != 1:
            return None, None, "ambiguous_quote"
        position = positions[0]
    elif occurrence_index >= len(positions):
        return None, None, "occurrence_out_of_range"
    else:
        position = positions[occurrence_index]
    return position, position + len(quote), None


def validate_direct_jd_items(
    case: DatasetCase,
    direct_case: DirectJDCase,
    items: Sequence[DirectJDEvidencePrediction],
) -> list[ValidatedEvidencePrediction]:
    evidence_ids = [item.evidence_id for item in items]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("duplicate_evidence_id")
    link_by_quote = {
        link.quote: link for link in direct_case.requirement_links
    }
    output_quotes = [item.jd_quote for item in items]
    if len(output_quotes) != len(set(output_quotes)):
        raise ValueError("duplicate_direct_jd_requirement_quote")
    if any(quote not in link_by_quote for quote in output_quotes):
        raise ValueError("direct_jd_unknown_requirement_quote")
    if set(output_quotes) != set(link_by_quote):
        raise ValueError("direct_jd_requirement_coverage_mismatch")

    validated_items = []
    for item in items:
        link = link_by_quote[item.jd_quote]
        spans = []
        source_errors = []
        for evidence in item.evidence_refs:
            start, end, error = _locate_global_quote(
                case.resume_text,
                evidence.quote,
                evidence.occurrence_index,
            )
            if error is not None:
                source_errors.append(error)
                continue
            assert start is not None and end is not None
            spans.append(
                EvidenceSpan(
                    start_char=start,
                    end_char=end,
                    quote=evidence.quote,
                )
            )
        prediction = EvidencePrediction(
            evidence_id=item.evidence_id,
            requirement_id=link.requirement_id,
            claim=item.claim,
            proposed_level=item.proposed_level,
            spans=spans,
            risk=item.risk,
            abstain=item.abstain or item.proposed_level == EvidenceLevel.NONE,
        )
        validated = validate_prediction_sources(
            case.resume_text,
            prediction,
            enforce_source_guard=True,
        )
        if source_errors:
            final_level = (
                EvidenceLevel.NONE
                if item.proposed_level == EvidenceLevel.NONE
                else EvidenceLevel.C0
            )
            validated = validated.model_copy(
                update={
                    "source_valid": False,
                    "final_level": final_level,
                    "reason_codes": (
                        validated.reason_codes
                        + [f"direct_jd_{error}" for error in sorted(set(source_errors))]
                        + ["direct_jd_source_guard_downgrade"]
                    ),
                }
            )
        elif item.proposed_level == EvidenceLevel.NONE and item.evidence_refs:
            validated = validated.model_copy(
                update={
                    "reason_codes": (
                        validated.reason_codes
                        + ["direct_jd_none_with_auditable_context"]
                    )
                }
            )
        validated_items.append(validated)
    return validated_items


def direct_jd_user_prompt(case: DatasetCase, direct_case: DirectJDCase) -> str:
    return json.dumps(
        {
            "job_description": direct_case.raw_jd_text,
            "resume": case.resume_text,
        },
        ensure_ascii=False,
    )


def direct_jd_system_prompt() -> str:
    return (
        "You are a one-shot JD-to-resume evidence mapper. Read the complete raw job "
        "description and complete resume without pre-supplied requirement IDs. Extract "
        "only technical lines under Core Requirements; Team Background and "
        "Other Information are context, not requirements. Return each true requirement "
        "exactly once. jd_quote must copy only the requirement text verbatim, excluding "
        "bullet or numbering markers. For each requirement, map resume evidence and choose "
        "a level: C3=action+artifact+direct result, C2=action+artifact, "
        "C1=learning/reproduction/limited participation, C0=keyword/context/team-only "
        "result, None=no positive evidence or explicit negation. evidence_refs contain "
        "verbatim resume quote and occurrence_index only when the quote repeats. Never "
        "output offsets. For None, refs may be empty or retain negative/context evidence. "
        "Do not borrow metrics across requirements, convert plans into completed work, "
        "or attribute team results to the candidate."
    )


def direct_jd_contract() -> dict:
    return {
        "variant": ExperimentVariant.JD_DIRECT.value,
        "prompt_version": DIRECT_JD_PROMPT_VERSION,
        "validator_version": DIRECT_JD_VALIDATOR_VERSION,
        "system_prompt": direct_jd_system_prompt(),
        "schema": DirectJDMappingOutput.model_json_schema(),
        "input_contract": "complete_raw_jd+complete_resume_without_requirement_ids",
        "jd_alignment": "exact_unique_requirement_quote",
        "resume_alignment": "exact_quote+optional_occurrence_index",
        "coverage_policy": "exact_all_true_requirements_no_extras",
        "source_failure_policy": "non_none_cap_c0",
        "offset_owner": "python",
        "upgrade_allowed": False,
    }
