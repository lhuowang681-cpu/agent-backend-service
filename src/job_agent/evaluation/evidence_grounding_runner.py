from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlparse

from pydantic import Field, ValidationError, model_validator

from job_agent.atomic_io import atomic_write_json
from job_agent.evaluation.evidence_grounding_atomic import (
    AtomicEvidenceRef,
    SourceLocationError,
    build_source_blocks,
    locate_atomic_evidence_refs,
    sha256_text,
)
from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    EntailmentRelation,
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
from job_agent.evaluation.evidence_grounding_validators import (
    apply_entailment_cap,
    canonicalize_unique_quote_offsets,
    locate_unique_quote,
    validate_prediction_sources,
)
from job_agent.llm.provider import (
    LLMProvider,
    ProviderError,
    ProviderResult,
    TraceContext,
)
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.schemas import EvidenceLevel, StrictModel


PROMPT_VERSION = "evidence-grounding-pilot-v1"
VALIDATOR_VERSION = "source-membership-v1.1+entailment-cap-v1"
C_BLOCK_ATTR_PROMPT_VERSION = "c-block-attr-v1.0"
C_BLOCK_ATTR_GUARD_VERSION = "attr-cap-v1.0"
_LEVEL_ORDER = {
    EvidenceLevel.NONE: -1,
    EvidenceLevel.C0: 0,
    EvidenceLevel.C1: 1,
    EvidenceLevel.C2: 2,
    EvidenceLevel.C3: 3,
}
_OWNERSHIP_MARKERS = (
    "团队",
    "参与",
    "协作",
    "共同",
    "team",
    "participated",
    "contributed",
)
_NEGATION_MARKERS = (
    "未实现",
    "未完成",
    "未记录",
    "没有",
    "尚未",
    "未声明",
    "not implemented",
    "did not",
    "without",
)
_PLAN_MARKERS = (
    "计划",
    "预计",
    "拟",
    "下季度",
    "将会",
    "plan to",
    "planned",
    "will ",
)
_NUMERIC_METRIC_RE = re.compile(
    r"\d|%|auc|准确率|成功率|完成率|命中率|召回率|延迟|吞吐|提升|降低",
    re.IGNORECASE,
)
_DEFAULT_VARIANTS = (
    ExperimentVariant.A_CURRENT_V1,
    ExperimentVariant.B_EXTRACTIVE_PROMPT,
    ExperimentVariant.C_MEMBERSHIP_GUARD,
    ExperimentVariant.D_ENTAILMENT_GATE,
    ExperimentVariant.E_LEXICAL_BASELINE,
)


class EvidenceGroundingRunError(RuntimeError):
    pass


class LegacyEvidenceOutput(StrictModel):
    evidence_id: str
    requirement_id: str
    claim: str
    level: EvidenceLevel
    proof: str
    risk: str = ""


class LegacyMappingOutput(StrictModel):
    items: list[LegacyEvidenceOutput] = Field(min_length=1)


class ExtractiveMappingOutput(StrictModel):
    items: list[EvidencePrediction] = Field(min_length=1)


class BlockEvidenceRef(StrictModel):
    source_block_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)


class BlockEvidencePrediction(StrictModel):
    evidence_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    proposed_level: EvidenceLevel
    evidence_refs: list[BlockEvidenceRef] = Field(default_factory=list)
    risk: str = ""
    abstain: bool = False

    @model_validator(mode="after")
    def validate_refs_for_level(self):
        if self.proposed_level != EvidenceLevel.NONE and not self.evidence_refs:
            raise ValueError("non-None block predictions require evidence refs")
        return self


class BlockMappingOutput(StrictModel):
    items: list[BlockEvidencePrediction] = Field(min_length=1)


class BlockAttrEvidencePrediction(BlockEvidencePrediction):
    actor: Literal["candidate", "team", "unknown"]
    polarity: Literal["positive", "negative", "uncertain"]
    completion_status: Literal["completed", "planned", "negated"]


class BlockAttrMappingOutput(StrictModel):
    items: list[BlockAttrEvidencePrediction] = Field(min_length=1)


class BlockAttrDecision(StrictModel):
    requirement_id: str = Field(min_length=1)
    actor: Literal["candidate", "team", "unknown"]
    polarity: Literal["positive", "negative", "uncertain"]
    completion_status: Literal["completed", "planned", "negated"]
    reason_code: str = Field(min_length=1)


class BlockAttrDecisionOutput(StrictModel):
    items: list[BlockAttrDecision] = Field(min_length=1)


class EntailmentDecision(StrictModel):
    requirement_id: str
    relation: EntailmentRelation
    reason_code: str = Field(min_length=1)


class EntailmentBatchOutput(StrictModel):
    items: list[EntailmentDecision] = Field(min_length=1)


class GroundingRunnerConfig(StrictModel):
    dataset_path: str = "data/eval/evidence_grounding_v2/pilot.jsonl"
    output_root: str = "output/private/evidence_grounding_v2/pilot"
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-4.7"
    api_key_env: str = "JOB_AGENT_LIVE_API_KEY"
    allow_network: bool = False
    stage: Literal["stage0", "pilot"] = "stage0"
    timeout_s: float = Field(default=90.0, gt=0, le=300)
    max_output_tokens: int = Field(default=4096, ge=512, le=8192)
    max_transport_retries: int = Field(default=1, ge=0, le=2)
    max_membership_repairs: int = Field(default=1, ge=0, le=1)
    max_provider_calls: int = Field(default=60, ge=1, le=300)
    variants: tuple[ExperimentVariant, ...] = _DEFAULT_VARIANTS
    seed: int = 20260726
    expected_dataset_hash: str | None = Field(default=None, min_length=64, max_length=64)
    allowed_hosts: tuple[str, ...] = ("open.bigmodel.cn",)

    @model_validator(mode="after")
    def validate_endpoint(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ValueError("grounding runner endpoint is not allowlisted")
        if not self.variants or len(self.variants) != len(set(self.variants)):
            raise ValueError("grounding runner variants must be unique and non-empty")
        if (
            ExperimentVariant.D_ENTAILMENT_GATE in self.variants
            and ExperimentVariant.C_MEMBERSHIP_GUARD not in self.variants
        ):
            raise ValueError("variant D requires variant C")
        return self


ProviderFactory = Callable[[GroundingRunnerConfig, str], LLMProvider]


class EvidenceGroundingRunner:
    def __init__(self, *, provider_factory: ProviderFactory | None = None) -> None:
        self.provider_factory = provider_factory or self._provider
        self._provider_calls = 0

    @property
    def provider_calls(self) -> int:
        return self._provider_calls

    def run(self, config: GroundingRunnerConfig) -> list[PredictionRecord]:
        if not config.allow_network:
            raise EvidenceGroundingRunError("network_not_explicitly_enabled")
        dataset_path = Path(config.dataset_path)
        dataset_hash = sha256_file(dataset_path)
        if config.expected_dataset_hash and dataset_hash != config.expected_dataset_hash:
            raise EvidenceGroundingRunError("dataset_hash_mismatch")
        dataset = load_jsonl_dataset(dataset_path)
        cases = list(dataset.cases[:3] if config.stage == "stage0" else dataset.cases)
        random.Random(config.seed).shuffle(cases)

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise EvidenceGroundingRunError("credential_not_injected")
        provider = self.provider_factory(config, api_key)
        del api_key

        protocol_payload = {
            "prompt_version": PROMPT_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "model": config.model,
            "temperature": 0.0,
            "max_output_tokens": config.max_output_tokens,
            "max_membership_repairs": config.max_membership_repairs,
            "stage": config.stage,
        }
        if config.variants != _DEFAULT_VARIANTS:
            protocol_payload["variants"] = [
                variant.value for variant in config.variants
            ]
        if ExperimentVariant.C_BLOCK_ATTR in config.variants:
            protocol_payload["c_block_attr_contract_hash"] = stable_protocol_hash(
                self.c_block_attr_contract()
            )
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
                "dataset_hash": dataset_hash,
                "case_ids": [case.case_id for case in cases],
                "label_provenance": "ai_draft",
                "review_status": "pending_human_review",
                "credential_source": config.api_key_env,
                "raw_artifacts": "private_only",
            },
        )

        records: list[PredictionRecord] = []
        for case in cases:
            case_records: dict[ExperimentVariant, PredictionRecord] = {}
            for variant in (
                ExperimentVariant.A_CURRENT_V1,
                ExperimentVariant.B_EXTRACTIVE_PROMPT,
                ExperimentVariant.C_MEMBERSHIP_GUARD,
                ExperimentVariant.C_BLOCK,
                ExperimentVariant.C_BLOCK_ATTR,
            ):
                if variant not in config.variants:
                    continue
                record = self._load_or_run_mapping(
                    case=case,
                    variant=variant,
                    provider=provider,
                    config=config,
                    dataset_hash=dataset_hash,
                    protocol_hash=protocol_hash,
                    prediction_root=prediction_root,
                    raw_root=raw_root,
                )
                records.append(record)
                case_records[variant] = record
            if ExperimentVariant.D_ENTAILMENT_GATE in config.variants:
                d_record = self._load_or_run_entailment(
                    case=case,
                    c_record=case_records[
                        ExperimentVariant.C_MEMBERSHIP_GUARD
                    ],
                    provider=provider,
                    config=config,
                    dataset_hash=dataset_hash,
                    protocol_hash=protocol_hash,
                    prediction_root=prediction_root,
                    raw_root=raw_root,
                )
                records.append(d_record)
            if ExperimentVariant.E_LEXICAL_BASELINE in config.variants:
                e_record = self._load_or_run_lexical(
                    case=case,
                    dataset_hash=dataset_hash,
                    protocol_hash=protocol_hash,
                    prediction_root=prediction_root,
                )
                records.append(e_record)
            if self._provider_calls > config.max_provider_calls:
                raise EvidenceGroundingRunError("provider_call_budget_exceeded")
        return records

    def _load_or_run_mapping(
        self,
        *,
        case: DatasetCase,
        variant: ExperimentVariant,
        provider: LLMProvider,
        config: GroundingRunnerConfig,
        dataset_hash: str,
        protocol_hash: str,
        prediction_root: Path,
        raw_root: Path,
    ) -> PredictionRecord:
        target = prediction_root / f"{case.case_id}_{variant.value}.json"
        cached = self._load_cache(target, dataset_hash, protocol_hash)
        if cached is not None:
            return cached
        if variant == ExperimentVariant.A_CURRENT_V1:
            output_schema = LegacyMappingOutput
            system_prompt = self._baseline_system_prompt()
            user_prompt = self._mapping_user_prompt(case)
        elif variant == ExperimentVariant.C_BLOCK:
            output_schema = BlockMappingOutput
            system_prompt = self._block_system_prompt()
            user_prompt = self._block_mapping_user_prompt(case)
        elif variant == ExperimentVariant.C_BLOCK_ATTR:
            output_schema = BlockAttrMappingOutput
            system_prompt = self._block_attr_system_prompt()
            user_prompt = self._block_mapping_user_prompt(case)
        else:
            output_schema = ExtractiveMappingOutput
            system_prompt = self._extractive_system_prompt(
                enforce_offsets=variant == ExperimentVariant.C_MEMBERSHIP_GUARD
            )
            user_prompt = self._mapping_user_prompt(case)
        result, traces, error = self._invoke(
            provider=provider,
            config=config,
            output_schema=output_schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            case_id=case.case_id,
            node_id=f"grounding:{variant.value}",
        )
        self._write_raw(raw_root, case.case_id, variant.value, result, error)
        if result is None or error is not None:
            record = self._failure_record(
                case,
                variant,
                dataset_hash,
                protocol_hash,
                traces,
                error or "provider_error",
            )
            self._write_record(target, record)
            return record
        try:
            if variant == ExperimentVariant.A_CURRENT_V1:
                parsed = LegacyMappingOutput.model_validate(result.parsed_output)
                predictions = [self._adapt_legacy(case, item) for item in parsed.items]
                items = [
                    validate_prediction_sources(
                        case.resume_text,
                        item,
                        enforce_source_guard=False,
                    )
                    for item in predictions
                ]
            elif variant in {
                ExperimentVariant.B_EXTRACTIVE_PROMPT,
                ExperimentVariant.C_MEMBERSHIP_GUARD,
            }:
                parsed = ExtractiveMappingOutput.model_validate(result.parsed_output)
                enforce = variant == ExperimentVariant.C_MEMBERSHIP_GUARD
                items = []
                for item in parsed.items:
                    canonicalized = 0
                    if enforce:
                        item, canonicalized = canonicalize_unique_quote_offsets(
                            case.resume_text,
                            item,
                        )
                    validated = validate_prediction_sources(
                        case.resume_text,
                        item,
                        enforce_source_guard=enforce,
                    )
                    if canonicalized:
                        validated = validated.model_copy(
                            update={
                                "reason_codes": validated.reason_codes
                                + ["span_offset_canonicalized"]
                            }
                        )
                    items.append(validated)
                if (
                    enforce
                    and any(not item.source_valid for item in items)
                    and config.max_membership_repairs
                ):
                    result, repair_trace, repair_error = self._repair_membership(
                        case=case,
                        invalid_items=items,
                        provider=provider,
                        config=config,
                    )
                    traces.extend(repair_trace)
                    self._write_raw(
                        raw_root,
                        case.case_id,
                        f"{variant.value}_repair",
                        result,
                        repair_error,
                    )
                    if result is None or repair_error is not None:
                        raise EvidenceGroundingRunError(
                            repair_error or "membership_repair_failed"
                        )
                    repaired = ExtractiveMappingOutput.model_validate(result.parsed_output)
                    items = []
                    for item in repaired.items:
                        item, canonicalized = canonicalize_unique_quote_offsets(
                            case.resume_text,
                            item,
                        )
                        validated = validate_prediction_sources(
                            case.resume_text,
                            item,
                            enforce_source_guard=True,
                        )
                        if canonicalized:
                            validated = validated.model_copy(
                                update={
                                    "reason_codes": validated.reason_codes
                                    + ["span_offset_canonicalized"]
                                }
                            )
                        items.append(validated)
            else:
                if variant == ExperimentVariant.C_BLOCK_ATTR:
                    parsed_items: list[BlockEvidencePrediction] = (
                        BlockAttrMappingOutput.model_validate(
                            result.parsed_output
                        ).items
                    )
                else:
                    parsed_items = BlockMappingOutput.model_validate(
                        result.parsed_output
                    ).items
                items = self._validate_block_items(case, parsed_items)
                if (
                    any(not item.source_valid for item in items)
                    and config.max_membership_repairs
                ):
                    result, repair_trace, repair_error = (
                        self._repair_block_membership(
                            case=case,
                            invalid_items=items,
                            provider=provider,
                            config=config,
                            variant=variant,
                        )
                    )
                    traces.extend(repair_trace)
                    self._write_raw(
                        raw_root,
                        case.case_id,
                        f"{variant.value}_repair",
                        result,
                        repair_error,
                    )
                    if result is None or repair_error is not None:
                        raise EvidenceGroundingRunError(
                            repair_error or "block_membership_repair_failed"
                        )
                    if variant == ExperimentVariant.C_BLOCK_ATTR:
                        parsed_items = BlockAttrMappingOutput.model_validate(
                            result.parsed_output
                        ).items
                    else:
                        parsed_items = BlockMappingOutput.model_validate(
                            result.parsed_output
                        ).items
                    items = self._validate_block_items(case, parsed_items)
                if variant == ExperimentVariant.C_BLOCK_ATTR:
                    attr_items = [
                        item
                        for item in parsed_items
                        if isinstance(item, BlockAttrEvidencePrediction)
                    ]
                    if len(attr_items) != len(parsed_items):
                        raise ValueError("attribute_contract_mismatch")
                    items, attr_traces = self._apply_block_attr_guard(
                        case=case,
                        attr_items=attr_items,
                        validated_items=items,
                        provider=provider,
                        config=config,
                        raw_root=raw_root,
                    )
                    traces.extend(attr_traces)
            self._validate_requirement_coverage(case, items)
            record = PredictionRecord(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                variant=variant,
                schema_valid=True,
                items=items,
                traces=traces,
            )
        except (ValidationError, ValueError, EvidenceGroundingRunError) as exc:
            record = self._failure_record(
                case,
                variant,
                dataset_hash,
                protocol_hash,
                traces,
                self._safe_error(exc),
            )
        self._write_record(target, record)
        return record

    def _load_or_run_entailment(
        self,
        *,
        case: DatasetCase,
        c_record: PredictionRecord,
        provider: LLMProvider,
        config: GroundingRunnerConfig,
        dataset_hash: str,
        protocol_hash: str,
        prediction_root: Path,
        raw_root: Path,
    ) -> PredictionRecord:
        variant = ExperimentVariant.D_ENTAILMENT_GATE
        target = prediction_root / f"{case.case_id}_{variant.value}.json"
        cached = self._load_cache(target, dataset_hash, protocol_hash)
        if cached is not None:
            return cached
        if not c_record.schema_valid:
            record = self._failure_record(
                case,
                variant,
                dataset_hash,
                protocol_hash,
                list(c_record.traces),
                "upstream_membership_failure",
            )
            self._write_record(target, record)
            return record
        validator_input = [
            {
                "requirement_id": item.requirement_id,
                "requirement": next(
                    unit.requirement
                    for unit in case.requirements
                    if unit.requirement_id == item.requirement_id
                ),
                "verified_spans": [
                    span.model_dump(mode="json")
                    for span in item.spans
                    if item.source_valid
                ],
            }
            for item in c_record.items
        ]
        result, validator_traces, error = self._invoke(
            provider=provider,
            config=config,
            output_schema=EntailmentBatchOutput,
            system_prompt=self._entailment_system_prompt(),
            user_prompt=json.dumps(
                {"units": validator_input},
                ensure_ascii=False,
            ),
            case_id=case.case_id,
            node_id="grounding:D:validator",
        )
        self._write_raw(raw_root, case.case_id, variant.value, result, error)
        traces = list(c_record.traces) + validator_traces
        if result is None or error is not None:
            record = self._failure_record(
                case,
                variant,
                dataset_hash,
                protocol_hash,
                traces,
                error or "validator_provider_error",
            )
            self._write_record(target, record)
            return record
        try:
            decisions = EntailmentBatchOutput.model_validate(result.parsed_output).items
            decision_by_id = {item.requirement_id: item for item in decisions}
            if len(decision_by_id) != len(decisions):
                raise ValueError("duplicate_validator_requirement_id")
            expected = {item.requirement_id for item in c_record.items}
            if set(decision_by_id) != expected:
                raise ValueError("validator_requirement_coverage_mismatch")
            items = [
                apply_entailment_cap(
                    item,
                    decision_by_id[item.requirement_id].relation,
                )
                for item in c_record.items
            ]
            record = PredictionRecord(
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                case_id=case.case_id,
                variant=variant,
                schema_valid=True,
                items=items,
                traces=traces,
            )
        except (ValidationError, ValueError) as exc:
            record = self._failure_record(
                case,
                variant,
                dataset_hash,
                protocol_hash,
                traces,
                self._safe_error(exc),
            )
        self._write_record(target, record)
        return record

    def _load_or_run_lexical(
        self,
        *,
        case: DatasetCase,
        dataset_hash: str,
        protocol_hash: str,
        prediction_root: Path,
    ) -> PredictionRecord:
        variant = ExperimentVariant.E_LEXICAL_BASELINE
        target = prediction_root / f"{case.case_id}_{variant.value}.json"
        cached = self._load_cache(target, dataset_hash, protocol_hash)
        if cached is not None:
            return cached
        lines = [line.strip() for line in case.resume_text.splitlines() if line.strip()]
        items: list[ValidatedEvidencePrediction] = []
        for index, unit in enumerate(case.requirements, start=1):
            tokens = {
                token.casefold()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}", unit.requirement)
                if len(token) >= 2
            }
            matches = [
                line
                for line in lines
                if any(token in line.casefold() for token in tokens)
            ]
            quote = matches[0] if matches else ""
            located = locate_unique_quote(case.resume_text, quote) if quote else None
            spans = (
                [EvidenceSpan(start_char=located[0], end_char=located[1], quote=quote)]
                if located
                else []
            )
            lowered = quote.casefold()
            if not quote:
                level = EvidenceLevel.NONE
            elif any(marker in lowered for marker in ("尚未", "没有", "计划", "了解", "技能：")):
                level = EvidenceLevel.C0
            elif re.search(r"\d", quote) and any(
                marker in lowered for marker in ("实现", "构建", "设计", "负责", "编写")
            ):
                level = EvidenceLevel.C3
            else:
                level = EvidenceLevel.C2
            item = EvidencePrediction(
                evidence_id=f"lex_{index:03d}",
                requirement_id=unit.requirement_id,
                claim=unit.requirement,
                proposed_level=level,
                spans=spans,
                risk="offline lexical baseline",
                abstain=level == EvidenceLevel.NONE,
            )
            items.append(validate_prediction_sources(case.resume_text, item))
        record = PredictionRecord(
            protocol_hash=protocol_hash,
            dataset_hash=dataset_hash,
            case_id=case.case_id,
            variant=variant,
            schema_valid=True,
            items=items,
            traces=[],
        )
        self._write_record(target, record)
        return record

    @staticmethod
    def _validate_block_items(
        case: DatasetCase,
        items: Sequence[BlockEvidencePrediction],
    ) -> list[ValidatedEvidencePrediction]:
        evidence_ids = [item.evidence_id for item in items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate_evidence_id")
        refs: list[AtomicEvidenceRef] = []
        refs_by_item: dict[str, list[str]] = {}
        for item in items:
            item_ref_ids = []
            for index, evidence in enumerate(item.evidence_refs, start=1):
                evidence_ref = f"{item.evidence_id}::ref-{index}"
                item_ref_ids.append(evidence_ref)
                refs.append(
                    AtomicEvidenceRef(
                        evidence_ref=evidence_ref,
                        source_block_id=evidence.source_block_id,
                        quote=evidence.quote,
                        occurrence_index=evidence.occurrence_index,
                    )
                )
            refs_by_item[item.evidence_id] = item_ref_ids
        _, located = locate_atomic_evidence_refs(
            case.resume_text,
            refs,
            expected_resume_sha256=sha256_text(case.resume_text),
        )
        located_by_id = {
            evidence.evidence_ref: evidence for evidence in located
        }
        validated_items: list[ValidatedEvidencePrediction] = []
        for item in items:
            item_locations = [
                located_by_id[evidence_ref]
                for evidence_ref in refs_by_item[item.evidence_id]
            ]
            valid_spans = [
                EvidenceSpan(
                    start_char=evidence.start_char,
                    end_char=evidence.end_char,
                    quote=evidence.raw_quote,
                )
                for evidence in item_locations
                if evidence.source_valid
                and evidence.start_char is not None
                and evidence.end_char is not None
                and evidence.raw_quote is not None
            ]
            prediction = EvidencePrediction(
                evidence_id=item.evidence_id,
                requirement_id=item.requirement_id,
                claim=item.claim,
                proposed_level=item.proposed_level,
                spans=valid_spans,
                risk=item.risk,
                abstain=(
                    item.abstain or item.proposed_level == EvidenceLevel.NONE
                ),
            )
            validated = validate_prediction_sources(
                case.resume_text,
                prediction,
                enforce_source_guard=True,
            )
            invalid = [
                evidence for evidence in item_locations if not evidence.source_valid
            ]
            if invalid:
                final_level = (
                    EvidenceLevel.NONE
                    if item.proposed_level == EvidenceLevel.NONE
                    else EvidenceLevel.C0
                )
                error_codes = sorted(
                    {
                        f"block_{evidence.source_error.value}"
                        for evidence in invalid
                        if evidence.source_error is not None
                    }
                )
                validated = validated.model_copy(
                    update={
                        "source_valid": False,
                        "final_level": final_level,
                        "reason_codes": (
                            validated.reason_codes
                            + error_codes
                            + ["block_source_guard_downgrade"]
                        ),
                    }
                )
            elif (
                item.proposed_level == EvidenceLevel.NONE
                and item.evidence_refs
            ):
                validated = validated.model_copy(
                    update={
                        "reason_codes": (
                            validated.reason_codes
                            + ["none_with_auditable_context_evidence"]
                        )
                    }
                )
            validated_items.append(validated)
        return validated_items

    def _repair_block_membership(
        self,
        *,
        case: DatasetCase,
        invalid_items: Sequence[ValidatedEvidencePrediction],
        provider: LLMProvider,
        config: GroundingRunnerConfig,
        variant: ExperimentVariant,
    ) -> tuple[ProviderResult | None, list[SanitizedTrace], str | None]:
        invalid_ids = [
            item.requirement_id for item in invalid_items if not item.source_valid
        ]
        is_attr = variant == ExperimentVariant.C_BLOCK_ATTR
        return self._invoke(
            provider=provider,
            config=config,
            output_schema=BlockAttrMappingOutput if is_attr else BlockMappingOutput,
            system_prompt=(
                self._block_attr_system_prompt()
                if is_attr
                else self._block_system_prompt()
            ),
            user_prompt=(
                self._block_mapping_user_prompt(case)
                + "\n\nThe previous output had invalid block/quote evidence for: "
                + ", ".join(invalid_ids)
                + ". Regenerate the COMPLETE mapping. Copy every quote verbatim "
                "from its declared source block. Do not output character offsets."
            ),
            case_id=case.case_id,
            node_id=f"grounding:{variant.value}:membership_repair",
        )

    def _apply_block_attr_guard(
        self,
        *,
        case: DatasetCase,
        attr_items: Sequence[BlockAttrEvidencePrediction],
        validated_items: Sequence[ValidatedEvidencePrediction],
        provider: LLMProvider,
        config: GroundingRunnerConfig,
        raw_root: Path,
    ) -> tuple[list[ValidatedEvidencePrediction], list[SanitizedTrace]]:
        initial_by_id = {item.requirement_id: item for item in attr_items}
        validated_by_id = {item.requirement_id: item for item in validated_items}
        if set(initial_by_id) != set(validated_by_id):
            raise ValueError("attribute_requirement_coverage_mismatch")
        triggers = {
            requirement_id: self._block_attr_risk_reasons(
                case,
                initial_by_id[requirement_id],
                validated,
            )
            for requirement_id, validated in validated_by_id.items()
        }
        triggered = {key: value for key, value in triggers.items() if value}
        decisions: dict[str, BlockAttrDecision] = {}
        verifier_failed = False
        traces: list[SanitizedTrace] = []
        if triggered:
            result, traces, error = self._invoke(
                provider=provider,
                config=config,
                output_schema=BlockAttrDecisionOutput,
                system_prompt=self._block_attr_verifier_system_prompt(),
                user_prompt=self._block_attr_verifier_user_prompt(
                    case=case,
                    initial_by_id=initial_by_id,
                    validated_by_id=validated_by_id,
                    triggers=triggered,
                ),
                case_id=case.case_id,
                node_id="grounding:C_BLOCK_ATTR:attribute_verifier",
            )
            self._write_raw(
                raw_root,
                case.case_id,
                f"{ExperimentVariant.C_BLOCK_ATTR.value}_attribute_verifier",
                result,
                error,
            )
            if result is None or error is not None:
                verifier_failed = True
            else:
                try:
                    verified = BlockAttrDecisionOutput.model_validate(
                        result.parsed_output
                    ).items
                    decisions = {item.requirement_id: item for item in verified}
                    if (
                        len(decisions) != len(verified)
                        or set(decisions) != set(triggered)
                    ):
                        raise ValueError("attribute_verifier_coverage_mismatch")
                except (ValidationError, ValueError):
                    verifier_failed = True

        guarded: list[ValidatedEvidencePrediction] = []
        for requirement_id, validated in validated_by_id.items():
            initial = initial_by_id[requirement_id]
            risk_reasons = triggers[requirement_id]
            if risk_reasons and verifier_failed:
                guarded.append(
                    self._cap_block_attr_item(
                        validated,
                        actor="unknown",
                        polarity="uncertain",
                        completion_status="negated",
                        extra_reasons=(
                            risk_reasons
                            + [
                                "attr_second_pass_triggered",
                                "attr_second_pass_failure_abstain",
                            ]
                        ),
                    )
                )
                continue
            decision = decisions.get(requirement_id)
            if decision is None:
                actor = initial.actor
                polarity = initial.polarity
                completion_status = initial.completion_status
                extra_reasons = risk_reasons
            else:
                actor = self._conservative_actor(initial.actor, decision.actor)
                polarity = self._conservative_polarity(
                    initial.polarity,
                    decision.polarity,
                )
                completion_status = self._conservative_completion(
                    initial.completion_status,
                    decision.completion_status,
                )
                extra_reasons = risk_reasons + [
                    "attr_second_pass_triggered",
                    "attr_second_pass_verified",
                ]
                if (
                    decision.actor != initial.actor
                    or decision.polarity != initial.polarity
                    or decision.completion_status != initial.completion_status
                ):
                    extra_reasons.append("attr_second_pass_disagreement")
                if (
                    actor != initial.actor
                    or polarity != initial.polarity
                    or completion_status != initial.completion_status
                ):
                    extra_reasons.append("attr_second_pass_conservative_merge")
            guarded.append(
                self._cap_block_attr_item(
                    validated,
                    actor=actor,
                    polarity=polarity,
                    completion_status=completion_status,
                    extra_reasons=extra_reasons,
                )
            )
        return guarded, traces

    @staticmethod
    def _block_attr_risk_reasons(
        case: DatasetCase,
        item: BlockAttrEvidencePrediction,
        validated: ValidatedEvidencePrediction,
    ) -> list[str]:
        requirement = next(
            unit.requirement
            for unit in case.requirements
            if unit.requirement_id == item.requirement_id
        )
        quoted = "\n".join(span.quote for span in validated.spans)
        combined = f"{requirement}\n{quoted}".casefold()
        reasons: list[str] = []
        if item.actor != "candidate":
            reasons.append("attr_risk_actor")
        if item.polarity != "positive":
            reasons.append("attr_risk_polarity")
        if item.completion_status != "completed":
            reasons.append("attr_risk_completion")
        if any(marker in combined for marker in _OWNERSHIP_MARKERS):
            reasons.append("attr_risk_ownership_lexical")
        if any(marker in combined for marker in _NEGATION_MARKERS):
            reasons.append("attr_risk_negation_lexical")
        if any(marker in combined for marker in _PLAN_MARKERS):
            reasons.append("attr_risk_plan_lexical")
        if _NUMERIC_METRIC_RE.search(combined):
            reasons.append("attr_risk_numeric_attribution")
        if not validated.source_valid:
            reasons.append("attr_risk_source_invalid")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _cap_block_attr_item(
        item: ValidatedEvidencePrediction,
        *,
        actor: Literal["candidate", "team", "unknown"],
        polarity: Literal["positive", "negative", "uncertain"],
        completion_status: Literal["completed", "planned", "negated"],
        extra_reasons: Sequence[str],
    ) -> ValidatedEvidencePrediction:
        caps: list[EvidenceLevel] = []
        if actor == "team":
            caps.append(EvidenceLevel.C0)
        elif actor == "unknown":
            caps.append(EvidenceLevel.C1)
        if polarity == "uncertain":
            caps.append(EvidenceLevel.C0)
        elif polarity == "negative":
            caps.append(EvidenceLevel.NONE)
        if completion_status in {"planned", "negated"}:
            caps.append(EvidenceLevel.NONE)
        final_level = item.final_level
        for cap in caps:
            if _LEVEL_ORDER[final_level] > _LEVEL_ORDER[cap]:
                final_level = cap
        reasons = item.reason_codes + [
            f"attr_actor_{actor}",
            f"attr_polarity_{polarity}",
            f"attr_completion_{completion_status}",
            *extra_reasons,
        ]
        if final_level != item.final_level:
            reasons.append("attr_guard_downgrade")
        if final_level == EvidenceLevel.NONE:
            reasons.append("attr_guard_abstain")
        return item.model_copy(
            update={
                "final_level": final_level,
                "abstain": item.abstain or final_level == EvidenceLevel.NONE,
                "reason_codes": list(dict.fromkeys(reasons)),
            }
        )

    @staticmethod
    def _conservative_actor(
        first: Literal["candidate", "team", "unknown"],
        second: Literal["candidate", "team", "unknown"],
    ) -> Literal["candidate", "team", "unknown"]:
        rank = {"team": 0, "unknown": 1, "candidate": 2}
        return first if rank[first] <= rank[second] else second

    @staticmethod
    def _conservative_polarity(
        first: Literal["positive", "negative", "uncertain"],
        second: Literal["positive", "negative", "uncertain"],
    ) -> Literal["positive", "negative", "uncertain"]:
        rank = {"negative": 0, "uncertain": 1, "positive": 2}
        return first if rank[first] <= rank[second] else second

    @staticmethod
    def _conservative_completion(
        first: Literal["completed", "planned", "negated"],
        second: Literal["completed", "planned", "negated"],
    ) -> Literal["completed", "planned", "negated"]:
        rank = {"negated": 0, "planned": 1, "completed": 2}
        return first if rank[first] <= rank[second] else second

    def _repair_membership(
        self,
        *,
        case: DatasetCase,
        invalid_items: Sequence[ValidatedEvidencePrediction],
        provider: LLMProvider,
        config: GroundingRunnerConfig,
    ) -> tuple[ProviderResult | None, list[SanitizedTrace], str | None]:
        invalid_ids = [
            item.requirement_id for item in invalid_items if not item.source_valid
        ]
        return self._invoke(
            provider=provider,
            config=config,
            output_schema=ExtractiveMappingOutput,
            system_prompt=self._extractive_system_prompt(enforce_offsets=True),
            user_prompt=(
                self._mapping_user_prompt(case)
                + "\n\nThe previous output had invalid source spans for: "
                + ", ".join(invalid_ids)
                + ". Regenerate the COMPLETE mapping. Every quote must be copied verbatim "
                "and every [start_char,end_char) must slice to that quote in Python."
            ),
            case_id=case.case_id,
            node_id="grounding:C:membership_repair",
        )

    def _invoke(
        self,
        *,
        provider: LLMProvider,
        config: GroundingRunnerConfig,
        output_schema,
        system_prompt: str,
        user_prompt: str,
        case_id: str,
        node_id: str,
        prompt_version: str | None = None,
    ) -> tuple[ProviderResult | None, list[SanitizedTrace], str | None]:
        traces: list[SanitizedTrace] = []
        trace_context = TraceContext(
            session_id=f"grounding:{case_id}",
            run_id=f"grounding:{case_id}:{config.stage}",
            node_id=node_id,
            skill_id="evidence-grounding-experiment",
            skill_version="v2-pilot",
            prompt_version=(
                prompt_version
                or (
                    C_BLOCK_ATTR_PROMPT_VERSION
                    if "C_BLOCK_ATTR" in node_id
                    else PROMPT_VERSION
                )
            ),
        )
        for attempt in range(config.max_transport_retries + 1):
            if self._provider_calls >= config.max_provider_calls:
                return None, traces, "provider_call_budget_exceeded"
            self._provider_calls += 1
            try:
                result = provider.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=output_schema,
                    tools=[],
                    temperature=0.0,
                    max_output_tokens=config.max_output_tokens,
                    trace=trace_context,
                )
            except ProviderError as exc:
                traces.append(
                    SanitizedTrace(
                        provider=str(getattr(provider, "provider_name", "unknown")),
                        model=str(getattr(provider, "model", config.model)),
                        latency_ms=0,
                        schema_valid=False,
                        error_code=exc.error_code,
                    )
                )
                if exc.retryable and attempt < config.max_transport_retries:
                    continue
                return None, traces, exc.error_code
            trace = SanitizedTrace.model_validate(
                result.model_dump(
                    include={
                        "provider",
                        "model",
                        "latency_ms",
                        "input_tokens",
                        "output_tokens",
                        "schema_valid",
                        "error_code",
                    }
                )
            )
            traces.append(trace)
            if not result.schema_valid or result.parsed_output is None:
                return result, traces, result.error_code or "schema_error"
            return result, traces, None
        return None, traces, "provider_error"

    @staticmethod
    def _adapt_legacy(case: DatasetCase, item: LegacyEvidenceOutput) -> EvidencePrediction:
        located = locate_unique_quote(case.resume_text, item.proof)
        spans = (
            [
                EvidenceSpan(
                    start_char=located[0],
                    end_char=located[1],
                    quote=item.proof,
                )
            ]
            if located
            else (
                []
                if item.level == EvidenceLevel.NONE
                else [EvidenceSpan(start_char=0, end_char=1, quote=item.proof)]
            )
        )
        return EvidencePrediction(
            evidence_id=item.evidence_id,
            requirement_id=item.requirement_id,
            claim=item.claim,
            proposed_level=item.level,
            spans=spans,
            risk=item.risk,
            abstain=item.level == EvidenceLevel.NONE,
        )

    @staticmethod
    def _validate_requirement_coverage(
        case: DatasetCase,
        items: Sequence[ValidatedEvidencePrediction],
    ) -> None:
        expected = {unit.requirement_id for unit in case.requirements}
        actual = [item.requirement_id for item in items]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("requirement_coverage_mismatch")
        evidence_ids = [item.evidence_id for item in items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate_evidence_id")

    @staticmethod
    def _mapping_user_prompt(case: DatasetCase) -> str:
        return json.dumps(
            {
                "requirements": [
                    {
                        "requirement_id": unit.requirement_id,
                        "requirement": unit.requirement,
                    }
                    for unit in case.requirements
                ],
                "resume": case.resume_text,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _block_mapping_user_prompt(case: DatasetCase) -> str:
        return json.dumps(
            {
                "resume_sha256": sha256_text(case.resume_text),
                "source_blocks": [
                    block.model_dump(mode="json")
                    for block in build_source_blocks(case.resume_text)
                ],
                "requirements": [
                    {
                        "requirement_id": unit.requirement_id,
                        "requirement": unit.requirement,
                    }
                    for unit in case.requirements
                ],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _block_attr_verifier_user_prompt(
        *,
        case: DatasetCase,
        initial_by_id: dict[str, BlockAttrEvidencePrediction],
        validated_by_id: dict[str, ValidatedEvidencePrediction],
        triggers: dict[str, list[str]],
    ) -> str:
        requirement_by_id = {
            unit.requirement_id: unit.requirement for unit in case.requirements
        }
        return json.dumps(
            {
                "units": [
                    {
                        "requirement_id": requirement_id,
                        "requirement": requirement_by_id[requirement_id],
                        "verified_quotes": [
                            span.quote
                            for span in validated_by_id[requirement_id].spans
                        ],
                        "initial_attributes": {
                            "actor": initial_by_id[requirement_id].actor,
                            "polarity": initial_by_id[requirement_id].polarity,
                            "completion_status": initial_by_id[
                                requirement_id
                            ].completion_status,
                        },
                        "risk_triggers": risk_triggers,
                    }
                    for requirement_id, risk_triggers in triggers.items()
                ]
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _baseline_system_prompt() -> str:
        return (
            "You map every requirement to resume evidence without inventing proof. "
            "Return exactly one item per requirement. Use levels C0, C1, C2, C3, or None. "
            "The proof field is free text, matching the current V1 contract. "
            "Use None when the resume has no evidence. Do not omit requirements."
        )

    @staticmethod
    def _extractive_system_prompt(*, enforce_offsets: bool) -> str:
        offset_rule = (
            "For every span, quote MUST be copied verbatim from resume and Python "
            "character offsets [start_char,end_char) MUST satisfy resume[start_char:end_char] "
            "== quote. Count Unicode characters, not UTF-8 bytes. "
            if enforce_offsets
            else "Prefer verbatim resume quotes with character offsets. "
        )
        return (
            "You are an evidence mapper. Return exactly one item per requirement and never "
            "invent evidence. Levels: C3=action+artifact+direct result, C2=action+artifact, "
            "C1=learning/reproduction/limited participation, C0=keyword or related concept, "
            "None=no evidence or explicit negation. "
            + offset_rule
            + "Use an empty spans list and abstain=true for None. Do not treat plans, "
            "negations, or metrics from another task as supporting evidence."
        )

    @staticmethod
    def _block_system_prompt() -> str:
        return (
            "You are an evidence mapper. Return exactly one item per requirement. "
            "Levels: C3=action+artifact+direct result, C2=action+artifact, "
            "C1=learning/reproduction/limited participation, C0=keyword/context/"
            "team-only result, None=no positive evidence or explicit negation. "
            "For non-None items, select one or more evidence_refs containing only "
            "source_block_id, verbatim quote, and occurrence_index when the same "
            "quote repeats in that block. Never output character offsets; Python "
            "will locate quotes. For None, evidence_refs may be empty or may retain "
            "verbatim negative/context evidence for audit; Python always forces the "
            "final level to None and abstain=true. "
            "Do not borrow metrics across requirements or attribute team results "
            "to the candidate. Do not omit requirements."
        )

    @staticmethod
    def _block_attr_system_prompt() -> str:
        return (
            "You are an evidence mapper. Return exactly one item per requirement. "
            "Levels: C3=action+artifact+direct result, C2=action+artifact, "
            "C1=learning/reproduction/limited participation, C0=keyword/context/"
            "team-only result, None=no positive evidence or explicit negation. "
            "For every item, classify actor as candidate, team, or unknown; polarity "
            "as positive, negative, or uncertain; and completion_status as completed, "
            "planned, or negated. Classify attributes relative to the requirement and "
            "selected quotes, not the resume globally. Candidate means the quote assigns "
            "the relevant action/result to the candidate; team means only a group owns it. "
            "For non-None items, select one or more evidence_refs containing only "
            "source_block_id, verbatim quote, and occurrence_index when the same quote "
            "repeats in that block. Never output character offsets; Python will locate "
            "quotes. For None, evidence_refs may be empty or may retain verbatim negative/"
            "context evidence for audit. Do not borrow metrics across requirements, turn "
            "plans into completed work, or attribute team results to the candidate. "
            "Python may only keep, downgrade, or abstain; it can never upgrade."
        )

    @staticmethod
    def _block_attr_verifier_system_prompt() -> str:
        return (
            "Independently verify attributes for only the supplied risk-triggered units. "
            "Use only the requirement and VERIFIED quotes. Return exactly one decision "
            "per supplied requirement_id. actor=candidate only when the relevant action/"
            "result is assigned to the candidate, team when only a group owns it, otherwise "
            "unknown. polarity=negative for explicit denial, uncertain for ambiguous or "
            "context-only evidence, otherwise positive. completion_status=planned for "
            "future intent, negated for explicitly uncompleted work, otherwise completed. "
            "Do not assign a numeric result from one task to another. This pass can only "
            "cause deterministic keep/downgrade/abstain and never an upgrade."
        )

    @classmethod
    def c_block_attr_contract(cls) -> dict:
        return {
            "candidate": ExperimentVariant.C_BLOCK_ATTR.value,
            "prompt_version": C_BLOCK_ATTR_PROMPT_VERSION,
            "guard_version": C_BLOCK_ATTR_GUARD_VERSION,
            "mapper_system_prompt": cls._block_attr_system_prompt(),
            "verifier_system_prompt": cls._block_attr_verifier_system_prompt(),
            "mapper_schema": BlockAttrMappingOutput.model_json_schema(),
            "verifier_schema": BlockAttrDecisionOutput.model_json_schema(),
            "risk_trigger_contract": {
                "attribute_non_default": True,
                "ownership_markers": list(_OWNERSHIP_MARKERS),
                "negation_markers": list(_NEGATION_MARKERS),
                "plan_markers": list(_PLAN_MARKERS),
                "numeric_metric_regex": _NUMERIC_METRIC_RE.pattern,
                "source_invalid": True,
            },
            "guard_caps": {
                "actor.team": "C0",
                "actor.unknown": "C1",
                "polarity.uncertain": "C0",
                "polarity.negative": "None",
                "completion_status.planned": "None",
                "completion_status.negated": "None",
            },
            "merge_policy": "most_conservative_attribute_wins",
            "verifier_failure_policy": "triggered_units_abstain",
            "upgrade_allowed": False,
        }

    @staticmethod
    def _entailment_system_prompt() -> str:
        return (
            "Independently judge whether the VERIFIED resume spans support each requirement. "
            "You cannot see and must not infer the mapper's proposed level. Return exactly "
            "one decision per requirement_id. Relations: entailed=direct support; "
            "partial=limited/learning support; related_only=keyword or adjacent concept; "
            "contradicted=span denies the requirement; uncertain=insufficient to decide. "
            "This validator only enables deterministic keep/downgrade/abstain and can never "
            "upgrade evidence."
        )

    @staticmethod
    def _provider(config: GroundingRunnerConfig, api_key: str) -> LLMProvider:
        return AnthropicCompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout_s=config.timeout_s,
        )

    @staticmethod
    def _failure_record(
        case: DatasetCase,
        variant: ExperimentVariant,
        dataset_hash: str,
        protocol_hash: str,
        traces: list[SanitizedTrace],
        error: str,
    ) -> PredictionRecord:
        return PredictionRecord(
            protocol_hash=protocol_hash,
            dataset_hash=dataset_hash,
            case_id=case.case_id,
            variant=variant,
            schema_valid=False,
            execution_error=re.sub(r"[^a-z0-9_]", "_", error.casefold())[:80],
            traces=traces,
        )

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc).casefold()
        for known in (
            "requirement_coverage_mismatch",
            "duplicate_evidence_id",
            "validator_requirement_coverage_mismatch",
            "duplicate_validator_requirement_id",
            "membership_repair_failed",
        ):
            if known in text:
                return known
        return "schema_or_contract_failure"

    @staticmethod
    def _load_cache(
        target: Path,
        dataset_hash: str,
        protocol_hash: str,
    ) -> PredictionRecord | None:
        if not target.is_file():
            return None
        try:
            record = PredictionRecord.model_validate_json(target.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None
        if record.dataset_hash != dataset_hash or record.protocol_hash != protocol_hash:
            raise EvidenceGroundingRunError("stale_prediction_cache")
        return record

    @classmethod
    def _write_record(cls, path: Path, record: PredictionRecord) -> None:
        cls._write_json(path, record.model_dump(mode="json"))

    @classmethod
    def _write_raw(
        cls,
        root: Path,
        case_id: str,
        suffix: str,
        result: ProviderResult | None,
        error: str | None,
    ) -> None:
        payload = {
            "case_id": case_id,
            "variant_or_stage": suffix,
            "schema_valid": bool(result and result.schema_valid),
            "error_code": error,
            "raw_output": result.raw_output if result is not None else None,
        }
        cls._write_json(root / f"{case_id}_{suffix}.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        atomic_write_json(path, payload)
