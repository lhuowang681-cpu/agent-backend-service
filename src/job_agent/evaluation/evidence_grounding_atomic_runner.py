from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from pydantic import Field, ValidationError, model_validator

from job_agent.atomic_io import atomic_write_json
from job_agent.evaluation.evidence_grounding_atomic import (
    AtomicClaim,
    AtomicClaimRole,
    AtomicEvidenceGraph,
    AtomicEvidenceGroup,
    AtomicEvidenceRef,
    AtomicRequirementDecision,
    ClaimActor,
    ClaimPolarity,
    ClaimSupportStatus,
    EvidenceRelation,
    ValidatedAtomicEvidenceGraph,
    build_source_blocks,
    sha256_text,
    validate_atomic_evidence_graph,
)
from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    EvidenceSpan,
    ExperimentVariant,
    PredictionRecord,
    SanitizedTrace,
    ValidatedEvidencePrediction,
    load_jsonl_dataset,
    sha256_file,
    stable_protocol_hash,
)
from job_agent.llm.provider import (
    LLMProvider,
    ProviderError,
    ProviderResult,
    TraceContext,
)
from job_agent.llm.providers.anthropic_compatible import AnthropicCompatibleProvider
from job_agent.schemas import EvidenceLevel, StrictModel


ATOMIC_PROMPT_VERSION = "evidence-grounding-c-atomic-v1"
ATOMIC_VALIDATOR_VERSION = "claim-evidence-graph-v1"


class AtomicEvidenceGroundingRunError(RuntimeError):
    pass


class AtomicRunnerConfig(StrictModel):
    dataset_path: str = "data/eval/evidence_grounding_v2/pilot.jsonl"
    output_root: str = "output/private/evidence_grounding_v2/c_atomic_pilot"
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-4.7"
    api_key_env: str = "JOB_AGENT_LIVE_API_KEY"
    allow_network: bool = False
    timeout_s: float = Field(default=90.0, gt=0, le=300)
    max_output_tokens: int = Field(default=4096, ge=512, le=8192)
    max_transport_retries: int = Field(default=1, ge=0, le=2)
    max_graph_repairs: int = Field(default=1, ge=0, le=1)
    max_provider_calls: int = Field(default=30, ge=1, le=200)
    seed: int = 20260726
    expected_dataset_hash: str | None = Field(default=None, min_length=64, max_length=64)
    allowed_hosts: tuple[str, ...] = ("open.bigmodel.cn",)

    @model_validator(mode="after")
    def validate_endpoint(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or parsed.hostname not in self.allowed_hosts:
            raise ValueError("atomic grounding runner endpoint is not allowlisted")
        return self


class AtomicValidatedArtifact(StrictModel):
    schema_version: str = "evidence-grounding-c-atomic-artifact-v1"
    protocol_hash: str = Field(min_length=64, max_length=64)
    dataset_hash: str = Field(min_length=64, max_length=64)
    case_id: str
    graph: ValidatedAtomicEvidenceGraph
    draft_canonicalization: dict[str, int] = Field(default_factory=dict)


class AtomicClaimDraft(StrictModel):
    claim_id: str = Field(min_length=1)
    requirement_ids: list[str] = Field(default_factory=list)
    claim: str = Field(min_length=1)
    actor: ClaimActor
    polarity: ClaimPolarity
    roles: list[AtomicClaimRole] = Field(min_length=1)
    evidence_groups: list[AtomicEvidenceGroup] = Field(min_length=1)


class AtomicEvidenceGraphDraft(StrictModel):
    schema_version: str = "evidence-grounding-atomic-v1"
    resume_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_refs: list[AtomicEvidenceRef] = Field(default_factory=list)
    claims: list[AtomicClaimDraft] = Field(default_factory=list)
    requirements: list[AtomicRequirementDecision] = Field(min_length=1)


def canonicalize_atomic_graph_draft(
    draft: AtomicEvidenceGraphDraft,
) -> tuple[AtomicEvidenceGraph, dict[str, int]]:
    required_claim_ids = {
        claim_id
        for decision in draft.requirements
        for claim_id in decision.claim_ids
    }
    requirements_by_claim: dict[str, set[str]] = {}
    for decision in draft.requirements:
        for claim_id in decision.claim_ids:
            requirements_by_claim.setdefault(claim_id, set()).add(
                decision.requirement_id
            )
    claims: list[AtomicClaim] = []
    repaired_links = 0
    for claim in draft.claims:
        if claim.claim_id not in required_claim_ids:
            continue
        inferred_requirements = sorted(
            requirements_by_claim.get(claim.claim_id, set())
        )
        if set(claim.requirement_ids) != set(inferred_requirements):
            repaired_links += 1
        claims.append(
            AtomicClaim(
                claim_id=claim.claim_id,
                requirement_ids=inferred_requirements,
                claim=claim.claim,
                actor=claim.actor,
                polarity=claim.polarity,
                roles=claim.roles,
                evidence_groups=claim.evidence_groups,
            )
        )
    used_evidence = {
        evidence_ref
        for claim in claims
        for group in claim.evidence_groups
        for evidence_ref in group.evidence_refs
    }
    evidence_refs = [
        evidence
        for evidence in draft.evidence_refs
        if evidence.evidence_ref in used_evidence
    ]
    graph = AtomicEvidenceGraph(
        resume_sha256=draft.resume_sha256,
        evidence_refs=evidence_refs,
        claims=claims,
        requirements=draft.requirements,
    )
    return graph, {
        "dropped_unused_claims": len(draft.claims) - len(claims),
        "dropped_unused_evidence_refs": (
            len(draft.evidence_refs) - len(evidence_refs)
        ),
        "repaired_claim_requirement_links": repaired_links,
    }


ProviderFactory = Callable[[AtomicRunnerConfig, str], LLMProvider]


class AtomicEvidenceGroundingRunner:
    def __init__(self, *, provider_factory: ProviderFactory | None = None) -> None:
        self.provider_factory = provider_factory or self._provider
        self._provider_calls = 0

    @property
    def provider_calls(self) -> int:
        return self._provider_calls

    def run(self, config: AtomicRunnerConfig) -> list[PredictionRecord]:
        if not config.allow_network:
            raise AtomicEvidenceGroundingRunError(
                "network_not_explicitly_enabled"
            )
        dataset_path = Path(config.dataset_path)
        dataset_hash = sha256_file(dataset_path)
        if config.expected_dataset_hash and dataset_hash != config.expected_dataset_hash:
            raise AtomicEvidenceGroundingRunError("dataset_hash_mismatch")
        dataset = load_jsonl_dataset(dataset_path)
        cases = list(dataset.cases)
        random.Random(config.seed).shuffle(cases)

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise AtomicEvidenceGroundingRunError("credential_not_injected")
        provider = self.provider_factory(config, api_key)
        del api_key

        protocol_payload = {
            "prompt_version": ATOMIC_PROMPT_VERSION,
            "validator_version": ATOMIC_VALIDATOR_VERSION,
            "model": config.model,
            "temperature": 0.0,
            "max_output_tokens": config.max_output_tokens,
            "max_graph_repairs": config.max_graph_repairs,
            "seed": config.seed,
        }
        protocol_hash = stable_protocol_hash(protocol_payload)
        output_root = Path(config.output_root)
        prediction_root = output_root / "predictions"
        graph_root = output_root / "graphs"
        raw_root = output_root / "raw"
        for path in (prediction_root, graph_root, raw_root):
            path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            output_root / "run_manifest.json",
            {
                **protocol_payload,
                "protocol_hash": protocol_hash,
                "dataset_hash": dataset_hash,
                "case_ids": [case.case_id for case in cases],
                "variant": ExperimentVariant.C_ATOMIC.value,
                "label_provenance": "ai_draft",
                "review_status": "pending_human_review",
                "credential_source": config.api_key_env,
                "raw_artifacts": "private_only",
            },
        )

        records = [
            self._load_or_run_case(
                case=case,
                provider=provider,
                config=config,
                dataset_hash=dataset_hash,
                protocol_hash=protocol_hash,
                prediction_root=prediction_root,
                graph_root=graph_root,
                raw_root=raw_root,
            )
            for case in cases
        ]
        return records

    def _load_or_run_case(
        self,
        *,
        case: DatasetCase,
        provider: LLMProvider,
        config: AtomicRunnerConfig,
        dataset_hash: str,
        protocol_hash: str,
        prediction_root: Path,
        graph_root: Path,
        raw_root: Path,
    ) -> PredictionRecord:
        prediction_path = (
            prediction_root
            / f"{case.case_id}_{ExperimentVariant.C_ATOMIC.value}.json"
        )
        graph_path = (
            graph_root
            / f"{case.case_id}_{ExperimentVariant.C_ATOMIC.value}.json"
        )
        cached = self._load_cache(
            prediction_path=prediction_path,
            graph_path=graph_path,
            dataset_hash=dataset_hash,
            protocol_hash=protocol_hash,
        )
        if cached is not None:
            return cached

        traces: list[SanitizedTrace] = []
        user_prompt = self._mapping_user_prompt(case)
        last_error = "atomic_graph_validation_error"
        for graph_attempt in range(config.max_graph_repairs + 1):
            result, call_traces, error = self._invoke(
                provider=provider,
                config=config,
                system_prompt=self._system_prompt(),
                user_prompt=(
                    user_prompt
                    if graph_attempt == 0
                    else user_prompt
                    + "\n\nPrevious output failed the deterministic graph/source "
                    "contract. Regenerate the COMPLETE graph. Recheck exact block "
                    "quotes, all IDs, actor, polarity, roles, and requirement coverage."
                ),
                case_id=case.case_id,
                node_id=(
                    "grounding:C_ATOMIC"
                    if graph_attempt == 0
                    else "grounding:C_ATOMIC:repair"
                ),
            )
            traces.extend(call_traces)
            self._write_raw(
                raw_root=raw_root,
                case_id=case.case_id,
                attempt=graph_attempt,
                result=result,
                error=error,
            )
            if result is None or error is not None:
                last_error = error or "provider_error"
                if last_error == "provider_call_budget_exceeded":
                    break
                continue
            try:
                draft = AtomicEvidenceGraphDraft.model_validate(
                    result.parsed_output
                )
                prediction, canonicalization = canonicalize_atomic_graph_draft(
                    draft
                )
                validated = validate_atomic_evidence_graph(
                    resume_text=case.resume_text,
                    prediction=prediction,
                    expected_requirement_ids={
                        unit.requirement_id for unit in case.requirements
                    },
                )
                record = self._to_prediction_record(
                    case=case,
                    prediction=prediction,
                    validated=validated,
                    dataset_hash=dataset_hash,
                    protocol_hash=protocol_hash,
                    traces=traces,
                )
                artifact = AtomicValidatedArtifact(
                    protocol_hash=protocol_hash,
                    dataset_hash=dataset_hash,
                    case_id=case.case_id,
                    graph=validated,
                    draft_canonicalization=canonicalization,
                )
                atomic_write_json(
                    graph_path,
                    artifact.model_dump(mode="json"),
                )
                atomic_write_json(
                    prediction_path,
                    record.model_dump(mode="json"),
                )
                return record
            except (ValidationError, ValueError) as exc:
                last_error = self._safe_validation_error(exc)

        record = PredictionRecord(
            protocol_hash=protocol_hash,
            dataset_hash=dataset_hash,
            case_id=case.case_id,
            variant=ExperimentVariant.C_ATOMIC,
            schema_valid=False,
            execution_error=last_error,
            traces=traces,
        )
        atomic_write_json(prediction_path, record.model_dump(mode="json"))
        return record

    def _invoke(
        self,
        *,
        provider: LLMProvider,
        config: AtomicRunnerConfig,
        system_prompt: str,
        user_prompt: str,
        case_id: str,
        node_id: str,
    ) -> tuple[ProviderResult | None, list[SanitizedTrace], str | None]:
        traces: list[SanitizedTrace] = []
        trace_context = TraceContext(
            session_id=f"grounding:{case_id}",
            run_id=f"grounding:{case_id}:c-atomic",
            node_id=node_id,
            skill_id="evidence-grounding-c-atomic",
            skill_version="v1",
            prompt_version=ATOMIC_PROMPT_VERSION,
        )
        for attempt in range(config.max_transport_retries + 1):
            if self._provider_calls >= config.max_provider_calls:
                return None, traces, "provider_call_budget_exceeded"
            self._provider_calls += 1
            try:
                result = provider.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_schema=AtomicEvidenceGraphDraft,
                    tools=[],
                    temperature=0.0,
                    max_output_tokens=config.max_output_tokens,
                    trace=trace_context,
                )
            except ProviderError as exc:
                traces.append(
                    SanitizedTrace(
                        provider=str(
                            getattr(provider, "provider_name", "unknown")
                        ),
                        model=str(getattr(provider, "model", config.model)),
                        latency_ms=0,
                        schema_valid=False,
                        error_code=exc.error_code,
                    )
                )
                if exc.retryable and attempt < config.max_transport_retries:
                    continue
                return None, traces, exc.error_code
            traces.append(
                SanitizedTrace.model_validate(
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
            )
            if not result.schema_valid or result.parsed_output is None:
                return result, traces, result.error_code or "schema_error"
            return result, traces, None
        return None, traces, "provider_error"

    @staticmethod
    def _to_prediction_record(
        *,
        case: DatasetCase,
        prediction: AtomicEvidenceGraph,
        validated: ValidatedAtomicEvidenceGraph,
        dataset_hash: str,
        protocol_hash: str,
        traces: list[SanitizedTrace],
    ) -> PredictionRecord:
        claims_by_id = {claim.claim_id: claim for claim in prediction.claims}
        validated_claims = {
            claim.claim_id: claim for claim in validated.claims
        }
        located_by_id = {
            evidence.evidence_ref: evidence
            for evidence in validated.located_evidence_refs
        }
        requirement_text = {
            unit.requirement_id: unit.requirement for unit in case.requirements
        }
        items: list[ValidatedEvidencePrediction] = []
        for decision, checked in zip(
            prediction.requirements,
            validated.requirements,
        ):
            evidence_ids: set[str] = set()
            claim_texts: list[str] = []
            if checked.final_level != EvidenceLevel.NONE:
                for claim_id in decision.claim_ids:
                    claim = claims_by_id[claim_id]
                    checked_claim = validated_claims[claim_id]
                    claim_texts.append(claim.claim)
                    include_support = (
                        claim.polarity == ClaimPolarity.POSITIVE
                        and checked_claim.support_status
                        in {
                            ClaimSupportStatus.SUPPORTED,
                            ClaimSupportStatus.CONFLICT,
                        }
                    )
                    include_context = (
                        checked_claim.support_status
                        == ClaimSupportStatus.CONTEXT_ONLY
                    )
                    if not (include_support or include_context):
                        continue
                    satisfied = set(checked_claim.satisfied_group_ids)
                    for group in claim.evidence_groups:
                        if group.group_id not in satisfied:
                            continue
                        if include_support and group.relation not in {
                            EvidenceRelation.SUPPORTS,
                            EvidenceRelation.CONTEXT_ONLY,
                        }:
                            continue
                        if include_context and group.relation != EvidenceRelation.CONTEXT_ONLY:
                            continue
                        evidence_ids.update(group.evidence_refs)
            spans = sorted(
                {
                    (
                        located_by_id[evidence_id].start_char,
                        located_by_id[evidence_id].end_char,
                        located_by_id[evidence_id].raw_quote,
                    )
                    for evidence_id in evidence_ids
                    if located_by_id[evidence_id].source_valid
                }
            )
            materialized_spans = [
                EvidenceSpan(
                    start_char=start,
                    end_char=end,
                    quote=quote,
                )
                for start, end, quote in spans
                if start is not None and end is not None and quote is not None
            ]
            items.append(
                ValidatedEvidencePrediction(
                    evidence_id=f"atomic-{decision.requirement_id}",
                    requirement_id=decision.requirement_id,
                    claim="；".join(claim_texts)
                    or requirement_text[decision.requirement_id],
                    proposed_level=decision.proposed_level,
                    spans=materialized_spans,
                    risk=decision.risk,
                    abstain=checked.final_level == EvidenceLevel.NONE,
                    source_valid=True,
                    final_level=checked.final_level,
                    reason_codes=checked.reason_codes,
                )
            )
        return PredictionRecord(
            protocol_hash=protocol_hash,
            dataset_hash=dataset_hash,
            case_id=case.case_id,
            variant=ExperimentVariant.C_ATOMIC,
            schema_valid=True,
            items=items,
            traces=traces,
        )

    @staticmethod
    def _mapping_user_prompt(case: DatasetCase) -> str:
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
    def _system_prompt() -> str:
        return (
            "Build one complete atomic claim-evidence graph for every requirement. "
            "Copy resume_sha256 exactly. Evidence refs contain source_block_id and "
            "a verbatim quote from that block; never output character offsets. Use "
            "occurrence_index only when the same quote repeats in one block. Every "
            "evidence ref and claim must be used, and every claim/requirement link "
            "must be bidirectional. Decompose compound statements into atomic claims. "
            "actor=candidate only for explicitly attributable candidate work; team "
            "results remain actor=team. polarity=negative with relation=SUPPORTS means "
            "the resume supports a negative claim; relation=CONTRADICTS means evidence "
            "refutes the claim. Roles determine safe level: C3 requires candidate "
            "ACTION+ARTIFACT+RESULT; C2 requires candidate ACTION+ARTIFACT or "
            "REPRODUCTION+ARTIFACT; C1 is learning/reproduction/limited participation/"
            "action-only; C0 is keyword/context/result-only; None is no valid positive "
            "support or contradiction-only. Use claim_operator=ALL_OF when every claim "
            "is necessary and ANY_OF only for genuine alternatives. The Python validator "
            "will only keep or downgrade proposed_level and will fail closed."
        )

    @staticmethod
    def _safe_validation_error(exc: Exception) -> str:
        message = str(exc)
        if "resume hash mismatch" in message:
            return "resume_hash_mismatch"
        if "requirement coverage mismatch" in message:
            return "requirement_coverage_mismatch"
        if "source" in message or "quote" in message:
            return "atomic_source_validation_error"
        return "atomic_graph_validation_error"

    @staticmethod
    def _load_cache(
        *,
        prediction_path: Path,
        graph_path: Path,
        dataset_hash: str,
        protocol_hash: str,
    ) -> PredictionRecord | None:
        if not prediction_path.exists():
            return None
        record = PredictionRecord.model_validate_json(
            prediction_path.read_text(encoding="utf-8")
        )
        if (
            record.dataset_hash != dataset_hash
            or record.protocol_hash != protocol_hash
        ):
            return None
        if record.schema_valid:
            if not graph_path.exists():
                return None
            artifact = AtomicValidatedArtifact.model_validate_json(
                graph_path.read_text(encoding="utf-8")
            )
            if (
                artifact.dataset_hash != dataset_hash
                or artifact.protocol_hash != protocol_hash
            ):
                return None
        return record

    @staticmethod
    def _write_raw(
        *,
        raw_root: Path,
        case_id: str,
        attempt: int,
        result: ProviderResult | None,
        error: str | None,
    ) -> None:
        suffix = "initial" if attempt == 0 else f"repair_{attempt}"
        atomic_write_json(
            raw_root
            / f"{case_id}_{ExperimentVariant.C_ATOMIC.value}_{suffix}.json",
            {
                "case_id": case_id,
                "variant": ExperimentVariant.C_ATOMIC.value,
                "attempt": attempt,
                "schema_valid": bool(result and result.schema_valid),
                "error_code": error,
                "raw_output": result.raw_output if result is not None else None,
            },
        )

    @staticmethod
    def _provider(config: AtomicRunnerConfig, api_key: str) -> LLMProvider:
        return AnthropicCompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            timeout_s=config.timeout_s,
        )
