from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from job_agent.schemas import EvidenceLevel, StrictModel


ATOMIC_SCHEMA_VERSION = "evidence-grounding-atomic-v1"

# Relation describes how located evidence relates to the claim. A supported
# negative statement is represented by polarity=NEGATIVE plus relation=SUPPORTS;
# CONTRADICTS means the evidence contradicts the claim text itself.


class EvidenceGroupOperator(str, Enum):
    ALL_OF = "all_of"
    ANY_OF = "any_of"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT_ONLY = "context_only"


class ClaimActor(str, Enum):
    CANDIDATE = "candidate"
    TEAM = "team"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClaimPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class AtomicClaimRole(str, Enum):
    ACTION = "action"
    ARTIFACT = "artifact"
    RESULT = "result"
    OWNERSHIP = "ownership"
    LEARNING = "learning"
    REPRODUCTION = "reproduction"
    LIMITED_PARTICIPATION = "limited_participation"
    KEYWORD = "keyword"
    CONTEXT = "context"


class SourceLocationError(str, Enum):
    UNKNOWN_BLOCK = "unknown_block"
    QUOTE_NOT_IN_BLOCK = "quote_not_in_block"
    AMBIGUOUS_QUOTE = "ambiguous_quote"
    OCCURRENCE_OUT_OF_RANGE = "occurrence_out_of_range"


class ClaimSupportStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    CONTEXT_ONLY = "context_only"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class SourceBlock(StrictModel):
    block_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_char <= self.start_char:
            raise ValueError("source block end_char must be greater than start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("source block text length does not match its range")
        return self


class AtomicEvidenceRef(StrictModel):
    evidence_ref: str = Field(min_length=1)
    source_block_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    occurrence_index: int | None = Field(default=None, ge=0)


class LocatedAtomicEvidenceRef(AtomicEvidenceRef):
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    raw_quote: str | None = None
    source_valid: bool
    source_error: SourceLocationError | None = None

    @model_validator(mode="after")
    def validate_location(self):
        has_location = self.start_char is not None or self.end_char is not None
        if has_location and (
            self.start_char is None
            or self.end_char is None
            or self.end_char <= self.start_char
        ):
            raise ValueError("located evidence offsets must form a valid range")
        if self.source_valid:
            if not has_location or self.raw_quote is None or self.source_error is not None:
                raise ValueError("valid source evidence requires a clean location")
        elif has_location or self.raw_quote is not None or self.source_error is None:
            raise ValueError("invalid source evidence must fail closed")
        return self


class AtomicEvidenceGroup(StrictModel):
    group_id: str = Field(min_length=1)
    relation: EvidenceRelation
    operator: EvidenceGroupOperator
    evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_refs(self):
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("duplicate evidence ref within an evidence group")
        return self


class AtomicClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    claim: str = Field(min_length=1)
    actor: ClaimActor
    polarity: ClaimPolarity
    roles: list[AtomicClaimRole] = Field(min_length=1)
    evidence_groups: list[AtomicEvidenceGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_members(self):
        if len(self.requirement_ids) != len(set(self.requirement_ids)):
            raise ValueError("duplicate requirement id within an atomic claim")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("duplicate role within an atomic claim")
        group_ids = [group.group_id for group in self.evidence_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate evidence group id within an atomic claim")
        return self


class AtomicRequirementDecision(StrictModel):
    requirement_id: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    claim_operator: EvidenceGroupOperator = EvidenceGroupOperator.ALL_OF
    proposed_level: EvidenceLevel
    risk: str = ""

    @model_validator(mode="after")
    def validate_claims_for_level(self):
        if len(self.claim_ids) != len(set(self.claim_ids)):
            raise ValueError("duplicate claim id within a requirement decision")
        if self.proposed_level != EvidenceLevel.NONE and not self.claim_ids:
            raise ValueError("non-NONE requirement decisions require atomic claims")
        return self


class AtomicEvidenceGraph(StrictModel):
    schema_version: Literal[
        "evidence-grounding-atomic-v1"
    ] = ATOMIC_SCHEMA_VERSION
    resume_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_refs: list[AtomicEvidenceRef] = Field(default_factory=list)
    claims: list[AtomicClaim] = Field(default_factory=list)
    requirements: list[AtomicRequirementDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_integrity(self):
        evidence_ids = [item.evidence_ref for item in self.evidence_refs]
        claim_ids = [item.claim_id for item in self.claims]
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence ref in atomic graph")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("duplicate claim id in atomic graph")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("duplicate requirement id in atomic graph")

        known_evidence = set(evidence_ids)
        known_claims = {item.claim_id: item for item in self.claims}
        known_requirements = set(requirement_ids)
        used_evidence: set[str] = set()
        used_claims: set[str] = set()
        group_ids: list[str] = []
        for claim in self.claims:
            unknown_requirements = set(claim.requirement_ids) - known_requirements
            if unknown_requirements:
                raise ValueError(
                    "atomic claim references unknown requirements: "
                    + ",".join(sorted(unknown_requirements))
                )
            for group in claim.evidence_groups:
                group_ids.append(group.group_id)
                unknown_evidence = set(group.evidence_refs) - known_evidence
                if unknown_evidence:
                    raise ValueError(
                        "atomic claim references unknown evidence: "
                        + ",".join(sorted(unknown_evidence))
                    )
                used_evidence.update(group.evidence_refs)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate evidence group id in atomic graph")

        for decision in self.requirements:
            unknown_claims = set(decision.claim_ids) - set(known_claims)
            if unknown_claims:
                raise ValueError(
                    "requirement decision references unknown claims: "
                    + ",".join(sorted(unknown_claims))
                )
            for claim_id in decision.claim_ids:
                if decision.requirement_id not in known_claims[claim_id].requirement_ids:
                    raise ValueError(
                        "requirement decision and atomic claim are not linked"
                    )
            used_claims.update(decision.claim_ids)
        if used_evidence != known_evidence:
            raise ValueError("atomic graph contains unused evidence refs")
        if used_claims != set(known_claims):
            raise ValueError("atomic graph contains unused claims")
        decisions_by_requirement = {
            item.requirement_id: set(item.claim_ids) for item in self.requirements
        }
        for claim in self.claims:
            for requirement_id in claim.requirement_ids:
                if claim.claim_id not in decisions_by_requirement[requirement_id]:
                    raise ValueError(
                        "atomic claim and requirement decision are not linked"
                    )
        return self


class ValidatedAtomicClaim(StrictModel):
    claim_id: str
    support_status: ClaimSupportStatus
    satisfied_group_ids: list[str]
    valid_evidence_refs: list[str]
    invalid_evidence_refs: list[str]


class ValidatedAtomicRequirementDecision(StrictModel):
    requirement_id: str
    proposed_level: EvidenceLevel
    level_cap: EvidenceLevel
    final_level: EvidenceLevel
    verified_claim_ids: list[str]
    verified_roles: list[AtomicClaimRole]
    reason_codes: list[str]


class ValidatedAtomicEvidenceGraph(StrictModel):
    schema_version: Literal[
        "evidence-grounding-atomic-validated-v1"
    ] = "evidence-grounding-atomic-validated-v1"
    resume_sha256: str
    source_block_count: int = Field(ge=0)
    located_evidence_refs: list[LocatedAtomicEvidenceRef]
    claims: list[ValidatedAtomicClaim]
    requirements: list[ValidatedAtomicRequirementDecision]


_LEVEL_ORDER = {
    EvidenceLevel.NONE: -1,
    EvidenceLevel.C0: 0,
    EvidenceLevel.C1: 1,
    EvidenceLevel.C2: 2,
    EvidenceLevel.C3: 3,
}
_ROLE_ORDER = {role: index for index, role in enumerate(AtomicClaimRole)}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_source_blocks(resume_text: str) -> list[SourceBlock]:
    blocks: list[SourceBlock] = []
    offset = 0
    for raw_line in resume_text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        start = offset
        end = start + len(content)
        if content.strip():
            blocks.append(
                SourceBlock(
                    block_id=f"resume_block_{len(blocks) + 1:03d}",
                    start_char=start,
                    end_char=end,
                    text=content,
                )
            )
        offset += len(raw_line)
    return blocks


def _quote_occurrences(text: str, quote: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = text.find(quote, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def locate_atomic_evidence_refs(
    resume_text: str,
    evidence_refs: list[AtomicEvidenceRef],
    *,
    expected_resume_sha256: str,
) -> tuple[list[SourceBlock], list[LocatedAtomicEvidenceRef]]:
    if sha256_text(resume_text) != expected_resume_sha256:
        raise ValueError("atomic evidence resume hash mismatch")
    blocks = build_source_blocks(resume_text)
    by_id = {block.block_id: block for block in blocks}
    located: list[LocatedAtomicEvidenceRef] = []
    for evidence in evidence_refs:
        block = by_id.get(evidence.source_block_id)
        if block is None:
            located.append(
                LocatedAtomicEvidenceRef(
                    **evidence.model_dump(),
                    source_valid=False,
                    source_error=SourceLocationError.UNKNOWN_BLOCK,
                )
            )
            continue
        positions = _quote_occurrences(block.text, evidence.quote)
        if not positions:
            located.append(
                LocatedAtomicEvidenceRef(
                    **evidence.model_dump(),
                    source_valid=False,
                    source_error=SourceLocationError.QUOTE_NOT_IN_BLOCK,
                )
            )
            continue
        if len(positions) > 1 and evidence.occurrence_index is None:
            located.append(
                LocatedAtomicEvidenceRef(
                    **evidence.model_dump(),
                    source_valid=False,
                    source_error=SourceLocationError.AMBIGUOUS_QUOTE,
                )
            )
            continue
        occurrence = evidence.occurrence_index or 0
        if occurrence >= len(positions):
            located.append(
                LocatedAtomicEvidenceRef(
                    **evidence.model_dump(),
                    source_valid=False,
                    source_error=SourceLocationError.OCCURRENCE_OUT_OF_RANGE,
                )
            )
            continue
        start = block.start_char + positions[occurrence]
        end = start + len(evidence.quote)
        raw_quote = resume_text[start:end]
        located.append(
            LocatedAtomicEvidenceRef(
                **evidence.model_dump(),
                start_char=start,
                end_char=end,
                raw_quote=raw_quote,
                source_valid=True,
            )
        )
    return blocks, located


def _group_satisfied(
    group: AtomicEvidenceGroup,
    located_by_id: dict[str, LocatedAtomicEvidenceRef],
) -> bool:
    valid = [
        located_by_id[evidence_ref].source_valid
        for evidence_ref in group.evidence_refs
    ]
    if group.operator == EvidenceGroupOperator.ALL_OF:
        return all(valid)
    return any(valid)


def _validate_claim(
    claim: AtomicClaim,
    located_by_id: dict[str, LocatedAtomicEvidenceRef],
) -> ValidatedAtomicClaim:
    satisfied: list[AtomicEvidenceGroup] = [
        group
        for group in claim.evidence_groups
        if _group_satisfied(group, located_by_id)
    ]
    has_support = any(
        group.relation == EvidenceRelation.SUPPORTS for group in satisfied
    )
    has_contradiction = any(
        group.relation == EvidenceRelation.CONTRADICTS for group in satisfied
    )
    has_context = any(
        group.relation == EvidenceRelation.CONTEXT_ONLY for group in satisfied
    )
    if has_support and has_contradiction:
        status = ClaimSupportStatus.CONFLICT
    elif has_support:
        status = ClaimSupportStatus.SUPPORTED
    elif has_contradiction:
        status = ClaimSupportStatus.CONTRADICTED
    elif has_context:
        status = ClaimSupportStatus.CONTEXT_ONLY
    else:
        status = ClaimSupportStatus.UNSUPPORTED
    all_refs = {
        evidence_ref
        for group in claim.evidence_groups
        for evidence_ref in group.evidence_refs
    }
    valid_refs = sorted(
        evidence_ref
        for evidence_ref in all_refs
        if located_by_id[evidence_ref].source_valid
    )
    return ValidatedAtomicClaim(
        claim_id=claim.claim_id,
        support_status=status,
        satisfied_group_ids=sorted(group.group_id for group in satisfied),
        valid_evidence_refs=valid_refs,
        invalid_evidence_refs=sorted(all_refs - set(valid_refs)),
    )


def _cap_for_candidate_roles(
    roles: set[AtomicClaimRole],
) -> EvidenceLevel:
    if {
        AtomicClaimRole.ACTION,
        AtomicClaimRole.ARTIFACT,
        AtomicClaimRole.RESULT,
    } <= roles:
        return EvidenceLevel.C3
    if {
        AtomicClaimRole.ACTION,
        AtomicClaimRole.ARTIFACT,
    } <= roles or {
        AtomicClaimRole.REPRODUCTION,
        AtomicClaimRole.ARTIFACT,
    } <= roles:
        return EvidenceLevel.C2
    if roles & {
        AtomicClaimRole.ACTION,
        AtomicClaimRole.LEARNING,
        AtomicClaimRole.REPRODUCTION,
        AtomicClaimRole.LIMITED_PARTICIPATION,
    }:
        return EvidenceLevel.C1
    return EvidenceLevel.C0 if roles else EvidenceLevel.NONE


def _cap_for_claim(claim: AtomicClaim) -> EvidenceLevel:
    roles = set(claim.roles)
    if claim.polarity != ClaimPolarity.POSITIVE:
        return EvidenceLevel.NONE
    if claim.actor == ClaimActor.CANDIDATE:
        return _cap_for_candidate_roles(roles)
    if claim.actor == ClaimActor.TEAM and roles & {
        AtomicClaimRole.ACTION,
        AtomicClaimRole.REPRODUCTION,
        AtomicClaimRole.LIMITED_PARTICIPATION,
    }:
        return EvidenceLevel.C1
    return EvidenceLevel.C0 if roles else EvidenceLevel.NONE


def _validate_requirement(
    decision: AtomicRequirementDecision,
    claims_by_id: dict[str, AtomicClaim],
    validated_claims: dict[str, ValidatedAtomicClaim],
) -> ValidatedAtomicRequirementDecision:
    claims = [claims_by_id[claim_id] for claim_id in decision.claim_ids]
    positive_supported = [
        claim
        for claim in claims
        if validated_claims[claim.claim_id].support_status
        == ClaimSupportStatus.SUPPORTED
        and claim.polarity == ClaimPolarity.POSITIVE
    ]
    negative_or_conflicted = [
        claim
        for claim in claims
        if (
            claim.polarity == ClaimPolarity.NEGATIVE
            and validated_claims[claim.claim_id].support_status
            == ClaimSupportStatus.SUPPORTED
        )
        or validated_claims[claim.claim_id].support_status
        in {ClaimSupportStatus.CONTRADICTED, ClaimSupportStatus.CONFLICT}
    ]
    context_only = any(
        validated_claims[claim.claim_id].support_status
        == ClaimSupportStatus.CONTEXT_ONLY
        for claim in claims
    )
    reasons: list[str] = []
    verified_roles: set[AtomicClaimRole] = set()
    if decision.claim_operator == EvidenceGroupOperator.ALL_OF:
        joint_support_valid = bool(claims) and len(positive_supported) == len(claims)
        if joint_support_valid:
            reasons.append("joint_support_valid")
        elif positive_supported:
            reasons.append("joint_support_incomplete")
        for claim in positive_supported:
            verified_roles.update(claim.roles)
        candidate_roles = {
            role
            for claim in positive_supported
            if claim.actor == ClaimActor.CANDIDATE
            for role in claim.roles
        }
        candidate_cap = _cap_for_candidate_roles(candidate_roles)
        team_cap = max(
            (
                _cap_for_claim(claim)
                for claim in positive_supported
                if claim.actor == ClaimActor.TEAM
            ),
            key=_LEVEL_ORDER.get,
            default=EvidenceLevel.NONE,
        )
        level_cap = (
            max((candidate_cap, team_cap), key=_LEVEL_ORDER.get)
            if joint_support_valid
            else EvidenceLevel.NONE
        )
    else:
        reasons.append(
            "alternative_support_valid"
            if positive_supported
            else "alternative_support_missing"
        )
        caps = [
            (_cap_for_claim(claim), claim)
            for claim in positive_supported
        ]
        if caps:
            level_cap, best_claim = max(caps, key=lambda item: _LEVEL_ORDER[item[0]])
            verified_roles.update(best_claim.roles)
        else:
            level_cap = EvidenceLevel.NONE

    if positive_supported and negative_or_conflicted:
        level_cap = EvidenceLevel.C0
        reasons.append("positive_negative_conflict")
    elif not positive_supported and negative_or_conflicted:
        level_cap = EvidenceLevel.NONE
        reasons.append("contradiction_only")
    elif not positive_supported and context_only:
        level_cap = EvidenceLevel.C0
        reasons.append("context_only")
    elif not positive_supported:
        level_cap = EvidenceLevel.NONE
        reasons.append("missing_valid_support")

    final_level = (
        decision.proposed_level
        if _LEVEL_ORDER[decision.proposed_level] <= _LEVEL_ORDER[level_cap]
        else level_cap
    )
    if final_level != decision.proposed_level:
        reasons.append("deterministic_level_cap")
    return ValidatedAtomicRequirementDecision(
        requirement_id=decision.requirement_id,
        proposed_level=decision.proposed_level,
        level_cap=level_cap,
        final_level=final_level,
        verified_claim_ids=sorted(claim.claim_id for claim in positive_supported),
        verified_roles=sorted(verified_roles, key=_ROLE_ORDER.get),
        reason_codes=reasons,
    )


def validate_atomic_evidence_graph(
    *,
    resume_text: str,
    prediction: AtomicEvidenceGraph,
    expected_requirement_ids: set[str],
) -> ValidatedAtomicEvidenceGraph:
    actual_ids = [item.requirement_id for item in prediction.requirements]
    if set(actual_ids) != expected_requirement_ids or len(actual_ids) != len(
        expected_requirement_ids
    ):
        raise ValueError("atomic requirement coverage mismatch")
    blocks, located = locate_atomic_evidence_refs(
        resume_text,
        prediction.evidence_refs,
        expected_resume_sha256=prediction.resume_sha256,
    )
    located_by_id = {item.evidence_ref: item for item in located}
    claims_by_id = {item.claim_id: item for item in prediction.claims}
    validated_claim_list = [
        _validate_claim(claim, located_by_id) for claim in prediction.claims
    ]
    validated_claims = {
        item.claim_id: item for item in validated_claim_list
    }
    requirements = [
        _validate_requirement(
            decision,
            claims_by_id,
            validated_claims,
        )
        for decision in prediction.requirements
    ]
    return ValidatedAtomicEvidenceGraph(
        resume_sha256=prediction.resume_sha256,
        source_block_count=len(blocks),
        located_evidence_refs=located,
        claims=validated_claim_list,
        requirements=requirements,
    )


def load_atomic_evidence_graph(path: Path | str) -> AtomicEvidenceGraph:
    return AtomicEvidenceGraph.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
