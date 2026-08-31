from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from job_agent.evaluation.evidence_grounding_contracts import (
    EntailmentRelation,
    EvidenceGroundingDataset,
    EvidenceSpan,
    load_jsonl_dataset,
    sha256_file,
)
from job_agent.schemas import EvidenceLevel, StrictModel


class HumanReviewQueueItem(StrictModel):
    case_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    resume_text: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    slices: list[str] = Field(min_length=1)


class HumanReviewQueue(StrictModel):
    schema_version: Literal[
        "evidence-grounding-human-review-queue-v1"
    ] = "evidence-grounding-human-review-queue-v1"
    base_file: str
    base_sha256: str = Field(min_length=64, max_length=64)
    annotation_guide_version: str = Field(min_length=1)
    review_mode: Literal["blind_first"] = "blind_first"
    total_units: int = Field(gt=0)
    already_reviewed_units: int = Field(ge=0)
    pending_units: int = Field(ge=0)
    items: list[HumanReviewQueueItem]

    @model_validator(mode="after")
    def validate_counts_and_unique_units(self):
        if self.already_reviewed_units + self.pending_units != self.total_units:
            raise ValueError("human review queue counts do not add up")
        if len(self.items) != self.pending_units:
            raise ValueError("pending item count mismatch")
        keys = [(item.case_id, item.requirement_id) for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate unit in human review queue")
        return self


class HumanReviewDecisionItem(StrictModel):
    case_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    status: Literal["pending", "completed"] = "pending"
    human_level: EvidenceLevel | None = None
    support_relation: EntailmentRelation | None = None
    support_spans: list[EvidenceSpan] = Field(default_factory=list)
    max_safe_claim: str | None = None
    risk_note: str | None = None
    confidence: Literal["low", "medium", "high"] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_completed_fields(self):
        if self.status == "pending":
            return self
        required_text = {
            "max_safe_claim": self.max_safe_claim,
            "risk_note": self.risk_note,
            "reason": self.reason,
        }
        missing = [
            field
            for field, value in required_text.items()
            if value is None or not value.strip()
        ]
        if self.human_level is None:
            missing.append("human_level")
        if self.support_relation is None:
            missing.append("support_relation")
        if self.confidence is None:
            missing.append("confidence")
        if missing:
            raise ValueError(
                "completed human review is missing fields: "
                + ",".join(sorted(missing))
            )
        if self.human_level == EvidenceLevel.NONE and self.support_spans:
            raise ValueError("NONE human review cannot contain supporting spans")
        if self.human_level != EvidenceLevel.NONE and not self.support_spans:
            raise ValueError(
                "non-NONE human review requires at least one supporting span"
            )
        return self


class HumanReviewSubmission(StrictModel):
    schema_version: Literal[
        "evidence-grounding-human-review-submission-v1"
    ] = "evidence-grounding-human-review-submission-v1"
    base_file: str
    base_sha256: str = Field(min_length=64, max_length=64)
    annotation_guide_version: str = Field(min_length=1)
    review_mode: Literal["blind_first"] = "blind_first"
    items: list[HumanReviewDecisionItem]

    @model_validator(mode="after")
    def validate_unique_units(self):
        keys = [(item.case_id, item.requirement_id) for item in self.items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate unit in human review submission")
        return self

    @property
    def completed_units(self) -> int:
        return sum(item.status == "completed" for item in self.items)

    @property
    def pending_units(self) -> int:
        return len(self.items) - self.completed_units


def _reviewed_keys(
    payload: dict[str, Any],
    *,
    expected_base_hash: str,
) -> set[tuple[str, str]]:
    if payload.get("base_sha256") != expected_base_hash:
        raise ValueError("human adjudication base hash mismatch")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("human adjudication decisions must be a list")
    keys: list[tuple[str, str]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("human adjudication decision must be an object")
        case_id = decision.get("case_id")
        requirement_id = decision.get("requirement_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("human adjudication case_id is required")
        if not isinstance(requirement_id, str) or not requirement_id:
            raise ValueError("human adjudication requirement_id is required")
        keys.append((case_id, requirement_id))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate human adjudication decision")
    return set(keys)


def _dataset_unit_keys(
    dataset: EvidenceGroundingDataset,
) -> set[tuple[str, str]]:
    return {
        (case.case_id, unit.requirement_id)
        for case in dataset.cases
        for unit in case.requirements
    }


def build_human_review_queue(
    *,
    base_path: Path | str,
    adjudication_path: Path | str,
    annotation_guide_version: str = "v1.1",
) -> HumanReviewQueue:
    base = Path(base_path)
    base_hash = sha256_file(base)
    dataset = load_jsonl_dataset(base)
    adjudication = json.loads(
        Path(adjudication_path).read_text(encoding="utf-8")
    )
    if not isinstance(adjudication, dict):
        raise ValueError("human adjudication payload must be an object")
    reviewed = _reviewed_keys(
        adjudication,
        expected_base_hash=base_hash,
    )
    known = _dataset_unit_keys(dataset)
    unknown = reviewed - known
    if unknown:
        rendered = ",".join(
            f"{case_id}/{requirement_id}"
            for case_id, requirement_id in sorted(unknown)
        )
        raise ValueError(f"human adjudication references unknown units: {rendered}")

    items = [
        HumanReviewQueueItem(
            case_id=case.case_id,
            requirement_id=unit.requirement_id,
            requirement=unit.requirement,
            resume_text=case.resume_text,
            source_group=case.source_group,
            source_kind=case.source_kind,
            slices=case.slices,
        )
        for case in dataset.cases
        for unit in case.requirements
        if (case.case_id, unit.requirement_id) not in reviewed
    ]
    total = len(known)
    return HumanReviewQueue(
        base_file=base.name,
        base_sha256=base_hash,
        annotation_guide_version=annotation_guide_version,
        total_units=total,
        already_reviewed_units=len(reviewed),
        pending_units=len(items),
        items=items,
    )


def build_human_review_submission_template(
    queue: HumanReviewQueue,
) -> HumanReviewSubmission:
    return HumanReviewSubmission(
        base_file=queue.base_file,
        base_sha256=queue.base_sha256,
        annotation_guide_version=queue.annotation_guide_version,
        items=[
            HumanReviewDecisionItem(
                case_id=item.case_id,
                requirement_id=item.requirement_id,
            )
            for item in queue.items
        ],
    )


def load_human_review_submission(
    path: Path | str,
) -> HumanReviewSubmission:
    return HumanReviewSubmission.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def validate_human_review_submission(
    *,
    queue: HumanReviewQueue,
    submission: HumanReviewSubmission,
    require_complete: bool = False,
) -> None:
    if submission.base_file != queue.base_file:
        raise ValueError("human review submission base file mismatch")
    if submission.base_sha256 != queue.base_sha256:
        raise ValueError("human review submission base hash mismatch")
    if submission.annotation_guide_version != queue.annotation_guide_version:
        raise ValueError("human review submission annotation guide mismatch")
    expected = {
        (item.case_id, item.requirement_id): item for item in queue.items
    }
    actual = {
        (item.case_id, item.requirement_id): item for item in submission.items
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise ValueError(
            "human review submission unit mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if require_complete and submission.pending_units:
        raise ValueError(
            "human review submission is incomplete: "
            f"pending={submission.pending_units}"
        )
    for key, decision in actual.items():
        if decision.status != "completed":
            continue
        resume_text = expected[key].resume_text
        for span in decision.support_spans:
            if span.end_char > len(resume_text):
                raise ValueError(
                    "human review span is outside resume text: "
                    f"{key[0]}/{key[1]}"
                )
            if resume_text[span.start_char : span.end_char] != span.quote:
                raise ValueError(
                    "human review span offset/quote mismatch: "
                    f"{key[0]}/{key[1]}"
                )


def render_human_review_markdown(queue: HumanReviewQueue) -> str:
    lines = [
        "# Evidence Grounding：剩余人工盲审队列",
        "",
        f"- Base SHA-256：`{queue.base_sha256}`",
        f"- Annotation guide：`{queue.annotation_guide_version}`",
        f"- 已复核：`{queue.already_reviewed_units}`",
        f"- 待复核：`{queue.pending_units}`",
        f"- 总计：`{queue.total_units}`",
        "",
        "> 这是 blind-first 表单，不显示原 AI level、span 或 rationale。"
        "请先独立判断，再与旧标签对照。",
        "",
        "每条都要填写：level、support relation、support quote、max safe claim、"
        "risk note、confidence 和理由。",
        "正式提交请填写同目录的 `review_submission.json`；Markdown 只用于阅读。",
        "",
    ]
    for index, item in enumerate(queue.items, start=1):
        lines.extend(
            [
                f"## {index}. `{item.case_id} / {item.requirement_id}`",
                "",
                f"- Source group：`{item.source_group}`",
                f"- Source kind：`{item.source_kind}`",
                f"- Slices：`{', '.join(item.slices)}`",
                f"- Requirement：{item.requirement}",
                "",
                "### Resume",
                "",
                "```text",
                item.resume_text,
                "```",
                "",
                "### Human decision",
                "",
                "- Level：`TODO`",
                "- Support relation：`TODO`",
                "- Support quote：`TODO`",
                "- Max safe claim：`TODO`",
                "- Risk note：`TODO`",
                "- Confidence：`TODO`",
                "- Reason：`TODO`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
