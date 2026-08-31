from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_PROMPT_ROOT = Path(__file__).parent
_TASK_MARKER = "<!-- TASK -->"
_REPAIR_MARKER = "<!-- REPAIR -->"


@dataclass(frozen=True)
class PromptDefinition:
    version: str
    relative_path: str
    language: str = "zh-CN"


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    version: str
    language: str
    path: Path
    content: str
    content_sha256: str
    task_instruction: str
    repair_instruction: str

    @property
    def identity(self) -> str:
        return f"{self.prompt_id}-{self.version}-{self.content_sha256[:12]}"


_DEFINITIONS = {
    "common.structured-output": PromptDefinition(
        "v1", "common/structured_output_zh.md"
    ),
    "common.evidence-boundary": PromptDefinition(
        "v1", "common/evidence_boundary_zh.md"
    ),
    "common.tool-use-contract": PromptDefinition(
        "v1", "common/tool_use_contract_zh.md"
    ),
    "semantic.jd-structurer": PromptDefinition(
        "v2", "semantic/jd_structurer_v2_zh.md"
    ),
    "semantic.evidence-mapping": PromptDefinition(
        "v2", "semantic/evidence_mapping_v2_zh.md"
    ),
    "semantic.resume-tailoring": PromptDefinition(
        "v3", "semantic/resume_tailoring_v3_zh.md"
    ),
    "semantic.interview-prep": PromptDefinition(
        "v3", "semantic/interview_prep_v3_zh.md"
    ),
    "semantic.answer-cards": PromptDefinition(
        "v3", "semantic/answer_cards_v3_zh.md"
    ),
    "semantic.post-interview-review": PromptDefinition(
        "v2", "semantic/post_interview_review_v2_zh.md"
    ),
    "semantic.mock-interview-evaluation": PromptDefinition(
        "v1", "semantic/mock_interview_evaluation_v1_zh.md"
    ),
    "domain.opportunity-research": PromptDefinition(
        "v2", "domain/opportunity_research_v2_zh.md"
    ),
    "domain.application-material": PromptDefinition(
        "v2", "domain/application_material_v2_zh.md"
    ),
    "domain.interview-coach": PromptDefinition(
        "v2", "domain/interview_coach_v2_zh.md"
    ),
    "domain.application-ops": PromptDefinition(
        "v2", "domain/application_ops_v2_zh.md"
    ),
}


def _sections(content: str) -> tuple[str, str]:
    if _TASK_MARKER not in content:
        return content.strip(), ""
    _, remainder = content.split(_TASK_MARKER, 1)
    if _REPAIR_MARKER not in remainder:
        return remainder.strip(), ""
    task, repair = remainder.split(_REPAIR_MARKER, 1)
    return task.strip(), repair.strip()


@lru_cache(maxsize=None)
def get_prompt(prompt_id: str) -> PromptSpec:
    try:
        definition = _DEFINITIONS[prompt_id]
    except KeyError as exc:
        raise KeyError(f"unknown prompt_id: {prompt_id}") from exc
    path = (_PROMPT_ROOT / definition.relative_path).resolve()
    if _PROMPT_ROOT.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(f"prompt asset missing: {definition.relative_path}")
    content = path.read_text(encoding="utf-8").strip()
    task, repair = _sections(content)
    return PromptSpec(
        prompt_id=prompt_id,
        version=definition.version,
        language=definition.language,
        path=path,
        content=content,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        task_instruction=task,
        repair_instruction=repair,
    )


def list_prompts() -> tuple[PromptSpec, ...]:
    return tuple(get_prompt(prompt_id) for prompt_id in sorted(_DEFINITIONS))
