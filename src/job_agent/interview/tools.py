from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from pydantic import Field

from job_agent.agent_runtime.contracts import ToolContext, ToolEffect
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.interview.contracts import InterviewContextSnapshot
from job_agent.interview.question_bank import QuestionAnchorLibrary
from job_agent.schemas import StrictModel


READ_BRIEF_TOOL = "interview.read_brief"
SEARCH_RESUME_TOOL = "interview.search_resume_claims"
SEARCH_EVIDENCE_TOOL = "interview.search_jd_evidence"
SEARCH_DOSSIER_TOOL = "interview.search_project_dossier"
SEARCH_ANCHORS_TOOL = "interview.search_question_anchors"
READ_SOURCE_TOOL = "interview.read_project_source"
INTERVIEW_READ_TOOLS = (
    READ_BRIEF_TOOL,
    SEARCH_RESUME_TOOL,
    SEARCH_EVIDENCE_TOOL,
    SEARCH_DOSSIER_TOOL,
    SEARCH_ANCHORS_TOOL,
    READ_SOURCE_TOOL,
)


class EmptyInput(StrictModel):
    pass


class SearchInput(StrictModel):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)


class AnchorSearchInput(StrictModel):
    topics: list[str] = Field(min_length=1, max_length=8)
    roles: list[str] = Field(default_factory=list, max_length=8)
    difficulty: str | None = None
    limit: int = Field(default=5, ge=1, le=5)


class SourceExcerptInput(StrictModel):
    relative_path: str = Field(min_length=1, max_length=300)
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=120, ge=1)


class ReadOutput(StrictModel):
    items: list[dict] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


def _query_terms(value: str) -> set[str]:
    return {term.casefold() for term in re.findall(r"[\w+#.-]{2,}", value, flags=re.UNICODE)}


def _matches(value: str, terms: set[str]) -> bool:
    lower = value.casefold()
    return not terms or any(term in lower for term in terms)


def _resume_items(snapshot: InterviewContextSnapshot, query: str, limit: int) -> ReadOutput:
    terms = _query_terms(query)
    items: list[dict] = []
    refs: list[str] = []
    for line_number, line in enumerate((snapshot.resume_text or "").splitlines(), start=1):
        text = line.strip().lstrip("-* ")
        if text and _matches(text, terms):
            ref = f"resume:targeted#L{line_number}"
            items.append({"text": text[:800], "line": line_number, "ref": ref})
            refs.append(ref)
            if len(items) >= limit:
                break
    return ReadOutput(items=items, evidence_refs=refs)


def _evidence_items(snapshot: InterviewContextSnapshot, query: str, limit: int) -> ReadOutput:
    terms = _query_terms(query)
    items: list[dict] = []
    refs: list[str] = []
    for index, item in enumerate(snapshot.jd_evidence):
        serialized = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if _matches(serialized, terms):
            identity = str(item.get("atom_id") or item.get("evidence_id") or item.get("requirement_id") or index)
            ref = f"jd_evidence:{identity}"
            items.append({"evidence": item, "ref": ref})
            refs.append(ref)
            if len(items) >= limit:
                break
    return ReadOutput(items=items, evidence_refs=refs)


def _dossier_items(snapshot: InterviewContextSnapshot, query: str, limit: int) -> ReadOutput:
    if snapshot.project_dossier is None:
        return ReadOutput()
    terms = _query_terms(query)
    items: list[dict] = []
    refs: list[str] = []
    for fact in snapshot.project_dossier.facts:
        if fact.status not in {"verified", "confirmed"} or not _matches(fact.statement, terms):
            continue
        ref = f"project_fact:{fact.fact_id}"
        items.append(
            {
                "fact_id": fact.fact_id,
                "kind": fact.kind,
                "statement": fact.statement,
                "evidence_refs": fact.evidence_refs,
                "ref": ref,
            }
        )
        refs.append(ref)
        if len(items) >= limit:
            break
    return ReadOutput(items=items, evidence_refs=refs)


def resolve_project_source(repo_root: Path, relative_path: str) -> Path:
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("project source path escapes the repository")
    if not normalized.parts:
        raise ValueError("project source path is empty")
    allowed = normalized.parts[0] in {"src", "tests"} or normalized.as_posix() == "pyproject.toml"
    if not allowed:
        raise ValueError("project source path is outside the allowlist")
    denied_names = {"glm.txt", ".env", ".git"}
    if any(part.casefold() in denied_names or "secret" in part.casefold() for part in normalized.parts):
        raise ValueError("project source path is sensitive")
    root = Path(repo_root).resolve()
    candidate = (root / Path(*normalized.parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("project source symlink escapes the repository")
    if not candidate.is_file():
        raise ValueError("project source does not exist")
    return candidate


def build_interview_registry(
    snapshot: InterviewContextSnapshot,
    *,
    question_library: QuestionAnchorLibrary,
    repo_root: Path,
    run_seed: str,
) -> ToolRegistry:
    registry = ToolRegistry()

    def read_brief(_: EmptyInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        items = [{"kind": snapshot.kind, "job_id": snapshot.job_id, "source_refs": [ref.ref_id for ref in snapshot.source_refs]}]
        if snapshot.jd:
            items.append({"jd": snapshot.jd})
        return ReadOutput(items=items, evidence_refs=[ref.ref_id for ref in snapshot.source_refs if ref.kind == "jd"])

    def search_resume(value: SearchInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        return _resume_items(snapshot, value.query, value.limit)

    def search_evidence(value: SearchInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        return _evidence_items(snapshot, value.query, value.limit)

    def search_dossier(value: SearchInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        return _dossier_items(snapshot, value.query, value.limit)

    def search_anchors(value: AnchorSearchInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        anchors = question_library.search(
            topics=set(value.topics),
            roles=set(value.roles),
            difficulty=value.difficulty,
            seed=run_seed,
            limit=value.limit,
        )
        return ReadOutput(
            items=[anchor.public_view() for anchor in anchors],
            evidence_refs=[f"question_anchor:{anchor.id}" for anchor in anchors],
        )

    def read_source(value: SourceExcerptInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        if value.end_line < value.start_line or value.end_line - value.start_line > 200:
            raise ValueError("project source excerpt range is invalid")
        path = resolve_project_source(repo_root, value.relative_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError("project source is not UTF-8 text") from exc
        excerpt = "\n".join(lines[value.start_line - 1 : value.end_line])
        ref = f"code:{PurePosixPath(value.relative_path).as_posix()}#L{value.start_line}-L{min(value.end_line, len(lines))}"
        return ReadOutput(items=[{"excerpt": excerpt[:12000], "ref": ref}], evidence_refs=[ref])

    registrations = (
        (READ_BRIEF_TOOL, "Read the frozen interview brief and stable JD source refs.", EmptyInput, read_brief),
        (SEARCH_RESUME_TOOL, "Search only the frozen job-linked resume snapshot.", SearchInput, search_resume),
        (SEARCH_EVIDENCE_TOOL, "Search only the frozen JD-to-resume evidence mapping.", SearchInput, search_evidence),
        (SEARCH_DOSSIER_TOOL, "Search verified or user-confirmed project dossier facts.", SearchInput, search_dossier),
        (SEARCH_ANCHORS_TOOL, "Retrieve role- and topic-filtered question anchors; anchors are optional aids.", AnchorSearchInput, search_anchors),
        (READ_SOURCE_TOOL, "Read a bounded excerpt from an allowlisted project source file.", SourceExcerptInput, read_source),
    )
    for name, description, input_model, handler in registrations:
        registry.register(
            name=name,
            description=description,
            input_model=input_model,
            output_model=ReadOutput,
            handler=handler,
            effect=ToolEffect.READ_ONLY,
        )
    return registry
