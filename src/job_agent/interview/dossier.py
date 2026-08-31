from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Sequence

from pydantic import Field, ValidationError

from job_agent.agent_runtime.budgets import AgentBudget, BudgetManager, BudgetProfile
from job_agent.agent_runtime.contracts import (
    AgentFinish,
    AgentRunStatus,
    AgentStepRecord,
    ToolContext,
    ToolEffect,
    ToolObservationStatus,
    VerificationResult,
)
from job_agent.agent_runtime.loop import AgentLoop, AgentModel
from job_agent.agent_runtime.policy import PolicyContext, PolicyEngine
from job_agent.agent_runtime.tool_registry import ToolRegistry
from job_agent.agent_runtime.tool_use_decision_model import ToolUseDecisionModel
from job_agent.interview.contracts import ProjectDossier, ProjectFact, ProjectFactKind
from job_agent.interview.run_store import _atomic_json
from job_agent.interview.tools import ReadOutput, SearchInput, SourceExcerptInput, resolve_project_source
from job_agent.llm.harness import NodePolicy
from job_agent.llm.skill_registry import SkillSpec
from job_agent.schemas import StrictModel


LIST_PROJECT_TOOL = "project.list_manifest"
SEARCH_PROJECT_TOOL = "project.search_source"
READ_PROJECT_TOOL = "project.read_source"
PROJECT_DOSSIER_TOOLS = (LIST_PROJECT_TOOL, SEARCH_PROJECT_TOOL, READ_PROJECT_TOOL)


class ProjectDossierError(RuntimeError):
    pass


class ManifestInput(StrictModel):
    limit: int = Field(default=200, ge=1, le=500)


class DossierFactDraft(StrictModel):
    kind: ProjectFactKind
    statement: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class DossierDraftResult(StrictModel):
    project_name: str = Field(min_length=1)
    facts: list[DossierFactDraft] = Field(min_length=1)


class FactConfirmation(StrictModel):
    fact_id: str = Field(min_length=1)
    decision: str = Field(pattern=r"^(confirmed|rejected)$")
    edited_statement: str | None = None


def repository_manifest(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    candidates: list[Path] = []
    for folder in (root / "src", root / "tests"):
        if folder.is_dir():
            candidates.extend(path for path in folder.rglob("*") if path.is_file())
    if (root / "pyproject.toml").is_file():
        candidates.append(root / "pyproject.toml")
    manifest: dict[str, str] = {}
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        parts = {part.casefold() for part in PurePosixPath(relative).parts}
        if "__pycache__" in parts or path.suffix.casefold() in {".pyc", ".pyo"}:
            continue
        if any(part in {".git", ".env", "glm.txt"} or "secret" in part for part in parts):
            continue
        raw = path.read_bytes()
        if len(raw) > 300_000:
            continue
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        manifest[relative] = hashlib.sha256(raw).hexdigest()
    if not manifest:
        raise ProjectDossierError("repository has no allowlisted UTF-8 sources")
    return manifest


def manifest_fingerprint(manifest: dict[str, str]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_dossier_registry(repo_root: Path, manifest: dict[str, str]) -> ToolRegistry:
    registry = ToolRegistry()

    def list_manifest(value: ManifestInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        items = [{"path": path, "sha256": digest} for path, digest in list(manifest.items())[: value.limit]]
        return ReadOutput(items=items, evidence_refs=[])

    def search_source(value: SearchInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        terms = [term.casefold() for term in value.query.split() if len(term) >= 2]
        items: list[dict] = []
        refs: list[str] = []
        for relative in manifest:
            path = resolve_project_source(repo_root, relative)
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if terms and not any(term in line.casefold() for term in terms):
                    continue
                ref = f"code:{relative}#L{line_number}-L{line_number}"
                items.append({"path": relative, "line": line_number, "text": line.strip()[:800], "ref": ref})
                refs.append(ref)
                if len(items) >= value.limit:
                    return ReadOutput(items=items, evidence_refs=refs)
        return ReadOutput(items=items, evidence_refs=refs)

    def read_source(value: SourceExcerptInput, context: ToolContext) -> ReadOutput:
        context.ensure_active()
        if value.end_line < value.start_line or value.end_line - value.start_line > 200:
            raise ValueError("project source excerpt range is invalid")
        path = resolve_project_source(repo_root, value.relative_path)
        lines = path.read_text(encoding="utf-8").splitlines()
        end = min(value.end_line, len(lines))
        ref = f"code:{PurePosixPath(value.relative_path).as_posix()}#L{value.start_line}-L{end}"
        return ReadOutput(
            items=[{"path": value.relative_path, "excerpt": "\n".join(lines[value.start_line - 1 : end])[:12000], "ref": ref}],
            evidence_refs=[ref],
        )

    registry.register(name=LIST_PROJECT_TOOL, description="List allowlisted repository text files and hashes.", input_model=ManifestInput, output_model=ReadOutput, handler=list_manifest, effect=ToolEffect.READ_ONLY)
    registry.register(name=SEARCH_PROJECT_TOOL, description="Search allowlisted repository sources for implementation evidence.", input_model=SearchInput, output_model=ReadOutput, handler=search_source, effect=ToolEffect.READ_ONLY)
    registry.register(name=READ_PROJECT_TOOL, description="Read a bounded excerpt from an allowlisted source file.", input_model=SourceExcerptInput, output_model=ReadOutput, handler=read_source, effect=ToolEffect.READ_ONLY)
    return registry


class DossierDraftVerifier:
    def verify(self, finish: AgentFinish, *, goal: dict, steps: Sequence[AgentStepRecord]) -> VerificationResult:
        try:
            result = DossierDraftResult.model_validate(finish.result)
        except ValidationError:
            return VerificationResult(passed=False, reason_code="dossier_schema_error", feedback="Submit one valid DossierDraftResult.")
        valid_refs: set[str] = set()
        for step in steps:
            if step.observation.status == ToolObservationStatus.OK:
                refs = step.observation.data.get("evidence_refs", [])
                if isinstance(refs, list):
                    valid_refs.update(str(ref) for ref in refs)
        if not valid_refs:
            return VerificationResult(passed=False, reason_code="dossier_no_source_evidence", feedback="Inspect repository sources before submitting facts.")
        if any(set(fact.evidence_refs) - valid_refs for fact in result.facts):
            return VerificationResult(passed=False, reason_code="dossier_unknown_source_ref", feedback="Every fact must cite a source ref returned by repository tools.")
        if not any(fact.kind == "implementation" for fact in result.facts):
            return VerificationResult(passed=False, reason_code="dossier_missing_implementation", feedback="Include at least one grounded implementation fact.")
        return VerificationResult(passed=True, reason_code="dossier_grounded", feedback="Dossier draft is grounded.", evidence_refs=sorted(valid_refs))


class ProjectDossierService:
    def __init__(self, output_root: Path, *, project_id: str = "agent-harness") -> None:
        self.output_root = Path(output_root).resolve()
        self.project_id = project_id

    @property
    def project_dir(self) -> Path:
        return self.output_root / "projects" / self.project_id

    def generate_draft(
        self,
        repo_root: Path,
        *,
        runtime=None,
        model: AgentModel | None = None,
    ) -> ProjectDossier:
        manifest = repository_manifest(repo_root)
        fingerprint = manifest_fingerprint(manifest)
        registry = _build_dossier_registry(Path(repo_root), manifest)
        if model is None:
            if runtime is None or getattr(runtime, "provider", None) is None:
                raise ProjectDossierError("live runtime is required")
            skill = SkillSpec(
                skill_id="project-dossier-builder",
                version="v1",
                instructions=(
                    "Inspect the allowlisted repository sources and submit a concise project fact draft. Code proves "
                    "implementation, not personal ownership. Mark facts by kind but do not claim employment, production "
                    "usage, ownership, metrics, or design intent without direct evidence. Every fact must cite tool refs."
                ),
                reference_paths=(),
                output_schema=DossierDraftResult,
                allowed_tools=PROJECT_DOSSIER_TOOLS,
                guardrails=("Never request docs/learning, credentials, or files outside the allowlist.", "Never invent source refs."),
            )
            model = ToolUseDecisionModel(
                provider=runtime.provider,
                skill=skill,
                tools=registry.provider_specs(PROJECT_DOSSIER_TOOLS),
                result_schema=DossierDraftResult,
                submit_tool_name="submit_project_dossier_draft",
                session_id=f"project-dossier:{self.project_id}",
                run_id=f"dossier-{fingerprint[:12]}",
                agent_id="project-dossier-builder",
                prompt_version="project-dossier-v1",
                max_runtime_tool_calls=10,
                policy=NodePolicy(timeout_s=90, max_retries=2, temperature=0.0, max_output_tokens=8192, allow_rule_fallback=False),
            )
        loop = AgentLoop(
            agent_id="project-dossier-builder",
            model=model,
            registry=registry,
            policy=PolicyEngine(),
            policy_context=PolicyContext(skill_allowed_tools=PROJECT_DOSSIER_TOOLS, agent_allowed_tools=PROJECT_DOSSIER_TOOLS, runtime_allowed_tools=PROJECT_DOSSIER_TOOLS),
            budget=BudgetManager(AgentBudget.for_profile(BudgetProfile.EVAL, max_model_calls=11, max_tool_calls=11, max_active_seconds=180)),
            verifier=DossierDraftVerifier(),
            max_same_verifier_reason=2,
        )
        result = loop.run(
            session_id=f"project-dossier:{self.project_id}",
            run_id=f"dossier-{fingerprint[:12]}",
            goal={"project_id": self.project_id, "manifest": manifest},
            tool_context=ToolContext(session_id=f"project-dossier:{self.project_id}", run_id=f"dossier-{fingerprint[:12]}", agent_id="project-dossier-builder", workspace_root=str(Path(repo_root).resolve())),
        )
        if result.state.status != AgentRunStatus.COMPLETED:
            raise ProjectDossierError(result.error_code or result.state.status.value)
        draft_result = DossierDraftResult.model_validate(result.result)
        revision = (self.load_current().revision + 1) if self.load_current() else 1
        facts = []
        for index, draft in enumerate(draft_result.facts, start=1):
            source_hashes = {}
            for ref in draft.evidence_refs:
                relative = ref.removeprefix("code:").split("#", 1)[0]
                if relative in manifest:
                    source_hashes[relative] = manifest[relative]
            status = "verified" if draft.kind == "implementation" and source_hashes else "needs_confirmation"
            facts.append(ProjectFact(fact_id=f"fact-{index:03d}", kind=draft.kind, statement=draft.statement, status=status, evidence_refs=draft.evidence_refs, source_hashes=source_hashes))
        dossier = ProjectDossier(project_id=self.project_id, name=draft_result.project_name, source_fingerprint=fingerprint, revision=revision, facts=facts)
        _atomic_json(self.project_dir / "draft.json", dossier.model_dump(mode="json"))
        return dossier

    def confirm(self, dossier: ProjectDossier, decisions: list[FactConfirmation]) -> ProjectDossier:
        decision_map = {item.fact_id: item for item in decisions}
        if len(decision_map) != len(decisions):
            raise ValueError("duplicate dossier fact confirmation")
        facts: list[ProjectFact] = []
        for fact in dossier.facts:
            decision = decision_map.get(fact.fact_id)
            if fact.status == "needs_confirmation" and decision is None:
                raise ValueError(f"fact requires confirmation: {fact.fact_id}")
            if decision is None:
                facts.append(fact)
                continue
            statement = (decision.edited_statement or fact.statement).strip()
            if not statement:
                raise ValueError("confirmed fact statement cannot be empty")
            facts.append(fact.model_copy(update={"statement": statement, "status": decision.decision}))
        confirmed = dossier.model_copy(update={"facts": facts, "confirmed_at": datetime.now(UTC).isoformat()})
        if not confirmed.ready_for_interview:
            raise ValueError("confirmed dossier must retain at least one implementation fact")
        revision_path = self.project_dir / "revisions" / f"{confirmed.revision}.json"
        if revision_path.exists():
            raise FileExistsError(f"dossier revision already exists: {confirmed.revision}")
        _atomic_json(revision_path, confirmed.model_dump(mode="json"))
        _atomic_json(self.project_dir / "current.json", {"revision": confirmed.revision, "path": f"revisions/{confirmed.revision}.json"})
        return confirmed

    def load_current(self, repo_root: Path | None = None) -> ProjectDossier | None:
        pointer = self.project_dir / "current.json"
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text(encoding="utf-8"))
        dossier = ProjectDossier.model_validate_json((self.project_dir / value["path"]).read_text(encoding="utf-8"))
        if repo_root is not None and manifest_fingerprint(repository_manifest(repo_root)) != dossier.source_fingerprint:
            return dossier.model_copy(update={"facts": [fact.model_copy(update={"status": "stale"}) for fact in dossier.facts]})
        return dossier
