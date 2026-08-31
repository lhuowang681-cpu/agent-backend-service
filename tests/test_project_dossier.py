from pathlib import Path

import pytest

from job_agent.agent_runtime.contracts import AgentAction, AgentFinish
from job_agent.agent_runtime.loop import ScriptedAgentModel
from job_agent.interview.dossier import FactConfirmation, ProjectDossierService, manifest_fingerprint, repository_manifest


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("def save_checkpoint():\n    return 'saved'\n", encoding="utf-8")
    (tmp_path / "tests" / "test_runtime.py").write_text("def test_save():\n    assert True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (tmp_path / "docs" / "learning").mkdir(parents=True)
    (tmp_path / "docs" / "learning" / "README.md").write_text("must not scan", encoding="utf-8")
    return tmp_path


def test_manifest_excludes_learning_and_changes_fingerprint(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = repository_manifest(repo)
    assert all(not path.startswith("docs/learning") for path in first)
    first_hash = manifest_fingerprint(first)
    (repo / "src" / "runtime.py").write_text("def save_checkpoint():\n    return 'changed'\n", encoding="utf-8")
    assert manifest_fingerprint(repository_manifest(repo)) != first_hash


def test_generate_confirm_and_stale_dossier(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    service = ProjectDossierService(tmp_path / "output")
    search = AgentAction(
        action_id="search-1",
        tool_name="project.search_source",
        tool_arguments={"query": "save_checkpoint", "limit": 5},
        expected_observation="source",
        progress_claim="search source",
    )
    result = {
        "project_name": "Agent Harness",
        "facts": [
            {"kind": "implementation", "statement": "The repository implements checkpoint persistence.", "evidence_refs": ["code:src/runtime.py#L1-L1"]},
            {"kind": "ownership", "statement": "I independently designed the checkpoint layer.", "evidence_refs": ["code:src/runtime.py#L1-L1"]}
        ]
    }
    model = ScriptedAgentModel([search, AgentFinish(result=result, completion_evidence=["code:src/runtime.py#L1-L1"], confidence=0.8)])
    draft = service.generate_draft(repo, model=model)
    assert draft.facts[0].status == "verified"
    assert draft.facts[1].status == "needs_confirmation"
    with pytest.raises(ValueError, match="requires confirmation"):
        service.confirm(draft, [])
    confirmed = service.confirm(draft, [FactConfirmation(fact_id="fact-002", decision="confirmed")])
    assert service.load_current().revision == confirmed.revision
    (repo / "src" / "runtime.py").write_text("changed", encoding="utf-8")
    assert all(fact.status == "stale" for fact in service.load_current(repo).facts)


def test_generation_failure_does_not_replace_current(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    service = ProjectDossierService(tmp_path / "output")
    search = AgentAction(action_id="search-1", tool_name="project.search_source", tool_arguments={"query": "save_checkpoint", "limit": 5}, expected_observation="source", progress_claim="search")
    good = {"project_name": "Agent Harness", "facts": [{"kind": "implementation", "statement": "Checkpoint exists.", "evidence_refs": ["code:src/runtime.py#L1-L1"]}]}
    draft = service.generate_draft(repo, model=ScriptedAgentModel([search, AgentFinish(result=good, completion_evidence=["code:src/runtime.py#L1-L1"], confidence=0.8)]))
    service.confirm(draft, [])
    bad = {"project_name": "Agent Harness", "facts": [{"kind": "implementation", "statement": "Invented.", "evidence_refs": ["code:missing.py#L1-L1"]}]}
    with pytest.raises(Exception):
        service.generate_draft(repo, model=ScriptedAgentModel([AgentFinish(result=bad, completion_evidence=["bad"], confidence=0.8), AgentFinish(result=bad, completion_evidence=["bad"], confidence=0.8)]))
    assert service.load_current().facts[0].statement == "Checkpoint exists."
