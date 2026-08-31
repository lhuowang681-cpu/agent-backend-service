import json

import pytest
from pydantic import ValidationError

from job_agent.evidence.evaluation.contracts import (
    EvidenceEvalManifest,
    LiveEvaluationApproval,
)
from job_agent.evidence.evaluation.runner import (
    default_manifest_path,
    load_manifest,
    require_live_approval,
)


def test_development_manifest_registers_three_fair_variants():
    manifest = load_manifest(default_manifest_path())
    assert {item.variant_id for item in manifest.variants} == {
        "direct_llm_judge",
        "atomized_single_link",
        "full_evidence_v2",
    }
    fairness = {
        (item.corpus_hash, item.provider, item.model, item.temperature, item.max_case_tokens)
        for item in manifest.variants
    }
    assert len(fairness) == 1
    assert manifest.synthetic_only is True
    assert manifest.held_out_frozen is False


def test_manifest_rejects_variant_budget_drift():
    payload = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    payload["variants"][0]["max_case_tokens"] += 1
    with pytest.raises(ValidationError, match="must share corpus"):
        EvidenceEvalManifest.model_validate(payload)


def test_manifest_has_no_evidence_v1_quality_variant():
    payload = default_manifest_path().read_text(encoding="utf-8")
    assert '"variant_id": "evidence_v1"' not in payload


def test_live_evaluation_requires_explicit_approval_and_environment_key(monkeypatch):
    monkeypatch.delenv("JOB_AGENT_LIVE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="explicit approval"):
        require_live_approval(None)
    approval = LiveEvaluationApproval(
        approved=True,
        endpoint="https://example.test",
        model="model",
        case_count=8,
        max_total_tokens=32768,
    )
    with pytest.raises(RuntimeError, match="JOB_AGENT_LIVE_API_KEY"):
        require_live_approval(approval)
    monkeypatch.setenv("JOB_AGENT_LIVE_API_KEY", "secret")
    require_live_approval(approval)
