import hashlib
import json
from pathlib import Path

from job_agent.evidence.evaluation.live_runner import load_corpus
from job_agent.evidence.evaluation.runner import load_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval" / "evidence_v2"


def test_v5_corpus_is_new_family_disjoint_and_frozen():
    audit = json.loads((DATA / "corpus_100_audit_v5.json").read_text(encoding="utf-8"))
    dev = set(audit["splits"]["development"]["family_ids"])
    held = set(audit["splits"]["held_out"]["family_ids"])
    assert len(dev) == 12
    assert len(held) == 8
    assert dev.isdisjoint(held)
    assert audit["v4_held_out_used_for_tuning"] is False
    assert audit["splits"]["held_out"]["frozen_before_live_run"] is True


def test_v5_hashes_contracts_and_artifact_metadata_are_valid():
    audit = json.loads((DATA / "corpus_100_audit_v5.json").read_text(encoding="utf-8"))
    observed_types: set[str] = set()
    total = 0
    for split, short in (("development", "dev"), ("held_out", "held_out")):
        corpus_path = DATA / f"cases_{short}_v5.json"
        manifest_path = DATA / f"manifest_{short}_v5.json"
        assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == audit["splits"][split]["corpus_sha256"]
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == audit["splits"][split]["manifest_sha256"]
        manifest = load_manifest(manifest_path)
        corpus = load_corpus(corpus_path, manifest)
        total += len(corpus.cases)
        for case in corpus.cases:
            observed_types.update(
                source.artifact_type for source in case.evidence_sources if source.artifact_type
            )
    assert total == 100
    assert {"source_code", "readme", "design_record", "comparison_record", "metric_record", "production_log", "incident_report"} <= observed_types
