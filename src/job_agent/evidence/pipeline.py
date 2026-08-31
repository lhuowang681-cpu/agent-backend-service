from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from job_agent.evidence.assessment import FitAssessmentPolicy, initial_fit_policy_v2
from job_agent.evidence.contracts import (
    EvidenceBundle,
    EvidenceRunManifest,
    SourceInput,
)
from job_agent.evidence.mapping import EvidenceMappingService
from job_agent.evidence.projections import EvidenceProjectionService
from job_agent.evidence.repository import EvidenceArtifactRepository
from job_agent.evidence.requirements import RequirementAnalyzer
from job_agent.evidence.sources import EvidenceSourceStore, sha256_bytes


@dataclass(frozen=True)
class EvidenceV2PipelineResult:
    bundle: EvidenceBundle
    resume_view: object
    interview_view: object
    display_view: object


class EvidenceV2Pipeline:
    def __init__(self, *, registry, harness) -> None:
        self.registry = registry
        self.harness = harness

    def run(
        self,
        *,
        session_dir: Path,
        raw_jd: str,
        original_resume: str,
        session_id: str,
        run_id: str | None = None,
        supporting_sources: tuple[SourceInput, ...] = (),
    ) -> EvidenceV2PipelineResult:
        run_id = run_id or f"ev2-{uuid4().hex}"
        source_store = EvidenceSourceStore(session_dir)
        catalog = source_store.snapshot(
            run_id,
            [
                SourceInput(kind="raw_jd", display_label="岗位 JD", text=raw_jd),
                SourceInput(kind="original_resume", display_label="原始简历", text=original_resume),
                *supporting_sources,
            ],
        )
        jd_source = next(item for item in catalog.documents if item.kind == "raw_jd")
        analysis = RequirementAnalyzer(
            registry=self.registry,
            harness=self.harness,
            source_store=source_store,
        ).analyze(jd_source, catalog=catalog, session_id=session_id, run_id=run_id)
        mapping = EvidenceMappingService(
            registry=self.registry,
            harness=self.harness,
            source_store=source_store,
        ).map(analysis, catalog=catalog, session_id=session_id, run_id=run_id)
        config = initial_fit_policy_v2()
        assessment = FitAssessmentPolicy().assess(analysis, mapping, config=config)
        provider = self.harness.provider
        fingerprint = sha256_bytes("".join(item.content_hash for item in catalog.documents).encode("utf-8"))
        bundle = EvidenceBundle(
            manifest=EvidenceRunManifest(
                run_id=run_id,
                created_at=datetime.now(UTC).isoformat(),
                jd_parser_version="evidence-requirements-v2",
                evidence_prompt_version="evidence-mapping-v2",
                fit_policy_version=config.version,
                provider=str(getattr(provider, "provider_name", "unknown")),
                model=str(getattr(provider, "model", "unknown")),
                source_fingerprint=fingerprint,
            ),
            sources=catalog,
            requirements=analysis,
            mapping=mapping,
            assessment=assessment,
        )
        repository = EvidenceArtifactRepository()
        repository.commit(session_dir, bundle)
        committed = repository.load_current(session_dir)
        assert committed is not None
        projections = EvidenceProjectionService()
        return EvidenceV2PipelineResult(
            bundle=committed,
            resume_view=projections.for_resume(committed),
            interview_view=projections.for_interview(committed),
            display_view=projections.for_ui(committed),
        )
