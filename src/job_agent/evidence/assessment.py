from __future__ import annotations

from job_agent.evidence.contracts import (
    AtomEvidenceResult,
    EvidenceAssessment,
    EvidenceDisplaySummary,
    EvidenceLevel,
    EvidenceMappingV2,
    EvidenceV2Error,
    FitPolicyConfig,
    ParentRequirement,
    RequirementAnalysis,
)


_VERIFIED_LEVELS = {EvidenceLevel.C2, EvidenceLevel.C3}


def initial_fit_policy_v2() -> FitPolicyConfig:
    """Initial development policy; evaluation must freeze a new version before quality claims."""
    return FitPolicyConfig(
        version="fit_policy_v2_initial",
        importance_weights={"must": 1.0, "should": 0.6, "nice": 0.2},
        fit_band_thresholds={"high": 0.70, "medium": 0.40},
        confidence_parameters={
            "C0": 0.0,
            "C1": 0.25,
            "C2": 0.70,
            "C3": 1.0,
            "None": 0.0,
            "high": 0.75,
            "medium": 0.40,
        },
    )


class FitAssessmentPolicy:
    def assess(
        self,
        analysis: RequirementAnalysis,
        mapping: EvidenceMappingV2,
        *,
        config: FitPolicyConfig,
    ) -> EvidenceAssessment:
        atoms = {item.atom_id: item for item in analysis.atoms}
        results = {item.atom_id: item for item in mapping.atom_results}
        if set(atoms) != set(results):
            raise EvidenceV2Error("mapping_atom_coverage_mismatch")
        if not atoms:
            return self._empty_assessment(config)

        estimated_by_atom = {key: self._estimated(results[key]) for key in atoms}
        verified_by_atom = {key: self._verified(results[key]) for key in atoms}
        confidence_by_atom = {key: self._confidence(results[key], config) for key in atoms}
        estimated = self._aggregate(analysis, estimated_by_atom, config)
        verified = self._aggregate(analysis, verified_by_atom, config)
        confidence = self._aggregate(analysis, confidence_by_atom, config)
        band = self._fit_band(estimated, config)

        hard_blockers: list[str] = []
        for parent in analysis.parents:
            if parent.hard_gate:
                hard_blockers.extend(
                    atom_id
                    for atom_id in parent.atom_ids
                    if not self._hard_gate_supported(results[atom_id])
                )
        unresolved = [
            atom_id
            for atom_id, result in results.items()
            if result.match_status in {"gap", "unsupported", "unverified_lead", "partial"}
        ]
        uncertainty: list[str] = []
        if len(atoms) < 3:
            uncertainty.append("insufficient_jd_detail")
        if any(item.has_contradiction for item in results.values()):
            uncertainty.append("contradictory_evidence")
        if hard_blockers:
            uncertainty.append("hard_gate_unverified")
        if (
            "source_span_offsets_canonicalized" in analysis.warnings
            or "source_span_offsets_canonicalized" in mapping.warnings
        ):
            uncertainty.append("source_span_offsets_canonicalized")
        if self._band_sensitive(analysis, estimated_by_atom, band, config):
            uncertainty.append("single_atom_band_sensitivity")
        needs_review = bool(uncertainty)
        confidence_label = self._confidence_band(confidence, config)
        strengths = [atoms[key].text for key, value in verified_by_atom.items() if value >= 1.0]
        needs_confirmation = [atoms[key].text for key in unresolved]
        critical_gaps = [atoms[key].text for key in hard_blockers]
        display_band = "insufficient_information" if len(atoms) < 3 else band
        display = EvidenceDisplaySummary(
            verdict_label=self._verdict_label(display_band, needs_review),
            confidence_label={"high": "高", "medium": "中", "low": "低"}[confidence_label],
            strengths=strengths,
            needs_confirmation=needs_confirmation,
            critical_gaps=critical_gaps,
        )
        return EvidenceAssessment(
            fit_policy_version=config.version,
            estimated_fit_band=display_band,
            evidence_confidence=confidence_label,
            verified_coverage=round(verified, 4),
            hard_blocker_atom_ids=hard_blockers,
            unresolved_atom_ids=unresolved,
            uncertainty_reasons=uncertainty,
            needs_review=needs_review,
            display_summary=display,
        )

    @staticmethod
    def _estimated(result: AtomEvidenceResult) -> float:
        values = [
            1.0 if link.support_status == "supported" else 0.5
            for link in result.links
            if link.support_status in {"supported", "partial"}
        ]
        return max(values, default=0.0)

    @staticmethod
    def _verified(result: AtomEvidenceResult) -> float:
        values = [
            1.0 if link.support_status == "supported" else 0.5
            for link in result.links
            if link.level in _VERIFIED_LEVELS and link.support_status in {"supported", "partial"}
        ]
        return max(values, default=0.0)

    @staticmethod
    def _hard_gate_supported(result: AtomEvidenceResult) -> bool:
        return any(
            link.level in _VERIFIED_LEVELS and link.support_status == "supported"
            for link in result.links
        )

    @staticmethod
    def _confidence(result: AtomEvidenceResult, config: FitPolicyConfig) -> float:
        return max(
            (config.confidence_parameters.get(link.level.value, 0.0) for link in result.links),
            default=0.0,
        )

    @staticmethod
    def _aggregate(
        analysis: RequirementAnalysis,
        values: dict[str, float],
        config: FitPolicyConfig,
    ) -> float:
        weighted = 0.0
        denominator = 0.0
        for parent in analysis.parents:
            parent_value = sum(values[item] for item in parent.atom_ids) / len(parent.atom_ids)
            weight = config.importance_weights[parent.importance]
            weighted += parent_value * weight
            denominator += weight
        return weighted / denominator if denominator else 0.0

    @staticmethod
    def _fit_band(score: float, config: FitPolicyConfig) -> str:
        if score >= config.fit_band_thresholds["high"]:
            return "high"
        if score >= config.fit_band_thresholds["medium"]:
            return "medium"
        return "low"

    @staticmethod
    def _confidence_band(score: float, config: FitPolicyConfig) -> str:
        if score >= config.confidence_parameters["high"]:
            return "high"
        if score >= config.confidence_parameters["medium"]:
            return "medium"
        return "low"

    def _band_sensitive(
        self,
        analysis: RequirementAnalysis,
        values: dict[str, float],
        current_band: str,
        config: FitPolicyConfig,
    ) -> bool:
        for atom_id, value in values.items():
            changed = dict(values)
            changed[atom_id] = 0.0 if value > 0 else 1.0
            if self._fit_band(self._aggregate(analysis, changed, config), config) != current_band:
                return True
        return False

    @staticmethod
    def _verdict_label(band: str, needs_review: bool) -> str:
        if band == "insufficient_information" or needs_review:
            return "信息不足"
        return {"high": "较匹配", "medium": "部分匹配", "low": "不匹配"}[band]

    @staticmethod
    def _empty_assessment(config: FitPolicyConfig) -> EvidenceAssessment:
        return EvidenceAssessment(
            fit_policy_version=config.version,
            estimated_fit_band="insufficient_information",
            evidence_confidence="low",
            verified_coverage=0.0,
            uncertainty_reasons=["insufficient_jd_detail"],
            needs_review=True,
            display_summary=EvidenceDisplaySummary(
                verdict_label="信息不足",
                confidence_label="低",
                needs_confirmation=["JD 未提供可验证的明确要求"],
            ),
        )
