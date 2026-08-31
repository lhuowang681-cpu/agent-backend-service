"""Evaluation-only components that do not mutate the production agent graph."""

from job_agent.evaluation.evidence_grounding_contracts import (
    DatasetCase,
    EvidenceGroundingDataset,
    EvidencePrediction,
    EvidenceSpan,
    PredictionRecord,
)
from job_agent.evaluation.evidence_grounding_metrics import evaluate_predictions
from job_agent.evaluation.evidence_grounding_validators import (
    apply_entailment_cap,
    canonicalize_unique_quote_offsets,
    validate_prediction_sources,
)

__all__ = [
    "DatasetCase",
    "EvidenceGroundingDataset",
    "EvidencePrediction",
    "EvidenceSpan",
    "PredictionRecord",
    "apply_entailment_cap",
    "canonicalize_unique_quote_offsets",
    "evaluate_predictions",
    "validate_prediction_sources",
]
