from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MIN_ACCURACY = 0.625
MIN_MACRO_RECALL = 0.625
MIN_STRONG_RECALL = 0.5
MIN_RISKY_RECALL = 0.5


def _load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics payload must be an object: {path}")
    required = {"decision_accuracy", "macro_recall", "schema_valid_rate", "fallback_rate", "per_class_recall", "prediction_distribution", "count"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Metrics payload missing {missing}: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, actual: float | int, expected: float | int, passed: bool) -> dict[str, object]:
    return {"name": name, "actual": actual, "expected": expected, "passed": passed}


def evaluate_candidate(
    *,
    candidate_name: str,
    candidate_metrics_path: Path,
    baseline_metrics_path: Path,
    corpus_path: Path,
) -> dict[str, object]:
    candidate = _load_metrics(candidate_metrics_path)
    baseline = _load_metrics(baseline_metrics_path)
    recalls = candidate["per_class_recall"]
    distribution = candidate["prediction_distribution"]
    if not isinstance(recalls, dict) or not isinstance(distribution, dict):
        raise ValueError("Metrics recall and distribution fields must be objects")

    count = candidate["count"]
    checks = [
        _check("schema_valid_rate", candidate["schema_valid_rate"], 1.0, candidate["schema_valid_rate"] == 1.0),
        _check("fallback_rate", candidate["fallback_rate"], 0.0, candidate["fallback_rate"] == 0.0),
        _check("decision_accuracy", candidate["decision_accuracy"], MIN_ACCURACY, candidate["decision_accuracy"] >= MIN_ACCURACY),
        _check("macro_recall", candidate["macro_recall"], MIN_MACRO_RECALL, candidate["macro_recall"] >= MIN_MACRO_RECALL),
        _check("strong_fit_recall", recalls.get("strong fit", 0.0), MIN_STRONG_RECALL, recalls.get("strong fit", 0.0) >= MIN_STRONG_RECALL),
        _check("risky_fit_recall", recalls.get("risky fit", 0.0), MIN_RISKY_RECALL, recalls.get("risky fit", 0.0) >= MIN_RISKY_RECALL),
        _check("no_majority_verdict", max(distribution.values(), default=0), count / 2, max(distribution.values(), default=0) <= count / 2),
        _check("evaluation_count_matches_baseline", candidate["count"], baseline["count"], candidate["count"] == baseline["count"]),
        _check("base_accuracy_non_regression", candidate["decision_accuracy"], baseline["decision_accuracy"], candidate["decision_accuracy"] >= baseline["decision_accuracy"]),
        _check("base_macro_recall_non_regression", candidate["macro_recall"], baseline["macro_recall"], candidate["macro_recall"] >= baseline["macro_recall"]),
    ]
    return {
        "candidate_name": candidate_name,
        "status": "pass" if all(check["passed"] for check in checks) else "reject",
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "candidate_metrics_path": str(candidate_metrics_path),
        "candidate_metrics_sha256": _sha256(candidate_metrics_path),
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_metrics_sha256": _sha256(baseline_metrics_path),
        "checks": checks,
    }
