from __future__ import annotations

from job_agent.schemas import Verdict


def evaluate_predictions(gold: list[Verdict], pred: list[Verdict]) -> dict[str, float | int]:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    labels = list(Verdict)
    if not gold:
        return {
            "decision_accuracy": 0.0,
            "count": 0,
            "macro_recall": 0.0,
            "per_class_recall": {label.value: 0.0 for label in labels},
            "prediction_distribution": {label.value: 0 for label in labels},
            "confusion_matrix": {label.value: {other.value: 0 for other in labels} for label in labels},
        }
    correct = sum(1 for expected, actual in zip(gold, pred) if expected == actual)
    confusion_matrix = {label.value: {other.value: 0 for other in labels} for label in labels}
    prediction_distribution = {label.value: 0 for label in labels}
    for expected, actual in zip(gold, pred):
        confusion_matrix[expected.value][actual.value] += 1
        prediction_distribution[actual.value] += 1
    per_class_recall = {}
    for label in labels:
        support = sum(confusion_matrix[label.value].values())
        per_class_recall[label.value] = confusion_matrix[label.value][label.value] / support if support else 0.0
    return {
        "decision_accuracy": correct / len(gold),
        "count": len(gold),
        "macro_recall": sum(per_class_recall.values()) / len(labels),
        "per_class_recall": per_class_recall,
        "prediction_distribution": prediction_distribution,
        "confusion_matrix": confusion_matrix,
    }
