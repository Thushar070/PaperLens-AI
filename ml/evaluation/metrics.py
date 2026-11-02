"""Strict, fixed-label evaluation metrics used by Phase 8."""

from __future__ import annotations

from typing import Iterable, Mapping

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


LABEL_IDS = (0, 1, 2, 3)
LABEL_NAMES = {
    0: "Algorithms",
    1: "NLP/AI",
    2: "Systems/Infra",
    3: "Vision",
}


def _validated_labels(values: Iterable[int], field: str) -> list[int]:
    result = list(values)
    if not result:
        raise ValueError(f"{field} must not be empty")
    if any(type(value) is not int or value not in LABEL_IDS for value in result):
        raise ValueError(f"{field} must contain integer labels in {LABEL_IDS}")
    return result


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    label_names: Mapping[int, str] | None = None,
) -> dict:
    """Compute full classification metrics in the fixed Phase 1 label order."""
    actual = _validated_labels(y_true, "y_true")
    predicted = _validated_labels(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred lengths differ")
    names = dict(LABEL_NAMES if label_names is None else label_names)
    if tuple(names) != LABEL_IDS or any(not isinstance(names[i], str) or not names[i] for i in LABEL_IDS):
        raise ValueError("label_names must map labels 0–3 in Phase 1 order")

    per_class = precision_recall_fscore_support(
        actual, predicted, labels=list(LABEL_IDS), average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        actual, predicted, labels=list(LABEL_IDS), average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        actual, predicted, labels=list(LABEL_IDS), average="weighted", zero_division=0
    )
    return {
        "records": len(actual),
        "label_order": list(LABEL_IDS),
        "label_names": {str(key): names[key] for key in LABEL_IDS},
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": [
            {
                "label": label,
                "name": names[label],
                "precision": float(per_class[0][position]),
                "recall": float(per_class[1][position]),
                "f1": float(per_class[2][position]),
                "support": int(per_class[3][position]),
            }
            for position, label in enumerate(LABEL_IDS)
        ],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(LABEL_IDS)).tolist(),
    }


def validate_experiment_record(record: Mapping) -> None:
    """Validate the core schema for a consolidated classification experiment."""
    required = {
        "model_id",
        "phase",
        "status",
        "selection_metric",
        "selected_by_validation",
        "configuration",
        "validation_metrics",
        "test_metrics",
        "sources",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Experiment record is missing required fields: {missing}")
    if not isinstance(record["model_id"], str) or not record["model_id"]:
        raise ValueError("model_id must be a non-empty string")
    if record["status"] not in {"complete", "partial", "failed"}:
        raise ValueError("status must be complete, partial, or failed")
    if type(record["selected_by_validation"]) is not bool:
        raise ValueError("selected_by_validation must be boolean")
    for split in ("validation_metrics", "test_metrics"):
        metrics = record[split]
        if not isinstance(metrics, Mapping) or not 0 <= float(metrics["macro_f1"]) <= 1:
            raise ValueError(f"{split}.macro_f1 must be a metric in [0, 1]")
    if not isinstance(record["configuration"], Mapping) or not isinstance(record["sources"], Mapping):
        raise ValueError("configuration and sources must be mappings")
