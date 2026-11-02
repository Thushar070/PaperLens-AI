from __future__ import annotations

import json
from pathlib import Path


class ExperimentCatalog:
    """Read-only view of the verified Phase 8 machine-readable evaluation."""

    def __init__(self, artifact_root: Path) -> None:
        phase8 = artifact_root / "phase8"
        self.classification = json.loads((phase8 / "classification_results.json").read_text(encoding="utf-8"))
        self.retrieval = json.loads((phase8 / "retrieval_results.json").read_text(encoding="utf-8"))
        if len(self.classification.get("models", [])) != 13 or len(self.retrieval.get("methods", {})) != 3:
            raise ValueError("Phase 8 result artifacts have unexpected model or retrieval-method counts")

    def response(self) -> dict:
        classifier_rows = []
        for model in self.classification["models"]:
            validation = model.get("validation_metrics") or {}
            test = model.get("test_metrics") or {}
            classifier_rows.append({
                "model_id": model["model_id"],
                "short_name": model["short_name"],
                "phase": int(model["phase"]),
                "family": model["family"],
                "validation_macro_f1": validation.get("macro_f1"),
                "test_accuracy": test.get("accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "selected_by_validation": bool(model.get("selected_by_validation")),
            })
        retrieval_rows = []
        for method_id, method in self.retrieval["methods"].items():
            metrics = method["macro_average"]
            retrieval_rows.append({
                "method_id": method_id,
                "display_name": method.get("display_name", method_id),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "mrr": float(metrics["mrr"]),
                "ndcg": float(metrics["ndcg"]),
                "unjudged_in_top_k": float(metrics["unjudged_in_top_k"]),
            })
        judgment = self.retrieval["judgments"]
        judgment_note = judgment["interpretation"]
        judgment_note += (
            f" Metrics use {len(self.retrieval['query_ids'])} queries and "
            f"{int(judgment['total_query_document_grades'])} document judgments."
        )
        return {
            "classifiers": classifier_rows,
            "retrieval": retrieval_rows,
            "classifier_selection": self.classification["selection_summary"],
            "retrieval_judgment_note": judgment_note,
        }
