from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.schemas import LABEL_NAMES
from ml.preprocessing.text import document


class ClassificationService:
    """One process-wide instance of the saved Phase 3 TF-IDF/SVM pipeline."""

    def __init__(self, artifact_root: Path, model_id: str = "P3/tfidf_svm") -> None:
        if model_id != "P3/tfidf_svm":
            raise ValueError("The Phase 9 default classifier is the validation-selected P3/tfidf_svm")
        phase1 = artifact_root / "phase1"
        phase2 = artifact_root / "phase2"
        phase3 = artifact_root / "phase3"
        self.model_id = model_id
        self.model = joblib.load(phase3 / "models" / "tfidf_svm.joblib")
        self.vectorizer = joblib.load(phase2 / "tfidf.joblib")
        mapping = json.loads((phase1 / "metadata.json").read_text(encoding="utf-8"))["label_mapping"]
        self.labels = {int(label): str(name) for label, name in mapping.items()}
        if self.labels != LABEL_NAMES:
            raise ValueError("Saved Phase 1 label mapping does not match the verified four-class mapping")

    def predict(self, title: str, abstract: str) -> dict[str, Any]:
        text_value = document({"title": title, "abstract": abstract})
        if not text_value.strip():
            raise ValueError("Title and abstract cannot both be empty")
        features = self.vectorizer.transform([text_value])
        label = int(self.model.predict(features)[0])
        margins = np.atleast_2d(self.model.decision_function(features))[0]
        scores = {self.labels[int(index)]: float(score) for index, score in zip(self.model.classes_, margins)}
        return {
            "model_id": self.model_id,
            "label": label,
            "category": self.labels[label],
            "score_kind": "decision_function_margin",
            "decision_scores": scores,
        }
