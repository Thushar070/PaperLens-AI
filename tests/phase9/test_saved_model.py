import sys
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.classification import ClassificationService
from ml.models.pretrained.data import read_manifest


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ClassificationService(ROOT / "artifacts" / "data")
        test_rows = read_manifest(ROOT / "artifacts" / "data" / "phase1" / "test.jsonl")
        saved = json.loads((ROOT / "artifacts" / "data" / "phase3" / "test" / "tfidf_svm_predictions.json").read_text(encoding="utf-8"))
        expected = {row["arxiv_id"]: int(row["predicted"]) for row in saved}
        cls.sample = test_rows[0]
        cls.expected_label = expected[cls.sample["arxiv_id"]]

    def test_saved_classifier_returns_real_label_and_non_probability_scores(self):
        result = self.service.predict(self.sample["title"], self.sample["abstract"])
        self.assertEqual(result["model_id"], "P3/tfidf_svm")
        self.assertIn(result["label"], range(4))
        self.assertEqual(result["label"], self.expected_label)
        self.assertEqual(result["category"], {0: "Algorithms", 1: "NLP/AI", 2: "Systems/Infra", 3: "Vision"}[result["label"]])
        self.assertEqual(result["score_kind"], "decision_function_margin")
        self.assertEqual(set(result["decision_scores"]), {"Algorithms", "NLP/AI", "Systems/Infra", "Vision"})
        self.assertFalse(hasattr(self.service.model, "predict_proba"))

    def test_empty_document_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot both be empty"):
            self.service.predict(" ", "\n")


if __name__ == "__main__":
    unittest.main()
