import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from app.services.experiments import ExperimentCatalog


class SavedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ExperimentCatalog(ROOT / "artifacts" / "data")
        cls.response = cls.catalog.response()

    def test_catalog_returns_all_saved_classifier_and_retrieval_rows(self):
        self.assertEqual(len(self.response["classifiers"]), 13)
        self.assertEqual(len(self.response["retrieval"]), 3)

    def test_selected_classifier_and_scores_match_phase8_artifact(self):
        selected = next(row for row in self.response["classifiers"] if row["model_id"] == "P3/tfidf_svm")
        self.assertTrue(selected["selected_by_validation"])
        self.assertAlmostEqual(selected["validation_macro_f1"], 0.9580092830870094)
        self.assertAlmostEqual(selected["test_macro_f1"], 0.9520151187335631)
        source = json.loads((ROOT / "artifacts/data/phase8/classification_results.json").read_text(encoding="utf-8"))
        self.assertEqual(selected["test_macro_f1"], next(m["test_metrics"]["macro_f1"] for m in source["models"] if m["model_id"] == "P3/tfidf_svm"))

    def test_sparse_retrieval_judgment_limit_is_returned(self):
        self.assertIn("27", self.response["retrieval_judgment_note"])
        self.assertIn("not human-reviewed", self.response["retrieval_judgment_note"])


if __name__ == "__main__":
    unittest.main()
