import unittest

from ml.evaluation.phase8 import (
    _manifest_and_integrity,
    build_classification_results,
    build_error_analysis,
    build_retrieval_results,
)


class SavedArtifactEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.integrity = _manifest_and_integrity()

    def test_all_saved_classification_predictions_are_aligned_and_recomputed(self):
        result, predictions, _sources = build_classification_results(self.splits, self.integrity)
        self.assertEqual(result["metadata"]["model_count"], 13)
        self.assertEqual(len(predictions), 13)
        self.assertTrue(all(row["saved_test_metric_check"] == "passed" for row in result["models"]))
        self.assertEqual(result["metadata"]["split_overlap_counts"], {"train_validation": 0, "train_test": 0, "validation_test": 0})
        self.assertEqual(result["selection_summary"]["best_observed_test_macro_f1"]["model_id"], "P3/tfidf_svm")
        self.assertIsNone(result["selection_summary"]["phase_selected_models"]["P5"])

    def test_retrieval_scores_match_saved_artifacts_and_local_rankings(self):
        result, _sources = build_retrieval_results()
        self.assertEqual(set(result["methods"]), {"phase4_tfidf", "phase4_word2vec", "phase7_minilm"})
        self.assertAlmostEqual(result["methods"]["phase4_tfidf"]["macro_average"]["ndcg"], 0.7801234381420308)
        reproduction = result["phase7_reproduction"]
        self.assertTrue(reproduction["saved_top10_ids_reproduced"])
        self.assertFalse(reproduction["investigation"]["bug_found"])
        self.assertFalse(reproduction["investigation"]["corrected_metric_output_created"])

    def test_error_records_keep_all_original_phase1_fields_and_shared_ids(self):
        result, predictions, _sources = build_classification_results(self.splits, self.integrity)
        analysis, error_rows = build_error_analysis(result, predictions, self.splits["test"])
        self.assertGreater(len(error_rows), 0)
        self.assertEqual(analysis["total_model_errors"], len(error_rows))
        self.assertTrue(all(field in error_rows[0] for field in ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")))
        self.assertTrue(analysis["representative_examples_by_common_confusion"])
        self.assertEqual(len(analysis["shared_error_ids"]["all_records"]), analysis["shared_error_ids"]["records_with_errors_in_at_least_two_models"])


if __name__ == "__main__":
    unittest.main()
