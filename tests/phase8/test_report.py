import unittest

from ml.evaluation.metrics import classification_metrics, LABEL_NAMES
from ml.evaluation.phase8 import _report_markdown


class ReportTests(unittest.TestCase):
    def test_report_contains_recorded_metrics_retrieval_errors_and_chart_references(self):
        ids = ["P3/tfidf_svm", "P5/bilstm_attention", "P6/custom_transformer", "P7/minilm_finetuned"]
        models = []
        for index, model_id in enumerate(ids):
            metrics = classification_metrics([0, 1, 2, 3], [0, 1, 2, 2 if index == 0 else 3])
            models.append({
                "model_id": model_id,
                "short_name": model_id,
                "phase": index + 3,
                "family": "fixture family",
                "validation_metrics": {"macro_f1": 0.8 + index / 100},
                "test_metrics": metrics,
                "selected_by_validation": index == 0,
            })
        classification = {"models": models, "metadata": {
            "label_order": [0, 1, 2, 3],
            "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
            "split_sizes": {"train": 15999, "validation": 2000, "test": 2000},
            "test_support": {"0": 500, "1": 500, "2": 500, "3": 500},
        }}
        classification["selection_summary"] = {
            "best_validation_macro_f1_across_saved_experiments": {"model_id": "P3/tfidf_svm", "value": 0.95},
            "best_observed_test_macro_f1": {"model_id": "P3/tfidf_svm", "value": 0.93},
            "phase3_original_comparison_baseline": "P3/tfidf_svm",
        }
        methods = {}
        for method_id, name in (("phase4_tfidf", "Phase 4 TF-IDF"), ("phase4_word2vec", "Phase 4 Word2Vec"), ("phase7_minilm", "Phase 7 MiniLM cosine")):
            per_query = {query: {"precision": 0.2, "recall": 0.2, "mrr": 0.5, "ndcg": 0.4, "unjudged_in_top_k": 1} for query in ("q1", "q2", "q3")}
            methods[method_id] = {"display_name": name, "per_query": per_query, "macro_average": per_query["q1"]}
        retrieval = {"methods": methods, "query_ids": ["q1", "q2", "q3"], "judgments": {
                "total_query_document_grades": 27,
                "grade_counts": {"0": 2, "1": 16, "2": 9},
                "relevant_per_query": {"q1": 8, "q2": 8, "q3": 9},
        }, "latency": {
                "phase4_tfidf": {"mean_seconds": 0.002},
                "phase4_word2vec": {"mean_seconds": 0.003},
                "phase7_minilm": {"mean": 32.0},
        }}
        efficiency = {"models": []}
        errors = {
            "total_model_errors": 3,
            "most_common_misclassification_pairs": [{"actual_name": "Algorithms", "predicted_name": "NLP/AI", "count_across_models": 2}],
            "shared_error_ids": {"records_with_errors_in_at_least_two_models": 1, "misclassified_by_all_models": []},
            "representative_examples_by_common_confusion": [{"actual_name": "Algorithms", "predicted_name": "NLP/AI", "count_across_models": 2, "examples": [{"arxiv_id": "example", "title": "fixture", "actual_label": 0, "predicted_label": 1, "misclassified_by_models": ["P3/tfidf_svm"]}]}],
        }
        chart_manifest = {"charts": [{"file": "classification_macro_f1.png", "title": "Test macro F1"}]}
        report = _report_markdown(classification, retrieval, efficiency, errors, chart_manifest)
        self.assertIn("P3/tfidf_svm", report)
        self.assertIn("Phase 4", report)
        self.assertIn("classification_macro_f1.png", report)
        self.assertIn("2 |", report)
        self.assertIn("recomputed independently", report)


if __name__ == "__main__":
    unittest.main()
