import unittest

from ml.evaluation.metrics import LABEL_NAMES, classification_metrics, validate_experiment_record


class ClassificationMetricTests(unittest.TestCase):
    def test_fixed_order_metrics_and_confusion_matrix(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["label_order"], [0, 1, 2, 3])
        self.assertEqual(result["label_names"]["2"], LABEL_NAMES[2])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual([row["support"] for row in result["per_class"]], [2, 2, 0, 0])
        self.assertAlmostEqual(result["macro_f1"], sum(row["f1"] for row in result["per_class"]) / 4)

    def test_rejects_bad_labels_empty_input_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([0], [0, 1])
        with self.assertRaises(ValueError):
            classification_metrics([4], [0])
        with self.assertRaises(ValueError):
            classification_metrics([True], [0])

    def test_experiment_record_schema(self):
        metrics = classification_metrics([0], [0])
        record = {
            "model_id": "P3/example",
            "phase": 3,
            "status": "complete",
            "selection_metric": "validation macro F1",
            "selected_by_validation": True,
            "configuration": {"C": 1.0},
            "validation_metrics": {"macro_f1": 1.0},
            "test_metrics": metrics,
            "sources": {"predictions": "saved.json"},
        }
        validate_experiment_record(record)
        with self.assertRaises(ValueError):
            validate_experiment_record({**record, "selected_by_validation": "yes"})
        with self.assertRaises(ValueError):
            validate_experiment_record({**record, "status": "unknown"})


if __name__ == "__main__":
    unittest.main()
