import unittest

from ml.retrieval.metrics import evaluate


class RetrievalMetricTests(unittest.TestCase):
    def test_top_k_metrics_and_unjudged_results(self):
        result = evaluate(["a", "b", "x"], {"a": 2, "b": 1, "c": 0}, 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["mrr"], 1.0)
        self.assertAlmostEqual(result["ndcg"], 1.0)
        self.assertEqual(result["unjudged_in_top_k"], 1)

    def test_duplicate_ranked_ids_and_no_relevant_judgments_fail(self):
        with self.assertRaises(ValueError):
            evaluate(["a", "a"], {"a": 1}, 2)
        with self.assertRaises(ValueError):
            evaluate(["a"], {"a": 0}, 1)

    def test_invalid_k_and_judgment_grades_fail(self):
        with self.assertRaises(ValueError):
            evaluate(["a"], {"a": 2}, 0)
        with self.assertRaises(ValueError):
            evaluate(["a"], {"a": 3}, 1)


if __name__ == "__main__":
    unittest.main()
