import json
import tempfile
import unittest
from pathlib import Path

from ml.evaluation.charts import generate_charts, make_chart_data
from ml.evaluation.metrics import classification_metrics, LABEL_NAMES


def fixture_inputs():
    model_ids = ["P3/tfidf_svm", "P5/bilstm_attention", "P6/custom_transformer", "P7/minilm_finetuned"]
    labels = [0, 1, 2, 3]
    records = []
    for index, model_id in enumerate(model_ids):
        metrics = classification_metrics(labels, [0, 1, 2, 3 if index else 2])
        records.append({
            "model_id": model_id,
            "short_name": model_id.split("/", 1)[1],
            "test_metrics": metrics,
        })
    classification = {"metadata": {"label_order": list(labels), "label_names": {str(key): value for key, value in LABEL_NAMES.items()}}, "models": records}
    methods = {}
    for method in ("phase4_tfidf", "phase4_word2vec", "phase7_minilm"):
        per_query = {query: {"precision": 0.2, "recall": 0.2, "mrr": 0.5, "ndcg": 0.4, "unjudged_in_top_k": 1} for query in ("q1", "q2", "q3")}
        methods[method] = {"display_name": method, "per_query": per_query, "macro_average": {key: value for key, value in per_query["q1"].items()}}
    retrieval = {"methods": methods, "query_ids": ["q1", "q2", "q3"], "macro_average": {"phase4_tfidf": methods["phase4_tfidf"]["macro_average"], "phase4_word2vec": methods["phase4_word2vec"]["macro_average"], "phase7_minilm": methods["phase7_minilm"]["macro_average"]}}
    efficiency = {"models": [
        {"model_id": record["model_id"], "training_seconds": 10 + index, "test_macro_f1": record["test_metrics"]["macro_f1"], "artifact_bytes": 1024 * (index + 1), "size_basis": "fixture"}
        for index, record in enumerate(records)
    ]}
    return classification, retrieval, efficiency


class ChartTests(unittest.TestCase):
    def test_chart_data_is_directly_derived_from_source_records(self):
        classification, retrieval, efficiency = fixture_inputs()
        data = make_chart_data(classification, retrieval, efficiency)
        self.assertEqual(len(data["classification_macro_f1"]), 4)
        self.assertEqual(data["classification_macro_f1"][0]["value"], classification["models"][0]["test_metrics"]["macro_f1"])
        self.assertEqual(data["retrieval_macro_metrics"], retrieval["macro_average"])
        self.assertEqual(data["artifact_size_vs_macro_f1"][0]["artifact_bytes"], efficiency["models"][0]["artifact_bytes"])

    def test_publication_charts_and_machine_readable_manifest(self):
        classification, retrieval, efficiency = fixture_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = generate_charts(classification, retrieval, efficiency, root)
            saved = json.loads((root / "chart_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["chart_count"], saved["chart_count"])
            self.assertGreaterEqual(manifest["chart_count"], 8)
            for chart in manifest["charts"]:
                data = (root / chart["file"]).read_bytes()
                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            chart_data = json.loads((root / "chart_data.json").read_text(encoding="utf-8"))
            self.assertEqual(chart_data["classification_macro_f1"][0]["value"], classification["models"][0]["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
