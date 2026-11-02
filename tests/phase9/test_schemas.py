import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from pydantic import ValidationError

from app.schemas import ClassificationRequest, PaperRecord, SearchRequest


class SchemaTests(unittest.TestCase):
    def test_paper_contract_preserves_six_source_fields(self):
        row = PaperRecord(
            arxiv_id="2603.08709",
            title="A Paper",
            abstract="An abstract.",
            primary_cat="cs.CV",
            all_cats="cs.CV | cs.AI",
            label=3,
        )
        self.assertEqual(set(row.model_dump()), {"arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label"})
        self.assertEqual(row.label, 3)

    def test_paper_rejects_unknown_label_and_extra_fields(self):
        fields = {
            "arxiv_id": "2603.08709", "title": "A", "abstract": "B",
            "primary_cat": "cs.CV", "all_cats": "cs.CV", "label": 4,
        }
        with self.assertRaises(ValidationError):
            PaperRecord(**fields)
        fields["label"] = 3
        fields["prediction"] = 3
        with self.assertRaises(ValidationError):
            PaperRecord(**fields)

    def test_classification_rejects_excessive_input_and_non_text(self):
        with self.assertRaises(ValidationError):
            ClassificationRequest(title="x" * 1001, abstract="")
        with self.assertRaises(ValidationError):
            ClassificationRequest(title=42, abstract="valid")

    def test_search_rejects_blank_queries_invalid_methods_and_out_of_range_k(self):
        for payload in (
            {"query": "   "},
            {"query": "graph", "method": "unknown"},
            {"query": "graph", "k": 51},
            {"query": "graph", "k": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                SearchRequest(**payload)


if __name__ == "__main__":
    unittest.main()
