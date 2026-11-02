import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from app.config import Settings
from app.main import AppServices, create_app
from app.schemas import ClassificationRequest, SearchRequest
from fastapi import HTTPException


class FakeRepository:
    def __init__(self):
        self.record = {
            "arxiv_id": "2603.00001", "title": "Graph methods", "abstract": "Node classification",
            "primary_cat": "cs.AI", "all_cats": "cs.AI", "label": 1, "split": "train",
        }

    def ready(self):
        return True

    def paper_count(self):
        return 1

    def label_counts(self):
        return {"NLP/AI": 1}

    def list_papers(self, **_kwargs):
        return [self.record], 1

    def get_paper(self, arxiv_id):
        return self.record if arxiv_id == self.record["arxiv_id"] else None


class FakeClassifier:
    def predict(self, title, abstract):
        if not title.strip() and not abstract.strip():
            raise ValueError("Title and abstract cannot both be empty")
        return {"model_id": "P3/tfidf_svm", "label": 1, "category": "NLP/AI", "score_kind": "decision_function_margin", "decision_scores": {"Algorithms": -0.1, "NLP/AI": 0.8, "Systems/Infra": -0.2, "Vision": -0.3}}


class FakeSearch:
    def search(self, query, method, k):
        return []


class FakeExperiments:
    def response(self):
        return {"classifiers": [], "retrieval": [], "classifier_selection": {}, "retrieval_judgment_note": "Test fixture only."}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = FakeRepository()
        services = AppServices(cls.repository, FakeClassifier(), FakeSearch(), FakeExperiments())
        settings = Settings("sqlite://", ROOT / "artifacts" / "data")
        cls.app = create_app(settings, services)
        cls.services = services
        cls.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=services)))
        cls.endpoints = {
            route.path: route.endpoint
            for route in cls.app.routes
            if hasattr(route, "path") and hasattr(route, "endpoint")
        }

    def test_health_and_readiness_handlers(self):
        health = self.endpoints["/api/v1/health"]()
        ready = self.endpoints["/api/v1/ready"](self.request)
        self.assertEqual(health.model_dump(), {"status": "ok", "database": None})
        self.assertEqual(ready.database, "connected")

    def test_paper_handlers_preserve_original_fields(self):
        response = self.endpoints["/api/v1/papers"](self.request, offset=0, limit=5)
        detail = self.endpoints["/api/v1/papers/{arxiv_id}"]("2603.00001", self.request)
        self.assertEqual(response.total, 1)
        self.assertEqual(detail.all_cats, "cs.AI")
        self.assertEqual(set(detail.model_dump()), {"arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label", "split"})

    def test_missing_paper_raises_not_found_and_invalid_input_fails_schema_validation(self):
        for candidate in ("missing", "../../phase1/metadata.json", "2603.00001; DROP TABLE papers"):
            with self.subTest(candidate=candidate), self.assertRaises(HTTPException) as missing:
                self.endpoints["/api/v1/papers/{arxiv_id}"](candidate, self.request)
            self.assertEqual(missing.exception.status_code, 404)
        with self.assertRaises(Exception):
            SearchRequest(query=" ", k=2)

    def test_classification_and_search_handlers_return_typed_contracts(self):
        classified = self.endpoints["/api/v1/classifications"](
            ClassificationRequest(title="Graph models", abstract="Graph representation learning."), self.request
        )
        searched = self.endpoints["/api/v1/search"](SearchRequest(query="graphs", method="tfidf", k=5), self.request)
        self.assertEqual(classified.category, "NLP/AI")
        self.assertEqual(searched.method, "tfidf")
        self.assertIn("not probability", searched.score_description)

    def test_route_prefixes_and_methods_are_registered(self):
        registered = {(method, route.path) for route in self.app.routes for method in getattr(route, "methods", set())}
        self.assertIn(("GET", "/api/v1/health"), registered)
        self.assertIn(("GET", "/api/v1/papers"), registered)
        self.assertIn(("POST", "/api/v1/classifications"), registered)
        self.assertIn(("POST", "/api/v1/search"), registered)

    def test_openapi_documents_typed_classification_and_search_requests(self):
        schema = self.app.openapi()
        self.assertIn("/api/v1/classifications", schema["paths"])
        self.assertIn("/api/v1/search", schema["paths"])
        components = schema["components"]["schemas"]
        self.assertIn("ClassificationRequest", components)
        self.assertIn("SearchRequest", components)
        self.assertEqual(components["SearchRequest"]["properties"]["method"]["default"], "tfidf")


if __name__ == "__main__":
    unittest.main()
