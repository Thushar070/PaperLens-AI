import json
import os
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


API_BASE_URL = os.environ.get("PAPERLENS_TEST_API_BASE_URL", "").rstrip("/")
_opener = build_opener(ProxyHandler({}))


def request_json(path, *, method="GET", body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        f"{API_BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with _opener.open(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@unittest.skipUnless(API_BASE_URL, "set PAPERLENS_TEST_API_BASE_URL to run live HTTP integration tests")
class LiveAPIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status, cls.papers = request_json("/papers?limit=1")
        if status != 200 or not cls.papers["items"]:
            raise RuntimeError(f"Live database did not return a sample paper (HTTP {status})")
        cls.sample = cls.papers["items"][0]

    def test_health_readiness_and_real_database_records(self):
        status, health = request_json("/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, ready = request_json("/ready")
        self.assertEqual(status, 200)
        self.assertEqual(ready["database"], "connected")
        status, papers = request_json("/papers?limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(papers["total"], 19999)
        self.assertEqual(len(papers["items"]), 3)
        self.assertEqual(set(papers["items"][0]), {
            "arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label", "split"
        })

    def test_filters_details_and_empty_results(self):
        status, filtered = request_json("/papers?label=3&limit=5")
        self.assertEqual(status, 200)
        self.assertEqual(filtered["total"], 5000)
        self.assertTrue(all(row["label"] == 3 for row in filtered["items"]))
        status, detail = request_json(f"/papers/{self.sample['arxiv_id']}")
        self.assertEqual(status, 200)
        self.assertEqual(detail["arxiv_id"], self.sample["arxiv_id"])
        status, missing = request_json("/papers/paperlens-no-such-id")
        self.assertEqual(status, 404)
        self.assertIn("not found", missing["detail"].lower())
        status, empty = request_json("/papers?query=paperlens-no-match-001")
        self.assertEqual(status, 200)
        self.assertEqual(empty["total"], 0)
        self.assertEqual(empty["items"], [])

    def test_saved_classifier_and_both_database_backed_retrieval_methods(self):
        status, prediction = request_json(
            "/classifications",
            method="POST",
            body={"title": self.sample["title"], "abstract": self.sample["abstract"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(prediction["model_id"], "P3/tfidf_svm")
        self.assertIn(prediction["label"], range(4))
        self.assertTrue(prediction["category"])

        for method in ("tfidf", "minilm"):
            with self.subTest(method=method):
                status, results = request_json(
                    "/search",
                    method="POST",
                    body={"query": "graph neural networks for node classification", "method": method, "k": 5},
                )
                self.assertEqual(status, 200)
                self.assertEqual(results["method"], method)
                self.assertEqual(len(results["items"]), 5)
                self.assertTrue(all("arxiv_id" in row and "score" in row for row in results["items"]))

    def test_overview_experiments_and_invalid_requests(self):
        status, overview = request_json("/overview")
        self.assertEqual(status, 200)
        self.assertEqual(overview["paper_count"], 19999)
        status, experiments = request_json("/experiments")
        self.assertEqual(status, 200)
        self.assertEqual(len(experiments["classifiers"]), 13)
        self.assertEqual(len(experiments["retrieval"]), 3)
        status, _ = request_json("/papers?label=9")
        self.assertEqual(status, 422)
        status, _ = request_json("/classifications", method="POST", body={})
        self.assertEqual(status, 422)
        status, _ = request_json(
            "/search", method="POST", body={"query": " ", "method": "tfidf", "k": 5}
        )
        self.assertEqual(status, 422)


if __name__ == "__main__":
    unittest.main()
