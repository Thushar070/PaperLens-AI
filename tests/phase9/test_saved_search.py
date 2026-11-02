import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from ml.retrieval.engine import Retriever


class SavedSearchSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(ROOT / "artifacts" / "data")

    def test_saved_tfidf_search_reproduces_phase8_top_five_order(self):
        started = time.perf_counter()
        results = self.retriever.search("graph neural networks node classification", method="tfidf", k=5)
        elapsed = time.perf_counter() - started
        self.assertEqual([row["arxiv_id"] for row in results], ["2505.15847", "2501.06236", "2311.14469", "2407.07713", "2510.1521"])
        self.assertTrue(all(0.0 <= row["score"] <= 1.0 for row in results))
        self.assertLess(elapsed, 20.0, f"saved TF-IDF query exceeded 20-second smoke limit: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
