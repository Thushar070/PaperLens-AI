import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from sqlalchemy import create_engine, text

from app.db.repository import DatabaseRepository


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        with self.engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL, "
                "primary_cat TEXT NOT NULL, all_cats TEXT NOT NULL, label INTEGER NOT NULL, split TEXT NOT NULL)"
            ))
            connection.execute(text(
                "INSERT INTO papers VALUES "
                "('2603.00001','Graph methods','Node classification','cs.AI','cs.AI',1,'train'),"
                "('2603.00002','Image models','Diffusion vision','cs.CV','cs.CV',3,'test')"
            ))
        self.repository = DatabaseRepository("sqlite://", engine=self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_list_filter_count_and_pagination(self):
        rows, total = self.repository.list_papers(query="graph", offset=0, limit=10)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["arxiv_id"], "2603.00001")
        self.assertEqual(rows[0]["split"], "train")

    def test_detail_and_order_preserving_id_lookup(self):
        self.assertEqual(self.repository.get_paper("2603.00002")["label"], 3)
        selected = self.repository.get_papers_by_ids(["2603.00002", "2603.00001"])
        self.assertEqual(set(selected), {"2603.00001", "2603.00002"})
        self.assertEqual(self.repository.get_papers_by_ids([]), {})

    def test_overview_counts_keep_original_category_mapping(self):
        self.assertEqual(self.repository.paper_count(), 2)
        self.assertEqual(self.repository.label_counts(), {"NLP/AI": 1, "Vision": 1})

    def test_query_text_is_bound_and_does_not_execute_sql(self):
        rows, total = self.repository.list_papers(query="%' OR 1=1 --", offset=0, limit=10)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_vector_search_rejects_wrong_dimensions_and_nonfinite_values(self):
        for values in ([0.0] * 383, [float("nan")] * 384):
            with self.subTest(length=len(values)), self.assertRaisesRegex(ValueError, "384 finite"):
                self.repository.vector_search(values, k=5)


if __name__ == "__main__":
    unittest.main()
