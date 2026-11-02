import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from sqlalchemy import create_engine, text

from app.db.ingest import ingest
from app.db.migrate import migrate
from app.db.repository import DatabaseRepository


DATABASE_URL = os.environ.get("PAPERLENS_TEST_DATABASE_URL")


@unittest.skipUnless(DATABASE_URL, "set PAPERLENS_TEST_DATABASE_URL to run live PostgreSQL integration tests")
class PostgreSQLIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        cls.repository = DatabaseRepository(DATABASE_URL, engine=cls.engine)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_database_schema_counts_labels_splits_and_embeddings(self):
        with self.engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT current_database()")).scalar_one(), "paperlens")
            self.assertIsNotNone(
                connection.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar_one_or_none()
            )
            counts = connection.execute(
                text("SELECT count(*), count(DISTINCT arxiv_id) FROM papers")
            ).one()
            self.assertEqual(tuple(counts), (19999, 19999))
            labels = dict(connection.execute(
                text("SELECT label, count(*) FROM papers GROUP BY label")
            ).all())
            self.assertEqual(labels, {0: 4999, 1: 5000, 2: 5000, 3: 5000})
            splits = dict(connection.execute(
                text("SELECT split, count(*) FROM papers GROUP BY split")
            ).all())
            self.assertEqual(splits, {"train": 15999, "validation": 2000, "test": 2000})
            dimensions = connection.execute(
                text("SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) FROM paper_embeddings")
            ).one()
            self.assertEqual(tuple(dimensions), (19999, 384, 384))
            missing_embeddings = connection.execute(text(
                "SELECT count(*) FROM papers p LEFT JOIN paper_embeddings e USING (arxiv_id) "
                "WHERE e.arxiv_id IS NULL"
            )).scalar_one()
            self.assertEqual(missing_embeddings, 0)

    def test_repository_filters_and_details_use_postgresql(self):
        rows, total = self.repository.list_papers(label=3, offset=0, limit=5)
        self.assertEqual(total, 5000)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["label"] == 3 for row in rows))
        sample = self.repository.get_paper("2603.08709")
        self.assertIsNotNone(sample)
        self.assertEqual(sample["label"], 3)
        self.assertEqual(sample["primary_cat"], "cs.CV")
        self.assertEqual(sample["split"], "train")

    def test_migration_and_artifact_ingestion_are_idempotent(self):
        self.assertEqual(migrate(DATABASE_URL), [])
        result = ingest(database_url=DATABASE_URL)
        self.assertEqual(result["papers_upserted"], 19999)
        self.assertEqual(result["embeddings_upserted"], 19999)
        self.assertEqual(result["experiments_upserted"], 16)
        self.assertEqual(self.repository.paper_count(), 19999)


if __name__ == "__main__":
    unittest.main()
