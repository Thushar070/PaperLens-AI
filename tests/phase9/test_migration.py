import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from app.db.migrate import MIGRATION_DIR


class MigrationDefinitionTests(unittest.TestCase):
    def test_migration_defines_source_fields_vector_dimension_and_indexes(self):
        migration = (MIGRATION_DIR / "0001_initial.sql").read_text(encoding="utf-8").lower()
        for field in ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label", "split"):
            with self.subTest(field=field):
                self.assertIn(field, migration)
        self.assertIn("vector(384)", migration)
        self.assertIn("using hnsw", migration)
        self.assertIn("vector_cosine_ops", migration)
        self.assertIn("validation", migration)
        self.assertIn("test", migration)
        self.assertIn("create table if not exists experiments", migration)

    def test_migration_is_rerunnable_and_contains_no_credentials(self):
        migration = (MIGRATION_DIR / "0001_initial.sql").read_text(encoding="utf-8").lower()
        self.assertGreaterEqual(migration.count("if not exists"), 7)
        for secret_word in ("password", "secret", "api_key", "token"):
            with self.subTest(secret_word=secret_word):
                self.assertNotIn(secret_word, migration)


if __name__ == "__main__":
    unittest.main()
