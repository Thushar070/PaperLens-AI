import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.db.ingest import IngestionSources, iter_aligned_rows, load_sources


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "data"


class IngestionSourceTests(unittest.TestCase):
    def test_existing_phase1_rows_and_phase7_embeddings_align_exactly(self):
        sources = load_sources(ARTIFACTS)
        self.assertEqual(sources.expected_count, 19999)
        self.assertEqual(sources.embeddings.shape, (19999, 384))
        counts = {"train": 0, "validation": 0, "test": 0}
        labels = {0: 0, 1: 0, 2: 0, 3: 0}
        first = None
        for record, split, vector in iter_aligned_rows(sources):
            counts[split] += 1
            labels[record["label"]] += 1
            self.assertEqual(set(record), {"arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label"})
            self.assertEqual(vector.shape, (384,))
            if first is None:
                first = record
        self.assertEqual(counts, {"train": 15999, "validation": 2000, "test": 2000})
        self.assertEqual(labels, {0: 4999, 1: 5000, 2: 5000, 3: 5000})
        self.assertEqual(first["arxiv_id"], "2603.08709")

    def test_offset_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.jsonl"
            record = {
                "arxiv_id": "x", "title": "t", "abstract": "a",
                "primary_cat": "cs.AI", "all_cats": "cs.AI", "label": 1,
                "embedding_offset": 4,
            }
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            sources = IngestionSources(
                Path(directory), {"x": ({key: record[key] for key in ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")}, "train")},
                np.ones((1, 384), dtype=np.float32), index, 1,
            )
            with self.assertRaisesRegex(ValueError, "offset mismatch"):
                list(iter_aligned_rows(sources))

    def test_metadata_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.jsonl"
            record = {
                "arxiv_id": "x", "title": "changed", "abstract": "a",
                "primary_cat": "cs.AI", "all_cats": "cs.AI", "label": 1,
                "embedding_offset": 0,
            }
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            source_record = {**record, "title": "source title"}
            source_record.pop("embedding_offset")
            sources = IngestionSources(Path(directory), {"x": (source_record, "train")}, np.ones((1, 384), dtype=np.float32), index, 1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                list(iter_aligned_rows(sources))


if __name__ == "__main__":
    unittest.main()
