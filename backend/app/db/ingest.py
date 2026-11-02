from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from sqlalchemy import text

from app.config import Settings
from app.db.repository import DatabaseRepository
from ml.models.pretrained.data import FIELDS, LABEL_NAMES, load_phase1_splits
import numpy as np


@dataclass(frozen=True)
class IngestionSources:
    artifact_root: Path
    rows_by_id: dict[str, tuple[dict, str]]
    embeddings: np.ndarray
    index_path: Path
    expected_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources(artifact_root: str | Path) -> IngestionSources:
    root = Path(artifact_root)
    phase1 = root / "phase1"
    phase7 = root / "phase7"
    phase1_meta = json.loads((phase1 / "metadata.json").read_text(encoding="utf-8"))
    expected_mapping = {str(index): name for index, name in enumerate(LABEL_NAMES)}
    if phase1_meta.get("label_mapping") != expected_mapping:
        raise ValueError("Phase 1 label mapping differs from the verified original mapping")
    expected_splits = phase1_meta["split_sizes"]
    split_rows = load_phase1_splits(phase1, expected=expected_splits)
    rows_by_id = {
        row["arxiv_id"]: (row, split)
        for split, rows in split_rows.items()
        for row in rows
    }
    if len(rows_by_id) != int(phase1_meta["selected_records"]):
        raise ValueError("Phase 1 split records disagree with selected_records metadata")

    metadata = json.loads((phase7 / "embedding_metadata.json").read_text(encoding="utf-8"))
    index_path = phase7 / "embedding_index.jsonl"
    matrix_path = phase7 / "paper_embeddings.npy"
    if metadata.get("status") != "complete":
        raise ValueError("Phase 7 embedding artifacts are not marked complete")
    if _sha256(index_path) != metadata.get("index_sha256"):
        raise ValueError("Phase 7 embedding index checksum does not match its metadata")
    if _sha256(matrix_path) != metadata.get("embedding_matrix_sha256"):
        raise ValueError("Phase 7 embedding matrix checksum does not match its metadata")
    embeddings = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    expected_count = int(metadata["paper_count"])
    expected_shape = tuple(metadata["embedding_matrix_shape"])
    if embeddings.shape != expected_shape or embeddings.shape != (expected_count, 384):
        raise ValueError("Phase 7 embedding matrix shape is not the verified 384-dimensional shape")
    if embeddings.dtype != np.float32 or expected_count != len(rows_by_id):
        raise ValueError("Phase 1 records and Phase 7 float32 embeddings are not row-aligned")
    return IngestionSources(root, rows_by_id, embeddings, index_path, expected_count)


def iter_aligned_rows(sources: IngestionSources) -> Iterator[tuple[dict, str, np.ndarray]]:
    seen: set[str] = set()
    with sources.index_path.open(encoding="utf-8") as stream:
        for offset, line in enumerate(stream):
            if not line.strip():
                continue
            index_row = json.loads(line)
            if int(index_row.get("embedding_offset", -1)) != offset:
                raise ValueError(f"Embedding offset mismatch at index row {offset}")
            arxiv_id = index_row.get("arxiv_id")
            if arxiv_id in seen or arxiv_id not in sources.rows_by_id:
                raise ValueError(f"Duplicate or unknown Phase 7 paper identifier: {arxiv_id}")
            record, split = sources.rows_by_id[arxiv_id]
            if any(index_row.get(field) != record[field] for field in FIELDS):
                raise ValueError(f"Phase 7 metadata does not match Phase 1 record {arxiv_id}")
            vector = sources.embeddings[offset]
            if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) == 0.0:
                raise ValueError(f"Invalid saved embedding for {arxiv_id}")
            seen.add(arxiv_id)
            yield record, split, vector
    if len(seen) != sources.expected_count:
        raise ValueError(f"Embedding index contains {len(seen)} rows; expected {sources.expected_count}")


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(format(float(value), ".8g") for value in vector) + "]"


def ingest(artifact_root: str | Path | None = None, database_url: str | None = None, batch_size: int = 200) -> dict:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    settings = Settings.from_environment()
    root = Path(artifact_root) if artifact_root else settings.artifact_root
    sources = load_sources(root)
    repository = DatabaseRepository(database_url or settings.database_url)
    engine = repository.engine
    papers_sql = text(
        "INSERT INTO papers(arxiv_id,title,abstract,primary_cat,all_cats,label,split) "
        "VALUES (:arxiv_id,:title,:abstract,:primary_cat,:all_cats,:label,:split) "
        "ON CONFLICT (arxiv_id) DO UPDATE SET title=EXCLUDED.title, abstract=EXCLUDED.abstract, "
        "primary_cat=EXCLUDED.primary_cat, all_cats=EXCLUDED.all_cats, label=EXCLUDED.label, split=EXCLUDED.split"
    )
    embeddings_sql = text(
        "INSERT INTO paper_embeddings(arxiv_id,embedding) VALUES (:arxiv_id,CAST(:embedding AS vector)) "
        "ON CONFLICT (arxiv_id) DO UPDATE SET embedding=EXCLUDED.embedding"
    )
    paper_batch: list[dict] = []
    embedding_batch: list[dict] = []
    try:
        count = 0
        with engine.begin() as connection:
            for record, split, vector in iter_aligned_rows(sources):
                paper_batch.append({**record, "split": split})
                embedding_batch.append({"arxiv_id": record["arxiv_id"], "embedding": _vector_literal(vector)})
                if len(paper_batch) >= batch_size:
                    connection.execute(papers_sql, paper_batch)
                    connection.execute(embeddings_sql, embedding_batch)
                    count += len(paper_batch)
                    paper_batch.clear()
                    embedding_batch.clear()
            if paper_batch:
                connection.execute(papers_sql, paper_batch)
                connection.execute(embeddings_sql, embedding_batch)
                count += len(paper_batch)
        phase8 = root / "phase8"
        classification = json.loads((phase8 / "classification_results.json").read_text(encoding="utf-8"))
        retrieval = json.loads((phase8 / "retrieval_results.json").read_text(encoding="utf-8"))
        experiment_sql = text(
            "INSERT INTO experiments(experiment_id,kind,phase,display_name,family,validation_metrics,test_metrics,source_artifact) "
            "VALUES (:experiment_id,:kind,:phase,:display_name,:family,CAST(:validation_metrics AS jsonb),CAST(:test_metrics AS jsonb),:source_artifact) "
            "ON CONFLICT (experiment_id) DO UPDATE SET kind=EXCLUDED.kind, phase=EXCLUDED.phase, "
            "display_name=EXCLUDED.display_name, family=EXCLUDED.family, validation_metrics=EXCLUDED.validation_metrics, "
            "test_metrics=EXCLUDED.test_metrics, source_artifact=EXCLUDED.source_artifact"
        )
        experiment_rows = []
        for model in classification["models"]:
            experiment_rows.append({
                "experiment_id": "classifier:" + model["model_id"],
                "kind": "classifier",
                "phase": int(model["phase"]),
                "display_name": model["short_name"],
                "family": model["family"],
                "validation_metrics": json.dumps(model.get("validation_metrics") or {}),
                "test_metrics": json.dumps(model.get("test_metrics") or {}),
                "source_artifact": "artifacts/data/phase8/classification_results.json",
            })
        for method_id, method in retrieval["methods"].items():
            experiment_rows.append({
                "experiment_id": "retrieval:" + method_id,
                "kind": "retrieval",
                "phase": int(method.get("phase", 4 if method_id.startswith("phase4") else 7)),
                "display_name": method.get("display_name", method_id),
                "family": "retrieval",
                "validation_metrics": "{}",
                "test_metrics": json.dumps(method["macro_average"]),
                "source_artifact": "artifacts/data/phase8/retrieval_results.json",
            })
        with engine.begin() as connection:
            connection.execute(experiment_sql, experiment_rows)
        return {
            "papers_upserted": count,
            "embeddings_upserted": count,
            "experiments_upserted": len(experiment_rows),
            "label_names": list(LABEL_NAMES),
        }
    finally:
        engine.dispose()


if __name__ == "__main__":
    print(ingest())
