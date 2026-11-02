"""Phase 7 Sentence Transformer retrieval, kept independent of Phase 4 methods."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

from ml.models.pretrained.data import FIELDS, validate_record

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TransformerRetriever:
    def __init__(self, records: list[Mapping[str, Any]], embeddings: np.ndarray, encoder):
        self.records = [validate_record(row) for row in records]
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(self.records) or vectors.shape[1] < 1:
            raise ValueError("Embedding matrix must be 2D and row-aligned with records")
        identifiers = [row["arxiv_id"] for row in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Paper IDs must be unique")
        self.encoder = encoder
        self.dimension = vectors.shape[1]
        norms = np.linalg.norm(vectors, axis=1)
        finite = np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
        self.valid_rows = finite
        self.embeddings = np.zeros_like(vectors, dtype=np.float32)
        self.embeddings[finite] = vectors[finite] / norms[finite, None]

    @classmethod
    def from_artifacts(cls, root: str | Path = "artifacts/data/phase7", model_path: str | Path | None = None, device: str | None = None):
        from ml.models.pretrained.embeddings import MODEL_SNAPSHOT, load_sentence_encoder

        root = Path(root)
        index_path = root / "embedding_index.jsonl"
        metadata_path = root / "embedding_metadata.json"
        matrix_path = root / "paper_embeddings.npy"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise ValueError("Embedding artifacts are not marked complete")
        if _sha256(matrix_path) != metadata.get("embedding_matrix_sha256") or _sha256(index_path) != metadata.get("index_sha256"):
            raise ValueError("Embedding matrix or metadata index failed checksum verification")
        records = []
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    records.append({field: value[field] for field in FIELDS})
        matrix = np.load(matrix_path, mmap_mode="r")
        if len(records) != metadata.get("paper_count") or matrix.shape != tuple(metadata.get("embedding_matrix_shape", ())):
            raise ValueError("Embedding artifact counts or shape disagree with metadata")
        path = model_path or MODEL_SNAPSHOT
        model, _ = load_sentence_encoder(path, device=device)
        return cls(records, matrix, model)

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        if not query.strip():
            return np.empty((0,), dtype=np.float32)
        vector = np.asarray(
            self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)[0],
            dtype=np.float32,
        ).reshape(-1)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Query embedding has invalid dimensions or values")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return np.empty((0,), dtype=np.float32)
        return vector / norm

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if not query.strip() or k == 0:
            return []
        query_vector = self.embed_query(query)
        if query_vector.size == 0:
            return []
        scores = self.embeddings @ query_vector
        scores[~self.valid_rows] = np.nan
        scores = np.clip(scores, -1.0, 1.0)
        ranked = [index for index, score in enumerate(scores) if np.isfinite(score)]
        ranked.sort(key=lambda index: (-float(scores[index]), self.records[index]["arxiv_id"]))
        results = []
        for index in ranked[:k]:
            row = {field: self.records[index][field] for field in FIELDS}
            row["score"] = float(scores[index])
            results.append(row)
        return results
