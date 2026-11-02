from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.db.repository import DatabaseRepository


class SearchService:
    def __init__(self, repository: DatabaseRepository, artifact_root: Path) -> None:
        self.repository = repository
        self.artifact_root = artifact_root
        self._tfidf_retriever = None
        self._encoder = None

    def _load_tfidf(self):
        if self._tfidf_retriever is None:
            from ml.retrieval.engine import Retriever

            self._tfidf_retriever = Retriever(root=self.artifact_root)
        return self._tfidf_retriever

    def _load_encoder(self):
        if self._encoder is None:
            from ml.models.pretrained.embeddings import load_sentence_encoder

            encoder, _device = load_sentence_encoder()
            self._encoder = encoder
        return self._encoder

    def search(self, query: str, method: str, k: int) -> list[dict[str, Any]]:
        if method == "tfidf":
            ranked = self._load_tfidf().search(query, method="tfidf", k=k)
            stored = self.repository.get_papers_by_ids([row["arxiv_id"] for row in ranked])
            return [
                {**stored[row["arxiv_id"]], "score": float(row["score"])}
                for row in ranked
                if row["arxiv_id"] in stored
            ]
        if method != "minilm":
            raise ValueError(f"Unsupported search method: {method}")
        vector = np.asarray(
            self._load_encoder().encode(
                [query], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
            )[0],
            dtype=np.float32,
        ).reshape(-1)
        if vector.shape != (384,) or not np.isfinite(vector).all():
            raise ValueError("Cached Phase 7 encoder returned an invalid 384-dimensional query vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return []
        return self.repository.vector_search(vector / norm, k=k)
