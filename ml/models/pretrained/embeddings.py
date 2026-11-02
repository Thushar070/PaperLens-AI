"""Incremental Sentence Transformer embedding generation for Phase 1 records."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ml.models.pretrained.checkpoint import update_checkpoint
from ml.models.pretrained.data import FIELDS, load_phase1_splits, paper_text

ROOT = Path(__file__).resolve().parents[3]
PHASE1 = ROOT / "artifacts" / "data" / "phase1"
OUTPUT = ROOT / "artifacts" / "data" / "phase7"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MODEL_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / MODEL_REVISION


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_ref(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _select_device() -> str:
    if torch.cuda.is_available():
        try:
            x = torch.ones((256, 256), device="cuda:0")
            _ = (x @ x).sum().item()
            torch.cuda.synchronize()
            del x
            return "cuda:0"
        except Exception as error:
            update_checkpoint("semantic_retrieval", {"status": "cuda_unusable_cpu_fallback", "cuda_probe_error": repr(error)})
    return "cpu"


def load_sentence_encoder(model_path: str | Path = MODEL_SNAPSHOT, device: str | None = None):
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Sentence Transformer model is not present locally: {path}")
    selected_device = device or _select_device()
    model = SentenceTransformer(str(path), device=selected_device, local_files_only=True, trust_remote_code=False)
    return model, selected_device


def encode_query(model, query: str) -> np.ndarray:
    if not isinstance(query, str):
        raise TypeError("Query must be a string")
    if not query.strip():
        return np.empty((0,), dtype=np.float32)
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)[0]
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.isfinite(vector).all() or not np.linalg.norm(vector) > 0:
        raise ValueError("Query encoder returned an invalid vector")
    return vector


def generate_embeddings(
    source: str | Path = PHASE1,
    output: str | Path = OUTPUT,
    *,
    model_path: str | Path = MODEL_SNAPSHOT,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    batch_size: int = 16,
    flush_every_batches: int = 8,
) -> dict:
    """Generate aligned float32 embeddings; safely resume from the last progress flush."""
    source, output = Path(source), Path(output)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    splits = load_phase1_splits(source, metadata["split_sizes"])
    rows = [row for split in ("train", "validation", "test") for row in splits[split]]
    ids = [row["arxiv_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Phase 1 manifests contain duplicate IDs")
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError("Embedding output must be separate from Phase 1 artifacts")
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "paper_embeddings.npy"
    index_path = output / "embedding_index.jsonl"
    progress_path = output / "embedding_progress.json"
    result_path = output / "embedding_metadata.json"
    if result_path.exists() and matrix_path.exists() and index_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        matrix = np.load(matrix_path, mmap_mode="r")
        indexed = sum(1 for line in index_path.open(encoding="utf-8") if line.strip())
        shape_ok = matrix.shape == (len(rows), existing.get("embedding_dimension")) and indexed == len(rows) and np.isfinite(matrix).all()
        hashes_ok = existing.get("embedding_matrix_sha256") == _sha256(matrix_path) and existing.get("index_sha256") == _sha256(index_path)
        if existing.get("status") == "complete":
            if not shape_ok or not hashes_ok:
                raise ValueError("Completed embedding artifacts failed shape, finiteness, count, or checksum verification")
            return existing
    model, device = load_sentence_encoder(model_path)
    dimension = int(model.get_sentence_embedding_dimension())
    max_length = int(model.max_seq_length)
    if dimension < 1 or max_length < 2:
        raise ValueError("Loaded Sentence Transformer has invalid dimension or token limit")
    if matrix_path.exists():
        matrix = np.load(matrix_path, mmap_mode="r+")
        if matrix.shape != (len(rows), dimension) or matrix.dtype != np.float32:
            raise ValueError("Existing embedding matrix shape/dtype does not match the selected model and dataset")
    else:
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
        matrix.flush()
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed_records": 0}
    completed = int(progress.get("completed_records", 0))
    if completed < 0 or completed > len(rows):
        raise ValueError("Embedding progress cursor is outside the dataset")
    if completed % batch_size:
        # Recompute an uncommitted partial batch. Progress is committed only at a batch boundary.
        completed -= completed % batch_size
    started = time.perf_counter()
    used_batch_sizes = list(progress.get("used_batch_sizes", []))
    batch_number = 0
    while completed < len(rows):
        current_batch = min(batch_size, len(rows) - completed)
        texts = [paper_text(row) for row in rows[completed:completed + current_batch]]
        try:
            encoded = model.encode(texts, batch_size=current_batch, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            if device == "cpu" or current_batch == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, current_batch // 2)
            used_batch_sizes.append(batch_size)
            continue
        values = np.asarray(encoded, dtype=np.float32)
        if values.shape != (current_batch, dimension) or not np.isfinite(values).all():
            raise ValueError(f"Invalid embedding batch shape or values: {values.shape}")
        matrix[completed:completed + current_batch] = values
        completed += current_batch
        batch_number += 1
        used_batch_sizes.append(current_batch)
        if batch_number % max(1, flush_every_batches) == 0 or completed == len(rows):
            matrix.flush()
            _write_json(progress_path, {"model_id": model_id, "model_revision": model_revision, "completed_records": completed, "total_records": len(rows), "used_batch_sizes": used_batch_sizes[-64:], "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            update_checkpoint("semantic_retrieval", {"status": "embedding_generation", "model_id": model_id, "model_revision": model_revision, "embedding_dimension": dimension, "max_sequence_length": max_length, "completed_records": completed, "total_records": len(rows), "progress_artifact": "artifacts/data/phase7/embedding_progress.json"})
    matrix.flush()
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("One or more paper embeddings are invalid or zero")
    with index_path.open("w", encoding="utf-8", newline="\n") as stream:
        for offset, row in enumerate(rows):
            record = {field: row[field] for field in FIELDS}
            record["embedding_offset"] = offset
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if len(rows) != len(ids) or len(set(ids)) != len(rows):
        raise ValueError("Embedding ID alignment validation failed")
    result = {
        "status": "complete",
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": model_id,
        "tokenizer_revision": model_revision,
        "license": "Apache-2.0",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda:0" else None,
        "embedding_dimension": dimension,
        "max_sequence_length": max_length,
        "paper_count": len(rows),
        "split_order": ["train", "validation", "test"],
        "split_sizes": metadata["split_sizes"],
        "metadata_fields_preserved": list(FIELDS),
        "embedding_dtype": "float32",
        "normalization": "The local Sentence Transformer pipeline includes its Normalize module; raw encode output norms are measured below.",
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_max": float(norms.max()),
        "generation_seconds": time.perf_counter() - started,
        "batch_size_requested": batch_size,
        "batch_sizes_used_tail": used_batch_sizes[-64:],
        "gpu_memory_bytes_after_generation": list(torch.cuda.mem_get_info(0)) if device == "cuda:0" else None,
        "embedding_matrix_path": _artifact_ref(matrix_path),
        "embedding_matrix_shape": list(matrix.shape),
        "embedding_matrix_dtype": str(matrix.dtype),
        "embedding_matrix_bytes": matrix_path.stat().st_size,
        "embedding_matrix_sha256": _sha256(matrix_path),
        "index_path": _artifact_ref(index_path),
        "index_bytes": index_path.stat().st_size,
        "index_sha256": _sha256(index_path),
        "progress_path": _artifact_ref(progress_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(result_path, result)
    update_checkpoint("semantic_retrieval", {"status": "embeddings_complete", "model_id": model_id, "model_revision": model_revision, "tokenizer_id": model_id, "tokenizer_revision": model_revision, "license": "Apache-2.0", "embedding_dimension": dimension, "max_sequence_length": max_length, "paper_count": len(rows), "device": device, "generation_seconds": result["generation_seconds"], "normalization": result["normalization"], "norm_range": [result["embedding_norm_min"], result["embedding_norm_max"]], "artifact_paths": {"embeddings": result["embedding_matrix_path"], "index": result["index_path"], "metadata": "artifacts/data/phase7/embedding_metadata.json"}, "retrieval_evaluation": "not_started"})
    return result


if __name__ == "__main__":
    print(json.dumps(generate_embeddings(), ensure_ascii=False, indent=2))
