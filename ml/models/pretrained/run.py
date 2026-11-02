"""Run Phase 7 classification and embedding/retrieval objectives locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.models.pretrained.checkpoint import update_checkpoint
from ml.models.pretrained.embeddings import generate_embeddings
from ml.models.pretrained.train import run_classification_experiments

ROOT = Path(__file__).resolve().parents[3]


def run_phase7(classification: bool = True, retrieval: bool = True) -> dict:
    result = {}
    if classification:
        result["classification"] = run_classification_experiments()
    if retrieval:
        embeddings_path = ROOT / "artifacts" / "data" / "phase7" / "embedding_metadata.json"
        if embeddings_path.exists():
            current = json.loads(embeddings_path.read_text(encoding="utf-8"))
            if current.get("status") == "complete":
                result["embeddings"] = current
            else:
                update_checkpoint("semantic_retrieval", {"status": "resuming_embeddings", "prior_completed_records": current.get("paper_count", 0)})
                result["embeddings"] = generate_embeddings()
        else:
            update_checkpoint("semantic_retrieval", {"status": "starting_embedding_generation", "model_id": "sentence-transformers/all-MiniLM-L6-v2", "model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"})
            result["embeddings"] = generate_embeddings()
        retrieval_path = ROOT / "artifacts" / "data" / "phase7" / "retrieval_evaluation.json"
        if retrieval_path.exists():
            existing = json.loads(retrieval_path.read_text(encoding="utf-8"))
            if existing.get("query_count") == 3:
                result["retrieval"] = existing
            else:
                from ml.retrieval.phase7_evaluation import evaluate_transformer_retrieval
                result["retrieval"] = evaluate_transformer_retrieval()
        else:
            from ml.retrieval.phase7_evaluation import evaluate_transformer_retrieval
            result["retrieval"] = evaluate_transformer_retrieval()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if args.classification_only and args.retrieval_only:
        parser.error("Choose at most one objective-only flag")
    result = run_phase7(classification=not args.retrieval_only, retrieval=not args.classification_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
