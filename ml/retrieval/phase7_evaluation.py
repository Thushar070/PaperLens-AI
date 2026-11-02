"""Compare transformer retrieval against the existing Phase 4 judgments."""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from ml.retrieval.evaluation import load_judgments
from ml.retrieval.metrics import evaluate
from ml.retrieval.transformer import TransformerRetriever
from ml.models.pretrained.checkpoint import CHECKPOINT, update_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def evaluate_transformer_retrieval(retriever: TransformerRetriever | None = None, k: int = 5) -> dict:
    retriever = retriever or TransformerRetriever.from_artifacts(ROOT / "artifacts" / "data" / "phase7")
    phase4 = ROOT / "artifacts" / "data" / "phase4"
    queries = json.loads((phase4 / "queries.json").read_text(encoding="utf-8"))
    judgment_data = load_judgments(phase4 / "judgments.json")
    all_rankings = {}
    per_query = {}
    latencies = []
    embedding_metadata = json.loads((ROOT / "artifacts" / "data" / "phase7" / "embedding_metadata.json").read_text(encoding="utf-8"))
    query_device = str(getattr(retriever.encoder, "device", embedding_metadata.get("device", "unknown")))
    # Exclude one warm-up query from latency measurements; it adds no judgment.
    retriever.search("transformer retrieval latency warmup", k=1)
    for query_id, query_text in queries.items():
        started = time.perf_counter()
        results = retriever.search(query_text, k=max(k, 10))
        latencies.append((time.perf_counter() - started) * 1000)
        all_rankings[query_id] = results
        # Phase 4 MRR used the same top-five candidate lists as its P@5, Recall@5,
        # and nDCG@5. Use that identical cutoff for fair comparison; top-10 rows
        # are retained in the separate rankings artifact for qualitative review.
        per_query[query_id] = evaluate([row["arxiv_id"] for row in results[:k]], judgment_data["judgments"][query_id], k)
    metric_names = ("precision", "recall", "mrr", "ndcg", "unjudged_in_top_k")
    average = {name: statistics.mean(row[name] for row in per_query.values()) for name in metric_names}
    phase4_results = json.loads((phase4 / "evaluation.json").read_text(encoding="utf-8"))
    result = {
        "k": k,
        "query_count": len(queries),
        "judgment_assessor": judgment_data["assessor"],
        "judgment_method": judgment_data["method"],
        "per_query": per_query,
        "macro_average": average,
        "query_latency_ms": {"per_query": {query: value for query, value in zip(queries, latencies)}, "mean": statistics.mean(latencies), "median": statistics.median(latencies), "p95": float(np_percentile(latencies, 95)), "includes_query_encoding_and_ranking": True},
        "embedding_dimension": retriever.dimension,
        "similarity": "cosine similarity over L2-normalized vectors",
        "paper_count": len(retriever.records),
        "query_device": query_device,
        "previous_phase4_macro_average": phase4_results["macro_average"],
        "comparison_condition": "Same three queries, same Phase 4 candidate judgments, and top-5 cutoff for Precision, Recall, MRR, and nDCG, matching the Phase 4 candidate-list evaluation. Top-10 rankings are saved for inspection. Unjudged results are treated as nonrelevant by the inherited metric function.",
        "limitations": ["Only three queries are available.", "Phase 4 relevance labels were assistant-assessed by reading titles and abstract excerpts and were not human-reviewed.", "The judgment pool was built from the earlier TF-IDF and Word2Vec top-five lists; transformer-only results can be unjudged.", "Unjudged results are scored as nonrelevant, so these metrics do not estimate complete-corpus relevance."],
        "rankings_path": "artifacts/data/phase7/retrieval_rankings.json",
    }
    output = ROOT / "artifacts" / "data" / "phase7"
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval_rankings.json").write_text(json.dumps(all_rankings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "retrieval_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    retrieval_state = dict(checkpoint_data.get("scope", {}).get("semantic_retrieval", {}))
    retrieval_state.update({"status": "complete", "retrieval_method": "Sentence Transformer embeddings + cosine similarity", "k": k, "query_count": len(queries), "macro_average": average, "query_latency_ms": result["query_latency_ms"], "limitations": result["limitations"], "artifact_paths": {**retrieval_state.get("artifact_paths", {}), "rankings": result["rankings_path"], "evaluation": "artifacts/data/phase7/retrieval_evaluation.json"}})
    update_checkpoint("semantic_retrieval", retrieval_state)
    return result


def np_percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
    print(json.dumps(evaluate_transformer_retrieval(), ensure_ascii=False, indent=2))
