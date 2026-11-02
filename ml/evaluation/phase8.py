"""Build Phase 8 comparisons from immutable Phase 1–7 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from ml.evaluation.charts import generate_charts
from ml.evaluation.metrics import LABEL_IDS, LABEL_NAMES, classification_metrics, validate_experiment_record
from ml.retrieval.metrics import evaluate as evaluate_retrieval


ROOT = Path(__file__).resolve().parents[2]
PHASE8_DATA = ROOT / "artifacts" / "data" / "phase8"
PHASE8_PLOTS = PHASE8_DATA / "plots"
PHASE8_DOCS = ROOT / "docs"
SOURCE_FIELDS = ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")
TOLERANCE = 1e-8


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_info(paths: list[Path]) -> dict[str, dict]:
    result = {}
    for path in sorted(set(paths)):
        result[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _assert_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        raise ValueError(f"Saved metric disagreement for {context}: recomputed={actual}, saved={expected}")


def _prediction_rows(path: Path) -> list[dict]:
    return _jsonl(path) if path.suffix == ".jsonl" else _json(path)


def _validate_alignment(rows: list[dict], manifest: list[dict], context: str, phase7: bool = False) -> tuple[list[int], list[int]]:
    ids = [row.get("arxiv_id") for row in rows]
    expected_ids = [row["arxiv_id"] for row in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate arxiv_id values")
    if ids != expected_ids:
        raise ValueError(f"{context} IDs or ordering differ from the Phase 1 split manifest")
    if phase7:
        actual = [row.get("actual_label") for row in rows]
        predicted = [row.get("predicted_label") for row in rows]
    else:
        actual = [row.get("actual") for row in rows]
        predicted = [row.get("predicted") for row in rows]
    expected_actual = [row["label"] for row in manifest]
    if actual != expected_actual:
        raise ValueError(f"{context} actual labels differ from the Phase 1 split manifest")
    if any(type(label) is not int or label not in LABEL_IDS for label in predicted):
        raise ValueError(f"{context} has an invalid predicted label")
    return actual, predicted


def _manifest_and_integrity() -> tuple[dict[str, list[dict]], dict]:
    phase1 = ROOT / "artifacts" / "data" / "phase1"
    splits = {name: _jsonl(phase1 / f"{name}.jsonl") for name in ("train", "validation", "test")}
    ids_by_split = {name: [row["arxiv_id"] for row in rows] for name, rows in splits.items()}
    for name, ids in ids_by_split.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate paper IDs in Phase 1 {name} split")
        for row in splits[name]:
            missing = set(SOURCE_FIELDS) - set(row)
            if missing:
                raise ValueError(f"Phase 1 {name} record missing source fields: {sorted(missing)}")
            if type(row["label"]) is not int or row["label"] not in LABEL_IDS:
                raise ValueError(f"Phase 1 {name} record has invalid label")
    overlap = {
        f"{left}_{right}": sorted(set(ids_by_split[left]) & set(ids_by_split[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"Phase 1 split leakage detected: { {k: len(v) for k, v in overlap.items()} }")
    metadata = _json(phase1 / "metadata.json")
    mapping = {int(key): value for key, value in metadata["label_mapping"].items()}
    if mapping != LABEL_NAMES:
        raise ValueError(f"Phase 1 label mapping changed unexpectedly: {mapping}")
    expected_sizes = {"train": 15999, "validation": 2000, "test": 2000}
    observed_sizes = {name: len(rows) for name, rows in splits.items()}
    if observed_sizes != expected_sizes:
        raise ValueError(f"Phase 1 split sizes differ from recorded split: {observed_sizes}")
    return splits, {
        "source": metadata.get("source"),
        "revision": metadata.get("revision"),
        "split_sizes": observed_sizes,
        "label_order": list(LABEL_IDS),
        "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "split_ids_unique": True,
        "split_ids_disjoint": True,
        "split_overlap_counts": {key: len(value) for key, value in overlap.items()},
        "test_support": {str(label): sum(row["label"] == label for row in splits["test"]) for label in LABEL_IDS},
    }


def _metric_summary(metrics: dict) -> dict:
    return {key: metrics[key] for key in (
        "records", "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "per_class", "confusion_matrix",
    )}


def _validate_saved_metric(summary: dict, saved: dict, model_id: str, split: str) -> None:
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        if key in saved:
            _assert_close(summary[key], saved[key], f"{model_id} {split} {key}")
    for field in ("macro_precision", "macro_recall", "weighted_precision", "weighted_recall"):
        if field in saved:
            _assert_close(summary[field], saved[field], f"{model_id} {split} {field}")
    matrix = saved.get("confusion_matrix")
    if matrix is not None and matrix != summary["confusion_matrix"]:
        raise ValueError(f"Saved confusion matrix disagrees for {model_id} {split}")


def _learned_parameter_count(model: Any) -> int | None:
    """Count learned numeric classifier coefficients from the serialized estimator."""
    for attribute in ("coef_", "feature_log_prob_"):
        value = getattr(model, attribute, None)
        if value is not None:
            count = int(np.asarray(value).size)
            bias = getattr(model, "intercept_", getattr(model, "class_log_prior_", None))
            if bias is not None:
                count += int(np.asarray(bias).size)
            return count
    return None


def _short_name(phase: str, key: str) -> str:
    return {
        ("P3", "bow_logistic"): "P3 BoW LR",
        ("P3", "bow_nb"): "P3 BoW NB",
        ("P3", "tfidf_logistic"): "P3 TF-IDF LR",
        ("P3", "tfidf_nb"): "P3 TF-IDF NB",
        ("P3", "tfidf_svm"): "P3 TF-IDF SVM",
        ("P3", "word2vec_logistic"): "P3 W2V LR",
        ("P5", "rnn"): "P5 RNN",
        ("P5", "lstm"): "P5 LSTM",
        ("P5", "gru"): "P5 GRU",
        ("P5", "bilstm"): "P5 BiLSTM",
        ("P5", "bilstm_attention"): "P5 BiLSTM Attn",
        ("P6", "custom_transformer"): "P6 Custom Transformer",
        ("P7", "minilm_finetuned"): "P7 MiniLM fine-tuned",
    }[(phase, key)]


def build_classification_results(splits: dict[str, list[dict]], integrity: dict) -> tuple[dict, dict, list[Path]]:
    test_rows, validation_rows = splits["test"], splits["validation"]
    registry: list[dict] = []
    internal_predictions: dict[str, list[dict]] = {}
    source_paths: list[Path] = [ROOT / "artifacts/data/phase1/metadata.json"]
    source_paths.extend(ROOT / f"artifacts/data/phase1/{name}.jsonl" for name in splits)

    p3 = ROOT / "artifacts/data/phase3"
    p3_selection = _json(p3 / "model_selection.json")
    source_paths.append(p3 / "model_selection.json")
    for prediction_path in sorted((p3 / "test").glob("*_predictions.json")):
        key = prediction_path.stem.removesuffix("_predictions")
        metric_path = p3 / "test" / f"{key}.json"
        validation_path = p3 / "validation" / f"{key}_predictions.json"
        validation_metric_path = p3 / "validation" / f"{key}.json"
        model_path = p3 / "models" / f"{key}.joblib"
        if not all(path.exists() for path in (metric_path, validation_path, validation_metric_path, model_path)):
            raise FileNotFoundError(f"Incomplete Phase 3 experiment artifacts for {key}")
        rows = _prediction_rows(prediction_path)
        y_true, y_pred = _validate_alignment(rows, test_rows, f"P3/{key}")
        metrics = classification_metrics(y_true, y_pred)
        saved_test = _json(metric_path)
        _validate_saved_metric(metrics, saved_test, f"P3/{key}", "test")
        val_rows = _prediction_rows(validation_path)
        val_true, val_pred = _validate_alignment(val_rows, validation_rows, f"P3/{key} validation")
        val_metrics = classification_metrics(val_true, val_pred)
        val_saved = _json(validation_metric_path)
        _validate_saved_metric(val_metrics, val_saved, f"P3/{key}", "validation")
        model = joblib.load(model_path)
        config = {"representation": {"bow": "bag-of-words count vectors", "tfidf": "TF-IDF vectors", "word2vec": "Word2Vec document vectors"}[key.split("_", 1)[0]], "estimator_class": type(model).__name__, "estimator_parameters": model.get_params(deep=False)}
        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is None and getattr(model, "coef_", None) is not None:
            feature_count = int(np.asarray(model.coef_).shape[-1])
        config["observed_classifier_input_dimensions"] = int(feature_count) if feature_count is not None else None
        model_id = f"P3/{key}"
        is_selected = p3_selection.get("selected") == key
        record = {
            "model_id": model_id,
            "short_name": _short_name("P3", key),
            "phase": 3,
            "family": "classical machine learning",
            "status": "complete",
            "phase_selected_model_id": f"P3/{p3_selection.get('selected')}",
            "selected_by_validation": is_selected,
            "selection_metric": "validation macro F1",
            "validation_metrics": _metric_summary(val_metrics),
            "test_metrics": _metric_summary(metrics),
            "configuration": config,
            "sources": {
                "test_predictions": prediction_path.relative_to(ROOT).as_posix(),
                "test_metrics": metric_path.relative_to(ROOT).as_posix(),
                "validation_predictions": validation_path.relative_to(ROOT).as_posix(),
                "model_artifact": model_path.relative_to(ROOT).as_posix(),
            },
            "saved_test_metric_check": "passed",
        }
        registry.append(record)
        internal_predictions[model_id] = rows
        source_paths.extend((prediction_path, metric_path, validation_path, validation_metric_path, model_path))
    source_paths.extend((ROOT / "artifacts/data/phase2/bow.joblib", ROOT / "artifacts/data/phase2/tfidf.joblib", ROOT / "artifacts/data/phase2/word2vec.model"))

    p5 = ROOT / "artifacts/data/phase5"
    p5_selection = _json(p5 / "selection.json")
    source_paths.append(p5 / "selection.json")
    p2_metadata = _json(ROOT / "artifacts/data/phase2/metadata.json")
    phase5_sequence_length = int(p2_metadata["sequence_length"])
    p5_selected = max(p5_selection, key=lambda key: p5_selection[key]["best_validation_macro_f1"])
    for prediction_path in sorted((p5 / "test").glob("*_predictions.json")):
        key = prediction_path.stem.removesuffix("_predictions")
        metric_path = p5 / "test" / f"{key}.json"
        validation_path = p5 / "validation" / f"{key}_predictions.json"
        validation_metric_path = p5 / "validation" / f"{key}.json"
        if not all(path.exists() for path in (metric_path, validation_path, validation_metric_path)):
            raise FileNotFoundError(f"Incomplete Phase 5 experiment artifacts for {key}")
        rows = _prediction_rows(prediction_path)
        y_true, y_pred = _validate_alignment(rows, test_rows, f"P5/{key}")
        metrics = classification_metrics(y_true, y_pred)
        saved_test = _json(metric_path)
        _validate_saved_metric(metrics, saved_test, f"P5/{key}", "test")
        val_rows = _prediction_rows(validation_path)
        val_true, val_pred = _validate_alignment(val_rows, validation_rows, f"P5/{key} validation")
        val_metrics = classification_metrics(val_true, val_pred)
        _validate_saved_metric(val_metrics, _json(validation_metric_path), f"P5/{key}", "validation")
        model_id = f"P5/{key}"
        record = {
            "model_id": model_id,
            "short_name": _short_name("P5", key),
            "phase": 5,
            "family": "recurrent neural network",
            "status": "complete",
            "phase_selected_model_id": None,
            "phase_best_validation_model_id": f"P5/{p5_selected}",
            "best_validation_in_phase": key == p5_selected,
            "selected_by_validation": True,
            "selection_metric": "validation macro F1 selected the learning-rate checkpoint within this architecture; Phase 5 did not record one architecture-wide selection",
            "validation_metrics": _metric_summary(val_metrics),
            "test_metrics": _metric_summary(metrics),
            "configuration": {**p5_selection[key]["config"], "input_limit_tokens": phase5_sequence_length, "input_limit_source": "artifacts/data/phase2/metadata.json"},
            "sources": {
                "test_predictions": prediction_path.relative_to(ROOT).as_posix(),
                "test_metrics": metric_path.relative_to(ROOT).as_posix(),
                "validation_predictions": validation_path.relative_to(ROOT).as_posix(),
                "selection": "artifacts/data/phase5/selection.json",
            },
            "saved_test_metric_check": "passed",
        }
        registry.append(record)
        internal_predictions[model_id] = rows
        source_paths.extend((prediction_path, metric_path, validation_path, validation_metric_path))
    source_paths.extend((ROOT / "artifacts/data/phase2/metadata.json", p5 / "verification.json"))

    p6 = ROOT / "artifacts/data/phase6"
    p6_selection = _json(p6 / "selection.json")
    p6_comparison = _json(p6 / "comparison.json")
    p6_selected = p6_selection["selected"]
    p6_prediction_path = p6 / "test/custom_transformer_predictions.json"
    p6_metric_path = p6 / "test/custom_transformer.json"
    p6_validation_path = p6 / "validation/custom_transformer_predictions.json"
    p6_val_metric_path = p6 / "validation/custom_transformer.json"
    rows = _prediction_rows(p6_prediction_path)
    y_true, y_pred = _validate_alignment(rows, test_rows, "P6/custom_transformer")
    metrics = classification_metrics(y_true, y_pred)
    saved_test = _json(p6_metric_path)
    _validate_saved_metric(metrics, saved_test, "P6/custom_transformer", "test")
    val_rows = _prediction_rows(p6_validation_path)
    val_true, val_pred = _validate_alignment(val_rows, validation_rows, "P6/custom_transformer validation")
    val_metrics = classification_metrics(val_true, val_pred)
    _validate_saved_metric(val_metrics, _json(p6_val_metric_path), "P6/custom_transformer", "validation")
    p6_best = p6_selected["name"]
    p6_id = "P6/custom_transformer"
    registry.append({
        "model_id": p6_id,
        "short_name": _short_name("P6", "custom_transformer"),
        "phase": 6,
        "family": "custom Transformer",
        "status": "complete",
        "phase_selected_model_id": p6_id,
        "selected_by_validation": p6_best == "custom_transformer_lr_0.001",
        "selection_metric": p6_selection["selection_metric"],
        "validation_metrics": _metric_summary(val_metrics),
        "test_metrics": _metric_summary(metrics),
        "configuration": p6_selected["config"],
        "sources": {
            "test_predictions": p6_prediction_path.relative_to(ROOT).as_posix(),
            "test_metrics": p6_metric_path.relative_to(ROOT).as_posix(),
            "validation_predictions": p6_validation_path.relative_to(ROOT).as_posix(),
            "selection": "artifacts/data/phase6/selection.json",
        },
        "saved_test_metric_check": "passed",
    })
    internal_predictions[p6_id] = rows
    source_paths.extend((p6_prediction_path, p6_metric_path, p6_validation_path, p6_val_metric_path, p6 / "selection.json", p6 / "comparison.json", p6 / "runs/custom_transformer_lr_0.001/best.pt"))

    p7 = ROOT / "artifacts/models/phase7"
    p7_results = _json(p7 / "classification_results.json")
    p7_experiments = _json(p7 / "classification_experiments.json")
    p7_selected = _json(p7 / "experiments/lr_3e-5/experiment_report.json")
    p7_rows_path = p7 / "evaluation/test_predictions.jsonl"
    p7_rows = _prediction_rows(p7_rows_path)
    y_true, y_pred = _validate_alignment(p7_rows, test_rows, "P7/minilm_finetuned", phase7=True)
    metrics = classification_metrics(y_true, y_pred)
    saved_test = p7_results["test_metrics"]
    _validate_saved_metric(metrics, saved_test, "P7/minilm_finetuned", "test")
    selected_exp = p7_experiments["selected_experiment"]
    if selected_exp != p7_selected["experiment"]:
        raise ValueError("Phase 7 selected experiment report differs from selection manifest")
    if p7_results["model_revision"] != p7_selected["model_revision"]:
        raise ValueError("Phase 7 selected model revision differs across saved artifacts")
    _assert_close(p7_results["validation_macro_f1"], p7_selected["best_validation_macro_f1"], "P7 validation macro F1")
    if p7_experiments.get("test_used_for_selection") is not False:
        raise ValueError("Phase 7 does not document validation-only selection")
    p7_id = "P7/minilm_finetuned"
    registry.append({
        "model_id": p7_id,
        "short_name": _short_name("P7", "minilm_finetuned"),
        "phase": 7,
        "family": "fine-tuned pretrained Sentence Transformer classifier",
        "status": "complete",
        "phase_selected_model_id": p7_id,
        "selected_by_validation": True,
        "selection_metric": "validation macro F1",
        "validation_metrics": {"macro_f1": p7_results["validation_macro_f1"], "other_metrics": None, "source": "saved Phase 7 experiment report; per-record validation predictions are unavailable"},
        "test_metrics": _metric_summary(metrics),
        "configuration": p7_selected["configuration"],
        "model_provenance": {"model_id": p7_results["model_id"], "revision": p7_results["model_revision"], "license": p7_selected["model_license"], "tokenizer_id": p7_selected["tokenizer_id"], "tokenizer_revision": p7_selected["tokenizer_revision"]},
        "sources": {
            "test_predictions": p7_rows_path.relative_to(ROOT).as_posix(),
            "test_metrics": "artifacts/models/phase7/classification_results.json",
            "validation_score": "artifacts/models/phase7/experiments/lr_3e-5/experiment_report.json",
            "selection": "artifacts/models/phase7/classification_experiments.json",
        },
        "saved_test_metric_check": "passed",
    })
    internal_predictions[p7_id] = p7_rows
    source_paths.extend((p7_rows_path, p7 / "classification_results.json", p7 / "classification_experiments.json", p7 / "experiments/lr_3e-5/experiment_report.json"))
    source_paths.extend(path for path in (p7 / "experiments/lr_3e-5/best_checkpoint").rglob("*") if path.is_file())

    if len(registry) != 13 or len({row["model_id"] for row in registry}) != 13:
        raise ValueError(f"Expected 13 complete Phase 3–7 classifier records; found {len(registry)}")
    for record in registry:
        validate_experiment_record(record)
    selected_by_phase = {
        "P3": p3_selection.get("selected"),
        "P5": None,
        "P6": "custom_transformer" if p6_best == "custom_transformer_lr_0.001" else p6_best,
        "P7": "minilm_finetuned" if selected_exp == "lr_3e-5" else selected_exp,
    }
    best_by_phase = {
        f"P{phase}": max((row for row in registry if row["phase"] == phase), key=lambda row: row["validation_metrics"]["macro_f1"])["model_id"]
        for phase in (3, 5, 6, 7)
    }
    best_validation = max(
        (row for row in registry if isinstance(row["validation_metrics"].get("macro_f1"), (int, float))),
        key=lambda row: row["validation_metrics"]["macro_f1"],
    )
    best_test = max(registry, key=lambda row: row["test_metrics"]["macro_f1"])
    summary = {
        "metadata": {**integrity, "model_count": len(registry), "metric_source": "independent recomputation from saved prediction artifacts", "test_selection_used": False},
        "models": registry,
        "selection_summary": {
            "phase_selected_models": selected_by_phase,
            "best_validation_model_within_each_phase": best_by_phase,
            "best_validation_macro_f1_across_saved_experiments": {"model_id": best_validation["model_id"], "value": best_validation["validation_metrics"]["macro_f1"]},
            "best_observed_test_macro_f1": {"model_id": best_test["model_id"], "value": best_test["test_metrics"]["macro_f1"], "retrospective_only": True},
            "phase3_original_comparison_baseline": "P3/tfidf_svm",
            "cross_phase_model_selection": "No single cross-phase selection was recorded; test ranking is retrospective and must not be treated as a validation selection.",
        },
    }
    return summary, internal_predictions, source_paths


def build_retrieval_results() -> tuple[dict, list[Path]]:
    p4 = ROOT / "artifacts/data/phase4"
    p7 = ROOT / "artifacts/data/phase7"
    queries = _json(p4 / "queries.json")
    judgment_doc = _json(p4 / "judgments.json")
    judgments = judgment_doc["judgments"]
    candidate_pool = _json(p4 / "candidate_pool.json")
    p7_rankings = _json(p7 / "retrieval_rankings.json")
    if set(queries) != set(judgments) or set(queries) != set(candidate_pool) or set(queries) != set(p7_rankings):
        raise ValueError("Phase 4 queries, judgments, candidate pools, and Phase 7 rankings do not align")
    if len(queries) != 3:
        raise ValueError(f"Expected three saved retrieval queries; found {len(queries)}")
    judgment_grade_counts = {str(grade): sum(value == grade for q in judgments.values() for value in q.values()) for grade in (0, 1, 2)}
    methods_spec = {
        "Phase 4 TF-IDF": ("phase4_tfidf", {query: candidate_pool[query]["tfidf"] for query in queries}),
        "Phase 4 Word2Vec": ("phase4_word2vec", {query: candidate_pool[query]["word2vec"] for query in queries}),
        "Phase 7 MiniLM cosine": ("phase7_minilm", p7_rankings),
    }
    methods: dict[str, dict] = {}
    all_rankings: dict[str, dict] = {}
    for display_name, (method_id, rankings) in methods_spec.items():
        per_query: dict[str, dict] = {}
        ranked_details: dict[str, list[dict]] = {}
        for query_id in queries:
            candidates = rankings[query_id]
            ids = [row["arxiv_id"] for row in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in {display_name} ranking for {query_id}")
            if len(ids) < 5:
                raise ValueError(f"{display_name} has fewer than five ranked results for {query_id}")
            top_ids = ids[:5]
            per_query[query_id] = evaluate_retrieval(top_ids, judgments[query_id], 5)
            ranked_details[query_id] = [
                {
                    "rank": rank,
                    "arxiv_id": row["arxiv_id"],
                    "score": row.get("score"),
                    "judgment_grade": judgments[query_id].get(row["arxiv_id"]),
                    "judgment_status": "unjudged" if row["arxiv_id"] not in judgments[query_id] else "judged",
                    "title": row.get("title"),
                }
                for rank, row in enumerate(candidates[:10], start=1)
            ]
        macro = {metric: float(statistics.mean(per_query[q][metric] for q in queries)) for metric in ("precision", "recall", "mrr", "ndcg", "unjudged_in_top_k")}
        methods[method_id] = {
            "display_name": display_name,
            "k": 5,
            "query_count": len(queries),
            "per_query": per_query,
            "macro_average": macro,
            "ranked_results": ranked_details,
            "unjudged_treatment": "Unjudged IDs have no positive grade and are counted as nonrelevant by the saved Phase 4 metric implementation.",
        }
        all_rankings[method_id] = rankings

    phase4_saved = _json(p4 / "evaluation.json")
    for mapping in (("tfidf", "phase4_tfidf"), ("word2vec", "phase4_word2vec")):
        saved = phase4_saved["macro_average"][mapping[0]]
        recomputed = methods[mapping[1]]["macro_average"]
        for metric in ("precision", "recall", "mrr", "ndcg"):
            _assert_close(recomputed[metric], saved[metric], f"Phase 4 {mapping[0]} {metric}")
    p7_saved = _json(p7 / "retrieval_evaluation.json")
    p7_recomputed = methods["phase7_minilm"]
    for query_id in queries:
        for metric in ("precision", "recall", "mrr", "ndcg", "unjudged_in_top_k"):
            _assert_close(p7_recomputed["per_query"][query_id][metric], p7_saved["per_query"][query_id][metric], f"Phase 7 retrieval {query_id} {metric}")
    for metric in ("precision", "recall", "mrr", "ndcg", "unjudged_in_top_k"):
        _assert_close(p7_recomputed["macro_average"][metric], p7_saved["macro_average"][metric], f"Phase 7 retrieval macro {metric}")

    reproduction = _verify_phase7_retrieval(queries, p7_rankings)

    timing = _json(p4 / "execution_metadata.json")
    p4_latency = {}
    for method, old_name in (("phase4_tfidf", "tfidf"), ("phase4_word2vec", "word2vec")):
        raw = [timing["query_seconds"][query][old_name] for query in queries]
        p4_latency[method] = {"per_query_seconds": dict(zip(queries, raw)), "mean_seconds": float(statistics.mean(raw)), "scope": "saved retrieval query timing; excludes Retriever construction/loading"}
    p7_latency = p7_saved.get("query_latency_ms", {})
    result = {
        "query_ids": list(queries),
        "queries": queries,
        "k": 5,
        "judgments": {
            "assessor": judgment_doc["assessor"],
            "method": judgment_doc["method"],
            "total_query_document_grades": sum(len(item) for item in judgments.values()),
            "grade_counts": judgment_grade_counts,
            "relevant_per_query": {query: sum(grade > 0 for grade in grades.values()) for query, grades in judgments.items()},
            "interpretation": "Sparse assistant-assessed judgments; not human-reviewed; the pool was built from Phase 4 candidate lists.",
        },
        "methods": methods,
        "macro_average": {key: methods[key]["macro_average"] for key in methods},
        "latency": {**p4_latency, "phase7_minilm": p7_latency},
        "phase7_reproduction": {
            "performed": True,
            "cached_local_model_only": True,
            "offline_mode": True,
            **reproduction,
        },
        "limitations": list(p7_saved.get("limitations", [])),
    }
    paths = [p4 / name for name in ("queries.json", "judgments.json", "candidate_pool.json", "evaluation.json", "execution_metadata.json")]
    paths.extend((p7 / "retrieval_rankings.json", p7 / "retrieval_evaluation.json", p7 / "embedding_metadata.json", p7 / "embedding_index.jsonl", p7 / "paper_embeddings.npy"))
    return result, paths


def _verify_phase7_retrieval(queries: dict, saved_rankings: dict) -> dict:
    """Re-run retrieval from the cached local encoder and checksum-verified index."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from ml.models.pretrained.runtime import expose_existing_base_site_packages

    expose_existing_base_site_packages()
    from transformers.utils import logging as transformer_logging

    transformer_logging.set_verbosity_error()
    transformer_logging.disable_progress_bar()
    from ml.retrieval.transformer import TransformerRetriever

    retriever = TransformerRetriever.from_artifacts(ROOT / "artifacts/data/phase7", device="cpu")
    p1_records = []
    for split in ("train", "validation", "test"):
        p1_records.extend(_jsonl(ROOT / f"artifacts/data/phase1/{split}.jsonl"))
    fields_aligned = len(p1_records) == len(retriever.records) and all(
        all(index_row[field] == source_row[field] for field in SOURCE_FIELDS)
        for index_row, source_row in zip(retriever.records, p1_records)
    )
    if not fields_aligned:
        raise ValueError("Phase 7 retrieval index does not align to Phase 1 corpus records")
    matrix = np.load(ROOT / "artifacts/data/phase7/paper_embeddings.npy", mmap_mode="r")
    norms = np.linalg.norm(matrix, axis=1)
    finite = np.isfinite(matrix).all(axis=1) & np.isfinite(norms)
    valid = finite & (norms > 0)
    if int(valid.sum()) != len(p1_records):
        raise ValueError("Phase 7 retrieval embedding matrix has invalid/zero rows")
    if matrix.shape != (len(p1_records), 384):
        raise ValueError(f"Unexpected Phase 7 embedding matrix shape: {matrix.shape}")
    id_match = True
    score_differences = []
    for query_id, query_text in queries.items():
        current = retriever.search(query_text, k=10)
        saved = saved_rankings[query_id]
        current_ids = [row["arxiv_id"] for row in current]
        saved_ids = [row["arxiv_id"] for row in saved]
        id_match = id_match and current_ids == saved_ids
        if current_ids != saved_ids:
            raise ValueError(f"Locally rerun Phase 7 ranking IDs differ for {query_id}")
        score_differences.extend(abs(float(a["score"]) - float(b["score"])) for a, b in zip(current, saved))
    return {
        "saved_top10_ids_reproduced": id_match,
        "max_absolute_score_difference": max(score_differences, default=0.0),
        "verification": "Local cached encoder and saved corpus embeddings reproduce every saved top-10 ID for all three queries; top-five order is identical.",
        "investigation": {
            "corpus_ids_and_six_phase1_fields_align": fields_aligned,
            "index_rows": len(retriever.records),
            "unique_ids": len({row["arxiv_id"] for row in retriever.records}),
            "embedding_dimension": int(matrix.shape[1]),
            "valid_embedding_rows": int(valid.sum()),
            "stored_norm_min": float(norms.min()),
            "stored_norm_mean": float(norms.mean()),
            "stored_norm_max": float(norms.max()),
            "ranking": "descending dot product after L2-normalizing document and query vectors; equivalent to cosine similarity",
            "query_document_encoding": "paper title and abstract were embedded as documents; raw natural-language query uses the same cached sentence encoder, as designed for a bi-encoder",
            "judgment_alignment": "Phase 4 query IDs and judgment maps match exactly; all ranked IDs are unique; unjudged results are explicit and counted nonrelevant by inherited metric code",
            "bug_found": False,
            "corrected_metric_output_created": False,
        },
    }


def _efficiency_records(classification: dict) -> dict:
    phase6_comparison = _json(ROOT / "artifacts/data/phase6/comparison.json")
    phase6_rows = {row["model"].lower(): row for row in phase6_comparison["results"]}
    p5_selection = _json(ROOT / "artifacts/data/phase5/selection.json")
    p5_verification = _json(ROOT / "artifacts/data/phase5/verification.json")
    p6_selected = _json(ROOT / "artifacts/data/phase6/selection.json")["selected"]
    p7_results = _json(ROOT / "artifacts/models/phase7/classification_results.json")
    p7_report = _json(ROOT / "artifacts/models/phase7/experiments/lr_3e-5/experiment_report.json")
    records = []
    for model in classification["models"]:
        mid = model["model_id"]
        phase, key = mid.split("/", 1)
        item = {
            "model_id": mid,
            "phase": int(phase[1:]),
            "test_macro_f1": model["test_metrics"]["macro_f1"],
            "training_seconds": None,
            "inference": {"seconds_per_document": None, "milliseconds_per_document": None, "timing_scope": "Not recorded in source artifacts", "raw_measurements": None},
            "parameter_count": None,
            "parameter_count_scope": None,
            "artifact_bytes": None,
            "size_basis": None,
            "size_source": None,
            "auxiliary_component_bytes": None,
            "auxiliary_component_source": None,
            "model_plus_auxiliary_bytes": None,
            "device": None,
            "batch_size": None,
            "input_limit_tokens": None,
            "notes": [],
        }
        if phase == "P3":
            model_path = ROOT / f"artifacts/data/phase3/models/{key}.joblib"
            representation_path = ROOT / {
                "bow": "artifacts/data/phase2/bow.joblib",
                "tfidf": "artifacts/data/phase2/tfidf.joblib",
                "word2vec": "artifacts/data/phase2/word2vec.model",
            }[key.split("_", 1)[0]]
            estimator = joblib.load(model_path)
            count = _learned_parameter_count(estimator)
            metric_path = ROOT / f"artifacts/data/phase3/test/{key}.json"
            metric = _json(metric_path)
            item.update({
                "parameter_count": count,
                "parameter_count_scope": "Count of learned classifier coefficient/log-probability entries and intercept/prior entries; excludes the shared Phase 2 feature representation.",
                "artifact_bytes": model_path.stat().st_size,
                "size_basis": "serialized Phase 3 classifier .joblib file",
                "size_source": model_path.relative_to(ROOT).as_posix(),
                "auxiliary_component_bytes": representation_path.stat().st_size,
                "auxiliary_component_source": representation_path.relative_to(ROOT).as_posix(),
                "model_plus_auxiliary_bytes": model_path.stat().st_size + representation_path.stat().st_size,
                "device": "CPU-based scikit-learn estimator; device was not explicitly logged",
                "inference": {
                    "seconds_per_document": metric["seconds_per_document"],
                    "milliseconds_per_document": metric["seconds_per_document"] * 1000,
                    "total_seconds": metric["prediction_seconds"],
                    "timing_scope": "Saved classifier predict time; excludes Phase 2 feature extraction and tokenization.",
                    "raw_measurements": {"prediction_seconds": metric["prediction_seconds"], "seconds_per_document": metric["seconds_per_document"]},
                },
                "notes": ["Training time was not logged for this individual Phase 3 run in its source artifacts."],
            })
            # The Phase 6 comparison did record a comparable training time for its selected Phase 3 baseline.
            if key == "tfidf_svm":
                historical = next((row for name, row in phase6_rows.items() if "phase 3 tf-idf + linear svm" in name), None)
                if historical:
                    item["training_seconds"] = historical.get("training_seconds")
                    item["training_seconds_source"] = "artifacts/data/phase6/comparison.json"
                    if historical.get("parameter_count") != count:
                        item["notes"].append("Phase 6 historical parameter count differs from the coefficient count defined here; both source values are retained.")
                        item["phase6_reported_parameter_count"] = historical.get("parameter_count")
        elif phase == "P5":
            selected = p5_selection[key]
            raw_test = _json(ROOT / f"artifacts/data/phase5/test/{key}.json")
            comparison = next((row for name, row in phase6_rows.items() if f"phase 5 {key}" in name), None)
            verification = p5_verification["models"][key]
            repeated = raw_test.get("inference_seconds_repeated", [])
            per_document = float(statistics.mean(repeated) / raw_test["report"]["weighted avg"]["support"]) if repeated else None
            model_only_batch_seconds = verification.get("model_only_seconds", [])
            model_only_batch_size = verification.get("model_only_batch_size")
            model_only_per_document = (float(statistics.mean(model_only_batch_seconds)) / model_only_batch_size) if model_only_batch_seconds and model_only_batch_size else None
            single_document = verification.get("end_to_end_single_seconds_repeated", [])
            item.update({
                "training_seconds": selected.get("training_seconds"),
                "training_seconds_source": "artifacts/data/phase5/selection.json",
                "inference": {
                    "seconds_per_document": per_document,
                    "milliseconds_per_document": per_document * 1000 if per_document is not None else None,
                    "model_only_batch_seconds_per_document": model_only_per_document,
                    "single_document_end_to_end_seconds": float(statistics.mean(single_document)) if single_document else None,
                    "total_seconds_per_test_pass": repeated,
                    "timing_scope": raw_test.get("latency_description", "Saved Phase 5 repeated test-set timing"),
                    "additional_timing_scope": verification.get("note"),
                    "raw_measurements": {
                        "test_pass_seconds": repeated,
                        "test_records": raw_test["report"]["weighted avg"]["support"],
                        "model_only_batch_seconds": model_only_batch_seconds,
                        "model_only_batch_size": model_only_batch_size,
                        "single_document_end_to_end_seconds": single_document,
                    },
                },
                "parameter_count": comparison.get("parameter_count") if comparison else None,
                "parameter_count_scope": "As recorded in Phase 6 comparison artifact; Phase 5 model file was not present for independent recounting." if comparison else None,
                "artifact_bytes": comparison.get("model_or_checkpoint_bytes") if comparison else None,
                "size_basis": "Recorded Phase 6 model_or_checkpoint_bytes" if comparison else None,
                "size_source": "artifacts/data/phase6/comparison.json" if comparison else None,
                "device": selected.get("device"),
                "batch_size": raw_test.get("batch_size"),
                "input_limit_tokens": int(_json(ROOT / "artifacts/data/phase2/metadata.json")["sequence_length"]),
                "notes": ["Latency excludes tokenization; it includes batch slicing, transfer, forward pass, and prediction transfer."],
            })
        elif phase == "P6":
            comparison = next(row for row in phase6_comparison["results"] if row["model"] == "Phase 6 custom Transformer")
            metric = _json(ROOT / "artifacts/data/phase6/test/custom_transformer.json")
            checkpoint = ROOT / "artifacts/data/phase6/runs/custom_transformer_lr_0.001/best.pt"
            latency = metric["inference_latency"]
            item.update({
                "training_seconds": metric["training_seconds"],
                "training_seconds_source": "artifacts/data/phase6/test/custom_transformer.json",
                "inference": {
                    "seconds_per_document": latency["model_only_mean_seconds_per_document"],
                    "milliseconds_per_document": latency["model_only_mean_seconds_per_document"] * 1000,
                    "model_only_seconds_per_document": latency["model_only_mean_seconds_per_document"],
                    "transfer_included_seconds_per_document": latency["data_transfer_and_prediction_mean_seconds_per_document"],
                    "timing_scope": "Preloaded token tensors; tokenization excluded. Separate transfer-included timing is also preserved.",
                    "raw_measurements": latency,
                },
                "parameter_count": metric["parameter_count"],
                "parameter_count_scope": "Saved total trainable parameter count.",
                "artifact_bytes": checkpoint.stat().st_size,
                "size_basis": "best.pt checkpoint file; verified against saved byte count",
                "size_source": checkpoint.relative_to(ROOT).as_posix(),
                "device": metric.get("device"),
                "batch_size": metric.get("batch_size"),
                "input_limit_tokens": p6_selected["config"]["model_config"]["max_length"],
                "notes": ["The phase's model-only latency excludes text tokenization; transfer-included timing excludes text tokenization too."],
            })
        elif phase == "P7":
            model_dir = ROOT / "artifacts/models/phase7/experiments/lr_3e-5/best_checkpoint"
            checkpoint_bytes = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
            item.update({
                "training_seconds": p7_report["training_seconds"],
                "training_seconds_source": "artifacts/models/phase7/experiments/lr_3e-5/experiment_report.json",
                "inference": {
                    "seconds_per_document": p7_results["inference_ms_per_document_batch32_includes_tokenization"] / 1000,
                    "milliseconds_per_document": p7_results["inference_ms_per_document_batch32_includes_tokenization"],
                    "total_seconds": p7_results["inference_seconds_batch32_includes_tokenization"],
                    "timing_scope": "Batch size 32; includes tokenization and model inference; excludes checkpoint loading.",
                    "raw_measurements": {"total_seconds": p7_results["inference_seconds_batch32_includes_tokenization"], "milliseconds_per_document": p7_results["inference_ms_per_document_batch32_includes_tokenization"]},
                },
                "parameter_count": p7_results["parameter_count"],
                "parameter_count_scope": "Saved total model parameter count.",
                "artifact_bytes": checkpoint_bytes,
                "recorded_artifact_bytes": p7_results["checkpoint_bytes"],
                "size_basis": "sum of files in the selected best_checkpoint directory",
                "size_source": "artifacts/models/phase7/experiments/lr_3e-5/best_checkpoint",
                "device": p7_results.get("evaluation_device"),
                "batch_size": 32,
                "input_limit_tokens": p7_report["effective_max_length"],
                "notes": [],
            })
            if checkpoint_bytes != p7_results["checkpoint_bytes"]:
                item["notes"].append("Directory byte sum differs from recorded checkpoint_bytes; both values are retained.")
        records.append(item)
    return {
        "timing_comparability": "Inference boundaries differ across phases; consult timing_scope and raw_measurements before comparing. Training hardware and software versions also differ.",
        "models": records,
        "retrieval": _retrieval_efficiency(),
    }


def _retrieval_efficiency() -> dict:
    p4 = _json(ROOT / "artifacts/data/phase4/execution_metadata.json")
    p7 = _json(ROOT / "artifacts/data/phase7/retrieval_evaluation.json")
    return {
        "Phase4_TFIDF": {"index_load_seconds": p4.get("initial_load_seconds"), "query_seconds": {q: p4["query_seconds"][q]["tfidf"] for q in p4["query_seconds"]}, "timing_scope": "Retriever construction plus vectorizer/model load is recorded separately; query timing excludes initialization."},
        "Phase4_Word2Vec": {"index_load_seconds": p4.get("initial_load_seconds"), "query_seconds": {q: p4["query_seconds"][q]["word2vec"] for q in p4["query_seconds"]}, "timing_scope": "Retriever construction plus vectorizer/model load is recorded separately; query timing excludes initialization."},
        "Phase7_MiniLM": {"index_load_seconds": None, "query_latency_ms": p7.get("query_latency_ms"), "timing_scope": "Warm query encoding and full-corpus ranking; excludes encoder/model and index load."},
    }


def build_error_analysis(classification: dict, predictions: dict[str, list[dict]], test_rows: list[dict]) -> tuple[dict, list[dict]]:
    model_ids = [row["model_id"] for row in classification["models"]]
    test_index = {row["arxiv_id"]: row for row in test_rows}
    predicted_by_model: dict[str, dict[str, int]] = {}
    for model_id, rows in predictions.items():
        if model_id.startswith("P7/"):
            predicted_by_model[model_id] = {row["arxiv_id"]: int(row["predicted_label"]) for row in rows}
        else:
            predicted_by_model[model_id] = {row["arxiv_id"]: int(row["predicted"]) for row in rows}
    if set(predicted_by_model) != set(model_ids):
        raise ValueError("Error analysis predictions do not cover all registry models")
    confusion = np.zeros((4, 4), dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = {}
    errors_jsonl: list[dict] = []
    per_model_errors: dict[str, int] = {}
    per_id_models: dict[str, list[str]] = {row["arxiv_id"]: [] for row in test_rows}
    for model_id in model_ids:
        local_error_count = 0
        for row in test_rows:
            paper_id = row["arxiv_id"]
            actual = int(row["label"])
            predicted = predicted_by_model[model_id][paper_id]
            confusion[actual, predicted] += 1
            if actual != predicted:
                local_error_count += 1
                per_id_models[paper_id].append(model_id)
                pair_counts[(actual, predicted)] = pair_counts.get((actual, predicted), 0) + 1
                record = {field: row[field] for field in SOURCE_FIELDS}
                record.update({
                    "model_id": model_id,
                    "actual_label": actual,
                    "actual_label_name": LABEL_NAMES[actual],
                    "predicted_label": predicted,
                    "predicted_label_name": LABEL_NAMES[predicted],
                    "misclassified": True,
                })
                errors_jsonl.append(record)
        per_model_errors[model_id] = local_error_count
    shared = []
    for paper_id, erroneous_models in per_id_models.items():
        if len(erroneous_models) >= 2:
            row = test_index[paper_id]
            shared.append({
                "arxiv_id": paper_id,
                "title": row["title"],
                "actual_label": row["label"],
                "actual_label_name": LABEL_NAMES[row["label"]],
                "misclassified_by_count": len(erroneous_models),
                "misclassified_by_models": erroneous_models,
            })
    shared.sort(key=lambda item: (-item["misclassified_by_count"], item["arxiv_id"]))
    pairs = [
        {"actual_label": actual, "actual_name": LABEL_NAMES[actual], "predicted_label": predicted, "predicted_name": LABEL_NAMES[predicted], "count_across_models": count}
        for (actual, predicted), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    by_model = {row["model_id"]: row for row in classification["models"]}
    examples_by_pair = []
    for pair in pairs[:8]:
        actual = pair["actual_label"]
        predicted = pair["predicted_label"]
        examples = []
        for row in test_rows:
            paper_id = row["arxiv_id"]
            if int(row["label"]) != actual:
                continue
            models_with_pair = [model_id for model_id in model_ids if predicted_by_model[model_id][paper_id] == predicted]
            if models_with_pair:
                examples.append({
                    "arxiv_id": paper_id,
                    "title": row["title"],
                    "actual_label": actual,
                    "actual_label_name": LABEL_NAMES[actual],
                    "predicted_label": predicted,
                    "predicted_label_name": LABEL_NAMES[predicted],
                    "misclassified_by_models": models_with_pair,
                })
                if len(examples) == 5:
                    break
        examples_by_pair.append({**pair, "examples": examples})
    per_model = {
        model_id: {
            "test_errors": per_model_errors[model_id],
            "error_rate": per_model_errors[model_id] / len(test_rows),
            "classwise_f1": {item["name"]: item["f1"] for item in by_model[model_id]["test_metrics"]["per_class"]},
        }
        for model_id in model_ids
    }
    result = {
        "test_records": len(test_rows),
        "model_count": len(model_ids),
        "total_model_errors": len(errors_jsonl),
        "per_model": per_model,
        "aggregate_confusion_counts_across_models": confusion.tolist(),
        "most_common_misclassification_pairs": pairs,
        "shared_error_ids": {"records_with_errors_in_at_least_two_models": len(shared), "misclassified_by_all_models": [item["arxiv_id"] for item in shared if item["misclassified_by_count"] == len(model_ids)], "all_records": shared, "top_25": shared[:25]},
        "representative_examples_by_common_confusion": examples_by_pair,
        "error_record_fields": list(SOURCE_FIELDS) + ["model_id", "actual_label", "actual_label_name", "predicted_label", "predicted_label_name", "misclassified"],
        "interpretation_limit": "Errors are measured on the shared fixed Phase 1 test split. The aggregate confusion totals sum over models and are not a count of unique papers.",
    }
    return result, errors_jsonl


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_classification_tables(classification: dict) -> None:
    models = classification["models"]
    flat = []
    registry_rows = []
    for row in models:
        result = {"model_id": row["model_id"], "short_name": row["short_name"], "phase": row["phase"], "family": row["family"], "validation_macro_f1": row["validation_metrics"].get("macro_f1"), "selected_by_validation": row["selected_by_validation"]}
        result.update({key: row["test_metrics"][key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "records")})
        for item in row["test_metrics"]["per_class"]:
            result[f"f1_{item['name']}"] = item["f1"]
            result[f"support_{item['name']}"] = item["support"]
        flat.append(result)
        registry_rows.append({
            "model_id": row["model_id"],
            "phase": row["phase"],
            "status": row["status"],
            "selected_by_validation": row["selected_by_validation"],
            "phase_selected_model_id": row.get("phase_selected_model_id"),
            "best_validation_in_phase": row.get("best_validation_in_phase"),
            "validation_macro_f1": row["validation_metrics"].get("macro_f1"),
            "test_macro_f1": row["test_metrics"]["macro_f1"],
            "configuration_json": json.dumps(row["configuration"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "sources_json": json.dumps(row["sources"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        })
    write_csv(PHASE8_DATA / "classification_metrics.csv", flat)
    write_csv(PHASE8_DATA / "experiment_registry.csv", registry_rows)
    per_class_rows = []
    cm_rows = []
    for row in models:
        for metric in row["test_metrics"]["per_class"]:
            per_class_rows.append({"model_id": row["model_id"], **metric})
        for actual, actual_name in enumerate(LABEL_NAMES.values()):
            for predicted, predicted_name in enumerate(LABEL_NAMES.values()):
                cm_rows.append({"model_id": row["model_id"], "actual_label": actual, "actual_name": actual_name, "predicted_label": predicted, "predicted_name": predicted_name, "count": row["test_metrics"]["confusion_matrix"][actual][predicted]})
    write_csv(PHASE8_DATA / "classification_per_class.csv", per_class_rows)
    write_csv(PHASE8_DATA / "confusion_matrices.csv", cm_rows)


def _write_retrieval_tables(retrieval: dict) -> None:
    rows = []
    for method_id, method in retrieval["methods"].items():
        for query, metric in method["per_query"].items():
            rows.append({"method_id": method_id, "display_name": method["display_name"], "query_id": query, **metric})
        rows.append({"method_id": method_id, "display_name": method["display_name"], "query_id": "MACRO", **method["macro_average"]})
    write_csv(PHASE8_DATA / "retrieval_metrics.csv", rows)
    ranking_rows = []
    for method_id, method in retrieval["methods"].items():
        for query, ranked in method["ranked_results"].items():
            for result in ranked:
                ranking_rows.append({"method_id": method_id, "query_id": query, **result})
    write_csv(PHASE8_DATA / "retrieval_rankings.csv", ranking_rows)


def _write_efficiency_tables(efficiency: dict) -> None:
    flat = []
    for row in efficiency["models"]:
        flat.append({
            "model_id": row["model_id"], "phase": row["phase"], "test_macro_f1": row["test_macro_f1"], "training_seconds": row["training_seconds"],
            "inference_seconds_per_document": row["inference"].get("seconds_per_document"), "inference_milliseconds_per_document": row["inference"].get("milliseconds_per_document"),
            "inference_scope": row["inference"].get("timing_scope"), "parameter_count": row["parameter_count"], "parameter_count_scope": row["parameter_count_scope"],
            "artifact_bytes": row["artifact_bytes"], "size_basis": row["size_basis"], "device": row["device"], "batch_size": row["batch_size"], "input_limit_tokens": row["input_limit_tokens"],
            "auxiliary_component_bytes": row["auxiliary_component_bytes"], "auxiliary_component_source": row["auxiliary_component_source"], "model_plus_auxiliary_bytes": row["model_plus_auxiliary_bytes"],
            "notes": "; ".join(row["notes"]),
        })
    write_csv(PHASE8_DATA / "efficiency_metrics.csv", flat)


def _write_error_tables(errors: dict, records: list[dict]) -> None:
    write_json(PHASE8_DATA / "error_analysis.json", errors)
    fieldnames = list(SOURCE_FIELDS) + ["model_id", "actual_label", "actual_label_name", "predicted_label", "predicted_label_name", "misclassified"]
    path = PHASE8_DATA / "classification_errors.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(PHASE8_DATA / "classification_errors.csv", records, fieldnames)
    write_csv(PHASE8_DATA / "common_error_pairs.csv", errors["most_common_misclassification_pairs"])
    write_csv(PHASE8_DATA / "shared_error_ids.csv", errors["shared_error_ids"]["all_records"])


def _report_markdown(classification: dict, retrieval: dict, efficiency: dict, errors: dict, chart_manifest: dict) -> str:
    model_rows = classification["models"]
    header = "| Model | Val macro F1 | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 | Config selected using validation |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"
    lines = [
        "# Phase 8 — Model Evaluation and Comparative Analysis",
        "",
        "## Scope and data provenance",
        "",
        f"This report compares {len(model_rows)} completed Phase 3–7 classifiers using their saved predictions on the same Phase 1 test split. Metrics were recomputed independently with labels fixed to `{LABEL_IDS}` and names `{', '.join(LABEL_NAMES.values())}`. No previous model was retrained, and no test metric was used to choose a model.",
        "",
        f"The verified Phase 1 split sizes are train={classification['metadata']['split_sizes']['train']:,}, validation={classification['metadata']['split_sizes']['validation']:,}, test={classification['metadata']['split_sizes']['test']:,}. Test support is {classification['metadata']['test_support']}; IDs are unique and split-disjoint. Every saved classifier prediction list matches the test manifest in exact ID order and actual labels.",
        "",
        "Machine-readable records, per-class CSV, confusion counts, retrieval results, efficiency measurements, provenance hashes, and chart data are under `artifacts/data/phase8/`.",
        "",
        "## Classification results",
        "",
        header,
    ]
    for row in model_rows:
        metric = row["test_metrics"]
        val = row["validation_metrics"].get("macro_f1")
        lines.append(f"| {row['short_name']} | {val:.4f} | {metric['accuracy']:.4f} | {metric['macro_precision']:.4f} | {metric['macro_recall']:.4f} | {metric['macro_f1']:.4f} | {metric['weighted_precision']:.4f} | {metric['weighted_recall']:.4f} | {metric['weighted_f1']:.4f} | {'Yes' if row['selected_by_validation'] else 'No'} |")
    selection = classification["selection_summary"]
    lines.extend([
        "",
        f"Best validation macro F1 among recorded Phase 3–7 runs: **{selection['best_validation_macro_f1_across_saved_experiments']['model_id']}** ({selection['best_validation_macro_f1_across_saved_experiments']['value']:.4f}). Best observed test macro F1: **{selection['best_observed_test_macro_f1']['model_id']}** ({selection['best_observed_test_macro_f1']['value']:.4f}); this is a retrospective comparison only. The Phase 3 original validation-selected baseline was `{selection['phase3_original_comparison_baseline']}`. Phase 5 selected the learning-rate checkpoint within each architecture; its selection file did not designate a single winning architecture. The project did not record one cross-phase model selection.",
        "",
        "The four test classes each have 500 records, so macro and weighted averages are numerically equal up to floating-point rounding. The full per-class precision, recall, F1, support, and all thirteen fixed-order confusion matrices are in the JSON/CSV outputs.",
        "",
        "### Per-class test F1",
        "",
        "| Model | Algorithms | NLP/AI | Systems/Infra | Vision |\n|---|---:|---:|---:|---:|",
    ])
    for row in model_rows:
        f1_by_name = {item["name"]: item["f1"] for item in row["test_metrics"]["per_class"]}
        lines.append(f"| {row['short_name']} | {f1_by_name['Algorithms']:.4f} | {f1_by_name['NLP/AI']:.4f} | {f1_by_name['Systems/Infra']:.4f} | {f1_by_name['Vision']:.4f} |")
    lines.extend([
        "",
        "## Retrieval comparison",
        "",
        "The three saved queries and Phase 4 assistant-assessed judgment maps were reused without editing. All scores are at k=5; recall denominator is the number of judged relevant IDs for that query. Unjudged IDs count as nonrelevant in the inherited metric implementation.",
        f"The judgment set contains {retrieval['judgments']['total_query_document_grades']} query-paper grades: {retrieval['judgments']['grade_counts']['0']} grade-0, {retrieval['judgments']['grade_counts']['1']} grade-1, and {retrieval['judgments']['grade_counts']['2']} grade-2. Judged relevant counts are {retrieval['judgments']['relevant_per_query']}.",
        "",
        "| Method | Precision@5 | Recall@5 | MRR@5 | nDCG@5 | Mean query latency (as recorded) | Unjudged in top 5 (mean) |\n|---|---:|---:|---:|---:|---:|---:|",
    ])
    latencies = retrieval["latency"]
    for method_id, method in retrieval["methods"].items():
        avg = method["macro_average"]
        if method_id.startswith("phase4"):
            latency = latencies[method_id]["mean_seconds"] * 1000
            latency_label = f"{latency:.3f} ms"
        else:
            latency = latencies[method_id].get("mean")
            latency_label = f"{latency:.3f} ms" if latency is not None else "Not recorded"
        lines.append(f"| {method['display_name']} | {avg['precision']:.4f} | {avg['recall']:.4f} | {avg['mrr']:.4f} | {avg['ndcg']:.4f} | {latency_label} | {avg['unjudged_in_top_k']:.2f} |")
    lines.extend([
        "",
        "The low Phase 7 MiniLM retrieval values were reproduced from the saved rankings and judgments, and the top ten were independently rerun using the locally cached model in offline mode. The rerun matched all saved IDs in order for all three queries; maximum score difference was 1.2e-7. The index aligns to all 19,999 Phase 1 rows, with 384-dimensional finite normalized embeddings and cosine-equivalent dot-product ranking. No ranking, normalization, ID, or metric bug was found, so no corrected metric output was created. The ranking scores reflect a very small, assistant-assessed Phase 4 pool built from earlier methods; most transformer top-five results are unjudged, which depresses these incomplete-pool scores.",
        "",
        "## Efficiency and saved configurations",
        "",
        "Training and inference timings are reported only when a source artifact recorded them. Timing scopes differ: Phase 3 excludes feature extraction; Phase 5 excludes tokenization but includes batch/device/prediction transfers; Phase 6 excludes tokenization and reports model-only plus transfer-included passes; Phase 7 batch-32 inference includes tokenization. These measurements should not be read as a hardware-normalized benchmark.",
        "",
        "See `artifacts/data/phase8/efficiency_metrics.csv` and `efficiency_results.json` for per-model training time, latency, parameter count, saved size basis, device, batch size, input limits, raw repetitions, and explicit missing values. Phase 3 model sizes are the serialized classifier `.joblib` files. Phase 5 size/parameter values are those recorded in the Phase 6 comparison artifact; the Phase 5 checkpoints are not available for independent recounting in the current Phase 5 output directory.",
        "",
        "| Model | Train seconds | Test inference ms/doc | Parameters | Classifier/checkpoint bytes | Auxiliary bytes | Device | Batch | Token limit |\n|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in efficiency["models"]:
        def fmt(value):
            if value is None:
                return "Not recorded"
            return f"{value:,.3f}" if isinstance(value, float) else f"{value:,}"
        lines.append(f"| {row['model_id']} | {fmt(row['training_seconds'])} | {fmt(row['inference'].get('milliseconds_per_document'))} | {fmt(row['parameter_count'])} | {fmt(row['artifact_bytes'])} | {fmt(row['auxiliary_component_bytes'])} | {row['device'] or 'Not recorded'} | {fmt(row['batch_size'])} | {fmt(row['input_limit_tokens'])} |")
    lines.extend([
        "",
        "The full selected configuration for each run, its status, validation/test metrics, and source paths are in `artifacts/data/phase8/experiment_registry.csv` and `classification_results.json`. The auxiliary Phase 3 bytes identify its saved BoW/TF-IDF vectorizer or Word2Vec model; the Phase 3 classifier/checkpoint column does not include those shared dependencies.",
        "",
        "## Error analysis",
        "",
        f"Across 13 models, the saved predictions contain {errors['total_model_errors']:,} model-level errors over the common test set. The most frequent aggregate actual-to-predicted confusions are:",
        "",
        "| Actual | Predicted | Count across models |\n|---|---|---:|",
    ])
    for item in errors["most_common_misclassification_pairs"][:8]:
        lines.append(f"| {item['actual_name']} | {item['predicted_name']} | {item['count_across_models']} |")
    lines.extend(["", "Representative examples from the test manifest:", ""])
    for pair in errors["representative_examples_by_common_confusion"][:3]:
        examples = []
        for example in pair["examples"][:3]:
            title = example["title"][:90].replace("|", "\\|")
            examples.append(f"`{example['arxiv_id']}` {title}")
        lines.append(f"- {pair['actual_name']} → {pair['predicted_name']} ({pair['count_across_models']} model-level confusions): " + "; ".join(examples))
    lines.extend([
        "",
        f"{errors['shared_error_ids']['records_with_errors_in_at_least_two_models']} test IDs are misclassified by at least two models; {len(errors['shared_error_ids']['misclassified_by_all_models'])} are misclassified by all thirteen. `classification_errors.jsonl` preserves each original Phase 1 field for every model error and adds true/predicted labels. Shared-error examples include titles and labels only.",
        "",
        "## Figures",
        "",
    ])
    for chart in chart_manifest["charts"]:
        lines.append(f"- `artifacts/data/phase8/plots/{chart['file']}` — {chart['title']}")
    lines.extend([
        "",
        "## Comparative interpretation",
        "",
        "The strongest saved test classifier is the validation-selected Phase 3 TF-IDF Linear SVM (macro F1 0.9520), followed closely by the Phase 7 fine-tuned MiniLM classifier (0.9504). Phase 7 uses a much larger pretrained model and its recorded batch-32 inference includes tokenization, while Phase 3 timing excludes feature extraction; their latency and storage comparisons therefore need the stated scope caveats. The recurrent runs span a wider range: the simple RNN is weakest (0.7245 macro F1), while the BiLSTM with attention is strongest among Phase 5 runs (0.9280). The custom Transformer reaches 0.9284. These are observed results for the stored configurations and this one stratified split, not evidence that one architecture is generally superior.",
        "",
        "## Verification status",
        "",
        "Every classifier has 2,000 unique, aligned test predictions with labels in the original mapping. Saved test metrics were checked against independent recomputation. Phase 3, 5, and 6 validation predictions were independently aligned and scored; Phase 7 validation scores were checked against its selected experiment record, which states test data was not used for selection. Phase 4 retrieval averages and Phase 7 per-query/macro metrics were recomputed and matched their saved summaries.",
        "",
        "See `docs/phase8_limitations.md` for threats to validity and `docs/phase8_test_report.md` for the actual Phase 8 and Phase 1–7 regression run results.",
        "",
    ])
    return "\n".join(lines)


def write_phase8_outputs() -> dict:
    """Validate prior artifacts, compute comparisons, and write Phase 8 artifacts."""
    PHASE8_DATA.mkdir(parents=True, exist_ok=True)
    splits, integrity = _manifest_and_integrity()
    classification, predictions, classification_sources = build_classification_results(splits, integrity)
    retrieval, retrieval_sources = build_retrieval_results()
    efficiency = _efficiency_records(classification)
    errors, error_rows = build_error_analysis(classification, predictions, splits["test"])

    provenance_paths = list(set(classification_sources + retrieval_sources))
    phase8_code_paths = sorted((ROOT / "ml/evaluation").glob("*.py")) + sorted((ROOT / "tests/phase8").glob("*.py"))
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": _source_info(provenance_paths),
        "phase8_implementation_files": _source_info(phase8_code_paths),
        "source_split_manifest_files": ["artifacts/data/phase1/train.jsonl", "artifacts/data/phase1/validation.jsonl", "artifacts/data/phase1/test.jsonl"],
        "protected_phase1_to_phase7_files_modified": False,
        "network_access_for_retrieval_reproduction": False,
        "git_or_github_commands_used": False,
    }
    write_json(PHASE8_DATA / "classification_results.json", classification)
    write_json(PHASE8_DATA / "retrieval_results.json", retrieval)
    write_json(PHASE8_DATA / "efficiency_results.json", efficiency)
    write_json(PHASE8_DATA / "provenance.json", provenance)
    _write_classification_tables(classification)
    _write_retrieval_tables(retrieval)
    _write_efficiency_tables(efficiency)
    _write_error_tables(errors, error_rows)
    chart_manifest = generate_charts(classification, retrieval, efficiency, PHASE8_PLOTS)
    write_json(PHASE8_DATA / "chart_manifest.json", chart_manifest)
    doc = _report_markdown(classification, retrieval, efficiency, errors, chart_manifest)
    (PHASE8_DOCS / "phase8_documentation.md").write_text(doc, encoding="utf-8")
    limitations = _limitations_markdown(integrity, classification, retrieval, efficiency)
    (PHASE8_DOCS / "phase8_limitations.md").write_text(limitations, encoding="utf-8")
    return {
        "classifier_count": len(classification["models"]),
        "retrieval_methods": list(retrieval["methods"]),
        "error_records": len(error_rows),
        "chart_count": chart_manifest["chart_count"],
        "output_files": [path.relative_to(ROOT).as_posix() for path in sorted(PHASE8_DATA.rglob("*")) if path.is_file()] + ["docs/phase8_documentation.md", "docs/phase8_limitations.md"],
        "classification": classification,
        "retrieval": retrieval,
        "efficiency": efficiency,
        "errors": errors,
        "charts": chart_manifest,
    }


def _limitations_markdown(integrity: dict, classification: dict, retrieval: dict, efficiency: dict) -> str:
    p7 = _json(ROOT / "artifacts/models/phase7/experiments/lr_3e-5/experiment_report.json")
    phase1_meta = _json(ROOT / "artifacts/data/phase1/metadata.json")
    exclusions = _json(ROOT / "artifacts/data/phase1/exclusions.json")
    lines = [
        "# Phase 8 — Limitations and Threats to Validity",
        "",
        "## Classification data and selection",
        "",
        f"- This comparison uses a single Phase 1 split (seed {phase1_meta.get('seed', 'not recorded')}) with {integrity['split_sizes']['test']:,} test records and 500 examples per label. It estimates performance for this split and source dataset, not all arXiv Computer Science papers or future records.",
        f"- Phase 1 source: `{phase1_meta.get('source')}`. Dataset revision and licensing are recorded in Phase 1 metadata. Phase 1 exclusion record: `{json.dumps(exclusions, ensure_ascii=False)}`.",
        "- Test set is reused to compare thirteen saved models. The best observed test score is retrospective and is not an unbiased selection estimate. Phase-specific selections were validated independently; no cross-phase model choice was recorded.",
        "- Test support is balanced across four broad labels. The labels do not represent the full taxonomy of arXiv subjects; all categories beyond the primary broad class are not modeled here.",
        "- Model families use different feature pipelines, training schedules, data limits, optimization settings, and computational devices. The comparison is descriptive, not a controlled architecture ablation.",
        "- Phase 5/6 recurrent and custom Transformer inputs use the saved Phase 2 sequence length of 290 tokens; the Phase 7 pretrained classifier uses an effective 256-token maximum. Truncated records may omit later text.",
        "- Phase 7 fine-tunes an external Apache-2.0 Sentence Transformer checkpoint, revision `" + str(p7.get("model_revision")) + "`. The earlier DistilBERT plan was not the model actually used; this report correctly identifies the saved MiniLM model.",
        "- Per-record Phase 7 validation predictions were not preserved. Its selected validation macro F1 and validation-only selection flag were checked from the saved experiment report; full independent validation metric recomputation is therefore unavailable for that run.",
        "- Phase 1 removed one exact duplicate record before splitting, as recorded in the exclusions artifact. This comparison does not reassess semantic near-duplicates beyond the saved Phase 1 cleaning decision.",
        "",
        "## Efficiency",
        "",
        "- Training and inference times were produced in different runs and with different scopes. Phase 3 timing excludes feature extraction; Phase 5 excludes tokenization; Phase 6 timing excludes tokenization; Phase 7 batch-32 classification timing includes tokenization. No normalized cross-device speed claim is justified.",
        "- Phase 5 checkpoint files are not available for independent inspection in the current Phase 5 artifact tree. Parameter counts and model/checkpoint sizes are retained from the saved Phase 6 comparison artifact and labeled as such.",
        "- Phase 3 stored model sizes are serialized classifier `.joblib` sizes; the shared Phase 2 features and Word2Vec model are external components. They are not full deployment package sizes.",
        "- Phase 7 checkpoint directory size includes its saved model/tokenizer/config files; other phases may report only a checkpoint or serialized estimator. Size bases are per-row in `efficiency_results.json`.",
        "- Retrieval timing is not an apples-to-apples latency test: Phase 4 records query times after its retrieval engine has been initialized; Phase 7 records warm query encoding plus full-corpus ranking but not model/index loading.",
        "",
        "## Retrieval judgments and coverage",
        "",
        f"- Retrieval evaluation has only {retrieval['judgments']['total_query_document_grades']} assistant-assessed grades across {len(retrieval['query_ids'])} queries. The grades were not human-reviewed.",
        "- The judgment pool was assembled from Phase 4 TF-IDF and Word2Vec top-five lists. This favors those methods' candidate space; Phase 7-only retrieved papers may be unjudged.",
        "- The inherited metric function counts an unjudged result as nonrelevant. The reported retrieval values therefore reflect incomplete pooled judgments and do not estimate exhaustive relevance.",
        "- Query wording and subject coverage are narrow; per-query variability is large. Three examples cannot support broad claims about semantic retrieval quality.",
        "- Retrieval classification labels are not relevance labels. Their presence on retrieved papers is descriptive and was not used to compute retrieval metrics.",
        "",
        "## Reproducibility and scope",
        "",
        "- Phase 8 consumes immutable saved predictions, rankings, judgments, and configuration/efficiency records. It does not retrain earlier models or regenerate earlier outputs.",
        "- Phase 7 top-ten retrieval was locally reproduced from the cached model and saved embeddings in offline mode. This verifies local artifact consistency, not transfer to another hardware/runtime environment.",
        "- Missing values are retained as null/Not recorded instead of estimated. In particular, absent training times and device logs are not imputed.",
        "- Phase 8 ends with comparative evaluation; Phase 9 is outside scope.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = write_phase8_outputs()
    print(json.dumps({key: result[key] for key in ("classifier_count", "retrieval_methods", "error_records", "chart_count", "output_files")}, indent=2, ensure_ascii=False))
