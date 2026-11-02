"""Create traceable Phase 6 comparisons and error-analysis artifacts."""

from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, f1_score
from ml.models.attention.train import load_model_checkpoint


ROOT = Path("artifacts/data")
PHASE6 = ROOT / "phase6"
NAMES = ["Algorithms", "NLP/AI", "Systems/Infra", "Vision"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_map(path: Path) -> dict[str, dict]:
    predictions = read_json(path)
    mapped = {str(row["arxiv_id"]): row for row in predictions}
    if len(mapped) != len(predictions):
        raise ValueError(f"Duplicate paper IDs in predictions: {path}")
    return mapped


def _verify_and_recompute(
    metric_path: Path, prediction_path: Path, expected_ids: set[str]
) -> tuple[dict, dict[str, dict]]:
    metrics = read_json(metric_path)
    predictions = _prediction_map(prediction_path)
    if set(predictions) != expected_ids:
        raise ValueError(f"Prediction IDs do not match the Phase 1 test split: {prediction_path}")
    truth = [int(predictions[paper_id]["actual"]) for paper_id in sorted(expected_ids)]
    predicted = [int(predictions[paper_id]["predicted"]) for paper_id in sorted(expected_ids)]
    report = classification_report(
        truth,
        predicted,
        labels=list(range(4)),
        target_names=NAMES,
        output_dict=True,
        zero_division=0,
    )
    if not np.isclose(metrics["accuracy"], accuracy_score(truth, predicted), atol=1e-12):
        raise ValueError(f"Saved accuracy does not reproduce from predictions: {metric_path}")
    if not np.isclose(metrics["macro_f1"], f1_score(truth, predicted, average="macro", zero_division=0), atol=1e-12):
        raise ValueError(f"Saved macro F1 does not reproduce from predictions: {metric_path}")
    return {**metrics, "recomputed_report": report}, predictions


def _phase3_fit_seconds() -> float:
    comparison = Path("docs/phase3_model_comparison.md").read_text(encoding="utf-8")
    for line in comparison.splitlines():
        if line.startswith("| tfidf_svm |"):
            fields = [field.strip() for field in line.strip("| ").split("|")]
            return float(fields[5])
    raise ValueError("Could not find the Phase 3 TF-IDF SVM fit time in its saved comparison")


def comparison_and_error_analysis() -> tuple[dict, dict]:
    phase6_metrics_path = PHASE6 / "test/custom_transformer.json"
    phase6_predictions_path = PHASE6 / "test/custom_transformer_predictions.json"
    phase6_rows = read_json(phase6_predictions_path)
    source_test_ids = set()
    with (ROOT / "phase1/test.jsonl").open(encoding="utf-8") as stream:
        source_test_ids = {str(json.loads(line)["arxiv_id"]) for line in stream}
    phase6_ids = {str(row["arxiv_id"]) for row in phase6_rows}
    if len(phase6_ids) != len(phase6_rows):
        raise ValueError("Duplicate IDs in Phase 6 predictions")
    if phase6_ids != source_test_ids:
        raise ValueError("Phase 6 predictions do not match the original Phase 1 test IDs")

    baseline_specs = [
        (
            "Phase 3 TF-IDF + Linear SVM",
            "phase3/test/tfidf_svm.json",
            "phase3/test/tfidf_svm_predictions.json",
            "phase3/validation/tfidf_svm.json",
        )
    ]
    phase5_selection = read_json(ROOT / "phase5/selection.json")
    for architecture in ("rnn", "lstm", "gru", "bilstm", "bilstm_attention"):
        baseline_specs.append(
            (
                f"Phase 5 {architecture}",
                f"phase5/test/{architecture}.json",
                f"phase5/test/{architecture}_predictions.json",
                f"phase5/validation/{architecture}.json",
            )
        )

    comparison_rows = []
    prediction_maps = {}
    phase3_model = joblib.load(ROOT / "phase3/models/tfidf_svm.joblib")
    if not hasattr(phase3_model, "coef_") or not hasattr(phase3_model, "intercept_"):
        raise ValueError("Saved Phase 3 SVM does not expose its trained linear parameters")
    phase3_parameters = int(phase3_model.coef_.size + phase3_model.intercept_.size)
    phase3_bytes = (ROOT / "phase3/models/tfidf_svm.joblib").stat().st_size

    for name, metric_relative, prediction_relative, validation_relative in baseline_specs:
        metric_path = ROOT / metric_relative
        prediction_path = ROOT / prediction_relative
        test_metrics, predictions = _verify_and_recompute(metric_path, prediction_path, phase6_ids)
        validation_metrics = read_json(ROOT / validation_relative)
        prediction_maps[name] = predictions
        if name.startswith("Phase 3"):
            parameters = phase3_parameters
            model_bytes = phase3_bytes
            training_seconds = _phase3_fit_seconds()
            latency = {
                "recorded_test_seconds_per_document": test_metrics.get("seconds_per_document"),
                "description": "Saved Phase 3 classifier prediction timing; excludes TF-IDF feature extraction.",
            }
        else:
            architecture = name.removeprefix("Phase 5 ")
            selected = phase5_selection[architecture]
            checkpoint = (
                ROOT
                / "phase5/runs"
                / f"{architecture}_{selected['config']['learning_rate']}"
                / "best.pt"
            )
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            parameters = sum(value.numel() for value in saved["model"].values())
            model_bytes = checkpoint.stat().st_size
            training_seconds = float(selected["training_seconds"])
            timings = test_metrics["inference_seconds_repeated"]
            latency = {
                "mean_recorded_test_seconds_per_document": float(np.mean(timings) / len(predictions)),
                "description": test_metrics.get("latency_description", "Recorded Phase 5 inference timing."),
            }
        comparison_rows.append(
            {
                "model": name,
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_weighted_f1": float(test_metrics["weighted_f1"]),
                "training_seconds": training_seconds,
                "inference_latency": latency,
                "parameter_count": int(parameters),
                "model_or_checkpoint_bytes": int(model_bytes),
                "test_metrics_recomputed_from_saved_predictions": True,
            }
        )

    phase6_test, phase6_predictions = _verify_and_recompute(
        phase6_metrics_path, phase6_predictions_path, phase6_ids
    )
    selected = read_json(PHASE6 / "selection.json")["selected"]
    phase6_validation = read_json(PHASE6 / "validation/custom_transformer.json")
    checkpoint = Path(selected["checkpoint"])
    model, _ = load_model_checkpoint(checkpoint, "cpu")
    phase6_parameters = sum(parameter.numel() for parameter in model.parameters())
    phase6_latency = phase6_test["inference_latency"]
    comparison_rows.append(
        {
            "model": "Phase 6 custom Transformer",
            "validation_macro_f1": float(phase6_validation["macro_f1"]),
            "test_macro_f1": float(phase6_test["macro_f1"]),
            "test_accuracy": float(phase6_test["accuracy"]),
            "test_weighted_f1": float(phase6_test["weighted_f1"]),
            "training_seconds": float(selected["training_seconds"]),
            "inference_latency": {
                "model_only_seconds_per_document": phase6_latency["model_only_mean_seconds_per_document"],
                "transfer_included_seconds_per_document": phase6_latency[
                    "data_transfer_and_prediction_mean_seconds_per_document"
                ],
                "description": "Five repeated test-set passes; model-only uses preloaded token tensors, transfer-included adds host/device and prediction transfers; no tokenization.",
            },
            "parameter_count": int(phase6_parameters),
            "model_or_checkpoint_bytes": checkpoint.stat().st_size,
            "test_metrics_recomputed_from_saved_predictions": True,
        }
    )
    prediction_maps["Phase 6 custom Transformer"] = phase6_predictions

    # Confirm every compared system predicts the same original test records and labels.
    expected_truth = {paper_id: int(row["actual"]) for paper_id, row in phase6_predictions.items()}
    for name, predictions in prediction_maps.items():
        if {paper_id: int(row["actual"]) for paper_id, row in predictions.items()} != expected_truth:
            raise ValueError(f"Ground-truth labels differ across systems: {name}")

    matrix = phase6_test["confusion_matrix"]
    mistakes = [row for row in phase6_rows if row["actual"] != row["predicted"]]
    confusions = []
    for actual in range(4):
        for predicted in range(4):
            if actual != predicted and matrix[actual][predicted]:
                confusions.append(
                    {
                        "actual": NAMES[actual],
                        "predicted": NAMES[predicted],
                        "count": int(matrix[actual][predicted]),
                    }
                )
    confusions.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))
    weakest = sorted(
        (
            {"class": name, "f1": float(phase6_test["per_class"][name]["f1-score"])}
            for name in NAMES
        ),
        key=lambda item: item["f1"],
    )

    model_names = list(prediction_maps)
    error_sets = {
        name: {
            paper_id
            for paper_id, row in predictions.items()
            if int(row["actual"]) != int(row["predicted"])
        }
        for name, predictions in prediction_maps.items()
    }
    phase3_name = "Phase 3 TF-IDF + Linear SVM"
    attention_name = "Phase 5 bilstm_attention"
    comparisons = {}
    for baseline_name in (phase3_name, attention_name):
        baseline = prediction_maps[baseline_name]
        current = phase6_predictions
        comparisons[baseline_name] = {
            "transformer_correct_baseline_wrong": sum(
                phase_id not in error_sets["Phase 6 custom Transformer"] and phase_id in error_sets[baseline_name]
                for phase_id in phase6_ids
            ),
            "baseline_correct_transformer_wrong": sum(
                phase_id in error_sets["Phase 6 custom Transformer"] and phase_id not in error_sets[baseline_name]
                for phase_id in phase6_ids
            ),
            "both_wrong": len(error_sets["Phase 6 custom Transformer"] & error_sets[baseline_name]),
            "both_correct": len(phase6_ids - error_sets["Phase 6 custom Transformer"] - error_sets[baseline_name]),
        }
    prior_names = [name for name in model_names if name != "Phase 6 custom Transformer"]
    error_consensus = set.intersection(*(error_sets[name] for name in prior_names))
    all_model_consensus = set.intersection(*(error_sets[name] for name in model_names))

    high_confidence_errors = sorted(
        (
            {
                "arxiv_id": row["arxiv_id"],
                "actual": NAMES[int(row["actual"])],
                "predicted": NAMES[int(row["predicted"])],
                "predicted_probability": float(row["probabilities"][NAMES[int(row["predicted"])] ]),
                "title": row["title"],
                "abstract_excerpt": row["abstract"][:600],
            }
            for row in mistakes
        ),
        key=lambda item: (-item["predicted_probability"], item["arxiv_id"]),
    )[:12]
    confusion_examples = []
    for confusion in confusions[:3]:
        actual_id = NAMES.index(confusion["actual"])
        predicted_id = NAMES.index(confusion["predicted"])
        examples = [
            row
            for row in mistakes
            if int(row["actual"]) == actual_id and int(row["predicted"]) == predicted_id
        ]
        examples.sort(
            key=lambda row: (
                -float(row["probabilities"][NAMES[predicted_id]]),
                str(row["arxiv_id"]),
            )
        )
        confusion_examples.append(
            {
                "actual": confusion["actual"],
                "predicted": confusion["predicted"],
                "count": confusion["count"],
                "examples": [
                    {
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "predicted_probability": float(
                            row["probabilities"][NAMES[predicted_id]]
                        ),
                        "abstract_excerpt": row["abstract"][:450],
                    }
                    for row in examples[:3]
                ],
            }
        )
    term_counts = Counter()
    for row in mistakes:
        text = f"{row['title']} {row['abstract']}".lower()
        terms = [
            word
            for word in text.replace("/", " ").replace("-", " ").split()
            if word.isalpha() and len(word) > 2 and word not in ENGLISH_STOP_WORDS
        ]
        term_counts.update(set(terms))

    error_analysis = {
        "test_records": len(phase6_rows),
        "misclassified_records": len(mistakes),
        "most_confused_class_pairs": confusions,
        "representative_examples_by_most_common_confusions": confusion_examples,
        "weakest_class_f1_first": weakest,
        "comparison_to_existing_models": comparisons,
        "records_misclassified_by_all_six_phase3_and_phase5_models": len(error_consensus),
        "records_misclassified_by_all_seven_phase3_phase5_and_phase6_models": len(all_model_consensus),
        "ids_misclassified_by_all_seven": sorted(all_model_consensus),
        "high_confidence_misclassifications": high_confidence_errors,
        "common_surface_terms_in_errors": [
            {"term": term, "error_records_containing_term": count}
            for term, count in term_counts.most_common(20)
        ],
        "interpretation_limit": "Surface-term counts and selected error examples are descriptive of this test split; they do not establish causal feature importance.",
    }

    comparison = {
        "dataset": {
            "source": "vrajpat3ll/arxiv-cs-papers",
            "revision": read_json(ROOT / "phase2/metadata.json")["dataset_revision"],
            "test_split": "original Phase 1 test split, matched by arxiv_id and verified actual label",
            "test_records": len(phase6_rows),
            "class_mapping": NAMES,
        },
        "latency_caveat": "Timing sources are not fully comparable: Phase 3 excludes feature extraction; Phase 5 includes batching/device/prediction transfers but excludes tokenization; Phase 6 reports both preloaded model-only and transfer-included passes, excluding text tokenization.",
        "results": comparison_rows,
    }

    PHASE6.mkdir(parents=True, exist_ok=True)
    (PHASE6 / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (PHASE6 / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison, error_analysis


def _fmt(value: float | int | None, digits: int = 4) -> str:
    return "not available" if value is None else f"{value:.{digits}f}"


def _write_training_curves() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for history_path in sorted((PHASE6 / "runs").glob("custom_transformer_lr_*/history.json")):
        history = read_json(history_path)
        label = history_path.parent.name.removeprefix("custom_transformer_lr_")
        epochs = [row["epoch"] for row in history]
        loss_axis.plot(epochs, [row["train"]["loss"] for row in history], marker="o", label=f"lr {label} train")
        loss_axis.plot(epochs, [row["validation"]["loss"] for row in history], marker="s", linestyle="--", label=f"lr {label} validation")
        f1_axis.plot(epochs, [row["validation"]["macro_f1"] for row in history], marker="o", label=f"lr {label}")
    loss_axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.legend(fontsize=8)
    f1_axis.set(title="Validation model-selection metric", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    f1_axis.legend(title="Learning rate")
    figure.tight_layout()
    destination = PHASE6 / "figures/training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _write_comparison_figure(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["model"].replace(" + ", "\n+ ").replace("custom Transformer", "custom\nTransformer") for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(
        positions - width / 2,
        [row["validation_macro_f1"] for row in rows],
        width,
        label="Validation macro F1",
    )
    axis.bar(
        positions + width / 2,
        [row["test_macro_f1"] for row in rows],
        width,
        label="Test macro F1",
    )
    axis.set(
        xticks=positions,
        xticklabels=names,
        ylabel="Macro F1",
        ylim=(0, 1),
        title="Phase 3–6 classification comparison",
    )
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    destination = PHASE6 / "figures/model_comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def write_documentation(comparison: dict, error_analysis: dict) -> None:
    rows = comparison["results"]
    table = [
        "| Model | Validation macro F1 | Test macro F1 | Accuracy | Weighted F1 | Training s | Inference latency | Parameters | Model/checkpoint bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        latency = row["inference_latency"]
        if row["model"].startswith("Phase 6"):
            latency_text = (
                f"{latency['model_only_seconds_per_document'] * 1000:.3f} ms/doc model-only; "
                f"{latency['transfer_included_seconds_per_document'] * 1000:.3f} ms/doc transfer-included"
            )
        else:
            seconds = latency.get(
                "recorded_test_seconds_per_document",
                latency.get("mean_recorded_test_seconds_per_document"),
            )
            latency_text = f"{seconds * 1000:.3f} ms/doc (recorded method)" if seconds is not None else "not available"
        table.append(
            "| {model} | {validation} | {test} | {accuracy} | {weighted} | {training} | {latency} | {parameters:,} | {bytes:,} |".format(
                model=row["model"],
                validation=_fmt(row["validation_macro_f1"]),
                test=_fmt(row["test_macro_f1"]),
                accuracy=_fmt(row["test_accuracy"]),
                weighted=_fmt(row["test_weighted_f1"]),
                training=_fmt(row["training_seconds"], 2),
                latency=latency_text,
                parameters=row["parameter_count"],
                bytes=row["model_or_checkpoint_bytes"],
            )
        )

    top_confusions = error_analysis["most_confused_class_pairs"][:5]
    confusion_text = "; ".join(
        f"{item['actual']} → {item['predicted']} ({item['count']})" for item in top_confusions
    ) or "No misclassifications"
    weakest = error_analysis["weakest_class_f1_first"][0]
    comparison_text = []
    for name, result in error_analysis["comparison_to_existing_models"].items():
        comparison_text.append(
            f"Against {name}: Transformer-only correct {result['transformer_correct_baseline_wrong']}, "
            f"baseline-only correct {result['baseline_correct_transformer_wrong']}, "
            f"both wrong {result['both_wrong']}, both correct {result['both_correct']}."
        )
    error_samples = []
    for row in error_analysis["high_confidence_misclassifications"][:5]:
        excerpt = " ".join(row["abstract_excerpt"].split())
        error_samples.append(
            f"- `{row['arxiv_id']}`: {row['actual']} predicted as {row['predicted']} "
            f"(confidence {row['predicted_probability']:.3f}); **{row['title']}** — {excerpt}"
        )
    frequent_confusion_text = []
    for pair in error_analysis["representative_examples_by_most_common_confusions"]:
        for example in pair["examples"][:2]:
            excerpt = " ".join(example["abstract_excerpt"].split())
            frequent_confusion_text.append(
                f"- {pair['actual']} predicted as {pair['predicted']} "
                f"(`{example['arxiv_id']}`, confidence {example['predicted_probability']:.3f}): "
                f"**{example['title']}** — {excerpt}"
            )

    selected = read_json(PHASE6 / "selection.json")["selected"]
    dataset = read_json(PHASE6 / "dataset_reuse.json")
    metrics = read_json(PHASE6 / "test/custom_transformer.json")
    execution = read_json(Path("docs/phase6_execution_checkpoint.json"))
    _write_training_curves()
    _write_comparison_figure(rows)
    phase3_score = next(
        row["test_macro_f1"] for row in rows if row["model"] == "Phase 3 TF-IDF + Linear SVM"
    )
    phase5_attention_score = next(
        row["test_macro_f1"] for row in rows if row["model"] == "Phase 5 bilstm_attention"
    )
    score_note = (
        f"The Transformer test macro F1 ({metrics['macro_f1']:.4f}) is "
        f"{metrics['macro_f1'] - phase3_score:+.4f} relative to the Phase 3 TF-IDF SVM "
        f"({phase3_score:.4f}) and {metrics['macro_f1'] - phase5_attention_score:+.4f} "
        f"relative to Phase 5 BiLSTM attention ({phase5_attention_score:.4f}). The latter "
        "is a small measured difference on this split, not evidence of general superiority."
    )
    per_class_table = [
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    per_class_table.extend(
        "| {name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support:.0f} |".format(
            name=name,
            precision=values["precision"],
            recall=values["recall"],
            f1=values["f1-score"],
            support=values["support"],
        )
        for name, values in metrics["per_class"].items()
    )
    docs = f"""# Phase 6 — Attention and custom Transformer

## Implementation

The custom PyTorch classifier is implemented in `ml/models/attention/components.py`. It uses the existing Phase 2 token IDs, a trainable token embedding with padding index 0, fixed sinusoidal positional encodings, explicit Q/K/V projections, scaled dot-product attention, four-way multi-head self-attention, two post-normalized residual encoder blocks, ReLU position-wise feed-forward layers, and masked mean pooling into four-class logits. No pretrained Transformer is used. Padding is masked as both keys and queries, and all-padding rows remain finite with zero pooled representations. The existing Phase 5 `RecurrentClassifier` `bilstm_attention` remains the project’s BiLSTM attention-pooling implementation and is reused unchanged.

## Data and reproducibility

- Source: `{dataset['source']}` at revision `{dataset['revision']}`.
- Reused unchanged Phase 1 splits and Phase 2 vocabulary/encoded arrays; vocabulary size `{dataset['vocabulary_size']}`, sequence length {dataset['sequence_length']}, seed {dataset['random_seed']}.
- Original label mapping: {', '.join(f"{key}: {value}" for key, value in dataset['label_mapping'].items())}.
- Split counts: train {dataset['splits']['train']['records']}, validation {dataset['splits']['validation']['records']}, test {dataset['splits']['test']['records']}.
- Class distributions, token-length summaries, and the test-after-selection check are in `artifacts/data/phase6/dataset_reuse.json`.
- Vocabulary and text normalization were not refit. Inference calls the existing `tokens`, `document`, and `encode` functions from Phase 2.

## Experiments

The plan does not specify a numeric Transformer size or learning-rate grid. A bounded controlled grid was used: fixed model dimension 128, four heads, two encoder layers, feed-forward dimension 256, dropout 0.1, batch size 32 (systematically halved only if CUDA OOM occurs), Adam, gradient clipping 1.0, seed 42, up to four epochs with patience two, and learning rates 0.001 and 0.002. The validation split selected the checkpoint by macro F1. The test split was opened after this selection.

Selected run: `{selected['name']}`, best epoch {selected['best_epoch']}, batch size {selected['effective_batch_size']}, device `{selected['device']}`, validation macro F1 {_fmt(selected['best_validation_macro_f1'])}, training time {_fmt(selected['training_seconds'], 2)} seconds. Runtime: Python {execution['environment']['python_training'].split()[0]}, PyTorch {selected['config']['torch_version']}, CUDA {selected['config']['cuda_version']}, {selected['config']['gpu_name']}. Peak allocated GPU memory: {selected.get('peak_gpu_allocated_bytes') if selected.get('peak_gpu_allocated_bytes') is not None else 'not recorded'} bytes. Test macro F1 {_fmt(metrics['macro_f1'])}, accuracy {_fmt(metrics['accuracy'])}, weighted F1 {_fmt(metrics['weighted_f1'])}. The selected checkpoint reload consistency check passed: {metrics['checkpoint_reload_consistency']['passed']} (maximum probability difference {metrics['checkpoint_reload_consistency']['max_probability_absolute_difference']:.3g}).

## Measured comparison

All listed systems use the same held-out test IDs and labels. Saved metrics were recomputed from their prediction files. Timing methods differ as described below.

{chr(10).join(table)}

{comparison['latency_caveat']}

{score_note}

## Error analysis

The Transformer made {error_analysis['misclassified_records']} errors out of {error_analysis['test_records']} test records. Most frequent confusions: {confusion_text}. Weakest class F1 was {weakest['class']} ({weakest['f1']:.4f}). {error_analysis['records_misclassified_by_all_six_phase3_and_phase5_models']} test records were misclassified by every Phase 3/5 comparator; {error_analysis['records_misclassified_by_all_seven_phase3_phase5_and_phase6_models']} were misclassified by all seven systems.

The most frequent errors are between NLP/AI and Vision in both directions. Inspected examples include papers on parameter-efficient adaptation of visual foundation models, cartoon visual question answering, and long-video frame selection. Systems/Vision confusions include radar-camera perception and vision-based wireless beam prediction. These titles and abstracts contain cross-domain task language; the observations are qualitative and do not establish why the classifier made each decision.

Per-class test metrics:

{chr(10).join(per_class_table)}

{chr(10).join(comparison_text)}

Selected high-confidence errors for inspection:

{chr(10).join(error_samples)}

Common surface terms among mistakes are saved in `artifacts/data/phase6/error_analysis.json`; they are descriptive only. The examples and counts describe this held-out split and do not establish causal feature importance.

Representative examples from the most frequent confusion pairs:

{chr(10).join(frequent_confusion_text)}

## Artifacts and verification

- Best checkpoints and full epoch histories: `artifacts/data/phase6/runs/`.
- Validation/test metrics and per-paper predictions: `artifacts/data/phase6/validation/` and `artifacts/data/phase6/test/`.
- Confusion matrix: `artifacts/data/phase6/figures/test_confusion_matrix.png`.
- Training curves and the cross-phase macro-F1 figure: `artifacts/data/phase6/figures/training_curves.png` and `artifacts/data/phase6/figures/model_comparison.png`.
- Comparison and error analysis: `artifacts/data/phase6/comparison.json` and `artifacts/data/phase6/error_analysis.json`.
- Persistent progress: `docs/phase6_execution_checkpoint.json`.
- Test suites and exact results are recorded in the execution checkpoint and final completion report.
- Protected Phase 1–5 files, `docs/plan.md`, and `docs/AGENTS.md` were hashed before implementation and checked again after experiments.

## Limitations

This is one seed and one fixed, bounded learning-rate grid on a single class-balanced test split. GPU timings are hardware-specific; Phase 3 excludes feature extraction and Phases 5/6 do not include tokenization. The Phase 6 model uses fixed sinusoidal positions and masked-mean pooling, not a learned classification token. No mixed precision was used.
"""
    Path("docs/phase6_documentation.md").write_text(docs, encoding="utf-8")

    comparison_md = "# Phase 6 measured model comparison\n\n" + "\n".join(table)
    comparison_md += (
        "\n\n"
        + comparison["latency_caveat"]
        + "\n\n"
        + score_note
        + "\n\nAll metrics were independently recomputed from saved test predictions and matched paper IDs/labels."
    )
    Path("docs/phase6_comparison.md").write_text(comparison_md, encoding="utf-8")

    analysis_md = (
        "# Phase 6 test error analysis\n\n"
        f"Errors: {error_analysis['misclassified_records']} / {error_analysis['test_records']}. "
        f"Weakest class: {weakest['class']} (F1 {weakest['f1']:.4f}).\n\n"
        f"Most frequent confusions: {confusion_text}.\n\n"
        + "\n".join(comparison_text)
        + "\n\n## High-confidence examples\n\n"
        + "\n".join(error_samples)
        + "\n\nSurface-term counts and examples are descriptive; they do not establish causal feature importance."
    )
    Path("docs/phase6_error_analysis.md").write_text(analysis_md, encoding="utf-8")


def main() -> None:
    comparison, errors = comparison_and_error_analysis()
    write_documentation(comparison, errors)
    print("Comparison rows:", len(comparison["results"]))
    print("Transformer errors:", errors["misclassified_records"])


if __name__ == "__main__":
    main()
