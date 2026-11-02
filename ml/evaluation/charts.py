"""Publication-ready Phase 8 charts driven only by consolidated JSON records."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_chart_data(classification: dict, retrieval: dict, efficiency: dict) -> dict:
    models = classification["models"]
    return {
        "classification_macro_f1": [
            {"model_id": row["model_id"], "value": row["test_metrics"]["macro_f1"]} for row in models
        ],
        "classification_accuracy": [
            {"model_id": row["model_id"], "value": row["test_metrics"]["accuracy"]} for row in models
        ],
        "macro_vs_weighted_f1": [
            {
                "model_id": row["model_id"],
                "macro_f1": row["test_metrics"]["macro_f1"],
                "weighted_f1": row["test_metrics"]["weighted_f1"],
            }
            for row in models
        ],
        "classwise_f1": [
            {
                "model_id": row["model_id"],
                "class_name": metric["name"],
                "label": metric["label"],
                "f1": metric["f1"],
            }
            for row in models
            for metric in row["test_metrics"]["per_class"]
        ],
        "confusion_matrices": {
            row["model_id"]: row["test_metrics"]["confusion_matrix"] for row in models
        },
        "retrieval_macro_metrics": retrieval["macro_average"],
        "retrieval_query_ndcg": [
            {"query_id": query_id, "method": method, "ndcg": metrics["ndcg"]}
            for method, result in retrieval["methods"].items()
            for query_id, metrics in result["per_query"].items()
        ],
        "training_seconds_vs_macro_f1": [
            {
                "model_id": row["model_id"],
                "training_seconds": row["training_seconds"],
                "macro_f1": row["test_macro_f1"],
            }
            for row in efficiency["models"]
            if row["training_seconds"] is not None and row["test_macro_f1"] is not None
        ],
        "artifact_size_vs_macro_f1": [
            {
                "model_id": row["model_id"],
                "artifact_bytes": row["artifact_bytes"],
                "size_basis": row["size_basis"],
                "macro_f1": row["test_macro_f1"],
            }
            for row in efficiency["models"]
            if row["artifact_bytes"] is not None and row["test_macro_f1"] is not None
        ],
        "source_class_order": classification["metadata"]["label_order"],
    }


def generate_charts(classification: dict, retrieval: dict, efficiency: dict, output_dir: str | Path) -> dict:
    """Write charts plus chart-data JSON. Returned manifest records every plot."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = make_chart_data(classification, retrieval, efficiency)
    data_path = output / "chart_data.json"
    data_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    models = classification["models"]
    model_ids = [row["model_id"] for row in models]
    short_names = [row["short_name"] for row in models]
    metrics_by_model = {row["model_id"]: row["test_metrics"] for row in models}
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 240, "font.size": 9, "axes.titlesize": 12, "axes.titleweight": "bold"})
    manifest: list[dict] = []

    def save(name: str, title: str, description: str, draw) -> None:
        fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
        draw(fig, ax)
        ax.set_title(title, pad=12, fontweight="bold")
        fig.savefig(output / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest.append({"file": name, "title": title, "description": description, "data_file": data_path.name})

    def classification_bars(field: str, ylabel: str, ylim: tuple[float, float], color: str):
        def draw(_fig, ax):
            values = [metrics_by_model[row["model_id"]][field] for row in models]
            bars = ax.bar(short_names, values, color=color, edgecolor="#ffffff", linewidth=0.7)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.22)
            ax.tick_params(axis="x", labelrotation=35)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        return draw

    save("classification_macro_f1.png", "Test macro F1 by saved experiment", "Independent fixed-label macro F1 recomputed from saved test predictions.", classification_bars("macro_f1", "Macro F1", (0.68, 1.01), "#3569a8"))
    save("classification_accuracy.png", "Test accuracy by saved experiment", "Accuracy recomputed from the same 2,000 aligned test predictions.", classification_bars("accuracy", "Accuracy", (0.68, 1.01), "#4d9b78"))

    def macro_weighted(_fig, ax):
        x = np.arange(len(models))
        width = 0.38
        macro = [metrics_by_model[row["model_id"]]["macro_f1"] for row in models]
        weighted = [metrics_by_model[row["model_id"]]["weighted_f1"] for row in models]
        ax.bar(x - width / 2, macro, width, label="Macro F1", color="#3569a8")
        ax.bar(x + width / 2, weighted, width, label="Weighted F1", color="#db9a32")
        ax.set_xticks(x, short_names, rotation=35, ha="right")
        ax.set_ylabel("F1")
        ax.set_ylim(0.68, 1.01)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    save("macro_vs_weighted_f1.png", "Macro and weighted test F1", "Both metrics use fixed labels; test support is 500 per class.", macro_weighted)

    def classwise(_fig, ax):
        for label, name in enumerate(classification["metadata"]["label_names"].values()):
            values = [next(item["f1"] for item in metrics_by_model[mid]["per_class"] if item["label"] == label) for mid in model_ids]
            ax.plot(short_names, values, marker="o", linewidth=1.7, markersize=4, label=name)
        ax.set_ylabel("Class F1")
        ax.set_ylim(0.55, 1.01)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", labelrotation=35)
        ax.legend(ncol=2, frameon=False, loc="lower left")
    save("classwise_f1.png", "Per-class test F1", "One-vs-rest F1 for all four labels, in original Phase 1 label order.", classwise)

    primary_models = [
        ("P3/tfidf_svm", "confusion_matrix_tfidf_svm.png"),
        ("P5/bilstm_attention", "confusion_matrix_bilstm_attention.png"),
        ("P6/custom_transformer", "confusion_matrix_custom_transformer.png"),
        ("P7/minilm_finetuned", "confusion_matrix_minilm_finetuned.png"),
    ]
    labels = list(classification["metadata"]["label_names"].values())
    by_id = {row["model_id"]: row for row in models}
    for model_id, filename in primary_models:
        matrix = np.asarray(by_id[model_id]["test_metrics"]["confusion_matrix"], dtype=int)

        def confusion(_fig, ax, matrix=matrix):
            image = ax.imshow(matrix, cmap="Blues")
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Records")
            ax.set_xticks(range(4), labels, rotation=25, ha="right")
            ax.set_yticks(range(4), labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            threshold = matrix.max() / 2 if matrix.size else 0
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "#18222e", fontsize=10)
        save(filename, f"Confusion matrix: {by_id[model_id]['short_name']}", "Counts are drawn directly from saved-prediction evaluation in the fixed label order.", confusion)

    retrieval_methods = list(retrieval["methods"])
    retrieval_labels = [retrieval["methods"][key]["display_name"].replace("Phase 4 ", "").replace("Phase 7 ", "") for key in retrieval_methods]
    retrieval_metrics = ("precision", "recall", "mrr", "ndcg")

    def retrieval_macro(_fig, ax):
        x = np.arange(len(retrieval_methods))
        width = 0.19
        for i, metric in enumerate(retrieval_metrics):
            values = [retrieval["methods"][name]["macro_average"][metric] for name in retrieval_methods]
            ax.bar(x + (i - 1.5) * width, values, width, label=metric.upper() if metric != "ndcg" else "nDCG")
        ax.set_xticks(x, retrieval_labels)
        ax.set_ylabel("Macro average at k=5")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, ncol=4)
    save("retrieval_macro_metrics.png", "Retrieval quality on the shared judgment pool", "P@5, Recall@5, MRR@5, and nDCG@5 use the same three queries and grades.", retrieval_macro)

    def retrieval_queries(_fig, ax):
        methods = list(retrieval["methods"])
        query_ids = retrieval["query_ids"]
        x = np.arange(len(query_ids))
        width = 0.22
        for i, method in enumerate(methods):
            vals = [retrieval["methods"][method]["per_query"][qid]["ndcg"] for qid in query_ids]
            label = retrieval["methods"][method]["display_name"].replace("Phase 4 ", "").replace("Phase 7 ", "")
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=label)
        ax.set_xticks(x, query_ids)
        ax.set_ylabel("nDCG@5")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    save("retrieval_query_ndcg.png", "Per-query retrieval nDCG@5", "Per-query values expose the three-query variability hidden by macro averages.", retrieval_queries)

    train_rows = data["training_seconds_vs_macro_f1"]

    def training(_fig, ax):
        for row in train_rows:
            ax.scatter(row["training_seconds"], row["macro_f1"], s=40, alpha=0.85)
            ax.annotate(row["model_id"].split("/", 1)[1], (row["training_seconds"], row["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Recorded training seconds")
        ax.set_ylabel("Test macro F1")
        ax.grid(alpha=0.22)
    save("training_time_vs_macro_f1.png", "Recorded training time and test macro F1", "Only experiments with a saved training-time measurement are plotted; timing environments differ.", training)

    size_rows = data["artifact_size_vs_macro_f1"]

    def size_scatter(_fig, ax):
        for row in size_rows:
            ax.scatter(row["artifact_bytes"] / (1024 * 1024), row["macro_f1"], s=42, alpha=0.82)
            ax.annotate(row["model_id"].split("/", 1)[1], (row["artifact_bytes"] / (1024 * 1024), row["macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Recorded model/checkpoint artifact size (MiB)")
        ax.set_ylabel("Test macro F1")
        ax.grid(alpha=0.22)
    save("artifact_size_vs_macro_f1.png", "Test macro F1 and recorded model artifact size", "Storage bases differ by phase and are identified in efficiency_results.json.", size_scatter)

    result = {
        "chart_count": len(manifest),
        "charts": manifest,
        "chart_data_file": data_path.name,
        "source_record_count": len(models),
        "source_model_ids": model_ids,
    }
    (output / "chart_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
