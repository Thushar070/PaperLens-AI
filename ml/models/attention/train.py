"""Train the custom Phase 6 Transformer on the saved Phase 2 sequences."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.attention.components import CustomTransformerClassifier


ROOT = Path("artifacts/data")
OUTPUT = ROOT / "phase6"
EXECUTION_CHECKPOINT = Path("docs/phase6_execution_checkpoint.json")
CLASS_NAMES = ["Algorithms", "NLP/AI", "Systems/Infra", "Vision"]
PAD_INDEX = 0
LEARNING_RATES = (0.001, 0.002)
INITIAL_BATCH_SIZE = 32
MINIMUM_BATCH_SIZE = 4
EPOCHS = 4
PATIENCE = 2
SEED = 42


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def update_execution(task: str, **changes: object) -> None:
    state = json.loads(EXECUTION_CHECKPOINT.read_text(encoding="utf-8"))
    state["current_experiment"] = task
    state["latest_completed_milestone"] = task
    state.update(changes)
    atomic_json(EXECUTION_CHECKPOINT, state)


def verify_protected_baseline() -> None:
    state = json.loads(EXECUTION_CHECKPOINT.read_text(encoding="utf-8"))
    for relative, expected in state["protected_baseline_sha256"].items():
        path = Path(relative)
        if not path.is_file():
            raise RuntimeError(f"Protected Phase 1-5 input is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Protected Phase 1-5 input changed: {relative}")


def load_split(name: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    if name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {name}")
    sequence_path = ROOT / "phase2" / f"{name}_sequences.npy"
    record_path = ROOT / "phase1" / f"{name}.jsonl"
    sequences = np.load(sequence_path, allow_pickle=False)
    with record_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    if sequences.ndim != 2 or sequences.shape != (len(records), 290):
        raise ValueError(f"{name} records and saved 290-token sequences are misaligned")
    if any(not isinstance(row.get("arxiv_id"), str) for row in records):
        raise ValueError(f"{name} records must preserve paper identifiers")
    if len({row["arxiv_id"] for row in records}) != len(records):
        raise ValueError(f"{name} contains duplicate paper identifiers")
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError(f"{name} contains labels outside the original four-class mapping")
    if (sequences < 0).any():
        raise ValueError(f"{name} contains negative token IDs")
    nonpadding = sequences != PAD_INDEX
    if (nonpadding[:, 1:] & ~nonpadding[:, :-1]).any():
        raise ValueError(f"{name} contains sequences that are not right padded")
    if not np.array_equal(labels, np.asarray([int(row["label"]) for row in records])):
        raise ValueError(f"{name} label alignment failed")
    return torch.from_numpy(sequences.astype(np.int64, copy=False)), torch.from_numpy(labels), records


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    sequences: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(sequences, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def run_epoch(
    model: nn.Module,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    seed: int = SEED,
) -> dict:
    if len(sequences) == 0 or len(sequences) != len(labels):
        raise ValueError("Epoch input must contain aligned, nonempty sequences and labels")
    training = optimizer is not None
    model.train(training)
    loader = make_loader(sequences, labels, batch_size, device, training, seed)
    total_loss = 0.0
    predictions: list[int] = []
    truth: list[int] = []

    with torch.set_grad_enabled(training):
        for batch_ids, batch_labels in loader:
            length = max(1, int(batch_ids.ne(PAD_INDEX).sum(dim=1).max().item()))
            batch_ids = batch_ids[:, :length].to(device, non_blocking=device.type == "cuda")
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            if next(model.parameters()).device != device or batch_ids.device != device:
                raise RuntimeError("Model and training batch must use the selected device")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch_ids)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            truth.extend(batch_labels.cpu().tolist())

    return {
        "loss": total_loss / len(sequences),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
    }


def model_config(vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "model_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "feedforward_dim": 256,
        "num_classes": 4,
        "dropout": 0.1,
        "max_length": 290,
        "pad_idx": PAD_INDEX,
    }


def summarize_split(sequences: torch.Tensor, labels: torch.Tensor, mapping: dict) -> dict:
    lengths = sequences.ne(PAD_INDEX).sum(dim=1).numpy()
    return {
        "records": len(labels),
        "class_distribution": {
            mapping[str(label)]: int((labels == label).sum().item())
            for label in range(4)
        },
        "encoded_token_length": {
            "minimum": int(lengths.min()),
            "median": float(np.median(lengths)),
            "mean": float(np.mean(lengths)),
            "p95": int(np.quantile(lengths, 0.95)),
            "maximum": int(lengths.max()),
            "empty_sequences": int((lengths == 0).sum()),
            "at_maximum_length": int((lengths == sequences.shape[1]).sum()),
        },
    }


def load_model_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = CustomTransformerClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _model_state(model: nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_experiment(
    learning_rate: float,
    train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    config: dict,
    provenance: dict,
) -> dict:
    name = f"custom_transformer_lr_{learning_rate:g}"
    folder = OUTPUT / "runs" / name
    latest_path, best_path = folder / "latest.pt", folder / "best.pt"
    history_path, completed_path = folder / "history.json", folder / "completed.json"
    run_config = {
        "model_config": config,
        "learning_rate": learning_rate,
        "initial_batch_size": INITIAL_BATCH_SIZE,
        "seed": SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "optimizer": "Adam",
        "loss": "cross_entropy",
        "gradient_clip_norm": 1.0,
        "selection_metric": "validation_macro_f1",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset_source": provenance["dataset_source"],
        "dataset_revision": provenance["dataset_revision"],
        "train_records": len(train[0]),
        "validation_records": len(validation[0]),
    }
    atomic_json(folder / "config.json", run_config)
    if completed_path.exists() and latest_path.exists() and best_path.exists():
        return json.loads(completed_path.read_text(encoding="utf-8"))

    set_seed(SEED)
    model = CustomTransformerClassifier(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_weights = _model_state(model)
    initial_torch_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    history: list[dict] = []
    next_epoch = 0
    best_f1 = -1.0
    bad_epochs = 0
    effective_batch_size = INITIAL_BATCH_SIZE
    elapsed_seconds = 0.0

    def restore_last_saved() -> None:
        nonlocal optimizer, history, next_epoch, best_f1, bad_epochs, effective_batch_size, elapsed_seconds
        if latest_path.is_file():
            saved = torch.load(latest_path, map_location=device, weights_only=False)
            if saved["config"] != run_config:
                raise ValueError("Saved run configuration differs from the requested experiment")
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            history = saved["history"]
            next_epoch = saved["epoch"] + 1
            best_f1 = saved["best_validation_macro_f1"]
            bad_epochs = saved["bad_epochs"]
            effective_batch_size = saved["effective_batch_size"]
            elapsed_seconds = saved["elapsed_seconds"]
            torch.set_rng_state(saved["torch_rng_state"].cpu())
            if device.type == "cuda" and saved["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device)
        else:
            model.load_state_dict(initial_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            history = []
            next_epoch = 0
            best_f1 = -1.0
            bad_epochs = 0
            effective_batch_size = INITIAL_BATCH_SIZE
            elapsed_seconds = 0.0
            torch.set_rng_state(initial_torch_rng)
            if device.type == "cuda" and initial_cuda_rng is not None:
                torch.cuda.set_rng_state(initial_cuda_rng, device)

    restore_last_saved()
    epoch = next_epoch
    while epoch < EPOCHS:
        epoch_start = time.perf_counter()
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training_metrics = run_epoch(
                model, *train, device, effective_batch_size, optimizer, SEED + epoch
            )
            validation_metrics = run_epoch(
                model, *validation, device, effective_batch_size, seed=SEED
            )
        except torch.cuda.OutOfMemoryError as exc:
            if device.type != "cuda" or effective_batch_size <= MINIMUM_BATCH_SIZE:
                raise
            failed_attempt_seconds = time.perf_counter() - epoch_start
            elapsed_seconds += failed_attempt_seconds
            previous = effective_batch_size
            effective_batch_size = max(MINIMUM_BATCH_SIZE, effective_batch_size // 2)
            update_execution(
                f"{name}: CUDA out of memory at batch {previous}; retrying from the last epoch checkpoint at batch {effective_batch_size}",
                errors_and_resolutions=json.loads(
                    EXECUTION_CHECKPOINT.read_text(encoding="utf-8")
                ).get("errors_and_resolutions", [])
                + [{"experiment": name, "error": str(exc), "resolution": f"Restored the last saved epoch and reduced batch size to {effective_batch_size}."}],
            )
            torch.cuda.empty_cache()
            restore_last_saved()
            elapsed_seconds += failed_attempt_seconds
            effective_batch_size = max(MINIMUM_BATCH_SIZE, min(effective_batch_size, previous // 2))
            epoch = next_epoch
            continue

        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            bad_epochs = 0
            atomic_torch(
                best_path,
                {
                    "model_config": config,
                    "model_state_dict": model.state_dict(),
                    "validation": validation_metrics,
                    "epoch": epoch + 1,
                    "selection_metric": "validation_macro_f1",
                },
            )
        else:
            bad_epochs += 1

        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        row = {
            "epoch": epoch + 1,
            "train": training_metrics,
            "validation": validation_metrics,
            "device": str(device),
            "batch_size": effective_batch_size,
            "gpu_peak_allocated_bytes": peak_allocated,
        }
        history.append(row)
        elapsed_seconds += time.perf_counter() - epoch_start
        state = {
            "config": run_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "epoch": epoch,
            "best_validation_macro_f1": best_f1,
            "bad_epochs": bad_epochs,
            "effective_batch_size": effective_batch_size,
            "elapsed_seconds": elapsed_seconds,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        atomic_torch(latest_path, state)
        atomic_json(history_path, history)
        progress = json.loads(EXECUTION_CHECKPOINT.read_text(encoding="utf-8"))
        experiment_configs = [
            item
            for item in progress.get("experiment_configurations", [])
            if item.get("learning_rate") != learning_rate
        ]
        experiment_configs.append(run_config)
        update_execution(
            f"{name}: completed epoch {epoch + 1}/{EPOCHS}",
            current_experiment=name,
            experiment_configurations=experiment_configs,
        )
        print(name, json.dumps(row), flush=True)
        epoch += 1
        if bad_epochs >= PATIENCE:
            break

    result = {
        "name": name,
        "config": run_config,
        "best_validation_macro_f1": best_f1,
        "best_epoch": torch.load(best_path, map_location="cpu", weights_only=False)["epoch"],
        "epochs_completed": len(history),
        "effective_batch_size": effective_batch_size,
        "training_seconds": elapsed_seconds,
        "peak_gpu_allocated_bytes": max(
            (row["gpu_peak_allocated_bytes"] or 0 for row in history), default=0
        )
        or None,
        "device": str(device),
        "checkpoint": str(best_path),
        "checkpoint_bytes": best_path.stat().st_size,
        "history": str(history_path),
    }
    atomic_json(completed_path, result)
    return result


def _prediction_pass(
    model: nn.Module,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    records: list[dict],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], list[int], list[float]]:
    model.eval()
    predictions: list[dict] = []
    truth: list[int] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            ids = sequences[start : start + batch_size]
            ys = labels[start : start + batch_size]
            rows = records[start : start + batch_size]
            length = max(1, int(ids.ne(PAD_INDEX).sum(dim=1).max().item()))
            ids = ids[:, :length].to(device, non_blocking=device.type == "cuda")
            logits = model(ids)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            predicted = logits.argmax(dim=1).cpu().tolist()
            for row, actual, label, scores in zip(rows, ys.tolist(), predicted, probabilities):
                predictions.append(
                    {
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "actual": int(actual),
                        "predicted": int(label),
                        "probabilities": {CLASS_NAMES[i]: float(scores[i]) for i in range(4)},
                    }
                )
                truth.append(int(actual))
    return predictions, truth, [float(row["probabilities"][CLASS_NAMES[row["predicted"]]]) for row in predictions]


def _measure_latency(
    model: nn.Module,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    repeats: int = 5,
) -> dict:
    model.eval()
    with torch.no_grad():
        gpu_sequences = sequences.to(device)
        warm = gpu_sequences[: min(batch_size, len(gpu_sequences))]
        model(warm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_only = []
        for _ in range(repeats):
            start = time.perf_counter()
            for offset in range(0, len(gpu_sequences), batch_size):
                batch = gpu_sequences[offset : offset + batch_size]
                length = max(1, int(batch.ne(PAD_INDEX).sum(dim=1).max().item()))
                model(batch[:, :length])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_only.append(time.perf_counter() - start)

        transfer_included = []
        for _ in range(repeats):
            start = time.perf_counter()
            for offset in range(0, len(sequences), batch_size):
                batch = sequences[offset : offset + batch_size]
                length = max(1, int(batch.ne(PAD_INDEX).sum(dim=1).max().item()))
                model(batch[:, :length].to(device, non_blocking=device.type == "cuda")).argmax(dim=1).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_included.append(time.perf_counter() - start)

    documents = len(sequences)
    return {
        "device": str(device),
        "batch_size": batch_size,
        "repeats": repeats,
        "model_only_preloaded_tokens_seconds": model_only,
        "model_only_mean_seconds_per_document": float(np.mean(model_only) / documents),
        "model_only_mean_seconds_per_batch": float(np.mean(model_only) / math_ceil_div(documents, batch_size)),
        "data_transfer_and_prediction_seconds": transfer_included,
        "data_transfer_and_prediction_mean_seconds_per_document": float(np.mean(transfer_included) / documents),
        "includes_tokenization": False,
        "includes_device_transfer": False,
        "prediction_pipeline_includes_device_transfer": True,
    }


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def evaluate(
    model: nn.Module,
    sequences: torch.Tensor,
    labels: torch.Tensor,
    records: list[dict],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict, list[dict]]:
    predictions, truth, _ = _prediction_pass(model, sequences, labels, records, device, batch_size)
    predicted = [row["predicted"] for row in predictions]
    report = classification_report(
        truth,
        predicted,
        labels=list(range(4)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(4))).tolist(),
        "records": len(truth),
        "batch_size": batch_size,
    }
    metrics["inference_latency"] = _measure_latency(model, sequences, labels, device, batch_size)
    return metrics, predictions


def _write_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Phase 6 custom Transformer test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    threshold = max(max(row) for row in matrix) / 2
    for row in range(4):
        for column in range(4):
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
                color="white" if matrix[row][column] > threshold else "black",
            )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    verify_protected_baseline()
    metadata = json.loads((ROOT / "phase2/metadata.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((ROOT / "phase2/vocabulary.json").read_text(encoding="utf-8"))
    if len(vocabulary) != metadata["vocabulary_size"] or vocabulary.get("<PAD>") != PAD_INDEX:
        raise ValueError("Saved Phase 2 vocabulary does not match the Phase 2 metadata")
    if metadata["sequence_length"] != 290 or len(vocabulary) != 32887:
        raise ValueError("Unexpected Phase 2 sequence representation")
    label_map = json.loads((ROOT / "phase1/metadata.json").read_text(encoding="utf-8"))["label_mapping"]
    if [label_map[str(i)] for i in range(4)] != CLASS_NAMES:
        raise ValueError("Original Phase 1 label mapping differs from the expected saved mapping")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device) if device.type == "cuda" else (None, None)
    train_x, train_y, train_rows = load_split("train")
    validation_x, validation_y, validation_rows = load_split("validation")
    if train_x.max().item() >= len(vocabulary) or validation_x.max().item() >= len(vocabulary):
        raise ValueError("Saved sequence contains a token ID outside the Phase 2 vocabulary")
    if set(row["arxiv_id"] for row in train_rows) & set(row["arxiv_id"] for row in validation_rows):
        raise ValueError("Training and validation paper IDs overlap")
    split_summary = {
        "source": metadata["dataset_source"],
        "revision": metadata["dataset_revision"],
        "random_seed": json.loads((ROOT / "phase1/metadata.json").read_text(encoding="utf-8"))["seed"],
        "vocabulary_size": len(vocabulary),
        "sequence_length": int(metadata["sequence_length"]),
        "label_mapping": label_map,
        "preprocessing": "Reused saved Phase 2 tokens, vocabulary, right-padded 290-token sequence arrays, and original Phase 1 splits.",
        "model_selection_split": "validation",
        "test_split_opened_after_selection": False,
        "splits": {
            "train": summarize_split(train_x, train_y, label_map),
            "validation": summarize_split(validation_x, validation_y, label_map),
        },
    }
    atomic_json(OUTPUT / "dataset_reuse.json", split_summary)
    update_execution(
        "Phase 6 training data verified; beginning controlled learning-rate experiments",
        gpu={
            "name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "device": str(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "free_bytes_before_training": free_bytes,
            "total_bytes": total_bytes,
        },
        dataset={
            "source": metadata["dataset_source"],
            "revision": metadata["dataset_revision"],
            "label_mapping": label_map,
            "sequence_length": metadata["sequence_length"],
            "vocabulary_size": metadata["vocabulary_size"],
            "train_records": len(train_rows),
            "validation_records": len(validation_rows),
        },
    )

    config = model_config(len(vocabulary))
    configurations = [
        fit_experiment(
            rate,
            (train_x, train_y),
            (validation_x, validation_y),
            device,
            config,
            metadata,
        )
        for rate in LEARNING_RATES
    ]
    selected = max(configurations, key=lambda item: item["best_validation_macro_f1"])
    selection = {
        "selection_metric": "validation_macro_f1",
        "tie_break": "first configuration in the recorded learning-rate order",
        "selected": selected,
        "experiments": configurations,
    }
    atomic_json(OUTPUT / "selection.json", selection)
    update_execution(
        "All learning-rate experiments completed; selected the validation macro-F1 checkpoint",
        selected_checkpoints=[selected["checkpoint"]],
        experiment_configurations=[item["config"] for item in configurations],
        artifact_paths=[selected["checkpoint"], str(OUTPUT / "selection.json")],
        current_experiment=None,
    )

    model, best_checkpoint = load_model_checkpoint(Path(selected["checkpoint"]), device)
    validation_metrics, validation_predictions = evaluate(
        model, validation_x, validation_y, validation_rows, device
    )
    validation_metrics.update(
        {
            "device": str(device),
            "checkpoint": selected["checkpoint"],
            "checkpoint_bytes": selected["checkpoint_bytes"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": selected["training_seconds"],
            "best_epoch": best_checkpoint["epoch"],
        }
    )
    atomic_json(OUTPUT / "validation/custom_transformer.json", validation_metrics)
    atomic_json(OUTPUT / "validation/custom_transformer_predictions.json", validation_predictions)

    # Test data is opened only after every experiment and checkpoint have been selected.
    update_execution("Validation checkpoint selected; opening the held-out test split")
    test_x, test_y, test_rows = load_split("test")
    if test_x.max().item() >= len(vocabulary):
        raise ValueError("Test sequences contain a token ID outside the Phase 2 vocabulary")
    train_ids = {row["arxiv_id"] for row in train_rows}
    validation_ids = {row["arxiv_id"] for row in validation_rows}
    test_ids = {row["arxiv_id"] for row in test_rows}
    if train_ids & test_ids or validation_ids & test_ids or train_ids & validation_ids:
        raise ValueError("Saved data splits contain overlapping paper IDs")
    split_summary["splits"]["test"] = summarize_split(test_x, test_y, label_map)
    split_summary["test_split_opened_after_selection"] = True
    atomic_json(OUTPUT / "dataset_reuse.json", split_summary)
    test_metrics, test_predictions = evaluate(model, test_x, test_y, test_rows, device)
    test_metrics.update(
        {
            "device": str(device),
            "checkpoint": selected["checkpoint"],
            "checkpoint_bytes": selected["checkpoint_bytes"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": selected["training_seconds"],
            "best_epoch": best_checkpoint["epoch"],
        }
    )

    reloaded, _ = load_model_checkpoint(Path(selected["checkpoint"]), device)
    sample = test_x[: min(32, len(test_x))]
    sample_length = max(1, int(sample.ne(PAD_INDEX).sum(dim=1).max().item()))
    with torch.no_grad():
        original_scores = torch.softmax(model(sample[:, :sample_length].to(device)), dim=1)
        reloaded_scores = torch.softmax(reloaded(sample[:, :sample_length].to(device)), dim=1)
    difference = float((original_scores - reloaded_scores).abs().max().cpu())
    test_metrics["checkpoint_reload_consistency"] = {
        "checked_records": len(sample),
        "max_probability_absolute_difference": difference,
        "passed": bool(torch.allclose(original_scores, reloaded_scores, atol=1e-7, rtol=1e-6)),
    }
    if not test_metrics["checkpoint_reload_consistency"]["passed"]:
        raise RuntimeError("Reloaded checkpoint predictions do not match the selected model")

    atomic_json(OUTPUT / "test/custom_transformer.json", test_metrics)
    atomic_json(OUTPUT / "test/custom_transformer_predictions.json", test_predictions)
    _write_confusion_matrix(test_metrics["confusion_matrix"], OUTPUT / "figures/test_confusion_matrix.png")
    from ml.models.attention.report import (
        comparison_and_error_analysis,
        write_documentation,
    )

    comparison, error_analysis = comparison_and_error_analysis()
    write_documentation(comparison, error_analysis)
    update_execution(
        "Transformer trained, selected, tested, compared, and checkpoint reload consistency verified",
        test_results=[test_metrics],
        artifact_paths=[
            str(OUTPUT / "dataset_reuse.json"),
            selected["checkpoint"],
            str(OUTPUT / "validation/custom_transformer.json"),
            str(OUTPUT / "test/custom_transformer.json"),
            str(OUTPUT / "test/custom_transformer_predictions.json"),
            str(OUTPUT / "figures/test_confusion_matrix.png"),
            str(OUTPUT / "comparison.json"),
            str(OUTPUT / "error_analysis.json"),
            "docs/phase6_documentation.md",
            "docs/phase6_comparison.md",
            "docs/phase6_error_analysis.md",
        ],
        current_experiment=None,
    )
    print("Selected validation macro F1:", selected["best_validation_macro_f1"], flush=True)
    print("Test metrics:", json.dumps(test_metrics), flush=True)


if __name__ == "__main__":
    main()
