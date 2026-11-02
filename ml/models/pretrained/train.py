"""Validation-selected fine-tuning for the Phase 7 pretrained classifier."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer

from ml.models.pretrained.checkpoint import update_checkpoint
from ml.models.pretrained.classifier import load_classifier
from ml.models.pretrained.data import LABEL_NAMES, collate_examples, read_manifest, tokenize_records

ROOT = Path(__file__).resolve().parents[3]
PHASE1 = ROOT / "artifacts" / "data" / "phase1"
OUTPUT = ROOT / "artifacts" / "models" / "phase7"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MODEL_SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / MODEL_REVISION
MAX_LENGTH = 256


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_ref(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def usable_cuda() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        value = torch.ones((256, 256), device="cuda:0")
        (value @ value).sum().backward() if value.requires_grad else None
        torch.cuda.synchronize()
        del value
        torch.cuda.empty_cache()
        return True
    except Exception as error:
        update_checkpoint("classification", {"status": "cuda_unusable_cpu_fallback", "cuda_probe_error": repr(error)})
        return False


def _batches(examples: list[dict], batch_size: int, pad_id: int, shuffle: bool, seed: int):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(examples), generator=generator).tolist()
    else:
        indices = list(range(len(examples)))
    for start in range(0, len(indices), batch_size):
        items = [examples[index] for index in indices[start:start + batch_size]]
        yield collate_examples(items, pad_id, torch)


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items() if key != "arxiv_ids"}


def _scores(y_true: list[int], y_pred: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABEL_NAMES))), zero_division=0
    )
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            LABEL_NAMES[index]: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index in range(len(LABEL_NAMES))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES)))).tolist(),
    }


@torch.inference_mode()
def evaluate_tokenized(model, examples: list[dict], tokenizer, device, batch_size: int, *, measure: bool = False):
    model.eval()
    actual, predicted, probabilities, losses = [], [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in _batches(examples, batch_size, tokenizer.pad_token_id, False, 0):
        moved = _move(batch, device)
        logits = model(**moved).logits
        if "labels" in moved:
            losses.append(float(torch.nn.functional.cross_entropy(logits.float(), moved["labels"]).item()))
            actual.extend(moved["labels"].cpu().tolist())
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probabilities.extend(probs.tolist())
        predicted.extend(probs.argmax(axis=1).tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {"actual": actual, "predicted": predicted, "probabilities": probabilities}
    if actual:
        result["metrics"] = _scores(actual, predicted)
        result["loss"] = float(np.mean(losses)) if losses else None
    if measure:
        result["elapsed_seconds"] = elapsed
        result["milliseconds_per_document"] = 1000 * elapsed / max(1, len(examples))
    return result


def _save_resume(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run_experiment(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    train_rows: list[dict],
    validation_rows: list[dict],
    config: dict,
    tokenizer=None,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    device_override: str | None = None,
) -> dict:
    """Fine-tune one LR/seed configuration, selecting only on validation macro F1."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "experiment_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            return existing
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(device_override or ("cuda:0" if usable_cuda() else "cpu"))
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    effective_max_length = min(int(config.get("max_length", MAX_LENGTH)), int(tokenizer.model_max_length), 512)
    train_examples = tokenize_records(tokenizer, train_rows, effective_max_length)
    validation_examples = tokenize_records(tokenizer, validation_rows, effective_max_length)
    train_truncated = sum(len(example["input_ids"]) >= effective_max_length for example in train_examples)
    validation_truncated = sum(len(example["input_ids"]) >= effective_max_length for example in validation_examples)
    batch_size = int(config.get("batch_size", 8))
    accumulation = int(config.get("gradient_accumulation_steps", 4))
    epochs = int(config.get("epochs", 3))
    patience = int(config.get("early_stopping_patience", 1))
    if min(batch_size, accumulation, epochs, patience) < 1:
        raise ValueError("Batch size, accumulation, epoch and patience settings must be positive")
    model = load_classifier(str(model_path), local_files_only=True).to(device)
    model.config.phase7_max_length = effective_max_length
    model.config.id2label = {index: name for index, name in enumerate(LABEL_NAMES)}
    model.config.label2id = {name: index for index, name in enumerate(LABEL_NAMES)}
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(config.get("weight_decay", 0.01))}, {"params": no_decay, "weight_decay": 0.0}],
        lr=float(config["learning_rate"]),
    )
    steps_per_epoch = math.ceil(len(train_examples) / batch_size / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * float(config.get("warmup_ratio", 0.06))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, max(0.0, (total_steps - step) / max(1, total_steps - warmup))),
    )
    amp_enabled = device.type == "cuda" and bool(config.get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    resume_path = output_dir / "resume_state.pt"
    best_dir = output_dir / "best_checkpoint"
    history_path = output_dir / "training_history.json"
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("config") != config or resume.get("model_revision") != model_revision:
            raise ValueError("Resume state does not match the current experiment configuration/model revision")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        history = resume["history"]
        best_f1, best_epoch, stale = resume["best_f1"], resume["best_epoch"], resume["stale"]
        start_epoch = resume["last_epoch"] + 1
    training_started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batches = list(_batches(train_examples, batch_size, tokenizer.pad_token_id, True, seed + epoch))
        for batch_index, batch in enumerate(batches):
            moved = _move(batch, device)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(**moved).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), moved["labels"])
                losses.append(float(loss.detach().item()))
                scaler.scale(loss / accumulation).backward()
                update_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches)
                if update_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            except torch.cuda.OutOfMemoryError as error:
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                update_checkpoint("classification", {"status": "failed_oom", "failed_experiment": output_dir.name, "epoch": epoch, "batch_index": batch_index, "error": str(error), "resume_instructions": "Rerun this configuration with batch_size=4 and gradient_accumulation_steps=8."})
                raise
        validation = evaluate_tokenized(model, validation_examples, tokenizer, device, max(batch_size, 16))
        score = validation["metrics"]["macro_f1"]
        improved = score > best_f1
        if improved:
            best_f1, best_epoch, stale = score, epoch, 0
            shutil.rmtree(best_dir, ignore_errors=True)
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        else:
            stale += 1
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "validation_metrics": validation["metrics"],
            "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
            "selected_as_best": bool(improved),
        }
        history.append(entry)
        write_json(history_path, history)
        _save_resume(resume_path, {
            "config": config,
            "model_revision": model_revision,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "stale": stale,
            "last_epoch": epoch,
        })
        report = {
            "status": "running",
            "experiment": output_dir.name,
            "model_id": model_id,
            "model_revision": model_revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": model_revision,
            "model_license": "Apache-2.0",
            "configuration": config,
            "effective_max_length": effective_max_length,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "mixed_precision": amp_enabled,
            "train_records": len(train_examples),
            "validation_records": len(validation_examples),
            "train_at_limit_count": train_truncated,
            "validation_at_limit_count": validation_truncated,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "history": history,
            "best_checkpoint": _artifact_ref(best_dir),
            "elapsed_seconds_so_far": time.perf_counter() - training_started,
            "status_message": f"Finished epoch {epoch} of {epochs}",
        }
        write_json(report_path, report)
        update_checkpoint("classification", {"status": "training", "active_experiment": output_dir.name, "completed_epoch": epoch, "epochs": epochs, "best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "experiment_report": str(report_path.relative_to(ROOT))})
        if stale >= patience:
            break
    training_seconds = time.perf_counter() - training_started
    checkpoint_reference = _artifact_ref(best_dir)
    report = {
        "status": "complete",
        "experiment": output_dir.name,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": model_id,
        "tokenizer_revision": model_revision,
        "model_license": "Apache-2.0",
        "configuration": config,
        "effective_max_length": effective_max_length,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "mixed_precision": amp_enabled,
        "train_records": len(train_examples),
        "validation_records": len(validation_examples),
        "train_at_limit_count": train_truncated,
        "validation_at_limit_count": validation_truncated,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "history": history,
        "best_checkpoint": checkpoint_reference,
        "training_seconds": training_seconds,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_bytes": sum(path.stat().st_size for path in best_dir.rglob("*") if path.is_file()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report_path, report)
    return report


def evaluate_selected_checkpoint(report: dict, test_rows: list[dict], output_dir: str | Path) -> dict:
    """Evaluate the already selected validation checkpoint once on the held-out test split."""
    from ml.models.pretrained.classifier import ClassifierPredictor

    output_dir = Path(output_dir)
    checkpoint = ROOT / report["best_checkpoint"]
    device = "cuda:0" if usable_cuda() else "cpu"
    predictor = ClassifierPredictor(str(checkpoint), device=device, max_length=report["effective_max_length"])
    # Warm the selected model before timing and synchronize actual device work.
    predictor.predict(test_rows[: min(16, len(test_rows))], batch_size=min(16, max(1, len(test_rows))))
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = predictor.predict(test_rows, batch_size=32)
    if predictor.device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    actual = [row["label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    metrics = _scores(actual, predicted)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            value = {key: row[key] for key in ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")}
            value.update({"actual_label": row["label"], "actual_label_name": LABEL_NAMES[row["label"]], "predicted_label": row["predicted_label"], "predicted_label_name": row["predicted_label_name"], "probabilities": row["probabilities"]})
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    errors = []
    for row in predictions:
        if row["label"] == row["predicted_label"]:
            continue
        value = {key: row[key] for key in ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")}
        value.update({"actual_label_name": LABEL_NAMES[row["label"]], "predicted_label": row["predicted_label"], "predicted_label_name": row["predicted_label_name"], "probabilities": row["probabilities"]})
        errors.append(value)
    result = {
        "model_id": report["model_id"],
        "model_revision": report["model_revision"],
        "validation_macro_f1": report["best_validation_macro_f1"],
        "test_records": len(test_rows),
        "test_metrics": metrics,
        "evaluation_device": str(predictor.device),
        "inference_seconds_batch32_includes_tokenization": inference_seconds,
        "inference_ms_per_document_batch32_includes_tokenization": 1000 * inference_seconds / max(1, len(test_rows)),
        "parameter_count": report["parameter_count"],
        "checkpoint_bytes": report["checkpoint_bytes"],
        "checkpoint_path": report["best_checkpoint"],
        "test_predictions_path": str((output_dir / "test_predictions.jsonl").relative_to(ROOT)),
        "error_count": len(errors),
        "error_examples_path": str((output_dir / "error_analysis.json").relative_to(ROOT)),
    }
    write_json(output_dir / "test_metrics.json", result)
    write_json(output_dir / "error_analysis.json", errors[:100])
    return result


def run_classification_experiments() -> dict:
    cached_results = OUTPUT / "classification_results.json"
    if cached_results.exists():
        cached = json.loads(cached_results.read_text(encoding="utf-8"))
        prediction_path = OUTPUT / "evaluation" / "test_predictions.jsonl"
        checkpoint_path = ROOT / cached.get("checkpoint_path", "")
        report_path = checkpoint_path.parent / "experiment_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        prediction_count = sum(1 for line in prediction_path.open(encoding="utf-8") if line.strip()) if prediction_path.exists() else 0
        checkpoint_weights = checkpoint_path / "model.safetensors"
        valid = (
            cached.get("test_records") == 2000
            and prediction_count == 2000
            and checkpoint_weights.exists()
            and report.get("status") == "complete"
            and report.get("best_validation_macro_f1") == cached.get("validation_macro_f1")
            and checkpoint_weights.stat().st_size > 0
        )
        if valid:
            return cached
        raise ValueError("Completed classifier artifacts failed report, prediction-count, or checkpoint integrity checks")
    if not MODEL_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"Selected model snapshot is not cached locally: {MODEL_SNAPSHOT}")
    metadata = json.loads((PHASE1 / "metadata.json").read_text(encoding="utf-8"))
    # Keep the held-out test manifest out of the training/tuning data structures.
    splits = {
        "train": read_manifest(PHASE1 / "train.jsonl"),
        "validation": read_manifest(PHASE1 / "validation.jsonl"),
    }
    for split in ("train", "validation"):
        if len(splits[split]) != metadata["split_sizes"][split]:
            raise ValueError(f"{split} record count disagrees with Phase 1 metadata")
    train_validation_ids = [row["arxiv_id"] for rows in splits.values() for row in rows]
    if len(train_validation_ids) != len(set(train_validation_ids)):
        raise ValueError("Training and validation manifests contain duplicate paper IDs")
    configs = [
        {"name": "lr_2e-5", "learning_rate": 2e-5, "batch_size": 8, "gradient_accumulation_steps": 4, "epochs": 3, "early_stopping_patience": 1, "weight_decay": 0.01, "max_grad_norm": 1.0, "max_length": MAX_LENGTH, "warmup_ratio": 0.06, "mixed_precision": True, "seed": 42},
        {"name": "lr_3e-5", "learning_rate": 3e-5, "batch_size": 8, "gradient_accumulation_steps": 4, "epochs": 3, "early_stopping_patience": 1, "weight_decay": 0.01, "max_grad_norm": 1.0, "max_length": MAX_LENGTH, "warmup_ratio": 0.06, "mixed_precision": True, "seed": 42},
    ]
    update_checkpoint("classification", {
        "status": "preparing_experiments",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "selected_encoder_max_length": MAX_LENGTH,
        "split_sizes_verified": {name: len(rows) for name, rows in splits.items()},
        "held_out_test_records": metadata["split_sizes"]["test"],
        "experiments": configs,
        "validation_selection_metric": "macro F1",
        "test_evaluation": "not_started",
    })
    experiment_reports = []
    for config in configs:
        exp_dir = OUTPUT / "experiments" / config["name"]
        update_checkpoint("classification", {"status": "experiment_starting", "active_experiment": config["name"], "configuration": config, "train_records": len(splits["train"]), "validation_records": len(splits["validation"]), "test_evaluation": "not_started"})
        report = run_experiment(model_path=MODEL_SNAPSHOT, output_dir=exp_dir, train_rows=splits["train"], validation_rows=splits["validation"], config=config)
        experiment_reports.append(report)
    selected = max(experiment_reports, key=lambda item: item["best_validation_macro_f1"])
    write_json(OUTPUT / "classification_experiments.json", {
        "selection_metric": "validation macro F1",
        "test_used_for_selection": False,
        "selected_experiment": selected["experiment"],
        "experiments": experiment_reports,
    })
    update_checkpoint("classification", {"status": "selected", "experiments": [{"name": item["experiment"], "best_validation_macro_f1": item["best_validation_macro_f1"], "best_epoch": item["best_epoch"], "report": str((OUTPUT / "experiments" / item["experiment"] / "experiment_report.json").relative_to(ROOT))} for item in experiment_reports], "selected_experiment": selected["experiment"], "selected_checkpoint": selected["best_checkpoint"], "test_evaluation": "not_started"})
    # Open the held-out records only after validation has selected the checkpoint.
    test_rows = read_manifest(PHASE1 / "test.jsonl")
    if len(test_rows) != metadata["split_sizes"]["test"]:
        raise ValueError("Test manifest size disagrees with Phase 1 metadata")
    if set(row["arxiv_id"] for row in test_rows) & set(train_validation_ids):
        raise ValueError("Test IDs overlap training or validation IDs")
    result = evaluate_selected_checkpoint(selected, test_rows, OUTPUT / "evaluation")
    result["experiments"] = [{"name": item["experiment"], "learning_rate": item["configuration"]["learning_rate"], "validation_macro_f1": item["best_validation_macro_f1"], "best_epoch": item["best_epoch"], "training_seconds": item["training_seconds"], "parameter_count": item["parameter_count"], "checkpoint_bytes": item["checkpoint_bytes"]} for item in experiment_reports]
    write_json(OUTPUT / "classification_results.json", result)
    update_checkpoint("classification", {"status": "complete", "model_id": selected["model_id"], "model_revision": selected["model_revision"], "tokenizer_id": selected["tokenizer_id"], "tokenizer_revision": selected["tokenizer_revision"], "license": selected["model_license"], "configuration": selected["configuration"], "selected_experiment": selected["experiment"], "best_validation_macro_f1": selected["best_validation_macro_f1"], "best_epoch": selected["best_epoch"], "test_metrics": result["test_metrics"], "device": selected["device"], "evaluation_device": result["evaluation_device"], "training_seconds_by_experiment": {item["experiment"]: item["training_seconds"] for item in experiment_reports}, "parameter_count": selected["parameter_count"], "checkpoint_bytes": selected["checkpoint_bytes"], "artifact_paths": {"checkpoint": selected["best_checkpoint"], "results": "artifacts/models/phase7/classification_results.json", "predictions": result["test_predictions_path"], "confusion_matrix": "embedded in test_metrics.json", "error_analysis": result["error_examples_path"]}})
    return result


if __name__ == "__main__":
    print(json.dumps(run_classification_experiments(), ensure_ascii=False, indent=2))
