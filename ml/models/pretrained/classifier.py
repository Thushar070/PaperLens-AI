"""Pretrained four-class encoder loading and reusable inference."""
from __future__ import annotations

from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from ml.models.pretrained.data import LABEL_NAMES, paper_text, validate_record


def load_tokenizer(model_path: str, *, local_files_only: bool = True):
    return AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)


def load_classifier(model_path: str, *, local_files_only: bool = True):
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    config.num_labels = len(LABEL_NAMES)
    config.id2label = {index: name for index, name in enumerate(LABEL_NAMES)}
    config.label2id = {name: index for index, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    if int(model.config.num_labels) != len(LABEL_NAMES):
        raise ValueError("Loaded classifier does not have four outputs")
    return model


class ClassifierPredictor:
    """Load a saved Phase 7 checkpoint and predict one or more paper records."""

    def __init__(self, checkpoint: str, device: str | None = None, max_length: int | None = None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = load_tokenizer(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
        if self.model.config.num_labels != len(LABEL_NAMES):
            raise ValueError("Checkpoint must contain a four-class head")
        self.model.to(self.device).eval()
        configured = int(getattr(self.model.config, "phase7_max_length", 256))
        self.max_length = min(int(max_length or configured), int(self.tokenizer.model_max_length), 512)
        if self.max_length < 2:
            raise ValueError("Invalid checkpoint token limit")

    @torch.inference_mode()
    def predict(self, rows, batch_size: int = 16) -> list[dict]:
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        rows = [validate_record(row) for row in rows]
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            encoded = self.tokenizer(
                [paper_text(row) for row in batch_rows],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            classes = probabilities.argmax(axis=1)
            for row, probs, label in zip(batch_rows, probabilities, classes):
                result = dict(row)
                result.update({
                    "predicted_label": int(label),
                    "predicted_label_name": LABEL_NAMES[int(label)],
                    "probabilities": [float(value) for value in probs],
                })
                predictions.append(result)
        return predictions

    def predict_one(self, row: dict) -> dict:
        return self.predict(row, batch_size=1)[0]
