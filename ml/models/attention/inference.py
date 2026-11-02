"""Inference using the Phase 2 vocabulary and selected Phase 6 checkpoint."""

import json
from pathlib import Path

import torch

from ml.models.attention.train import load_model_checkpoint
from ml.preprocessing.text import document, tokens
from ml.preprocessing.vocabulary import encode


def predict(
    title: str,
    abstract: str,
    checkpoint_path: str | Path = "artifacts/data/phase6/selection.json",
    root: str | Path = "artifacts/data",
    device: str | torch.device | None = None,
) -> dict:
    text = document({"title": title, "abstract": abstract})
    if not text.strip():
        raise ValueError("Title and abstract cannot both be empty")

    root = Path(root)
    selection_path = Path(checkpoint_path)
    if selection_path.suffix.lower() == ".json":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_path = selection["selected"]["checkpoint"]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
    selected_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    model, _ = load_model_checkpoint(Path(checkpoint_path), selected_device)

    vocabulary = json.loads((root / "phase2/vocabulary.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "phase2/metadata.json").read_text(encoding="utf-8"))
    label_mapping = json.loads((root / "phase1/metadata.json").read_text(encoding="utf-8"))["label_mapping"]
    token_ids = encode(tokens(text), vocabulary, int(metadata["sequence_length"]))
    inputs = torch.tensor([token_ids], dtype=torch.long, device=selected_device)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].cpu().tolist()
    label = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "label": label,
        "category": label_mapping[str(label)],
        "probabilities": {
            label_mapping[str(index)]: float(value)
            for index, value in enumerate(probabilities)
        },
        "device": str(selected_device),
        "checkpoint": str(checkpoint_path),
    }
