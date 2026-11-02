"""Phase 7 records, tokenizer inputs, and dynamically padded batches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

FIELDS = ("arxiv_id", "title", "abstract", "primary_cat", "all_cats", "label")
LABEL_NAMES = ("Algorithms", "NLP/AI", "Systems/Infra", "Vision")


def text_value(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def paper_text(row: Mapping[str, Any]) -> str:
    """Use the same title + abstract serialization for training and inference."""
    title = text_value(row.get("title")).strip()
    abstract = text_value(row.get("abstract")).strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def validate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("Each paper record must be a mapping")
    missing = [field for field in FIELDS if field not in row]
    if missing:
        raise ValueError(f"Paper record is missing required fields: {missing}")
    result = {field: row[field] for field in FIELDS}
    if not isinstance(result["arxiv_id"], str) or not result["arxiv_id"].strip():
        raise ValueError("arxiv_id must be a nonempty string")
    if isinstance(result["label"], bool) or not isinstance(result["label"], int) or result["label"] not in range(len(LABEL_NAMES)):
        raise ValueError("label must be an integer from 0 through 3")
    # Text fields are normalized only for model input; the six source values remain
    # untouched in the returned record and in generated prediction/index metadata.
    return result


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON in {path} at line {line_number}") from error
            rows.append(validate_record(value))
    return rows


def load_phase1_splits(source: str | Path, expected: Mapping[str, int] | None = None) -> dict[str, list[dict[str, Any]]]:
    source = Path(source)
    split_rows = {name: read_manifest(source / f"{name}.jsonl") for name in ("train", "validation", "test")}
    seen: set[str] = set()
    for split, rows in split_rows.items():
        if expected is not None and len(rows) != int(expected[split]):
            raise ValueError(f"{split} has {len(rows)} rows; expected {expected[split]}")
        for row in rows:
            if row["arxiv_id"] in seen:
                raise ValueError(f"Duplicate arxiv_id across Phase 1 splits: {row['arxiv_id']}")
            seen.add(row["arxiv_id"])
    return split_rows


def tokenize_records(tokenizer: Any, rows: Iterable[Mapping[str, Any]], max_length: int) -> list[dict[str, Any]]:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    rows = [validate_record(row) for row in rows]
    if not rows:
        return []
    texts = [paper_text(row) for row in rows]
    # Keep examples unpadded; the collator pads only to the longest item in a batch.
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False, add_special_tokens=True)
    result = []
    for row, ids, mask in zip(rows, encoded["input_ids"], encoded["attention_mask"]):
        if not ids:
            ids = [tokenizer.cls_token_id or tokenizer.pad_token_id, tokenizer.sep_token_id or tokenizer.pad_token_id]
            mask = [1, 1]
        result.append({"arxiv_id": row["arxiv_id"], "input_ids": list(ids), "attention_mask": list(mask), "label": row["label"]})
    return result


def collate_examples(examples: list[Mapping[str, Any]], pad_token_id: int, torch: Any) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in examples]
    masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in examples]
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0),
        "labels": torch.tensor([item["label"] for item in examples], dtype=torch.long),
        "arxiv_ids": [item["arxiv_id"] for item in examples],
    }
