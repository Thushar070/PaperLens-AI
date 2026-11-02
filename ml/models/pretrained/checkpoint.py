"""Atomic updates to the persistent Phase 7 execution checkpoint."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "docs" / "phase7_execution_checkpoint.json"


def update_checkpoint(section: str, values: dict) -> dict:
    data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data.setdefault("scope", {})[section] = values
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="phase7-checkpoint-", suffix=".json", dir=CHECKPOINT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, CHECKPOINT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return data
