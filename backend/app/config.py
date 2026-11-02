from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str
    artifact_root: Path
    model_id: str = "P3/tfidf_svm"
    api_prefix: str = "/api/v1"
    max_page_size: int = 100
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://127.0.0.1:3000")

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_url=os.environ.get(
                "PAPERLENS_DATABASE_URL",
                "postgresql+psycopg://localhost:5432/paperlens",
            ),
            artifact_root=Path(os.environ.get("PAPERLENS_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts" / "data")),
            cors_origins=tuple(
                origin.strip()
                for origin in os.environ.get(
                    "PAPERLENS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
                ).split(",")
                if origin.strip()
            ),
        )
