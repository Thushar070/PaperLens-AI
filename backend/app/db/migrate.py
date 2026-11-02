from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from app.config import Settings
from app.db.repository import DatabaseRepository


MIGRATION_DIR = Path(__file__).with_name("migrations")


def migrate(database_url: str | None = None) -> list[str]:
    settings = Settings.from_environment()
    repository = DatabaseRepository(database_url or settings.database_url)
    engine = repository.engine
    applied: list[str] = []
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            for migration in sorted(MIGRATION_DIR.glob("[0-9]*.sql")):
                version = migration.stem
                already_applied = connection.execute(
                    text("SELECT 1 FROM schema_migrations WHERE version = :version"), {"version": version}
                ).first()
                if already_applied:
                    continue
                for statement in migration.read_text(encoding="utf-8").split(";"):
                    if statement.strip():
                        connection.exec_driver_sql(statement)
                connection.execute(
                    text("INSERT INTO schema_migrations(version) VALUES (:version)"), {"version": version}
                )
                applied.append(version)
    finally:
        engine.dispose()
    return applied


if __name__ == "__main__":
    print({"applied_migrations": migrate()})
